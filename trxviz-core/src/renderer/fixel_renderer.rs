use wgpu::util::DeviceExt;

use crate::data::odx_data::FixelInstance;
use crate::lighting::{SceneLightingParams, WorkflowRender3D};
use crate::renderer::viewport::ViewportUniformSet;

pub struct FixelResources {
    pipeline: wgpu::RenderPipeline,
    slice_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: Option<wgpu::Buffer>,
    viewports: ViewportUniformSet<FixelUniforms>,
    num_instances: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FixelUniforms {
    view_proj: [[f32; 4]; 4], // [0..64]
    camera_pos: [f32; 3],     // [64..76]
    slab_half_width: f32,     // [76..80]  0 = disabled
    slab_normal: [f32; 3],    // [80..92]
    draw_step: u32,           // [92..96]
    slab_center: [f32; 3],    // [96..108]
    line_width: f32,          // [108..112]
    ambient_strength: f32,    // [112..116]
    key_strength: f32,        // [116..120]
    fill_strength: f32,       // [120..124]
    opacity: f32,             // [124..128]
    fog_color: [f32; 4],      // [128..144]
    fog_params: [f32; 4],     // [144..160]
    post_params: [f32; 4],    // [160..176]
    color_params: [f32; 4],   // [176..192] x=colormap, y=scalar_min, z=scalar_max, w=reserved
    opacity_gate: [f32; 4],   // [192..208] x=range_min, y=range_max, z=below, w=above
}

/// Unit quad: 4 vertices, 2 triangles.
///
/// `x` in `[-1, 1]` runs along the fixel direction.
/// `y` in `[-0.5, 0.5]` gives the screen-space perpendicular offset.
const QUAD_VERTICES: [[f32; 2]; 4] = [[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]];

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

impl FixelResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fixel_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fixel.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fixel_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fixel_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let default_uniforms = FixelUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 1.0],
            slab_half_width: 0.0,
            slab_normal: [0.0, 0.0, 1.0],
            draw_step: 1,
            slab_center: [0.0; 3],
            line_width: 0.006,
            ambient_strength: 0.5,
            key_strength: 0.5,
            fill_strength: 0.0,
            opacity: 1.0,
            fog_color: [0.0; 4],
            fog_params: [0.0, 1.0, 0.0, 0.0],
            post_params: [1.0, 1.0, 0.12, 0.0],
            color_params: [0.0, 0.0, 1.0, 0.0],
            // Default opacity gate is pass-through (every scalar → 1.0).
            opacity_gate: [0.0, 0.0, 1.0, 1.0],
        };

        let viewports =
            ViewportUniformSet::new(device, "fixel_uniform", &default_uniforms, |i, buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fixel_bg_{i}")),
                    layout: &bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 8, // 2 × f32
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FixelInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // center
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3, // direction
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32, // length
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32, // scalar
                },
            ],
        };

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fixel_quad_verts"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fixel_quad_indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let pipeline = make_fixel_pipeline(
            device,
            &layout,
            &shader,
            target_format,
            &[vertex_layout.clone(), instance_layout.clone()],
            wgpu::CompareFunction::LessEqual,
            true,
        );
        let slice_pipeline = make_fixel_pipeline(
            device,
            &layout,
            &shader,
            target_format,
            &[vertex_layout, instance_layout],
            wgpu::CompareFunction::Always,
            false,
        );

        Self {
            pipeline,
            slice_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer: None,
            viewports,
            num_instances: 0,
        }
    }

    pub fn set_fixels(&mut self, device: &wgpu::Device, instances: &[FixelInstance]) {
        if instances.is_empty() {
            self.instance_buffer = None;
            self.num_instances = 0;
            return;
        }
        self.instance_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fixel_instances"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
        self.num_instances = instances.len() as u32;
    }

    pub fn clear(&mut self) {
        self.instance_buffer = None;
        self.num_instances = 0;
    }

    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        slab_normal: glam::Vec3,
        slab_center: glam::Vec3,
        slab_half_width: f32,
        draw_step: u32,
        line_width: f32,
        opacity: f32,
        opacity_gate: [f32; 4],
        scene_lighting: SceneLightingParams,
        render_3d: &WorkflowRender3D,
        fog_near: f32,
        fog_far: f32,
    ) {
        let uniforms = FixelUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.into(),
            slab_half_width,
            slab_normal: slab_normal.into(),
            draw_step: draw_step.max(1),
            slab_center: slab_center.into(),
            line_width,
            ambient_strength: scene_lighting.ambient_strength(),
            key_strength: scene_lighting.key_strength(),
            fill_strength: scene_lighting.fill_strength(),
            opacity: opacity.clamp(0.0, 1.0),
            fog_color: [
                render_3d.fog_color[0],
                render_3d.fog_color[1],
                render_3d.fog_color[2],
                if render_3d.fog_enabled { 1.0 } else { 0.0 },
            ],
            fog_params: [fog_near, fog_far.max(fog_near + 0.001), 0.0, 0.0],
            post_params: [
                render_3d.exposure,
                render_3d.contrast,
                render_3d.vignette_strength,
                0.0,
            ],
            color_params: [0.0, 0.0, 1.0, 0.0],
            opacity_gate,
        };
        self.viewports.update(queue, viewport, &uniforms);
    }

    /// Patch colormap + scalar range via `color_params` at offset 176.
    pub fn update_colormap(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        code: u32,
        range: (f32, f32),
    ) {
        const OFFSET: u64 = 176;
        let packed = [code as f32, range.0, range.1, 0.0f32];
        self.viewports
            .write(queue, viewport, OFFSET, bytemuck::bytes_of(&packed));
    }

    /// Patch the length multiplier (fixel instance length × mul) via `post_params.w`.
    /// Pass 0.0 to fall back to the shader default of 1.0.
    pub fn update_length_mul(&self, queue: &wgpu::Queue, viewport: usize, length_mul: f32) {
        // Offset of `post_params[3]` in `FixelUniforms`: 160 + 12.
        const OFFSET: u64 = 160 + 12;
        self.viewports
            .write(queue, viewport, OFFSET, bytemuck::bytes_of(&length_mul));
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>, viewport: usize, slice: bool) {
        if self.num_instances == 0 {
            return;
        }
        let Some(inst) = &self.instance_buffer else {
            return;
        };
        render_pass.set_pipeline(if slice {
            &self.slice_pipeline
        } else {
            &self.pipeline
        });
        render_pass.set_bind_group(0, self.viewports.bind_group(viewport), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, inst.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..6, 0, 0..self.num_instances);
    }
}

fn make_fixel_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    depth_compare: wgpu::CompareFunction,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fixel_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers,
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: depth_write,
            depth_compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview: None,
        cache: None,
    })
}

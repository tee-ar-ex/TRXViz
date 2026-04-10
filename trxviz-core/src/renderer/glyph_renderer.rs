use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::data::orientation_field::{BoundaryContactField, BoundaryGlyphColorMode};
use crate::lighting::SceneLightingParams;

pub struct GlyphResources {
    pipeline: wgpu::RenderPipeline,
    slice_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    amplitude_buffer: Option<wgpu::Buffer>,
    uniform_buffers: [wgpu::Buffer; 4],
    bind_groups: [wgpu::BindGroup; 4],
    num_indices: u32,
    num_instances: u32,
    current_bins: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphUniforms {
    view_proj: [[f32; 4]; 4],
    slab_axis: u32,
    color_mode: u32,
    draw_step: u32,
    slab_min: f32,
    slab_max: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad1: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    pub center: [f32; 3],
    pub scale: f32,
    pub amplitude_offset: u32,
    pub min_contacts: u32,
    pub contact_count: u32,
    pub _pad: u32,
}

impl GlyphResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/glyph.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glyph_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let uniforms = GlyphUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            slab_axis: 3,
            color_mode: 0,
            draw_step: 1,
            slab_min: 0.0,
            slab_max: 0.0,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            _pad1: [0.0; 2],
        };
        let amplitude_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_amplitudes_empty"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("glyph_uniform_{i}")),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("glyph_bg_{i}")),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: amplitude_buffer.as_entire_binding(),
                    },
                ],
            })
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlyphInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        let pipeline = make_pipeline(
            device,
            &layout,
            &shader,
            target_format,
            &[vertex_layout.clone(), instance_layout.clone()],
            wgpu::CompareFunction::LessEqual,
            true,
        );
        let slice_pipeline = make_pipeline(
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
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            amplitude_buffer: Some(amplitude_buffer),
            uniform_buffers,
            bind_groups,
            num_indices: 0,
            num_instances: 0,
            current_bins: 0,
        }
    }

    pub fn set_field(
        &mut self,
        device: &wgpu::Device,
        field: Arc<BoundaryContactField>,
        scale: f32,
        min_contacts: u32,
    ) {
        let vertices = &field.sphere.vertices;
        let indices = &field.sphere.indices;
        let nbins = field.sphere.vertices.len() as u32;

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("glyph_vertices"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("glyph_indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );

        let mut instances = Vec::new();
        let mut amplitudes = Vec::new();
        for (compact_idx, &flat) in field.occupied_voxels().iter().enumerate() {
            let contact_count = field.contact_count(compact_idx);
            let coords = field.grid.unflatten(flat);
            let center = field.grid.voxel_center(coords[0], coords[1], coords[2]);
            let offset = amplitudes.len() as u32;
            amplitudes.extend_from_slice(field.histogram_for_voxel(compact_idx));
            instances.push(GlyphInstance {
                center: center.to_array(),
                scale,
                amplitude_offset: offset,
                min_contacts,
                contact_count,
                _pad: 0,
            });
        }

        self.instance_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("glyph_instances"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glyph_amplitudes"),
            contents: bytemuck::cast_slice(&amplitudes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.amplitude_buffer = Some(amplitude_buffer);
        self.num_indices = indices.len() as u32;
        self.num_instances = instances.len() as u32;
        self.current_bins = nbins;

        for i in 0..4 {
            self.bind_groups[i] = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("glyph_bg_dynamic"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self
                            .amplitude_buffer
                            .as_ref()
                            .expect("amplitude buffer")
                            .as_entire_binding(),
                    },
                ],
            });
        }
    }

    pub fn clear(&mut self) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.instance_buffer = None;
        self.num_indices = 0;
        self.num_instances = 0;
        self.current_bins = 0;
    }

    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        slab_axis: u32,
        slab_min: f32,
        slab_max: f32,
        color_mode: BoundaryGlyphColorMode,
        draw_step: u32,
        scene_lighting: SceneLightingParams,
    ) {
        let uniforms = GlyphUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            slab_axis,
            color_mode: match color_mode {
                BoundaryGlyphColorMode::DirectionRgb => 0,
                BoundaryGlyphColorMode::Monochrome => 1,
            },
            draw_step: draw_step.max(1),
            slab_min,
            slab_max,
            ambient_strength: scene_lighting.ambient_strength(),
            key_strength: scene_lighting.key_strength(),
            fill_strength: scene_lighting.fill_strength(),
            headlight_mix: scene_lighting.headlight_mix(),
            specular_strength: scene_lighting.specular_strength(),
            _pad1: [0.0; 2],
        };
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>, viewport: usize, slice: bool) {
        if self.num_indices == 0 || self.num_instances == 0 {
            return;
        }
        let (Some(vb), Some(ib), Some(inst)) = (
            &self.vertex_buffer,
            &self.index_buffer,
            &self.instance_buffer,
        ) else {
            return;
        };
        render_pass.set_pipeline(if slice {
            &self.slice_pipeline
        } else {
            &self.pipeline
        });
        render_pass.set_bind_group(0, &self.bind_groups[viewport], &[]);
        render_pass.set_vertex_buffer(0, vb.slice(..));
        render_pass.set_vertex_buffer(1, inst.slice(..));
        render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    depth_compare: wgpu::CompareFunction,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("glyph_pipeline"),
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

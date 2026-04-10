use wgpu::util::DeviceExt;

use crate::data::trx_data::{TrxGpuData, TubeMeshVertex};
use crate::lighting::SceneLightingParams;

/// Holds StreamlineResources for all loaded TRX files, keyed by FileId.
pub struct AllStreamlineResources {
    pub entries: Vec<(usize, StreamlineResources)>,
}

/// GPU resources for streamline rendering.
pub struct StreamlineResources {
    /// Line-based pipeline (Flat / Illuminated / DepthCue modes). 3D viewport.
    pub pipeline: wgpu::RenderPipeline,
    /// Line-based pipeline for slice views (depth Always).
    pub slice_pipeline: wgpu::RenderPipeline,
    /// Streamtube mesh pipeline. 3D viewport.
    pub tube_pipeline: wgpu::RenderPipeline,
    pub position_buffer: wgpu::Buffer,
    pub color_buffer: wgpu::Buffer,
    pub tangent_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    /// Tube mesh geometry — rebuilt whenever the selected set changes.
    pub tube_vertex_buffer: Option<wgpu::Buffer>,
    pub tube_index_buffer: Option<wgpu::Buffer>,
    pub num_tube_indices: u32,
    /// Per-viewport uniform buffers: [3D, axial, coronal, sagittal].
    pub uniform_buffers: [wgpu::Buffer; 4],
    /// Per-viewport bind groups.
    pub bind_groups: [wgpu::BindGroup; 4],
    pub num_indices: u32,
}

/// Must match the Uniforms struct in both streamline.wgsl and tube.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    /// 0=flat, 1=illuminated, 2=tubes (unused in line shader), 3=depth_cue
    render_style: u32,
    /// Slab clipping axis: 0=X, 1=Y, 2=Z, 3=disabled.
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
    /// Tube radius (mm). Reused as depth_far for depth-cue mode.
    tube_radius: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad: [f32; 3],
}

impl StreamlineResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        data: &TrxGpuData,
    ) -> Self {
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("streamline_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/streamline.wgsl").into()),
        });

        let tube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tube_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tube.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streamline_bind_group_layout"),
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

        let default_uniforms = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            render_style: 0,
            slab_axis: 3,
            slab_min: 0.0,
            slab_max: 0.0,
            tube_radius: 0.5,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            _pad: [0.0; 3],
        };

        let uniform_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("streamline_uniforms_{i}")),
                contents: bytemuck::bytes_of(&default_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("streamline_bind_group_{i}")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffers[i].as_entire_binding(),
                }],
            })
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamline_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // ── Line vertex buffer layouts (position + color + tangent) ──────────
        let position_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };
        let color_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            }],
        };
        let tangent_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        // ── Tube mesh vertex buffer layout (single interleaved buffer) ───────
        let tube_stride = std::mem::size_of::<TubeMeshVertex>() as wgpu::BufferAddress;
        let tube_layout = wgpu::VertexBufferLayout {
            array_stride: tube_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }, // position
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }, // normal
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                }, // color
            ],
        };

        let pipeline = Self::make_line_pipeline(
            device,
            &pipeline_layout,
            &line_shader,
            target_format,
            &[
                position_layout.clone(),
                color_layout.clone(),
                tangent_layout.clone(),
            ],
            wgpu::CompareFunction::Less,
            true,
        );

        let slice_pipeline = Self::make_line_pipeline(
            device,
            &pipeline_layout,
            &line_shader,
            target_format,
            &[position_layout, color_layout, tangent_layout],
            wgpu::CompareFunction::Always,
            false,
        );

        let tube_pipeline = Self::make_tube_pipeline(
            device,
            &pipeline_layout,
            &tube_shader,
            target_format,
            &[tube_layout],
            wgpu::CompareFunction::Less,
            true,
        );

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_positions"),
            contents: bytemuck::cast_slice(&data.positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_colors"),
            contents: bytemuck::cast_slice(&data.colors),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let tangent_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_tangents"),
            contents: bytemuck::cast_slice(&data.tangents),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_indices"),
            contents: bytemuck::cast_slice(&data.all_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            slice_pipeline,
            tube_pipeline,
            position_buffer,
            color_buffer,
            tangent_buffer,
            index_buffer,
            tube_vertex_buffer: None,
            tube_index_buffer: None,
            num_tube_indices: 0,
            uniform_buffers,
            bind_groups,
            num_indices: data.all_indices.len() as u32,
        }
    }

    fn make_line_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
        buffers: &[wgpu::VertexBufferLayout<'_>],
        depth_compare: wgpu::CompareFunction,
        depth_write: bool,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamline_line_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers,
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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

    fn make_tube_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
        buffers: &[wgpu::VertexBufferLayout<'_>],
        depth_compare: wgpu::CompareFunction,
        depth_write: bool,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamline_tube_pipeline"),
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

    /// Update uniforms for a specific viewport.
    /// `viewport`: 0=3D, 1=axial, 2=coronal, 3=sagittal.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        render_style: u32,
        slab_axis: u32,
        slab_min: f32,
        slab_max: f32,
        tube_radius: f32,
        scene_lighting: SceneLightingParams,
    ) {
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.into(),
            render_style,
            slab_axis,
            slab_min,
            slab_max,
            tube_radius,
            ambient_strength: scene_lighting.ambient_strength(),
            key_strength: scene_lighting.key_strength(),
            fill_strength: scene_lighting.fill_strength(),
            headlight_mix: scene_lighting.headlight_mix(),
            specular_strength: scene_lighting.specular_strength(),
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    /// Replace the tube geometry buffers.
    pub fn update_tube_geometry(
        &mut self,
        device: &wgpu::Device,
        vertices: &[TubeMeshVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() {
            self.tube_vertex_buffer = None;
            self.tube_index_buffer = None;
            self.num_tube_indices = 0;
            return;
        }

        // Clamp to the device's 1 GB buffer limit.
        const MAX_BUFFER_BYTES: usize = 1 << 30; // 1 GB
        let max_verts = MAX_BUFFER_BYTES / std::mem::size_of::<TubeMeshVertex>();
        let truncated = vertices.len() > max_verts;
        let vertices = &vertices[..vertices.len().min(max_verts)];

        let mut filtered_indices = Vec::with_capacity(indices.len());
        for tri in indices.chunks_exact(3) {
            if tri.iter().all(|&idx| idx < vertices.len() as u32) {
                filtered_indices.extend_from_slice(tri);
            } else {
                break;
            }
        }

        if truncated {
            log::warn!(
                "Tube geometry truncated to {} vertices due to GPU buffer size limit.",
                vertices.len(),
            );
        }

        self.tube_vertex_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("tube_vertices"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.tube_index_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("tube_indices"),
                contents: bytemuck::cast_slice(&filtered_indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.num_tube_indices = filtered_indices.len() as u32;
    }
}

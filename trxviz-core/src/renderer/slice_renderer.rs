use wgpu::util::DeviceExt;

use crate::data::nifti_data::NiftiVolume;
use crate::renderer::viewport::ViewportUniformSet;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SliceVertex {
    pub position: [f32; 3],
    pub tex_coord: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceUniforms {
    view_proj: [[f32; 4]; 4],
    window_center: f32,
    window_width: f32,
    colormap: u32, // 0=Grayscale, 1=Hot, 2=Cool, 3=RedYellow, 4=BlueLightblue
    opacity: f32,  // alpha multiplier
}

#[derive(Clone, Copy)]
pub enum SliceAxis {
    Axial,
    Coronal,
    Sagittal,
}

/// GPU resources for rendering NIfTI volume slices.
///
/// We maintain one uniform/bind-group pair per viewport so different
/// view-projection matrices can be active simultaneously.
pub struct SliceResources {
    pub pipeline: wgpu::RenderPipeline,
    viewports: ViewportUniformSet<SliceUniforms>,
    pub quad_buffers: [wgpu::Buffer; 3],
    pub quad_index_buffer: wgpu::Buffer,
    pub dims: [usize; 3],
}

impl SliceResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        volume: &NiftiVolume,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slice_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/slice.wgsl").into()),
        });

        // Upload volume as 3D texture
        let dims = volume.dims;
        let texture_size = wgpu::Extent3d {
            width: dims[0] as u32,
            height: dims[1] as u32,
            depth_or_array_layers: dims[2] as u32,
        };

        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&volume.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] as u32 * 4),
                rows_per_image: Some(dims[1] as u32),
            },
            texture_size,
        );

        let texture_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slice_bind_group_layout"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create 4 uniform buffers and bind groups (one per viewport context)
        let default_uniforms = SliceUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            window_center: 0.5,
            window_width: 1.0,
            colormap: 0,
            opacity: 1.0,
        };

        let viewports =
            ViewportUniformSet::new(device, "slice_uniform", &default_uniforms, |i, buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(match i {
                        0 => "slice_3d",
                        1 => "slice_axial",
                        2 => "slice_coronal",
                        _ => "slice_sagittal",
                    }),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
            });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slice_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SliceVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slice_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Quad index buffer
        let quad_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slice_quad_indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Per-slice quad vertex buffers
        let empty_verts = [SliceVertex {
            position: [0.0; 3],
            tex_coord: [0.0; 3],
        }; 4];
        let quad_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("axial_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("coronal_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sagittal_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        Self {
            pipeline,
            viewports,
            quad_buffers,
            quad_index_buffer,
            dims,
        }
    }

    /// Update the quad vertices for a given slice axis and index.
    pub fn update_slice(
        &self,
        queue: &wgpu::Queue,
        axis: SliceAxis,
        slice_index: usize,
        volume: &NiftiVolume,
    ) {
        let (corners, tex_coords) = match axis {
            SliceAxis::Axial => {
                let c = volume.axial_slice_corners(slice_index);
                let w = (slice_index as f32 + 0.5) / self.dims[2] as f32;
                let tc = [[0.0, 0.0, w], [1.0, 0.0, w], [1.0, 1.0, w], [0.0, 1.0, w]];
                (c, tc)
            }
            SliceAxis::Coronal => {
                let c = volume.coronal_slice_corners(slice_index);
                let v = (slice_index as f32 + 0.5) / self.dims[1] as f32;
                let tc = [[0.0, v, 0.0], [1.0, v, 0.0], [1.0, v, 1.0], [0.0, v, 1.0]];
                (c, tc)
            }
            SliceAxis::Sagittal => {
                let c = volume.sagittal_slice_corners(slice_index);
                let u = (slice_index as f32 + 0.5) / self.dims[0] as f32;
                let tc = [[u, 0.0, 0.0], [u, 1.0, 0.0], [u, 1.0, 1.0], [u, 0.0, 1.0]];
                (c, tc)
            }
        };

        let vertices: [SliceVertex; 4] = std::array::from_fn(|i| SliceVertex {
            position: corners[i].into(),
            tex_coord: tex_coords[i],
        });

        let buffer_index = match axis {
            SliceAxis::Axial => 0,
            SliceAxis::Coronal => 1,
            SliceAxis::Sagittal => 2,
        };

        queue.write_buffer(
            &self.quad_buffers[buffer_index],
            0,
            bytemuck::cast_slice(&vertices),
        );
    }

    /// Update the view-projection uniform for a specific viewport context.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        window_center: f32,
        window_width: f32,
        colormap: u32,
        opacity: f32,
    ) {
        let uniforms = SliceUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            window_center,
            window_width,
            colormap,
            opacity,
        };
        self.viewports.update(queue, viewport, &uniforms);
    }

    pub fn bind_group(&self, viewport: usize) -> &wgpu::BindGroup {
        self.viewports.bind_group(viewport)
    }
}

/// Slice resources are either a scalar 3D-texture path (`File` and
/// `InMemory` backings) or a composite 2D-texture-per-axis path
/// (`Composite` backing produced by `VolumeOverlayStackOp`). The
/// renderer dispatches based on the kind.
pub enum SliceResourceKind {
    Scalar(SliceResources),
    Composite(crate::renderer::slice_composite::CompositeSliceResources),
}

/// Wrapper for multiple slice resources (one per loaded NIfTI volume
/// or workflow-produced volume). Needed because
/// egui_wgpu::CallbackResources is a TypeMap.
pub struct AllSliceResources {
    pub entries: Vec<(usize, SliceResourceKind)>, // (slice_key, resources)
}

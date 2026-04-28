//! Composite slice rendering for `VolumeBacking::Composite` sources.
//!
//! Layers are composed on the CPU into a 2D RGBA buffer per displayed
//! axis (one per axial/coronal/sagittal), uploaded into a 2D texture,
//! and drawn via a passthrough shader. This keeps slider scrubbing
//! cheap (~256² samples per layer per axis) and lets the same
//! `composite_slice_into` helper feed both the live renderer and the
//! GLB exporter.

use glam::{Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::data::loaded_files::VolumeColormap;
use crate::renderer::slice_renderer::{SliceAxis, SliceVertex};
use crate::renderer::viewport::ViewportUniformSet;
use crate::workflow::{CompositeVolumeStack, Interp, VolumeOverlayLayerConfig};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    view_proj: [[f32; 4]; 4],
    opacity: f32,
    _pad: [f32; 3],
}

/// One axis-aligned slice plane: a 2D RGBA texture sized to the
/// matching plane of the stack's base grid.
struct AxisResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_groups: [wgpu::BindGroup; 4],
    size: [u32; 2],
}

pub struct CompositeSliceResources {
    pub pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    viewports: ViewportUniformSet<CompositeUniforms>,
    axes: [AxisResources; 3],
    pub quad_buffers: [wgpu::Buffer; 3],
    pub quad_index_buffer: wgpu::Buffer,
    pub dims: [usize; 3],
}

impl CompositeSliceResources {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        stack: &CompositeVolumeStack,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slice_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/slice_composite.wgsl").into(),
            ),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("slice_composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slice_composite_bind_group_layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Allocate one 2D texture per axis sized to that axis's plane
        // in the base grid.
        let make_axis = |axis: SliceAxis| -> AxisResources {
            let (w, h) = plane_size(stack.dims, axis);
            create_axis_resources(device, &bind_group_layout, &sampler, w, h)
        };
        let axes = [
            make_axis(SliceAxis::Axial),
            make_axis(SliceAxis::Coronal),
            make_axis(SliceAxis::Sagittal),
        ];

        // Per-viewport uniform buffers; the bind groups in
        // `viewports` reference axis 0's texture, but at draw time
        // we'll bind the per-axis bind group from `axes`. The
        // viewport uniform buffers are reused via `update`.
        let default_uniforms = CompositeUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            opacity: 1.0,
            _pad: [0.0; 3],
        };
        let viewports = ViewportUniformSet::new(
            device,
            "slice_composite_uniform",
            &default_uniforms,
            |_i, buffer| {
                // Bind group placeholders pointing at the first axis's
                // view; never actually used to draw — the per-axis
                // bind groups in `axes` are what we bind. We need
                // *some* bind group here to satisfy the API.
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("slice_composite_uniform_only"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&axes[0].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
            },
        );

        // Rebuild each axis's per-viewport bind groups now that the
        // viewport uniform buffers exist — the placeholder ones from
        // `create_axis_resources` reference a dummy uniform buffer.
        let mut axes = axes;
        for axis in 0..3 {
            axes[axis].bind_groups = std::array::from_fn(|vp| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("slice_composite_axis_bind_group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: viewports.buffer(vp).as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&axes[axis].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
            });
        }

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slice_composite_pipeline_layout"),
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
                    // Only the first 2 components of the SliceVertex
                    // tex_coord are used in this shader; the third is
                    // ignored. Reusing the layout means SliceVertex
                    // can be shared between scalar and composite paths.
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slice_composite_pipeline"),
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

        let quad_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slice_composite_quad_indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let empty_verts = [SliceVertex {
            position: [0.0; 3],
            tex_coord: [0.0; 3],
        }; 4];
        let quad_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("composite_axial_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("composite_coronal_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("composite_sagittal_quad"),
                contents: bytemuck::cast_slice(&empty_verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            viewports,
            axes,
            quad_buffers,
            quad_index_buffer,
            dims: stack.dims,
        }
    }

    /// Recompose and re-upload the 2D RGBA slice for the given axis.
    /// Also rewrites the quad vertex buffer for that axis.
    pub fn update_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        axis: SliceAxis,
        slice_index: usize,
        stack: &CompositeVolumeStack,
    ) {
        let buffer_index = axis_index(axis);

        // Rebuild axis texture if dims changed (defensive — handle
        // changes already trigger a fresh CompositeSliceResources, but
        // allow a clean re-init if anything slipped through).
        let (w, h) = plane_size(stack.dims, axis);
        if self.axes[buffer_index].size != [w, h] {
            self.axes[buffer_index] =
                create_axis_resources(device, &self.bind_group_layout, &self.sampler, w, h);
            // Re-bind the new texture's view into each viewport's
            // bind group, pointing at the real uniform buffers.
            self.axes[buffer_index].bind_groups = std::array::from_fn(|vp| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("slice_composite_axis_bind_group_resized"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.viewports.buffer(vp).as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.axes[buffer_index].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            });
        }

        // CPU composite into an RGBA8 buffer.
        let mut buf = vec![0u8; (w as usize) * (h as usize) * 4];
        composite_slice_into(&mut buf, stack, axis, slice_index);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.axes[buffer_index].texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &buf,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        // Quad geometry (corners in RAS).
        let corners = slice_corners(stack.dims, stack.voxel_to_ras, axis, slice_index);
        let uvs: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let vertices: [SliceVertex; 4] = std::array::from_fn(|i| SliceVertex {
            position: corners[i].into(),
            tex_coord: uvs[i],
        });
        queue.write_buffer(
            &self.quad_buffers[buffer_index],
            0,
            bytemuck::cast_slice(&vertices),
        );
    }

    /// Update the per-viewport uniform — `view_proj` and the
    /// stack-level opacity multiplier (the per-layer opacities are
    /// already baked into the composite RGBA).
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        opacity: f32,
    ) {
        let uniforms = CompositeUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            opacity,
            _pad: [0.0; 3],
        };
        self.viewports.update(queue, viewport, &uniforms);
    }

    /// Per-axis bind group for the given viewport. The renderer binds
    /// this before drawing each axis's quad, so the shader samples the
    /// correct 2D texture.
    pub fn bind_group(&self, viewport: usize, axis: SliceAxis) -> &wgpu::BindGroup {
        &self.axes[axis_index(axis)].bind_groups[viewport]
    }
}

fn axis_index(axis: SliceAxis) -> usize {
    match axis {
        SliceAxis::Axial => 0,
        SliceAxis::Coronal => 1,
        SliceAxis::Sagittal => 2,
    }
}

fn plane_size(dims: [usize; 3], axis: SliceAxis) -> (u32, u32) {
    match axis {
        SliceAxis::Axial => (dims[0] as u32, dims[1] as u32),
        SliceAxis::Coronal => (dims[0] as u32, dims[2] as u32),
        SliceAxis::Sagittal => (dims[1] as u32, dims[2] as u32),
    }
}

fn create_axis_resources(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    w: u32,
    h: u32,
) -> AxisResources {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite_slice_axis_tex"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Placeholder bind groups — the real ones are attached by
    // `rebind_axes` once the parent's uniform buffers are in place.
    let bind_groups = std::array::from_fn(|_| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("placeholder_axis_bind_group"),
            layout: bind_group_layout,
            entries: &[
                // No uniform binding here — replaced by rebind_axes
                // before first draw. We just need any valid resource
                // so creation succeeds; uniform will be a tiny dummy.
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("composite_axis_dummy_uniform"),
                            contents: bytemuck::bytes_of(&CompositeUniforms {
                                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                                opacity: 1.0,
                                _pad: [0.0; 3],
                            }),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        })
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    });
    AxisResources {
        texture,
        view,
        bind_groups,
        size: [w, h],
    }
}

/// Slice plane corners in RAS, identical math to the scalar-volume
/// path in `nifti_data.rs`.
fn slice_corners(
    dims: [usize; 3],
    voxel_to_ras: glam::Mat4,
    axis: SliceAxis,
    slice_index: usize,
) -> [Vec3; 4] {
    let to_world = |voxel: Vec3| -> Vec3 { voxel_to_ras.transform_point3(voxel) };
    match axis {
        SliceAxis::Axial => {
            let kf = slice_index as f32;
            let i0 = -0.5;
            let i1 = dims[0] as f32 - 0.5;
            let j0 = -0.5;
            let j1 = dims[1] as f32 - 0.5;
            [
                to_world(Vec3::new(i0, j0, kf)),
                to_world(Vec3::new(i1, j0, kf)),
                to_world(Vec3::new(i1, j1, kf)),
                to_world(Vec3::new(i0, j1, kf)),
            ]
        }
        SliceAxis::Coronal => {
            let jf = slice_index as f32;
            let i0 = -0.5;
            let i1 = dims[0] as f32 - 0.5;
            let k0 = -0.5;
            let k1 = dims[2] as f32 - 0.5;
            [
                to_world(Vec3::new(i0, jf, k0)),
                to_world(Vec3::new(i1, jf, k0)),
                to_world(Vec3::new(i1, jf, k1)),
                to_world(Vec3::new(i0, jf, k1)),
            ]
        }
        SliceAxis::Sagittal => {
            let if_ = slice_index as f32;
            let j0 = -0.5;
            let j1 = dims[1] as f32 - 0.5;
            let k0 = -0.5;
            let k1 = dims[2] as f32 - 0.5;
            [
                to_world(Vec3::new(if_, j0, k0)),
                to_world(Vec3::new(if_, j1, k0)),
                to_world(Vec3::new(if_, j1, k1)),
                to_world(Vec3::new(if_, j0, k1)),
            ]
        }
    }
}

/// Compose the requested 2D slice plane into `out` (RGBA8, length =
/// w * h * 4 where (w, h) match `plane_size(stack.dims, axis)`).
///
/// Walks the base grid's plane in voxel coords. For each output
/// pixel, transforms to RAS, then for each enabled layer:
/// - Layer 0: direct lookup at the output's voxel coord.
/// - Other layers: RAS → that layer's voxel via `ras_to_voxel`,
///   sampled with `Interp::Trilinear` or `Interp::Nearest`.
/// Out-of-bounds samples contribute alpha = 0 and are skipped.
///
/// Each layer's value is windowed → thresholded → mapped through its
/// colormap → premultiplied by `opacity` → alpha-over composited atop
/// the running color.
pub fn composite_slice_into(
    out: &mut [u8],
    stack: &CompositeVolumeStack,
    axis: SliceAxis,
    slice_index: usize,
) {
    let (w, h) = plane_size(stack.dims, axis);
    debug_assert_eq!(out.len(), (w as usize) * (h as usize) * 4);

    // Pre-compute per-layer ras_to_voxel for non-base layers.
    let layer_ras_to_voxel: Vec<glam::Mat4> = stack
        .layers
        .iter()
        .map(|(s, _)| s.voxel_to_ras.inverse())
        .collect();

    let stack_voxel_to_ras = stack.voxel_to_ras;
    let dims = stack.dims;

    for j in 0..h as usize {
        for i in 0..w as usize {
            // Map (i, j) of the 2D plane to a base-grid voxel coord.
            let base_voxel = match axis {
                SliceAxis::Axial => Vec3::new(i as f32, j as f32, slice_index as f32),
                SliceAxis::Coronal => Vec3::new(i as f32, slice_index as f32, j as f32),
                SliceAxis::Sagittal => Vec3::new(slice_index as f32, i as f32, j as f32),
            };
            let ras = stack_voxel_to_ras.transform_point3(base_voxel);

            // Running premultiplied RGBA, starting fully transparent.
            let mut acc = Vec4::ZERO;

            for (layer_idx, (scalars, cfg)) in stack.layers.iter().enumerate() {
                if !cfg.enabled {
                    continue;
                }

                let value: f32 = if layer_idx == 0 {
                    // Direct base-grid lookup: i, j, slice_index in
                    // the layer's own indexing.
                    let (xi, yi, zi) = match axis {
                        SliceAxis::Axial => (i, j, slice_index),
                        SliceAxis::Coronal => (i, slice_index, j),
                        SliceAxis::Sagittal => (slice_index, i, j),
                    };
                    if xi >= dims[0] || yi >= dims[1] || zi >= dims[2] {
                        continue;
                    }
                    scalars.values[xi + dims[0] * (yi + dims[1] * zi)]
                } else {
                    let layer_voxel = layer_ras_to_voxel[layer_idx].transform_point3(ras);
                    let sampled = match cfg.interpolation {
                        Interp::Trilinear => crate::data::sampling::trilinear(scalars, layer_voxel),
                        Interp::Nearest => crate::data::sampling::nearest(scalars, layer_voxel),
                    };
                    match sampled {
                        Some(v) => v,
                        None => continue,
                    }
                };

                // Threshold gate.
                if !value.is_finite() || value < cfg.threshold_min || value > cfg.threshold_max {
                    continue;
                }

                // Window/level → [0, 1].
                let lo = cfg.window_center - cfg.window_width * 0.5;
                let hi = cfg.window_center + cfg.window_width * 0.5;
                let t = ((value - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);

                // Colormap.
                let rgb = colormap_sample(cfg.colormap, t);

                // Per-pixel alpha. Base layer (index 0) paints with
                // uniform `opacity` so the anatomy fills the slice
                // solidly. Overlay layers modulate alpha by the
                // windowed value `t`, mirroring mrview's behavior:
                // zero-valued voxels are transparent so the base
                // shows through, bright voxels approach full
                // opacity. Avoids the "opaque black background of
                // the overlay obscures everything" pitfall.
                let alpha = if layer_idx == 0 {
                    cfg.opacity
                } else {
                    cfg.opacity * t
                };

                // Porter-Duff "src over dst" with `src` = current
                // layer (front), `acc` = accumulated layers (back).
                // Premultiplied form: out = src + dst * (1 - src.a).
                let src = Vec4::new(rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha);
                acc = src + acc * (1.0 - alpha);
            }

            let idx = (j * w as usize + i) * 4;
            out[idx] = (acc.x.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 1] = (acc.y.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 2] = (acc.z.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 3] = (acc.w.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
}

/// CPU mirror of the WGSL colormap functions in `slice.wgsl`.
fn colormap_sample(cmap: VolumeColormap, t: f32) -> [f32; 3] {
    match cmap {
        VolumeColormap::Grayscale => [t, t, t],
        VolumeColormap::Hot => [
            (t * 2.5).clamp(0.0, 1.0),
            (t * 2.5 - 1.0).clamp(0.0, 1.0),
            (t * 5.0 - 4.0).clamp(0.0, 1.0),
        ],
        VolumeColormap::Cool => [t, 1.0 - t, 1.0],
        VolumeColormap::RedYellow => [1.0, t, 0.0],
        VolumeColormap::BlueLightblue => [0.0, t, 1.0],
    }
}

// Holding `VolumeOverlayLayerConfig` import even if unused after edits
// — keeps the API self-contained.
#[allow(dead_code)]
const _: Option<VolumeOverlayLayerConfig> = None;

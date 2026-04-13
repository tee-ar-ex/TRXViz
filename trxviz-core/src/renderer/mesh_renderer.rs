use std::collections::HashMap;

use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::data::bundle_mesh::{BundleMesh, BundleMeshVertex};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::lighting::{SceneLightingParams, WorkflowRender3D};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshUniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
    camera_pos: [f32; 3],
    shininess: f32,
    map_opacity: f32,
    map_threshold: f32,
    scalar_min: f32,
    scalar_max: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    scalar_enabled: u32,
    colormap: u32,
    _pad0: f32,
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    post_params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[derive(Clone)]
pub struct MeshDrawStyle {
    pub color: [f32; 4],
    pub scalar_min: f32,
    pub scalar_max: f32,
    pub scalar_enabled: bool,
    pub colormap: SurfaceColormap,
    pub gloss: f32,
    pub map_opacity: f32,
    pub map_threshold: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SurfaceColormap {
    BlueWhiteRed = 0,
    Viridis = 1,
    Inferno = 2,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BundleUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    opacity: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad0: [f32; 7],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    post_params: [f32; 4],
}

struct BundleGpuSurface {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    transparent_index_buffers: [wgpu::Buffer; 6],
    num_indices: u32,
    centroid: [f32; 3],
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct MeshResources {
    pub opaque_pipeline: wgpu::RenderPipeline,
    pub transparent_pipeline: wgpu::RenderPipeline,
    surfaces: HashMap<usize, GpuSurface>,
    // Bundle surfaces — keyed by file_id, one Vec per TRX file
    pub bundle_opaque_pipeline: wgpu::RenderPipeline,
    pub bundle_transparent_pipeline: wgpu::RenderPipeline,
    bundle_surfaces: HashMap<usize, Vec<BundleGpuSurface>>,
}

struct GpuSurface {
    vertex_buffer: wgpu::Buffer,
    scalar_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    transparent_index_buffers: [wgpu::Buffer; 6],
    num_indices: u32,
    centroid: [f32; 3],
    uniform_buffers: [wgpu::Buffer; 4],
    bind_groups: [wgpu::BindGroup; 4],
}

impl MeshResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_bind_group_layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_mesh_pipeline = |label: &'static str, depth_write_enabled: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
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
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            }],
                        },
                    ],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled,
                    depth_compare: wgpu::CompareFunction::LessEqual,
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
            })
        };
        let opaque_pipeline = make_mesh_pipeline("mesh_opaque_pipeline", true);
        let transparent_pipeline = make_mesh_pipeline("mesh_transparent_pipeline", false);

        // ── Bundle mesh pipeline ──────────────────────────────────────────────
        let bundle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bundle_mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bundle_mesh.wgsl").into()),
        });

        let bundle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bundle_mesh_bgl"),
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

        let bundle_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bundle_mesh_pl_layout"),
            bind_group_layouts: &[&bundle_bgl],
            push_constant_ranges: &[],
        });

        let bundle_stride = std::mem::size_of::<BundleMeshVertex>() as wgpu::BufferAddress;
        let make_bundle_pipeline = |label: &'static str, depth_write_enabled: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bundle_pl_layout),
                vertex: wgpu::VertexState {
                    module: &bundle_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: bundle_stride,
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
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, // two-sided (handled in shader)
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &bundle_shader,
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
            })
        };
        let bundle_opaque_pipeline = make_bundle_pipeline("bundle_mesh_opaque_pipeline", true);
        let bundle_transparent_pipeline =
            make_bundle_pipeline("bundle_mesh_transparent_pipeline", false);

        Self {
            opaque_pipeline,
            transparent_pipeline,
            surfaces: HashMap::new(),
            bundle_opaque_pipeline,
            bundle_transparent_pipeline,
            bundle_surfaces: HashMap::new(),
        }
    }

    pub fn add_surface(
        &mut self,
        surface_id: usize,
        device: &wgpu::Device,
        surface: &GiftiSurfaceData,
    ) {
        let vertices: Vec<MeshVertex> = surface
            .vertices
            .iter()
            .zip(surface.normals.iter())
            .map(|(position, normal)| MeshVertex {
                position: *position,
                normal: *normal,
            })
            .collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_indices"),
            contents: bytemuck::cast_slice(&surface.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let transparent_index_orders =
            build_transparent_index_orders(&surface.vertices, &surface.indices);
        let transparent_index_buffers: [wgpu::Buffer; 6] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("gifti_surface_indices_sorted_{i}")),
                contents: bytemuck::cast_slice(&transparent_index_orders[i]),
                usage: wgpu::BufferUsages::INDEX,
            })
        });
        let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_scalars"),
            contents: bytemuck::cast_slice(&vec![0.0f32; surface.vertices.len()]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = self.opaque_pipeline.get_bind_group_layout(0);
        let default_uniforms = MeshUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color: [0.7, 0.7, 0.7, 1.0],
            camera_pos: [0.0, 0.0, 1.0],
            shininess: 24.0,
            map_opacity: 1.0,
            map_threshold: 0.0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            scalar_enabled: 0,
            colormap: SurfaceColormap::BlueWhiteRed as u32,
            _pad0: 0.0,
            fog_color: [0.0, 0.0, 0.0, 0.0],
            fog_params: [0.0, 1.0, 0.0, 0.0],
            post_params: [1.0, 1.0, 0.12, 0.0],
        };

        let uniform_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh_uniforms_{i}")),
                contents: bytemuck::bytes_of(&default_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("mesh_bind_group_{i}")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffers[i].as_entire_binding(),
                }],
            })
        });

        self.surfaces.insert(
            surface_id,
            GpuSurface {
                vertex_buffer,
                scalar_buffer,
                index_buffer,
                transparent_index_buffers,
                num_indices: surface.indices.len() as u32,
                centroid: compute_centroid(&surface.vertices),
                uniform_buffers,
                bind_groups,
            },
        );
    }

    pub fn update_surface_uniforms(
        &self,
        queue: &wgpu::Queue,
        surface_index: usize,
        viewport: usize,
        view_proj: glam::Mat4,
        style: &MeshDrawStyle,
        camera_pos: glam::Vec3,
        scene_lighting: SceneLightingParams,
        render_3d: &WorkflowRender3D,
        fog_near: f32,
        fog_far: f32,
    ) {
        if let Some(surface) = self.surfaces.get(&surface_index) {
            let uniforms = MeshUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                color: style.color,
                camera_pos: camera_pos.into(),
                shininess: 8.0 + 80.0 * style.gloss.clamp(0.0, 1.0),
                map_opacity: style.map_opacity.clamp(0.0, 1.0),
                map_threshold: style.map_threshold.clamp(0.0, 1.0),
                scalar_min: style.scalar_min,
                scalar_max: style.scalar_max,
                ambient_strength: scene_lighting.ambient_strength(),
                key_strength: scene_lighting.key_strength(),
                fill_strength: scene_lighting.fill_strength(),
                headlight_mix: scene_lighting.headlight_mix(),
                specular_strength: scene_lighting.specular_strength()
                    * (0.15 + 0.85 * style.gloss.clamp(0.0, 1.0)),
                scalar_enabled: if style.scalar_enabled { 1 } else { 0 },
                colormap: style.colormap as u32,
                _pad0: 0.0,
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
            };
            queue.write_buffer(
                &surface.uniform_buffers[viewport],
                0,
                bytemuck::bytes_of(&uniforms),
            );
        }
    }

    pub fn update_surface_scalars(
        &self,
        queue: &wgpu::Queue,
        surface_index: usize,
        scalars: &[f32],
    ) {
        if let Some(surface) = self.surfaces.get(&surface_index) {
            queue.write_buffer(&surface.scalar_buffer, 0, bytemuck::cast_slice(scalars));
        }
    }

    pub fn paint_opaque(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        viewport: usize,
        draw_calls: &[(usize, MeshDrawStyle)],
    ) {
        render_pass.set_pipeline(&self.opaque_pipeline);
        for (surface_index, style) in draw_calls {
            if let Some(surface) = self.surfaces.get(surface_index) {
                if style.color[3] <= 0.999 {
                    continue;
                }
                render_pass.set_bind_group(0, &surface.bind_groups[viewport], &[]);
                render_pass.set_vertex_buffer(0, surface.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                render_pass
                    .set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..surface.num_indices, 0, 0..1);
            }
        }
    }

    pub fn paint_transparent(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        viewport: usize,
        draw_calls: &[(usize, MeshDrawStyle)],
        bundle_draw_calls: &[(usize, f32)],
        camera_pos: glam::Vec3,
        camera_dir: glam::Vec3,
    ) {
        let transparent_order = transparent_view_bucket(camera_dir);
        enum TransparentDraw<'a> {
            Surface {
                surface: &'a GpuSurface,
                viewport: usize,
                depth: f32,
            },
            Bundle {
                surface: &'a BundleGpuSurface,
                depth: f32,
            },
        }

        let mut draws = Vec::new();
        for (surface_index, style) in draw_calls {
            if style.color[3] <= 0.001 || style.color[3] >= 0.999 {
                continue;
            }
            if let Some(surface) = self.surfaces.get(surface_index) {
                let centroid = Vec3::from_array(surface.centroid);
                let depth = (centroid - camera_pos).dot(camera_dir);
                draws.push(TransparentDraw::Surface {
                    surface,
                    viewport,
                    depth,
                });
            }
        }
        for (file_id, opacity) in bundle_draw_calls {
            if *opacity <= 0.001 || *opacity >= 0.999 {
                continue;
            }
            if let Some(surfaces) = self.bundle_surfaces.get(file_id) {
                for surface in surfaces {
                    let centroid = Vec3::from_array(surface.centroid);
                    let depth = (centroid - camera_pos).dot(camera_dir);
                    draws.push(TransparentDraw::Bundle { surface, depth });
                }
            }
        }

        draws.sort_by(|lhs, rhs| {
            let lhs_depth = match lhs {
                TransparentDraw::Surface { depth, .. } | TransparentDraw::Bundle { depth, .. } => {
                    *depth
                }
            };
            let rhs_depth = match rhs {
                TransparentDraw::Surface { depth, .. } | TransparentDraw::Bundle { depth, .. } => {
                    *depth
                }
            };
            rhs_depth.total_cmp(&lhs_depth)
        });

        let mut using_bundle_pipeline = false;
        for draw in draws {
            match draw {
                TransparentDraw::Surface {
                    surface, viewport, ..
                } => {
                    if using_bundle_pipeline {
                        render_pass.set_pipeline(&self.transparent_pipeline);
                        using_bundle_pipeline = false;
                    } else if !using_bundle_pipeline {
                        render_pass.set_pipeline(&self.transparent_pipeline);
                    }
                    render_pass.set_bind_group(0, &surface.bind_groups[viewport], &[]);
                    render_pass.set_vertex_buffer(0, surface.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                    render_pass.set_index_buffer(
                        surface.transparent_index_buffers[transparent_order].slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..surface.num_indices, 0, 0..1);
                }
                TransparentDraw::Bundle { surface, .. } => {
                    if !using_bundle_pipeline {
                        render_pass.set_pipeline(&self.bundle_transparent_pipeline);
                        using_bundle_pipeline = true;
                    }
                    render_pass.set_bind_group(0, &surface.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, surface.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        surface.transparent_index_buffers[transparent_order].slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..surface.num_indices, 0, 0..1);
                }
            }
        }
    }

    // ── Bundle meshes (voxel density surfaces) ───────────────────────────────

    fn make_bundle_gpu_surface(
        &self,
        device: &wgpu::Device,
        mesh: &BundleMesh,
        label: &str,
    ) -> BundleGpuSurface {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_verts_{label}")),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_idx_{label}")),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bundle_positions: Vec<[f32; 3]> =
            mesh.vertices.iter().map(|vertex| vertex.position).collect();
        let transparent_index_orders =
            build_transparent_index_orders(&bundle_positions, &mesh.indices);
        let transparent_index_buffers: [wgpu::Buffer; 6] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("bundle_idx_sorted_{label}_{i}")),
                contents: bytemuck::cast_slice(&transparent_index_orders[i]),
                usage: wgpu::BufferUsages::INDEX,
            })
        });
        let default_uniforms = BundleUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            opacity: 0.5,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            _pad0: [0.0; 7],
            fog_color: [0.0, 0.0, 0.0, 0.0],
            fog_params: [0.0, 1.0, 0.0, 0.0],
            post_params: [1.0, 1.0, 0.12, 0.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_uni_{label}")),
            contents: bytemuck::bytes_of(&default_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = self.bundle_opaque_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bundle_bg_{label}")),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        BundleGpuSurface {
            vertex_buffer,
            index_buffer,
            transparent_index_buffers,
            num_indices: mesh.indices.len() as u32,
            centroid: compute_centroid(&bundle_positions),
            uniform_buffer,
            bind_group,
        }
    }

    /// Replace the bundle surfaces for one TRX file.
    /// `meshes` is a slice of `(mesh, label)` pairs — one entry per source
    /// (e.g. one for all streamlines, or one per group).
    pub fn set_bundle_meshes(
        &mut self,
        file_id: usize,
        device: &wgpu::Device,
        meshes: &[(BundleMesh, String)],
    ) {
        let surfaces = meshes
            .iter()
            .map(|(m, label)| self.make_bundle_gpu_surface(device, m, label))
            .collect();
        self.bundle_surfaces.insert(file_id, surfaces);
    }

    /// Update per-frame uniforms for one TRX file's bundle surfaces.
    pub fn update_bundle_uniforms(
        &self,
        file_id: usize,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        opacity: f32,
        scene_lighting: SceneLightingParams,
        render_3d: &WorkflowRender3D,
        fog_near: f32,
        fog_far: f32,
    ) {
        if let Some(surfaces) = self.bundle_surfaces.get(&file_id) {
            let uniforms = BundleUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                camera_pos: camera_pos.into(),
                opacity,
                ambient_strength: scene_lighting.ambient_strength(),
                key_strength: scene_lighting.key_strength(),
                fill_strength: scene_lighting.fill_strength(),
                headlight_mix: scene_lighting.headlight_mix(),
                specular_strength: scene_lighting.specular_strength(),
                _pad0: [0.0; 7],
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
            };
            for bs in surfaces {
                queue.write_buffer(&bs.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            }
        }
    }

    /// Draw bundle surfaces only for the active file ids in the current frame.
    pub fn paint_bundle_opaque(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        draw_calls: &[(usize, f32)],
    ) {
        if self.bundle_surfaces.is_empty() || draw_calls.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.bundle_opaque_pipeline);
        for (file_id, opacity) in draw_calls {
            if *opacity <= 0.999 {
                continue;
            }
            if let Some(surfaces) = self.bundle_surfaces.get(file_id) {
                for bs in surfaces {
                    render_pass.set_bind_group(0, &bs.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, bs.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(bs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..bs.num_indices, 0, 0..1);
                }
            }
        }
    }

    pub fn clear_bundle_mesh(&mut self, file_id: usize) {
        self.bundle_surfaces.remove(&file_id);
    }
}

fn compute_centroid(vertices: &[[f32; 3]]) -> [f32; 3] {
    if vertices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut sum = Vec3::ZERO;
    for vertex in vertices {
        sum += Vec3::from_array(*vertex);
    }
    (sum / vertices.len() as f32).to_array()
}

fn build_transparent_index_orders(vertices: &[[f32; 3]], indices: &[u32]) -> [Vec<u32>; 6] {
    let triangle_count = indices.len() / 3;
    let triangle_order_by_axis: [Vec<(f32, usize)>; 3] = std::array::from_fn(|axis| {
        let mut order: Vec<(f32, usize)> = (0..triangle_count)
            .map(|tri_index| {
                let base = tri_index * 3;
                let a = Vec3::from(vertices[indices[base] as usize]);
                let b = Vec3::from(vertices[indices[base + 1] as usize]);
                let c = Vec3::from(vertices[indices[base + 2] as usize]);
                let centroid = (a + b + c) / 3.0;
                (centroid[axis], tri_index)
            })
            .collect();
        order.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
        order
    });

    std::array::from_fn(|bucket| {
        let axis = bucket % 3;
        let descending = bucket >= 3;
        let tri_indices: Box<dyn Iterator<Item = usize>> = if descending {
            Box::new(
                triangle_order_by_axis[axis]
                    .iter()
                    .rev()
                    .map(|(_, tri_index)| *tri_index),
            )
        } else {
            Box::new(
                triangle_order_by_axis[axis]
                    .iter()
                    .map(|(_, tri_index)| *tri_index),
            )
        };

        let mut sorted = Vec::with_capacity(indices.len());
        for tri_index in tri_indices {
            let base = tri_index * 3;
            sorted.extend_from_slice(&indices[base..base + 3]);
        }
        sorted
    })
}

fn transparent_view_bucket(view_dir: glam::Vec3) -> usize {
    let abs = view_dir.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        if view_dir.x >= 0.0 { 3 } else { 0 }
    } else if abs.y >= abs.z {
        if view_dir.y >= 0.0 { 4 } else { 1 }
    } else if view_dir.z >= 0.0 {
        5
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::{build_transparent_index_orders, transparent_view_bucket};
    use glam::Vec3;

    #[test]
    fn transparent_bucket_uses_dominant_axis_and_sign() {
        assert_eq!(transparent_view_bucket(Vec3::new(-1.0, 0.2, 0.1)), 0);
        assert_eq!(transparent_view_bucket(Vec3::new(1.0, 0.2, 0.1)), 3);
        assert_eq!(transparent_view_bucket(Vec3::new(0.2, -1.0, 0.1)), 1);
        assert_eq!(transparent_view_bucket(Vec3::new(0.2, 1.0, 0.1)), 4);
        assert_eq!(transparent_view_bucket(Vec3::new(0.2, 0.1, -1.0)), 2);
        assert_eq!(transparent_view_bucket(Vec3::new(0.2, 0.1, 1.0)), 5);
    }

    #[test]
    fn transparent_orders_sort_triangle_centroids_by_axis() {
        let vertices = vec![
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.5, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let orders = build_transparent_index_orders(&vertices, &indices);

        assert_eq!(orders[0], vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(orders[3], vec![3, 4, 5, 0, 1, 2]);
    }
}

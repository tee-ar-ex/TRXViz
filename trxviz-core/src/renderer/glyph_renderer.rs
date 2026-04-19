use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::data::odx_data::{OdfChunkWorklist, OdxGlyphSourceKind};
use crate::data::orientation_field::{BoundaryContactField, BoundaryGlyphColorMode};
use crate::lighting::{SceneLightingParams, WorkflowRender3D};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OdxGpuGlyphMode {
    PreSampledOdf,
    OdfSliceGather,
    ShCompute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OdxGlyphResourceKey {
    pub scene_ptr: usize,
    pub source_kind: OdxGlyphSourceKind,
    pub mode: OdxGpuGlyphMode,
    pub sphere_vertex_count: usize,
    pub sphere_index_count: usize,
    pub sh_order: Option<usize>,
    pub slice_axis: Option<u8>,
    pub slice_index: Option<u32>,
    pub opacity_gate_fingerprint: u64,
    pub size_gate_fingerprint: u64,
}

pub struct GlyphResources {
    pipeline: wgpu::RenderPipeline,
    slice_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    instance_buffer: Option<wgpu::Buffer>,
    amplitude_buffer: Option<wgpu::Buffer>,
    opacity_buffer: Option<wgpu::Buffer>,
    size_buffer: Option<wgpu::Buffer>,
    uniform_buffers: [wgpu::Buffer; 4],
    bind_groups: [wgpu::BindGroup; 4],
    buffer_capacities: GlyphBufferCapacities,
    odx_static_geometry_key: Option<(usize, usize, usize)>,
    odx_mode: Option<OdxGpuGlyphMode>,
    odx_odf_pipeline: wgpu::ComputePipeline,
    odx_odf_bind_group_layout: wgpu::BindGroupLayout,
    odx_odf_source_chunks: Vec<wgpu::Buffer>,
    odx_odf_source_key: Option<(usize, usize, usize, usize)>,
    odx_odf_worklist_buffer: Option<wgpu::Buffer>,
    odx_odf_worklist_capacity_bytes: u64,
    odx_odf_params_buffer: Option<wgpu::Buffer>,
    odx_sh_pipeline: wgpu::ComputePipeline,
    odx_sh_bind_group_layout: wgpu::BindGroupLayout,
    odx_sh_bind_group: Option<wgpu::BindGroup>,
    odx_sh_coeff_buffer: Option<wgpu::Buffer>,
    odx_sh_transform_buffer: Option<wgpu::Buffer>,
    odx_sh_slice_buffer: Option<wgpu::Buffer>,
    odx_sh_params_buffer: Option<wgpu::Buffer>,
    odx_sh_source_bins: u32,
    odx_sh_ncoeffs: u32,
    num_indices: u32,
    num_instances: u32,
    current_bins: u32,
}

#[derive(Default)]
struct GlyphBufferCapacities {
    instance_bytes: u64,
    amplitude_bytes: u64,
    opacity_bytes: u64,
    size_bytes: u64,
}

#[derive(Clone, Copy)]
enum GateBufferKind {
    Opacity,
    Size,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    slab_half_width: f32,
    slab_normal: [f32; 3],
    color_mode: u32,
    slab_center: [f32; 3],
    draw_step: u32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    opacity: f32,
    gloss: f32,
    scale_mul: f32,
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    post_params: [f32; 4],
    opacity_gate: [f32; 4],
    size_gate: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OdxShComputeParams {
    full_bins: u32,
    source_bins: u32,
    ncoeffs: u32,
    slice_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OdxOdfComputeParams {
    counts: [u32; 4],
    scales: [f32; 4],
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

const COMPUTE_WORKGROUP_SIZE_X: u32 = 64;
const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
const MAX_COMPUTE_INVOCATIONS_PER_DISPATCH: usize =
    (COMPUTE_WORKGROUP_SIZE_X as usize) * (MAX_COMPUTE_WORKGROUPS_PER_DIMENSION as usize);

fn max_rows_per_dispatch(row_width: usize) -> usize {
    if row_width == 0 {
        1
    } else {
        (MAX_COMPUTE_INVOCATIONS_PER_DISPATCH / row_width).max(1)
    }
}

impl GlyphResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/glyph.wgsl").into()),
        });
        let odf_gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("odx_odf_gather_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/odx_odf_gather.wgsl").into()),
        });
        let sh_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("odx_sh_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/odx_sh_compute.wgsl").into()),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let odx_sh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("odx_sh_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let odx_odf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("odx_odf_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        let odx_sh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("odx_sh_compute_layout"),
            bind_group_layouts: &[&odx_sh_bgl],
            push_constant_ranges: &[],
        });
        let odx_odf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("odx_odf_compute_layout"),
            bind_group_layouts: &[&odx_odf_bgl],
            push_constant_ranges: &[],
        });

        let uniforms = GlyphUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 1.0],
            slab_half_width: 0.0,
            slab_normal: [0.0, 0.0, 1.0],
            color_mode: 0,
            slab_center: [0.0; 3],
            draw_step: 1,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            opacity: 0.95,
            gloss: 0.0,
            scale_mul: 0.0,
            fog_color: [0.0, 0.0, 0.0, 0.0],
            fog_params: [0.0, 1.0, 0.0, 0.0],
            post_params: [1.0, 1.0, 0.12, 0.0],
            opacity_gate: [0.0, 1.0, 1.0, 1.0],
            size_gate: [0.0, 1.0, 1.0, 1.0],
        };
        let amplitude_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_amplitudes_empty"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let opacity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glyph_opacity_gate_samples_empty"),
            contents: bytemuck::cast_slice(&[f32::NAN]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glyph_size_gate_samples_empty"),
            contents: bytemuck::cast_slice(&[f32::NAN]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: opacity_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: size_buffer.as_entire_binding(),
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
        let odx_sh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("odx_sh_compute_pipeline"),
            layout: Some(&odx_sh_layout),
            module: &sh_compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let odx_odf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("odx_odf_compute_pipeline"),
            layout: Some(&odx_odf_layout),
            module: &odf_gather_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            slice_pipeline,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            amplitude_buffer: Some(amplitude_buffer),
            opacity_buffer: Some(opacity_buffer),
            size_buffer: Some(size_buffer),
            uniform_buffers,
            bind_groups,
            buffer_capacities: GlyphBufferCapacities {
                instance_bytes: 0,
                amplitude_bytes: 16,
                opacity_bytes: std::mem::size_of::<f32>() as u64,
                size_bytes: std::mem::size_of::<f32>() as u64,
            },
            odx_static_geometry_key: None,
            odx_mode: None,
            odx_odf_pipeline,
            odx_odf_bind_group_layout: odx_odf_bgl,
            odx_odf_source_chunks: Vec::new(),
            odx_odf_source_key: None,
            odx_odf_worklist_buffer: None,
            odx_odf_worklist_capacity_bytes: 0,
            odx_odf_params_buffer: None,
            odx_sh_pipeline,
            odx_sh_bind_group_layout: odx_sh_bgl,
            odx_sh_bind_group: None,
            odx_sh_coeff_buffer: None,
            odx_sh_transform_buffer: None,
            odx_sh_slice_buffer: None,
            odx_sh_params_buffer: None,
            odx_sh_source_bins: 0,
            odx_sh_ncoeffs: 0,
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
        self.odx_static_geometry_key = None;
        self.clear_odx_odf_gather_state();
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
        self.buffer_capacities.instance_bytes =
            (instances.len() * std::mem::size_of::<GlyphInstance>()) as u64;

        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glyph_amplitudes"),
            contents: bytemuck::cast_slice(&amplitudes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.amplitude_buffer = Some(amplitude_buffer);
        self.buffer_capacities.amplitude_bytes =
            (amplitudes.len() * std::mem::size_of::<f32>()) as u64;
        self.opacity_buffer = Some(make_gate_sample_buffer(
            device,
            "glyph_opacity_gate_samples",
            &vec![f32::NAN; instances.len().max(1)],
        ));
        self.buffer_capacities.opacity_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.size_buffer = Some(make_gate_sample_buffer(
            device,
            "glyph_size_gate_samples",
            &vec![f32::NAN; instances.len().max(1)],
        ));
        self.buffer_capacities.size_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.num_indices = indices.len() as u32;
        self.num_instances = instances.len() as u32;
        self.current_bins = nbins;
        self.odx_mode = None;
        self.clear_odx_sh_compute();
        self.rebuild_bind_groups(device, "glyph_bg_dynamic");
    }

    pub fn set_odx_field(
        &mut self,
        device: &wgpu::Device,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
        instances: &[GlyphInstance],
        amplitudes: &[f32],
        opacity_samples: Option<&[f32]>,
        size_samples: Option<&[f32]>,
    ) {
        if instances.is_empty() {
            self.clear();
            return;
        }

        self.odx_static_geometry_key = None;
        self.clear_odx_odf_gather_state();
        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_vertices"),
                contents: bytemuck::cast_slice(sphere_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_indices"),
                contents: bytemuck::cast_slice(sphere_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.instance_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_instances"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
        self.buffer_capacities.instance_bytes =
            (instances.len() * std::mem::size_of::<GlyphInstance>()) as u64;

        self.amplitude_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_amplitudes"),
                contents: bytemuck::cast_slice(amplitudes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.buffer_capacities.amplitude_bytes =
            (amplitudes.len() * std::mem::size_of::<f32>()) as u64;
        self.opacity_buffer = Some(make_gate_sample_buffer(
            device,
            "odx_glyph_opacity_gate_samples",
            gate_samples_for_instances(instances.len(), opacity_samples).as_slice(),
        ));
        self.buffer_capacities.opacity_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.size_buffer = Some(make_gate_sample_buffer(
            device,
            "odx_glyph_size_gate_samples",
            gate_samples_for_instances(instances.len(), size_samples).as_slice(),
        ));
        self.buffer_capacities.size_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.num_indices = sphere_indices.len() as u32;
        self.num_instances = instances.len() as u32;
        self.current_bins = sphere_vertices.len() as u32;
        self.odx_mode = Some(OdxGpuGlyphMode::PreSampledOdf);
        self.clear_odx_sh_compute();
        self.rebuild_bind_groups(device, "odx_glyph_bg");
    }

    pub fn set_odx_odf_volume(
        &mut self,
        device: &wgpu::Device,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
        instances: &[GlyphInstance],
        amplitudes: &[f32],
        opacity_samples: Option<&[f32]>,
        size_samples: Option<&[f32]>,
    ) {
        self.set_odx_field(
            device,
            sphere_vertices,
            sphere_indices,
            instances,
            amplitudes,
            opacity_samples,
            size_samples,
        );
        self.odx_mode = Some(OdxGpuGlyphMode::PreSampledOdf);
    }

    pub fn set_odx_slice_odf(
        &mut self,
        device: &wgpu::Device,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
        instances: &[GlyphInstance],
        amplitudes: &[f32],
        opacity_samples: Option<&[f32]>,
        size_samples: Option<&[f32]>,
    ) {
        self.set_odx_field(
            device,
            sphere_vertices,
            sphere_indices,
            instances,
            amplitudes,
            opacity_samples,
            size_samples,
        );
        self.odx_mode = Some(OdxGpuGlyphMode::PreSampledOdf);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_odx_slice_gather(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_key: usize,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
        instances: &[GlyphInstance],
        source_amplitudes: &[f32],
        source_rows: usize,
        source_bins: usize,
        full_bins: usize,
        rows_per_chunk: usize,
        chunk_worklists: &[OdfChunkWorklist],
        amp_norm: f32,
        opacity_samples: Option<&[f32]>,
        size_samples: Option<&[f32]>,
    ) {
        if instances.is_empty() {
            self.clear();
            return;
        }

        self.ensure_odx_geometry(device, scene_key, sphere_vertices, sphere_indices);
        self.ensure_instance_buffer(
            device,
            queue,
            "odx_glyph_instances",
            bytemuck::cast_slice(instances),
        );

        let mut bind_groups_dirty = self.ensure_output_amplitude_buffer(
            device,
            (instances
                .len()
                .saturating_mul(full_bins)
                .saturating_mul(std::mem::size_of::<f32>())) as u64,
        );
        let opacity_values = gate_samples_for_instances(instances.len(), opacity_samples);
        bind_groups_dirty |= self.ensure_gate_buffer(
            device,
            queue,
            "odx_glyph_opacity_gate_samples",
            &opacity_values,
            GateBufferKind::Opacity,
        );
        let size_values = gate_samples_for_instances(instances.len(), size_samples);
        bind_groups_dirty |= self.ensure_gate_buffer(
            device,
            queue,
            "odx_glyph_size_gate_samples",
            &size_values,
            GateBufferKind::Size,
        );
        if bind_groups_dirty {
            self.rebuild_bind_groups(device, "odx_glyph_bg");
        }

        self.ensure_odx_odf_source_chunks(
            device,
            scene_key,
            source_amplitudes,
            source_rows,
            source_bins,
            full_bins,
            rows_per_chunk.max(1),
        );

        self.num_indices = sphere_indices.len() as u32;
        self.num_instances = instances.len() as u32;
        self.current_bins = full_bins as u32;
        self.odx_mode = Some(OdxGpuGlyphMode::OdfSliceGather);
        self.clear_odx_sh_compute();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("interactive_odx_odf_slice_encoder"),
        });
        self.dispatch_odx_odf_slice_gather(device, &mut encoder, chunk_worklists, amp_norm);
        queue.submit(std::iter::once(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_odx_sh_volume(
        &mut self,
        device: &wgpu::Device,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
        instances: &[GlyphInstance],
        coefficients: &[f32],
        ncoeffs: usize,
        transform: &[f32],
        source_bins: usize,
        full_bins: usize,
        opacity_samples: Option<&[f32]>,
        size_samples: Option<&[f32]>,
    ) {
        if instances.is_empty() {
            self.clear();
            return;
        }

        self.odx_static_geometry_key = None;
        self.clear_odx_odf_gather_state();
        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_glyph_vertices"),
                contents: bytemuck::cast_slice(sphere_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_glyph_indices"),
                contents: bytemuck::cast_slice(sphere_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.instance_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_glyph_instances"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
        self.buffer_capacities.instance_bytes =
            (instances.len() * std::mem::size_of::<GlyphInstance>()) as u64;
        let zero_amplitudes = vec![0.0f32; instances.len() * full_bins];
        self.amplitude_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_glyph_amplitudes"),
                contents: bytemuck::cast_slice(&zero_amplitudes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.buffer_capacities.amplitude_bytes =
            (zero_amplitudes.len() * std::mem::size_of::<f32>()) as u64;
        self.opacity_buffer = Some(make_gate_sample_buffer(
            device,
            "odx_sh_glyph_opacity_gate_samples",
            gate_samples_for_instances(instances.len(), opacity_samples).as_slice(),
        ));
        self.buffer_capacities.opacity_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.size_buffer = Some(make_gate_sample_buffer(
            device,
            "odx_sh_glyph_size_gate_samples",
            gate_samples_for_instances(instances.len(), size_samples).as_slice(),
        ));
        self.buffer_capacities.size_bytes =
            (instances.len().max(1) * std::mem::size_of::<f32>()) as u64;
        self.odx_sh_coeff_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_coefficients"),
                contents: bytemuck::cast_slice(coefficients),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.odx_sh_transform_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_transform"),
                contents: bytemuck::cast_slice(transform),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.odx_sh_slice_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_slice_indices"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        let params = OdxShComputeParams {
            full_bins: full_bins as u32,
            source_bins: source_bins as u32,
            ncoeffs: ncoeffs as u32,
            slice_count: 0,
        };
        self.odx_sh_params_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("odx_sh_compute_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.odx_sh_source_bins = source_bins as u32;
        self.odx_sh_ncoeffs = ncoeffs as u32;
        self.num_indices = sphere_indices.len() as u32;
        self.num_instances = instances.len() as u32;
        self.current_bins = sphere_vertices.len() as u32;
        self.odx_mode = Some(OdxGpuGlyphMode::ShCompute);
        self.rebuild_bind_groups(device, "odx_sh_glyph_bg");
        self.rebuild_odx_sh_bind_group(device);
    }

    pub fn dispatch_odx_sh_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slice_compact_indices: &[u32],
    ) {
        if self.odx_mode != Some(OdxGpuGlyphMode::ShCompute) {
            return;
        }
        if self.odx_sh_params_buffer.is_none() {
            return;
        }

        let stored = self.current_odx_sh_params();
        let full_bins = stored.full_bins as usize;
        if full_bins == 0 || slice_compact_indices.is_empty() {
            return;
        }

        let rows_per_dispatch = max_rows_per_dispatch(full_bins);
        for slice_chunk in slice_compact_indices.chunks(rows_per_dispatch) {
            let next = OdxShComputeParams {
                full_bins: stored.full_bins,
                source_bins: stored.source_bins,
                ncoeffs: stored.ncoeffs,
                slice_count: slice_chunk.len() as u32,
            };
            let Some(params_buffer) = self.odx_sh_params_buffer.as_ref() else {
                return;
            };
            queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&next));

            self.odx_sh_slice_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("odx_sh_slice_indices"),
                    contents: bytemuck::cast_slice(slice_chunk),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            ));
            self.rebuild_odx_sh_bind_group(device);

            let total = (slice_chunk.len() * full_bins) as u32;
            if total == 0 {
                continue;
            }
            let Some(bind_group) = self.odx_sh_bind_group.as_ref() else {
                return;
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("odx_sh_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.odx_sh_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(total.div_ceil(COMPUTE_WORKGROUP_SIZE_X), 1, 1);
        }
    }

    fn ensure_odx_geometry(
        &mut self,
        device: &wgpu::Device,
        scene_key: usize,
        sphere_vertices: &[[f32; 3]],
        sphere_indices: &[u32],
    ) {
        let key = (scene_key, sphere_vertices.len(), sphere_indices.len());
        if self.odx_static_geometry_key == Some(key)
            && self.vertex_buffer.is_some()
            && self.index_buffer.is_some()
        {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_vertices"),
                contents: bytemuck::cast_slice(sphere_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odx_glyph_indices"),
                contents: bytemuck::cast_slice(sphere_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.odx_static_geometry_key = Some(key);
    }

    fn ensure_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        bytes: &[u8],
    ) {
        let needed = bytes.len().max(1) as u64;
        let should_recreate =
            self.instance_buffer.is_none() || self.buffer_capacities.instance_bytes < needed;
        if should_recreate {
            self.instance_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
            self.buffer_capacities.instance_bytes = needed;
        } else if let Some(buffer) = self.instance_buffer.as_ref() {
            queue.write_buffer(buffer, 0, bytes);
        }
    }

    fn ensure_output_amplitude_buffer(&mut self, device: &wgpu::Device, needed: u64) -> bool {
        let needed = needed.max(16);
        let should_recreate =
            self.amplitude_buffer.is_none() || self.buffer_capacities.amplitude_bytes < needed;
        if should_recreate {
            self.amplitude_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("odx_slice_amplitudes"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.buffer_capacities.amplitude_bytes = needed;
        }
        should_recreate
    }

    fn ensure_gate_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        values: &[f32],
        which: GateBufferKind,
    ) -> bool {
        let bytes = bytemuck::cast_slice(values);
        let needed = bytes.len().max(std::mem::size_of::<f32>()) as u64;
        let (slot, capacity) = match which {
            GateBufferKind::Opacity => (
                &mut self.opacity_buffer,
                &mut self.buffer_capacities.opacity_bytes,
            ),
            GateBufferKind::Size => (
                &mut self.size_buffer,
                &mut self.buffer_capacities.size_bytes,
            ),
        };
        let should_recreate = slot.is_none() || *capacity < needed;
        if should_recreate {
            *slot = Some(make_gate_sample_buffer(device, label, values));
            *capacity = needed;
        } else if let Some(buffer) = slot.as_ref() {
            queue.write_buffer(buffer, 0, bytes);
        }
        should_recreate
    }

    fn ensure_odx_odf_source_chunks(
        &mut self,
        device: &wgpu::Device,
        scene_key: usize,
        source_amplitudes: &[f32],
        source_rows: usize,
        source_bins: usize,
        full_bins: usize,
        rows_per_chunk: usize,
    ) {
        let key = (scene_key, source_rows, source_bins, rows_per_chunk);
        if self.odx_odf_source_key == Some(key) {
            return;
        }

        self.odx_odf_source_chunks.clear();
        for chunk_start_row in (0..source_rows).step_by(rows_per_chunk.max(1)) {
            let chunk_end_row = (chunk_start_row + rows_per_chunk).min(source_rows);
            let flat_start = chunk_start_row * source_bins;
            let flat_end = chunk_end_row * source_bins;
            self.odx_odf_source_chunks.push(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("odx_odf_source_chunk"),
                    contents: bytemuck::cast_slice(&source_amplitudes[flat_start..flat_end]),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));
        }
        self.odx_odf_source_key = Some(key);
        self.current_bins = full_bins as u32;
    }

    fn dispatch_odx_odf_slice_gather(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        chunk_worklists: &[OdfChunkWorklist],
        amp_norm: f32,
    ) {
        let Some(output_buffer) = self.amplitude_buffer.as_ref() else {
            return;
        };
        let Some((_, _, source_bins, _)) = self.odx_odf_source_key else {
            return;
        };

        for chunk in chunk_worklists {
            if chunk.work_items.is_empty() {
                continue;
            }
            let Some(source_buffer) = self.odx_odf_source_chunks.get(chunk.chunk_index) else {
                continue;
            };
            let rows_per_dispatch = max_rows_per_dispatch(self.current_bins as usize);
            for work_item_chunk in chunk.work_items.chunks(rows_per_dispatch) {
                let params = OdxOdfComputeParams {
                    counts: [
                        self.current_bins,
                        source_bins as u32,
                        work_item_chunk.len() as u32,
                        0,
                    ],
                    scales: [if amp_norm > 0.0 { amp_norm } else { 1.0 }, 0.0, 0.0, 0.0],
                };
                let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("odx_odf_compute_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let worklist_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("odx_odf_worklist"),
                        contents: bytemuck::cast_slice(work_item_chunk),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("odx_odf_compute_bg"),
                    layout: &self.odx_odf_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: source_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: worklist_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buffer.as_entire_binding(),
                        },
                    ],
                });

                let total = (work_item_chunk.len() * self.current_bins as usize) as u32;
                if total == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("odx_odf_gather_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.odx_odf_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total.div_ceil(COMPUTE_WORKGROUP_SIZE_X), 1, 1);
            }
        }
    }

    pub fn clear(&mut self) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.instance_buffer = None;
        self.amplitude_buffer = None;
        self.opacity_buffer = None;
        self.size_buffer = None;
        self.buffer_capacities = GlyphBufferCapacities::default();
        self.num_indices = 0;
        self.num_instances = 0;
        self.current_bins = 0;
        self.odx_static_geometry_key = None;
        self.odx_mode = None;
        self.clear_odx_odf_gather_state();
        self.clear_odx_sh_compute();
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
        color_mode: BoundaryGlyphColorMode,
        draw_step: u32,
        opacity: f32,
        gloss: f32,
        scene_lighting: SceneLightingParams,
        render_3d: &WorkflowRender3D,
        fog_near: f32,
        fog_far: f32,
    ) {
        let uniforms = GlyphUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.into(),
            slab_half_width,
            slab_normal: slab_normal.into(),
            color_mode: match color_mode {
                BoundaryGlyphColorMode::DirectionRgb => 0,
                BoundaryGlyphColorMode::Monochrome => 1,
            },
            slab_center: slab_center.into(),
            draw_step: draw_step.max(1),
            ambient_strength: scene_lighting.ambient_strength(),
            key_strength: scene_lighting.key_strength(),
            fill_strength: scene_lighting.fill_strength(),
            headlight_mix: scene_lighting.headlight_mix(),
            specular_strength: scene_lighting.specular_strength(),
            opacity: opacity.clamp(0.0, 1.0),
            gloss: gloss.clamp(0.0, 1.0),
            scale_mul: 0.0,
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
            opacity_gate: [0.0, 1.0, 1.0, 1.0],
            size_gate: [0.0, 1.0, 1.0, 1.0],
        };
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    pub fn update_amp_norm(&self, queue: &wgpu::Queue, viewport: usize, amp_norm: f32) {
        const OFFSET: u64 = 176 + 12;
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            OFFSET,
            bytemuck::bytes_of(&amp_norm),
        );
    }

    pub fn update_color_mode(&self, queue: &wgpu::Queue, viewport: usize, mode: u32) {
        const OFFSET: u64 = 92;
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            OFFSET,
            bytemuck::bytes_of(&mode),
        );
    }

    pub fn update_scale_mul(&self, queue: &wgpu::Queue, viewport: usize, scale_mul: f32) {
        const OFFSET: u64 = 140;
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            OFFSET,
            bytemuck::bytes_of(&scale_mul),
        );
    }

    pub fn update_opacity_gate(&self, queue: &wgpu::Queue, viewport: usize, params: [f32; 4]) {
        const OFFSET: u64 = 192;
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            OFFSET,
            bytemuck::bytes_of(&params),
        );
    }

    pub fn update_size_gate(&self, queue: &wgpu::Queue, viewport: usize, params: [f32; 4]) {
        const OFFSET: u64 = 208;
        queue.write_buffer(
            &self.uniform_buffers[viewport],
            OFFSET,
            bytemuck::bytes_of(&params),
        );
    }

    pub fn has_geometry(&self) -> bool {
        self.num_indices != 0 && self.num_instances != 0
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

    fn rebuild_bind_groups(&mut self, device: &wgpu::Device, label: &str) {
        let amplitude = self.amplitude_buffer.as_ref().expect("amplitude buffer");
        let opacity = self.opacity_buffer.as_ref().expect("opacity gate buffer");
        let size = self.size_buffer.as_ref().expect("size gate buffer");
        for i in 0..4 {
            self.bind_groups[i] = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: amplitude.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: opacity.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: size.as_entire_binding(),
                    },
                ],
            });
        }
    }

    fn rebuild_odx_sh_bind_group(&mut self, device: &wgpu::Device) {
        let (Some(params), Some(coeffs), Some(transform), Some(slice_indices), Some(amplitudes)) = (
            self.odx_sh_params_buffer.as_ref(),
            self.odx_sh_coeff_buffer.as_ref(),
            self.odx_sh_transform_buffer.as_ref(),
            self.odx_sh_slice_buffer.as_ref(),
            self.amplitude_buffer.as_ref(),
        ) else {
            self.odx_sh_bind_group = None;
            return;
        };
        self.odx_sh_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("odx_sh_compute_bg"),
            layout: &self.odx_sh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: transform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: slice_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: amplitudes.as_entire_binding(),
                },
            ],
        }));
    }

    fn clear_odx_odf_gather_state(&mut self) {
        self.odx_odf_source_chunks.clear();
        self.odx_odf_source_key = None;
        self.odx_odf_worklist_buffer = None;
        self.odx_odf_worklist_capacity_bytes = 0;
        self.odx_odf_params_buffer = None;
    }

    fn clear_odx_sh_compute(&mut self) {
        self.odx_sh_bind_group = None;
        self.odx_sh_coeff_buffer = None;
        self.odx_sh_transform_buffer = None;
        self.odx_sh_slice_buffer = None;
        self.odx_sh_params_buffer = None;
        self.odx_sh_source_bins = 0;
        self.odx_sh_ncoeffs = 0;
    }

    fn current_odx_sh_params(&self) -> OdxShComputeParams {
        OdxShComputeParams {
            full_bins: self.current_bins,
            source_bins: self.odx_sh_source_bins,
            ncoeffs: self.odx_sh_ncoeffs,
            slice_count: 0,
        }
    }
}

fn gate_samples_for_instances(instance_count: usize, samples: Option<&[f32]>) -> Vec<f32> {
    match samples {
        Some(values) if values.len() == instance_count => values.to_vec(),
        _ => vec![f32::NAN; instance_count.max(1)],
    }
}

fn make_gate_sample_buffer(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: depth_write,
            depth_compare,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    })
}

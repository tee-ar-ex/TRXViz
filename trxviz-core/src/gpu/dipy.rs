/// GPU probabilistic tractography via wgpu compute shader.
///
/// Uses the WGSL shader at `shaders/tractography_prob.wgsl`.
/// One workgroup = one seed, workgroup size (32, 1, 1).
/// Seeds are processed in batches to bound GPU memory usage.
use std::sync::Arc;

use glam::Mat4;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::data::loaded_files::StreamlineBacking;
use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;
use crate::workflow::PostFilter;
use crate::workflow::tracking_filters::{
    streamline_endpoint_in, streamline_hits_all_rois, streamline_passes_hausdorff,
    streamline_satisfies_end_masks,
};
use crate::workflow::{
    DipyDirectionGetter, DipyTractographyPlan, StreamlineDataset, StreamlineFlow,
};

/// Maximum seeds dispatched in a single GPU batch.
const BATCH_SIZE: u32 = 2048;

/// Bytes per f32.
const F32_SIZE: u64 = 4;

// ── shared input prep ────────────────────────────────────────────────────
//
// The probabilistic and PTT GPU paths both need the same scene-derived
// inputs (SH coefficients, B-matrix, voxel LUT, sphere vertices, seed
// points in voxel space, affine transforms). They differ only in which
// shader and per-step parameters they upload. `DipyGpuInputs` bundles
// everything shared so each path is just `prep + shader-specific
// dispatch + apply_post_hoc_filters`.

/// Shared inputs for any DIPY-style GPU tracker. Built once per
/// `run_gpu_dipy*` call by `prepare_dipy_inputs`.
pub(super) struct DipyGpuInputs {
    // Scene metadata.
    pub(super) ncoeffs: u32,
    pub(super) n_dirs: u32,
    pub(super) nx: u32,
    pub(super) ny: u32,
    pub(super) nz: u32,
    pub(super) max_pts: u32,
    pub(super) min_pts: u32,
    pub(super) step_size_vox: f32,
    pub(super) max_angle_cos: f32,
    pub(super) vox_to_ras: Mat4,

    // Seeds expanded to (n_voxels × seeds_per_voxel) in voxel space.
    pub(super) seeds_flat: Vec<f32>,
    pub(super) total_seeds: u32,

    // Persistent GPU buffers (uniform across batches).
    pub(super) sh_buf: wgpu::Buffer,
    pub(super) b_buf: wgpu::Buffer,
    pub(super) lut_buf: wgpu::Buffer,
    pub(super) sv_buf: wgpu::Buffer,
}

/// Build all the GPU-uniform inputs for a DIPY tracker. Returns
/// `Ok(None)` when the seed mask is empty (caller should use `empty_flow`).
/// Returns `Err` when the ODX lacks SH coefficients (PTT and probabilistic
/// both need them).
pub(super) fn prepare_dipy_inputs(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
) -> WorkflowResult<Option<DipyGpuInputs>> {
    let scene = &plan.odx_scene;

    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation("ODX file has no SH coefficients. Re-derive with odx-rs.".into())
    })?;
    let ncoeffs = sh_view.ncols() as u32;
    let nb_voxels = sh_view.nrows();

    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for GPU tractography.".into())
    })?;
    let sphere_verts = mesh.vertices();
    let n_dirs = sphere_verts.len() as u32;

    let b_matrix_flat: Vec<f32> = mesh.transform_flat().to_vec();

    let sh_flat: Vec<f32> = (0..nb_voxels)
        .flat_map(|i| sh_view.row(i).iter().copied())
        .collect();

    let dims = scene.dimensions();
    let [nx, ny, nz] = [dims[0] as u32, dims[1] as u32, dims[2] as u32];
    let lut_len = (nx * ny * nz) as usize;
    let mut lut: Vec<u32> = vec![0xFFFF_FFFFu32; lut_len];
    for (compact_idx, &[ix, iy, iz]) in scene.ijk_lookup().iter().enumerate() {
        let flat =
            ix as usize * ny as usize * nz as usize + iy as usize * nz as usize + iz as usize;
        lut[flat] = compact_idx as u32;
    }

    let sphere_verts_flat: Vec<f32> = sphere_verts
        .iter()
        .flat_map(|v: &[f32; 3]| v.iter().copied())
        .collect();

    let vox_to_ras: Mat4 = scene.voxel_to_ras();
    let ras_to_vox: Mat4 = vox_to_ras.inverse();

    // Smallest voxel dimension (mm) used to convert step_size_mm → voxel
    // units. The shader works in voxel space because the LUT lives there.
    let smallest_vs = vox_to_ras
        .col(0)
        .truncate()
        .length()
        .min(vox_to_ras.col(1).truncate().length())
        .min(vox_to_ras.col(2).truncate().length())
        .max(1e-3);
    let step_size_vox = plan.step_size_mm / smallest_vs;
    let max_angle_cos = plan.max_angle_deg.to_radians().cos();

    let seeds_ras = plan.seed_mask.nonzero_voxel_centers_ras();
    let seeds_vox: Vec<[f32; 3]> = seeds_ras
        .iter()
        .flat_map(|seed| {
            (0..plan.seeds_per_voxel).map(move |_| {
                let v = ras_to_vox.transform_point3(glam::Vec3::from_array(*seed));
                [v.x, v.y, v.z]
            })
        })
        .collect();
    let total_seeds = seeds_vox.len() as u32;
    if total_seeds == 0 {
        return Ok(None);
    }

    let seeds_flat: Vec<f32> = seeds_vox
        .iter()
        .flat_map(|s: &[f32; 3]| s.iter().copied())
        .collect();

    let max_pts = plan.max_points;
    let min_pts = (plan.min_len_mm / plan.step_size_mm).ceil() as u32;

    let sh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_sh_coeffs"),
        contents: bytemuck::cast_slice(&sh_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_b_matrix"),
        contents: bytemuck::cast_slice(&b_matrix_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_lut"),
        contents: bytemuck::cast_slice(&lut),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_sphere_verts"),
        contents: bytemuck::cast_slice(&sphere_verts_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    Ok(Some(DipyGpuInputs {
        ncoeffs,
        n_dirs,
        nx,
        ny,
        nz,
        max_pts,
        min_pts,
        step_size_vox,
        max_angle_cos,
        vox_to_ras,
        seeds_flat,
        total_seeds,
        sh_buf,
        b_buf,
        lut_buf,
        sv_buf,
    }))
}

/// Decision applied to a single GPU-produced streamline before it lands
/// in the output flow. Same set of post-hoc filters that
/// `cpu_dipy::try_one_attempt` runs serially.
///
/// Per-step masks (`limiting_mask`, `roa_mask`, `term_mask`) are
/// **not** enforced — they would require per-step lookups inside the
/// shader. Documented gap; same as the existing prob path.
pub(super) fn keep_streamline_for_plan(
    plan: &DipyTractographyPlan,
    streamline: &[[f32; 3]],
) -> bool {
    if !plan.roi_masks.is_empty() && !streamline_hits_all_rois(streamline, &plan.roi_masks) {
        return false;
    }
    if let Some(ne) = plan.no_end_mask.as_deref() {
        if streamline_endpoint_in(streamline, ne) {
            return false;
        }
    }
    if !plan.end_masks.is_empty() && !streamline_satisfies_end_masks(streamline, &plan.end_masks) {
        return false;
    }
    if let Some(PostFilter::Hausdorff {
        reference_points_ras,
        max_mm,
    }) = plan.post_filter.as_ref()
    {
        if !streamline_passes_hausdorff(streamline, reference_points_ras, *max_mm) {
            return false;
        }
    }
    true
}

/// Wrap a positions+offsets pair into a `StreamlineFlow` ready to
/// return from a tracker.
pub(super) fn assemble_flow(
    plan: &DipyTractographyPlan,
    positions: Vec<[f32; 3]>,
    offsets: Vec<u32>,
) -> StreamlineFlow {
    let nb_streamlines = offsets.len() - 1;
    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(positions, offsets));
    let selected: Vec<StreamlineIndex> = (0..nb_streamlines as u32).map(StreamlineIndex).collect();
    let dataset = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
    });
    StreamlineFlow {
        dataset,
        selected_streamlines: selected,
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    }
}

/// Precompute FOD amplitudes evaluated at every sphere vertex for every
/// voxel: `amp[v, d] = max(0, Σ_k sh[v, k] · b_matrix[d, k])`.
///
/// Used by the PTT shader so the per-step FOD-at-direction lookup is a
/// single trilinear interp of one channel, instead of recomputing the
/// SH dot product (8 corners × ncoeffs ≈ 360 mults) on every call.
///
/// Cost: nb_voxels × n_dirs × ncoeffs mults on the CPU. For typical
/// dimensions (50k voxels × 162 dirs × 45 coeffs ≈ 365M mults) this is
/// ~1-2 s single-threaded; we parallelize the outer-voxel loop with
/// rayon to amortize across cores. The amortized cost per streamline
/// is negligible since this happens once per `run_gpu_dipy_ptt`.
///
/// Output buffer size: nb_voxels × n_dirs × 4 bytes. For the same
/// example: 32 MB. Fits comfortably in any GPU memory budget.
fn precompute_fod_amplitudes(
    plan: &DipyTractographyPlan,
) -> WorkflowResult<(Vec<f32>, u32, u32, usize)> {
    let scene = &plan.odx_scene;
    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation("ODX file has no SH coefficients. Re-derive with odx-rs.".into())
    })?;
    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for GPU tractography.".into())
    })?;
    let n_dirs = mesh.vertices().len();
    let ncoeffs = sh_view.ncols();
    let nb_voxels = sh_view.nrows();
    let b_matrix = mesh.transform_flat();

    // Parallel over voxels: each thread computes `n_dirs` amplitudes
    // for one voxel, writing into its assigned slice of the output.
    // (b_matrix is read-only and shared; sh_view is also read-only.)
    let mut amps = vec![0.0f32; nb_voxels * n_dirs];
    amps.par_chunks_mut(n_dirs)
        .enumerate()
        .for_each(|(v, voxel_slice)| {
            let sh_row = sh_view.row(v);
            for d in 0..n_dirs {
                let b_base = d * ncoeffs;
                let mut sum = 0.0f32;
                for k in 0..ncoeffs {
                    sum += sh_row[k] * b_matrix[b_base + k];
                }
                voxel_slice[d] = sum.max(0.0);
            }
        });

    Ok((amps, n_dirs as u32, ncoeffs as u32, nb_voxels))
}

pub fn run_gpu_dipy(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WorkflowResult<StreamlineFlow> {
    // DG dispatch. Each variant has its own `run_gpu_dipy_*`; this
    // outer function is now just a router. Shared input prep lives in
    // `prepare_dipy_inputs`.
    match plan.direction_getter {
        DipyDirectionGetter::Probabilistic => {}
        DipyDirectionGetter::Ptt { .. } => {
            return run_gpu_dipy_ptt(plan, device, queue);
        }
    }

    let inputs = match prepare_dipy_inputs(plan, device)? {
        Some(i) => i,
        None => return empty_flow(plan),
    };
    let DipyGpuInputs {
        ncoeffs,
        n_dirs,
        nx,
        ny,
        nz,
        max_pts,
        min_pts,
        step_size_vox,
        max_angle_cos,
        vox_to_ras,
        seeds_flat,
        total_seeds,
        ref sh_buf,
        ref b_buf,
        ref lut_buf,
        ref sv_buf,
    } = inputs;

    eprintln!(
        "[gpu_dipy_prob] '{}': {} seeds, {} dirs, {} coeffs",
        plan.label, total_seeds, n_dirs, ncoeffs,
    );

    // ── compute pipeline ─────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tractography_prob"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tractography_prob.wgsl").into()),
    });

    let group0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trac_group0_bgl"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
        ],
    });

    let group1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trac_group1_bgl"),
        entries: &[
            storage_entry(0, false), // out_points (read_write)
            storage_entry(1, false), // out_lengths (read_write)
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("trac_pipeline_layout"),
        bind_group_layouts: &[&group0_bgl, &group1_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tractography_prob_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ── batch loop ────────────────────────────────────────────────────────
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_offsets: Vec<u32> = vec![0];
    let t0 = std::time::Instant::now();

    let out_pts_floats_per_batch = BATCH_SIZE as u64 * 2 * max_pts as u64 * 3;
    let out_pts_bytes = out_pts_floats_per_batch * F32_SIZE;
    let out_len_bytes = BATCH_SIZE as u64 * 2 * 4; // u32

    let out_pts_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trac_out_points"),
        size: out_pts_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_len_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trac_out_lengths"),
        size: out_len_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_pts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trac_staging_pts"),
        size: out_pts_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_len = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trac_staging_len"),
        size: out_len_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut batch_offset = 0u32;
    while batch_offset < total_seeds {
        let batch_size = (total_seeds - batch_offset).min(BATCH_SIZE);

        // ── params uniform (64 bytes, 16-byte aligned) ─────────────────
        // Match TractographyParams in the shader exactly.
        let params_data: [u32; 16] = {
            let mut p = [0u32; 16];
            p[0] = batch_size;
            p[1] = ncoeffs;
            p[2] = n_dirs;
            p[3] = nx;
            p[4] = ny;
            p[5] = nz;
            p[6] = max_pts;
            p[7] = batch_offset;
            p[8] = step_size_vox.to_bits();
            p[9] = max_angle_cos.to_bits();
            p[10] = plan.relative_peak_threshold.to_bits();
            // rng_seed: mix plan.rng_seed + batch_offset
            let rng = (plan.rng_seed as u32).wrapping_add(batch_offset);
            p[11] = rng;
            // pads 12..15 = 0
            p
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trac_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── batch seeds buffer ─────────────────────────────────────────
        let seeds_start = batch_offset as usize * 3;
        let seeds_end = seeds_start + batch_size as usize * 3;
        let seeds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trac_seeds"),
            contents: bytemuck::cast_slice(&seeds_flat[seeds_start..seeds_end]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── bind groups ────────────────────────────────────────────────
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trac_group0"),
            layout: &group0_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sv_buf.as_entire_binding(),
                },
            ],
        });

        let batch_out_pts_bytes = batch_size as u64 * 2 * max_pts as u64 * 3 * F32_SIZE;
        let batch_out_len_bytes = batch_size as u64 * 2 * 4;

        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trac_group1"),
            layout: &group1_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &out_pts_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(batch_out_pts_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &out_len_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(batch_out_len_bytes),
                    }),
                },
            ],
        });

        // ── dispatch ───────────────────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trac_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trac_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            // One workgroup per seed; each workgroup has 32 threads.
            pass.dispatch_workgroups(batch_size, 1, 1);
        }

        // Copy output to staging buffers.
        encoder.copy_buffer_to_buffer(&out_pts_buf, 0, &staging_pts, 0, batch_out_pts_bytes);
        encoder.copy_buffer_to_buffer(&out_len_buf, 0, &staging_len, 0, batch_out_len_bytes);

        queue.submit(std::iter::once(encoder.finish()));

        // ── readback ───────────────────────────────────────────────────
        let pts_slice = staging_pts.slice(..batch_out_pts_bytes);
        let len_slice = staging_len.slice(..batch_out_len_bytes);
        map_slices_blocking(device, &[pts_slice, len_slice], GPU_READBACK_TIMEOUT)?;

        let lengths: Vec<u32> = {
            let view = len_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging_len.unmap();

        let points: Vec<f32> = {
            let view = pts_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging_pts.unmap();

        // ── decode output ──────────────────────────────────────────────
        // lengths layout: [back_len, fwd_len] per seed (batch_size * 2 entries).
        // points layout:  [seed * 2 * max_pts * 3 + dir * max_pts * 3 + step * 3 + xyz]
        for s in 0..batch_size as usize {
            let back_len = lengths[s * 2] as usize;
            let fwd_len = lengths[s * 2 + 1] as usize;
            let total_pts = back_len + fwd_len;
            if total_pts < min_pts as usize {
                continue;
            }

            let back_base = s * 2 * max_pts as usize * 3;
            let fwd_base = s * 2 * max_pts as usize * 3 + max_pts as usize * 3;

            // Backward segment is stored head-first; reverse it. We
            // assemble + convert vox→RAS into a temporary `streamline`
            // buffer because the post-hoc filters (`roi_masks`,
            // `end_masks`, Hausdorff, ...) need the whole streamline in
            // RAS space to score it. Only commit to `all_positions` if
            // it survives `keep_streamline_for_plan`.
            let mut streamline: Vec<[f32; 3]> = Vec::with_capacity(total_pts);
            for step in (0..back_len).rev() {
                let o = back_base + step * 3;
                let v = glam::Vec3::new(points[o], points[o + 1], points[o + 2]);
                streamline.push(vox_to_ras.transform_point3(v).to_array());
            }
            for step in 0..fwd_len {
                let o = fwd_base + step * 3;
                let v = glam::Vec3::new(points[o], points[o + 1], points[o + 2]);
                streamline.push(vox_to_ras.transform_point3(v).to_array());
            }

            if !keep_streamline_for_plan(plan, &streamline) {
                continue;
            }

            all_positions.extend_from_slice(&streamline);
            all_offsets.push(all_positions.len() as u32);
        }

        if batch_offset % (BATCH_SIZE * 4) == 0 && batch_offset > 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let rate = batch_offset as f32 / elapsed;
            eprintln!(
                "[gpu_dipy_prob] {}/{} seeds ({:.0}/s) {} streamlines so far",
                batch_offset,
                total_seeds,
                rate,
                all_offsets.len() - 1,
            );
        }

        batch_offset += batch_size;
    }

    eprintln!(
        "[gpu_dipy_prob] '{}': done in {:.1}s — {} streamlines",
        plan.label,
        t0.elapsed().as_secs_f32(),
        all_offsets.len() - 1,
    );

    Ok(assemble_flow(plan, all_positions, all_offsets))
}

/// PTT (Aydogan & Shi 2021) on the GPU.
///
/// Mirrors `run_gpu_dipy` (the probabilistic path) almost line-for-line
/// — same input prep, same output buffer shape, same batching, same
/// post-hoc filter pipeline. The differences are:
///   - Different shader (`dipy_ptt.wgsl`) — implements the PTT
///     direction getter (frame propagation + rejection-sampled
///     candidate arcs).
///   - Different params struct (24 u32 slots = 96 bytes vs. prob's
///     16 u32 = 64 bytes) — adds 8 PTT-specific knobs.
///
/// See `docs/ptt-implementation-notes.md` for algorithm + design notes
/// (nibrary vs. DIPY vs. GPUStreamlines comparison).
fn run_gpu_dipy_ptt(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WorkflowResult<StreamlineFlow> {
    // Extract the PTT-specific knobs up front. The outer router has
    // already verified the variant.
    let (
        probe_length_mm,
        probe_quality,
        probe_radius_mm,
        probe_count,
        max_curvature_per_mm,
        data_support_exponent,
        min_data_support,
        rejection_sampling_max_try,
    ) = match plan.direction_getter {
        DipyDirectionGetter::Ptt {
            probe_length_mm,
            probe_quality,
            probe_radius_mm,
            probe_count,
            max_curvature_per_mm,
            data_support_exponent,
            min_data_support,
            rejection_sampling_max_try,
        } => (
            probe_length_mm,
            probe_quality,
            probe_radius_mm,
            probe_count,
            max_curvature_per_mm,
            data_support_exponent,
            min_data_support,
            rejection_sampling_max_try,
        ),
        _ => unreachable!("router only dispatches Ptt to this fn"),
    };

    let inputs = match prepare_dipy_inputs(plan, device)? {
        Some(i) => i,
        None => return empty_flow(plan),
    };
    let DipyGpuInputs {
        ncoeffs,
        n_dirs,
        nx,
        ny,
        nz,
        max_pts,
        min_pts,
        step_size_vox,
        max_angle_cos,
        vox_to_ras,
        seeds_flat,
        total_seeds,
        // sh_buf + b_buf NOT used by PTT — we upload a precomputed
        // fod_amp buffer below instead. They're built unconditionally
        // by `prepare_dipy_inputs` (shared with the prob path) but the
        // GPU doesn't bind them here. Cheap memory waste; could be
        // optimized later by making the inputs split per-DG.
        sh_buf: _,
        b_buf: _,
        ref lut_buf,
        ref sv_buf,
    } = inputs;

    // ── precompute FOD amplitudes ─────────────────────────────────────
    //
    // Skip the per-shader-call SH dot product by baking
    // `amp[v, d] = max(0, sh[v] · b_matrix[d])` once on the host.
    // Each shader-side FOD-at-direction lookup then becomes 8 corner
    // reads of one channel instead of 8 corners × ncoeffs SH-dot mults.
    // ~4.5× speedup per FOD call (the dominant inner-loop cost).
    let fod_t0 = std::time::Instant::now();
    let (fod_amps, _, _, _) = precompute_fod_amplitudes(plan)?;
    eprintln!(
        "[gpu_dipy_ptt] FOD precompute: {} voxels × {} dirs in {:.1}s",
        fod_amps.len() / n_dirs as usize,
        n_dirs,
        fod_t0.elapsed().as_secs_f32(),
    );
    let fod_amp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ptt_fod_amplitudes"),
        contents: bytemuck::cast_slice(&fod_amps),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Convert plan's per-mm probe/curvature params into voxel units.
    let smallest_vs = if step_size_vox > 1e-6 {
        plan.step_size_mm / step_size_vox
    } else {
        1.0
    };
    let probe_length_vox = probe_length_mm / smallest_vs;
    let probe_radius_vox = probe_radius_mm / smallest_vs;
    let max_curvature_per_vox = max_curvature_per_mm * smallest_vs; // 1/mm × mm/vox = 1/vox

    eprintln!(
        "[gpu_dipy_ptt] '{}': {} seeds, {} dirs, {} coeffs (probe {:.2}vox × q={}, max_curv {:.3}/vox)",
        plan.label,
        total_seeds,
        n_dirs,
        ncoeffs,
        probe_length_vox,
        probe_quality,
        max_curvature_per_vox,
    );

    // ── compute pipeline ─────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dipy_ptt"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dipy_ptt.wgsl").into()),
    });

    // PTT bind group (5 slots, vs prob's 6): we replace the
    // sh_coeffs + b_matrix pair with a single fod_amp buffer.
    //   0: params (uniform)
    //   1: seeds_vox
    //   2: fod_amp (precomputed FOD per voxel × sphere vertex)
    //   3: lut
    //   4: sphere_verts
    let group0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ptt_group0_bgl"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
        ],
    });

    let group1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ptt_group1_bgl"),
        entries: &[storage_entry(0, false), storage_entry(1, false)],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ptt_pipeline_layout"),
        bind_group_layouts: &[&group0_bgl, &group1_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dipy_ptt_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ── output buffers ────────────────────────────────────────────────
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_offsets: Vec<u32> = vec![0];
    let t0 = std::time::Instant::now();

    let out_pts_floats_per_batch = BATCH_SIZE as u64 * 2 * max_pts as u64 * 3;
    let out_pts_bytes = out_pts_floats_per_batch * F32_SIZE;
    let out_len_bytes = BATCH_SIZE as u64 * 2 * 4;

    let out_pts_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptt_out_points"),
        size: out_pts_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_len_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptt_out_lengths"),
        size: out_len_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_pts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptt_staging_pts"),
        size: out_pts_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_len = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ptt_staging_len"),
        size: out_len_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut batch_offset = 0u32;
    while batch_offset < total_seeds {
        let batch_size = (total_seeds - batch_offset).min(BATCH_SIZE);

        // ── PTT params uniform (24 u32 slots = 96 bytes) ──────────────
        // Layout MUST match `PttParams` in dipy_ptt.wgsl exactly.
        let params_data: [u32; 24] = {
            let mut p = [0u32; 24];
            p[0] = batch_size;
            p[1] = ncoeffs;
            p[2] = n_dirs;
            p[3] = nx;
            p[4] = ny;
            p[5] = nz;
            p[6] = max_pts;
            p[7] = batch_offset;
            p[8] = step_size_vox.to_bits();
            p[9] = max_angle_cos.to_bits();
            p[10] = (plan.rng_seed as u32).wrapping_add(batch_offset);
            p[11] = probe_length_vox.to_bits();
            p[12] = probe_quality;
            p[13] = probe_radius_vox.to_bits();
            p[14] = probe_count;
            p[15] = max_curvature_per_vox.to_bits();
            p[16] = data_support_exponent.to_bits();
            p[17] = min_data_support.to_bits();
            p[18] = rejection_sampling_max_try;
            // pads 19..23 = 0
            p
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ptt_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── batch seeds buffer ────────────────────────────────────────
        let seeds_start = batch_offset as usize * 3;
        let seeds_end = seeds_start + batch_size as usize * 3;
        let seeds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ptt_seeds"),
            contents: bytemuck::cast_slice(&seeds_flat[seeds_start..seeds_end]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── bind groups ───────────────────────────────────────────────
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ptt_group0"),
            layout: &group0_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fod_amp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sv_buf.as_entire_binding(),
                },
            ],
        });

        let batch_out_pts_bytes = batch_size as u64 * 2 * max_pts as u64 * 3 * F32_SIZE;
        let batch_out_len_bytes = batch_size as u64 * 2 * 4;

        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ptt_group1"),
            layout: &group1_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &out_pts_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(batch_out_pts_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &out_len_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(batch_out_len_bytes),
                    }),
                },
            ],
        });

        // ── dispatch ──────────────────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ptt_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ptt_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            // PTT workgroup is (32, 2, 1) — 2 streamlines per workgroup.
            // `ceil(batch_size / 2)` workgroups cover all seeds; the
            // shader early-returns on the trailing tidy=1 slot when
            // batch_size is odd.
            let wg_count = batch_size.div_ceil(2);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_pts_buf, 0, &staging_pts, 0, batch_out_pts_bytes);
        encoder.copy_buffer_to_buffer(&out_len_buf, 0, &staging_len, 0, batch_out_len_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // ── readback ──────────────────────────────────────────────────
        let pts_slice = staging_pts.slice(..batch_out_pts_bytes);
        let len_slice = staging_len.slice(..batch_out_len_bytes);
        map_slices_blocking(device, &[pts_slice, len_slice], GPU_READBACK_TIMEOUT)?;

        let lengths: Vec<u32> = {
            let view = len_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging_len.unmap();
        let points: Vec<f32> = {
            let view = pts_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging_pts.unmap();

        // ── decode + post-hoc filters (identical to prob path) ────────
        for s in 0..batch_size as usize {
            let back_len = lengths[s * 2] as usize;
            let fwd_len = lengths[s * 2 + 1] as usize;
            let total_pts = back_len + fwd_len;
            if total_pts < min_pts as usize {
                continue;
            }

            let back_base = s * 2 * max_pts as usize * 3;
            let fwd_base = s * 2 * max_pts as usize * 3 + max_pts as usize * 3;

            let mut streamline: Vec<[f32; 3]> = Vec::with_capacity(total_pts);
            for step in (0..back_len).rev() {
                let o = back_base + step * 3;
                let v = glam::Vec3::new(points[o], points[o + 1], points[o + 2]);
                streamline.push(vox_to_ras.transform_point3(v).to_array());
            }
            for step in 0..fwd_len {
                let o = fwd_base + step * 3;
                let v = glam::Vec3::new(points[o], points[o + 1], points[o + 2]);
                streamline.push(vox_to_ras.transform_point3(v).to_array());
            }

            if !keep_streamline_for_plan(plan, &streamline) {
                continue;
            }

            all_positions.extend_from_slice(&streamline);
            all_offsets.push(all_positions.len() as u32);
        }

        if batch_offset % (BATCH_SIZE * 4) == 0 && batch_offset > 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let rate = batch_offset as f32 / elapsed;
            eprintln!(
                "[gpu_dipy_ptt] {}/{} seeds ({:.0}/s) {} streamlines so far",
                batch_offset,
                total_seeds,
                rate,
                all_offsets.len() - 1,
            );
        }

        batch_offset += batch_size;
    }

    eprintln!(
        "[gpu_dipy_ptt] '{}': done in {:.1}s — {} streamlines",
        plan.label,
        t0.elapsed().as_secs_f32(),
        all_offsets.len() - 1,
    );

    Ok(assemble_flow(plan, all_positions, all_offsets))
}

fn empty_flow(plan: &DipyTractographyPlan) -> WorkflowResult<StreamlineFlow> {
    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(vec![], vec![0]));
    let dataset = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
    });
    Ok(StreamlineFlow {
        dataset,
        selected_streamlines: vec![],
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    })
}

// ── readback helper ──────────────────────────────────────────────────────

/// Timeout for a single batch's GPU readback. 30s is ~100× the worst real
/// batch we've observed; crossing it almost certainly means the driver
/// wedged and we'd rather return an error than hang the worker thread.
const GPU_READBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Map the given buffer slices for reading, blocking until all maps
/// complete, the map errors, or the global timeout fires. Replaces the
/// older `map_async(..., |_| {})` + `poll(wait_indefinitely)` pattern,
/// which silently ate `BufferAsyncError` and would hang forever if the
/// driver never fired the callback.
fn map_slices_blocking(
    device: &wgpu::Device,
    slices: &[wgpu::BufferSlice<'_>],
    timeout: std::time::Duration,
) -> WorkflowResult<()> {
    use std::sync::mpsc::{RecvTimeoutError, channel};

    let (tx, rx) = channel::<Result<(), wgpu::BufferAsyncError>>();
    for slice in slices {
        let tx = tx.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    }
    drop(tx);

    let n = slices.len();
    let start = std::time::Instant::now();
    let mut got = 0usize;
    while got < n {
        // Some backends only fire callbacks when the device is polled.
        let _ = device.poll(wgpu::PollType::Poll);
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(Ok(())) => got += 1,
            Ok(Err(e)) => {
                return Err(WorkflowError::Evaluation(format!(
                    "GPU buffer map failed: {e}"
                )));
            }
            Err(RecvTimeoutError::Timeout) => {
                if start.elapsed() > timeout {
                    return Err(WorkflowError::Evaluation(format!(
                        "GPU buffer map exceeded {:.0}s timeout ({}/{} buffers mapped); \
                         driver or shader may be wedged",
                        timeout.as_secs_f32(),
                        got,
                        n,
                    )));
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(WorkflowError::Evaluation(format!(
                    "GPU buffer map channel closed after only {}/{} buffers mapped",
                    got, n,
                )));
            }
        }
    }
    Ok(())
}

// ── bind group layout entry helpers ──────────────────────────────────────

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

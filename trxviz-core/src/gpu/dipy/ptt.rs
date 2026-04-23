//! Parallel Transport Tractography (Aydogan & Shi 2021) on the GPU.
//!
//! WGSL shader: `shaders/dipy_ptt.wgsl`. Workgroup size (32, 2, 1) — 2
//! streamlines per workgroup. Inputs mirror the probabilistic path's
//! shared prep, except the SH + B-matrix pair is replaced by a single
//! precomputed `fod_amp[v, d]` buffer (see `shared::precompute_fod_amplitudes`).
//! The params uniform grows to 24 u32 slots (96 bytes) to carry the 8
//! PTT-specific knobs (probe length / quality / radius / count /
//! max curvature / data support / min support / rejection max try).
//!
//! See `docs/ptt-implementation-notes.md` for algorithm + design notes
//! (nibrary vs. DIPY vs. GPUStreamlines comparison).

use wgpu::util::DeviceExt;

use crate::error::WorkflowResult;
use crate::workflow::{DipyDirectionGetter, DipyTractographyPlan, StreamlineFlow};

use super::readback::{GPU_READBACK_TIMEOUT, map_slices_blocking};
use super::shared::{
    BATCH_SIZE, DipyGpuInputs, F32_SIZE, assemble_flow, empty_flow, keep_streamline_for_plan,
    precompute_fod_amplitudes, prepare_dipy_inputs, storage_entry, uniform_entry,
};

pub(super) fn run(
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
    // `step_size_vox = step_size_mm / smallest_vs`, so recovering
    // `smallest_vs` is `step_size_mm / step_size_vox`. Units:
    // mm / vox = mm/vox ✓.
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
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dipy_ptt.wgsl").into()),
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

        let seeds_start = batch_offset as usize * 3;
        let seeds_end = seeds_start + batch_size as usize * 3;
        let seeds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ptt_seeds"),
            contents: bytemuck::cast_slice(&seeds_flat[seeds_start..seeds_end]),
            usage: wgpu::BufferUsages::STORAGE,
        });

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

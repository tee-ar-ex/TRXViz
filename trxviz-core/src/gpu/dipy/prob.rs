//! Probabilistic Dipy tractography on the GPU.
//!
//! WGSL shader: `shaders/tractography_prob.wgsl`. One workgroup per seed,
//! workgroup size (32, 1, 1). At each step the shader evaluates the SH
//! PMF on the sphere at the current voxel, zeros directions outside the
//! angular cone, renormalizes, and draws a weighted sample.

use wgpu::util::DeviceExt;

use crate::error::WorkflowResult;
use crate::workflow::{DipyTractographyPlan, StreamlineFlow};

use super::readback::{GPU_READBACK_TIMEOUT, map_slices_blocking};
use super::shared::{
    BATCH_SIZE, DipyGpuInputs, F32_SIZE, assemble_flow, empty_flow, keep_streamline_for_plan,
    prepare_dipy_inputs, storage_entry, uniform_entry,
};

pub(super) fn run(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WorkflowResult<StreamlineFlow> {
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
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/tractography_prob.wgsl").into(),
        ),
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
        entries: &[storage_entry(0, false), storage_entry(1, false)],
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
    let out_len_bytes = BATCH_SIZE as u64 * 2 * 4;

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
            // Mix plan.rng_seed + batch_offset so each batch's seeds
            // produce a different random stream in the shader.
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

        // ── decode + post-hoc filters ──────────────────────────────────
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
            // assemble + convert vox→RAS into a temporary streamline
            // buffer because the post-hoc filters need the whole
            // streamline in RAS space. Only commit if it survives
            // `keep_streamline_for_plan`.
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

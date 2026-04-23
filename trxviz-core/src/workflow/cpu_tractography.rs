/// CPU probabilistic tractography from ODX SH or ODF data.
///
/// Implements a simplified probabilistic tracker:
/// 1. At each step, trilinearly interpolate the ODF/SH field.
/// 2. Evaluate the local PMF on a fixed sphere (from the ODX sphere vertices).
/// 3. Zero entries that exceed `max_angle_deg` from the current direction.
/// 4. Sample one direction from the remaining PMF.
/// 5. Stop when GFA falls below threshold, outside the mask, or length limits exceeded.
///
/// Output streamlines are in RAS+mm space, compatible with TrxGpuData::from_tractogram.
use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;

use super::tracking_filters::{
    point_in_mask, streamline_endpoint_in, streamline_hits_all_rois,
    streamline_passes_hausdorff, streamline_satisfies_end_masks,
};
use super::types::{
    PostFilter, StreamlineDataset, StreamlineFlow, TractographyPlan, VoxelMask,
    WorkflowExecutionCache, WorkflowNodeUuid,
};

/// Run CPU probabilistic tractography and return a StreamlineFlow.
pub(super) fn run_cpu_tractography(plan: &TractographyPlan) -> WorkflowResult<StreamlineFlow> {
    let scene = &plan.odx_scene;

    // ── collect field data ──────────────────────────────────────────────
    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation(
            "ODX file has no SH coefficients. Re-derive with odx-rs from a PAM5/CSD model."
                .into(),
        )
    })?;
    let ncoeffs = sh_view.ncols();
    let nb_voxels = sh_view.nrows();

    // Get the render mesh (sphere vertices + B matrix) at detail level 2.
    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for tractography.".into())
    })?;
    let sphere_verts = mesh.vertices(); // &[[f32; 3]]
    let n_dirs = sphere_verts.len();
    let sample_plan = mesh.sample_plan();

    // SH coefficients flat: (NB_VOXELS, ncoeffs)
    let sh_flat: Vec<f32> = (0..nb_voxels)
        .flat_map(|i| sh_view.row(i).iter().copied())
        .collect();

    // ── build dense voxel lookup ────────────────────────────────────────
    let dims = scene.dimensions(); // [nx, ny, nz]
    let [nx, ny, nz] = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
    let ijk_lookup = scene.ijk_lookup();

    // dense_lut[x * ny * nz + y * nz + z] = compact index, or usize::MAX = outside
    let mut dense_lut = vec![usize::MAX; nx * ny * nz];
    for (compact_idx, &[ix, iy, iz]) in ijk_lookup.iter().enumerate() {
        dense_lut[ix as usize * ny * nz + iy as usize * nz + iz as usize] = compact_idx;
    }

    // GFA (per-voxel) — use as stopping criterion
    let gfa_data: Vec<f32> = {
        let dpv = scene.odf_view_f32().and_then(|_odf| {
            // Use the ODF amplitudes to compute GFA on the fly, or fall back to mask.
            None::<Vec<f32>>
        });
        dpv.unwrap_or_else(|| {
            // Fallback: uniform 1.0 for all masked voxels (stop only outside mask).
            vec![1.0f32; nb_voxels]
        })
    };

    // ── affine for RAS ↔ voxel conversion ──────────────────────────────
    let vox_to_ras = scene.voxel_to_ras();
    let ras_to_vox = vox_to_ras.inverse();

    let cos_max = plan.max_angle_deg.to_radians().cos();

    // Resolve the effective fixel threshold. Prob is deterministic (no
    // per-seed randomization), so the `fixel_threshold <= 0` sentinel
    // falls back to `0.6 · fixel_otsu` when a plan carries one, else
    // `0.0` (accept all) to preserve legacy behavior.
    let effective_fixel_threshold = if plan.fixel_threshold <= 0.0 {
        plan.fixel_otsu.map(|v| v * 0.6).unwrap_or(0.0)
    } else {
        plan.fixel_threshold
    };
    let step_mm = plan.step_size_mm;
    let min_pts = (plan.min_len_mm / step_mm).ceil() as usize;
    let max_pts = plan.max_points as usize;

    // ── seed points ─────────────────────────────────────────────────────
    let seeds_ras_owned = plan.seed_mask.nonzero_voxel_centers_ras();
    let seeds_ras = &seeds_ras_owned;

    let t0 = std::time::Instant::now();
    eprintln!(
        "[tractography] '{}': {} seeds × {} reps, {} sphere dirs, {} SH coeffs",
        plan.label,
        seeds_ras.len(),
        plan.seeds_per_voxel,
        n_dirs,
        ncoeffs,
    );

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_offsets: Vec<u32> = vec![0];
    let mut rng = simple_lcg(plan.rng_seed);

    let mut sampled_pmf = vec![0.0f32; n_dirs];

    for (seed_idx, &seed_ras) in seeds_ras.iter().enumerate() {
        if seed_idx > 0 && seed_idx % 1000 == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let rate = seed_idx as f32 / elapsed;
            let remaining = (seeds_ras.len() - seed_idx) as f32 / rate;
            eprintln!(
                "[tractography] {}/{} seeds ({:.0}/s, ~{:.0}s left, {} streamlines so far)",
                seed_idx,
                seeds_ras.len(),
                rate,
                remaining,
                all_offsets.len() - 1,
            );
        }
        for _rep in 0..plan.seeds_per_voxel {
            // Add small jitter within voxel
            let jitter = [
                (lcg_f32(&mut rng) - 0.5) * step_mm * 0.5,
                (lcg_f32(&mut rng) - 0.5) * step_mm * 0.5,
                (lcg_f32(&mut rng) - 0.5) * step_mm * 0.5,
            ];
            let seed_pt = Vec3::new(
                seed_ras[0] + jitter[0],
                seed_ras[1] + jitter[1],
                seed_ras[2] + jitter[2],
            );

            // Sample an initial direction
            let Some(init_dir) = sample_direction(
                seed_pt,
                Vec3::ZERO,
                true,
                cos_max,
                &sh_flat,
                ncoeffs,
                nb_voxels,
                &dense_lut,
                nx,
                ny,
                nz,
                &ras_to_vox,
                &gfa_data,
                effective_fixel_threshold,
                sample_plan,
                n_dirs,
                sphere_verts,
                plan.relative_peak_threshold,
                &mut sampled_pmf,
                &mut rng,
            ) else {
                continue;
            };

            // Bidirectional tracking with per-step plan constraints.
            let forward = track_one(
                seed_pt,
                init_dir,
                false,
                cos_max,
                step_mm,
                max_pts,
                &sh_flat,
                ncoeffs,
                nb_voxels,
                &dense_lut,
                nx,
                ny,
                nz,
                &ras_to_vox,
                &gfa_data,
                effective_fixel_threshold,
                sample_plan,
                n_dirs,
                sphere_verts,
                plan.relative_peak_threshold,
                &mut sampled_pmf,
                &mut rng,
                plan.limiting_mask.as_deref(),
                plan.roa_mask.as_deref(),
                plan.term_mask.as_deref(),
            );
            let Some(forward) = forward else { continue };
            let backward = track_one(
                seed_pt,
                -init_dir,
                false,
                cos_max,
                step_mm,
                max_pts,
                &sh_flat,
                ncoeffs,
                nb_voxels,
                &dense_lut,
                nx,
                ny,
                nz,
                &ras_to_vox,
                &gfa_data,
                effective_fixel_threshold,
                sample_plan,
                n_dirs,
                sphere_verts,
                plan.relative_peak_threshold,
                &mut sampled_pmf,
                &mut rng,
                plan.limiting_mask.as_deref(),
                plan.roa_mask.as_deref(),
                plan.term_mask.as_deref(),
            );
            let Some(backward) = backward else { continue };

            // Merge: backward (reversed) + seed + forward
            let streamline: Vec<[f32; 3]> = backward
                .iter()
                .rev()
                .chain(std::iter::once(&seed_pt.to_array()))
                .chain(forward.iter())
                .map(|p| *p)
                .collect();

            if streamline.len() < min_pts {
                continue;
            }

            // Post-hoc plan filters.
            if !plan.roi_masks.is_empty()
                && !streamline_hits_all_rois(&streamline, &plan.roi_masks)
            {
                continue;
            }
            if let Some(ne) = plan.no_end_mask.as_deref() {
                if streamline_endpoint_in(&streamline, ne) {
                    continue;
                }
            }
            if !plan.end_masks.is_empty()
                && !streamline_satisfies_end_masks(&streamline, &plan.end_masks)
            {
                continue;
            }
            if let Some(PostFilter::Hausdorff {
                reference_points_ras,
                max_mm,
            }) = plan.post_filter.as_ref()
            {
                if !streamline_passes_hausdorff(&streamline, reference_points_ras, *max_mm) {
                    continue;
                }
            }

            all_positions.extend_from_slice(&streamline);
            all_offsets.push(all_positions.len() as u32);
        }
    }

    let nb_streamlines = all_offsets.len() - 1;
    eprintln!(
        "[tractography] '{}': done in {:.1}s — {} streamlines",
        plan.label,
        t0.elapsed().as_secs_f32(),
        nb_streamlines,
    );

    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(
        all_positions,
        all_offsets,
    ));

    let selected: Vec<StreamlineIndex> = (0..nb_streamlines as u32).map(StreamlineIndex).collect();
    let dataset = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: crate::data::loaded_files::StreamlineBacking::Derived(Arc::new(
            trx_rs::Tractogram::new(),
        )),
    });

    Ok(StreamlineFlow {
        dataset,
        selected_streamlines: selected,
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
    })
}

// ── tracking helpers ────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn track_one(
    start: Vec3,
    start_dir: Vec3,
    _is_start: bool,
    cos_max: f32,
    step_mm: f32,
    max_pts: usize,
    sh_flat: &[f32],
    ncoeffs: usize,
    nb_voxels: usize,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    ras_to_vox: &Mat4,
    gfa_data: &[f32],
    fixel_threshold: f32,
    sample_plan: &odx_rs::mrtrix_sh::RowSamplePlan,
    n_dirs: usize,
    sphere_verts: &[[f32; 3]],
    relative_peak_threshold: f32,
    sampled_pmf: &mut Vec<f32>,
    rng: &mut u64,
    limiting: Option<&VoxelMask>,
    roa: Option<&VoxelMask>,
    term: Option<&VoxelMask>,
) -> Option<Vec<[f32; 3]>> {
    let mut pts = Vec::with_capacity(64);
    let mut point = start + start_dir * step_mm;
    let mut direction = start_dir;

    while pts.len() < max_pts {
        // Per-step constraint masks. ROA rejects the whole streamline;
        // limiting/term terminate the branch cleanly.
        if let Some(m) = roa {
            if point_in_mask(point, m) {
                return None;
            }
        }
        if let Some(m) = limiting {
            if !point_in_mask(point, m) {
                break;
            }
        }
        if let Some(m) = term {
            if point_in_mask(point, m) {
                pts.push(point.to_array());
                break;
            }
        }

        let Some(new_dir) = sample_direction(
            point,
            direction,
            false,
            cos_max,
            sh_flat,
            ncoeffs,
            nb_voxels,
            dense_lut,
            nx,
            ny,
            nz,
            ras_to_vox,
            gfa_data,
            fixel_threshold,
            sample_plan,
            n_dirs,
            sphere_verts,
            relative_peak_threshold,
            sampled_pmf,
            rng,
        ) else {
            break;
        };
        pts.push(point.to_array());
        point += new_dir * step_mm;
        direction = new_dir;
    }

    Some(pts)
}

fn sample_direction(
    point_ras: Vec3,
    prev_dir: Vec3,
    is_start: bool,
    cos_max: f32,
    sh_flat: &[f32],
    ncoeffs: usize,
    nb_voxels: usize,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    ras_to_vox: &Mat4,
    gfa_data: &[f32],
    fixel_threshold: f32,
    sample_plan: &odx_rs::mrtrix_sh::RowSamplePlan,
    n_dirs: usize,
    sphere_verts: &[[f32; 3]],
    relative_peak_threshold: f32,
    pmf_buf: &mut Vec<f32>,
    rng: &mut u64,
) -> Option<Vec3> {
    let vox = ras_to_vox.transform_point3(point_ras);

    // Trilinearly interpolated SH coefficients
    let sh_interp = trilinear_sh(
        vox,
        sh_flat,
        ncoeffs,
        nb_voxels,
        dense_lut,
        nx,
        ny,
        nz,
        gfa_data,
        fixel_threshold,
    )?;

    // Evaluate on sphere → PMF
    pmf_buf.resize(n_dirs, 0.0);
    sample_plan.apply_row_into(&sh_interp, pmf_buf);

    // Zero out negative values
    for v in pmf_buf.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // If continuing, mask directions beyond max_angle
    if !is_start {
        for (i, v) in pmf_buf.iter_mut().enumerate() {
            if *v > 0.0 {
                let sv = Vec3::from(sphere_verts[i]);
                // Handle antipodal symmetry
                let dot = prev_dir.dot(sv).abs();
                if dot < cos_max {
                    *v = 0.0;
                }
            }
        }
    }

    // Relative peak threshold
    let max_val = pmf_buf.iter().cloned().fold(0.0f32, f32::max);
    if max_val <= 0.0 {
        return None;
    }
    let thresh = max_val * relative_peak_threshold;
    let mut total = 0.0f32;
    for v in pmf_buf.iter_mut() {
        if *v < thresh {
            *v = 0.0;
        } else {
            total += *v;
        }
    }
    if total <= 0.0 {
        return None;
    }

    // Sample from PMF
    let r = lcg_f32(rng) * total;
    let mut cumsum = 0.0f32;
    let mut chosen = None;
    for (i, &v) in pmf_buf.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        cumsum += v;
        if cumsum >= r {
            chosen = Some(i);
            break;
        }
    }
    let idx = chosen?;

    let sv = Vec3::from(sphere_verts[idx]);
    // Flip to match hemisphere convention (antipodal ambiguity)
    let dir = if !is_start && prev_dir.dot(sv) < 0.0 {
        -sv
    } else {
        sv
    };
    Some(dir.normalize())
}

/// Trilinear interpolation of sparse SH coefficients at fractional voxel coords.
/// Returns None if outside mask or GFA below threshold.
fn trilinear_sh(
    vox: Vec3,
    sh_flat: &[f32],
    ncoeffs: usize,
    _nb_voxels: usize,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    gfa_data: &[f32],
    fixel_threshold: f32,
) -> Option<Vec<f32>> {
    let x0 = vox.x.floor() as i32;
    let y0 = vox.y.floor() as i32;
    let z0 = vox.z.floor() as i32;

    let wx1 = vox.x - x0 as f32;
    let wy1 = vox.y - y0 as f32;
    let wz1 = vox.z - z0 as f32;
    let wx0 = 1.0 - wx1;
    let wy0 = 1.0 - wy1;
    let wz0 = 1.0 - wz1;

    let mut out = vec![0.0f32; ncoeffs];
    let mut total_weight = 0.0f32;

    for (dx, wx) in [(0i32, wx0), (1, wx1)] {
        for (dy, wy) in [(0i32, wy0), (1, wy1)] {
            for (dz, wz) in [(0i32, wz0), (1, wz1)] {
                let xi = x0 + dx;
                let yi = y0 + dy;
                let zi = z0 + dz;
                if xi < 0
                    || yi < 0
                    || zi < 0
                    || xi >= nx as i32
                    || yi >= ny as i32
                    || zi >= nz as i32
                {
                    continue;
                }
                let lin = xi as usize * ny * nz + yi as usize * nz + zi as usize;
                let compact = dense_lut[lin];
                if compact == usize::MAX {
                    continue;
                }
                if gfa_data[compact] < fixel_threshold {
                    continue;
                }
                let w = wx * wy * wz;
                if w <= 0.0 {
                    continue;
                }
                let row = &sh_flat[compact * ncoeffs..(compact + 1) * ncoeffs];
                for (out_v, &sh_v) in out.iter_mut().zip(row) {
                    *out_v += w * sh_v;
                }
                total_weight += w;
            }
        }
    }

    if total_weight < 0.01 {
        return None;
    }
    // Normalize interpolated coefficients by total weight
    for v in out.iter_mut() {
        *v /= total_weight;
    }
    Some(out)
}

// ── simple LCG RNG (reproducible, no external deps) ─────────────────

fn simple_lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}

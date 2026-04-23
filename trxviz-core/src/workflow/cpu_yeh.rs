//! Fixel-based CPU tractography in the Yeh (DSI-Studio) style.
//!
//! Random-sampling seeding: at each iteration pick a random seed voxel,
//! attempt one bidirectional streamline, keep it if it meets the length
//! floor. Stop when either `target_streamlines` is reached or
//! `max_seed_attempts` has been consumed.
//!
//! When `seed_mask` is absent, seed from every voxel that has at least one
//! fixel peak.
use std::sync::Arc;

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;

use super::tracking_filters::{
    point_in_mask, streamline_endpoint_in, streamline_hits_all_rois,
    streamline_passes_hausdorff, streamline_satisfies_end_masks,
};
use super::types::{
    PostFilter, StreamlineDataset, StreamlineFlow, VoxelMask, YehTractographyPlan,
};

/// Outcome of a single seed attempt. Keeps the kept streamline inline so
/// the parent loop can drain them into the shared output without heap
/// contention across threads.
enum AttemptOutcome {
    Kept(Vec<[f32; 3]>),
    SkipEmpty,         // picked voxel has no fixels or is outside seed mask
    InitialPeakFailed, // no viable initial direction above fixel threshold
    RejectRoa,         // per-step ROA hit
    RejectMinLen,
    RejectRoi,
    RejectEnd,
    RejectNoEnd,
    RejectHausdorff,
}

/// Shared read-only context for per-attempt tracking. Constructed once per
/// `run_cpu_yeh` and passed as a `&` to every parallel closure.
struct YehCtx<'a> {
    plan: &'a YehTractographyPlan,
    offsets: &'a [u32],
    directions: &'a [[f32; 3]],
    fixel_amplitude: &'a [f32],
    dense_lut: &'a [usize],
    ijk_lookup: &'a [[u32; 3]],
    centers_ras: &'a [[f32; 3]],
    nx: usize,
    ny: usize,
    nz: usize,
    nb_voxels: usize,
    vox_to_ras: Mat4,
    ras_to_vox: Mat4,
    max_pts: usize,
    min_pts: usize,
    max_len_pts: usize,
    mask_grid_matches: bool,
}

pub(super) fn run_cpu_yeh(plan: &YehTractographyPlan) -> WorkflowResult<StreamlineFlow> {
    let scene = &plan.odx_scene;
    let dataset = scene.dataset();

    let offsets = dataset.offsets().to_vec();
    if offsets.len() <= 1 {
        return Err(WorkflowError::Evaluation(
            "ODX file has no fixels; Yeh tracking requires a peak representation.".into(),
        ));
    }
    let directions: Vec<[f32; 3]> = dataset.directions().to_vec();
    let n_fixels = directions.len();

    // Per-fixel tracking amplitude: prefer "qa" then "fa"; fall back
    // to uniform 1.0. Name is intentionally metric-agnostic — the
    // scalar may be QA (GQI/GQI2), AFD (CSD), FA (DTI), or any other
    // DPF the user chose.
    let fixel_amplitude: Vec<f32> = ["qa", "fa"]
        .iter()
        .find_map(|name| dataset.scalar_dpf_f32(name).ok())
        .unwrap_or_else(|| vec![1.0f32; n_fixels]);

    let dims = scene.dimensions();
    let [nx, ny, nz] = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
    let ijk_lookup = scene.ijk_lookup();

    let mut dense_lut = vec![usize::MAX; nx * ny * nz];
    for (compact_idx, &[ix, iy, iz]) in ijk_lookup.iter().enumerate() {
        dense_lut[ix as usize * ny * nz + iy as usize * nz + iz as usize] = compact_idx;
    }

    let vox_to_ras = scene.voxel_to_ras();
    let ras_to_vox = vox_to_ras.inverse();

    let max_pts = plan.max_points as usize;
    let min_pts = ((plan.min_len_mm / plan.step_size_mm.max(0.01)).ceil() as usize).max(2);
    let max_len_pts = ((plan.max_len_mm / plan.step_size_mm.max(0.01)).ceil() as usize).max(2);

    // If a seed mask is supplied, express it in the ODX grid frame so we can
    // test membership of a randomly sampled compact voxel. When the mask's
    // grid matches the ODX grid exactly we can test via ijk directly; this is
    // the expected case for masks built from ROI-from-* producers driven by
    // this same ODX.
    let centers_ras = scene.centers_ras();
    let nb_voxels = centers_ras.len();
    if nb_voxels == 0 {
        log::warn!("[yeh] '{}': ODX has no mask voxels", plan.label);
        return Ok(empty_flow(&plan.label));
    }
    let mask_grid_matches = plan
        .seed_mask
        .as_ref()
        .map(|m| {
            m.dims[0] as usize == nx
                && m.dims[1] as usize == ny
                && m.dims[2] as usize == nz
        })
        .unwrap_or(true);
    if !mask_grid_matches {
        log::warn!(
            "[yeh] '{}': seed mask grid does not match ODX grid — ignoring mask",
            plan.label
        );
    }

    let target = plan.target_streamlines as usize;
    let attempt_budget = plan.max_seed_attempts as usize;
    let t0 = std::time::Instant::now();
    let n_threads = rayon::current_num_threads();
    log::info!(
        "[yeh] '{}': {} mask voxels, target {} streamlines, max {} attempts ({} rayon threads)",
        plan.label,
        nb_voxels,
        target,
        attempt_budget,
        n_threads,
    );

    let ctx = YehCtx {
        plan,
        offsets: &offsets,
        directions: &directions,
        fixel_amplitude: &fixel_amplitude,
        dense_lut: &dense_lut,
        ijk_lookup,
        centers_ras,
        nx,
        ny,
        nz,
        nb_voxels,
        vox_to_ras,
        ras_to_vox,
        max_pts,
        min_pts,
        max_len_pts,
        mask_grid_matches,
    };

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_offsets: Vec<u32> = vec![0];
    let mut attempts_used = 0usize;
    let mut rejected_roa = 0usize;
    let mut rejected_min_len = 0usize;
    let mut rejected_roi = 0usize;
    let mut rejected_end = 0usize;
    let mut rejected_no_end = 0usize;
    let mut rejected_hausdorff = 0usize;
    let mut rejected_skip = 0usize;
    let mut rejected_initial = 0usize;

    // Process attempts in chunks so we can stop cleanly once we hit the
    // streamline target without wasting an entire 10M-attempt parallel
    // sweep. Chunk sizes scale with thread count.
    let chunk_size = (n_threads * 256).max(1024);
    let mut next_attempt: u64 = 0;
    while all_offsets.len() - 1 < target && (next_attempt as usize) < attempt_budget {
        let end = ((next_attempt as usize) + chunk_size).min(attempt_budget);
        let outcomes: Vec<AttemptOutcome> = (next_attempt..end as u64)
            .into_par_iter()
            .map(|idx| try_one_attempt(&ctx, idx))
            .collect();
        attempts_used = end;
        next_attempt = end as u64;

        for outcome in outcomes {
            if all_offsets.len() - 1 >= target {
                break;
            }
            match outcome {
                AttemptOutcome::Kept(streamline) => {
                    all_positions.extend_from_slice(&streamline);
                    all_offsets.push(all_positions.len() as u32);
                }
                AttemptOutcome::SkipEmpty => rejected_skip += 1,
                AttemptOutcome::InitialPeakFailed => rejected_initial += 1,
                AttemptOutcome::RejectRoa => rejected_roa += 1,
                AttemptOutcome::RejectMinLen => rejected_min_len += 1,
                AttemptOutcome::RejectRoi => rejected_roi += 1,
                AttemptOutcome::RejectEnd => rejected_end += 1,
                AttemptOutcome::RejectNoEnd => rejected_no_end += 1,
                AttemptOutcome::RejectHausdorff => rejected_hausdorff += 1,
            }
        }

        log::info!(
            "[yeh] {} streamlines after {} attempts",
            all_offsets.len() - 1,
            attempts_used,
        );
    }
    let attempts = attempts_used;

    let nb_streamlines = all_offsets.len() - 1;
    log::info!(
        "[yeh] '{}': done in {:.1}s — {} streamlines from {} attempts \
         (rejected: skip={} init={} min_len={} roa/term={} roi={} end={} no_end={} hausdorff={})",
        plan.label,
        t0.elapsed().as_secs_f32(),
        nb_streamlines,
        attempts,
        rejected_skip,
        rejected_initial,
        rejected_min_len,
        rejected_roa,
        rejected_roi,
        rejected_end,
        rejected_no_end,
        rejected_hausdorff,
    );

    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(
        all_positions,
        all_offsets,
    ));
    let selected: Vec<StreamlineIndex> =
        (0..nb_streamlines as u32).map(StreamlineIndex).collect();
    let dataset_out = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: crate::data::loaded_files::StreamlineBacking::Derived(Arc::new(
            trx_rs::Tractogram::new(),
        )),
    });

    Ok(StreamlineFlow {
        dataset: dataset_out,
        selected_streamlines: selected,
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
    })
}

fn empty_flow(label: &str) -> StreamlineFlow {
    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(Vec::new(), vec![0]));
    let dataset = Arc::new(StreamlineDataset {
        name: label.to_string(),
        gpu_data,
        backing: crate::data::loaded_files::StreamlineBacking::Derived(Arc::new(
            trx_rs::Tractogram::new(),
        )),
    });
    StreamlineFlow {
        dataset,
        selected_streamlines: Vec::new(),
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
    }
}

/// One deterministic seed-attempt. Builds its own RNG from
/// `(plan.rng_seed, attempt_idx)` so the stream of attempts is reproducible
/// across CPU counts and parallelism decisions.
fn try_one_attempt(ctx: &YehCtx<'_>, attempt_idx: u64) -> AttemptOutcome {
    let plan = ctx.plan;
    // Derive a well-dispersed u64 from (rng_seed, attempt_idx). The two
    // multiply-adds produce a SplitMix-style state uncorrelated with the
    // input order so adjacent attempts don't produce similar streamlines.
    let mut rng = plan
        .rng_seed
        .wrapping_add(0x9E3779B97F4A7C15)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        .wrapping_add(attempt_idx.wrapping_mul(0x94D049BB133111EB));

    // Rejection-sample a random mask voxel with at least one fixel and (if
    // a seed mask is wired) inside it.
    let compact_idx = (lcg_u32(&mut rng) as usize) % ctx.nb_voxels;
    if compact_idx + 1 >= ctx.offsets.len() {
        return AttemptOutcome::SkipEmpty;
    }
    if ctx.offsets[compact_idx + 1] == ctx.offsets[compact_idx] {
        return AttemptOutcome::SkipEmpty;
    }
    if let (Some(mask), true) = (&plan.seed_mask, ctx.mask_grid_matches) {
        let [i, j, k] = ctx.ijk_lookup[compact_idx];
        let idx = (i as usize)
            + (mask.dims[0] as usize)
                * ((j as usize) + (mask.dims[1] as usize) * (k as usize));
        if mask.data.get(idx).copied().unwrap_or(0) == 0 {
            return AttemptOutcome::SkipEmpty;
        }
    }
    let seed_ras = ctx.centers_ras[compact_idx];

    // DSI-Studio-style sentinels (see the outer doc).
    let step_mm = if plan.step_size_mm <= 0.0 {
        // DSI-Studio `tracking_thread.cpp`: `param.step_size = vs[0] ·
        // (0.5 + rand())` — step is expressed in voxel units, not mm.
        let vs = ctx.vox_to_ras.col(0).truncate().length()
            .min(ctx.vox_to_ras.col(1).truncate().length())
            .min(ctx.vox_to_ras.col(2).truncate().length())
            .max(1e-3);
        vs * (0.5 + lcg_f32(&mut rng))
    } else {
        let _ = lcg_f32(&mut rng);
        plan.step_size_mm
    };
    let angle_rad = if plan.max_angle_deg <= 0.0 {
        // DSI-Studio `tracking_thread.cpp`: when turning angle is
        // unspecified (autotrack mode), randomize the half-angle in
        // [15°, 90°] per attempt — `15 + 75 · rand()`.
        let t = lcg_f32(&mut rng);
        (15.0 + 75.0 * t).to_radians()
    } else {
        let _ = lcg_f32(&mut rng);
        plan.max_angle_deg.to_radians()
    };
    let cos_max = angle_rad.cos();
    let fixel_threshold = if plan.fixel_threshold <= 0.0 {
        // DSI-Studio `tracking_thread.cpp:215–221`: jitter in
        // `[(default_otsu − 0.1) · fixel_otsu, (default_otsu + 0.1) · fixel_otsu]`
        // with `default_otsu = 0.6`. When the plan carries no
        // `fixel_otsu` we fall back to the legacy absolute 0.1 base so
        // existing graphs/tests keep working.
        let base = plan.fixel_otsu.unwrap_or(0.1);
        // lcg_f32 ∈ [0,1) → jitter ∈ [-0.1, 0.1)
        let jitter = lcg_f32(&mut rng) - 0.5;
        (base * (0.6 + jitter * 0.2)).clamp(0.0, 1.0)
    } else {
        let _ = lcg_f32(&mut rng);
        plan.fixel_threshold
    };
    let smooth_jitter = if plan.smooth_fraction >= 1.0 {
        lcg_f32(&mut rng) * 0.95
    } else {
        let _ = lcg_f32(&mut rng);
        plan.smooth_fraction.clamp(0.0, 0.95)
    };

    // DSI-Studio `tracking_thread.cpp`: seed jitter is uniform in
    // `[-0.5, +0.5]` voxels — covers the full seed voxel rather than a
    // central quarter.
    let jitter_vox = Vec3::new(
        lcg_f32(&mut rng) - 0.5,
        lcg_f32(&mut rng) - 0.5,
        lcg_f32(&mut rng) - 0.5,
    );
    let seed_vox = ctx.ras_to_vox.transform_point3(Vec3::from(seed_ras)) + jitter_vox;
    let seed_pt = ctx.vox_to_ras.transform_point3(seed_vox);

    let Some(init_peak) = pick_initial_peak(
        seed_vox,
        ctx.offsets,
        ctx.directions,
        ctx.fixel_amplitude,
        fixel_threshold,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &mut rng,
    ) else {
        return AttemptOutcome::InitialPeakFailed;
    };
    let sign = if lcg_f32(&mut rng) < 0.5 { 1.0 } else { -1.0 };
    let init_dir = init_peak * sign;

    let forward = track_one(
        seed_pt,
        init_dir,
        step_mm,
        cos_max,
        smooth_jitter,
        fixel_threshold,
        ctx.max_pts / 2,
        ctx.max_len_pts / 2,
        ctx.offsets,
        ctx.directions,
        ctx.fixel_amplitude,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &ctx.ras_to_vox,
        plan.limiting_mask.as_deref(),
        plan.roa_mask.as_deref(),
        plan.term_mask.as_deref(),
    );
    let Some(forward) = forward else {
        return AttemptOutcome::RejectRoa;
    };
    let backward = track_one(
        seed_pt,
        -init_dir,
        step_mm,
        cos_max,
        smooth_jitter,
        fixel_threshold,
        ctx.max_pts / 2,
        ctx.max_len_pts / 2,
        ctx.offsets,
        ctx.directions,
        ctx.fixel_amplitude,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &ctx.ras_to_vox,
        plan.limiting_mask.as_deref(),
        plan.roa_mask.as_deref(),
        plan.term_mask.as_deref(),
    );
    let Some(backward) = backward else {
        return AttemptOutcome::RejectRoa;
    };

    let streamline: Vec<[f32; 3]> = backward
        .iter()
        .rev()
        .chain(std::iter::once(&seed_pt.to_array()))
        .chain(forward.iter())
        .copied()
        .collect();

    if streamline.len() < ctx.min_pts {
        return AttemptOutcome::RejectMinLen;
    }
    if !plan.roi_masks.is_empty()
        && !streamline_hits_all_rois(&streamline, &plan.roi_masks)
    {
        return AttemptOutcome::RejectRoi;
    }
    if let Some(ne) = plan.no_end_mask.as_deref() {
        if streamline_endpoint_in(&streamline, ne) {
            return AttemptOutcome::RejectNoEnd;
        }
    }
    if !plan.end_masks.is_empty()
        && !streamline_satisfies_end_masks(&streamline, &plan.end_masks)
    {
        return AttemptOutcome::RejectEnd;
    }
    if let Some(PostFilter::Hausdorff {
        reference_points_ras,
        max_mm,
    }) = plan.post_filter.as_ref()
    {
        if !streamline_passes_hausdorff(&streamline, reference_points_ras, *max_mm) {
            return AttemptOutcome::RejectHausdorff;
        }
    }

    AttemptOutcome::Kept(streamline)
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Track one branch entirely in RAS+mm. Returns `Some(points)` on a clean
/// termination (including clean limiting/term-mask exits), or `None` when
/// the streamline entered a `roa_mask` — the whole streamline should be
/// discarded in that case.
#[allow(clippy::too_many_arguments)]
fn track_one(
    start: Vec3,
    mut dir: Vec3,
    step_mm: f32,
    cos_max: f32,
    smooth: f32,
    fixel_threshold: f32,
    max_pts: usize,
    max_len_pts: usize,
    offsets: &[u32],
    directions: &[[f32; 3]],
    fixel_amplitude: &[f32],
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    ras_to_vox: &Mat4,
    limiting: Option<&VoxelMask>,
    roa: Option<&VoxelMask>,
    term: Option<&VoxelMask>,
) -> Option<Vec<[f32; 3]>> {
    let mut out = Vec::<[f32; 3]>::new();
    if dir.length_squared() < 1e-8 {
        return Some(out);
    }
    dir = dir.normalize();

    let mut pt_ras = start;
    let cap = max_pts.min(max_len_pts);

    for _ in 0..cap {
        pt_ras += dir * step_mm;

        let pt_vox = ras_to_vox.transform_point3(pt_ras);
        let Some(compact_idx) = voxel_at(pt_vox, dense_lut, nx, ny, nz) else {
            break;
        };
        if compact_idx + 1 >= offsets.len() {
            break;
        }

        // Per-step constraint masks. ROA rejects the whole streamline;
        // limiting/term terminate cleanly.
        if let Some(m) = roa {
            if point_in_mask(pt_ras, m) {
                return None;
            }
        }
        if let Some(m) = limiting {
            if !point_in_mask(pt_ras, m) {
                break;
            }
        }
        if let Some(m) = term {
            if point_in_mask(pt_ras, m) {
                out.push(pt_ras.to_array());
                break;
            }
        }

        let Some((best_dir, best_fa)) = best_peak(
            compact_idx,
            offsets,
            directions,
            fixel_amplitude,
            &dir,
            cos_max,
        ) else {
            break;
        };
        if best_fa < fixel_threshold {
            break;
        }

        let new_dir = ((1.0 - smooth) * best_dir + smooth * dir).normalize_or_zero();
        if new_dir.length_squared() < 1e-8 {
            break;
        }
        dir = new_dir;

        out.push(pt_ras.to_array());
    }

    Some(out)
}

fn voxel_at(
    pt_vox: Vec3,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Option<usize> {
    let x = pt_vox.x.floor() as i32;
    let y = pt_vox.y.floor() as i32;
    let z = pt_vox.z.floor() as i32;
    if x < 0 || y < 0 || z < 0 {
        return None;
    }
    let (x, y, z) = (x as usize, y as usize, z as usize);
    if x >= nx || y >= ny || z >= nz {
        return None;
    }
    let lut_idx = x * ny * nz + y * nz + z;
    let compact = dense_lut[lut_idx];
    if compact == usize::MAX {
        None
    } else {
        Some(compact)
    }
}

fn best_peak(
    compact_idx: usize,
    offsets: &[u32],
    directions: &[[f32; 3]],
    fixel_amplitude: &[f32],
    incoming: &Vec3,
    cos_max: f32,
) -> Option<(Vec3, f32)> {
    let start = offsets[compact_idx] as usize;
    let end = offsets[compact_idx + 1] as usize;
    let mut best: Option<(Vec3, f32, f32)> = None;
    for k in start..end {
        if k >= directions.len() {
            break;
        }
        let peak = Vec3::from(directions[k]);
        if peak.length_squared() < 1e-8 {
            continue;
        }
        let peak = peak.normalize();
        let d = peak.dot(*incoming);
        let abs_d = d.abs();
        if abs_d < cos_max {
            continue;
        }
        let amplitude = fixel_amplitude.get(k).copied().unwrap_or(1.0);
        let flipped = if d >= 0.0 { peak } else { -peak };
        match &best {
            Some((_, _, best_abs)) if abs_d <= *best_abs => {}
            _ => best = Some((flipped, amplitude, abs_d)),
        }
    }
    best.map(|(d, amp, _)| (d, amp))
}

fn pick_initial_peak(
    seed_vox: Vec3,
    offsets: &[u32],
    directions: &[[f32; 3]],
    fixel_amplitude: &[f32],
    fixel_threshold: f32,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    rng: &mut u64,
) -> Option<Vec3> {
    let compact_idx = voxel_at(seed_vox, dense_lut, nx, ny, nz)?;
    if compact_idx + 1 >= offsets.len() {
        return None;
    }
    let start = offsets[compact_idx] as usize;
    let end = offsets[compact_idx + 1] as usize;
    let mut candidates: Vec<Vec3> = Vec::new();
    for k in start..end {
        if k >= directions.len() {
            break;
        }
        let amplitude = fixel_amplitude.get(k).copied().unwrap_or(1.0);
        if amplitude < fixel_threshold {
            continue;
        }
        let peak = Vec3::from(directions[k]);
        if peak.length_squared() < 1e-8 {
            continue;
        }
        candidates.push(peak.normalize());
    }
    if candidates.is_empty() {
        None
    } else {
        let idx = (lcg_f32(rng) * candidates.len() as f32) as usize;
        Some(candidates[idx.min(candidates.len() - 1)])
    }
}

/// Advance the LCG and return a uniform `f32` in `[0, 1)`. The high 32 bits
/// of the state are used so the returned value spans the full range.
fn lcg_f32(state: &mut u64) -> f32 {
    (lcg_u32(state) as f32) / 4_294_967_296.0
}

/// Advance the LCG and return the high 32 bits as a `u32`.
fn lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

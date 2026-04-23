//! Fixel-based CPU tractography in the Yeh (DSI-Studio) style.
//!
//! This module is the thin "how to turn a `YehTractographyPlan` into
//! streamlines" wrapper: load the fixel arrays from the ODX, build the
//! dense voxel-index LUT, decide per-attempt randomized params (step,
//! angle, threshold, smoothing) from DSI-Studio's sentinel rules, fire
//! the rayon fold+reduce loop, and assemble the output `StreamlineFlow`.
//! The actual per-step algorithm lives in `tracking::dg_yeh::YehFixelDG`,
//! and the bidirectional assembly + post-hoc filtering lives in
//! `tracking::tracker::try_one_attempt`.
//!
//! Random-sampling seeding: at each iteration pick a random masked voxel
//! with fixels, attempt one bidirectional streamline, keep it if it
//! survives all filters. Stop when `target_streamlines` is reached or
//! `max_seed_attempts` has been consumed.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use glam::Vec3;
use rayon::prelude::*;

use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;

use super::tracking::accum::{AttemptOutcome, ThreadAccum};
use super::tracking::dg_yeh::{YehFixelDG, YehFixelGlobal, YehFixelScratch};
use super::tracking::masks::{PerStepMasks, PostFilterSet};
use super::tracking::rng::{lcg_f32, lcg_u32, split_mix_init};
use super::tracking::tracker::{TrackingLimits, try_one_attempt};
use super::types::{StreamlineDataset, StreamlineFlow, YehTractographyPlan};

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

    let centers_ras = scene.centers_ras();
    let nb_voxels = centers_ras.len();
    if nb_voxels == 0 {
        log::warn!("[yeh] '{}': ODX has no mask voxels", plan.label);
        return Ok(empty_flow(&plan.label));
    }
    let mask_grid_matches = plan
        .seed_mask
        .as_ref()
        .map(|m| m.dims[0] as usize == nx && m.dims[1] as usize == ny && m.dims[2] as usize == nz)
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

    let global = YehFixelGlobal {
        offsets: &offsets,
        directions: &directions,
        fixel_amplitude: &fixel_amplitude,
        dense_lut: &dense_lut,
        nx,
        ny,
        nz,
        ras_to_vox,
    };

    let limits = TrackingLimits {
        max_pts_per_branch: max_pts / 2,
        max_len_pts_per_branch: max_len_pts / 2,
        min_pts,
    };

    let ctx = YehAttemptCtx {
        plan,
        global: &global,
        limits,
        ijk_lookup,
        centers_ras,
        nb_voxels,
        vox_to_ras_smallest_vs: smallest_voxel_dim_mm(&vox_to_ras),
        mask_grid_matches,
    };

    // Single rayon par-iter over the whole attempt budget. Each worker
    // keeps its own ThreadAccum alive; reduce merges them at the end.
    // Determinism: each attempt's RNG is derived from (rng_seed, idx),
    // so individual streamlines are bit-identical across runs — only
    // the *order* of kept streamlines in the output may shift with
    // thread scheduling.
    let target_atomic = AtomicUsize::new(0);
    let merged: ThreadAccum<YehFixelScratch> = (0..attempt_budget)
        .into_par_iter()
        .with_min_len(64)
        .fold(ThreadAccum::<YehFixelScratch>::new, |mut acc, idx| {
            if target_atomic.load(Ordering::Relaxed) >= target {
                return acc;
            }
            let outcome = try_yeh_attempt(&ctx, idx as u64, &mut acc);
            acc.counts.bump(outcome);
            if matches!(outcome, AttemptOutcome::Kept) {
                target_atomic.fetch_add(1, Ordering::Relaxed);
            }
            acc
        })
        .reduce(ThreadAccum::<YehFixelScratch>::new, ThreadAccum::merge);

    let mut all_positions = merged.positions;
    let mut all_offsets = merged.offsets;
    let counts = merged.counts;

    // Trim overshoot from the racy target check — workers may have kept
    // a few extra streamlines after the target was hit.
    if all_offsets.len() - 1 > target {
        let cutoff = all_offsets[target] as usize;
        all_positions.truncate(cutoff);
        all_offsets.truncate(target + 1);
    }

    let nb_streamlines = all_offsets.len() - 1;
    log::info!(
        "[yeh] '{}': done in {:.1}s — {} streamlines from {} attempts \
         (rejected: skip={} init={} min_len={} roa/term={} roi={} end={} no_end={} hausdorff={})",
        plan.label,
        t0.elapsed().as_secs_f32(),
        nb_streamlines,
        counts.total_attempts(),
        counts.skip_empty,
        counts.no_initial,
        counts.min_len,
        counts.roa,
        counts.roi,
        counts.end,
        counts.no_end,
        counts.hausdorff,
    );

    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(
        all_positions,
        all_offsets,
    ));
    let selected: Vec<StreamlineIndex> = (0..nb_streamlines as u32).map(StreamlineIndex).collect();
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
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
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
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    }
}

/// Smallest column-length of the vox→RAS affine (mm per voxel), clamped
/// to avoid divide-by-zero blow-ups on degenerate affines. Matches the
/// legacy inline calc at the top of `try_one_attempt`.
fn smallest_voxel_dim_mm(vox_to_ras: &glam::Mat4) -> f32 {
    vox_to_ras
        .col(0)
        .truncate()
        .length()
        .min(vox_to_ras.col(1).truncate().length())
        .min(vox_to_ras.col(2).truncate().length())
        .max(1e-3)
}

/// Read-only shared state for one `run_cpu_yeh` call. Bundled so the per-
/// attempt closure takes one `&` instead of ~10 individual references.
struct YehAttemptCtx<'a> {
    plan: &'a YehTractographyPlan,
    global: &'a YehFixelGlobal<'a>,
    limits: TrackingLimits,
    ijk_lookup: &'a [[u32; 3]],
    centers_ras: &'a [[f32; 3]],
    nb_voxels: usize,
    vox_to_ras_smallest_vs: f32,
    mask_grid_matches: bool,
}

/// One deterministic seed-attempt. RNG derived from `(rng_seed, idx)` so
/// the streamline content is reproducible across runs and thread counts.
/// On `Kept`, the streamline has already been pushed to the accumulator.
fn try_yeh_attempt(
    ctx: &YehAttemptCtx<'_>,
    attempt_idx: u64,
    acc: &mut ThreadAccum<YehFixelScratch>,
) -> AttemptOutcome {
    let plan = ctx.plan;
    let mut rng = split_mix_init(plan.rng_seed, attempt_idx);

    // Rejection-sample a random masked voxel with at least one fixel.
    let compact_idx = (lcg_u32(&mut rng) as usize) % ctx.nb_voxels;
    if compact_idx + 1 >= ctx.global.offsets.len() {
        return AttemptOutcome::SkipEmpty;
    }
    if ctx.global.offsets[compact_idx + 1] == ctx.global.offsets[compact_idx] {
        return AttemptOutcome::SkipEmpty;
    }
    if let (Some(mask), true) = (&plan.seed_mask, ctx.mask_grid_matches) {
        let [i, j, k] = ctx.ijk_lookup[compact_idx];
        let idx = (i as usize)
            + (mask.dims[0] as usize) * ((j as usize) + (mask.dims[1] as usize) * (k as usize));
        if mask.data.get(idx).copied().unwrap_or(0) == 0 {
            return AttemptOutcome::SkipEmpty;
        }
    }
    let seed_ras = ctx.centers_ras[compact_idx];

    // DSI-Studio autotrack sentinels: `≤ 0` (or `≥ 1` for smooth) means
    // "randomize per attempt." We preserve those exactly because the
    // benches baseline against this sequence. RNG consumption is
    // deterministic per attempt_idx, so skip-paths still burn the same
    // `lcg_f32` calls to stay bit-identical.
    let step_mm = if plan.step_size_mm <= 0.0 {
        // DSI-Studio `tracking_thread.cpp`: `step = vs · (0.5 + rand())`.
        ctx.vox_to_ras_smallest_vs * (0.5 + lcg_f32(&mut rng))
    } else {
        let _ = lcg_f32(&mut rng);
        plan.step_size_mm
    };
    let angle_rad = if plan.max_angle_deg <= 0.0 {
        let t = lcg_f32(&mut rng);
        (15.0 + 75.0 * t).to_radians()
    } else {
        let _ = lcg_f32(&mut rng);
        plan.max_angle_deg.to_radians()
    };
    let cos_max = angle_rad.cos();
    let fixel_threshold = if plan.fixel_threshold <= 0.0 {
        // DSI-Studio `tracking_thread.cpp:215–221`: jitter in
        // `[(default_otsu - 0.1) * fixel_otsu, (default_otsu + 0.1) * fixel_otsu]`
        // with `default_otsu = 0.6`. Legacy fallback base 0.1 preserves
        // pre-Otsu behavior for plans without a `fixel_otsu`.
        let base = plan.fixel_otsu.unwrap_or(0.1);
        let jitter = lcg_f32(&mut rng) - 0.5;
        (base * (0.6 + jitter * 0.2)).clamp(0.0, 1.0)
    } else {
        let _ = lcg_f32(&mut rng);
        plan.fixel_threshold
    };
    let smooth = if plan.smooth_fraction >= 1.0 {
        lcg_f32(&mut rng) * 0.95
    } else {
        let _ = lcg_f32(&mut rng);
        plan.smooth_fraction.clamp(0.0, 0.95)
    };

    // DSI-Studio seed jitter: uniform in [-0.5, +0.5] voxels, applied in
    // voxel space then transformed back to RAS mm.
    let jitter_vox = Vec3::new(
        lcg_f32(&mut rng) - 0.5,
        lcg_f32(&mut rng) - 0.5,
        lcg_f32(&mut rng) - 0.5,
    );
    let seed_vox = ctx.global.ras_to_vox.transform_point3(Vec3::from(seed_ras)) + jitter_vox;
    let vox_to_ras = ctx.global.ras_to_vox.inverse();
    let seed_pt = vox_to_ras.transform_point3(seed_vox);

    let dg = YehFixelDG {
        global: ctx.global,
        cos_max,
        step_mm,
        smooth,
        fixel_threshold,
    };

    let masks = PerStepMasks {
        limiting: plan.limiting_mask.as_deref(),
        roa: plan.roa_mask.as_deref(),
        term: plan.term_mask.as_deref(),
    };
    let post_filters = PostFilterSet {
        roi_masks: &plan.roi_masks,
        end_masks: &plan.end_masks,
        no_end_mask: plan.no_end_mask.as_deref(),
        post_filter: plan.post_filter.as_ref(),
    };

    try_one_attempt(
        &dg,
        seed_pt,
        ctx.limits,
        &masks,
        &post_filters,
        &mut rng,
        acc,
    )
}

//! CPU probabilistic tractography from ODX SH data.
//!
//! Thin wrapper around `tracking::dg_prob::DipyProbDG`: load SH +
//! sphere + per-voxel GFA from the scene, build the dense voxel-index
//! LUT, enumerate `seed_mask × seeds_per_voxel` attempts, fire rayon
//! fold+reduce, and assemble the output `StreamlineFlow`. The actual
//! probabilistic direction sampling lives in `tracking::dg_prob`.
//!
//! PTT is declared on `DipyDirectionGetter` so the plan/op/scene-plan
//! API is ready for it, but the CPU port isn't implemented (GPU only).
//! Surfaced here as a hard error; PR 2b's `validate()` method on
//! `WorkflowOp` will turn this into a pre-dispatch diagnostic so the
//! worker thread never even spawns.

use std::sync::Arc;

use rayon::prelude::*;

use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;

use super::tracking::accum::{AttemptOutcome, ThreadAccum};
use super::tracking::dg_prob::{DipyProbDG, DipyProbGlobal, DipyProbScratch};
use super::tracking::masks::{PerStepMasks, PostFilterSet};
use super::tracking::rng::{lcg_f32, split_mix_init};
use super::tracking::tracker::{TrackingLimits, try_one_attempt};
use super::types::{DipyDirectionGetter, DipyTractographyPlan, StreamlineDataset, StreamlineFlow};

pub(super) fn run_cpu_dipy(plan: &DipyTractographyPlan) -> WorkflowResult<StreamlineFlow> {
    match plan.direction_getter {
        DipyDirectionGetter::Probabilistic => {}
        DipyDirectionGetter::Ptt { .. } => {
            return Err(WorkflowError::Evaluation(
                "DipyDirectionGetter::Ptt has no CPU implementation. Run on a GPU-capable \
                 device, or set direction_getter = Probabilistic."
                    .into(),
            ));
        }
    }

    let scene = &plan.odx_scene;

    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation(
            "ODX file has no SH coefficients. Re-derive with odx-rs from a PAM5/CSD model.".into(),
        )
    })?;
    let ncoeffs = sh_view.ncols();
    let nb_voxels = sh_view.nrows();

    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for tractography.".into())
    })?;
    let sphere_verts = mesh.vertices();
    let n_dirs = sphere_verts.len();
    let sample_plan = mesh.sample_plan();

    // Flatten SH: (nb_voxels, ncoeffs) row-major.
    let sh_flat: Vec<f32> = (0..nb_voxels)
        .flat_map(|i| sh_view.row(i).iter().copied())
        .collect();

    let dims = scene.dimensions();
    let [nx, ny, nz] = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
    let ijk_lookup = scene.ijk_lookup();

    let mut dense_lut = vec![usize::MAX; nx * ny * nz];
    for (compact_idx, &[ix, iy, iz]) in ijk_lookup.iter().enumerate() {
        dense_lut[ix as usize * ny * nz + iy as usize * nz + iz as usize] = compact_idx;
    }

    // Per-voxel GFA as the stopping metric. Placeholder uniform 1.0
    // matches the pre-refactor behavior; the TODO to compute real GFA
    // from the ODF view stays as-is — it's independent of this refactor.
    let gfa_data: Vec<f32> = {
        let dpv = scene.odf_view_f32().and_then(|_odf| None::<Vec<f32>>);
        dpv.unwrap_or_else(|| vec![1.0f32; nb_voxels])
    };

    let vox_to_ras = scene.voxel_to_ras();
    let ras_to_vox = vox_to_ras.inverse();

    let cos_max = plan.max_angle_deg.to_radians().cos();

    // Effective fixel threshold. Prob is deterministic (no per-seed
    // randomization), so `fixel_threshold ≤ 0` falls back to
    // `0.6 · fixel_otsu` when the plan carries one, else 0.0 (accept
    // all) to preserve legacy behavior.
    let effective_fixel_threshold = if plan.fixel_threshold <= 0.0 {
        plan.fixel_otsu.map(|v| v * 0.6).unwrap_or(0.0)
    } else {
        plan.fixel_threshold
    };
    let step_mm = plan.step_size_mm;
    let min_pts = ((plan.min_len_mm / step_mm.max(0.01)).ceil() as usize).max(2);
    let max_pts = plan.max_points as usize;

    // Seed points: wired mask if present, else whole-brain (every
    // voxel in the ODX compact mask). Matches Yeh's whole-brain default
    // when no mask is wired.
    let seeds_ras: Vec<[f32; 3]> = match plan.seed_mask.as_deref() {
        Some(mask) => mask.nonzero_voxel_centers_ras(),
        None => scene.centers_ras().to_vec(),
    };
    let n_seeds = seeds_ras.len();
    let n_rep = plan.seeds_per_voxel as usize;
    let n_attempts = n_seeds * n_rep;

    let t0 = std::time::Instant::now();
    log::info!(
        "[dipy] '{}': {} seeds × {} reps, {} sphere dirs, {} SH coeffs",
        plan.label,
        n_seeds,
        n_rep,
        n_dirs,
        ncoeffs,
    );

    let global = DipyProbGlobal {
        sh_flat: &sh_flat,
        ncoeffs,
        dense_lut: &dense_lut,
        nx,
        ny,
        nz,
        ras_to_vox,
        gfa_data: &gfa_data,
        fixel_threshold: effective_fixel_threshold,
        sample_plan,
        n_dirs,
        sphere_verts,
        relative_peak_threshold: plan.relative_peak_threshold,
    };

    let limits = TrackingLimits {
        // Dipy doesn't have a separate max_len_pts; use max_pts for both.
        max_pts_per_branch: max_pts,
        max_len_pts_per_branch: max_pts,
        min_pts,
    };

    let ctx = DipyAttemptCtx {
        plan,
        global: &global,
        limits,
        seeds_ras: &seeds_ras,
        n_rep,
        cos_max,
        step_mm,
    };

    let merged: ThreadAccum<DipyProbScratch> = (0..n_attempts)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            || new_prob_accum(n_dirs),
            |mut acc, attempt_idx| {
                let outcome = try_dipy_attempt(&ctx, attempt_idx, &mut acc);
                acc.counts.bump(outcome);
                acc
            },
        )
        .reduce(|| new_prob_accum(n_dirs), ThreadAccum::merge);

    let all_positions = merged.positions;
    let all_offsets = merged.offsets;
    let counts = merged.counts;

    let nb_streamlines = all_offsets.len() - 1;
    log::info!(
        "[dipy] '{}': done in {:.1}s — {} streamlines from {} attempts \
         (rejected: no_init={} roa={} min_len={} roi={} end={} no_end={} hausdorff={})",
        plan.label,
        t0.elapsed().as_secs_f32(),
        nb_streamlines,
        n_attempts,
        counts.no_initial,
        counts.roa,
        counts.min_len,
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
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    })
}

/// Pre-size the per-thread PMF scratch to n_dirs so the first
/// `apply_row_into` doesn't trigger a resize. Tiny one-time cost per
/// rayon worker; avoids any allocation inside the hot per-attempt loop.
fn new_prob_accum(n_dirs: usize) -> ThreadAccum<DipyProbScratch> {
    let mut acc = ThreadAccum::<DipyProbScratch>::new();
    acc.dg_scratch.pmf = vec![0.0; n_dirs];
    acc
}

struct DipyAttemptCtx<'a> {
    plan: &'a DipyTractographyPlan,
    global: &'a DipyProbGlobal<'a>,
    limits: TrackingLimits,
    seeds_ras: &'a [[f32; 3]],
    n_rep: usize,
    cos_max: f32,
    step_mm: f32,
}

fn try_dipy_attempt(
    ctx: &DipyAttemptCtx<'_>,
    attempt_idx: usize,
    acc: &mut ThreadAccum<DipyProbScratch>,
) -> AttemptOutcome {
    let plan = ctx.plan;
    let seed_idx = attempt_idx / ctx.n_rep;
    let seed_ras = ctx.seeds_ras[seed_idx];

    let mut rng = split_mix_init(plan.rng_seed, attempt_idx as u64);

    // Small RAS-space jitter around the seed so the `seeds_per_voxel`
    // reps don't all start at the same point. ±¼ step_mm keeps the
    // jitter sub-voxel for any reasonable step size.
    let jitter = [
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
    ];
    let seed_pt = glam::Vec3::new(
        seed_ras[0] + jitter[0],
        seed_ras[1] + jitter[1],
        seed_ras[2] + jitter[2],
    );

    let dg = DipyProbDG {
        global: ctx.global,
        cos_max: ctx.cos_max,
        step_mm: ctx.step_mm,
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

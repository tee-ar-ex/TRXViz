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
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Outcome of a single seed attempt. No streamline payload — when an
/// attempt is kept, `try_one_attempt` has already appended the points
/// directly into the caller's thread-local accumulator. This enum is now
/// just a tiny tag (one byte at runtime), so passing it around in a hot
/// loop is essentially free.
#[derive(Clone, Copy)]
enum AttemptOutcome {
    Kept,
    SkipEmpty,         // picked voxel has no fixels or is outside seed mask
    InitialPeakFailed, // no viable initial direction above fixel threshold
    RejectRoa,         // per-step ROA hit
    RejectMinLen,
    RejectRoi,
    RejectEnd,
    RejectNoEnd,
    RejectHausdorff,
}

/// Per-thread scratch + output buffers. Each rayon worker keeps one of
/// these alive across all the attempts it processes; reusing the buffers
/// (via `clear()`) keeps their underlying allocations and means we pay the
/// `Vec::with_capacity` cost ~once per thread instead of ~once per attempt.
///
/// `positions` and `offsets` are the local share of the final output: at
/// the end of the parallel section we `reduce` (merge) all per-thread
/// `ThreadAccum`s into a single output by appending positions and
/// rebasing offsets.
struct ThreadAccum {
    /// Concatenated streamline points for streamlines this thread kept.
    positions: Vec<[f32; 3]>,
    /// Offsets into `positions`. Always starts with `0`; pushes one
    /// entry per kept streamline equal to the new `positions.len()`.
    /// (After merge, offsets are rebased relative to the global vector.)
    offsets: Vec<u32>,
    /// Reusable scratch buffer for `track_one`'s forward branch.
    fwd_scratch: Vec<[f32; 3]>,
    /// Reusable scratch buffer for `track_one`'s backward branch.
    bwd_scratch: Vec<[f32; 3]>,
    /// Reusable scratch buffer for `pick_initial_peak` candidate list.
    peak_scratch: Vec<Vec3>,
    /// Per-thread rejection counters (summed during reduce).
    counts: RejectionCounts,
}

#[derive(Default, Clone, Copy)]
struct RejectionCounts {
    skip: usize,
    initial: usize,
    roa: usize,
    min_len: usize,
    roi: usize,
    end: usize,
    no_end: usize,
    hausdorff: usize,
    kept: usize,
}

impl ThreadAccum {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            // `offsets` mirrors the TRX/streamline-set convention: N
            // streamlines → N+1 offsets, where `offsets[i]..offsets[i+1]`
            // is the slice of points for streamline i. The leading 0 is
            // the start of streamline 0.
            offsets: vec![0u32],
            fwd_scratch: Vec::new(),
            bwd_scratch: Vec::new(),
            peak_scratch: Vec::new(),
            counts: RejectionCounts::default(),
        }
    }

    /// Merge `other` into `self`. Used as the `reduce` step after the
    /// parallel `fold`.  We append `other`'s positions then translate
    /// (rebase) its offsets by the byte... err, the *element* count we
    /// had before the append. This is O(streamlines) per merge, not
    /// O(positions), so it's cheap.
    fn merge(mut self, other: ThreadAccum) -> ThreadAccum {
        let base = self.positions.len() as u32;
        self.positions.extend(other.positions);
        // Skip other.offsets[0] (always 0) — our last offset already
        // marks the boundary between the two thread-locals' streamlines.
        for off in other.offsets.into_iter().skip(1) {
            self.offsets.push(base + off);
        }
        self.counts.skip += other.counts.skip;
        self.counts.initial += other.counts.initial;
        self.counts.roa += other.counts.roa;
        self.counts.min_len += other.counts.min_len;
        self.counts.roi += other.counts.roi;
        self.counts.end += other.counts.end;
        self.counts.no_end += other.counts.no_end;
        self.counts.hausdorff += other.counts.hausdorff;
        self.counts.kept += other.counts.kept;
        self
    }
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

    // ── Parallel section: fold + reduce ─────────────────────────────────
    //
    // Strategy: fire one rayon par-iter over the whole `0..attempt_budget`
    // range (no wave-loop, no per-wave `.collect()` barrier). Each rayon
    // worker keeps its own `ThreadAccum` alive across the chunk it owns,
    // appending kept streamlines and reusing scratch buffers. After the
    // parallel section we `reduce` (merge) the per-thread accumulators
    // into a single output.
    //
    // Why this is faster than the prior wave loop:
    //   1. No barrier between waves — workers stay busy until the global
    //      target is hit.
    //   2. No per-streamline `Vec` allocation (was: `track_one` returned
    //      a fresh Vec; outer loop returned an `AttemptOutcome::Kept(Vec)`).
    //      Streamlines are now appended directly into the worker's local
    //      `positions` buffer — at most ~1 reallocation per worker.
    //   3. `with_min_len(64)` keeps rayon from over-splitting cheap
    //      attempts. Without it, rayon's adaptive splitter can hand out
    //      tasks small enough that a `SkipEmpty` rejection (a few ns of
    //      work) is dominated by scheduling overhead. 64 was picked as a
    //      conservative "even worst-case-cheap items batch into ~µs of
    //      work" floor — there's headroom to tune higher (256, 1024) once
    //      benchmarks land.
    //
    // Determinism note: each individual streamline at attempt index `i`
    // is bit-identical to before — RNG is derived from `(rng_seed, i)`.
    // What changes: the *order* of kept streamlines in the output may
    // shift between runs because thread scheduling determines which
    // worker's keeps land first during reduce. If you need stable order,
    // sort by attempt_idx (not done here — would cost an extra `u64` per
    // kept streamline).
    let target_atomic = AtomicUsize::new(0);

    // `into_par_iter()` on a `Range` returns an indexed parallel iterator.
    // Rayon recursively splits the range; `with_min_len` says "don't split
    // any branch below 64 items".
    // Note: iterate as `usize` (not `u64`) — only `usize`/`u32`/`i32`/etc.
    // ranges implement `IndexedParallelIterator` in rayon. `attempt_budget`
    // is already a `usize`, so this is the natural type. We convert to
    // `u64` inside the closure where `try_one_attempt` wants it.
    let merged: ThreadAccum = (0..attempt_budget)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            // `fold` takes TWO closures:
            //   (1) the identity / init closure — called by each worker
            //       to create its starting accumulator. Called O(workers)
            //       times, not O(items). This is where `ThreadAccum::new`
            //       runs (cheap — empty Vecs).
            ThreadAccum::new,
            //   (2) the per-item closure — folds one attempt index into
            //       the worker-local accumulator. We pass `&mut acc` in,
            //       then return `acc` (rayon's `fold` is move-based, so
            //       you must return the accumulator each step).
            |mut acc, idx| {
                // Early-termination check. `Relaxed` is the cheapest
                // atomic ordering — we don't need a happens-before
                // relationship with anything; we just need eventual
                // visibility. Worst case: a few extra attempts run after
                // target is hit. They're harmless — we truncate at the
                // end. Reading an atomic is ~1 ns on modern x86/ARM.
                if target_atomic.load(Ordering::Relaxed) >= target {
                    return acc;
                }
                let outcome = try_one_attempt(&ctx, idx as u64, &mut acc);
                match outcome {
                    AttemptOutcome::Kept => {
                        acc.counts.kept += 1;
                        target_atomic.fetch_add(1, Ordering::Relaxed);
                    }
                    AttemptOutcome::SkipEmpty => acc.counts.skip += 1,
                    AttemptOutcome::InitialPeakFailed => acc.counts.initial += 1,
                    AttemptOutcome::RejectRoa => acc.counts.roa += 1,
                    AttemptOutcome::RejectMinLen => acc.counts.min_len += 1,
                    AttemptOutcome::RejectRoi => acc.counts.roi += 1,
                    AttemptOutcome::RejectEnd => acc.counts.end += 1,
                    AttemptOutcome::RejectNoEnd => acc.counts.no_end += 1,
                    AttemptOutcome::RejectHausdorff => acc.counts.hausdorff += 1,
                }
                acc
            },
        )
        // `fold` produces a `ParallelIterator` of partial accumulators
        // (one per rayon split). `reduce` then combines them pairwise in
        // a tree, also in parallel. `ThreadAccum::merge` does the offset
        // rebasing and counter sum.
        .reduce(ThreadAccum::new, ThreadAccum::merge);

    let mut all_positions = merged.positions;
    let mut all_offsets = merged.offsets;
    let counts = merged.counts;

    // Trim any overshoot. Because the atomic check is racy by design,
    // multiple workers may have crossed the target boundary in flight —
    // we may have e.g. target+5 kept streamlines. Truncate to exactly
    // `target` (or leave alone if we underran the budget).
    if (all_offsets.len() - 1) > target {
        // offsets has N+1 entries for N streamlines; cap at target+1.
        let cutoff = all_offsets[target] as usize;
        all_positions.truncate(cutoff);
        all_offsets.truncate(target + 1);
    }

    // Attempts actually consumed: workers stop pulling fresh `idx` values
    // after target_atomic >= target, but they finish whatever idx they
    // had loaded. Reporting "we touched at least this many" is honest.
    // The atomic only counts kept streamlines, so we approximate attempts
    // as the sum of all rejection counters + kept.
    let attempts = counts.skip
        + counts.initial
        + counts.roa
        + counts.min_len
        + counts.roi
        + counts.end
        + counts.no_end
        + counts.hausdorff
        + counts.kept;

    let nb_streamlines = all_offsets.len() - 1;
    log::info!(
        "[yeh] '{}': done in {:.1}s — {} streamlines from {} attempts \
         (rejected: skip={} init={} min_len={} roa/term={} roi={} end={} no_end={} hausdorff={})",
        plan.label,
        t0.elapsed().as_secs_f32(),
        nb_streamlines,
        attempts,
        counts.skip,
        counts.initial,
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

/// One deterministic seed-attempt. Builds its own RNG from
/// `(plan.rng_seed, attempt_idx)` so the *content* of every attempt is
/// reproducible across CPU counts and parallelism decisions — only the
/// *order* of kept streamlines in the output may shift between runs.
///
/// On a `Kept` outcome, the streamline points have already been pushed
/// onto `acc.positions` and a new boundary appended to `acc.offsets`.
/// The caller does not need to do anything else.
fn try_one_attempt(
    ctx: &YehCtx<'_>,
    attempt_idx: u64,
    acc: &mut ThreadAccum,
) -> AttemptOutcome {
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
        &mut acc.peak_scratch,
    ) else {
        return AttemptOutcome::InitialPeakFailed;
    };
    let sign = if lcg_f32(&mut rng) < 0.5 { 1.0 } else { -1.0 };
    let init_dir = init_peak * sign;

    // Reuse the per-thread scratch buffers. `clear()` keeps the underlying
    // capacity, so after the first ~few attempts these allocations are a
    // no-op — this is most of the per-streamline allocation savings.
    acc.fwd_scratch.clear();
    acc.bwd_scratch.clear();

    if track_one(
        &mut acc.fwd_scratch,
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
    )
    .is_err()
    {
        return AttemptOutcome::RejectRoa;
    }
    if track_one(
        &mut acc.bwd_scratch,
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
    )
    .is_err()
    {
        return AttemptOutcome::RejectRoa;
    }

    // Pre-flight length check (cheap) so we don't pay for the post-hoc
    // mask filters on streamlines that won't survive the length floor.
    let streamline_len = acc.bwd_scratch.len() + 1 + acc.fwd_scratch.len();
    if streamline_len < ctx.min_pts {
        return AttemptOutcome::RejectMinLen;
    }

    // Build the assembled streamline directly into the accumulator's
    // `positions` vector. We append in [reversed backward, seed, forward]
    // order, then commit by pushing the new offset. If a post-hoc filter
    // rejects, we truncate `positions` back to the saved length — cheaper
    // than the prior pattern of building a temporary `Vec` and only later
    // copying it into the output.
    let commit_start = acc.positions.len();
    acc.positions.extend(acc.bwd_scratch.iter().rev().copied());
    acc.positions.push(seed_pt.to_array());
    acc.positions.extend(acc.fwd_scratch.iter().copied());

    // Each filter borrows the freshly-appended slice immutably, runs the
    // check, and on rejection rolls back `positions` via `truncate` (cheap:
    // truncate just resets the length, leaving the capacity allocated for
    // the next attempt). We can't keep a `let streamline = &acc.positions[..]`
    // binding alive across the truncate because that would conflict with
    // the &mut borrow — so we re-slice inline at each filter.

    if !plan.roi_masks.is_empty()
        && !streamline_hits_all_rois(&acc.positions[commit_start..], &plan.roi_masks)
    {
        acc.positions.truncate(commit_start);
        return AttemptOutcome::RejectRoi;
    }
    if let Some(ne) = plan.no_end_mask.as_deref() {
        if streamline_endpoint_in(&acc.positions[commit_start..], ne) {
            acc.positions.truncate(commit_start);
            return AttemptOutcome::RejectNoEnd;
        }
    }
    if !plan.end_masks.is_empty()
        && !streamline_satisfies_end_masks(&acc.positions[commit_start..], &plan.end_masks)
    {
        acc.positions.truncate(commit_start);
        return AttemptOutcome::RejectEnd;
    }
    if let Some(PostFilter::Hausdorff {
        reference_points_ras,
        max_mm,
    }) = plan.post_filter.as_ref()
    {
        if !streamline_passes_hausdorff(
            &acc.positions[commit_start..],
            reference_points_ras,
            *max_mm,
        ) {
            acc.positions.truncate(commit_start);
            return AttemptOutcome::RejectHausdorff;
        }
    }

    // Survived all filters — commit by recording the new boundary in
    // `offsets`. The streamline points are already in `positions`.
    acc.offsets.push(acc.positions.len() as u32);
    AttemptOutcome::Kept
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Track one branch entirely in RAS+mm. Appends points to `out` as it goes.
/// Returns `Ok(())` on a clean termination (including clean limiting/term-mask
/// exits), or `Err(())` when the streamline entered a `roa_mask` — the whole
/// streamline should be discarded in that case.
///
/// The caller supplies `out`. We do NOT allocate or clear here; the caller is
/// responsible for `clear()`-ing the scratch buffer between streamlines (this
/// preserves the underlying capacity, so the second-and-later allocations are
/// near-free — a major win when running tens of thousands of attempts).
#[allow(clippy::too_many_arguments)]
fn track_one(
    out: &mut Vec<[f32; 3]>,
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
) -> Result<(), ()> {
    if dir.length_squared() < 1e-8 {
        return Ok(());
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
                return Err(());
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

    Ok(())
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

/// Pick a random initial peak among the seed voxel's fixels that exceed
/// `fixel_threshold`. Uses a caller-supplied scratch buffer so the
/// candidate list is allocated once per thread, not once per attempt.
#[allow(clippy::too_many_arguments)]
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
    candidates: &mut Vec<Vec3>,
) -> Option<Vec3> {
    let compact_idx = voxel_at(seed_vox, dense_lut, nx, ny, nz)?;
    if compact_idx + 1 >= offsets.len() {
        return None;
    }
    let start = offsets[compact_idx] as usize;
    let end = offsets[compact_idx + 1] as usize;
    candidates.clear();
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

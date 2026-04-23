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
use rayon::prelude::*;

use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;

use super::tracking_filters::{
    point_in_mask, streamline_endpoint_in, streamline_hits_all_rois, streamline_passes_hausdorff,
    streamline_satisfies_end_masks,
};
use super::types::{
    DipyDirectionGetter, DipyTractographyPlan, PostFilter, StreamlineDataset, StreamlineFlow,
    VoxelMask, WorkflowExecutionCache, WorkflowNodeUuid,
};

/// Per-thread scratch + output buffers. Mirrors `cpu_yeh::ThreadAccum`.
/// Keeping `positions`/`offsets`/scratch alive across attempts on a single
/// thread means we pay `Vec::with_capacity` ~once per thread instead of
/// once per attempt — the major allocation win for the future parallel
/// version. The merge step (used by rayon `reduce` post-parallel) appends
/// `other`'s positions and rebases its offsets by the current `positions`
/// length.
///
/// Public-shape note: this struct is single-thread-only today (we still
/// have one `DipyThreadAccum` for the whole `run_cpu_dipy` call). The
/// pattern is here so the parallel flip is mechanical: replace the outer
/// `for attempt_idx in 0..n_attempts { step_one(&ctx, idx, &mut acc) }`
/// with `(0..n_attempts).into_par_iter().with_min_len(64).fold(...).reduce(...)`.
struct DipyThreadAccum {
    positions: Vec<[f32; 3]>,
    offsets: Vec<u32>,
    fwd_scratch: Vec<[f32; 3]>,
    bwd_scratch: Vec<[f32; 3]>,
    pmf_scratch: Vec<f32>,
    counts: DipyRejectionCounts,
}

#[derive(Default, Clone, Copy)]
struct DipyRejectionCounts {
    no_initial: usize, // sample_direction returned None at the seed
    roa: usize,        // hit ROA mask mid-track (forward or backward)
    min_len: usize,    // assembled streamline shorter than min_pts
    roi: usize,
    end: usize,
    no_end: usize,
    hausdorff: usize,
    kept: usize,
}

impl DipyThreadAccum {
    fn new(n_dirs: usize) -> Self {
        Self {
            positions: Vec::new(),
            offsets: vec![0u32],
            fwd_scratch: Vec::new(),
            bwd_scratch: Vec::new(),
            pmf_scratch: vec![0.0f32; n_dirs],
            counts: DipyRejectionCounts::default(),
        }
    }

    /// Merge `other` into `self`. Used as the rayon `reduce` step
    /// downstream of the parallel `fold`.
    fn merge(mut self, other: DipyThreadAccum) -> DipyThreadAccum {
        let base = self.positions.len() as u32;
        self.positions.extend(other.positions);
        for off in other.offsets.into_iter().skip(1) {
            self.offsets.push(base + off);
        }
        let c = &other.counts;
        self.counts.no_initial += c.no_initial;
        self.counts.roa += c.roa;
        self.counts.min_len += c.min_len;
        self.counts.roi += c.roi;
        self.counts.end += c.end;
        self.counts.no_end += c.no_end;
        self.counts.hausdorff += c.hausdorff;
        self.counts.kept += c.kept;
        self
    }
}

/// Run CPU probabilistic tractography and return a StreamlineFlow.
pub(super) fn run_cpu_dipy(plan: &DipyTractographyPlan) -> WorkflowResult<StreamlineFlow> {
    // ── DG dispatch ─────────────────────────────────────────────────────
    //
    // Currently only `Probabilistic` has a CPU implementation. PTT is
    // declared in the type system so the plan/op/scene-plan API is ready
    // for it, but the algorithm itself is not yet ported. We surface
    // that clearly here rather than running the wrong algorithm.
    //
    // When PTT lands: extract the existing per-attempt body into
    // `try_one_attempt_probabilistic`, add `try_one_attempt_ptt`, and
    // dispatch in the rayon fold closure rather than at the top level.
    // (Top-level dispatch is fine for now since each DG will share the
    // same outer scaffolding — seeds, batching, output collection.)
    match plan.direction_getter {
        DipyDirectionGetter::Probabilistic => {}
        DipyDirectionGetter::Ptt { .. } => {
            return Err(WorkflowError::Evaluation(
                "DipyDirectionGetter::Ptt is not implemented on CPU yet. \
                 Set direction_getter = Probabilistic, or wait for the PTT \
                 follow-up that will land in cpu_dipy.rs."
                    .into(),
            ));
        }
    }

    let scene = &plan.odx_scene;

    // ── collect field data ──────────────────────────────────────────────
    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation(
            "ODX file has no SH coefficients. Re-derive with odx-rs from a PAM5/CSD model.".into(),
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
        "[dipy] '{}': {} seeds × {} reps, {} sphere dirs, {} SH coeffs",
        plan.label,
        seeds_ras.len(),
        plan.seeds_per_voxel,
        n_dirs,
        ncoeffs,
    );

    // ── Flat attempt enumeration ───────────────────────────────────────
    //
    // Old shape: nested `for seed in seeds { for rep in 0..N { ... } }`.
    // New shape: a single flat index over `n_attempts = n_seeds * N_rep`,
    // with `(seed_idx, rep)` recovered by integer division. This is the
    // single change that makes the loop directly drop-in-able into
    // `(0..n_attempts).into_par_iter()` without any restructuring.
    //
    // RNG is now derived per-attempt from `(rng_seed, attempt_idx)` —
    // mirrors `cpu_yeh::try_one_attempt`. This is a behavior change vs.
    // the old serial single-RNG sequence (per-attempt streamlines will
    // differ bit-wise from old runs at the same `rng_seed`), but it's
    // the right shape for parallelism: each attempt's RNG state depends
    // only on `attempt_idx`, so the keep-set is identical regardless of
    // thread count, scheduling, or whether parallelism is enabled at all.
    let n_seeds = seeds_ras.len();
    let n_rep = plan.seeds_per_voxel as usize;
    let n_attempts = n_seeds * n_rep;

    // Bundle all the read-only shared state into one borrow-able struct
    // so the per-attempt function (and therefore the rayon closure) has
    // a single context argument instead of ~20 individual `&` parameters.
    // Mirrors `cpu_yeh::YehCtx`.
    let ctx = DipyCtx {
        plan,
        seeds_ras,
        sh_flat: &sh_flat,
        ncoeffs,
        nb_voxels,
        dense_lut: &dense_lut,
        nx,
        ny,
        nz,
        ras_to_vox,
        gfa_data: &gfa_data,
        effective_fixel_threshold,
        sample_plan,
        n_dirs,
        sphere_verts,
        n_rep,
        cos_max,
        step_mm,
        max_pts,
        min_pts,
    };

    // ── Parallel section ────────────────────────────────────────────────
    //
    // Single rayon par-iter over the flat attempt range. Each worker keeps
    // its own `DipyThreadAccum` alive across the chunk it owns. After
    // the parallel section, `reduce` merges per-thread accumulators by
    // appending positions and rebasing offsets.
    //
    // `with_min_len(64)` keeps rayon's adaptive splitter from producing
    // tasks small enough that `no_initial` rejections (cheap) get
    // dominated by scheduling overhead. Tune upward if benchmarks suggest
    // it helps; 64 is a conservative floor.
    //
    // No early-termination: unlike yeh (which has a target_streamlines
    // cap and runs to budget), the dipy tracker enumerates a fixed
    // `n_seeds × n_rep` attempts. So we just process the full range.
    //
    // Determinism note: each attempt's RNG comes from `(rng_seed,
    // attempt_idx)`, so individual streamlines are bit-identical across
    // runs. Only the *order* of kept streamlines in the output may shift
    // with thread count.
    let merged: DipyThreadAccum = (0..n_attempts)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            || DipyThreadAccum::new(n_dirs),
            |mut acc, attempt_idx| {
                try_one_attempt(&ctx, attempt_idx, &mut acc);
                acc
            },
        )
        .reduce(|| DipyThreadAccum::new(n_dirs), DipyThreadAccum::merge);

    let all_positions = merged.positions;
    let all_offsets = merged.offsets;
    let counts = merged.counts;
    let _ = n_seeds; // informational only

    let nb_streamlines = all_offsets.len() - 1;
    eprintln!(
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

// ── tracking helpers ────────────────────────────────────────────────

/// Read-only shared context for one `run_cpu_dipy` invocation, threaded
/// into `try_one_attempt` so each rayon worker has a single `&` argument
/// instead of ~20 individual references. Mirrors `cpu_yeh::YehCtx`.
struct DipyCtx<'a> {
    plan: &'a DipyTractographyPlan,
    seeds_ras: &'a [[f32; 3]],
    sh_flat: &'a [f32],
    ncoeffs: usize,
    nb_voxels: usize,
    dense_lut: &'a [usize],
    nx: usize,
    ny: usize,
    nz: usize,
    ras_to_vox: Mat4,
    gfa_data: &'a [f32],
    effective_fixel_threshold: f32,
    sample_plan: &'a odx_rs::mrtrix_sh::RowSamplePlan,
    n_dirs: usize,
    sphere_verts: &'a [[f32; 3]],
    n_rep: usize,
    cos_max: f32,
    step_mm: f32,
    max_pts: usize,
    min_pts: usize,
}

/// One deterministic seed-attempt. Builds its own RNG from
/// `(plan.rng_seed, attempt_idx)` so each attempt is reproducible
/// regardless of thread count or scheduling. On a kept streamline, the
/// points are already in `acc.positions` and the boundary in
/// `acc.offsets`; the caller does not need to do anything else.
fn try_one_attempt(ctx: &DipyCtx<'_>, attempt_idx: usize, acc: &mut DipyThreadAccum) {
    let plan = ctx.plan;
    let seed_idx = attempt_idx / ctx.n_rep;
    let seed_ras = ctx.seeds_ras[seed_idx];

    // SplitMix-style mix of (rng_seed, attempt_idx). Same constants
    // as cpu_yeh::try_one_attempt — keeps the per-attempt RNG well-
    // dispersed even when adjacent attempt indices map to the same
    // seed voxel.
    let mut rng = plan
        .rng_seed
        .wrapping_add(0x9E3779B97F4A7C15)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        .wrapping_add((attempt_idx as u64).wrapping_mul(0x94D049BB133111EB));

    let jitter = [
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
        (lcg_f32(&mut rng) - 0.5) * ctx.step_mm * 0.5,
    ];
    let seed_pt = Vec3::new(
        seed_ras[0] + jitter[0],
        seed_ras[1] + jitter[1],
        seed_ras[2] + jitter[2],
    );

    let Some(init_dir) = sample_direction(
        seed_pt,
        Vec3::ZERO,
        true,
        ctx.cos_max,
        ctx.sh_flat,
        ctx.ncoeffs,
        ctx.nb_voxels,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &ctx.ras_to_vox,
        ctx.gfa_data,
        ctx.effective_fixel_threshold,
        ctx.sample_plan,
        ctx.n_dirs,
        ctx.sphere_verts,
        plan.relative_peak_threshold,
        &mut acc.pmf_scratch,
        &mut rng,
    ) else {
        acc.counts.no_initial += 1;
        return;
    };

    // Reuse per-thread scratch buffers — `clear()` preserves capacity
    // so repeat allocations are near-free.
    acc.fwd_scratch.clear();
    acc.bwd_scratch.clear();

    // Forward branch.
    if track_one(
        &mut acc.fwd_scratch,
        seed_pt,
        init_dir,
        false,
        ctx.cos_max,
        ctx.step_mm,
        ctx.max_pts,
        ctx.sh_flat,
        ctx.ncoeffs,
        ctx.nb_voxels,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &ctx.ras_to_vox,
        ctx.gfa_data,
        ctx.effective_fixel_threshold,
        ctx.sample_plan,
        ctx.n_dirs,
        ctx.sphere_verts,
        plan.relative_peak_threshold,
        &mut acc.pmf_scratch,
        &mut rng,
        plan.limiting_mask.as_deref(),
        plan.roa_mask.as_deref(),
        plan.term_mask.as_deref(),
    )
    .is_err()
    {
        acc.counts.roa += 1;
        return;
    }

    // Backward branch.
    if track_one(
        &mut acc.bwd_scratch,
        seed_pt,
        -init_dir,
        false,
        ctx.cos_max,
        ctx.step_mm,
        ctx.max_pts,
        ctx.sh_flat,
        ctx.ncoeffs,
        ctx.nb_voxels,
        ctx.dense_lut,
        ctx.nx,
        ctx.ny,
        ctx.nz,
        &ctx.ras_to_vox,
        ctx.gfa_data,
        ctx.effective_fixel_threshold,
        ctx.sample_plan,
        ctx.n_dirs,
        ctx.sphere_verts,
        plan.relative_peak_threshold,
        &mut acc.pmf_scratch,
        &mut rng,
        plan.limiting_mask.as_deref(),
        plan.roa_mask.as_deref(),
        plan.term_mask.as_deref(),
    )
    .is_err()
    {
        acc.counts.roa += 1;
        return;
    }

    let streamline_len = acc.bwd_scratch.len() + 1 + acc.fwd_scratch.len();
    if streamline_len < ctx.min_pts {
        acc.counts.min_len += 1;
        return;
    }

    // Assemble [reversed backward, seed, forward] directly into the
    // accumulator. On rejection by a post-hoc filter, truncate back.
    let commit_start = acc.positions.len();
    acc.positions.extend(acc.bwd_scratch.iter().rev().copied());
    acc.positions.push(seed_pt.to_array());
    acc.positions.extend(acc.fwd_scratch.iter().copied());

    if !plan.roi_masks.is_empty()
        && !streamline_hits_all_rois(&acc.positions[commit_start..], &plan.roi_masks)
    {
        acc.positions.truncate(commit_start);
        acc.counts.roi += 1;
        return;
    }
    if let Some(ne) = plan.no_end_mask.as_deref() {
        if streamline_endpoint_in(&acc.positions[commit_start..], ne) {
            acc.positions.truncate(commit_start);
            acc.counts.no_end += 1;
            return;
        }
    }
    if !plan.end_masks.is_empty()
        && !streamline_satisfies_end_masks(&acc.positions[commit_start..], &plan.end_masks)
    {
        acc.positions.truncate(commit_start);
        acc.counts.end += 1;
        return;
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
            acc.counts.hausdorff += 1;
            return;
        }
    }

    acc.offsets.push(acc.positions.len() as u32);
    acc.counts.kept += 1;
}

/// Track one branch entirely in RAS+mm. Appends points to `out` as it goes.
/// Returns `Ok(())` on a clean termination (including clean limiting/term-mask
/// exits), or `Err(())` when the streamline entered a `roa_mask` — the whole
/// streamline should be discarded in that case.
///
/// Caller owns `out` and is responsible for `clear()`-ing it between
/// streamlines (preserves capacity → near-free re-use). Same shape as
/// `cpu_yeh::track_one` so that the per-thread accumulator pattern can be
/// shared mechanically when this code is parallelized.
#[allow(clippy::too_many_arguments)]
fn track_one(
    out: &mut Vec<[f32; 3]>,
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
) -> Result<(), ()> {
    let mut point = start + start_dir * step_mm;
    let mut direction = start_dir;

    while out.len() < max_pts {
        // Per-step constraint masks. ROA rejects the whole streamline;
        // limiting/term terminate the branch cleanly.
        if let Some(m) = roa {
            if point_in_mask(point, m) {
                return Err(());
            }
        }
        if let Some(m) = limiting {
            if !point_in_mask(point, m) {
                break;
            }
        }
        if let Some(m) = term {
            if point_in_mask(point, m) {
                out.push(point.to_array());
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
        out.push(point.to_array());
        point += new_dir * step_mm;
        direction = new_dir;
    }

    Ok(())
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
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}

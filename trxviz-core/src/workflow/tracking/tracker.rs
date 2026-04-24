//! The shared outer tracker: runs the bidirectional branches, assembles
//! the streamline, applies post-hoc filters, and commits to the
//! accumulator. Generic over the `DirectionGetter` so Yeh and Dipy share
//! this machinery — only the direction-picking is different.

use glam::Vec3;

use super::accum::{AttemptOutcome, ThreadAccum};
use super::direction_getter::DirectionGetter;
use super::masks::{PerStepMasks, PostFilterSet, StepMaskDecision};

/// Length caps for one attempt. `max_pts_per_branch` is the tracker's
/// hard ceiling on points per branch (forward / backward); `max_len_pts`
/// is the length-derived ceiling (`max_len_mm / step_mm`). Both have the
/// same effect in the loop — we cap at `min(max_pts_per_branch, max_len_pts)`.
/// `min_pts` is applied after both branches join.
#[derive(Clone, Copy)]
pub struct TrackingLimits {
    pub max_pts_per_branch: usize,
    pub max_len_pts_per_branch: usize,
    pub min_pts: usize,
}

/// Result of a single branch. The tracker interprets `RoaHit` as an
/// attempt-wide rejection (both branches discarded); `Clean` allows the
/// streamline to be assembled.
enum BranchResult {
    Clean,
    RoaHit,
}

/// Run one seed attempt end-to-end. On `AttemptOutcome::Kept`, the
/// assembled streamline points have been pushed to `acc.positions` and a
/// new boundary pushed to `acc.offsets`. On any rejection outcome,
/// `acc` is logically unchanged (scratch buffers may have grown, but no
/// streamline was committed).
///
/// The RNG state is threaded through so every RNG consumption is
/// reproducible: same `rng_seed` + same `attempt_idx` → same streamline,
/// independent of thread count or scheduling.
pub fn try_one_attempt<D: DirectionGetter>(
    dg: &D,
    seed_ras: Vec3,
    limits: TrackingLimits,
    masks: &PerStepMasks<'_>,
    post_filters: &PostFilterSet<'_>,
    rng: &mut u64,
    acc: &mut ThreadAccum<D::Scratch>,
) -> AttemptOutcome {
    // 1. Pick an initial direction from the seed. None = no viable peak
    //    at this seed (outside mask, below threshold, empty PMF, etc.).
    let Some(init_dir) = dg.initial_direction(seed_ras, rng, &mut acc.dg_scratch) else {
        return AttemptOutcome::NoInitial;
    };

    // 2. Forward + backward branches into per-thread scratch buffers.
    //    Reusing them across attempts is the whole reason we have
    //    ThreadAccum — `.clear()` preserves capacity, so a run of
    //    millions of attempts pays `Vec::with_capacity` ~once per thread.
    acc.fwd_scratch.clear();
    acc.bwd_scratch.clear();

    // Forward branch from the seed in `+init_dir`.
    if matches!(
        track_one_branch(
            dg,
            seed_ras,
            init_dir,
            limits,
            masks,
            rng,
            &mut acc.dg_scratch,
            &mut acc.fwd_scratch,
        ),
        BranchResult::RoaHit
    ) {
        return AttemptOutcome::RejectRoa;
    }

    // Backward branch from the seed in `-init_dir`. Yeh's convention: the
    // two branches share the seed point and extend outward.
    if matches!(
        track_one_branch(
            dg,
            seed_ras,
            -init_dir,
            limits,
            masks,
            rng,
            &mut acc.dg_scratch,
            &mut acc.bwd_scratch,
        ),
        BranchResult::RoaHit
    ) {
        return AttemptOutcome::RejectRoa;
    }

    // 3. Minimum-length pre-flight: each branch's scratch holds post-seed
    //    points; the final streamline is `[reversed bwd, seed, fwd]`.
    let streamline_len = acc.bwd_scratch.len() + 1 + acc.fwd_scratch.len();
    if streamline_len < limits.min_pts {
        return AttemptOutcome::RejectMinLen;
    }

    // 4. Assemble directly into acc.positions. If a post-hoc filter
    //    rejects, `truncate` rolls back — cheaper than building a
    //    temporary Vec. Capacity survives the truncate.
    let commit_start = acc.positions.len();
    acc.positions.extend(acc.bwd_scratch.iter().rev().copied());
    acc.positions.push(seed_ras.to_array());
    acc.positions.extend(acc.fwd_scratch.iter().copied());

    // 5. Post-hoc filters (ROI / end / no_end / hausdorff). Any rejection
    //    rolls back the assembled slice.
    if let Some(outcome) = post_filters.evaluate(&acc.positions[commit_start..]) {
        acc.positions.truncate(commit_start);
        return outcome;
    }

    // 6. Commit: record the boundary for this streamline. Points are
    //    already in `acc.positions`.
    acc.offsets.push(acc.positions.len() as u32);
    AttemptOutcome::Kept
}

/// Track one branch entirely in RAS+mm. Appends points to `out` as it
/// goes. The caller owns `out` and is responsible for `clear`-ing it
/// between branches (preserves capacity → near-free reuse).
///
/// Convention matches the pre-refactor Yeh loop: advance → look up masks
/// → sample next direction → push the post-advance point.
fn track_one_branch<D: DirectionGetter>(
    dg: &D,
    start: Vec3,
    init_dir: Vec3,
    limits: TrackingLimits,
    masks: &PerStepMasks<'_>,
    rng: &mut u64,
    scratch: &mut D::Scratch,
    out: &mut Vec<[f32; 3]>,
) -> BranchResult {
    if init_dir.length_squared() < 1e-8 {
        return BranchResult::Clean;
    }
    let step_mm = dg.step_size_mm();
    let mut dir = init_dir.normalize();
    let mut pt = start;

    let cap = limits.max_pts_per_branch.min(limits.max_len_pts_per_branch);
    for _ in 0..cap {
        pt += dir * step_mm;

        match masks.evaluate_at(pt) {
            StepMaskDecision::Continue => {}
            StepMaskDecision::Terminate => break,
            StepMaskDecision::TerminateAt => {
                out.push(pt.to_array());
                break;
            }
            StepMaskDecision::RejectAll => return BranchResult::RoaHit,
        }

        let Some(new_dir) = dg.next_direction(pt, dir, rng, scratch) else {
            break;
        };
        dir = new_dir;

        out.push(pt.to_array());
    }

    BranchResult::Clean
}

//! The one trait that earns its keep: what distinguishes Yeh from
//! Dipy-probabilistic from PTT at the algorithmic level is *how the next
//! direction is picked*. Everything around that — seeding, the bidirectional
//! assembly, per-step mask checks, post-hoc filters, thread accumulators,
//! RNG seeding — is the same.
//!
//! Implementations:
//!   - `dg_yeh::YehFixelDG` — discrete-peak selection from the ODX fixel
//!     field.
//!   - `dg_prob::DipyProbDG` — trilinear SH interpolation → PMF on the
//!     sphere → weighted sample.
//!   - `dg_ptt::DipyPttDG` — deferred; GPU-only for now (gated at
//!     `DipyTractographyOp::validate`).
//!
//! The trait is intentionally small (three methods, one associated type)
//! so impls stay focused on the algorithmic core. Everything a step needs
//! that isn't algorithm-specific — the current point in RAS+mm, the
//! incoming direction, the per-attempt RNG, and per-thread scratch —
//! flows through parameters.

use glam::Vec3;

/// Signature for "pick a direction from local orientation data."
///
/// `Sync` because the outer tracker hands out `&Self` to each rayon worker
/// inside `try_one_attempt`; impls must be safe to share by shared
/// reference.
///
/// Per-attempt state that isn't the incoming direction (e.g. the random
/// "max-angle" chosen for a Yeh attempt, the step size when randomized)
/// lives on the impl struct itself; the outer code constructs a DG value
/// once per attempt, cheaply.
pub trait DirectionGetter: Sync {
    /// Reusable scratch owned by the per-thread accumulator. Yeh uses this
    /// for the initial-peak candidate list; Dipy uses it for the PMF
    /// evaluated on the sphere. Kept across attempts on the same rayon
    /// worker so per-attempt alloc traffic stays near zero.
    type Scratch: Default + Send;

    /// Pick an initial direction at a seed point. Returns `None` when the
    /// seed voxel has no viable orientation (below threshold, outside the
    /// mask, etc.) — the outer tracker tags the attempt as `NoInitial`.
    ///
    /// Yeh randomly flips the peak's sign so the "forward" and "backward"
    /// branches end up on opposite hemispheres; Dipy's PMF sample already
    /// lives on a full sphere so no flip is needed. Either way, the
    /// caller tracks in `+init_dir` for the forward branch and `-init_dir`
    /// for the backward branch.
    fn initial_direction(
        &self,
        seed_ras: Vec3,
        rng: &mut u64,
        scratch: &mut Self::Scratch,
    ) -> Option<Vec3>;

    /// Pick the next direction given the current point and the previous
    /// step's direction. Returns `None` to terminate this branch cleanly
    /// (e.g. dropped below a threshold, ran out of peaks inside the cone,
    /// PMF went to zero, or left the mask).
    fn next_direction(
        &self,
        pt_ras: Vec3,
        prev_dir: Vec3,
        rng: &mut u64,
        scratch: &mut Self::Scratch,
    ) -> Option<Vec3>;

    /// Step size (mm) for this attempt. Yeh randomizes per attempt when
    /// `plan.step_size_mm <= 0` (DSI-Studio autotrack sentinel); Dipy is
    /// always fixed. The tracker uses this to advance the point each step.
    fn step_size_mm(&self) -> f32;
}

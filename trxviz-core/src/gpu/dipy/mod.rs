//! GPU tractography pipelines for Dipy-style ODF trackers.
//!
//! Split by concern:
//!   - [`shared`] — input prep (`DipyGpuInputs`, `prepare_dipy_inputs`),
//!     FOD precompute, post-hoc filter helpers, streamline assembly,
//!     shared bind-group-layout entries, and the two constants used by
//!     both paths.
//!   - [`readback`] — `map_slices_blocking` + GPU readback timeout. GPU
//!     failures return an error rather than hanging the worker thread
//!     on an un-callbacked `map_async` — see PR 1 bug #5.
//!   - [`prob`] — the probabilistic path (PMF-on-sphere sampling in the
//!     shader).
//!   - [`ptt`] — the PTT path (Parallel Transport Tractography; probe
//!     arcs with rejection sampling).
//!
//! `run_gpu_dipy` is the public entry point; it dispatches on
//! `plan.direction_getter` to `prob::run` or `ptt::run`. The outer
//! workflow job in `trxviz-core::workflow::jobs` only sees this router.

use crate::error::WorkflowResult;
use crate::workflow::tracking::CancelFlag;
use crate::workflow::{DipyDirectionGetter, DipyTractographyPlan, StreamlineFlow};

pub(super) mod prob;
pub(super) mod ptt;
pub(super) mod readback;
pub(super) mod shared;

/// Dispatch a Dipy-style GPU tractography run.
pub fn run_gpu_dipy(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cancel: &CancelFlag,
) -> WorkflowResult<StreamlineFlow> {
    match plan.direction_getter {
        DipyDirectionGetter::Probabilistic => prob::run(plan, device, queue, cancel),
        DipyDirectionGetter::Ptt { .. } => ptt::run(plan, device, queue, cancel),
    }
}

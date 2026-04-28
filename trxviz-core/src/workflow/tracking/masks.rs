//! Borrowed views of the per-step + post-hoc mask sets that every
//! tractography op may wire. Plans own `Arc<VoxelMask>`s; the tracker just
//! needs cheap borrowed views, so we pass these view structs instead of
//! threading seven individual references through `track_one_streamline`.

use std::sync::Arc;

use super::super::tracking_filters::{
    point_in_mask, streamline_endpoint_in, streamline_hits_all_rois, streamline_passes_hausdorff,
    streamline_satisfies_end_masks,
};
use super::super::types::{PostFilter, VoxelMask};
use super::accum::AttemptOutcome;

/// Per-step masks consulted inside the tracking loop.
///
/// - `roa` — "region of avoidance". If a streamline enters a non-zero
///   voxel here, the whole streamline is rejected (`AttemptOutcome::RejectRoa`).
/// - `limiting` — anatomical-extent mask. Tracking terminates cleanly
///   when the streamline leaves a non-zero voxel (branch ends; streamline
///   may still be kept if it met min-length).
/// - `term` — termination mask. Tracking appends the hit point and
///   terminates cleanly.
#[derive(Default, Clone, Copy)]
pub struct PerStepMasks<'a> {
    pub limiting: Option<&'a VoxelMask>,
    pub roa: Option<&'a VoxelMask>,
    pub term: Option<&'a VoxelMask>,
}

/// Post-hoc filter set applied once the full bidirectional streamline is
/// assembled. Each filter returns an `AttemptOutcome::Reject*` to roll the
/// streamline back.
pub struct PostFilterSet<'a> {
    pub roi_masks: &'a [Arc<VoxelMask>],
    pub end_masks: &'a [Arc<VoxelMask>],
    pub no_end_mask: Option<&'a VoxelMask>,
    pub post_filter: Option<&'a PostFilter>,
}

impl<'a> PostFilterSet<'a> {
    /// Apply every filter in order. Returns `None` on acceptance; on
    /// rejection, returns the first failing outcome.
    pub fn evaluate(&self, streamline: &[[f32; 3]]) -> Option<AttemptOutcome> {
        if !self.roi_masks.is_empty() && !streamline_hits_all_rois(streamline, self.roi_masks) {
            return Some(AttemptOutcome::RejectRoi);
        }
        if let Some(ne) = self.no_end_mask
            && streamline_endpoint_in(streamline, ne)
        {
            return Some(AttemptOutcome::RejectNoEnd);
        }
        if !self.end_masks.is_empty() && !streamline_satisfies_end_masks(streamline, self.end_masks)
        {
            return Some(AttemptOutcome::RejectEnd);
        }
        match self.post_filter {
            Some(PostFilter::Hausdorff {
                reference_points_ras,
                max_mm,
            }) => {
                if !streamline_passes_hausdorff(streamline, reference_points_ras, *max_mm) {
                    return Some(AttemptOutcome::RejectHausdorff);
                }
            }
            None => {}
        }
        None
    }
}

/// Outcome of a single per-step mask check. Used by the tracker to decide
/// whether to continue, cleanly terminate, or reject the whole streamline.
#[derive(Clone, Copy)]
pub(super) enum StepMaskDecision {
    /// No mask fired; continue tracking.
    Continue,
    /// `term` mask fired; push `pt` and terminate this branch cleanly.
    TerminateAt,
    /// `limiting` mask fired (outside the limiting region); terminate
    /// cleanly without pushing `pt`.
    Terminate,
    /// `roa` mask fired; reject the entire streamline.
    RejectAll,
}

impl<'a> PerStepMasks<'a> {
    /// Evaluate every per-step mask at `pt`. Order: roa (reject) → limiting
    /// (terminate) → term (terminate-with-push). Matches the legacy Yeh /
    /// Dipy inner-loop order.
    pub(super) fn evaluate_at(&self, pt: glam::Vec3) -> StepMaskDecision {
        if let Some(m) = self.roa
            && point_in_mask(pt, m)
        {
            return StepMaskDecision::RejectAll;
        }
        if let Some(m) = self.limiting
            && !point_in_mask(pt, m)
        {
            return StepMaskDecision::Terminate;
        }
        if let Some(m) = self.term
            && point_in_mask(pt, m)
        {
            return StepMaskDecision::TerminateAt;
        }
        StepMaskDecision::Continue
    }
}

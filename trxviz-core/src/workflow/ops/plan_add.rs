//! Chain-augmentation ops for `TrackingPlan`. Each op takes
//! `[TrackingPlan, VoxelMask]` and returns `[TrackingPlan]` with the mask
//! appended / assigned to the appropriate region role. This lets users
//! build up arbitrarily complex constraint sets without variadic ports.

use std::sync::Arc;

use crate::error::WorkflowResult;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{TrackingPlan, VoxelMask, WorkflowValue};

use super::super::{EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp};

/// Unpack `[TrackingPlan, VoxelMask]` inputs. The plan input is required;
/// the mask input may be None (returns Ok((plan, None))) so that an
/// unwired mask is a no-op rather than an error.
fn unpack_inputs(
    ctx: &EvalCtx<'_, '_>,
    label: &str,
) -> WorkflowResult<(Arc<TrackingPlan>, Option<Arc<VoxelMask>>)> {
    let plan = match ctx.inputs.first().and_then(|v| v.as_ref()) {
        Some(ev) => match &ev.value {
            WorkflowValue::TrackingPlan(p) => p.clone(),
            _ => {
                return Err(crate::error::WorkflowError::Evaluation(format!(
                    "{label}: first input must be a TrackingPlan"
                )));
            }
        },
        None => {
            return Err(crate::error::WorkflowError::Evaluation(format!(
                "{label}: first input (TrackingPlan) is required"
            )));
        }
    };
    let mask = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
        Some(ev) => match &ev.value {
            WorkflowValue::VoxelMask(m) => Some(m.clone()),
            _ => {
                return Err(crate::error::WorkflowError::Evaluation(format!(
                    "{label}: second input must be a VoxelMask (or unconnected)"
                )));
            }
        },
        None => None,
    };
    Ok((plan, mask))
}

fn ports_two() -> &'static [PortKind] {
    &[PortKind::TrackingPlan, PortKind::VoxelMask]
}

fn ports_one() -> &'static [PortKind] {
    &[PortKind::TrackingPlan]
}

// ── AddRoi ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddRoiOp;

impl WorkflowOp for AddRoiOp {
    fn tag(&self) -> &'static str {
        "add_roi"
    }
    fn title(&self) -> &'static str {
        "Add ROI (waypoint)"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            new_plan.roi_masks.push(m);
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddRoiOp> for WorkflowNodeKind {
    fn from(_: AddRoiOp) -> Self {
        Self::AddRoi
    }
}

// ── AddRoa ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddRoaOp;

impl WorkflowOp for AddRoaOp {
    fn tag(&self) -> &'static str {
        "add_roa"
    }
    fn title(&self) -> &'static str {
        "Add ROA (exclusion)"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            new_plan.roa_mask = Some(match new_plan.roa_mask.take() {
                Some(existing) => Arc::new(union_masks(&existing, &m)),
                None => m,
            });
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddRoaOp> for WorkflowNodeKind {
    fn from(_: AddRoaOp) -> Self {
        Self::AddRoa
    }
}

// ── AddEndRegion ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddEndRegionOp;

impl WorkflowOp for AddEndRegionOp {
    fn tag(&self) -> &'static str {
        "add_end_region"
    }
    fn title(&self) -> &'static str {
        "Add End Region"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            new_plan.end_masks.push(m);
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddEndRegionOp> for WorkflowNodeKind {
    fn from(_: AddEndRegionOp) -> Self {
        Self::AddEndRegion
    }
}

// ── AddLimiting ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddLimitingOp;

impl WorkflowOp for AddLimitingOp {
    fn tag(&self) -> &'static str {
        "add_limiting"
    }
    fn title(&self) -> &'static str {
        "Add Limiting Region"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            // Limiting is an AND constraint — streamlines must stay inside.
            // Multiple limiting regions aren't especially meaningful, so we
            // overwrite rather than union; wire a MergeVoxelMasks upstream
            // if you want the intersection of several masks.
            new_plan.limiting_mask = Some(m);
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddLimitingOp> for WorkflowNodeKind {
    fn from(_: AddLimitingOp) -> Self {
        Self::AddLimiting
    }
}

// ── AddTerm ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddTermOp;

impl WorkflowOp for AddTermOp {
    fn tag(&self) -> &'static str {
        "add_term"
    }
    fn title(&self) -> &'static str {
        "Add Terminative Region"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            new_plan.term_mask = Some(match new_plan.term_mask.take() {
                Some(existing) => Arc::new(union_masks(&existing, &m)),
                None => m,
            });
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddTermOp> for WorkflowNodeKind {
    fn from(_: AddTermOp) -> Self {
        Self::AddTerm
    }
}

// ── AddNoEnd ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct AddNoEndOp;

impl WorkflowOp for AddNoEndOp {
    fn tag(&self) -> &'static str {
        "add_no_end"
    }
    fn title(&self) -> &'static str {
        "Add No-End Region"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        ports_two()
    }
    fn output_ports(&self) -> &'static [PortKind] {
        ports_one()
    }
    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }
    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let (plan, mask) = unpack_inputs(ctx, self.title())?;
        let mut new_plan: TrackingPlan = (*plan).clone();
        if let Some(m) = mask {
            new_plan.no_end_mask = Some(match new_plan.no_end_mask.take() {
                Some(existing) => Arc::new(union_masks(&existing, &m)),
                None => m,
            });
        }
        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(new_plan)).into()])
    }
}
impl From<AddNoEndOp> for WorkflowNodeKind {
    fn from(_: AddNoEndOp) -> Self {
        Self::AddNoEnd
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Pointwise OR union. Only well-defined when both masks share the same
/// grid; otherwise we conservatively keep `a` and warn in the log.
fn union_masks(a: &VoxelMask, b: &VoxelMask) -> VoxelMask {
    if a.dims != b.dims || a.data.len() != b.data.len() {
        log::warn!(
            "union_masks: grid mismatch (a.dims={:?}, b.dims={:?}); keeping first mask",
            a.dims,
            b.dims
        );
        return a.clone();
    }
    let data: Vec<u8> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if *x != 0 || *y != 0 { 1 } else { 0 })
        .collect();
    VoxelMask {
        dims: a.dims,
        voxel_to_ras: a.voxel_to_ras,
        data,
        ..Default::default()
    }
}

//! `PrepareSimplePlanOp` — builds a `TrackingPlan` from a seed `VoxelMask`
//! plus per-parameter manual overrides. Each override has an `enabled` flag:
//! if false, the plan leaves the field as `None` and the downstream tracker
//! uses its own slider.
//!
//! Designed to compose with future `AddRoi`/`AddRoa`/`AddEndRegion` ops that
//! take a `TrackingPlan` + `VoxelMask` and return a plan with the region
//! appended. This chain pattern scales to arbitrary DSI-Studio-style
//! multi-region constraints without variadic ports.

use std::sync::Arc;

use crate::error::WorkflowResult;
use crate::workflow::types::{TrackingPlan, VoxelMask, WorkflowValue};

use super::super::{EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp};

fn default_step() -> f32 {
    1.0
}
fn default_angle() -> f32 {
    60.0
}
fn default_min_len() -> f32 {
    15.0
}
fn default_max_len() -> f32 {
    100.0
}
fn default_fa() -> f32 {
    0.05
}
fn default_smooth() -> f32 {
    0.25
}
fn default_fixel_otsu() -> f32 {
    0.1
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PrepareSimplePlanOp {
    // Each override has an `enabled` flag. When false, the plan leaves the
    // field None and the tracker slider wins.
    #[serde(default)]
    pub override_step: bool,
    #[serde(default = "default_step")]
    pub step_size_mm: f32,

    #[serde(default)]
    pub override_angle: bool,
    #[serde(default = "default_angle")]
    pub max_angle_deg: f32,

    #[serde(default)]
    pub override_min_len: bool,
    #[serde(default = "default_min_len")]
    pub min_len_mm: f32,

    #[serde(default)]
    pub override_max_len: bool,
    #[serde(default = "default_max_len")]
    pub max_len_mm: f32,

    #[serde(default)]
    pub override_fixel_threshold: bool,
    #[serde(default = "default_fa")]
    pub fixel_threshold: f32,

    #[serde(default)]
    pub override_smooth: bool,
    #[serde(default = "default_smooth")]
    pub smooth_fraction: f32,

    /// When true, the plan writes `fixel_otsu = Some(self.fixel_otsu)`.
    /// Otherwise (default) the tracker consults the scene's default
    /// Otsu via its own fallback path.
    #[serde(default)]
    pub override_fixel_otsu: bool,
    #[serde(default = "default_fixel_otsu")]
    pub fixel_otsu: f32,
}

impl Default for PrepareSimplePlanOp {
    fn default() -> Self {
        Self {
            override_step: false,
            step_size_mm: default_step(),
            override_angle: false,
            max_angle_deg: default_angle(),
            override_min_len: false,
            min_len_mm: default_min_len(),
            override_max_len: false,
            max_len_mm: default_max_len(),
            override_fixel_threshold: false,
            fixel_threshold: default_fa(),
            override_smooth: false,
            smooth_fraction: default_smooth(),
            override_fixel_otsu: false,
            fixel_otsu: default_fixel_otsu(),
        }
    }
}

impl WorkflowOp for PrepareSimplePlanOp {
    fn tag(&self) -> &'static str {
        "prepare_simple_plan"
    }

    fn title(&self) -> &'static str {
        "Prepare Simple Plan"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::VoxelMask]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::TrackingPlan]
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        // Optional seed mask. When unwired, the tracker will seed across the
        // whole fixel/ODF mask.
        let seed_mask: Option<Arc<VoxelMask>> = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Prepare Simple Plan: input must be a VoxelMask (or unconnected)".into(),
                    ));
                }
            },
            None => None,
        };

        // Infer a placeholder grid if no mask is wired. The tracker only
        // reads `seed_mask` + override fields from this plan, so grid_dims +
        // voxel_to_ras are informational here.
        let (grid_dims, voxel_to_ras) = match &seed_mask {
            Some(m) => (m.dims, m.voxel_to_ras),
            None => ([0, 0, 0], glam::Mat4::IDENTITY),
        };

        let plan = TrackingPlan {
            label: ctx.node.label.clone(),
            grid_dims,
            voxel_to_ras,
            seed_mask,
            limiting_mask: None,
            roa_mask: None,
            term_mask: None,
            roi_masks: Vec::new(),
            end_masks: Vec::new(),
            no_end_mask: None,
            post_filter: None,
            min_len_mm: self.override_min_len.then_some(self.min_len_mm),
            max_len_mm: self.override_max_len.then_some(self.max_len_mm),
            max_angle_deg: self.override_angle.then_some(self.max_angle_deg),
            step_size_mm: self.override_step.then_some(self.step_size_mm),
            fixel_threshold: self
                .override_fixel_threshold
                .then_some(self.fixel_threshold),
            smooth_fraction: self.override_smooth.then_some(self.smooth_fraction),
            tolerance_mm: None,
            fixel_otsu: self.override_fixel_otsu.then_some(self.fixel_otsu),
        };

        Ok(vec![WorkflowValue::TrackingPlan(Arc::new(plan)).into()])
    }
}

impl From<PrepareSimplePlanOp> for WorkflowNodeKind {
    fn from(op: PrepareSimplePlanOp) -> Self {
        Self::PrepareSimplePlan {
            override_step: op.override_step,
            step_size_mm: op.step_size_mm,
            override_angle: op.override_angle,
            max_angle_deg: op.max_angle_deg,
            override_min_len: op.override_min_len,
            min_len_mm: op.min_len_mm,
            override_max_len: op.override_max_len,
            max_len_mm: op.max_len_mm,
            override_fixel_threshold: op.override_fixel_threshold,
            fixel_threshold: op.fixel_threshold,
            override_smooth: op.override_smooth,
            smooth_fraction: op.smooth_fraction,
            override_fixel_otsu: op.override_fixel_otsu,
            fixel_otsu: op.fixel_otsu,
        }
    }
}

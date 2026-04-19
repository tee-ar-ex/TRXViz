use std::sync::Arc;

use super::super::{
    EvalCtx, PortKind, ReactiveStreamlineOp, ReactiveStreamlinePlan, WorkflowNodeKind, WorkflowOp,
    evaluate_derived_streamline_plan, expect_parcel_selection_input, expect_streamline_input,
};
use crate::data::parcellation_data::ParcellationVolume;

#[derive(Debug, Clone)]
pub struct ParcelRoiOp;

#[derive(Debug, Clone)]
pub struct ParcelRoaOp;

#[derive(Debug, Clone)]
pub struct ParcelEndOp {
    pub endpoint_count: usize,
}

#[derive(Debug, Clone)]
pub struct ParcelCropOp {
    pub keep_inside: bool,
}

impl Default for ParcelRoiOp {
    fn default() -> Self {
        Self
    }
}

impl Default for ParcelRoaOp {
    fn default() -> Self {
        Self
    }
}

impl Default for ParcelEndOp {
    fn default() -> Self {
        Self { endpoint_count: 1 }
    }
}

fn lookup_parcellation(
    ctx: &EvalCtx<'_, '_>,
    source_id: usize,
    label: &str,
) -> crate::error::WorkflowResult<Arc<ParcellationVolume>> {
    let parcellation = ctx.parcellation_assets.get(&source_id).ok_or_else(|| {
        crate::error::WorkflowError::Evaluation(format!("{label} is missing its parcellation"))
    })?;
    Ok(parcellation.asset.data.clone())
}

fn derived_reactive_op(
    ctx: &mut EvalCtx<'_, '_>,
    label: &str,
    op: ReactiveStreamlineOp,
) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
    let flow = expect_streamline_input(ctx.inputs, label)?;
    let plan = ReactiveStreamlinePlan {
        node_uuid: ctx.node.uuid,
        label: ctx.node.label.clone(),
        op,
        left: flow.clone(),
        right: flow,
    };
    ctx.scene_plan.reactive_streamline_plans.push(plan.clone());
    evaluate_derived_streamline_plan(
        ctx.node,
        plan,
        ctx.inputs,
        ctx.execution_cache,
        ctx.node_state,
    )
}

impl WorkflowOp for ParcelRoiOp {
    fn tag(&self) -> &'static str {
        "parcel_roi"
    }

    fn title(&self) -> &'static str {
        "Parcel ROI"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::ParcelSelection]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let selection = expect_parcel_selection_input(ctx.inputs, self.title())?;
        let parcellation = lookup_parcellation(ctx, selection.source_id, self.title())?;
        derived_reactive_op(
            ctx,
            self.title(),
            ReactiveStreamlineOp::ParcelROI {
                parcellation,
                labels: selection.labels,
            },
        )
    }
}

impl WorkflowOp for ParcelRoaOp {
    fn tag(&self) -> &'static str {
        "parcel_roa"
    }

    fn title(&self) -> &'static str {
        "Parcel ROA"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::ParcelSelection]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let selection = expect_parcel_selection_input(ctx.inputs, self.title())?;
        let parcellation = lookup_parcellation(ctx, selection.source_id, self.title())?;
        derived_reactive_op(
            ctx,
            self.title(),
            ReactiveStreamlineOp::ParcelROA {
                parcellation,
                labels: selection.labels,
            },
        )
    }
}

impl WorkflowOp for ParcelEndOp {
    fn tag(&self) -> &'static str {
        "parcel_end"
    }

    fn title(&self) -> &'static str {
        "Parcel End"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::ParcelSelection]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let selection = expect_parcel_selection_input(ctx.inputs, self.title())?;
        let parcellation = lookup_parcellation(ctx, selection.source_id, self.title())?;
        derived_reactive_op(
            ctx,
            self.title(),
            ReactiveStreamlineOp::ParcelEnd {
                parcellation,
                labels: selection.labels,
                endpoint_count: self.endpoint_count,
            },
        )
    }
}

impl WorkflowOp for ParcelCropOp {
    fn tag(&self) -> &'static str {
        if self.keep_inside {
            "parcel_limiting"
        } else {
            "parcel_terminative"
        }
    }

    fn title(&self) -> &'static str {
        if self.keep_inside {
            "Parcel Limiting"
        } else {
            "Parcel Terminative"
        }
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::ParcelSelection]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let selection = expect_parcel_selection_input(ctx.inputs, self.title())?;
        let parcellation = lookup_parcellation(ctx, selection.source_id, self.title())?;
        derived_reactive_op(
            ctx,
            self.title(),
            ReactiveStreamlineOp::ParcelCrop {
                parcellation,
                labels: selection.labels,
                keep_inside: self.keep_inside,
            },
        )
    }
}

impl From<ParcelRoiOp> for WorkflowNodeKind {
    fn from(_: ParcelRoiOp) -> Self {
        Self::ParcelROI
    }
}

impl From<ParcelRoaOp> for WorkflowNodeKind {
    fn from(_: ParcelRoaOp) -> Self {
        Self::ParcelROA
    }
}

impl From<ParcelEndOp> for WorkflowNodeKind {
    fn from(op: ParcelEndOp) -> Self {
        Self::ParcelEnd {
            endpoint_count: op.endpoint_count,
        }
    }
}

impl From<ParcelCropOp> for WorkflowNodeKind {
    fn from(op: ParcelCropOp) -> Self {
        if op.keep_inside {
            Self::ParcelLimiting
        } else {
            Self::ParcelTerminative
        }
    }
}

use trx_rs::DuplicateRemovalParams;

use super::super::{
    EvalCtx, PortKind, ReactiveStreamlineOp, ReactiveStreamlinePlan, WorkflowNodeKind, WorkflowOp,
    evaluate_derived_streamline_plan, expect_streamline_input,
};
use crate::workflow::methods::OpCategory;

#[derive(Debug, Clone)]
pub struct RemoveDuplicatesOp {
    pub params: DuplicateRemovalParams,
}

impl Default for RemoveDuplicatesOp {
    fn default() -> Self {
        Self {
            params: DuplicateRemovalParams::default(),
        }
    }
}

impl WorkflowOp for RemoveDuplicatesOp {
    fn tag(&self) -> &'static str {
        "remove_duplicates"
    }

    fn title(&self) -> &'static str {
        "Remove Duplicates"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn category(&self) -> OpCategory {
        OpCategory::StreamlineFilter
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let plan = ReactiveStreamlinePlan {
            node_uuid: ctx.node.uuid,
            label: ctx.node.label.clone(),
            op: ReactiveStreamlineOp::RemoveDuplicates {
                params: self.params.clone(),
            },
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
}

impl From<RemoveDuplicatesOp> for WorkflowNodeKind {
    fn from(op: RemoveDuplicatesOp) -> Self {
        Self::RemoveDuplicates { params: op.params }
    }
}

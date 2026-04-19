use super::super::{
    EvalCtx, PortKind, ReactiveStreamlineOp, ReactiveStreamlinePlan, WorkflowNodeKind, WorkflowOp,
    WorkflowValue, evaluate_derived_streamline_plan, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct MergeOp;

impl Default for MergeOp {
    fn default() -> Self {
        Self
    }
}

impl WorkflowOp for MergeOp {
    fn tag(&self) -> &'static str {
        "merge"
    }

    fn title(&self) -> &'static str {
        "Merge"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let left = expect_streamline_input(ctx.inputs, self.title())?;
        let right = match ctx.inputs.get(1).cloned().flatten() {
            Some(value) => match value.value {
                WorkflowValue::Streamline(flow) => flow,
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Merge needs a right streamline input".to_string(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Merge needs a right streamline input".to_string(),
                ));
            }
        };
        let plan = ReactiveStreamlinePlan {
            node_uuid: ctx.node.uuid,
            label: ctx.node.label.clone(),
            op: ReactiveStreamlineOp::Merge,
            left,
            right,
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

impl From<MergeOp> for WorkflowNodeKind {
    fn from(_: MergeOp) -> Self {
        Self::Merge
    }
}

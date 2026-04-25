use super::super::{
    EvalCtx, PortKind, ReactiveStreamlineOp, ReactiveStreamlinePlan, WorkflowNodeKind, WorkflowOp,
    WorkflowValue, evaluate_derived_streamline_plan, expect_streamline_input,
};
use crate::workflow::methods::OpCategory;

#[derive(Debug, Clone, Copy)]
pub struct AddGroupsFromParcellationOp;

impl Default for AddGroupsFromParcellationOp {
    fn default() -> Self {
        Self
    }
}

impl WorkflowOp for AddGroupsFromParcellationOp {
    fn tag(&self) -> &'static str {
        "add_groups_from_parcellation"
    }

    fn title(&self) -> &'static str {
        "Add Groups From Parcellation"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Parcellation]
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
        let source_id = match ctx.inputs.get(1).cloned().flatten() {
            Some(value) => match value.value {
                WorkflowValue::Parcellation(source_id) => source_id,
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Add Groups From Parcellation needs a parcellation input".to_string(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Add Groups From Parcellation needs a parcellation input".to_string(),
                ));
            }
        };
        let parcellation = ctx.parcellation_assets.get(&source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing parcellation {source_id}"))
        })?;
        let plan = ReactiveStreamlinePlan {
            node_uuid: ctx.node.uuid,
            label: ctx.node.label.clone(),
            op: ReactiveStreamlineOp::AddGroupsFromParcellation {
                parcellation: parcellation.asset.data.clone(),
                parcellation_name: parcellation.asset.name.clone(),
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

impl From<AddGroupsFromParcellationOp> for WorkflowNodeKind {
    fn from(_: AddGroupsFromParcellationOp) -> Self {
        Self::AddGroupsFromParcellation
    }
}

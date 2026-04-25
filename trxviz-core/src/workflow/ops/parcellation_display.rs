use super::super::{
    EvalCtx, ParcelIdSet, ParcellationDrawPlan, PortKind, WorkflowNodeKind, WorkflowOp,
    expect_parcellation_input, resolve_selected_labels,
};
use crate::workflow::methods::OpCategory;

#[derive(Debug, Clone)]
pub struct ParcellationDisplayOp {
    pub labels: ParcelIdSet,
    pub opacity: f32,
}

impl Default for ParcellationDisplayOp {
    fn default() -> Self {
        Self {
            labels: ParcelIdSet::default(),
            opacity: 0.9,
        }
    }
}

impl WorkflowOp for ParcellationDisplayOp {
    fn tag(&self) -> &'static str {
        "parcellation_display"
    }

    fn title(&self) -> &'static str {
        "Parcellation Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Parcellation]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let source_id = expect_parcellation_input(ctx.inputs, self.title())?;
        let parcellation = ctx.parcellation_assets.get(&source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing parcellation {source_id}"))
        })?;
        let labels = resolve_selected_labels(&self.labels, &parcellation.asset.data);
        ctx.scene_plan
            .parcellation_draws
            .push(ParcellationDrawPlan {
                source_id,
                labels,
                opacity: self.opacity,
            });
        Ok(Vec::new())
    }
}

impl From<ParcellationDisplayOp> for WorkflowNodeKind {
    fn from(op: ParcellationDisplayOp) -> Self {
        Self::ParcellationDisplay {
            labels: op.labels,
            opacity: op.opacity,
        }
    }
}

use super::super::{
    EvalCtx, ParcelIdSet, ParcelSelection, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_parcellation_input, resolve_selected_labels,
};

#[derive(Debug, Clone)]
pub struct ParcelSelectOp {
    pub labels: ParcelIdSet,
}

impl Default for ParcelSelectOp {
    fn default() -> Self {
        Self {
            labels: ParcelIdSet::default(),
        }
    }
}

impl WorkflowOp for ParcelSelectOp {
    fn tag(&self) -> &'static str {
        "parcel_select"
    }

    fn title(&self) -> &'static str {
        "Parcel Select"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Parcellation]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::ParcelSelection]
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
        Ok(vec![
            WorkflowValue::ParcelSelection(ParcelSelection { source_id, labels }).into(),
        ])
    }
}

impl From<ParcelSelectOp> for WorkflowNodeKind {
    fn from(op: ParcelSelectOp) -> Self {
        Self::ParcelSelect { labels: op.labels }
    }
}

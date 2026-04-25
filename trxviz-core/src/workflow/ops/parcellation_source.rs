use crate::data::loaded_files::FileId;
use crate::workflow::methods::OpCategory;

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

#[derive(Debug, Clone, Copy)]
pub struct ParcellationSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for ParcellationSourceOp {
    fn tag(&self) -> &'static str {
        "parcellation_source"
    }

    fn title(&self) -> &'static str {
        "Parcellation Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Parcellation]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Source
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        ctx.parcellation_assets
            .get(&self.source_id)
            .ok_or_else(|| {
                crate::error::WorkflowError::Evaluation(format!(
                    "Missing parcellation source {}",
                    self.source_id
                ))
            })?;
        Ok(vec![WorkflowValue::Parcellation(self.source_id).into()])
    }
}

impl From<ParcellationSourceOp> for WorkflowNodeKind {
    fn from(op: ParcellationSourceOp) -> Self {
        Self::ParcellationSource {
            source_id: op.source_id,
        }
    }
}

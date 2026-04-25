use crate::data::loaded_files::FileId;
use crate::workflow::methods::OpCategory;

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

#[derive(Debug, Clone, Copy)]
pub struct CiftiSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for CiftiSourceOp {
    fn tag(&self) -> &'static str {
        "cifti_source"
    }

    fn title(&self) -> &'static str {
        "CIFTI Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Cifti]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Source
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        ctx.cifti_assets.get(&self.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!(
                "Missing CIFTI source {}",
                self.source_id
            ))
        })?;
        Ok(vec![WorkflowValue::Cifti(self.source_id).into()])
    }
}

impl From<CiftiSourceOp> for WorkflowNodeKind {
    fn from(op: CiftiSourceOp) -> Self {
        Self::CiftiSource {
            source_id: op.source_id,
        }
    }
}

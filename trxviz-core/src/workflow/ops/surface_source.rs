use crate::data::loaded_files::FileId;
use crate::workflow::methods::OpCategory;

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

#[derive(Debug, Clone, Copy)]
pub struct SurfaceSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for SurfaceSourceOp {
    fn tag(&self) -> &'static str {
        "surface_source"
    }

    fn title(&self) -> &'static str {
        "Surface Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Surface]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Source
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        ctx.surface_assets.get(&self.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!(
                "Missing surface source {}",
                self.source_id
            ))
        })?;
        Ok(vec![WorkflowValue::Surface(self.source_id).into()])
    }
}

impl From<SurfaceSourceOp> for WorkflowNodeKind {
    fn from(op: SurfaceSourceOp) -> Self {
        Self::SurfaceSource {
            source_id: op.source_id,
        }
    }
}

use crate::data::loaded_files::FileId;
use crate::workflow::methods::OpCategory;

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

#[derive(Debug, Clone, Copy)]
pub struct VolumeSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for VolumeSourceOp {
    fn tag(&self) -> &'static str {
        "volume_source"
    }

    fn title(&self) -> &'static str {
        "Volume Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Volume]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Source
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        ctx.volume_assets.get(&self.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!(
                "Missing volume source {}",
                self.source_id
            ))
        })?;
        Ok(vec![WorkflowValue::Volume(self.source_id).into()])
    }
}

impl From<VolumeSourceOp> for WorkflowNodeKind {
    fn from(op: VolumeSourceOp) -> Self {
        Self::VolumeSource {
            source_id: op.source_id,
        }
    }
}

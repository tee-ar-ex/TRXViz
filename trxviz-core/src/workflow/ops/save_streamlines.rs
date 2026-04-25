use std::path::PathBuf;

use super::super::{
    EvalCtx, PortKind, SaveStreamlinePlan, WorkflowNodeKind, WorkflowOp, expect_streamline_input,
};
use crate::workflow::methods::OpCategory;

#[derive(Debug, Clone)]
pub struct SaveStreamlinesOp {
    pub output_path: String,
}

impl Default for SaveStreamlinesOp {
    fn default() -> Self {
        Self {
            output_path: String::new(),
        }
    }
}

impl WorkflowOp for SaveStreamlinesOp {
    fn tag(&self) -> &'static str {
        "save_streamlines"
    }

    fn title(&self) -> &'static str {
        "Save Streamlines"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Io
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        if self.output_path.trim().is_empty() {
            return Err(crate::error::WorkflowError::Evaluation(
                "Save Streamlines needs an output path".to_string(),
            ));
        }
        ctx.save_targets.insert(
            ctx.node.uuid,
            SaveStreamlinePlan {
                node_uuid: ctx.node.uuid,
                output_path: PathBuf::from(&self.output_path),
                flow,
            },
        );
        Ok(Vec::new())
    }
}

impl From<SaveStreamlinesOp> for WorkflowNodeKind {
    fn from(op: SaveStreamlinesOp) -> Self {
        Self::SaveStreamlines {
            output_path: op.output_path,
        }
    }
}

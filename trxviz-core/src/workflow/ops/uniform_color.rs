use crate::data::trx_data::ColorMode;
use crate::workflow::methods::OpCategory;

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct UniformColorOp {
    pub color: [f32; 4],
}

impl Default for UniformColorOp {
    fn default() -> Self {
        Self {
            color: [0.95, 0.8, 0.2, 1.0],
        }
    }
}

impl WorkflowOp for UniformColorOp {
    fn tag(&self) -> &'static str {
        "uniform_color"
    }

    fn title(&self) -> &'static str {
        "Uniform Color"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Coloring
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        Ok(vec![
            WorkflowValue::Streamline(super::super::StreamlineFlow {
                color_mode: ColorMode::Uniform(self.color),
                ..flow
            })
            .into(),
        ])
    }
}

impl From<UniformColorOp> for WorkflowNodeKind {
    fn from(op: UniformColorOp) -> Self {
        Self::UniformColor { color: op.color }
    }
}

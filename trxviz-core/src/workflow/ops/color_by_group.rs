use crate::data::trx_data::ColorMode;
use crate::workflow::methods::OpCategory;

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct ColorByGroupOp;

impl Default for ColorByGroupOp {
    fn default() -> Self {
        Self
    }
}

impl WorkflowOp for ColorByGroupOp {
    fn tag(&self) -> &'static str {
        "color_by_group"
    }

    fn title(&self) -> &'static str {
        "Color By Group"
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
                color_mode: ColorMode::Group,
                ..flow
            })
            .into(),
        ])
    }
}

impl From<ColorByGroupOp> for WorkflowNodeKind {
    fn from(_: ColorByGroupOp) -> Self {
        Self::ColorByGroup
    }
}

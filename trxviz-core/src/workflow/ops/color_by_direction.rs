use crate::data::trx_data::ColorMode;

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct ColorByDirectionOp;

impl Default for ColorByDirectionOp {
    fn default() -> Self {
        Self
    }
}

impl WorkflowOp for ColorByDirectionOp {
    fn tag(&self) -> &'static str {
        "color_by_direction"
    }

    fn title(&self) -> &'static str {
        "Color By Direction"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        Ok(vec![
            WorkflowValue::Streamline(super::super::StreamlineFlow {
                color_mode: ColorMode::DirectionRgb,
                ..flow
            })
            .into(),
        ])
    }
}

impl From<ColorByDirectionOp> for WorkflowNodeKind {
    fn from(_: ColorByDirectionOp) -> Self {
        Self::ColorByDirection
    }
}

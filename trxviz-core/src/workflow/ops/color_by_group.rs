use crate::data::trx_data::ColorMode;

use super::super::{EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_streamline_input};

#[derive(Debug, Clone, Copy)]
pub struct ColorByGroupOp;

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

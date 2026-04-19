use crate::data::trx_data::ColorMode;

use super::super::{EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_streamline_input};

#[derive(Debug, Clone, Copy)]
pub struct UniformColorOp {
    pub color: [f32; 4],
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

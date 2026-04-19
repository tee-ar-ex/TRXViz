use crate::data::trx_data::ColorMode;

use super::super::{
    DpvFieldName, EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone)]
pub struct ColorByDpvOp {
    pub field: DpvFieldName,
}

impl WorkflowOp for ColorByDpvOp {
    fn tag(&self) -> &'static str {
        "color_by_dpv"
    }

    fn title(&self) -> &'static str {
        "Color By DPV"
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
                color_mode: ColorMode::Dpv(self.field.as_str().to_string()),
                ..flow
            })
            .into(),
        ])
    }
}

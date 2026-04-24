use crate::data::trx_data::ColorMode;
use crate::renderer::mesh_renderer::SurfaceColormap;

use super::super::{
    DpvFieldName, EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input,
};

#[derive(Debug, Clone)]
pub struct ColorByDpvOp {
    pub field: DpvFieldName,
    pub colormap: SurfaceColormap,
}

impl Default for ColorByDpvOp {
    fn default() -> Self {
        Self {
            field: DpvFieldName::default(),
            colormap: SurfaceColormap::default(),
        }
    }
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
                scalar_colormap: self.colormap,
                ..flow
            })
            .into(),
        ])
    }
}

impl From<ColorByDpvOp> for WorkflowNodeKind {
    fn from(op: ColorByDpvOp) -> Self {
        Self::ColorByDPV {
            field: op.field,
            colormap: op.colormap,
        }
    }
}

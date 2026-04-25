use crate::data::trx_data::ColorMode;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::workflow::methods::OpCategory;

use super::super::{
    DpsFieldName, EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input,
};

#[derive(Debug, Clone)]
pub struct ColorByDpsOp {
    pub field: DpsFieldName,
    pub colormap: SurfaceColormap,
}

impl Default for ColorByDpsOp {
    fn default() -> Self {
        Self {
            field: DpsFieldName::default(),
            colormap: SurfaceColormap::default(),
        }
    }
}

impl WorkflowOp for ColorByDpsOp {
    fn tag(&self) -> &'static str {
        "color_by_dps"
    }

    fn title(&self) -> &'static str {
        "Color By DPS"
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
                color_mode: ColorMode::Dps(self.field.as_str().to_string()),
                scalar_colormap: self.colormap,
                ..flow
            })
            .into(),
        ])
    }
}

impl From<ColorByDpsOp> for WorkflowNodeKind {
    fn from(op: ColorByDpsOp) -> Self {
        Self::ColorByDPS {
            field: op.field,
            colormap: op.colormap,
        }
    }
}

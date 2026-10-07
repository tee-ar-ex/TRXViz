use crate::data::trx_data::ColorMode;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::workflow::methods::OpCategory;

use super::super::{
    DpvFieldName, EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input,
};

/// Colour streamlines by a per-vertex DPV field AND drive per-vertex OPACITY
/// from the same field's magnitude: values at (or near) zero are fully
/// transparent, growing opaque toward the extremes. With a diverging colormap
/// (e.g. `BlueWhiteRed`) and a signed, significance-masked field (0 where not
/// significant), this yields a translucent bundle whose significant fixels
/// glow — sign by hue, effect size by opacity — in a single draw.
#[derive(Debug, Clone)]
pub struct ColorByDpvGlowOp {
    pub field: DpvFieldName,
    pub colormap: SurfaceColormap,
}

impl Default for ColorByDpvGlowOp {
    fn default() -> Self {
        Self {
            field: DpvFieldName::default(),
            colormap: SurfaceColormap::BlueWhiteRed,
        }
    }
}

impl WorkflowOp for ColorByDpvGlowOp {
    fn tag(&self) -> &'static str {
        "color_by_dpv_glow"
    }

    fn title(&self) -> &'static str {
        "Color By DPV (glow)"
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
                color_mode: ColorMode::DpvGlow(self.field.as_str().to_string()),
                scalar_colormap: self.colormap,
                ..flow
            })
            .into(),
        ])
    }
}

impl From<ColorByDpvGlowOp> for WorkflowNodeKind {
    fn from(op: ColorByDpvGlowOp) -> Self {
        Self::ColorByDpvGlow {
            field: op.field,
            colormap: op.colormap,
        }
    }
}

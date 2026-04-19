use crate::renderer::mesh_renderer::SurfaceColormap;

use super::super::{
    EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_fixel_scalars_input, expect_fixels_input,
};

#[derive(Debug, Clone, Copy)]
pub struct ColorByFixelScalarsOp {
    pub colormap: SurfaceColormap,
    pub range: Option<(f32, f32)>,
    pub length_scale_by_scalar: bool,
}

impl WorkflowOp for ColorByFixelScalarsOp {
    fn tag(&self) -> &'static str {
        "color_by_fixel_scalars"
    }

    fn title(&self) -> &'static str {
        "Color By Fixel Scalars"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Fixels, PortKind::FixelScalars]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Fixels]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let mut field = expect_fixels_input(ctx.inputs, self.title())?;
        let scalars = expect_fixel_scalars_input(ctx.inputs, self.title())?;
        if scalars.fixel_count != field.scalars.fixel_count {
            return Err(crate::error::WorkflowError::Evaluation(format!(
                "Fixel count mismatch: scalars have {} fixels, field has {}",
                scalars.fixel_count, field.scalars.fixel_count
            )));
        }
        field.colormap_code = match self.colormap {
            SurfaceColormap::BlueWhiteRed => 5,
            SurfaceColormap::Viridis => 3,
            SurfaceColormap::Inferno => 4,
        };
        field.scalar_range = self.range.unwrap_or(scalars.range);
        field.scalars = scalars;
        let output_scalars = field.scalars.clone();
        let _ = self.length_scale_by_scalar;
        Ok(vec![
            WorkflowValue::Fixels(field).into(),
            WorkflowValue::FixelScalars(output_scalars).into(),
        ])
    }
}

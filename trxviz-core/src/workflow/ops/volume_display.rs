use super::super::{
    EvalCtx, PortKind, VolumeDrawPlan, VolumeScalarDrawPlan, WorkflowNodeKind, WorkflowOp,
    expect_volume_input, expect_volume_scalars_input,
};
use crate::data::loaded_files::VolumeColormap;

#[derive(Debug, Clone, Copy)]
pub struct VolumeDisplayOp {
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub window_center: f32,
    pub window_width: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeScalarsDisplayOp {
    pub colormap: VolumeColormap,
    pub opacity: f32,
}

impl Default for VolumeDisplayOp {
    fn default() -> Self {
        Self {
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            window_center: 0.5,
            window_width: 1.0,
        }
    }
}

impl Default for VolumeScalarsDisplayOp {
    fn default() -> Self {
        Self {
            colormap: VolumeColormap::Hot,
            opacity: 1.0,
        }
    }
}

impl WorkflowOp for VolumeDisplayOp {
    fn tag(&self) -> &'static str {
        "volume_display"
    }

    fn title(&self) -> &'static str {
        "Volume Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Volume]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let source_id = expect_volume_input(ctx.inputs, self.title())?;
        if ctx.volume_assets.get(&source_id).is_none() && ctx.odx_assets.get(&source_id).is_none() {
            return Err(crate::error::WorkflowError::Evaluation(format!(
                "Missing volume {source_id}"
            )));
        }
        ctx.scene_plan.volume_draws.push(VolumeDrawPlan {
            source_id,
            colormap: self.colormap,
            opacity: self.opacity,
            window_center: self.window_center,
            window_width: self.window_width,
        });
        Ok(Vec::new())
    }
}

impl WorkflowOp for VolumeScalarsDisplayOp {
    fn tag(&self) -> &'static str {
        "volume_scalars_display"
    }

    fn title(&self) -> &'static str {
        "Volume Scalars Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::VolumeScalars]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let scalars = expect_volume_scalars_input(ctx.inputs, self.title())?;
        ctx.scene_plan
            .volume_scalar_draws
            .push(VolumeScalarDrawPlan {
                dims: scalars.dims,
                voxel_to_ras: scalars.voxel_to_ras.to_cols_array_2d(),
                colormap: self.colormap,
                opacity: self.opacity,
            });
        Ok(Vec::new())
    }
}

impl From<VolumeDisplayOp> for WorkflowNodeKind {
    fn from(op: VolumeDisplayOp) -> Self {
        Self::VolumeDisplay {
            colormap: op.colormap,
            opacity: op.opacity,
            window_center: op.window_center,
            window_width: op.window_width,
        }
    }
}

impl From<VolumeScalarsDisplayOp> for WorkflowNodeKind {
    fn from(op: VolumeScalarsDisplayOp) -> Self {
        Self::VolumeScalarsDisplay {
            colormap: op.colormap,
            opacity: op.opacity,
        }
    }
}

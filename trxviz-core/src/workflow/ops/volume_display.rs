use super::super::{
    EvalCtx, EvaluatedValue, PortKind, VolumeBacking, VolumeDrawPlan, WorkflowNodeKind, WorkflowOp,
    WorkflowValue,
};
use crate::data::loaded_files::VolumeColormap;
use crate::workflow::methods::OpCategory;

/// Renders a volume as orthogonal slices in the 3D viewport. Accepts a
/// single `Volume` input which can be either a file-backed NIfTI
/// (`VolumeBacking::File`) or an in-memory scalar volume
/// (`VolumeBacking::InMemory`, produced by e.g. pyAFQ probability maps,
/// CIFTI subcortical, or ODX DPV).
#[derive(Debug, Clone, Copy)]
pub struct VolumeDisplayOp {
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub window_center: f32,
    pub window_width: f32,
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

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<EvaluatedValue>> {
        let backing = match ctx.inputs.first().cloned().flatten() {
            Some(EvaluatedValue {
                value: WorkflowValue::Volume(b),
                ..
            }) => b,
            _ => return Ok(Vec::new()),
        };
        match &backing {
            VolumeBacking::File(id) => {
                if ctx.volume_assets.get(id).is_none() && ctx.odx_assets.get(id).is_none() {
                    return Err(crate::error::WorkflowError::Evaluation(format!(
                        "Missing volume {id}"
                    )));
                }
            }
            VolumeBacking::InMemory { scalars, .. } => {
                if scalars.dims.contains(&0) || scalars.values.is_empty() {
                    return Ok(Vec::new());
                }
            }
            VolumeBacking::Composite { stack, .. } => {
                if stack.dims.contains(&0) {
                    return Ok(Vec::new());
                }
            }
        }
        ctx.scene_plan.volume_draws.push(VolumeDrawPlan {
            source: backing,
            colormap: self.colormap,
            opacity: self.opacity,
            window_center: self.window_center,
            window_width: self.window_width,
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

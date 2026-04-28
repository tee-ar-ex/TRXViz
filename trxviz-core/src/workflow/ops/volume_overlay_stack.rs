//! `VolumeOverlayStackOp` — composes N volume scalar layers into a
//! single composited slice draw, analogous to `SurfaceOverlayStackOp`.
//!
//! Layer 0 defines the target grid (dims + voxel_to_ras). Subsequent
//! layers can be on any grid; the slice renderer resamples them per
//! displayed slice via the layer's `interpolation` kernel.
//!
//! Composition itself happens at render time inside the slice
//! renderer — this op only validates inputs and packages a
//! `CompositeVolumeStack` plan.

use std::sync::Arc;

use crate::data::cifti::ScalarKind;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{
    CompositeVolumeStack, Interp, VolumeBacking, VolumeOverlayLayerConfig,
    default_volume_overlay_layers,
};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
};

#[derive(Debug, Clone)]
pub struct VolumeOverlayStackOp {
    pub layers: Vec<VolumeOverlayLayerConfig>,
}

impl Default for VolumeOverlayStackOp {
    fn default() -> Self {
        Self {
            layers: default_volume_overlay_layers(),
        }
    }
}

impl WorkflowOp for VolumeOverlayStackOp {
    fn tag(&self) -> &'static str {
        "volume_overlay_stack"
    }

    fn title(&self) -> &'static str {
        "Volume Overlay Stack"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        // Dynamic; resolved via WorkflowNodeKind::inputs() override.
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Volume]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<EvaluatedValue>> {
        // Pull each wired layer's scalars and pair with its config.
        // The base layer (input 0) is required — it defines the
        // composite's target grid.
        let mut layer_scalars: Vec<Option<Arc<crate::data::cifti::VolumeScalars>>> =
            Vec::with_capacity(self.layers.len());
        for (i, _cfg) in self.layers.iter().enumerate() {
            match ctx.inputs.get(i).cloned().flatten() {
                Some(EvaluatedValue {
                    value: WorkflowValue::Volume(backing),
                    ..
                }) => {
                    let scalars = ctx.scalars_for(&backing)?.into_owned();
                    layer_scalars.push(Some(Arc::new(scalars)));
                }
                _ => layer_scalars.push(None),
            }
        }

        let Some(base) = layer_scalars.first().and_then(|l| l.clone()) else {
            return Err(crate::error::WorkflowError::Evaluation(
                "Volume Overlay Stack needs a base volume on input 0".into(),
            ));
        };

        let mut layers: Vec<(
            Arc<crate::data::cifti::VolumeScalars>,
            VolumeOverlayLayerConfig,
        )> = Vec::with_capacity(self.layers.len());
        for (i, cfg) in self.layers.iter().enumerate() {
            let Some(scalars) = layer_scalars[i].clone() else {
                continue;
            };
            // Apply kind-driven interpolation default the first time
            // this layer has a value: if the on-disk default is
            // Trilinear and the layer is a Label map, switch to
            // Nearest. The user's explicit edits in the GUI always
            // win; we only intervene on the still-default Trilinear.
            let mut cfg = cfg.clone();
            if matches!(cfg.interpolation, Interp::Trilinear)
                && matches!(scalars.kind, ScalarKind::Label)
            {
                cfg.interpolation = Interp::Nearest;
            }
            layers.push((scalars, cfg));
        }

        let stack = CompositeVolumeStack {
            dims: base.dims,
            voxel_to_ras: base.voxel_to_ras,
            layers,
        };
        let handle = stack.handle();

        Ok(vec![
            WorkflowValue::Volume(VolumeBacking::Composite {
                handle,
                stack: Arc::new(stack),
            })
            .into(),
        ])
    }
}

impl From<VolumeOverlayStackOp> for WorkflowNodeKind {
    fn from(op: VolumeOverlayStackOp) -> Self {
        Self::VolumeOverlayStack { layers: op.layers }
    }
}

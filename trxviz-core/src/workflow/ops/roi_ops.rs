use std::collections::BTreeSet;
use std::sync::Arc;

use glam::Vec3;

use crate::error::WorkflowResult;
use crate::units::{Millimeters, ParcelId};
use crate::workflow::eval_inputs::{
    expect_odf_field_input, expect_parcellation_input, expect_volume_input,
};
use crate::workflow::types::{ParcelIdSet, VoxelMask};

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

// ── RoiFromParcelOp ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RoiFromParcelOp {
    pub labels: ParcelIdSet,
}

impl Default for RoiFromParcelOp {
    fn default() -> Self {
        Self {
            labels: ParcelIdSet::default(),
        }
    }
}

impl WorkflowOp for RoiFromParcelOp {
    fn tag(&self) -> &'static str {
        "roi_from_parcel"
    }

    fn title(&self) -> &'static str {
        "ROI from Parcellation"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Parcellation]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::VoxelMask]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let parcel_id = expect_parcellation_input(ctx.inputs, self.title())?;
        let loaded = ctx.parcellation_assets.get(&parcel_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation("Missing parcellation asset".into())
        })?;
        let vol = &loaded.asset.data;

        let selected: BTreeSet<ParcelId> =
            crate::workflow::eval_inputs::resolve_selected_labels(&self.labels, vol);

        let [nx, ny, nz] = vol.dims;
        let n_voxels = nx * ny * nz;
        let mut data = vec![0u8; n_voxels];
        for (lin_idx, &label_id) in vol.labels.iter().enumerate() {
            if label_id.0 == 0 || !selected.contains(&label_id) {
                continue;
            }
            if lin_idx < n_voxels {
                data[lin_idx] = 1;
            }
        }

        Ok(vec![WorkflowValue::VoxelMask(Arc::new(VoxelMask {
            dims: [nx as u32, ny as u32, nz as u32],
            voxel_to_ras: vol.voxel_to_ras,
            data,
        }))
        .into()])
    }
}

impl From<RoiFromParcelOp> for WorkflowNodeKind {
    fn from(op: RoiFromParcelOp) -> Self {
        Self::RoiFromParcel {
            labels: op.labels,
        }
    }
}

// ── RoiFromVolumeOp ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct RoiFromVolumeOp {
    pub threshold: f32,
}

impl Default for RoiFromVolumeOp {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl WorkflowOp for RoiFromVolumeOp {
    fn tag(&self) -> &'static str {
        "roi_from_volume"
    }

    fn title(&self) -> &'static str {
        "ROI from Volume Mask"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Volume]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::VoxelMask]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let vol_id = expect_volume_input(ctx.inputs, self.title())?;
        let loaded = ctx.volume_assets.get(&vol_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation("Missing volume asset".into())
        })?;
        let vol = &loaded.volume;
        let [nx, ny, nz] = vol.dims;
        let n_voxels = nx * ny * nz;

        let mut data = vec![0u8; n_voxels];
        // NiftiVolume.data is [i][j][k] order: linear index = i + nx*(j + ny*k)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + nx * (j + ny * k);
                    if vol.data[idx] > self.threshold {
                        data[idx] = 1;
                    }
                }
            }
        }

        Ok(vec![WorkflowValue::VoxelMask(Arc::new(VoxelMask {
            dims: [nx as u32, ny as u32, nz as u32],
            voxel_to_ras: vol.voxel_to_ras,
            data,
        }))
        .into()])
    }
}

impl From<RoiFromVolumeOp> for WorkflowNodeKind {
    fn from(op: RoiFromVolumeOp) -> Self {
        Self::RoiFromVolume {
            threshold: op.threshold,
        }
    }
}

// ── RoiFromShapeOp ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RoiShape {
    Sphere,
    Box,
}

impl Default for RoiShape {
    fn default() -> Self {
        Self::Sphere
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoiFromShapeOp {
    pub center_ras: [f32; 3],
    pub radius_or_half_extent_mm: Millimeters,
    pub shape: RoiShape,
}

impl Default for RoiFromShapeOp {
    fn default() -> Self {
        Self {
            center_ras: [0.0; 3],
            radius_or_half_extent_mm: Millimeters(10.0),
            shape: RoiShape::Sphere,
        }
    }
}

impl WorkflowOp for RoiFromShapeOp {
    fn tag(&self) -> &'static str {
        "roi_from_shape"
    }

    fn title(&self) -> &'static str {
        "ROI from Shape"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::OdfField]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::VoxelMask]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let odf_field = expect_odf_field_input(ctx.inputs, self.title())?;
        let loaded = ctx.odx_assets.get(&odf_field.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation("Missing ODX asset".into())
        })?;
        let scene = &loaded.scene;
        let dims64 = scene.dimensions();
        let dims = [dims64[0] as u32, dims64[1] as u32, dims64[2] as u32];
        let voxel_to_ras = scene.voxel_to_ras();

        let center = Vec3::from(self.center_ras);
        let r = self.radius_or_half_extent_mm.0;

        let [nx, ny, nz] = dims;
        let n_voxels = (nx as usize) * (ny as usize) * (nz as usize);
        let mut data = vec![0u8; n_voxels];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let pt = voxel_to_ras
                        .transform_point3(Vec3::new(x as f32, y as f32, z as f32));
                    let inside = match self.shape {
                        RoiShape::Sphere => pt.distance(center) <= r,
                        RoiShape::Box => {
                            let d = (pt - center).abs();
                            d.x <= r && d.y <= r && d.z <= r
                        }
                    };
                    if inside {
                        let idx = (x as usize)
                            + (nx as usize) * ((y as usize) + (ny as usize) * (z as usize));
                        data[idx] = 1;
                    }
                }
            }
        }

        Ok(vec![WorkflowValue::VoxelMask(Arc::new(VoxelMask {
            dims,
            voxel_to_ras,
            data,
        }))
        .into()])
    }
}

impl From<RoiFromShapeOp> for WorkflowNodeKind {
    fn from(op: RoiFromShapeOp) -> Self {
        Self::RoiFromShape {
            center_ras: op.center_ras,
            radius_or_half_extent_mm: op.radius_or_half_extent_mm,
            shape: op.shape,
        }
    }
}

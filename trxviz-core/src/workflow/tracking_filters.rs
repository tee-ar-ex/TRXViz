//! Shared constraint helpers for the CPU tractography backends (Prob +
//! Yeh). Each tracker owns its own inner step loop; these helpers implement
//! the plan-level region semantics consistently between them.

use glam::Vec3;
use std::sync::Arc;

use super::types::VoxelMask;

/// True when the RAS point lies within a non-zero voxel of `mask`. Honors
/// the mask's own `voxel_to_ras`, so masks on different grids work.
pub(crate) fn point_in_mask(pt_ras: Vec3, mask: &VoxelMask) -> bool {
    let ras_to_vox = mask.voxel_to_ras.inverse();
    let v = ras_to_vox.transform_point3(pt_ras);
    let x = v.x.floor() as i32;
    let y = v.y.floor() as i32;
    let z = v.z.floor() as i32;
    if x < 0 || y < 0 || z < 0 {
        return false;
    }
    let (x, y, z) = (x as u32, y as u32, z as u32);
    if x >= mask.dims[0] || y >= mask.dims[1] || z >= mask.dims[2] {
        return false;
    }
    let idx = mask.lin_idx(x, y, z);
    mask.data.get(idx).copied().unwrap_or(0) != 0
}

/// Does the streamline pass through **every** waypoint mask? AND-semantics.
pub(crate) fn streamline_hits_all_rois(
    streamline: &[[f32; 3]],
    rois: &[Arc<VoxelMask>],
) -> bool {
    rois.iter().all(|roi| {
        streamline
            .iter()
            .any(|p| point_in_mask(Vec3::from_array(*p), roi))
    })
}

/// Is either endpoint of the streamline inside `mask`?
pub(crate) fn streamline_endpoint_in(streamline: &[[f32; 3]], mask: &VoxelMask) -> bool {
    if streamline.len() < 2 {
        return false;
    }
    let first = Vec3::from_array(streamline[0]);
    let last = Vec3::from_array(*streamline.last().unwrap());
    point_in_mask(first, mask) || point_in_mask(last, mask)
}

/// End-region rule (simplified): at least one endpoint must lie in **each**
/// mask in `end_masks`.
pub(crate) fn streamline_satisfies_end_masks(
    streamline: &[[f32; 3]],
    end_masks: &[Arc<VoxelMask>],
) -> bool {
    if streamline.len() < 2 {
        return false;
    }
    let first = Vec3::from_array(streamline[0]);
    let last = Vec3::from_array(*streamline.last().unwrap());
    end_masks
        .iter()
        .all(|m| point_in_mask(first, m) || point_in_mask(last, m))
}

/// Mean-min Hausdorff test: each candidate point's minimum distance to the
/// reference cloud, averaged over the streamline; pass iff mean ≤ `max_mm`.
pub(crate) fn streamline_passes_hausdorff(
    streamline: &[[f32; 3]],
    reference_points: &[[f32; 3]],
    max_mm: f32,
) -> bool {
    if reference_points.is_empty() || streamline.is_empty() {
        return true;
    }
    let mut acc = 0.0f32;
    for p in streamline {
        let p = Vec3::from_array(*p);
        let mut min_d2 = f32::INFINITY;
        for q in reference_points {
            let d2 = (Vec3::from_array(*q) - p).length_squared();
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        acc += min_d2.sqrt();
    }
    let mean = acc / streamline.len() as f32;
    mean <= max_mm
}

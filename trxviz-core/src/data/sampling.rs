//! Voxel sampling kernels for `VolumeScalars`.
//!
//! Both kernels accept a *voxel-space* point (after the caller has
//! applied `ras_to_voxel`) and return `None` for out-of-bounds samples
//! so callers can treat the volume as transparent outside its grid.

use glam::Vec3;

use crate::data::cifti::VolumeScalars;

/// Trilinear (8-tap) sample. `voxel` is a voxel-space coordinate.
/// Returns `None` if the 8-neighbor cube would step outside the
/// volume.
pub fn trilinear(scalars: &VolumeScalars, voxel: Vec3) -> Option<f32> {
    let (dx, dy, dz) = (
        scalars.dims[0] as i32,
        scalars.dims[1] as i32,
        scalars.dims[2] as i32,
    );
    let x0 = voxel.x.floor() as i32;
    let y0 = voxel.y.floor() as i32;
    let z0 = voxel.z.floor() as i32;
    if x0 < 0 || y0 < 0 || z0 < 0 || x0 + 1 >= dx || y0 + 1 >= dy || z0 + 1 >= dz {
        return None;
    }
    let fx = voxel.x - x0 as f32;
    let fy = voxel.y - y0 as f32;
    let fz = voxel.z - z0 as f32;
    let dxs = scalars.dims[0];
    let dys = scalars.dims[1];
    let idx =
        |x: i32, y: i32, z: i32| -> usize { x as usize + dxs * (y as usize + dys * z as usize) };
    let v000 = scalars.values[idx(x0, y0, z0)];
    let v100 = scalars.values[idx(x0 + 1, y0, z0)];
    let v010 = scalars.values[idx(x0, y0 + 1, z0)];
    let v110 = scalars.values[idx(x0 + 1, y0 + 1, z0)];
    let v001 = scalars.values[idx(x0, y0, z0 + 1)];
    let v101 = scalars.values[idx(x0 + 1, y0, z0 + 1)];
    let v011 = scalars.values[idx(x0, y0 + 1, z0 + 1)];
    let v111 = scalars.values[idx(x0 + 1, y0 + 1, z0 + 1)];
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c00 = lerp(v000, v100, fx);
    let c10 = lerp(v010, v110, fx);
    let c01 = lerp(v001, v101, fx);
    let c11 = lerp(v011, v111, fx);
    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    Some(lerp(c0, c1, fz))
}

/// Nearest-neighbor sample. `voxel` is a voxel-space coordinate.
/// Returns `None` if the rounded index falls outside the volume.
pub fn nearest(scalars: &VolumeScalars, voxel: Vec3) -> Option<f32> {
    let (dx, dy, dz) = (
        scalars.dims[0] as i32,
        scalars.dims[1] as i32,
        scalars.dims[2] as i32,
    );
    let x = voxel.x.round() as i32;
    let y = voxel.y.round() as i32;
    let z = voxel.z.round() as i32;
    if x < 0 || y < 0 || z < 0 || x >= dx || y >= dy || z >= dz {
        return None;
    }
    let dxs = scalars.dims[0];
    let dys = scalars.dims[1];
    let idx = x as usize + dxs * (y as usize + dys * z as usize);
    Some(scalars.values[idx])
}

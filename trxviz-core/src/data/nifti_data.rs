use std::path::Path;

use glam::{Mat4, Vec3, Vec4};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

/// Convert NIfTI qform quaternion parameters to a 4x4 voxel-to-RAS affine.
/// Matches trx-cpp's `quatern_to_mat44` exactly.
fn quatern_to_mat44(
    qb: f32,
    qc: f32,
    qd: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    qfac: f32,
) -> Mat4 {
    let mut a = 1.0f64 - (qb * qb + qc * qc + qd * qd) as f64;
    let mut b = qb as f64;
    let mut c = qc as f64;
    let mut d = qd as f64;

    if a < 1e-7 {
        let s = 1.0 / (b * b + c * c + d * d).sqrt();
        a = 0.0;
        b *= s;
        c *= s;
        d *= s;
    } else {
        a = a.sqrt();
    }

    let xd = if dx > 0.0 { dx as f64 } else { 1.0 };
    let yd = if dy > 0.0 { dy as f64 } else { 1.0 };
    let zd_abs = if dz > 0.0 { dz as f64 } else { 1.0 };
    let zd = if qfac < 0.0 { -zd_abs } else { zd_abs };

    let r00 = ((a * a + b * b - c * c - d * d) * xd) as f32;
    let r01 = ((2.0 * (b * c - a * d)) * yd) as f32;
    let r02 = ((2.0 * (b * d + a * c)) * zd) as f32;
    let r10 = ((2.0 * (b * c + a * d)) * xd) as f32;
    let r11 = ((a * a + c * c - b * b - d * d) * yd) as f32;
    let r12 = ((2.0 * (c * d - a * b)) * zd) as f32;
    let r20 = ((2.0 * (b * d - a * c)) * xd) as f32;
    let r21 = ((2.0 * (c * d + a * b)) * yd) as f32;
    let r22 = ((a * a + d * d - c * c - b * b) * zd) as f32;

    // Row-major form → glam column-major
    Mat4::from_cols(
        Vec4::new(r00, r10, r20, 0.0),
        Vec4::new(r01, r11, r21, 0.0),
        Vec4::new(r02, r12, r22, 0.0),
        Vec4::new(qx, qy, qz, 1.0),
    )
}

/// Voxel axis and world axis of a view's slices: axial slices step along `k`
/// and sit at a world z, coronal along `j` at a y, sagittal along `i` at an x.
fn slice_axes(axis_index: usize) -> (usize, usize) {
    match axis_index {
        0 => (2, 2),
        1 => (1, 1),
        _ => (0, 0),
    }
}

/// The middle voxel of the volume, moved to `index` along `voxel_axis`.
fn centre_voxel(dims: [usize; 3], voxel_axis: usize, index: f32) -> Vec3 {
    let mut voxel = Vec3::new(
        dims[0].saturating_sub(1) as f32 * 0.5,
        dims[1].saturating_sub(1) as f32 * 0.5,
        dims[2].saturating_sub(1) as f32 * 0.5,
    );
    voxel[voxel_axis] = index;
    voxel
}

/// World position of slice `index` along the view axis (z for axial, y for
/// coronal, x for sagittal), taken where the slice crosses the line of voxel
/// centres through the middle of the volume. [`nearest_slice_index`] inverts
/// it exactly. Both use the same point because on an oblique grid a slice's
/// world z changes across the slice.
pub fn slice_world_position(
    voxel_to_ras: &Mat4,
    dims: [usize; 3],
    axis_index: usize,
    index: usize,
) -> f32 {
    let (voxel_axis, world_axis) = slice_axes(axis_index);
    voxel_to_ras.transform_point3(centre_voxel(dims, voxel_axis, index as f32))[world_axis]
}

/// The slice whose [`slice_world_position`] is nearest `world_pos`, clamped to
/// the volume.
pub fn nearest_slice_index(
    voxel_to_ras: &Mat4,
    dims: [usize; 3],
    axis_index: usize,
    world_pos: f32,
) -> usize {
    let (voxel_axis, world_axis) = slice_axes(axis_index);
    let base = voxel_to_ras.transform_point3(centre_voxel(dims, voxel_axis, 0.0))[world_axis];
    let step = voxel_to_ras.col(voxel_axis)[world_axis];
    let coord = if step.abs() > 1e-6 {
        (world_pos - base) / step
    } else {
        // The slice axis lies in the view plane, so no world position picks
        // a slice; keep the middle one.
        dims[voxel_axis].saturating_sub(1) as f32 * 0.5
    };
    let max = dims[voxel_axis].saturating_sub(1) as isize;
    (coord.round() as isize).clamp(0, max) as usize
}

/// NIfTI volume data ready for GPU upload.
pub struct NiftiVolume {
    /// 3D voxel data normalized to `[0, 1]` as `f32`, stored in `\[i]\[j]\[k]` order.
    pub data: Vec<f32>,
    /// Volume dimensions (i, j, k).
    pub dims: [usize; 3],
    /// Voxel-to-RAS+ affine (4x4, column-major for glam).
    pub voxel_to_ras: Mat4,
}

impl NiftiVolume {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let obj = ReaderOptions::new().read_file(path)?;
        let header = obj.header();

        // Build voxel-to-RAS affine using trx-cpp priority: qform first, then sform.
        // This matches how trx-cpp's read_nifti_voxel_to_rasmm() works.
        let voxel_to_ras = if header.qform_code > 0 {
            let qfac = if header.pixdim[0] < 0.0 {
                -1.0f32
            } else {
                1.0f32
            };
            quatern_to_mat44(
                header.quatern_b,
                header.quatern_c,
                header.quatern_d,
                header.quatern_x,
                header.quatern_y,
                header.quatern_z,
                header.pixdim[1],
                header.pixdim[2],
                header.pixdim[3],
                qfac,
            )
        } else if header.sform_code > 0 {
            // srow_x/y/z are rows of the 4x4 affine (row-major) → convert to column-major
            let sx = header.srow_x;
            let sy = header.srow_y;
            let sz = header.srow_z;
            Mat4::from_cols(
                Vec4::new(sx[0], sy[0], sz[0], 0.0),
                Vec4::new(sx[1], sy[1], sz[1], 0.0),
                Vec4::new(sx[2], sy[2], sz[2], 0.0),
                Vec4::new(sx[3], sy[3], sz[3], 1.0),
            )
        } else {
            return Err(anyhow::anyhow!(
                "NIfTI header has neither valid qform nor sform code"
            ));
        };

        // Get volume dimensions from header dim array
        let dims = [
            header.dim[1] as usize,
            header.dim[2] as usize,
            header.dim[3] as usize,
        ];

        // Convert volume to f32 ndarray
        let volume = obj.into_volume();
        let array = volume.into_ndarray::<f32>()?;

        // Find intensity range
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in array.iter() {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        // Normalize to [0, 1].
        // IMPORTANT: Use as_slice_memory_order() (Fortran/i-fastest order) NOT .iter()
        // (which gives C/k-fastest order). The GPU texture upload uses bytes_per_row = ni*4,
        // which expects i to vary fastest in memory. nifti-rs creates a Fortran-ordered
        // ndarray, so its memory order matches the GPU layout.
        let range = max_val - min_val;
        let raw = array
            .as_slice_memory_order()
            .expect("nifti-rs array is always contiguous");
        let data: Vec<f32> = if range > 0.0 {
            raw.iter()
                .map(|&v| ((v - min_val) / range).clamp(0.0, 1.0))
                .collect()
        } else {
            vec![0.0; raw.len()]
        };

        Ok(Self {
            data,
            dims,
            voxel_to_ras,
        })
    }

    /// Compute the half-extents (in world-space mm) for each slice camera axis so the
    /// slice quad fills almost the entire viewport.  Returns [axial, coronal, sagittal].
    pub fn slice_half_extents(&self) -> [f32; 3] {
        fn quad_half_extent(
            corners: &[Vec3; 4],
            a: impl Fn(&Vec3) -> f32,
            b: impl Fn(&Vec3) -> f32,
        ) -> f32 {
            let ca = corners.iter().map(|c| a(c)).sum::<f32>() / 4.0;
            let cb = corners.iter().map(|c| b(c)).sum::<f32>() / 4.0;
            let ha = corners
                .iter()
                .map(|c| (a(c) - ca).abs())
                .fold(0f32, f32::max);
            let hb = corners
                .iter()
                .map(|c| (b(c) - cb).abs())
                .fold(0f32, f32::max);
            ha.max(hb) * 1.05
        }
        let mid_k = self.dims[2] / 2;
        let mid_j = self.dims[1] / 2;
        let mid_i = self.dims[0] / 2;
        [
            quad_half_extent(&self.axial_slice_corners(mid_k), |c| c.x, |c| c.y),
            quad_half_extent(&self.coronal_slice_corners(mid_j), |c| c.x, |c| c.z),
            quad_half_extent(&self.sagittal_slice_corners(mid_i), |c| c.y, |c| c.z),
        ]
    }

    /// Transform a voxel coordinate to RAS+ world space.
    pub fn voxel_to_world(&self, voxel: Vec3) -> Vec3 {
        let v = self.voxel_to_ras * voxel.extend(1.0);
        Vec3::new(v.x, v.y, v.z)
    }

    /// See the free function [`slice_world_position`].
    pub fn slice_world_position(&self, axis_index: usize, index: usize) -> f32 {
        slice_world_position(&self.voxel_to_ras, self.dims, axis_index, index)
    }

    /// See the free function [`nearest_slice_index`].
    pub fn nearest_slice_index(&self, axis_index: usize, world_pos: f32) -> usize {
        nearest_slice_index(&self.voxel_to_ras, self.dims, axis_index, world_pos)
    }

    /// Compute the 4 corner positions (in RAS+ space) of an axial slice at voxel index k.
    ///
    /// The NIfTI affine maps integer voxel indices to the *center* of each voxel.
    /// So the outer edges of the slice quad span from voxel -0.5 to dim-0.5 in
    /// each in-plane axis, ensuring the quad covers the full extent of all voxels.
    /// The slice itself sits at voxel center k (not offset).
    pub fn axial_slice_corners(&self, k: usize) -> [Vec3; 4] {
        let kf = k as f32;
        let i0 = -0.5;
        let i1 = self.dims[0] as f32 - 0.5;
        let j0 = -0.5;
        let j1 = self.dims[1] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(i0, j0, kf)),
            self.voxel_to_world(Vec3::new(i1, j0, kf)),
            self.voxel_to_world(Vec3::new(i1, j1, kf)),
            self.voxel_to_world(Vec3::new(i0, j1, kf)),
        ]
    }

    /// Compute the 4 corner positions of a coronal slice at voxel index j.
    pub fn coronal_slice_corners(&self, j: usize) -> [Vec3; 4] {
        let jf = j as f32;
        let i0 = -0.5;
        let i1 = self.dims[0] as f32 - 0.5;
        let k0 = -0.5;
        let k1 = self.dims[2] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(i0, jf, k0)),
            self.voxel_to_world(Vec3::new(i1, jf, k0)),
            self.voxel_to_world(Vec3::new(i1, jf, k1)),
            self.voxel_to_world(Vec3::new(i0, jf, k1)),
        ]
    }

    /// Compute the 4 corner positions of a sagittal slice at voxel index i.
    pub fn sagittal_slice_corners(&self, i: usize) -> [Vec3; 4] {
        let if_ = i as f32;
        let j0 = -0.5;
        let j1 = self.dims[1] as f32 - 0.5;
        let k0 = -0.5;
        let k1 = self.dims[2] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(if_, j0, k0)),
            self.voxel_to_world(Vec3::new(if_, j1, k0)),
            self.voxel_to_world(Vec3::new(if_, j1, k1)),
            self.voxel_to_world(Vec3::new(if_, j0, k1)),
        ]
    }

    /// Return `(normal, center)` for the slice plane at the given voxel index.
    ///
    /// `axis_index`: 0 = axial (k-slices), 1 = coronal (j-slices), 2 = sagittal (i-slices).
    ///
    /// `normal` is the unit normal to the slice plane in RAS+ space.  For an
    /// axis-aligned volume it is exactly one of ±X/±Y/±Z; for oblique volumes
    /// it follows the affine column direction.
    ///
    /// `center` is the world-space centre of the slice quad (mid-i, mid-j or mid-k
    /// for the two in-plane axes, `slice_index` for the slice axis).
    pub fn slice_plane(&self, axis_index: usize, slice_index: usize) -> (Vec3, Vec3) {
        // The column of the affine that corresponds to the varying voxel axis.
        let col = match axis_index {
            0 => self.voxel_to_ras.col(2), // k-column
            1 => self.voxel_to_ras.col(1), // j-column
            _ => self.voxel_to_ras.col(0), // i-column
        };
        let normal = Vec3::new(col.x, col.y, col.z).normalize();

        let ci = self.dims[0] as f32 / 2.0;
        let cj = self.dims[1] as f32 / 2.0;
        let ck = self.dims[2] as f32 / 2.0;
        let si = slice_index as f32;
        let center_voxel = match axis_index {
            0 => Vec3::new(ci, cj, si),
            1 => Vec3::new(ci, si, ck),
            _ => Vec3::new(si, cj, ck),
        };
        (normal, self.voxel_to_world(center_voxel))
    }
}

#[cfg(test)]
mod tests {
    use glam::{EulerRot, Mat4, Quat, Vec3, Vec4};

    use super::{nearest_slice_index, slice_world_position};

    /// Like odf-testdata's `slab`: 1.5 mm voxels stored LAS, the grid rotated
    /// by (12, -7, 5) degrees, 10 axial slices.
    fn oblique_slab() -> (Mat4, [usize; 3]) {
        let rotation = Quat::from_euler(
            EulerRot::XYZ,
            12f32.to_radians(),
            -7f32.to_radians(),
            5f32.to_radians(),
        );
        let affine = Mat4::from_rotation_translation(rotation, Vec3::new(70.0, -95.0, 20.0))
            * Mat4::from_scale(Vec3::new(-1.5, 1.5, 1.5));
        (affine, [94, 126, 10])
    }

    #[test]
    fn slice_positions_round_trip_on_an_oblique_grid() {
        let (affine, dims) = oblique_slab();
        for (axis, n) in [(0, dims[2]), (1, dims[1]), (2, dims[0])] {
            for index in 0..n {
                let world = slice_world_position(&affine, dims, axis, index);
                assert_eq!(nearest_slice_index(&affine, dims, axis, world), index);
            }
        }
    }

    #[test]
    fn oblique_axial_slices_line_up_across_storage_orders() {
        // The same grid with j reversed, as DSI Studio stores an LAS image.
        let (las, dims) = oblique_slab();
        let flip_j = Mat4::from_cols(
            Vec4::X,
            -Vec4::Y,
            Vec4::Z,
            Vec4::new(0.0, (dims[1] - 1) as f32, 0.0, 1.0),
        );
        let lps = las * flip_j;
        for k in 0..dims[2] {
            let world = slice_world_position(&las, dims, 0, k);
            assert_eq!(nearest_slice_index(&lps, dims, 0, world), k);
        }
    }

    #[test]
    fn axis_aligned_slices_sit_at_their_world_coordinate() {
        let affine = Mat4::from_cols(
            Vec4::new(-2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(90.0, -126.0, -72.0, 1.0),
        );
        let dims = [91, 109, 91];
        assert_eq!(slice_world_position(&affine, dims, 0, 5), -62.0);
        assert_eq!(slice_world_position(&affine, dims, 2, 3), 84.0);
        assert_eq!(nearest_slice_index(&affine, dims, 0, -61.6), 5);
        assert_eq!(nearest_slice_index(&affine, dims, 2, 84.9), 3);
        assert_eq!(nearest_slice_index(&affine, dims, 0, 1000.0), 90);
    }
}

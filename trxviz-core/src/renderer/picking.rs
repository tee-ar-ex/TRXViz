//! Screen → world picking for the value probe. glam-only (no egui): the
//! GUI converts cursor positions to NDC before calling in.
//!
//! Axis-index convention (matches the GUI, `NiftiVolume::slice_plane`, and
//! `OdxScene::slice_plane`): `0 = axial (k-slices)`, `1 = coronal
//! (j-slices)`, `2 = sagittal (i-slices)`. Note this is the *opposite* of
//! `OdxScene::glyphs_for_slice` / `fixels_for_slice` (0 = i, 1 = j,
//! 2 = k); probe data lookups key on `ijk` so no conversion is needed —
//! only "is this voxel on the rendered glyph slice" checks go through
//! `WorkflowSliceViewKind::odx_axis()`.

use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};

/// A world-space ray through the cursor. For orthographic projections the
/// direction is the constant view direction and `origin` slides with the
/// cursor, so the same plane-intersection code serves both camera kinds.
#[derive(Clone, Copy, Debug)]
pub struct PickRay {
    pub origin: Vec3,
    pub dir: Vec3,
}

/// Where a pick landed: a world position on the slice plane of
/// `axis_index` (GUI convention, see module docs).
#[derive(Clone, Copy, Debug)]
pub struct ProbeHit {
    pub world: Vec3,
    pub axis_index: usize,
}

/// Unproject a cursor NDC position (x right, y up, both in [-1, 1]) into a
/// world-space ray using the inverse view-projection. Returns `None` for a
/// singular matrix or degenerate unprojection.
pub fn pick_ray(view_proj: Mat4, ndc: Vec2) -> Option<PickRay> {
    let inv = view_proj.inverse();
    if !inv.is_finite() {
        return None;
    }
    // wgpu clip space: z in [0, 1].
    let near = inv * ndc.extend(0.0).extend(1.0);
    let far = inv * ndc.extend(1.0).extend(1.0);
    if near.w.abs() < f32::EPSILON || far.w.abs() < f32::EPSILON {
        return None;
    }
    let near = near.xyz() / near.w;
    let far = far.xyz() / far.w;
    let dir = far - near;
    let len = dir.length();
    if !len.is_finite() || len < f32::EPSILON {
        return None;
    }
    Some(PickRay {
        origin: near,
        dir: dir / len,
    })
}

/// Intersect a ray with the plane through `point` with unit `normal`.
/// Returns the intersection point, or `None` when the ray is (near)
/// parallel to the plane or the hit lies behind the origin.
pub fn ray_plane_intersect(ray: &PickRay, normal: Vec3, point: Vec3) -> Option<Vec3> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - ray.origin).dot(normal) / denom;
    if !t.is_finite() || t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}

/// Forward transform: world position to NDC (x right, y up). `None` when
/// the point is behind the camera (w <= 0). The inverse pair of
/// [`pick_ray`]; also the shared core for the GUI's world→screen overlay
/// projections.
pub fn project_world_to_ndc(view_proj: Mat4, world: Vec3) -> Option<Vec2> {
    let clip = view_proj * world.extend(1.0);
    if clip.w <= f32::EPSILON {
        return None;
    }
    Some(Vec2::new(clip.x / clip.w, clip.y / clip.w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
    use crate::renderer::slice_renderer::SliceAxis;

    #[test]
    fn ortho_pick_round_trip() {
        let camera = OrthoSliceCamera::new(SliceAxis::Axial, Vec3::new(1.0, -2.0, 0.0), 200.0);
        let vp = camera.view_projection(1.5, 12.0);
        // A world point on the slice plane (z = slice position).
        let world = Vec3::new(7.0, -3.0, 12.0);
        let ndc = project_world_to_ndc(vp, world).expect("in front");
        let ray = pick_ray(vp, ndc).expect("ray");
        // The ortho ray is parallel to the slice normal (±Z for axial).
        assert!(ray.dir.x.abs() < 1e-4 && ray.dir.y.abs() < 1e-4);
        let hit = ray_plane_intersect(&ray, Vec3::Z, Vec3::new(0.0, 0.0, 12.0)).expect("hit");
        assert!((hit - world).length() < 1e-3, "{hit:?} vs {world:?}");
    }

    #[test]
    fn perspective_center_ray_hits_camera_axis() {
        let camera = OrbitCamera::new(Vec3::new(5.0, -8.0, 20.0), 150.0);
        let vp = camera.view_projection(1.0);
        let ray = pick_ray(vp, Vec2::ZERO).expect("ray");
        // The NDC-center ray runs from the eye toward the orbit center.
        let expected = (camera.center - camera.eye()).normalize();
        assert!((ray.dir - expected).length() < 1e-3, "{ray:?}");
        // And intersecting the plane through the center perpendicular to
        // the view recovers the center.
        let hit = ray_plane_intersect(&ray, expected, camera.center).expect("hit");
        // f32 inverse-perspective at distance 150 with far=10000 leaves
        // ~1e-4 direction noise → ~1.5e-2 mm at the hit; allow 0.1 mm.
        assert!((hit - camera.center).length() < 0.1);
    }

    #[test]
    fn ray_plane_oblique_and_degenerate() {
        let ray = PickRay {
            origin: Vec3::ZERO,
            dir: Vec3::new(1.0, 1.0, 1.0).normalize(),
        };
        let normal = Vec3::new(0.3, -0.5, 0.81).normalize();
        let point = Vec3::new(4.0, 2.0, 6.0);
        let hit = ray_plane_intersect(&ray, normal, point).expect("hit");
        assert!((hit - point).dot(normal).abs() < 1e-4);
        // Parallel ray misses.
        let parallel = PickRay {
            origin: Vec3::ZERO,
            dir: normal.cross(Vec3::X).normalize(),
        };
        assert!(ray_plane_intersect(&parallel, normal, point + normal).is_none());
        // Behind-origin hit is rejected.
        let away = PickRay {
            origin: point + normal,
            dir: normal,
        };
        assert!(ray_plane_intersect(&away, normal, point).is_none());
    }
}

//! Per-streamline u-fiber **triangle** features from a parabola fit.
//!
//! Each streamline is abstracted to: two cortical endpoints `e1`/`e2`,
//! the deepest "apex" of the U, and the unit normal of the U-plane.
//! The fit averages per-vertex sampling noise across the whole
//! polyline rather than picking three raw sample points:
//!
//! 1. PCA on the streamline points (covariance + a self-contained
//!    Jacobi symmetric-3×3 eigensolver — no nalgebra) → least-squares
//!    best-fit plane; smallest-eigenvalue eigenvector is the U-plane
//!    normal.
//! 2. In-plane axes `(u, v)`: `u` = chord (last − first) projected
//!    onto the plane; `v` ⟂ `u` in-plane, sign-flipped so the
//!    streamline body sits on +v.
//! 3. Least-squares parabola `v = a·u² + b·u + c`; the vertex is the
//!    clean apex.
//! 4. `e1`/`e2` = the raw first/last points snapped onto the parabola.
//!
//! Keep in sync with `ufixels/src/triangle.rs::fit_streamline_parabola`
//! (same algorithm; this copy is glam-only so TRXViz stays independent
//! of the ufixels research crate).

use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct FundusTriangle {
    pub apex: Vec3,
    pub e1: Vec3,
    pub e2: Vec3,
    /// Unit normal of the U-plane, sign-canonicalised via
    /// `(e2 - e1) × (apex - midchord)`.
    pub plane_normal: Vec3,
}

/// Fit a u-fiber triangle to a streamline. Returns `None` for
/// streamlines with `< 4` points or a degenerate fit (no plane, no
/// chord, or a near-flat parabola).
pub fn fit_streamline_triangle(points: &[Vec3]) -> Option<FundusTriangle> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f32;
    let raw_e1 = points[0];
    let raw_e2 = points[points.len() - 1];

    // Centroid.
    let mut centroid = Vec3::ZERO;
    for p in points {
        centroid += *p;
    }
    centroid /= n;

    // 3×3 covariance of centred points (symmetric), accumulated in
    // f64 so the (near-zero) plane-normal eigenvalue is resolved
    // cleanly.
    let mut cov = [[0.0_f64; 3]; 3];
    for p in points {
        let d = *p - centroid;
        let da = [d.x as f64, d.y as f64, d.z as f64];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += da[i] * da[j];
            }
        }
    }
    let (eigvals, eigvecs) = jacobi_eigen_symmetric_3x3(cov);
    // Smallest-eigenvalue eigenvector = plane normal.
    let mut idx_min = 0usize;
    for i in 1..3 {
        if eigvals[i] < eigvals[idx_min] {
            idx_min = i;
        }
    }
    let plane_n = Vec3::new(
        eigvecs[0][idx_min],
        eigvecs[1][idx_min],
        eigvecs[2][idx_min],
    );
    if plane_n.length_squared() < 1e-12 {
        return None;
    }
    let plane_n = plane_n.normalize();

    // u_axis = chord projected onto the plane.
    let raw_chord = raw_e2 - raw_e1;
    let u_raw = raw_chord - plane_n * raw_chord.dot(plane_n);
    if u_raw.length_squared() < 1e-12 {
        return None;
    }
    let u_axis = u_raw.normalize();
    let v_raw = plane_n.cross(u_axis);
    if v_raw.length_squared() < 1e-12 {
        return None;
    }
    let mut v_axis = v_raw.normalize();
    // Sign v so the streamline midpoint sits on +v.
    let mid_pt = points[points.len() / 2];
    if (mid_pt - centroid).dot(v_axis) < 0.0 {
        v_axis = -v_axis;
    }

    // Project to (u, v), accumulate least-squares sums, and track the
    // streamline's actual extent along u so we never extrapolate the
    // parabola past where the data exists.
    let (mut s_u, mut s_u2, mut s_u3, mut s_u4) = (0.0, 0.0, 0.0, 0.0);
    let (mut s_v, mut s_uv, mut s_u2v) = (0.0, 0.0, 0.0);
    let mut u_min = f32::INFINITY;
    let mut u_max = f32::NEG_INFINITY;
    for p in points {
        let d = *p - centroid;
        let u = d.dot(u_axis);
        let v = d.dot(v_axis);
        if u < u_min {
            u_min = u;
        }
        if u > u_max {
            u_max = u;
        }
        let u2 = u * u;
        let u3 = u2 * u;
        s_u += u;
        s_u2 += u2;
        s_u3 += u3;
        s_u4 += u3 * u;
        s_v += v;
        s_uv += u * v;
        s_u2v += u2 * v;
    }
    // Solve M·[a,b,c]ᵀ = [Σu²v, Σuv, Σv]ᵀ,
    // M = [[Σu⁴,Σu³,Σu²],[Σu³,Σu²,Σu],[Σu²,Σu,n]].
    let m = [
        [s_u4, s_u3, s_u2],
        [s_u3, s_u2, s_u],
        [s_u2, s_u, n],
    ];
    let rhs = [s_u2v, s_uv, s_v];
    let coef = solve_3x3(m, rhs)?;
    let (a, b, c) = (coef[0], coef[1], coef[2]);
    if a.abs() < 1e-6 {
        return None; // near-flat: not a real U
    }

    let to_world = |u: f32, v: f32| centroid + u_axis * u + v_axis * v;
    // Parabola vertex → apex. For gentle / non-U arcs the analytic
    // vertex `-b/2a` lies far outside the sampled range (it flies off
    // screen), so clamp it to the streamline's actual u-extent: the
    // apex must sit on the streamline, not on its extrapolation.
    let u_vertex = (-b / (2.0 * a)).clamp(u_min, u_max);
    let v_vertex = a * u_vertex * u_vertex + b * u_vertex + c;
    let apex = to_world(u_vertex, v_vertex);
    // Snap raw endpoints onto the parabola (keep u within range,
    // predict v).
    let snap = |raw: Vec3| {
        let d = raw - centroid;
        let u = d.dot(u_axis).clamp(u_min, u_max);
        to_world(u, a * u * u + b * u + c)
    };
    let e1 = snap(raw_e1);
    let e2 = snap(raw_e2);

    let chord_vec = e2 - e1;
    if chord_vec.length_squared() < 1e-12 {
        return None;
    }
    let mid = (e1 + e2) * 0.5;
    let arm = apex - mid;
    let n_cross = chord_vec.cross(arm);
    if n_cross.length_squared() < 1e-12 {
        return None;
    }
    let plane_normal = n_cross.normalize();

    Some(FundusTriangle {
        apex,
        e1,
        e2,
        plane_normal,
    })
}

/// Cyclic Jacobi eigendecomposition of a symmetric 3×3 matrix
/// (classic Rutishauser formulation — numerically robust, converges
/// in a few sweeps for 3×3). Returns (eigenvalues,
/// eigenvectors-as-columns of `v`).
fn jacobi_eigen_symmetric_3x3(mut a: [[f64; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [
        [1.0_f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for _sweep in 0..24 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-15 {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // Symmetric update of the active 2×2 block.
            let app = a[p][p];
            let aqq = a[q][q];
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            // Rotate the remaining off-block entries.
            for k in 0..3 {
                if k == p || k == q {
                    continue;
                }
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
            // Accumulate eigenvectors (columns of v).
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    (
        [a[0][0] as f32, a[1][1] as f32, a[2][2] as f32],
        [
            [v[0][0] as f32, v[0][1] as f32, v[0][2] as f32],
            [v[1][0] as f32, v[1][1] as f32, v[1][2] as f32],
            [v[2][0] as f32, v[2][1] as f32, v[2][2] as f32],
        ],
    )
}

/// Solve a 3×3 linear system by Cramer's rule. `None` if singular.
fn solve_3x3(m: [[f32; 3]; 3], r: [f32; 3]) -> Option<[f32; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let col = |c: usize| {
        let mut mm = m;
        for row in 0..3 {
            mm[row][c] = r[row];
        }
        (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
            - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
            + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]))
            * inv_det
    };
    Some([col(0), col(1), col(2)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_parabola_in_tilted_plane() {
        // Symmetric parabola v = u² in a tilted plane, offset from
        // origin. Endpoints are at u = ±2 (exactly representable,
        // symmetric) so the chord is exactly along `ua` and the
        // geometric apex is the vertex at u=0,v=0 → world `origin`.
        let ua = Vec3::new(1.0, 0.0, 0.0);
        let va = Vec3::new(0.0, 0.6, 0.8); // unit, not axis-aligned
        let origin = Vec3::new(5.0, -2.0, 3.0);
        let us = [
            -2.0_f32, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0,
        ];
        let pts: Vec<Vec3> = us
            .iter()
            .map(|&u| origin + ua * u + va * (u * u))
            .collect();
        let t = fit_streamline_triangle(&pts).expect("fit");
        assert!(
            (t.apex - origin).length() < 1e-2,
            "apex {:?} vs {:?}",
            t.apex,
            origin
        );
        let n_true = ua.cross(va).normalize();
        assert!(
            t.plane_normal.dot(n_true).abs() > 0.999,
            "normal {:?} vs {:?}",
            t.plane_normal,
            n_true
        );
    }

    #[test]
    fn rejects_straight_line() {
        let pts: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        assert!(fit_streamline_triangle(&pts).is_none());
    }

    #[test]
    fn rejects_too_short() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(fit_streamline_triangle(&pts).is_none());
    }

    #[test]
    fn gentle_asymmetric_arc_apex_stays_in_range() {
        // Gentle, asymmetric arc: tiny curvature + a linear ramp, so
        // the analytic parabola vertex (-b/2a) lands far outside the
        // sampled u-range. The apex must be clamped onto the data,
        // not flung off into extrapolation.
        let pts: Vec<Vec3> = (0..=20)
            .map(|i| {
                let u = i as f32 * 0.5; // u in [0, 10]
                Vec3::new(u, 0.005 * u * u + 0.3 * u, 0.4)
            })
            .collect();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        if let Some(t) = fit_streamline_triangle(&pts) {
            let eps = 1.0; // mm slack
            for v in [t.apex, t.e1, t.e2] {
                assert!(
                    v.cmpge(lo - eps).all() && v.cmple(hi + eps).all(),
                    "vertex {:?} escaped streamline bbox [{:?},{:?}]",
                    v,
                    lo,
                    hi
                );
            }
        }
    }
}

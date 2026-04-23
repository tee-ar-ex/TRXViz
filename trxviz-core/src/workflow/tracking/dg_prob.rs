//! Dipy-style probabilistic direction getter.
//!
//! At each step: trilinearly interpolate SH coefficients, evaluate on a
//! fixed sphere to get a PMF, zero entries outside the angular cone,
//! apply the relative-peak threshold, renormalize, and sample one vertex
//! weighted by the PMF. The sphere vertices and their SH transform live
//! once in `DipyProbGlobal`; the per-attempt `DipyProbDG` just carries the
//! step + angular knobs.

use glam::{Mat4, Vec3};

use super::direction_getter::DirectionGetter;
use super::rng::lcg_f32;

/// Lifetime-of-run state shared by every Dipy-prob attempt.
///
/// The SH buffer is a flat `Vec<f32>` of `(nb_voxels, ncoeffs)` (row-major),
/// stored sparse: row `v` is the SH vector of compact voxel `v`. The
/// `sample_plan` is an odx-rs `RowSamplePlan` — a precomputed mapping
/// from SH → sphere amplitudes, cached so `sample_direction` is a single
/// matrix-vector product per voxel.
pub struct DipyProbGlobal<'a> {
    pub sh_flat: &'a [f32],
    pub ncoeffs: usize,
    pub dense_lut: &'a [usize],
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub ras_to_vox: Mat4,
    /// Per-voxel GFA (or whatever stopping metric the caller computed) to
    /// gate the trilinear interp: if any of the 8 corners dips below
    /// `fixel_threshold`, that corner contributes 0 (and the whole step
    /// returns None if no corners survive).
    pub gfa_data: &'a [f32],
    pub fixel_threshold: f32,
    pub sample_plan: &'a odx_rs::mrtrix_sh::RowSamplePlan,
    pub n_dirs: usize,
    pub sphere_verts: &'a [[f32; 3]],
    /// Relative PMF threshold: any sphere vertex whose PMF value is
    /// below `relative_peak_threshold * max(PMF)` is zeroed before
    /// sampling.
    pub relative_peak_threshold: f32,
}

/// Per-thread scratch: PMF-on-sphere buffer, sized to `n_dirs`. Reused
/// across every step of every attempt on the thread, so PMF evaluation
/// does no allocation after the first step.
#[derive(Default)]
pub struct DipyProbScratch {
    pub pmf: Vec<f32>,
}

/// Per-attempt direction getter.
pub struct DipyProbDG<'a> {
    pub global: &'a DipyProbGlobal<'a>,
    /// `cos(max_angle_deg)`. Sphere vertices whose absolute dot product
    /// with the previous direction falls below this are zeroed in the
    /// PMF before sampling.
    pub cos_max: f32,
    pub step_mm: f32,
}

impl<'a> DirectionGetter for DipyProbDG<'a> {
    type Scratch = DipyProbScratch;

    fn initial_direction(
        &self,
        seed_ras: Vec3,
        rng: &mut u64,
        scratch: &mut Self::Scratch,
    ) -> Option<Vec3> {
        // is_start = true: ignore the angular cone (no previous direction
        // yet) and allow any PMF peak.
        sample_direction(
            self.global,
            seed_ras,
            Vec3::ZERO,
            true,
            self.cos_max,
            rng,
            scratch,
        )
    }

    fn next_direction(
        &self,
        pt_ras: Vec3,
        prev_dir: Vec3,
        rng: &mut u64,
        scratch: &mut Self::Scratch,
    ) -> Option<Vec3> {
        sample_direction(
            self.global,
            pt_ras,
            prev_dir,
            false,
            self.cos_max,
            rng,
            scratch,
        )
    }

    fn step_size_mm(&self) -> f32 {
        self.step_mm
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

fn sample_direction(
    global: &DipyProbGlobal<'_>,
    point_ras: Vec3,
    prev_dir: Vec3,
    is_start: bool,
    cos_max: f32,
    rng: &mut u64,
    scratch: &mut DipyProbScratch,
) -> Option<Vec3> {
    let vox = global.ras_to_vox.transform_point3(point_ras);
    let sh_interp = trilinear_sh(vox, global)?;

    // PMF on sphere. `apply_row_into` is a precomputed matrix-vector
    // product (SH → amplitudes on the current sphere).
    scratch.pmf.resize(global.n_dirs, 0.0);
    global
        .sample_plan
        .apply_row_into(&sh_interp, &mut scratch.pmf);

    // Clamp negatives to zero: the SH reconstruction can over/undershoot.
    for v in scratch.pmf.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // When continuing, zero any sphere vertex outside the angular cone.
    // Uses absolute dot product to respect antipodal symmetry.
    if !is_start {
        for (i, v) in scratch.pmf.iter_mut().enumerate() {
            if *v > 0.0 {
                let sv = Vec3::from(global.sphere_verts[i]);
                if prev_dir.dot(sv).abs() < cos_max {
                    *v = 0.0;
                }
            }
        }
    }

    // Relative-peak threshold: keep only vertices whose amplitude is at
    // least `rel * max`. Filters PMF noise from the SH reconstruction.
    let max_val = scratch.pmf.iter().copied().fold(0.0f32, f32::max);
    if max_val <= 0.0 {
        return None;
    }
    let thresh = max_val * global.relative_peak_threshold;
    let mut total = 0.0f32;
    for v in scratch.pmf.iter_mut() {
        if *v < thresh {
            *v = 0.0;
        } else {
            total += *v;
        }
    }
    if total <= 0.0 {
        return None;
    }

    // Weighted sample: walk the CDF until we pass a uniform `[0, total)`.
    let r = lcg_f32(rng) * total;
    let mut cumsum = 0.0f32;
    let mut chosen = None;
    for (i, &v) in scratch.pmf.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        cumsum += v;
        if cumsum >= r {
            chosen = Some(i);
            break;
        }
    }
    let idx = chosen?;

    let sv = Vec3::from(global.sphere_verts[idx]);
    // Hemisphere convention for a continuation step: flip the sampled
    // vertex into the hemisphere of `prev_dir` if it's on the opposite
    // side. No flip at the seed because `prev_dir = ZERO`.
    let dir = if !is_start && prev_dir.dot(sv) < 0.0 {
        -sv
    } else {
        sv
    };
    Some(dir.normalize())
}

/// Trilinear interpolation of sparse SH coefficients at a fractional
/// voxel coordinate. Returns None when the point is outside the mask or
/// when every corner fails the `fixel_threshold` check.
fn trilinear_sh(vox: Vec3, global: &DipyProbGlobal<'_>) -> Option<Vec<f32>> {
    let x0 = vox.x.floor() as i32;
    let y0 = vox.y.floor() as i32;
    let z0 = vox.z.floor() as i32;

    let wx1 = vox.x - x0 as f32;
    let wy1 = vox.y - y0 as f32;
    let wz1 = vox.z - z0 as f32;
    let wx0 = 1.0 - wx1;
    let wy0 = 1.0 - wy1;
    let wz0 = 1.0 - wz1;

    let (nx, ny, nz) = (global.nx, global.ny, global.nz);
    let ncoeffs = global.ncoeffs;

    let mut out = vec![0.0f32; ncoeffs];
    let mut total_weight = 0.0f32;

    for (dx, wx) in [(0i32, wx0), (1, wx1)] {
        for (dy, wy) in [(0i32, wy0), (1, wy1)] {
            for (dz, wz) in [(0i32, wz0), (1, wz1)] {
                let xi = x0 + dx;
                let yi = y0 + dy;
                let zi = z0 + dz;
                if xi < 0
                    || yi < 0
                    || zi < 0
                    || xi >= nx as i32
                    || yi >= ny as i32
                    || zi >= nz as i32
                {
                    continue;
                }
                let lin = xi as usize * ny * nz + yi as usize * nz + zi as usize;
                let compact = global.dense_lut[lin];
                if compact == usize::MAX {
                    continue;
                }
                if global.gfa_data[compact] < global.fixel_threshold {
                    continue;
                }
                let w = wx * wy * wz;
                if w <= 0.0 {
                    continue;
                }
                let row = &global.sh_flat[compact * ncoeffs..(compact + 1) * ncoeffs];
                for (out_v, &sh_v) in out.iter_mut().zip(row) {
                    *out_v += w * sh_v;
                }
                total_weight += w;
            }
        }
    }

    if total_weight < 0.01 {
        return None;
    }
    for v in out.iter_mut() {
        *v /= total_weight;
    }
    Some(out)
}

//! Yeh (DSI-Studio style) fixel-peak direction getter.
//!
//! The ODX carries a precomputed fixel field: for each compact voxel, a
//! list of peak directions and amplitudes. `next_direction` picks the
//! peak inside a cone around the incoming direction with the largest
//! absolute dot product, optionally blended with the incoming direction
//! by `smooth_fraction`. `initial_direction` picks a random peak above
//! the (possibly per-attempt-randomized) fixel threshold and flips its
//! sign 50/50 so the bidirectional tracker's two branches cover opposite
//! hemispheres.
//!
//! Per-attempt randomization of `cos_max`, `step_mm`, `fixel_threshold`,
//! and `smooth` lives on `YehFixelDG` itself: the outer CPU runner
//! constructs a fresh DG for each attempt by consuming four f32s from
//! the RNG. Mirrors DSI-Studio's `tracking_thread.cpp` when the user has
//! left any of those params at their "randomize" sentinel (≤ 0 or ≥ 1).

use glam::{Mat4, Vec3};

use super::direction_getter::DirectionGetter;
use super::rng::lcg_f32;

/// Lifetime-of-run state shared by every Yeh attempt. Holds borrows into
/// the ODX fixel arrays, the dense voxel-index LUT, and the grid affine.
/// Built once per `run_cpu_yeh` call; one of these is referenced by every
/// per-attempt `YehFixelDG` so the DG itself stays a handful of bytes.
pub struct YehFixelGlobal<'a> {
    /// Per-voxel fixel-index offset into `directions` / `fixel_amplitude`.
    /// Voxel `v`'s fixels are `offsets[v]..offsets[v+1]`.
    pub offsets: &'a [u32],
    /// Peak direction table, indexed by fixel index (not voxel).
    pub directions: &'a [[f32; 3]],
    /// Per-fixel tracking amplitude (QA / FA / AFD / …). Legacy uniform
    /// 1.0 fallback when the ODX carries none is assembled by the caller.
    pub fixel_amplitude: &'a [f32],
    /// Dense 3-D voxel → compact-index LUT, size `nx * ny * nz`. Entries
    /// equal to `usize::MAX` mean the voxel is outside the mask.
    pub dense_lut: &'a [usize],
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub ras_to_vox: Mat4,
}

/// Per-thread scratch: candidate-peak list reused by `initial_direction`.
/// Kept as `Vec<Vec3>` rather than `Vec<[f32; 3]>` because we immediately
/// need `Vec3` arithmetic inside the sampler.
#[derive(Default)]
pub struct YehFixelScratch {
    pub peaks: Vec<Vec3>,
}

/// Per-attempt direction getter. Cheap to construct (just a pointer + four
/// f32s) and carries the attempt-specific randomized params so the trait
/// methods stay pure `&self`.
pub struct YehFixelDG<'a> {
    pub global: &'a YehFixelGlobal<'a>,
    /// `cos(max_angle_deg)`. A candidate peak is accepted only if its
    /// absolute dot product with the incoming direction ≥ `cos_max`.
    pub cos_max: f32,
    pub step_mm: f32,
    /// Fraction of the incoming direction to blend into the chosen peak.
    /// `0.0` = no smoothing, peak direction taken as-is. `1.0` ≈ no
    /// turning. Clamped to ≤ 0.95 by the caller before reaching us.
    pub smooth: f32,
    /// Per-attempt threshold on fixel amplitude (QA/FA/…). Peaks below
    /// this are ignored during both initial-peak selection and the
    /// stopping test inside the step loop.
    pub fixel_threshold: f32,
}

impl<'a> DirectionGetter for YehFixelDG<'a> {
    type Scratch = YehFixelScratch;

    fn initial_direction(
        &self,
        seed_ras: Vec3,
        rng: &mut u64,
        scratch: &mut Self::Scratch,
    ) -> Option<Vec3> {
        let seed_vox = self.global.ras_to_vox.transform_point3(seed_ras);
        let peak = pick_initial_peak(
            seed_vox,
            self.global.offsets,
            self.global.directions,
            self.global.fixel_amplitude,
            self.fixel_threshold,
            self.global.dense_lut,
            self.global.nx,
            self.global.ny,
            self.global.nz,
            rng,
            &mut scratch.peaks,
        )?;
        // Random hemisphere flip so the bidirectional tracker's two
        // branches don't both sample the same side of the fixel cone.
        let sign = if lcg_f32(rng) < 0.5 { 1.0 } else { -1.0 };
        Some(peak * sign)
    }

    fn next_direction(
        &self,
        pt_ras: Vec3,
        prev_dir: Vec3,
        _rng: &mut u64,
        _scratch: &mut Self::Scratch,
    ) -> Option<Vec3> {
        let pt_vox = self.global.ras_to_vox.transform_point3(pt_ras);
        let compact = voxel_at(
            pt_vox,
            self.global.dense_lut,
            self.global.nx,
            self.global.ny,
            self.global.nz,
        )?;
        if compact + 1 >= self.global.offsets.len() {
            return None;
        }
        let (best_dir, best_fa) = best_peak(
            compact,
            self.global.offsets,
            self.global.directions,
            self.global.fixel_amplitude,
            &prev_dir,
            self.cos_max,
        )?;
        if best_fa < self.fixel_threshold {
            return None;
        }
        let new_dir = ((1.0 - self.smooth) * best_dir + self.smooth * prev_dir).normalize_or_zero();
        if new_dir.length_squared() < 1e-8 {
            return None;
        }
        Some(new_dir)
    }

    fn step_size_mm(&self) -> f32 {
        self.step_mm
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Dense-LUT voxel lookup. Returns the compact voxel index for a point in
/// voxel coordinates, or `None` if the point is outside the grid or in a
/// voxel with no mask entry.
pub(super) fn voxel_at(
    pt_vox: Vec3,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Option<usize> {
    let x = pt_vox.x.floor() as i32;
    let y = pt_vox.y.floor() as i32;
    let z = pt_vox.z.floor() as i32;
    if x < 0 || y < 0 || z < 0 {
        return None;
    }
    let (x, y, z) = (x as usize, y as usize, z as usize);
    if x >= nx || y >= ny || z >= nz {
        return None;
    }
    let compact = dense_lut[x * ny * nz + y * nz + z];
    if compact == usize::MAX {
        None
    } else {
        Some(compact)
    }
}

/// Best peak inside the angular cone at a voxel. Returns the flipped peak
/// (oriented to match `incoming`'s half-space) and its amplitude, or
/// `None` when no peak is inside the cone.
fn best_peak(
    compact_idx: usize,
    offsets: &[u32],
    directions: &[[f32; 3]],
    fixel_amplitude: &[f32],
    incoming: &Vec3,
    cos_max: f32,
) -> Option<(Vec3, f32)> {
    let start = offsets[compact_idx] as usize;
    let end = offsets[compact_idx + 1] as usize;
    let mut best: Option<(Vec3, f32, f32)> = None;
    for k in start..end {
        if k >= directions.len() {
            break;
        }
        let peak = Vec3::from(directions[k]);
        if peak.length_squared() < 1e-8 {
            continue;
        }
        let peak = peak.normalize();
        let d = peak.dot(*incoming);
        let abs_d = d.abs();
        if abs_d < cos_max {
            continue;
        }
        let amplitude = fixel_amplitude.get(k).copied().unwrap_or(1.0);
        let flipped = if d >= 0.0 { peak } else { -peak };
        match &best {
            Some((_, _, best_abs)) if abs_d <= *best_abs => {}
            _ => best = Some((flipped, amplitude, abs_d)),
        }
    }
    best.map(|(d, amp, _)| (d, amp))
}

/// Pick a random initial peak among the seed voxel's fixels above
/// `fixel_threshold`. Uses the caller's `candidates` scratch so we don't
/// allocate per attempt. Returns `None` if no peak qualifies.
#[allow(clippy::too_many_arguments)]
fn pick_initial_peak(
    seed_vox: Vec3,
    offsets: &[u32],
    directions: &[[f32; 3]],
    fixel_amplitude: &[f32],
    fixel_threshold: f32,
    dense_lut: &[usize],
    nx: usize,
    ny: usize,
    nz: usize,
    rng: &mut u64,
    candidates: &mut Vec<Vec3>,
) -> Option<Vec3> {
    let compact_idx = voxel_at(seed_vox, dense_lut, nx, ny, nz)?;
    if compact_idx + 1 >= offsets.len() {
        return None;
    }
    let start = offsets[compact_idx] as usize;
    let end = offsets[compact_idx + 1] as usize;
    candidates.clear();
    for k in start..end {
        if k >= directions.len() {
            break;
        }
        let amplitude = fixel_amplitude.get(k).copied().unwrap_or(1.0);
        if amplitude < fixel_threshold {
            continue;
        }
        let peak = Vec3::from(directions[k]);
        if peak.length_squared() < 1e-8 {
            continue;
        }
        candidates.push(peak.normalize());
    }
    if candidates.is_empty() {
        None
    } else {
        let idx = (lcg_f32(rng) * candidates.len() as f32) as usize;
        Some(candidates[idx.min(candidates.len() - 1)])
    }
}

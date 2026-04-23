//! Purifibre: FOD-coherence streamline cleanup (Aydogan, nibrary).
//!
//! Port of nibrary's `purifibre` algorithm
//! (`nibrary/src/dMRI/tractography/utility/purifibre.cpp`). The idea:
//! streamlines whose tangent directions consistently match the local
//! bundle orientation are "coherent" and should be kept; streamlines
//! that wander into directions the local bundle never takes are outliers
//! and should be dropped.
//!
//! trxviz already builds the per-voxel × sphere-direction histogram
//! that nibrary calls a **sTODI** — it's `BoundaryContactField`, the
//! same structure that drives the boundary-glyph visualization. So
//! purifibre here is just: take streamlines + a prebuilt boundary field,
//! score each streamline against the field, filter.
//!
//! ### Algorithm sketch
//!
//! For each streamline:
//!
//! 1. Trim `trim_fraction` of segments off each end — endpoint segments
//!    have systematically lower bundle support because bundles taper,
//!    and we don't want to penalize honest streamlines for their own
//!    boundary behavior. Matches nibrary's `trimFactor`.
//! 2. For each remaining segment, look up the field histogram at the
//!    segment's midpoint voxel × the sphere bin nearest the segment's
//!    tangent direction. Track the minimum support across segments
//!    — nibrary calls this "SECO" (segment-to-bundle coupling).
//! 3. **FICO** (fiber-to-bundle coupling) score = `ln(min_SECO + 1)`.
//!    The log compresses the tail; the `+1` keeps it defined at zero.
//!
//! After scoring, discard the bottom `puri_fraction` of streamlines
//! ranked by FICO ascending.
//!
//! ### Optional: on-sphere smoothing
//!
//! nibrary smooths the sTODI histograms on the sphere (with a Gaussian
//! whose width is given by `sphericalSmoothing` degrees) before
//! scoring, so that a segment's tangent doesn't need to exactly hit a
//! sphere vertex to find support — it just needs to be close. We do
//! the same: build a per-voxel smoothed histogram buffer once, then
//! use it for the scoring loop.

use glam::Vec3;
use rayon::prelude::*;

use crate::data::orientation_field::{BoundaryContactField, SphereTemplate};
use crate::data::trx_data::TrxGpuData;
use crate::units::StreamlineIndex;

/// User-tunable parameters. Defaults mirror nibrary's purifibre.h.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PurifibreParams {
    /// Fraction (0.0..0.5) of segments to skip at each streamline end
    /// before scoring. Matches nibrary's `trimFactor / 100.0`.
    pub trim_fraction: f32,
    /// Fraction (0.0..1.0) of streamlines to discard, ranked from
    /// lowest FICO upward. Matches nibrary's `puriFactor / 100.0`.
    pub puri_fraction: f32,
    /// Gaussian FWHM (degrees, on the sphere) for smoothing the
    /// histograms before scoring. `0.0` disables smoothing — the
    /// scoring loop uses raw histograms as-is. Matches nibrary's
    /// `sphericalSmoothing`.
    pub spherical_smoothing_deg: f32,
}

impl Default for PurifibreParams {
    fn default() -> Self {
        // Defaults from nibrary/src/dMRI/tractography/utility/purifibre.h.
        Self {
            trim_fraction: 0.1,
            puri_fraction: 0.1,
            spherical_smoothing_deg: 15.0,
        }
    }
}

/// Output of one purifibre run.
#[derive(Debug, Clone)]
pub struct PurifibreReport {
    /// One FICO score per streamline in the full dataset (length =
    /// `gpu.nb_streamlines`). Entries outside the scored selection are
    /// `NaN`. NaN also marks streamlines that had fewer than one
    /// scoreable segment (post-trim) or fell entirely outside the
    /// boundary field.
    pub fico: Vec<f32>,
    /// Subset of the input `selected` that survived the threshold.
    pub survivors: Vec<StreamlineIndex>,
    /// FICO value at the cutoff percentile. Informational.
    pub threshold: f32,
    /// Short user-facing summary (e.g. "42000 → 37800 (FICO ≥ 2.13)").
    pub summary: String,
}

/// Score every streamline in `selected` against `field` and return
/// survivors + per-streamline FICO. Runs on all CPU cores via rayon.
pub fn purifibre_score(
    gpu: &TrxGpuData,
    selected: &[StreamlineIndex],
    field: &BoundaryContactField,
    params: &PurifibreParams,
) -> PurifibreReport {
    // ── optional on-sphere smoothing ────────────────────────────────
    //
    // Build a smoothed copy of the field's histograms if requested.
    // The smoothing kernel is computed once and applied per voxel. We
    // keep the original field read-only (no interior mutability); the
    // smoothed buffer lives in a local `Vec<f32>` and is used by the
    // scoring loop below.
    let smoothed_storage;
    let histograms: &[f32] = if params.spherical_smoothing_deg > 0.0 {
        smoothed_storage = smooth_histograms_on_sphere(
            field.histograms_flat(),
            &field.sphere,
            params.spherical_smoothing_deg,
        );
        &smoothed_storage
    } else {
        field.histograms_flat()
    };

    let nb_streamlines = gpu.nb_streamlines;
    let nbins = field.sphere.directions.len();
    let trim = params.trim_fraction.clamp(0.0, 0.5);

    // ── per-streamline FICO (parallel) ──────────────────────────────
    //
    // Score in parallel, collect (index, score) pairs, then merge into
    // a dense NaN-initialized `fico` vector. This keeps the closure
    // purely returning-data (no shared mut state) and gives rayon full
    // work-stealing latitude.
    let scored: Vec<(usize, f32)> = selected
        .par_iter()
        .map(|&idx| {
            let i = idx.0 as usize;
            let start = gpu.offsets[i] as usize;
            let end = gpu.offsets[i + 1] as usize;
            // `n_segments` = vertex pairs in this streamline.
            let n_segments = end.saturating_sub(start + 1);
            if n_segments == 0 {
                return (i, f32::NAN);
            }

            // Skip `trim_count` segments on each end. `trim_count` is
            // truncated, matching nibrary's `floor(trim * n)` behavior.
            let trim_count = (n_segments as f32 * trim).floor() as usize;
            let seg_begin = start + trim_count;
            let seg_end = end.saturating_sub(trim_count + 1);
            if seg_end <= seg_begin {
                return (i, f32::NAN);
            }

            let mut min_support = f32::INFINITY;
            let mut scored_any = false;
            for s in seg_begin..seg_end {
                let p0 = Vec3::from(gpu.positions[s]);
                let p1 = Vec3::from(gpu.positions[s + 1]);
                let tangent = p1 - p0;
                let len_sq = tangent.length_squared();
                if len_sq < 1e-10 {
                    // Degenerate zero-length segment — skip. Very rare.
                    continue;
                }
                let tangent_n = tangent * len_sq.sqrt().recip();
                let midpoint = 0.5 * (p0 + p1);

                // sTODI lookup: (voxel, bin) → histogram entry.
                let Some(voxel) = field.grid.point_to_voxel(midpoint) else {
                    continue; // segment midpoint outside field bbox
                };
                let flat = field.grid.flat_index(voxel[0], voxel[1], voxel[2]);
                let Some(compact) = field.compact_index_for(flat) else {
                    continue; // voxel had no contacts when field was built
                };
                let bin = field.sphere.nearest_bin(tangent_n);
                let support = histograms[compact * nbins + bin];

                if support < min_support {
                    min_support = support;
                }
                scored_any = true;
            }

            let fico = if scored_any && min_support.is_finite() {
                // nibrary purifibre.cpp: FICO = log(min_support + 1).
                // Log compresses the long tail; +1 keeps FICO defined
                // (and nonnegative) when min_support is 0 or tiny.
                (min_support + 1.0).ln()
            } else {
                f32::NAN
            };
            (i, fico)
        })
        .collect();

    let mut fico = vec![f32::NAN; nb_streamlines];
    for (i, score) in scored {
        fico[i] = score;
    }

    let (survivors, threshold) = apply_puri_threshold(&fico, selected, params.puri_fraction);
    let summary = format!(
        "{} → {} (FICO ≥ {:.3})",
        selected.len(),
        survivors.len(),
        threshold,
    );

    PurifibreReport {
        fico,
        survivors,
        threshold,
        summary,
    }
}

/// Re-threshold an existing per-streamline FICO vector. Used by the op
/// when only `puri_fraction` changed — avoids re-walking segments.
///
/// Returns `(survivors, threshold)`.
pub fn apply_puri_threshold(
    fico: &[f32],
    selected: &[StreamlineIndex],
    puri_fraction: f32,
) -> (Vec<StreamlineIndex>, f32) {
    // Sort the finite-FICO subset of the `selected` set; the threshold
    // is the FICO at the `puri_fraction` quantile. Streamlines whose
    // FICO is NaN (degenerate, outside field, etc.) are dropped along
    // with the low-FICO streamlines — they're not scoreable so we
    // can't keep them in good conscience.
    let mut sorted: Vec<f32> = selected
        .iter()
        .map(|idx| fico[idx.0 as usize])
        .filter(|s| s.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let puri = puri_fraction.clamp(0.0, 1.0);
    let threshold = if sorted.is_empty() {
        0.0
    } else {
        let cut = ((sorted.len() as f32) * puri).floor() as usize;
        // When puri == 0.0 (keep everything) we want threshold to be
        // the min finite FICO so the ≥ check passes for all.
        sorted[cut.min(sorted.len() - 1)]
    };

    let survivors: Vec<StreamlineIndex> = selected
        .iter()
        .copied()
        .filter(|&idx| {
            let s = fico[idx.0 as usize];
            s.is_finite() && s >= threshold
        })
        .collect();
    (survivors, threshold)
}

// ── on-sphere smoothing helpers ─────────────────────────────────────

/// Smooth a full `[n_voxels × n_bins]` histogram buffer with a Gaussian
/// kernel over angular distance on the sphere. The kernel width is
/// specified as a FWHM in degrees; internally converted to the sigma
/// of a Gaussian over angular distance (radians).
///
/// Allocates and returns a new owned buffer the same shape as `src`.
/// O(n_voxels × n_bins²) — fine for typical bin counts (≤200) and
/// voxel counts (≤10⁵). Runs in parallel across voxels.
fn smooth_histograms_on_sphere(src: &[f32], sphere: &SphereTemplate, fwhm_deg: f32) -> Vec<f32> {
    let nbins = sphere.directions.len();
    debug_assert!(
        src.len() % nbins == 0,
        "src must be a whole number of voxels"
    );
    let n_voxels = src.len() / nbins;

    // Build the per-bin Gaussian weight matrix once. `weights[b]` is
    // the kernel row centered on bin `b`: `weights[b][c]` is the
    // contribution of bin c to bin b after smoothing, already
    // row-normalized so each row sums to 1.
    let weights = build_sphere_smoothing_kernel(sphere, fwhm_deg);

    // Apply weight matrix per voxel. Parallelize across voxels — no
    // cross-voxel dependency, trivial work split.
    let mut dst = vec![0.0f32; src.len()];
    dst.par_chunks_mut(nbins)
        .zip(src.par_chunks(nbins))
        .for_each(|(dst_voxel, src_voxel)| {
            for b in 0..nbins {
                let row = &weights[b * nbins..(b + 1) * nbins];
                let mut acc = 0.0f32;
                for c in 0..nbins {
                    acc += row[c] * src_voxel[c];
                }
                dst_voxel[b] = acc;
            }
        });
    let _ = n_voxels; // silence unused; captured by chunks math above.
    dst
}

/// Construct an `n_bins × n_bins` row-stochastic Gaussian smoothing
/// kernel over angular distance on the unit sphere. Row `b` gives the
/// weights for smoothing bin `b` from all bins.
///
/// FWHM → σ conversion: FWHM = 2·sqrt(2·ln(2))·σ ≈ 2.3548σ.
/// We fold that into a single factor when filling the row.
fn build_sphere_smoothing_kernel(sphere: &SphereTemplate, fwhm_deg: f32) -> Vec<f32> {
    let n = sphere.directions.len();
    let sigma_rad = (fwhm_deg.to_radians() / 2.354_820_045).max(1e-4);
    let inv_two_sigma_sq = 0.5 / (sigma_rad * sigma_rad);

    let mut weights = vec![0.0f32; n * n];
    for b in 0..n {
        let dir_b = sphere.directions[b];
        let row = &mut weights[b * n..(b + 1) * n];
        let mut row_sum = 0.0f32;
        for c in 0..n {
            // Angular distance on the unit sphere. Clamp the dot to
            // [-1, 1] to guard against tiny numerical overshoot
            // (acos NaNs otherwise).
            let cos_theta = dir_b.dot(sphere.directions[c]).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let w = (-theta * theta * inv_two_sigma_sq).exp();
            row[c] = w;
            row_sum += w;
        }
        // Row-normalize so smoothed bins stay in the same "units" as
        // the unsmoothed ones (no amplitude drift).
        if row_sum > 0.0 {
            let inv = row_sum.recip();
            for c in 0..n {
                row[c] *= inv;
            }
        }
    }
    weights
}

//! Region derivation for `PrepareHausdorffPlanOp`:
//! reference bundle → coverage → limiting → seed / no_end masks + Hausdorff
//! post-filter point cloud.

use glam::{Mat4, Vec3};

use crate::data::odx_data::OdxScene;
use crate::data::trx_data::TrxGpuData;
use crate::workflow::{PostFilter, TrackingPlan, VoxelMask};
use std::sync::Arc;

pub struct HausdorffPlanParams {
    /// DSI-Studio-style tolerance (mm). Governs the limiting-mask dilation
    /// radius, the post-filter distance threshold, and the ±2·tolerance
    /// padding on min/max streamline length — all with the same value,
    /// matching `auto_track.cpp` semantics.
    pub tolerance_mm: f32,
    /// Dilation radius for the **seed** mask. DSI-Studio's autotrack
    /// seeds from the undilated atlas coverage (or a very small dilation
    /// of it) — seeding from the full `tolerance_mm` envelope wastes most
    /// attempts on voxels with no reference support. Keep this small
    /// (≤ 2 mm typical) even when `tolerance_mm` is large.
    pub seed_tolerance_mm: f32,
    /// Resolved DPF name (e.g. "qa", "afd", "amplitude") — the scalar whose
    /// primary-peak-per-voxel projection gates the seed / no_end masks.
    pub tracking_metric: String,
    /// Otsu threshold in `tracking_metric`'s native units; passed through
    /// to `TrackingPlan.fixel_otsu` so the tracker can scale its sentinel.
    pub fixel_otsu: f32,
    /// Seed voxels = limiting ∩ (metric ≥ factor × fixel_otsu).
    pub seed_fixel_otsu_factor: f32,
    /// No-end voxels = limiting ∩ (metric > factor × fixel_otsu).
    pub not_end_fixel_otsu_factor: f32,
    pub max_reference_points: usize,
}

pub struct HausdorffPlanOutputs {
    pub plan: TrackingPlan,
    pub seed_mask: Arc<VoxelMask>,
    pub limiting_mask: Arc<VoxelMask>,
    pub no_end_mask: Arc<VoxelMask>,
}

/// Build a Hausdorff plan from an ODX scene + reference streamline point cloud.
///
/// `reference_positions` is a flat point list in RAS+mm (typically
/// `TrxGpuData::positions`); `reference_offsets` is a CSR prefix-sum of
/// streamline point counts (typically `TrxGpuData::offsets`).
pub fn build_hausdorff_plan(
    scene: &OdxScene,
    reference_gpu: &TrxGpuData,
    selected_streamlines: &[u32],
    label: String,
    params: &HausdorffPlanParams,
) -> HausdorffPlanOutputs {
    let dims64 = scene.dimensions();
    let dims = [dims64[0] as u32, dims64[1] as u32, dims64[2] as u32];
    let voxel_to_ras = scene.voxel_to_ras();
    let ras_to_vox = voxel_to_ras.inverse();

    // Step 1: voxelize reference streamlines → coverage mask.
    let coverage = voxelize_streamlines(
        dims,
        ras_to_vox,
        reference_gpu,
        selected_streamlines,
    );

    // Step 2: EDT-dilate coverage by `tolerance_mm` → limiting mask, and
    // separately by `seed_tolerance_mm` → seed envelope. If
    // `seed_tolerance_mm >= tolerance_mm`, reuse the limiting mask.
    let min_vs = min_voxel_size_mm(&voxel_to_ras).max(1e-6);
    let tol_vox = (params.tolerance_mm / min_vs).max(0.0);
    let limiting_data = dilate_mask(&coverage, dims, tol_vox);
    let seed_tol_mm = params.seed_tolerance_mm.max(0.0).min(params.tolerance_mm);
    let seed_envelope: Vec<u8> = if seed_tol_mm >= params.tolerance_mm {
        limiting_data.clone()
    } else {
        dilate_mask(&coverage, dims, (seed_tol_mm / min_vs).max(0.0))
    };

    // Step 3: scatter the chosen tracking metric's primary-peak projection
    // onto the dense grid, then threshold using Otsu-scaled factors. This
    // matches DSI-Studio's `roi.cpp:95–98` semantics (threshold = factor ×
    // fixel_otsu, applied to the primary-peak-per-voxel projection).
    let amplitude = scatter_primary_peak_metric(scene, dims, &params.tracking_metric);
    let seed_threshold = params.seed_fixel_otsu_factor * params.fixel_otsu;
    let not_end_threshold = params.not_end_fixel_otsu_factor * params.fixel_otsu;

    let n_voxels = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let mut seed_data = vec![0u8; n_voxels];
    let mut no_end_data = vec![0u8; n_voxels];
    for i in 0..n_voxels {
        if limiting_data[i] != 0 && amplitude[i] > not_end_threshold {
            no_end_data[i] = 1;
        }
        if seed_envelope[i] != 0 && amplitude[i] >= seed_threshold {
            seed_data[i] = 1;
        }
    }

    let limiting_mask = Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: limiting_data,
    });
    let seed_mask = Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: seed_data,
    });
    let no_end_mask = Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: no_end_data,
    });

    // Step 4: flatten + subsample reference points for Hausdorff post-filter.
    let reference_points_ras = Arc::new(subsample_reference_points(
        reference_gpu,
        selected_streamlines,
        params.max_reference_points.max(1),
    ));

    // Step 5: compute reference-bundle min/max arc-length, clamp by ±2·tol
    // (DSI-Studio auto_track.cpp:265–267).
    let (ref_min, ref_max) =
        reference_length_extents(reference_gpu, selected_streamlines);
    let tol = params.tolerance_mm.max(0.0);
    let min_len_mm = if ref_min > 0.0 {
        Some(tol.max(ref_min - 2.0 * tol).max(0.0))
    } else {
        None
    };
    let max_len_mm = if ref_max > 0.0 {
        Some(ref_max + 2.0 * tol)
    } else {
        None
    };

    let plan = TrackingPlan {
        label,
        grid_dims: dims,
        voxel_to_ras,
        seed_mask: Some(seed_mask.clone()),
        limiting_mask: Some(limiting_mask.clone()),
        roa_mask: None,
        term_mask: None,
        roi_masks: Vec::new(),
        end_masks: Vec::new(),
        no_end_mask: Some(no_end_mask.clone()),
        post_filter: Some(PostFilter::Hausdorff {
            reference_points_ras,
            max_mm: params.tolerance_mm,
        }),
        min_len_mm,
        max_len_mm,
        max_angle_deg: None,
        step_size_mm: None,
        fixel_threshold: None,
        smooth_fraction: None,
        tolerance_mm: Some(params.tolerance_mm),
        fixel_otsu: Some(params.fixel_otsu),
    };

    HausdorffPlanOutputs {
        plan,
        seed_mask,
        limiting_mask,
        no_end_mask,
    }
}

/// Rasterize each streamline segment onto the grid via 3D DDA.
fn voxelize_streamlines(
    dims: [u32; 3],
    ras_to_vox: Mat4,
    gpu: &TrxGpuData,
    selected: &[u32],
) -> Vec<u8> {
    let n_voxels = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let mut mask = vec![0u8; n_voxels];
    let [nx, ny, nz] = dims;
    let positions = &gpu.positions;
    let offsets = &gpu.offsets;

    let set = |x: i32, y: i32, z: i32, mask: &mut [u8]| {
        if x < 0 || y < 0 || z < 0 {
            return;
        }
        let (x, y, z) = (x as u32, y as u32, z as u32);
        if x >= nx || y >= ny || z >= nz {
            return;
        }
        let idx = (x as usize)
            + (nx as usize) * ((y as usize) + (ny as usize) * (z as usize));
        mask[idx] = 1;
    };

    let rasterize_segment = |a: Vec3, b: Vec3, mask: &mut [u8]| {
        let va = ras_to_vox.transform_point3(a);
        let vb = ras_to_vox.transform_point3(b);
        let dx = vb.x - va.x;
        let dy = vb.y - va.y;
        let dz = vb.z - va.z;
        let steps = dx.abs().max(dy.abs()).max(dz.abs()).ceil() as i32;
        if steps <= 0 {
            set(va.x.floor() as i32, va.y.floor() as i32, va.z.floor() as i32, mask);
            return;
        }
        let inv = 1.0 / steps as f32;
        for s in 0..=steps {
            let t = s as f32 * inv;
            let x = (va.x + dx * t).floor() as i32;
            let y = (va.y + dy * t).floor() as i32;
            let z = (va.z + dz * t).floor() as i32;
            set(x, y, z, mask);
        }
    };

    let iter_streamline = |sid: u32, mask: &mut [u8]| {
        let sid = sid as usize;
        if sid + 1 >= offsets.len() {
            return;
        }
        let start = offsets[sid] as usize;
        let end = offsets[sid + 1] as usize;
        if end <= start + 1 {
            return;
        }
        for window in positions[start..end].windows(2) {
            let a = Vec3::from_array(window[0]);
            let b = Vec3::from_array(window[1]);
            rasterize_segment(a, b, mask);
        }
    };

    if selected.is_empty() {
        // No explicit selection → rasterize every streamline in the dataset.
        if offsets.len() >= 2 {
            for sid in 0..(offsets.len() as u32 - 1) {
                iter_streamline(sid, &mut mask);
            }
        }
    } else {
        for &sid in selected {
            iter_streamline(sid, &mut mask);
        }
    }

    mask
}

/// Dilate `mask` by a radius expressed in voxels using an exact 3-pass 1-D
/// EDT (Saito–Toriwaki). Returns the binary dilation (distance ≤ radius).
fn dilate_mask(mask: &[u8], dims: [u32; 3], radius_vox: f32) -> Vec<u8> {
    let nx = dims[0] as usize;
    let ny = dims[1] as usize;
    let nz = dims[2] as usize;
    let n = nx * ny * nz;
    if n == 0 {
        return Vec::new();
    }

    // Squared-distance result, in voxels²; initialized to a large value.
    const INF: u64 = u64::MAX / 4;
    let mut g2: Vec<u64> = Vec::with_capacity(n);

    // Pass along x: for each (y, z) row, compute 1-D squared distance to the
    // nearest set voxel in that row.
    for k in 0..nz {
        for j in 0..ny {
            // forward
            let mut d: u64 = INF;
            for i in 0..nx {
                let idx = i + nx * (j + ny * k);
                if mask[idx] != 0 {
                    d = 0;
                } else if d != INF {
                    d = d.saturating_add(1);
                }
                g2.push(if d == INF { INF } else { d * d });
            }
            // backward: update the just-pushed row in place
            let row_start = nx * (j + ny * k);
            let mut d: u64 = INF;
            for i in (0..nx).rev() {
                let idx = row_start + i;
                if mask[idx] != 0 {
                    d = 0;
                } else if d != INF {
                    d = d.saturating_add(1);
                }
                let sq = if d == INF { INF } else { d * d };
                if sq < g2[idx] {
                    g2[idx] = sq;
                }
            }
        }
    }

    // Pass along y then z using a lower-envelope parabola sweep.
    let mut tmp = vec![0u64; n];
    // y-pass
    for k in 0..nz {
        for i in 0..nx {
            // collect column values
            let mut f = vec![0u64; ny];
            for j in 0..ny {
                f[j] = g2[i + nx * (j + ny * k)];
            }
            let out = lower_envelope(&f);
            for j in 0..ny {
                tmp[i + nx * (j + ny * k)] = out[j];
            }
        }
    }
    // z-pass
    let mut out = vec![0u64; n];
    for j in 0..ny {
        for i in 0..nx {
            let mut f = vec![0u64; nz];
            for k in 0..nz {
                f[k] = tmp[i + nx * (j + ny * k)];
            }
            let sqd = lower_envelope(&f);
            for k in 0..nz {
                out[i + nx * (j + ny * k)] = sqd[k];
            }
        }
    }

    let r2 = (radius_vox * radius_vox).max(0.0);
    let r2_u = if r2 > (u64::MAX / 4) as f32 {
        u64::MAX / 4
    } else {
        r2.ceil() as u64
    };
    out.into_iter()
        .map(|d2| if d2 <= r2_u { 1u8 } else { 0u8 })
        .collect()
}

/// Felzenszwalb–Huttenlocher 1-D squared-distance transform (lower envelope
/// of parabolas). Input `f[q]` is the already-squared distance contribution
/// from previous dimensions; returns the same.
fn lower_envelope(f: &[u64]) -> Vec<u64> {
    let n = f.len();
    if n == 0 {
        return Vec::new();
    }
    const INF: u64 = u64::MAX / 4;
    let inf_f = INF as f64;
    let fd = |q: usize| if f[q] >= INF { inf_f } else { f[q] as f64 };

    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let mut k: usize = 0;
    for q in 1..n {
        loop {
            let vk = v[k] as f64;
            let qf = q as f64;
            let s = ((fd(q) + qf * qf) - (fd(v[k]) + vk * vk)) / (2.0 * (qf - vk));
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }

    let mut out = vec![0u64; n];
    let mut k: usize = 0;
    for q in 0..n {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let vk = v[k];
        let vkf = vk as f64;
        let d = (qf - vkf) * (qf - vkf) + fd(vk);
        out[q] = if d >= inf_f {
            INF
        } else {
            d.round() as u64
        };
    }
    out
}

fn min_voxel_size_mm(voxel_to_ras: &Mat4) -> f32 {
    let x = voxel_to_ras.col(0).truncate().length();
    let y = voxel_to_ras.col(1).truncate().length();
    let z = voxel_to_ras.col(2).truncate().length();
    x.min(y).min(z)
}

/// Scatter the primary-peak-per-voxel projection of a per-fixel scalar
/// (DPF `metric`) onto the dense grid. This is the value the seed /
/// no_end masks compare against — matches DSI-Studio's `fa[0][voxel]`
/// (the "fa" there is DSI-Studio naming; the value is generic per-voxel
/// primary-peak amplitude, not necessarily fractional anisotropy)
/// (`roi.cpp:95–98`). Falls back to `1.0` for every masked voxel only as
/// a last-ditch safety net; in practice the caller has already resolved
/// the metric via `OdxScene::fixel_otsu`, so the DPF must exist.
fn scatter_primary_peak_metric(scene: &OdxScene, dims: [u32; 3], metric: &str) -> Vec<f32> {
    let n = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let nx = dims[0] as usize;
    let ny = dims[1] as usize;
    let ijk = scene.ijk_lookup();
    let mut out = vec![0.0f32; n];

    let dpf_values = scene.dataset().scalar_dpf_f32(metric).ok();
    let offsets = scene.dataset().offsets();

    if let Some(values) = dpf_values {
        for (compact_idx, ijk) in ijk.iter().enumerate() {
            if compact_idx + 1 >= offsets.len() {
                break;
            }
            let start = offsets[compact_idx] as usize;
            let end = offsets[compact_idx + 1] as usize;
            if end <= start {
                continue;
            }
            let primary = *values.get(start).unwrap_or(&0.0);
            let i = ijk[0] as usize;
            let j = ijk[1] as usize;
            let k = ijk[2] as usize;
            let flat = i + nx * (j + ny * k);
            if flat < n {
                out[flat] = primary;
            }
        }
    } else {
        log::warn!(
            "scatter_primary_peak_metric: DPF '{metric}' not found; masks will be vacuous"
        );
        for ijk in ijk.iter() {
            let i = ijk[0] as usize;
            let j = ijk[1] as usize;
            let k = ijk[2] as usize;
            let flat = i + nx * (j + ny * k);
            if flat < n {
                out[flat] = 1.0;
            }
        }
    }
    out
}

/// Sum arc-length of each reference streamline and report `(min, max)` in mm.
/// Returns `(0.0, 0.0)` when the selection is empty.
fn reference_length_extents(gpu: &TrxGpuData, selected: &[u32]) -> (f32, f32) {
    let offsets = &gpu.offsets;
    let positions = &gpu.positions;
    let mut mn = f32::INFINITY;
    let mut mx: f32 = 0.0;
    let mut seen = false;

    let mut accumulate = |sid: usize| {
        if sid + 1 >= offsets.len() {
            return;
        }
        let start = offsets[sid] as usize;
        let end = offsets[sid + 1] as usize;
        if end <= start + 1 {
            return;
        }
        let mut len = 0.0f32;
        for w in positions[start..end].windows(2) {
            let a = Vec3::from_array(w[0]);
            let b = Vec3::from_array(w[1]);
            len += (b - a).length();
        }
        if len > 0.0 {
            if !seen || len < mn {
                mn = len;
            }
            if len > mx {
                mx = len;
            }
            seen = true;
        }
    };

    if selected.is_empty() {
        if offsets.len() >= 2 {
            for sid in 0..(offsets.len() - 1) {
                accumulate(sid);
            }
        }
    } else {
        for &sid in selected {
            accumulate(sid as usize);
        }
    }

    if seen { (mn, mx) } else { (0.0, 0.0) }
}

fn subsample_reference_points(
    gpu: &TrxGpuData,
    selected: &[u32],
    max_points: usize,
) -> Vec<[f32; 3]> {
    let offsets = &gpu.offsets;
    let positions = &gpu.positions;

    let mut collected: Vec<[f32; 3]> = Vec::new();
    if selected.is_empty() {
        collected.extend_from_slice(positions);
    } else {
        for &sid in selected {
            let sid = sid as usize;
            if sid + 1 >= offsets.len() {
                continue;
            }
            let start = offsets[sid] as usize;
            let end = offsets[sid + 1] as usize;
            collected.extend_from_slice(&positions[start..end]);
        }
    }

    if collected.len() <= max_points {
        return collected;
    }

    // Stride-based subsampling to keep arc-length roughly uniform.
    let stride = collected.len().div_ceil(max_points);
    collected.into_iter().step_by(stride).collect()
}

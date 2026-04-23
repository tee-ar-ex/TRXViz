//! Topology-Informed Pruning (Yeh 2019, DSI-Studio parity).
//!
//! Iteratively removes streamlines that traverse voxels occupied by
//! few other streamlines. Pure CPU implementation with rayon, no
//! coupling to `OdxScene` or any plan — operates on a raw point cloud.

use glam::Vec3;
use rayon::prelude::*;

use crate::data::trx_data::TrxGpuData;
use crate::units::StreamlineIndex;

#[derive(Debug, Clone, Copy)]
pub struct TipParams {
    /// Voxel size in mm for the density grid. DSI-Studio defaults to the
    /// subject's native voxel size; a good generic default is 1.0.
    pub voxel_size_mm: f32,
    /// Max pruning iterations. DSI-Studio autotrack uses 16.
    pub iterations: u32,
    /// Voxels with `count <= min_support` are "unsupported". `1` matches
    /// DSI-Studio — a voxel visited by only one streamline contributes no
    /// mutual support.
    pub min_support: u32,
    /// Fraction of a streamline's touched voxels that may be unsupported
    /// before the streamline is dropped. `0.0` = strict DSI-Studio parity
    /// (any unsupported voxel kills the streamline); `1.0` = passthrough.
    pub max_unsupported_fraction: f32,
}

impl Default for TipParams {
    fn default() -> Self {
        Self {
            voxel_size_mm: 1.0,
            iterations: 16,
            min_support: 1,
            max_unsupported_fraction: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TipReport {
    pub iterations_run: u32,
    pub kept: usize,
    pub removed: usize,
    pub max_density: u32,
}

/// Run TIP on `selected` in place. Returns a summary. Leaves
/// `selected` empty (`kept = 0`) if pruning collapses everything.
pub fn prune_by_topology(
    gpu: &TrxGpuData,
    selected: &mut Vec<StreamlineIndex>,
    params: &TipParams,
) -> TipReport {
    let before = selected.len();
    let vs = params.voxel_size_mm.max(1e-3);
    let iters = params.iterations.max(1);
    let max_unsupported = params.max_unsupported_fraction.clamp(0.0, 1.0);

    let mut max_density = 0u32;
    let mut iterations_run = 0u32;

    for iter in 0..iters {
        iterations_run = iter + 1;
        if selected.is_empty() {
            break;
        }

        // Derive a fresh bbox-aligned grid from the currently selected streamlines.
        let Some(grid) = compute_grid(gpu, selected, vs) else {
            break;
        };

        // Build density map in parallel via per-chunk locals + reduce.
        let density = build_density(gpu, selected, &grid);
        max_density = density.iter().copied().max().unwrap_or(0);

        // Partition streamlines by support. Each streamline's voxel-id set is
        // re-derived from scratch — cheap vs. storing it per iteration.
        let before_iter = selected.len();
        let min_support = params.min_support;
        let kept: Vec<StreamlineIndex> = selected
            .par_iter()
            .filter(|&&sid| streamline_supported(gpu, sid.0, &grid, &density, min_support, max_unsupported))
            .copied()
            .collect();

        let removed_this_iter = before_iter - kept.len();
        *selected = kept;
        if removed_this_iter == 0 {
            break;
        }
    }

    TipReport {
        iterations_run,
        kept: selected.len(),
        removed: before - selected.len(),
        max_density,
    }
}

struct Grid {
    origin: Vec3,
    inv_vs: f32,
    dims: [u32; 3],
}

impl Grid {
    #[inline]
    fn index(&self, v: Vec3) -> Option<usize> {
        let x = ((v.x - self.origin.x) * self.inv_vs).floor();
        let y = ((v.y - self.origin.y) * self.inv_vs).floor();
        let z = ((v.z - self.origin.z) * self.inv_vs).floor();
        if x < 0.0 || y < 0.0 || z < 0.0 {
            return None;
        }
        let (x, y, z) = (x as u32, y as u32, z as u32);
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return None;
        }
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        Some((x as usize) + nx * ((y as usize) + ny * (z as usize)))
    }

    #[inline]
    fn n_voxels(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }
}

fn compute_grid(gpu: &TrxGpuData, selected: &[StreamlineIndex], vs: f32) -> Option<Grid> {
    let offsets = &gpu.offsets;
    let positions = &gpu.positions;
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut seen = false;

    for sid in selected {
        let s = sid.0 as usize;
        if s + 1 >= offsets.len() {
            continue;
        }
        let a = offsets[s] as usize;
        let b = offsets[s + 1] as usize;
        for p in &positions[a..b] {
            let v = Vec3::from_array(*p);
            min = min.min(v);
            max = max.max(v);
            seen = true;
        }
    }
    if !seen {
        return None;
    }

    // Pad by one voxel so boundary points stay in-grid even with floor().
    let origin = min - Vec3::splat(vs);
    let extent = max - origin + Vec3::splat(vs);
    let inv_vs = 1.0 / vs;
    let dims = [
        (extent.x * inv_vs).ceil().max(1.0) as u32,
        (extent.y * inv_vs).ceil().max(1.0) as u32,
        (extent.z * inv_vs).ceil().max(1.0) as u32,
    ];
    // Sanity cap: refuse absurd grids (e.g. bad input at 0.01mm on a whole brain).
    let n = (dims[0] as u64) * (dims[1] as u64) * (dims[2] as u64);
    if n == 0 || n > 256 * 1024 * 1024 {
        log::warn!(
            "[tip] refusing grid {:?} at vs={vs}mm ({n} voxels) — returning no-op",
            dims
        );
        return None;
    }
    Some(Grid {
        origin,
        inv_vs,
        dims,
    })
}

/// Walk each segment of each selected streamline and increment a voxel counter
/// once per unique voxel *per streamline*. Uses a cheap "skip when same as
/// previous voxel" dedup which is exact for locally-coherent segments — the
/// same approximation DSI-Studio uses.
fn build_density(
    gpu: &TrxGpuData,
    selected: &[StreamlineIndex],
    grid: &Grid,
) -> Vec<u32> {
    let n = grid.n_voxels();
    let offsets = &gpu.offsets;
    let positions = &gpu.positions;

    // Chunked parallel reduction: each chunk produces a local Vec<u32>,
    // summed at the end. Avoids atomics and keeps memory linear.
    selected
        .par_chunks(512.max(selected.len() / (rayon::current_num_threads() * 4).max(1)))
        .map(|chunk| {
            let mut local = vec![0u32; n];
            for sid in chunk {
                for_each_voxel(gpu, sid.0, grid, offsets, positions, |idx, prev| {
                    if Some(idx) != prev {
                        local[idx] = local[idx].saturating_add(1);
                    }
                });
            }
            local
        })
        .reduce(
            || vec![0u32; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = x.saturating_add(*y);
                }
                a
            },
        )
}

/// Returns true if the streamline's unsupported-voxel fraction is ≤ threshold.
/// Per-streamline voxel set is rebuilt in a small `Vec` (points/3 bound).
fn streamline_supported(
    gpu: &TrxGpuData,
    sid: u32,
    grid: &Grid,
    density: &[u32],
    min_support: u32,
    max_unsupported_fraction: f32,
) -> bool {
    let mut touched: Vec<usize> = Vec::with_capacity(64);
    for_each_voxel(
        gpu,
        sid,
        grid,
        &gpu.offsets,
        &gpu.positions,
        |idx, prev| {
            if Some(idx) != prev {
                touched.push(idx);
            }
        },
    );
    if touched.is_empty() {
        return false;
    }
    touched.sort_unstable();
    touched.dedup();

    let mut unsupported = 0usize;
    for &idx in &touched {
        if density[idx] <= min_support {
            unsupported += 1;
        }
    }
    let frac = unsupported as f32 / touched.len() as f32;
    frac <= max_unsupported_fraction
}

/// Rasterize segments of one streamline through `grid`; invoke `cb(idx,
/// prev_idx)` at each step so callers can cheaply dedup adjacent hits.
fn for_each_voxel(
    _gpu: &TrxGpuData,
    sid: u32,
    grid: &Grid,
    offsets: &[u32],
    positions: &[[f32; 3]],
    mut cb: impl FnMut(usize, Option<usize>),
) {
    let s = sid as usize;
    if s + 1 >= offsets.len() {
        return;
    }
    let a = offsets[s] as usize;
    let b = offsets[s + 1] as usize;
    if b <= a + 1 {
        return;
    }
    let mut prev: Option<usize> = None;
    for w in positions[a..b].windows(2) {
        let pa = Vec3::from_array(w[0]);
        let pb = Vec3::from_array(w[1]);
        // Step length = ceil(max-axis voxel delta) to avoid skipping voxels.
        let d = (pb - pa) * grid.inv_vs;
        let steps = d.x.abs().max(d.y.abs()).max(d.z.abs()).ceil() as i32;
        if steps <= 0 {
            if let Some(idx) = grid.index(pa) {
                cb(idx, prev);
                prev = Some(idx);
            }
            continue;
        }
        let inv = 1.0 / steps as f32;
        for s_i in 0..=steps {
            let t = s_i as f32 * inv;
            let p = pa + (pb - pa) * t;
            if let Some(idx) = grid.index(p) {
                cb(idx, prev);
                prev = Some(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu(streamlines: Vec<Vec<[f32; 3]>>) -> TrxGpuData {
        let mut positions = Vec::new();
        let mut offsets = vec![0u32];
        for s in &streamlines {
            positions.extend_from_slice(s);
            offsets.push(positions.len() as u32);
        }
        let n = streamlines.len();
        TrxGpuData {
            positions,
            offsets,
            bbox_min: Vec3::ZERO,
            bbox_max: Vec3::ZERO,
            nb_streamlines: n,
            nb_vertices: 0,
            dpv_names: vec![],
            dps_names: vec![],
            groups: vec![],
            group_colors: vec![],
            dpv_data: vec![],
            dps_data: vec![],
            tangents: vec![],
            colors: vec![],
            all_indices: vec![],
            aabbs: vec![],
        }
    }

    #[test]
    fn single_streamline_is_fully_pruned() {
        let gpu = make_gpu(vec![(0..20)
            .map(|i| [i as f32, 0.0, 0.0])
            .collect()]);
        let mut sel: Vec<StreamlineIndex> = vec![StreamlineIndex(0)];
        let r = prune_by_topology(&gpu, &mut sel, &TipParams::default());
        assert_eq!(r.kept, 0);
        assert_eq!(r.removed, 1);
    }

    #[test]
    fn overlapping_streamlines_are_kept() {
        let line: Vec<[f32; 3]> = (0..20).map(|i| [i as f32, 0.0, 0.0]).collect();
        let gpu = make_gpu(vec![line.clone(), line.clone(), line]);
        let mut sel: Vec<StreamlineIndex> = vec![
            StreamlineIndex(0),
            StreamlineIndex(1),
            StreamlineIndex(2),
        ];
        let r = prune_by_topology(&gpu, &mut sel, &TipParams::default());
        assert_eq!(r.kept, 3);
        assert_eq!(r.removed, 0);
    }

    #[test]
    fn perpendicular_outlier_is_dropped() {
        // 6 identical core streamlines along +x, one outlier along +y.
        let core: Vec<[f32; 3]> = (0..30).map(|i| [i as f32, 0.0, 0.0]).collect();
        let outlier: Vec<[f32; 3]> = (0..30).map(|i| [5.0, i as f32, 0.0]).collect();
        let streamlines = vec![
            core.clone(), core.clone(), core.clone(),
            core.clone(), core.clone(), core,
            outlier,
        ];
        let gpu = make_gpu(streamlines);
        let mut sel: Vec<StreamlineIndex> =
            (0..7u32).map(StreamlineIndex).collect();
        let r = prune_by_topology(&gpu, &mut sel, &TipParams::default());
        assert_eq!(r.kept, 6);
        assert!(!sel.iter().any(|s| s.0 == 6));
    }

    #[test]
    fn passthrough_when_fraction_is_one() {
        let streamlines: Vec<Vec<[f32; 3]>> = (0..5)
            .map(|k| (0..10).map(|i| [i as f32, k as f32 * 10.0, 0.0]).collect())
            .collect();
        let gpu = make_gpu(streamlines);
        let mut sel: Vec<StreamlineIndex> =
            (0..5u32).map(StreamlineIndex).collect();
        let params = TipParams {
            max_unsupported_fraction: 1.0,
            ..Default::default()
        };
        let r = prune_by_topology(&gpu, &mut sel, &params);
        assert_eq!(r.kept, 5);
        assert_eq!(r.removed, 0);
    }
}

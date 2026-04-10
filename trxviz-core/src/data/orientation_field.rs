use std::collections::HashMap;

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BoundaryGlyphNormalization {
    RawCount,
    GlobalPeak,
    UnitPeak,
    UnitMass,
}

impl BoundaryGlyphNormalization {
    pub fn label(self) -> &'static str {
        match self {
            Self::RawCount => "Raw count",
            Self::GlobalPeak => "Global peak",
            Self::UnitPeak => "Per-voxel peak",
            Self::UnitMass => "Unit mass",
        }
    }

    pub const ALL: [Self; 4] = [
        Self::RawCount,
        Self::GlobalPeak,
        Self::UnitPeak,
        Self::UnitMass,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BoundaryGlyphColorMode {
    DirectionRgb,
    Monochrome,
}

impl BoundaryGlyphColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::DirectionRgb => "Directional RGB",
            Self::Monochrome => "Monochrome",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct BoundaryGlyphParams {
    pub voxel_size_mm: f32,
    pub sphere_lod: u32,
    pub scale: f32,
    pub normalization: BoundaryGlyphNormalization,
    pub density_3d_step: usize,
    pub slice_density_step: usize,
    pub color_mode: BoundaryGlyphColorMode,
    pub min_contacts: u32,
}

impl Default for BoundaryGlyphParams {
    fn default() -> Self {
        Self {
            voxel_size_mm: 3.0,
            sphere_lod: 12,
            scale: 2.0,
            normalization: BoundaryGlyphNormalization::GlobalPeak,
            density_3d_step: 2,
            slice_density_step: 1,
            color_mode: BoundaryGlyphColorMode::DirectionRgb,
            min_contacts: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrientationGridSpec {
    pub origin_ras: Vec3,
    pub dims: [u32; 3],
    pub voxel_size_mm: f32,
}

impl OrientationGridSpec {
    pub fn voxel_center(&self, ix: u32, iy: u32, iz: u32) -> Vec3 {
        self.origin_ras
            + Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5) * self.voxel_size_mm
    }

    pub fn voxel_box_min(&self, ix: u32, iy: u32, iz: u32) -> Vec3 {
        self.origin_ras + Vec3::new(ix as f32, iy as f32, iz as f32) * self.voxel_size_mm
    }

    pub fn flat_index(&self, ix: u32, iy: u32, iz: u32) -> u32 {
        ix + iy * self.dims[0] + iz * self.dims[0] * self.dims[1]
    }

    pub fn unflatten(&self, flat: u32) -> [u32; 3] {
        let nx = self.dims[0];
        let ny = self.dims[1];
        let iz = flat / (nx * ny);
        let rem = flat % (nx * ny);
        let iy = rem / nx;
        let ix = rem % nx;
        [ix, iy, iz]
    }

    pub fn point_to_voxel(&self, point: Vec3) -> Option<[u32; 3]> {
        let rel = (point - self.origin_ras) / self.voxel_size_mm;
        let ix = rel.x.floor() as i32;
        let iy = rel.y.floor() as i32;
        let iz = rel.z.floor() as i32;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.dims[0] as i32
            || iy >= self.dims[1] as i32
            || iz >= self.dims[2] as i32
        {
            return None;
        }
        Some([ix as u32, iy as u32, iz as u32])
    }
}

#[derive(Clone)]
pub struct SphereTemplate {
    pub vertices: Vec<[f32; 3]>,
    pub directions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub neighbors: Vec<Vec<usize>>,
}

impl SphereTemplate {
    pub fn new(_segments: u32) -> Self {
        let obj = include_str!("../assets/glyph_sphere.obj");
        let mut vertices = Vec::new();
        let mut directions = Vec::new();
        let mut indices = Vec::new();
        for line in obj.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("v ") {
                let vals: Vec<f32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if vals.len() == 3 {
                    let dir = Vec3::new(vals[0], vals[1], vals[2]).normalize_or_zero();
                    vertices.push(dir.to_array());
                    directions.push(dir);
                }
            } else if let Some(rest) = line.strip_prefix("f ") {
                let vals: Vec<u32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.split('/').next()?.parse::<u32>().ok())
                    .collect();
                if vals.len() == 3 {
                    indices.extend(vals.into_iter().map(|v| v - 1));
                }
            }
        }

        subdivide_sphere_once(&mut vertices, &mut directions, &mut indices);

        let mut neighbor_sets = vec![std::collections::HashSet::<usize>::new(); vertices.len()];
        for tri in indices.chunks_exact(3) {
            let a = tri[0] as usize;
            let b = tri[1] as usize;
            let c = tri[2] as usize;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                neighbor_sets[u].insert(v);
                neighbor_sets[v].insert(u);
            }
        }
        let neighbors = neighbor_sets
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();

        Self {
            vertices,
            directions,
            indices,
            neighbors,
        }
    }

    pub fn nearest_bin(&self, dir: Vec3) -> usize {
        let dir = dir.normalize_or_zero();
        let mut best_idx = 0usize;
        let mut best_dot = f32::NEG_INFINITY;
        for (idx, candidate) in self.directions.iter().enumerate() {
            let dot = dir.dot(*candidate);
            if dot > best_dot {
                best_dot = dot;
                best_idx = idx;
            }
        }
        best_idx
    }
}

fn subdivide_sphere_once(
    vertices: &mut Vec<[f32; 3]>,
    directions: &mut Vec<Vec3>,
    indices: &mut Vec<u32>,
) {
    let mut midpoint_cache = HashMap::<(u32, u32), u32>::new();
    let mut midpoint =
        |a: u32, b: u32, vertices: &mut Vec<[f32; 3]>, directions: &mut Vec<Vec3>| {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = midpoint_cache.get(&key) {
                return idx;
            }
            let va = Vec3::from(vertices[a as usize]);
            let vb = Vec3::from(vertices[b as usize]);
            let vm = (va + vb).normalize_or_zero();
            let idx = vertices.len() as u32;
            vertices.push(vm.to_array());
            directions.push(vm);
            midpoint_cache.insert(key, idx);
            idx
        };

    let old = indices.clone();
    indices.clear();
    for tri in old.chunks_exact(3) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let ab = midpoint(a, b, vertices, directions);
        let bc = midpoint(b, c, vertices, directions);
        let ca = midpoint(c, a, vertices, directions);
        indices.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
    }
}

#[derive(Clone)]
pub struct BoundaryContactField {
    pub grid: OrientationGridSpec,
    pub sphere: SphereTemplate,
    compact_lookup: HashMap<u32, usize>,
    occupied_voxels: Vec<u32>,
    histograms: Vec<f32>,
    contact_counts: Vec<u32>,
    #[allow(dead_code)]
    summary_vectors: Vec<[f32; 3]>,
    #[allow(dead_code)]
    summary_tensors: Vec<[f32; 6]>,
}

impl BoundaryContactField {
    pub fn build_from_streamlines(
        streamlines: &[StreamlineSet],
        params: &BoundaryGlyphParams,
    ) -> Option<Self> {
        let voxel_size = params.voxel_size_mm.max(0.5);
        let mut bbox_min = Vec3::splat(f32::INFINITY);
        let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);
        let mut any_points = false;
        for set in streamlines {
            for p in &set.positions {
                let v = Vec3::from(*p);
                bbox_min = bbox_min.min(v);
                bbox_max = bbox_max.max(v);
                any_points = true;
            }
        }
        if !any_points {
            return None;
        }

        let pad = Vec3::splat(voxel_size);
        bbox_min -= pad;
        bbox_max += pad;
        let extent = (bbox_max - bbox_min).max(Vec3::splat(voxel_size));
        let dims = [
            ((extent.x / voxel_size).ceil() as u32).max(1),
            ((extent.y / voxel_size).ceil() as u32).max(1),
            ((extent.z / voxel_size).ceil() as u32).max(1),
        ];
        let grid = OrientationGridSpec {
            origin_ras: bbox_min,
            dims,
            voxel_size_mm: voxel_size,
        };
        let sphere = SphereTemplate::new(params.sphere_lod);
        let nbins = sphere.vertices.len();

        let mut flat_to_compact: HashMap<u32, usize> = HashMap::new();
        let mut occupied_voxels = Vec::new();
        let mut histograms = Vec::<f32>::new();
        let mut contact_counts = Vec::<u32>::new();
        let mut summary_vectors = Vec::<[f32; 3]>::new();
        let mut summary_tensors = Vec::<[f32; 6]>::new();

        for set in streamlines {
            for win in set.offsets.windows(2) {
                let start = win[0] as usize;
                let end = win[1] as usize;
                if end.saturating_sub(start) < 2 {
                    continue;
                }
                for vi in start..(end - 1) {
                    let p0 = Vec3::from(set.positions[vi]);
                    let p1 = Vec3::from(set.positions[vi + 1]);
                    let Some(v0) = grid.point_to_voxel(p0) else {
                        continue;
                    };
                    let Some(v1) = grid.point_to_voxel(p1) else {
                        continue;
                    };
                    if v0 == v1 {
                        continue;
                    }

                    let flat0 = grid.flat_index(v0[0], v0[1], v0[2]);
                    let flat1 = grid.flat_index(v1[0], v1[1], v1[2]);

                    if segment_exit_from_voxel_box(p0, p1, v0, &grid).is_some() {
                        let dir = (p1 - p0).normalize_or_zero();
                        if dir.length_squared() > 0.0 {
                            let compact = ensure_voxel(
                                flat0,
                                nbins,
                                &mut flat_to_compact,
                                &mut occupied_voxels,
                                &mut histograms,
                                &mut contact_counts,
                                &mut summary_vectors,
                                &mut summary_tensors,
                            );
                            accumulate_contact(
                                compact,
                                dir,
                                &sphere,
                                nbins,
                                &mut histograms,
                                &mut contact_counts,
                                &mut summary_vectors,
                                &mut summary_tensors,
                            );
                        }
                    }

                    if segment_exit_from_voxel_box(p1, p0, v1, &grid).is_some() {
                        let dir = (p1 - p0).normalize_or_zero();
                        if dir.length_squared() > 0.0 {
                            let compact = ensure_voxel(
                                flat1,
                                nbins,
                                &mut flat_to_compact,
                                &mut occupied_voxels,
                                &mut histograms,
                                &mut contact_counts,
                                &mut summary_vectors,
                                &mut summary_tensors,
                            );
                            accumulate_contact(
                                compact,
                                dir,
                                &sphere,
                                nbins,
                                &mut histograms,
                                &mut contact_counts,
                                &mut summary_vectors,
                                &mut summary_tensors,
                            );
                        }
                    }
                }
            }
        }

        if occupied_voxels.is_empty() {
            return None;
        }

        let mut out = Self {
            grid,
            sphere,
            compact_lookup: flat_to_compact,
            occupied_voxels,
            histograms,
            contact_counts,
            summary_vectors,
            summary_tensors,
        };
        out.smooth_histograms();
        out.normalize(params.normalization);
        Some(out)
    }

    fn smooth_histograms(&mut self) {
        const SMOOTH_ITERS: usize = 2;
        const SMOOTH_WEIGHT: f32 = 0.35;
        let nbins = self.sphere.vertices.len();
        for voxel_idx in 0..self.occupied_voxels.len() {
            let start = voxel_idx * nbins;
            let mut src = self.histograms[start..start + nbins].to_vec();
            let mut dst = vec![0.0f32; nbins];
            for _ in 0..SMOOTH_ITERS {
                for bin in 0..nbins {
                    let neigh = &self.sphere.neighbors[bin];
                    let avg = if neigh.is_empty() {
                        src[bin]
                    } else {
                        neigh.iter().map(|&n| src[n]).sum::<f32>() / neigh.len() as f32
                    };
                    dst[bin] = src[bin] * (1.0 - SMOOTH_WEIGHT) + avg * SMOOTH_WEIGHT;
                }
                std::mem::swap(&mut src, &mut dst);
            }
            self.histograms[start..start + nbins].copy_from_slice(&src);
        }
    }

    fn normalize(&mut self, mode: BoundaryGlyphNormalization) {
        let nbins = self.sphere.vertices.len();
        let global_peak = if matches!(mode, BoundaryGlyphNormalization::GlobalPeak) {
            self.histograms.iter().copied().fold(0.0, f32::max)
        } else {
            0.0
        };
        for i in 0..self.occupied_voxels.len() {
            let start = i * nbins;
            let slice = &mut self.histograms[start..start + nbins];
            match mode {
                BoundaryGlyphNormalization::RawCount => {}
                BoundaryGlyphNormalization::GlobalPeak => {
                    if global_peak > 0.0 {
                        for v in slice {
                            *v /= global_peak;
                        }
                    }
                }
                BoundaryGlyphNormalization::UnitPeak => {
                    let peak = slice.iter().copied().fold(0.0, f32::max);
                    if peak > 0.0 {
                        for v in slice {
                            *v /= peak;
                        }
                    }
                }
                BoundaryGlyphNormalization::UnitMass => {
                    let sum: f32 = slice.iter().sum();
                    if sum > 0.0 {
                        for v in slice {
                            *v /= sum;
                        }
                    }
                }
            }
        }
    }

    pub fn occupied_voxels(&self) -> &[u32] {
        &self.occupied_voxels
    }

    pub fn histogram_for_voxel(&self, compact_index: usize) -> &[f32] {
        let nbins = self.sphere.vertices.len();
        let start = compact_index * nbins;
        &self.histograms[start..start + nbins]
    }

    pub fn contact_count(&self, compact_index: usize) -> u32 {
        self.contact_counts[compact_index]
    }

    #[cfg(test)]
    pub fn total_contacts(&self) -> u64 {
        self.contact_counts.iter().map(|&v| v as u64).sum()
    }

    #[allow(dead_code)]
    pub fn sample_summary_vector(&self, world_pos: Vec3) -> Vec3 {
        let Some(voxel) = self.grid.point_to_voxel(world_pos) else {
            return Vec3::ZERO;
        };
        let flat = self.grid.flat_index(voxel[0], voxel[1], voxel[2]);
        if let Some(&idx) = self.compact_lookup.get(&flat) {
            return Vec3::from(self.summary_vectors[idx]);
        }

        let mut sum = Vec3::ZERO;
        let mut weight_sum = 0.0f32;
        let vx = voxel[0] as i32;
        let vy = voxel[1] as i32;
        let vz = voxel[2] as i32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = vx + dx;
                    let ny = vy + dy;
                    let nz = vz + dz;
                    if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= self.grid.dims[0] as i32
                        || ny >= self.grid.dims[1] as i32
                        || nz >= self.grid.dims[2] as i32
                    {
                        continue;
                    }
                    let flat = self.grid.flat_index(nx as u32, ny as u32, nz as u32);
                    let Some(&idx) = self.compact_lookup.get(&flat) else {
                        continue;
                    };
                    let v = Vec3::from(self.summary_vectors[idx]);
                    if v.length_squared() <= 1e-10 {
                        continue;
                    }
                    let center = self.grid.voxel_center(nx as u32, ny as u32, nz as u32);
                    let d2 = (center - world_pos).length_squared();
                    let w = 1.0 / d2.max(1e-4);
                    sum += v * w;
                    weight_sum += w;
                }
            }
        }
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            Vec3::ZERO
        }
    }

    #[allow(dead_code)]
    pub fn sample_tensor(&self, world_pos: Vec3) -> [f32; 6] {
        let Some(voxel) = self.grid.point_to_voxel(world_pos) else {
            return [0.0; 6];
        };
        let flat = self.grid.flat_index(voxel[0], voxel[1], voxel[2]);
        if let Some(&idx) = self.compact_lookup.get(&flat) {
            self.summary_tensors[idx]
        } else {
            [0.0; 6]
        }
    }
}

fn ensure_voxel(
    flat: u32,
    nbins: usize,
    flat_to_compact: &mut HashMap<u32, usize>,
    occupied_voxels: &mut Vec<u32>,
    histograms: &mut Vec<f32>,
    contact_counts: &mut Vec<u32>,
    summary_vectors: &mut Vec<[f32; 3]>,
    summary_tensors: &mut Vec<[f32; 6]>,
) -> usize {
    if let Some(&idx) = flat_to_compact.get(&flat) {
        idx
    } else {
        let idx = occupied_voxels.len();
        flat_to_compact.insert(flat, idx);
        occupied_voxels.push(flat);
        histograms.resize((idx + 1) * nbins, 0.0);
        contact_counts.push(0);
        summary_vectors.push([0.0; 3]);
        summary_tensors.push([0.0; 6]);
        idx
    }
}

#[derive(Clone)]
pub struct StreamlineSet {
    pub positions: Vec<[f32; 3]>,
    pub offsets: Vec<u32>,
}

fn accumulate_contact(
    compact: usize,
    dir: Vec3,
    sphere: &SphereTemplate,
    nbins: usize,
    histograms: &mut [f32],
    contact_counts: &mut [u32],
    summary_vectors: &mut [[f32; 3]],
    summary_tensors: &mut [[f32; 6]],
) {
    let bin = sphere.nearest_bin(dir);
    histograms[compact * nbins + bin] += 1.0;
    contact_counts[compact] += 1;
    let sum = Vec3::from(summary_vectors[compact]) + dir;
    summary_vectors[compact] = sum.to_array();
    let t = &mut summary_tensors[compact];
    t[0] += dir.x * dir.x;
    t[1] += dir.x * dir.y;
    t[2] += dir.x * dir.z;
    t[3] += dir.y * dir.y;
    t[4] += dir.y * dir.z;
    t[5] += dir.z * dir.z;
}

fn segment_exit_from_voxel_box(
    start: Vec3,
    end: Vec3,
    voxel: [u32; 3],
    grid: &OrientationGridSpec,
) -> Option<Vec3> {
    let dir = end - start;
    if dir.length_squared() <= 1e-12 {
        return None;
    }
    let bmin = grid.voxel_box_min(voxel[0], voxel[1], voxel[2]);
    let bmax = bmin + Vec3::splat(grid.voxel_size_mm);
    let mut best_t = f32::INFINITY;

    for axis in 0..3 {
        let axis_dir = dir[axis];
        if axis_dir.abs() <= 1e-8 {
            continue;
        }
        let plane = if axis_dir > 0.0 {
            bmax[axis]
        } else {
            bmin[axis]
        };
        let t = (plane - start[axis]) / axis_dir;
        if !(1e-5..=1.0).contains(&t) {
            continue;
        }
        let p = start + dir * t;
        let in_other_axes = match axis {
            0 => {
                p.y >= bmin.y - 1e-4
                    && p.y <= bmax.y + 1e-4
                    && p.z >= bmin.z - 1e-4
                    && p.z <= bmax.z + 1e-4
            }
            1 => {
                p.x >= bmin.x - 1e-4
                    && p.x <= bmax.x + 1e-4
                    && p.z >= bmin.z - 1e-4
                    && p.z <= bmax.z + 1e-4
            }
            _ => {
                p.x >= bmin.x - 1e-4
                    && p.x <= bmax.x + 1e-4
                    && p.y >= bmin.y - 1e-4
                    && p.y <= bmax.y + 1e-4
            }
        };
        if in_other_axes && t < best_t {
            best_t = t;
        }
    }

    if best_t.is_finite() {
        Some(start + dir * best_t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_roundtrip_flatten() {
        let grid = OrientationGridSpec {
            origin_ras: Vec3::new(-5.0, -5.0, -5.0),
            dims: [4, 5, 6],
            voxel_size_mm: 2.0,
        };
        let flat = grid.flat_index(2, 3, 4);
        assert_eq!(grid.unflatten(flat), [2, 3, 4]);
    }

    #[test]
    fn crossing_point_found_on_adjacent_voxels() {
        let grid = OrientationGridSpec {
            origin_ras: Vec3::ZERO,
            dims: [4, 4, 4],
            voxel_size_mm: 2.0,
        };
        let p0 = Vec3::new(0.5, 0.5, 0.5);
        let p1 = Vec3::new(2.5, 0.5, 0.5);
        let crossing = segment_exit_from_voxel_box(p0, p1, [0, 0, 0], &grid).unwrap();
        assert!((crossing.x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn build_field_combines_streamline_sets() {
        let params = BoundaryGlyphParams::default();
        let set_a = StreamlineSet {
            positions: vec![[0.2, 0.2, 0.2], [8.2, 0.2, 0.2]],
            offsets: vec![0, 2],
        };
        let set_b = StreamlineSet {
            positions: vec![[0.2, 0.2, 0.2], [0.2, 8.2, 0.2]],
            offsets: vec![0, 2],
        };
        let field = BoundaryContactField::build_from_streamlines(&[set_a, set_b], &params).unwrap();
        assert!(!field.occupied_voxels().is_empty());
        assert!(field.total_contacts() >= 2);
    }

    #[test]
    fn crossing_votes_use_segment_direction() {
        let params = BoundaryGlyphParams::default();
        let set = StreamlineSet {
            positions: vec![[0.2, 0.2, 0.2], [8.2, 0.2, 0.2]],
            offsets: vec![0, 2],
        };
        let field = BoundaryContactField::build_from_streamlines(&[set], &params).unwrap();
        let summary = field.sample_summary_vector(Vec3::new(4.2, 0.2, 0.2));
        assert!(summary.x > 0.5);
        assert!(summary.y.abs() < 1e-4);
        assert!(summary.z.abs() < 1e-4);
    }

    #[test]
    fn sphere_template_matches_odf8_like_density() {
        let sphere = SphereTemplate::new(12);
        assert_eq!(sphere.vertices.len(), 642);
        assert_eq!(sphere.directions.len(), 642);
        assert_eq!(sphere.indices.len() / 3, 1280);
    }
}

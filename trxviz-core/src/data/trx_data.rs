use crate::data::gifti_data::GiftiSurfaceData;
use crate::units::{Millimeters, StreamlineIndex};
use glam::Vec3;
use std::collections::HashMap;
use std::collections::HashSet;
use trx_rs::{
    AnyTrxFile, DataArray, StreamlineAabb, Tractogram, build_streamline_aabbs_from_slices,
};

/// Rendering style for streamlines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RenderStyle {
    /// Plain colored lines.
    Flat = 0,
    /// Zoeckler illuminated lines (tangent-based Phong).
    Illuminated = 1,
    /// True triangle-mesh streamtubes.
    Tubes = 2,
    /// Depth-cued lines (brightness fades with camera distance).
    DepthCue = 3,
}

/// Available coloring modes for streamline visualization.
#[derive(Clone, Debug, PartialEq)]
pub enum ColorMode {
    /// RGB from absolute local tangent direction.
    DirectionRgb,
    /// Scalar per-vertex field mapped to colormap.
    Dpv(String),
    /// Scalar per-streamline field expanded to all vertices.
    Dps(String),
    /// Distinct color per group.
    Group,
    /// Single uniform color for all vertices.
    Uniform([f32; 4]),
}

/// GPU-ready streamline data extracted from a TRX file.
#[derive(Clone)]
pub struct TrxGpuData {
    /// Raw positions in RAS+ space.
    pub positions: Vec<[f32; 3]>,
    /// Streamline offsets (length = nb_streamlines + 1).
    pub offsets: Vec<u32>,
    /// Bounding box min/max in RAS+ space.
    pub bbox_min: Vec3,
    pub bbox_max: Vec3,
    /// Metadata counts.
    pub nb_streamlines: usize,
    pub nb_vertices: usize,
    /// Available DPV field names.
    pub dpv_names: Vec<String>,
    /// Available DPS field names.
    pub dps_names: Vec<String>,
    /// Available group names and their streamline indices.
    pub groups: Vec<(String, Vec<StreamlineIndex>)>,
    /// Optional per-group colors from DPG metadata, aligned with `groups`.
    pub group_colors: Vec<Option<[f32; 4]>>,
    /// Cached DPV data: each entry is a flat `Vec<f32>` with one value per vertex.
    pub dpv_data: Vec<(String, Vec<f32>)>,
    /// Cached DPS data: each entry is a flat `Vec<f32>` with one value per streamline.
    pub dps_data: Vec<(String, Vec<f32>)>,
    /// Per-vertex tangent directions (normalized, unsigned).
    pub tangents: Vec<[f32; 3]>,
    /// Current vertex colors.
    pub colors: Vec<[f32; 4]>,
    /// Full line-segment index buffer (all streamlines).
    pub all_indices: Vec<u32>,
    /// Per-streamline axis-aligned bounding boxes.
    pub aabbs: Vec<StreamlineAabb>,
}

/// Per-vertex data for true streamline tube meshes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TubeMeshVertex {
    /// Vertex position in world space.
    pub position: [f32; 3],
    /// Outward-facing normal in world space.
    pub normal: [f32; 3],
    /// Vertex color.
    pub color: [f32; 4],
}

/// Tableau-20 palette (matplotlib `tab20`) as linear RGB floats [0, 1].
/// Index matches the Python `tableau_20[i]` convention.
const TABLEAU_20: [[f32; 3]; 20] = [
    [0.122, 0.467, 0.706], // 0  blue
    [0.682, 0.780, 0.910], // 1  light blue
    [1.000, 0.498, 0.055], // 2  orange
    [1.000, 0.733, 0.471], // 3  light orange
    [0.173, 0.627, 0.173], // 4  green
    [0.596, 0.875, 0.541], // 5  light green
    [0.839, 0.153, 0.157], // 6  red
    [1.000, 0.596, 0.588], // 7  light red / salmon
    [0.580, 0.404, 0.741], // 8  purple
    [0.773, 0.690, 0.835], // 9  light purple
    [0.549, 0.337, 0.294], // 10 brown
    [0.769, 0.612, 0.580], // 11 light brown
    [0.890, 0.467, 0.761], // 12 pink
    [0.969, 0.714, 0.824], // 13 light pink
    [0.498, 0.498, 0.498], // 14 grey
    [0.780, 0.780, 0.780], // 15 light grey
    [0.737, 0.741, 0.133], // 16 olive / yellow-green
    [0.859, 0.859, 0.553], // 17 light yellow-green
    [0.090, 0.745, 0.812], // 18 cyan
    [0.620, 0.855, 0.898], // 19 light cyan
];

/// Two extra colors used by `tableau_extension` in the Python COLOR_DICT.
const TABLEAU_EXTENSION: [[f32; 3]; 2] = [
    [0.400, 0.761, 0.647], // 0  teal
    [0.988, 0.553, 0.384], // 1  coral
];

/// Return the reference color for a well-known bundle group name, or `None` if
/// the name is not recognised.  Colors are [R, G, B, A] linear floats in [0, 1].
pub(crate) fn group_name_color(name: &str) -> Option<[f32; 4]> {
    let t = |i: usize| -> [f32; 4] {
        let [r, g, b] = TABLEAU_20[i];
        [r, g, b, 1.0]
    };
    let e = |i: usize| -> [f32; 4] {
        let [r, g, b] = TABLEAU_EXTENSION[i];
        [r, g, b, 1.0]
    };
    let rgb = |r: f32, g: f32, b: f32| -> [f32; 4] { [r, g, b, 1.0] };

    match name {
        "Left Anterior Thalamic" | "C_L" => Some(t(0)),
        "Right Anterior Thalamic" | "C_R" => Some(t(1)),
        "Left Corticospinal" => Some(t(2)),
        "Right Corticospinal" => Some(t(3)),
        "Left Cingulum Cingulate" | "MCP" => Some(t(4)),
        "Right Cingulum Cingulate" | "CCMid" => Some(t(5)),
        "Left Posterior Arcuate" => Some(t(6)),
        "Right Posterior Arcuate" => Some(t(7)),
        "Forceps Minor" | "CC_ForcepsMinor" => Some(t(8)),
        "Forceps Major" | "CC_ForcepsMajor" => Some(t(9)),
        "Left Inferior Fronto-occipital" | "IFOF_L" => Some(t(10)),
        "Right Inferior Fronto-occipital" | "IFOF_R" => Some(t(11)),
        "Left Inferior Longitudinal" | "F_L" => Some(t(12)),
        "Right Inferior Longitudinal" | "F_R" => Some(t(13)),
        "Left Superior Longitudinal" => Some(t(14)),
        "Right Superior Longitudinal" => Some(t(15)),
        "Left Uncinate" | "UF_L" => Some(t(16)),
        "Right Uncinate" | "UF_R" => Some(t(17)),
        "Left Arcuate" | "AF_L" => Some(t(18)),
        "Right Arcuate" | "AF_R" => Some(t(19)),
        "Left Vertical Occipital" => Some(e(0)),
        "Right Vertical Occipital" => Some(e(1)),
        "median" => Some(t(6)),
        // Paul Tol's palette for callosal bundles
        "Callosum Orbital" => Some(rgb(0.20, 0.13, 0.53)),
        "Callosum Anterior Frontal" => Some(rgb(0.07, 0.47, 0.20)),
        "Callosum Superior Frontal" => Some(rgb(0.27, 0.67, 0.60)),
        "Callosum Motor" => Some(rgb(0.53, 0.80, 0.93)),
        "Callosum Superior Parietal" => Some(rgb(0.87, 0.80, 0.47)),
        "Callosum Posterior Parietal" => Some(rgb(0.80, 0.40, 0.47)),
        "Callosum Occipital" => Some(rgb(0.67, 0.27, 0.60)),
        "Callosum Temporal" => Some(rgb(0.53, 0.13, 0.33)),
        _ => None,
    }
}

impl TrxGpuData {
    /// Build from raw positions and offsets only (no per-vertex/streamline metadata).
    /// Used to package synthetic tractography output into the standard data type.
    pub fn from_positions_and_offsets(positions: Vec<[f32; 3]>, offsets: Vec<u32>) -> Self {
        let nb_streamlines = offsets.len().saturating_sub(1);
        let nb_vertices = positions.len();
        Self::from_components(
            positions,
            offsets,
            nb_streamlines,
            nb_vertices,
            ExtractedMetadata {
                dpv_data: Vec::new(),
                dps_data: Vec::new(),
                groups: Vec::new(),
                group_colors: Vec::new(),
            },
        )
    }

    pub fn from_tractogram(tractogram: &Tractogram) -> anyhow::Result<Self> {
        let positions = tractogram.positions().to_vec();
        let nb_vertices = positions.len();
        let nb_streamlines = tractogram.nb_streamlines();
        let offsets = tractogram.offsets().to_vec();
        let groups: Vec<(String, Vec<StreamlineIndex>)> = tractogram
            .groups()
            .iter()
            .map(|(name, members)| {
                (
                    name.clone(),
                    members.iter().copied().map(StreamlineIndex).collect(),
                )
            })
            .collect();
        let group_colors = groups
            .iter()
            .map(|(name, _)| tractogram.dpg().get(name).and_then(extract_group_color))
            .collect();

        Ok(Self::from_components(
            positions,
            offsets,
            nb_streamlines,
            nb_vertices,
            ExtractedMetadata {
                dpv_data: Vec::new(),
                dps_data: Vec::new(),
                groups,
                group_colors,
            },
        ))
    }

    pub fn from_any_trx(any: &AnyTrxFile) -> anyhow::Result<Self> {
        let positions = any.positions_f32();
        let nb_vertices = positions.len();
        let nb_streamlines = any.nb_streamlines();
        let offsets = any.offsets_vec();
        let metadata = extract_metadata(any, nb_vertices, nb_streamlines);

        Ok(Self::from_components(
            positions,
            offsets,
            nb_streamlines,
            nb_vertices,
            metadata,
        ))
    }

    fn from_components(
        positions: Vec<[f32; 3]>,
        offsets: Vec<u32>,
        nb_streamlines: usize,
        nb_vertices: usize,
        metadata: ExtractedMetadata,
    ) -> Self {
        let (bbox_min, bbox_max) = compute_bounding_box(&positions);
        let all_indices = build_line_indices(&offsets);
        let tangents = compute_tangents(&positions, &offsets);
        let colors = direction_colors_from_tangents(&tangents);
        let aabbs = build_streamline_aabbs_from_slices(&positions, &offsets);

        Self {
            positions,
            offsets,
            bbox_min,
            bbox_max,
            nb_streamlines,
            nb_vertices,
            dpv_names: metadata.dpv_data.iter().map(|(n, _)| n.clone()).collect(),
            dps_names: metadata.dps_data.iter().map(|(n, _)| n.clone()).collect(),
            groups: metadata.groups,
            group_colors: metadata.group_colors,
            dpv_data: metadata.dpv_data,
            dps_data: metadata.dps_data,
            tangents,
            colors,
            all_indices,
            aabbs,
        }
    }

    /// Recompute vertex colors based on the given color mode.
    /// For scalar modes (DPV/DPS), `scalar_range` overrides automatic
    /// range detection and `colormap` selects the colormap.
    pub fn recolor(
        &mut self,
        mode: &ColorMode,
        scalar_range: Option<(f32, f32)>,
        colormap: crate::renderer::mesh_renderer::SurfaceColormap,
    ) {
        self.colors = match mode {
            ColorMode::DirectionRgb => direction_colors_from_tangents(&self.tangents),
            ColorMode::Dpv(name) => {
                if let Some((_, data)) = self.dpv_data.iter().find(|(n, _)| n == name) {
                    let (lo, hi) = scalar_range.unwrap_or_else(|| scalar_auto_range(data));
                    scalar_to_colors_ranged(data, lo, hi, colormap)
                } else {
                    vec![[0.5, 0.5, 0.5, 1.0]; self.nb_vertices]
                }
            }
            ColorMode::Dps(name) => {
                if let Some((_, data)) = self.dps_data.iter().find(|(n, _)| n == name) {
                    let (lo, hi) = scalar_range.unwrap_or_else(|| scalar_auto_range(data));
                    expand_dps_to_vertices_ranged(
                        data,
                        &self.offsets,
                        self.nb_vertices,
                        lo,
                        hi,
                        colormap,
                    )
                } else {
                    vec![[0.5, 0.5, 0.5, 1.0]; self.nb_vertices]
                }
            }
            ColorMode::Group => self.compute_group_colors(),
            ColorMode::Uniform(c) => vec![*c; self.nb_vertices],
        };
    }

    /// Query streamlines intersecting a sphere. Returns matching streamline indices.
    /// Uses AABB broad phase then per-vertex distance check.
    pub fn query_sphere(&self, center: Vec3, radius: Millimeters) -> HashSet<StreamlineIndex> {
        let r2 = radius.0 * radius.0;
        let sphere_min = center - Vec3::splat(radius.0);
        let sphere_max = center + Vec3::splat(radius.0);
        let mut result = HashSet::new();

        for si in 0..self.nb_streamlines {
            let aabb = &self.aabbs[si];
            // AABB broad phase
            if !aabb.overlaps_box(sphere_min.to_array(), sphere_max.to_array()) {
                continue;
            }
            // Vertex-level narrow phase
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if (p - center).length_squared() <= r2 {
                    result.insert(StreamlineIndex(si as u32));
                    break;
                }
            }
        }
        result
    }

    /// Query streamlines that pass within `depth_mm` of the surface.
    /// Uses streamline AABB and surface AABB broad phase, then nearest surface-vertex distance.
    pub fn query_near_surface(
        &self,
        surface: &GiftiSurfaceData,
        depth_mm: Millimeters,
    ) -> HashSet<StreamlineIndex> {
        let mut result = HashSet::new();
        if depth_mm.0 <= 0.0 || surface.vertices.is_empty() {
            return result;
        }
        let depth2 = depth_mm.0 * depth_mm.0;
        let smin = surface.bbox_min - Vec3::splat(depth_mm.0);
        let smax = surface.bbox_max + Vec3::splat(depth_mm.0);
        let grid = SurfaceSpatialGrid::build(&surface.vertices, depth_mm.0.max(0.5));

        for si in 0..self.nb_streamlines {
            let aabb = &self.aabbs[si];
            if !aabb_overlaps_expanded_surface(aabb, smin, smax) {
                continue;
            }
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if p.x < smin.x
                    || p.x > smax.x
                    || p.y < smin.y
                    || p.y > smax.y
                    || p.z < smin.z
                    || p.z > smax.z
                {
                    continue;
                }
                if let Some((_, d2)) = grid.nearest_vertex(&surface.vertices, p)
                    && d2 <= depth2
                {
                    result.insert(StreamlineIndex(si as u32));
                    break;
                }
            }
        }
        result
    }

    /// Project selected streamlines onto surface vertices.
    /// Returns (density_count, mean_dps_value_or_nan) per surface vertex.
    pub fn project_selected_to_surface(
        &self,
        surface: &GiftiSurfaceData,
        selected_streamlines: &[StreamlineIndex],
        depth_mm: Millimeters,
        dps_values: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = surface.vertices.len();
        let mut density = vec![0.0f32; n];
        let mut dps_sum = vec![0.0f32; n];
        let mut dps_count = vec![0u32; n];
        if depth_mm.0 <= 0.0 || n == 0 {
            return (density, vec![f32::NAN; n]);
        }

        let depth2 = depth_mm.0 * depth_mm.0;
        let smin = surface.bbox_min - Vec3::splat(depth_mm.0);
        let smax = surface.bbox_max + Vec3::splat(depth_mm.0);
        let grid = SurfaceSpatialGrid::build(&surface.vertices, depth_mm.0.max(0.5));

        for &si_u32 in selected_streamlines {
            let si = si_u32.0 as usize;
            if si + 1 >= self.offsets.len() {
                continue;
            }
            let dps_val = dps_values.and_then(|d| d.get(si)).copied().unwrap_or(0.0);
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if p.x < smin.x
                    || p.x > smax.x
                    || p.y < smin.y
                    || p.y > smax.y
                    || p.z < smin.z
                    || p.z > smax.z
                {
                    continue;
                }
                if let Some((nearest_idx, d2)) = grid.nearest_vertex(&surface.vertices, p)
                    && d2 <= depth2
                {
                    density[nearest_idx] += 1.0;
                    if dps_values.is_some() {
                        dps_sum[nearest_idx] += dps_val;
                        dps_count[nearest_idx] += 1;
                    }
                }
            }
        }

        let mean_dps: Vec<f32> = dps_sum
            .iter()
            .zip(dps_count.iter())
            .map(|(&sum, &cnt)| if cnt > 0 { sum / cnt as f32 } else { f32::NAN })
            .collect();

        (density, mean_dps)
    }

    fn compute_group_colors(&self) -> Vec<[f32; 4]> {
        let mut colors = vec![[0.5f32, 0.5, 0.5, 1.0]; self.nb_vertices];

        // Fallback palette when group name is not in the reference dict.
        let palette: &[[f32; 4]] = &[
            [1.0, 0.2, 0.2, 1.0], // red
            [0.2, 0.7, 1.0, 1.0], // blue
            [0.2, 1.0, 0.3, 1.0], // green
            [1.0, 0.8, 0.1, 1.0], // yellow
            [1.0, 0.4, 0.8, 1.0], // pink
            [0.6, 0.3, 1.0, 1.0], // purple
            [1.0, 0.6, 0.2, 1.0], // orange
            [0.3, 1.0, 0.8, 1.0], // cyan
        ];

        for (gi, (name, members)) in self.groups.iter().enumerate() {
            let color = self
                .group_colors
                .get(gi)
                .copied()
                .flatten()
                .or_else(|| group_name_color(name))
                .unwrap_or(palette[gi % palette.len()]);
            for &streamline_idx in members {
                let si = streamline_idx.0 as usize;
                if si + 1 < self.offsets.len() {
                    let start = self.offsets[si] as usize;
                    let end = self.offsets[si + 1] as usize;
                    for vertex_color in colors
                        .iter_mut()
                        .take(end.min(self.nb_vertices))
                        .skip(start)
                    {
                        *vertex_color = color;
                    }
                }
            }
        }
        colors
    }

    /// Return the world-space positions and current vertex colors for a set of
    /// selected streamlines (by streamline index).  Used to feed the bundle mesh.
    pub fn selected_vertex_data(
        &self,
        selected_streamlines: &[StreamlineIndex],
    ) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
        let total: usize = selected_streamlines
            .iter()
            .map(|&si| (self.offsets[si.0 as usize + 1] - self.offsets[si.0 as usize]) as usize)
            .sum();
        let mut positions = Vec::with_capacity(total);
        let mut colors = Vec::with_capacity(total);
        for &si in selected_streamlines {
            let start = self.offsets[si.0 as usize] as usize;
            let end = self.offsets[si.0 as usize + 1] as usize;
            positions.extend_from_slice(&self.positions[start..end]);
            colors.extend_from_slice(&self.colors[start..end]);
        }
        (positions, colors)
    }

    /// Gather selected streamline positions/colors into contiguous arrays and offsets.
    pub fn selected_tube_data(
        &self,
        selected_streamlines: &[StreamlineIndex],
    ) -> (Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>) {
        let total: usize = selected_streamlines
            .iter()
            .map(|&si| (self.offsets[si.0 as usize + 1] - self.offsets[si.0 as usize]) as usize)
            .sum();
        let mut positions = Vec::with_capacity(total);
        let mut colors = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(selected_streamlines.len() + 1);
        offsets.push(0);

        for &si in selected_streamlines {
            let start = self.offsets[si.0 as usize] as usize;
            let end = self.offsets[si.0 as usize + 1] as usize;
            positions.extend_from_slice(&self.positions[start..end]);
            colors.extend_from_slice(&self.colors[start..end]);
            offsets.push(positions.len() as u32);
        }

        (positions, colors, offsets)
    }

    pub fn subset_streamlines(&self, selected_streamlines: &[StreamlineIndex]) -> Self {
        let total_vertices: usize = selected_streamlines
            .iter()
            .map(|&si| (self.offsets[si.0 as usize + 1] - self.offsets[si.0 as usize]) as usize)
            .sum();

        let mut positions = Vec::with_capacity(total_vertices);
        let mut tangents = Vec::with_capacity(total_vertices);
        let mut colors = Vec::with_capacity(total_vertices);
        let mut offsets = Vec::with_capacity(selected_streamlines.len() + 1);
        let mut old_to_new = HashMap::new();
        offsets.push(0);

        for (new_index, &old_index_u32) in selected_streamlines.iter().enumerate() {
            let old_index = old_index_u32.0 as usize;
            let start = self.offsets[old_index] as usize;
            let end = self.offsets[old_index + 1] as usize;
            positions.extend_from_slice(&self.positions[start..end]);
            tangents.extend_from_slice(&self.tangents[start..end]);
            colors.extend_from_slice(&self.colors[start..end]);
            offsets.push(positions.len() as u32);
            old_to_new.insert(old_index_u32, StreamlineIndex(new_index as u32));
        }

        let groups: Vec<(String, Vec<StreamlineIndex>)> = self
            .groups
            .iter()
            .map(|(name, members)| {
                let remapped = members
                    .iter()
                    .filter_map(|member| old_to_new.get(member).copied())
                    .collect();
                (name.clone(), remapped)
            })
            .collect();

        let dps_data = self
            .dps_data
            .iter()
            .map(|(name, data)| {
                let subset = selected_streamlines
                    .iter()
                    .filter_map(|index| data.get(index.0 as usize).copied())
                    .collect();
                (name.clone(), subset)
            })
            .collect();

        let dpv_data = self
            .dpv_data
            .iter()
            .map(|(name, data)| {
                let mut subset = Vec::with_capacity(total_vertices);
                for &streamline_index in selected_streamlines {
                    let start = self.offsets[streamline_index.0 as usize] as usize;
                    let end = self.offsets[streamline_index.0 as usize + 1] as usize;
                    subset.extend_from_slice(&data[start..end]);
                }
                (name.clone(), subset)
            })
            .collect();

        let (bbox_min, bbox_max) = compute_bounding_box(&positions);
        let all_indices = build_line_indices(&offsets);
        let aabbs = build_streamline_aabbs_from_slices(&positions, &offsets);

        Self {
            positions,
            offsets,
            bbox_min,
            bbox_max,
            nb_streamlines: selected_streamlines.len(),
            nb_vertices: total_vertices,
            dpv_names: self.dpv_names.clone(),
            dps_names: self.dps_names.clone(),
            groups,
            group_colors: self.group_colors.clone(),
            dpv_data,
            dps_data,
            tangents,
            colors,
            all_indices,
            aabbs,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.bbox_min + self.bbox_max) * 0.5
    }

    pub fn extent(&self) -> f32 {
        (self.bbox_max - self.bbox_min).length()
    }
}

/// Extract DPV, DPS, and group data from a TRX file.
struct ExtractedMetadata {
    dpv_data: Vec<(String, Vec<f32>)>,
    dps_data: Vec<(String, Vec<f32>)>,
    groups: Vec<(String, Vec<StreamlineIndex>)>,
    group_colors: Vec<Option<[f32; 4]>>,
}

fn extract_metadata(
    trx: &AnyTrxFile,
    nb_vertices: usize,
    nb_streamlines: usize,
) -> ExtractedMetadata {
    let dpv_data = trx
        .dpv_entries()
        .into_iter()
        .filter(|(_, info)| info.ncols == 1)
        .filter_map(|(name, _)| {
            trx.scalar_dpv_f32(&name)
                .ok()
                .filter(|flat| flat.len() == nb_vertices)
                .map(|flat| (name, flat))
        })
        .collect();

    let dps_data = trx
        .dps_entries()
        .into_iter()
        .filter(|(_, info)| info.ncols == 1)
        .filter_map(|(name, _)| {
            trx.scalar_dps_f32(&name)
                .ok()
                .filter(|flat| flat.len() == nb_streamlines)
                .map(|flat| (name, flat))
        })
        .collect();

    let groups: Vec<(String, Vec<StreamlineIndex>)> = trx
        .groups_owned()
        .into_iter()
        .map(|(name, members)| {
            (
                name,
                members.into_iter().map(StreamlineIndex).collect::<Vec<_>>(),
            )
        })
        .collect();
    let group_colors = groups
        .iter()
        .map(|(name, _)| extract_group_color_from_any(trx, name))
        .collect();

    ExtractedMetadata {
        dpv_data,
        dps_data,
        groups,
        group_colors,
    }
}

fn compute_bounding_box(positions: &[[f32; 3]]) -> (Vec3, Vec3) {
    let mut bbox_min = Vec3::splat(f32::INFINITY);
    let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        let v = Vec3::from(*p);
        bbox_min = bbox_min.min(v);
        bbox_max = bbox_max.max(v);
    }
    (bbox_min, bbox_max)
}

fn extract_group_color_from_any(trx: &AnyTrxFile, group: &str) -> Option<[f32; 4]> {
    trx.with_typed(
        |inner| {
            inner
                .dpg::<u8>(group, "color")
                .ok()
                .and_then(color_view_to_rgba)
        },
        |inner| {
            inner
                .dpg::<u8>(group, "color")
                .ok()
                .and_then(color_view_to_rgba)
        },
        |inner| {
            inner
                .dpg::<u8>(group, "color")
                .ok()
                .and_then(color_view_to_rgba)
        },
    )
}

fn extract_group_color(entries: &HashMap<String, DataArray>) -> Option<[f32; 4]> {
    entries.get("color").and_then(|array| {
        if array.dtype() != trx_rs::DType::UInt8 || array.ncols() != 3 || array.nrows() != 1 {
            return None;
        }
        let view = array.typed_view::<u8>();
        color_view_to_rgba(view)
    })
}

fn color_view_to_rgba(view: trx_rs::TypedView2D<'_, u8>) -> Option<[f32; 4]> {
    if view.nrows() != 1 || view.ncols() != 3 {
        return None;
    }
    let row = view.row(0);
    Some([
        row[0] as f32 / 255.0,
        row[1] as f32 / 255.0,
        row[2] as f32 / 255.0,
        1.0,
    ])
}

/// Compute per-vertex tangent directions (unsigned, normalized).
pub fn compute_tangents(positions: &[[f32; 3]], offsets: &[u32]) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut tangents = vec![[0.5f32, 0.5, 0.5]; n];

    for win in offsets.windows(2) {
        let start = win[0] as usize;
        let end = win[1] as usize;
        if end - start < 2 {
            continue;
        }
        for i in start..end {
            let raw = if i == start {
                Vec3::from(positions[i + 1]) - Vec3::from(positions[i])
            } else if i == end - 1 {
                Vec3::from(positions[i]) - Vec3::from(positions[i - 1])
            } else {
                Vec3::from(positions[i + 1]) - Vec3::from(positions[i - 1])
            };
            let t = raw.normalize_or_zero();
            tangents[i] = t.into();
        }
    }

    tangents
}

/// Compute direction-RGB colors from pre-computed tangents.
fn direction_colors_from_tangents(tangents: &[[f32; 3]]) -> Vec<[f32; 4]> {
    tangents
        .iter()
        .map(|&t| {
            let v = Vec3::from(t).abs();
            [v.x, v.y, v.z, 1.0]
        })
        .collect()
}

/// Build line-segment indices for PrimitiveTopology::LineList.
fn build_line_indices(offsets: &[u32]) -> Vec<u32> {
    let mut indices = Vec::new();
    for win in offsets.windows(2) {
        let start = win[0];
        let end = win[1];
        for i in start..end.saturating_sub(1) {
            indices.push(i);
            indices.push(i + 1);
        }
    }
    indices
}

/// Compute the robust scalar range (2nd–98th percentile) for a slice of values.
pub fn scalar_auto_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo = finite[((n as f32 * 0.02) as usize).min(n - 1)];
    let hi = finite[((n as f32 * 0.98) as usize).min(n - 1)];
    (lo, hi.max(lo + 1e-6))
}

/// Map scalar values to the chosen colormap using an explicit range.
pub fn scalar_to_colors_ranged(
    values: &[f32],
    min_v: f32,
    max_v: f32,
    colormap: crate::renderer::mesh_renderer::SurfaceColormap,
) -> Vec<[f32; 4]> {
    let range = (max_v - min_v).max(1e-10);
    values
        .iter()
        .map(|&v| {
            let t = ((v - min_v) / range).clamp(0.0, 1.0);
            let [r, g, b] = crate::renderer::colormap::surface_colormap_rgb(t, colormap);
            [r, g, b, 1.0]
        })
        .collect()
}

/// Blue (0) → White (0.5) → Red (1) colormap at parameter t ∈ [0, 1].
pub fn colormap_bwr(t: f32) -> [f32; 4] {
    if t < 0.5 {
        let s = t * 2.0;
        [s, s, 1.0, 1.0]
    } else {
        let s = (1.0 - t) * 2.0;
        [1.0, s, s, 1.0]
    }
}

#[derive(Clone, Copy)]
struct TubePoint {
    position: [f32; 3],
    color: [f32; 4],
}

fn simplify_streamline_points(positions: &[[f32; 3]], colors: &[[f32; 4]]) -> Vec<TubePoint> {
    let mut points = Vec::with_capacity(positions.len());
    let mut last: Option<Vec3> = None;

    for (&position, &color) in positions.iter().zip(colors.iter()) {
        let pos = Vec3::from(position);
        if last.is_some_and(|prev| prev.distance_squared(pos) < 1e-8) {
            continue;
        }
        points.push(TubePoint { position, color });
        last = Some(pos);
    }

    points
}

pub fn build_tube_vertices_from_data(
    positions: &[[f32; 3]],
    colors: &[[f32; 4]],
    offsets: &[u32],
    radius: f32,
    sides: u32,
) -> (Vec<TubeMeshVertex>, Vec<u32>) {
    let mut vertices: Vec<TubeMeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let sides = sides.max(3) as usize;
    let radius = radius.max(0.001);

    for win in offsets.windows(2) {
        let start = win[0] as usize;
        let end = win[1] as usize;
        let points = simplify_streamline_points(&positions[start..end], &colors[start..end]);
        if points.len() < 2 {
            continue;
        }

        let tangents = compute_streamline_tangents(&points);
        let frame_normals = build_parallel_transport_frames(&tangents);
        let ring_base = vertices.len() as u32;

        for ((point, tangent), frame_normal) in
            points.iter().zip(tangents.iter()).zip(frame_normals.iter())
        {
            let center = Vec3::from(point.position);
            let normal = *frame_normal;
            let mut binormal = tangent.cross(normal).normalize_or_zero();
            if binormal.length_squared() < 1e-8 {
                binormal = arbitrary_perpendicular(*tangent);
            }

            for side in 0..sides {
                let theta = std::f32::consts::TAU * (side as f32 / sides as f32);
                let radial = normal * theta.cos() + binormal * theta.sin();
                vertices.push(TubeMeshVertex {
                    position: (center + radial * radius).to_array(),
                    normal: radial.to_array(),
                    color: point.color,
                });
            }
        }

        let ring_count = points.len() as u32;
        for ring in 0..ring_count.saturating_sub(1) {
            let curr = ring_base + ring * sides as u32;
            let next = curr + sides as u32;
            for side in 0..sides as u32 {
                let side_next = (side + 1) % sides as u32;
                indices.extend_from_slice(&[
                    curr + side,
                    next + side,
                    next + side_next,
                    curr + side,
                    next + side_next,
                    curr + side_next,
                ]);
            }
        }

        let start_center = vertices.len() as u32;
        vertices.push(TubeMeshVertex {
            position: points[0].position,
            normal: (-tangents[0]).to_array(),
            color: points[0].color,
        });
        for side in 0..sides as u32 {
            let side_next = (side + 1) % sides as u32;
            indices.extend_from_slice(&[start_center, ring_base + side_next, ring_base + side]);
        }

        let end_center = vertices.len() as u32;
        let last_ring = ring_base + (ring_count - 1) * sides as u32;
        let last_point = *points.last().unwrap();
        let last_tangent = *tangents.last().unwrap();
        vertices.push(TubeMeshVertex {
            position: last_point.position,
            normal: last_tangent.to_array(),
            color: last_point.color,
        });
        for side in 0..sides as u32 {
            let side_next = (side + 1) % sides as u32;
            indices.extend_from_slice(&[end_center, last_ring + side, last_ring + side_next]);
        }
    }

    (vertices, indices)
}

fn compute_streamline_tangents(points: &[TubePoint]) -> Vec<Vec3> {
    let mut tangents = Vec::with_capacity(points.len());

    for i in 0..points.len() {
        let tangent = if i == 0 {
            Vec3::from(points[1].position) - Vec3::from(points[0].position)
        } else if i + 1 == points.len() {
            Vec3::from(points[i].position) - Vec3::from(points[i - 1].position)
        } else {
            Vec3::from(points[i + 1].position) - Vec3::from(points[i - 1].position)
        };
        tangents.push(safe_normalize(tangent));
    }

    tangents
}

fn build_parallel_transport_frames(tangents: &[Vec3]) -> Vec<Vec3> {
    let mut normals = Vec::with_capacity(tangents.len());
    if tangents.is_empty() {
        return normals;
    }

    let mut normal = arbitrary_perpendicular(tangents[0]);
    normals.push(normal);

    for window in tangents.windows(2) {
        let prev_t = window[0];
        let curr_t = window[1];
        let axis = prev_t.cross(curr_t);

        if axis.length_squared() > 1e-8 {
            let angle = prev_t.dot(curr_t).clamp(-1.0, 1.0).acos();
            normal = glam::Quat::from_axis_angle(axis.normalize(), angle) * normal;
        }

        normal = (normal - curr_t * normal.dot(curr_t)).normalize_or_zero();
        if normal.length_squared() < 1e-8 {
            normal = arbitrary_perpendicular(curr_t);
        }
        normals.push(normal);
    }

    normals
}

fn arbitrary_perpendicular(tangent: Vec3) -> Vec3 {
    let tangent = safe_normalize(tangent);
    let reference = if tangent.z.abs() < 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let perp = tangent.cross(reference);
    if perp.length_squared() < 1e-8 {
        Vec3::X
    } else {
        perp.normalize()
    }
}

fn safe_normalize(v: Vec3) -> Vec3 {
    let n = v.normalize_or_zero();
    if n.length_squared() < 1e-8 {
        Vec3::X
    } else {
        n
    }
}

/// Expand per-streamline scalar values to per-vertex colors using an explicit range.
fn expand_dps_to_vertices_ranged(
    dps_values: &[f32],
    offsets: &[u32],
    nb_vertices: usize,
    min_v: f32,
    max_v: f32,
    colormap: crate::renderer::mesh_renderer::SurfaceColormap,
) -> Vec<[f32; 4]> {
    let mut per_vertex = vec![0.0f32; nb_vertices];
    for (si, &val) in dps_values.iter().enumerate() {
        if si + 1 < offsets.len() {
            let start = offsets[si] as usize;
            let end = offsets[si + 1] as usize;
            for vertex_value in per_vertex.iter_mut().take(end.min(nb_vertices)).skip(start) {
                *vertex_value = val;
            }
        }
    }
    scalar_to_colors_ranged(&per_vertex, min_v, max_v, colormap)
}

fn aabb_overlaps_expanded_surface(aabb: &StreamlineAabb, smin: Vec3, smax: Vec3) -> bool {
    aabb.overlaps_box(smin.to_array(), smax.to_array())
}

struct SurfaceSpatialGrid {
    cell_size: f32,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SurfaceSpatialGrid {
    fn build(vertices: &[[f32; 3]], cell_size: f32) -> Self {
        let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, v) in vertices.iter().enumerate() {
            let key = Self::key(Vec3::from(*v), cell_size);
            buckets.entry(key).or_default().push(i);
        }
        Self { cell_size, buckets }
    }

    fn key(p: Vec3, cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
    }

    fn nearest_vertex(&self, vertices: &[[f32; 3]], p: Vec3) -> Option<(usize, f32)> {
        let (cx, cy, cz) = Self::key(p, self.cell_size);
        let mut best: Option<(usize, f32)> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    let Some(list) = self.buckets.get(&key) else {
                        continue;
                    };
                    for &idx in list {
                        let d2 = (Vec3::from(vertices[idx]) - p).length_squared();
                        match best {
                            Some((_, best_d2)) if d2 >= best_d2 => {}
                            _ => best = Some((idx, d2)),
                        }
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_tractogram_preserves_geometry_and_groups() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .unwrap();
        tractogram.push_streamline(&[[7.0, 8.0, 9.0]]).unwrap();
        tractogram.insert_group("bundle_a", vec![0]);

        let gpu = TrxGpuData::from_tractogram(&tractogram).unwrap();
        assert_eq!(gpu.nb_streamlines, 2);
        assert_eq!(gpu.nb_vertices, 3);
        assert_eq!(gpu.offsets, vec![0, 2, 3]);
        assert_eq!(gpu.groups.len(), 1);
        assert_eq!(gpu.groups[0].0, "bundle_a");
        assert!(gpu.dpv_data.is_empty());
        assert!(gpu.dps_data.is_empty());
    }

    #[test]
    fn imported_dpg_color_overrides_name_palette() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .unwrap();
        tractogram.insert_group("bundle_a", vec![0]);
        tractogram.insert_dpg(
            "bundle_a",
            "color",
            DataArray::owned_bytes(vec![10, 20, 30], 3, trx_rs::DType::UInt8),
        );

        let mut gpu = TrxGpuData::from_tractogram(&tractogram).unwrap();
        gpu.recolor(
            &ColorMode::Group,
            None,
            crate::renderer::mesh_renderer::SurfaceColormap::default(),
        );
        assert_eq!(
            gpu.colors[0],
            [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0]
        );
        assert_eq!(
            gpu.colors[1],
            [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0]
        );
    }

    #[test]
    fn group_name_palette_used_when_no_dpg_color_is_present() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .unwrap();
        tractogram.insert_group("AF_L", vec![0]);

        let mut gpu = TrxGpuData::from_tractogram(&tractogram).unwrap();
        gpu.recolor(
            &ColorMode::Group,
            None,
            crate::renderer::mesh_renderer::SurfaceColormap::default(),
        );
        assert_eq!(gpu.colors[0], [0.09, 0.745, 0.812, 1.0]);
    }
}

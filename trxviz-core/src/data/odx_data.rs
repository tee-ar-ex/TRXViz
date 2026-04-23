use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec3, Vec4};
use odx_rs::formats::dsistudio_odf8;
use odx_rs::typed_view::TypedView2D;
use odx_rs::{OdxDataset, mrtrix_sh};

use crate::data::cifti::VolumeScalars;
use crate::data::loaded_files::FileId;
use crate::data::nifti_data::NiftiVolume;
use crate::renderer::glyph_renderer::GlyphInstance;

pub use odx_rs::qc::{FixelOtsu, OtsuScope};

/// Loaded ODX dataset with precomputed spatial lookups for slice-local rendering.
pub struct OdxScene {
    dataset: OdxDataset,
    /// `[i, j, k]` grid position for each compact (masked) voxel.
    ijk_lookup: Vec<[u32; 3]>,
    /// Compact voxel indices grouped by slice for each ODX axis (i, j, k).
    slice_compact_indices: [Vec<Vec<usize>>; 3],
    /// RAS+ center for each compact voxel.
    centers_ras: Vec<[f32; 3]>,
    odf_source: Option<OdfGlyphSource>,
    sh_source: Option<ShGlyphSource>,
    glyph_warnings: Vec<String>,
    odf_slice_cache: Mutex<OdfSliceMetadataCache>,
    sh_render_mesh_cache: Mutex<HashMap<u32, Arc<ShRenderMesh>>>,
    /// Memoized per-(metric, scope) Otsu thresholds. Pre-populated with
    /// the "default" entry (auto-resolved metric, all-fixel scope) at
    /// scene construction time so display ops can query without paying
    /// compute latency.
    fixel_otsu_cache: Arc<std::sync::RwLock<HashMap<(String, OtsuScope), FixelOtsu>>>,
    default_fixel_otsu: Option<FixelOtsu>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OdxGlyphSourceKind {
    Odf,
    Sh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OdfSampleDomain {
    FullSphere,
    Hemisphere,
}

impl OdfSampleDomain {
    fn is_hemisphere(self) -> bool {
        matches!(self, Self::Hemisphere)
    }
}

#[derive(Debug)]
struct OdfGlyphSource {
    name: String,
    sample_domain: OdfSampleDomain,
    ncols: usize,
    render_vertices: Vec<[f32; 3]>,
    render_indices: Vec<u32>,
}

#[derive(Debug)]
struct ShGlyphSource {
    name: String,
    ncoeffs: usize,
    sh_order: usize,
}

pub struct ShRenderMesh {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    sample_plan: mrtrix_sh::RowSamplePlan,
}

impl ShRenderMesh {
    pub fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn row_width(&self) -> usize {
        self.vertices.len()
    }

    pub fn transform_flat(&self) -> &[f32] {
        self.sample_plan.transform_flat()
    }

    pub fn source_dir_count(&self) -> usize {
        self.sample_plan.source_dir_count()
    }

    pub fn sample_plan(&self) -> &mrtrix_sh::RowSamplePlan {
        &self.sample_plan
    }
}

pub struct SliceGlyphData {
    pub instances: Vec<GlyphInstance>,
    pub amplitudes: Vec<f32>,
    pub amp_norm: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OdfChunkWorkItem {
    pub local_row: u32,
    pub output_row: u32,
}

#[derive(Clone, Debug)]
pub struct OdfChunkWorklist {
    pub chunk_index: usize,
    pub work_items: Vec<OdfChunkWorkItem>,
}

#[derive(Clone)]
pub struct OdfSliceMetadata {
    pub instances: Vec<GlyphInstance>,
    pub chunk_worklists: Vec<OdfChunkWorklist>,
    pub amp_norm: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OdfAmplitudeConditioning {
    pub subtract_iso: bool,
    pub norm_within_voxel: bool,
}

impl OdfAmplitudeConditioning {
    pub const fn new(subtract_iso: bool, norm_within_voxel: bool) -> Self {
        Self {
            subtract_iso,
            norm_within_voxel,
        }
    }
}

#[derive(Default)]
struct OdfSliceMetadataCache {
    order: VecDeque<(usize, u32, usize)>,
    entries: HashMap<(usize, u32, usize), Arc<OdfSliceMetadata>>,
}

impl Default for SliceGlyphData {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            amplitudes: Vec::new(),
            amp_norm: 1.0,
        }
    }
}

/// A fixel (peak) line-segment instance ready for GPU upload.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FixelInstance {
    pub center: [f32; 3],
    pub direction: [f32; 3],
    pub length: f32,
    pub scalar: f32,
}

impl OdxScene {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let dataset = OdxDataset::open(path)?;
        Self::from_dataset(dataset)
    }

    pub fn from_dataset(dataset: OdxDataset) -> anyhow::Result<Self> {
        let ijk_lookup = dataset.compact_to_ijk();
        let dims = dataset.header().dimensions;
        let mut slice_compact_indices: [Vec<Vec<usize>>; 3] =
            std::array::from_fn(|axis| vec![Vec::new(); dims[axis] as usize]);
        for (compact_idx, ijk) in ijk_lookup.iter().enumerate() {
            for axis in 0..3 {
                if let Some(entries) = slice_compact_indices[axis].get_mut(ijk[axis] as usize) {
                    entries.push(compact_idx);
                }
            }
        }
        let centers_ras = dataset.mask_voxel_centers_ras();
        let mut glyph_warnings = Vec::new();
        let odf_source = resolve_odf_source(&dataset, &mut glyph_warnings);
        let sh_source = resolve_sh_source(&dataset, &mut glyph_warnings);

        // Eagerly resolve the default Otsu so every display op sees it
        // without paying compute latency. Silent on failure — data
        // without a usable DPF metric simply has no default threshold.
        let default_fixel_otsu =
            odx_rs::qc::compute_fixel_otsu(&dataset, None, OtsuScope::AllFixels).ok();
        let mut cache = HashMap::new();
        if let Some(otsu) = &default_fixel_otsu {
            cache.insert((otsu.metric_name.clone(), otsu.scope), otsu.clone());
        }

        Ok(Self {
            dataset,
            ijk_lookup,
            slice_compact_indices,
            centers_ras,
            odf_source,
            sh_source,
            glyph_warnings,
            odf_slice_cache: Mutex::new(OdfSliceMetadataCache::default()),
            sh_render_mesh_cache: Mutex::new(HashMap::new()),
            fixel_otsu_cache: Arc::new(std::sync::RwLock::new(cache)),
            default_fixel_otsu,
        })
    }

    /// Return the Otsu threshold for the given metric + scope, computing
    /// and memoizing it on first request. `metric = None` auto-resolves
    /// in the odx-rs priority order (amplitude → afd → qa).
    pub fn fixel_otsu(
        &self,
        metric: Option<&str>,
        scope: OtsuScope,
    ) -> anyhow::Result<FixelOtsu> {
        // Fast path: if the caller's `metric` matches the eager default
        // entry's resolved name, skip the lookup-then-write dance.
        if let Some(entry) = self.default_fixel_otsu.as_ref() {
            if scope == entry.scope
                && (metric.is_none() || metric == Some(entry.metric_name.as_str()))
            {
                return Ok(entry.clone());
            }
        }
        // Try the cache.
        if let Some(name) = metric {
            let key = (name.to_string(), scope);
            if let Some(hit) = self.fixel_otsu_cache.read().ok().and_then(|c| c.get(&key).cloned()) {
                return Ok(hit);
            }
        }
        // Compute + memoize. For `metric = None` the resolved name isn't
        // known until after the call, so we key by the resolved name.
        let result = odx_rs::qc::compute_fixel_otsu(&self.dataset, metric, scope)?;
        if let Ok(mut cache) = self.fixel_otsu_cache.write() {
            cache.insert((result.metric_name.clone(), result.scope), result.clone());
        }
        Ok(result)
    }

    /// Eager default Otsu, computed at scene construction time from the
    /// auto-resolved tracking metric and `AllFixels` scope. `None` when
    /// the ODX has no usable DPF metric.
    pub fn default_fixel_otsu(&self) -> Option<&FixelOtsu> {
        self.default_fixel_otsu.as_ref()
    }

    /// Number of full-volume masked voxels.
    pub fn nb_voxels(&self) -> usize {
        self.ijk_lookup.len()
    }

    pub fn compact_voxel_count(&self) -> usize {
        self.ijk_lookup.len()
    }

    pub fn has_glyph_field(&self) -> bool {
        self.glyph_source_kind().is_some()
    }

    pub fn glyph_source_kind(&self) -> Option<OdxGlyphSourceKind> {
        if self.sh_source.is_some() {
            Some(OdxGlyphSourceKind::Sh)
        } else if self.odf_source.is_some() {
            Some(OdxGlyphSourceKind::Odf)
        } else {
            None
        }
    }

    pub fn glyph_warnings(&self) -> &[String] {
        &self.glyph_warnings
    }

    pub fn odf_render_geometry(&self) -> Option<(&[[f32; 3]], &[u32])> {
        self.odf_source.as_ref().map(|source| {
            (
                source.render_vertices.as_slice(),
                source.render_indices.as_slice(),
            )
        })
    }

    pub fn odf_render_row_width(&self) -> Option<usize> {
        self.odf_source
            .as_ref()
            .map(|source| source.render_vertices.len())
    }

    pub fn sh_render_mesh(&self, detail: u32) -> Option<Arc<ShRenderMesh>> {
        self.sh_source.as_ref()?;
        let detail = detail.max(1);
        if let Ok(cache) = self.sh_render_mesh_cache.lock()
            && let Some(mesh) = cache.get(&detail)
        {
            return Some(mesh.clone());
        }

        let (vertices, indices) = build_full_icosphere_mesh(detail);
        let ncoeffs = self.sh_source.as_ref()?.ncoeffs;
        let sample_plan =
            mrtrix_sh::RowSamplePlan::for_sh_rows_nonnegative(&vertices, ncoeffs).ok()?;
        let mesh = Arc::new(ShRenderMesh {
            vertices,
            indices,
            sample_plan,
        });
        if let Ok(mut cache) = self.sh_render_mesh_cache.lock() {
            cache.insert(detail, mesh.clone());
        }
        Some(mesh)
    }

    pub fn clamp_sh_detail_for_slice(
        &self,
        axis: usize,
        slice_idx: u32,
        requested_detail: u32,
        max_storage_bytes: usize,
    ) -> u32 {
        if self.sh_source.is_none() {
            return requested_detail.max(1);
        }
        let requested_detail = requested_detail.max(1);
        let instance_count = self.slice_compact_indices(axis, slice_idx).len();
        if instance_count == 0 {
            return requested_detail;
        }

        let mut safe_detail = 1u32;
        for detail in 1..=requested_detail {
            let Some(row_width) = self.sh_render_row_width(detail) else {
                break;
            };
            let needed = instance_count
                .saturating_mul(row_width)
                .saturating_mul(std::mem::size_of::<f32>());
            if needed <= max_storage_bytes {
                safe_detail = detail;
            } else {
                break;
            }
        }
        safe_detail
    }

    pub fn max_sh_detail_for_slice(
        &self,
        axis: usize,
        slice_idx: u32,
        max_storage_bytes: usize,
        max_detail: u32,
    ) -> u32 {
        self.clamp_sh_detail_for_slice(axis, slice_idx, max_detail, max_storage_bytes)
    }

    pub fn glyph_instances_full_volume(&self, row_width: usize) -> Vec<GlyphInstance> {
        let scale = self.default_glyph_scale();
        self.centers_ras
            .iter()
            .enumerate()
            .map(|(compact_idx, &center)| GlyphInstance {
                center,
                scale,
                amplitude_offset: (compact_idx * row_width) as u32,
                min_contacts: 0,
                contact_count: 1,
                _pad: 0,
            })
            .collect()
    }

    pub fn glyph_instances_for_slice(
        &self,
        axis: usize,
        slice_idx: u32,
        row_width: usize,
    ) -> Vec<GlyphInstance> {
        let scale = self.default_glyph_scale();
        self.slice_compact_indices(axis, slice_idx)
            .iter()
            .enumerate()
            .map(|(local_idx, &compact_idx)| GlyphInstance {
                center: self.centers_ras[compact_idx],
                scale,
                amplitude_offset: (local_idx * row_width) as u32,
                min_contacts: 0,
                contact_count: 1,
                _pad: 0,
            })
            .collect()
    }

    pub fn slice_compact_indices(&self, axis: usize, slice_idx: u32) -> &[usize] {
        self.slice_compact_indices
            .get(axis)
            .and_then(|slices| slices.get(slice_idx as usize))
            .map(|indices| indices.as_slice())
            .unwrap_or(&[])
    }

    pub fn nearest_nonempty_slice(&self, axis: usize, preferred: u32) -> Option<u32> {
        let slices = self.slice_compact_indices.get(axis)?;
        if slices.is_empty() {
            return None;
        }
        let preferred = preferred.min((slices.len().saturating_sub(1)) as u32) as usize;
        if !slices[preferred].is_empty() {
            return Some(preferred as u32);
        }
        for delta in 1..slices.len() {
            if let Some(lower) = preferred.checked_sub(delta)
                && !slices[lower].is_empty()
            {
                return Some(lower as u32);
            }
            let upper = preferred + delta;
            if upper < slices.len() && !slices[upper].is_empty() {
                return Some(upper as u32);
            }
        }
        None
    }

    pub fn odf_view_f32(&self) -> Option<TypedView2D<'_, f32>> {
        self.odf_source
            .as_ref()
            .and_then(|source| self.dataset.odf::<f32>(&source.name).ok())
    }

    pub fn odf_amplitudes_full_sphere(&self) -> Option<Vec<f32>> {
        let source = self.odf_source.as_ref()?;
        let view = self.odf_view_f32()?;
        if !source.sample_domain.is_hemisphere() {
            return Some(view.as_flat_slice().to_vec());
        }
        let mut out = Vec::with_capacity(view.nrows() * source.render_vertices.len());
        for row in view.rows() {
            append_mirrored_hemisphere_row(row, &mut out);
        }
        Some(out)
    }

    pub fn conditioned_odf_amplitudes_full_sphere(
        &self,
        conditioning: OdfAmplitudeConditioning,
    ) -> Option<Vec<f32>> {
        let source = self.odf_source.as_ref()?;
        let view = self.odf_view_f32()?;
        let peak_target = self.default_normalized_peak_length_mm();
        let mut out = Vec::with_capacity(view.nrows() * source.render_vertices.len());
        for row in view.rows() {
            let row_start = out.len();
            if source.sample_domain.is_hemisphere() {
                append_mirrored_hemisphere_row(row, &mut out);
            } else {
                out.extend_from_slice(row);
            }
            condition_odf_amplitudes_in_place(&mut out[row_start..], conditioning, peak_target);
        }
        Some(out)
    }

    pub fn odf_amplitudes_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        let source = self.odf_source.as_ref()?;
        let view = self.odf_view_f32()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let mut out = Vec::with_capacity(slice_indices.len() * source.render_vertices.len());
        for &compact_idx in slice_indices {
            let row = view.row(compact_idx);
            if source.sample_domain.is_hemisphere() {
                append_mirrored_hemisphere_row(row, &mut out);
            } else {
                out.extend_from_slice(row);
            }
        }
        Some(out)
    }

    pub fn conditioned_odf_amplitudes_for_slice(
        &self,
        axis: usize,
        slice_idx: u32,
        conditioning: OdfAmplitudeConditioning,
    ) -> Option<Vec<f32>> {
        let source = self.odf_source.as_ref()?;
        let view = self.odf_view_f32()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let peak_target = self.default_normalized_peak_length_mm();
        let mut out = Vec::with_capacity(slice_indices.len() * source.render_vertices.len());
        for &compact_idx in slice_indices {
            let row_start = out.len();
            let row = view.row(compact_idx);
            if source.sample_domain.is_hemisphere() {
                append_mirrored_hemisphere_row(row, &mut out);
            } else {
                out.extend_from_slice(row);
            }
            condition_odf_amplitudes_in_place(&mut out[row_start..], conditioning, peak_target);
        }
        Some(out)
    }

    pub fn odf_source_row_width(&self) -> Option<usize> {
        self.odf_source.as_ref().map(|source| source.ncols)
    }

    pub fn odf_rows_per_chunk(&self, max_storage_bytes: usize) -> Option<usize> {
        let row_width = self.odf_source_row_width()?;
        let row_bytes = row_width.saturating_mul(std::mem::size_of::<f32>());
        if row_bytes == 0 {
            return None;
        }
        Some((max_storage_bytes / row_bytes).max(1))
    }

    pub fn prewarm_odf_slice_metadata(&self, axis: usize, slice_idx: u32, rows_per_chunk: usize) {
        let _ = self.odf_slice_metadata(axis, slice_idx, rows_per_chunk);
    }

    pub fn odf_slice_metadata(
        &self,
        axis: usize,
        slice_idx: u32,
        rows_per_chunk: usize,
    ) -> Option<Arc<OdfSliceMetadata>> {
        let key = (axis, slice_idx, rows_per_chunk.max(1));
        {
            let mut cache = self.odf_slice_cache.lock().ok()?;
            if let Some(metadata) = cache.entries.get(&key).cloned() {
                promote_odf_slice_cache_key(&mut cache, key);
                return Some(metadata);
            }
        }

        let metadata = Arc::new(self.build_odf_slice_metadata(axis, slice_idx, key.2)?);
        let mut cache = self.odf_slice_cache.lock().ok()?;
        cache.entries.insert(key, metadata.clone());
        cache.order.push_back(key);
        while cache.order.len() > 12 {
            if let Some(oldest) = cache.order.pop_front() {
                cache.entries.remove(&oldest);
            }
        }
        Some(metadata)
    }

    pub fn sh_view_f32(&self) -> Option<TypedView2D<'_, f32>> {
        self.sh_source
            .as_ref()
            .and_then(|source| self.dataset.sh::<f32>(&source.name).ok())
    }

    pub fn sh_coefficients_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        self.sh_source.as_ref()?;
        let view = self.sh_view_f32()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let mut out = Vec::with_capacity(slice_indices.len() * view.ncols());
        for &compact_idx in slice_indices {
            out.extend_from_slice(view.row(compact_idx));
        }
        Some(out)
    }

    pub fn glyph_amplitudes_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        match self.glyph_source_kind()? {
            OdxGlyphSourceKind::Odf => self.odf_amplitudes_for_slice(axis, slice_idx),
            OdxGlyphSourceKind::Sh => {
                let mesh = self.sh_render_mesh(3)?;
                let view = self.sh_view_f32()?;
                let slice_indices = self.slice_compact_indices(axis, slice_idx);
                let mut out = Vec::with_capacity(slice_indices.len() * mesh.row_width());
                let mut sampled = vec![0.0f32; mesh.row_width()];
                for &compact_idx in slice_indices {
                    mesh.sample_plan()
                        .apply_row_into(view.row(compact_idx), &mut sampled);
                    out.extend_from_slice(&sampled);
                }
                Some(out)
            }
        }
    }

    pub fn sh_order(&self) -> Option<usize> {
        self.sh_source.as_ref().map(|source| source.sh_order)
    }

    pub fn sh_source_dir_count(&self, detail: u32) -> Option<usize> {
        self.sh_render_mesh(detail)
            .map(|mesh| mesh.source_dir_count())
    }

    pub fn sh_render_row_width(&self, detail: u32) -> Option<usize> {
        self.sh_render_mesh(detail).map(|mesh| mesh.row_width())
    }

    pub fn sh_transform_flat(&self, detail: u32) -> Option<Arc<ShRenderMesh>> {
        self.sh_render_mesh(detail)
    }

    pub fn glyph_source_is_hemisphere(&self) -> bool {
        self.odf_source
            .as_ref()
            .map(|source| source.sample_domain.is_hemisphere())
            .unwrap_or(false)
    }

    /// Volume dimensions `[nx, ny, nz]`.
    pub fn dimensions(&self) -> [u64; 3] {
        self.dataset.header().dimensions
    }

    /// RAS+ centers for all masked voxels.
    pub fn centers_ras(&self) -> &[[f32; 3]] {
        &self.centers_ras
    }

    /// Per-voxel IJK indices for all masked voxels.
    pub fn ijk_lookup(&self) -> &[[u32; 3]] {
        &self.ijk_lookup
    }

    /// Voxel→RAS affine (row-major source converted to `glam::Mat4`).
    pub fn voxel_to_ras(&self) -> Mat4 {
        let a = &self.dataset.header().voxel_to_rasmm;
        Mat4::from_cols(
            Vec4::new(a[0][0] as f32, a[1][0] as f32, a[2][0] as f32, 0.0),
            Vec4::new(a[0][1] as f32, a[1][1] as f32, a[2][1] as f32, 0.0),
            Vec4::new(a[0][2] as f32, a[1][2] as f32, a[2][2] as f32, 0.0),
            Vec4::new(a[0][3] as f32, a[1][3] as f32, a[2][3] as f32, 1.0),
        )
    }

    /// Return `(normal, center)` for the slice plane at the given voxel index.
    ///
    /// `axis_index`: 0 = axial (k-slices), 1 = coronal (j-slices), 2 = sagittal (i-slices).
    pub fn slice_plane(&self, axis_index: usize, slice_index: usize) -> (glam::Vec3, glam::Vec3) {
        let voxel_to_ras = self.voxel_to_ras();
        let col = match axis_index {
            0 => voxel_to_ras.col(2), // k-column
            1 => voxel_to_ras.col(1), // j-column
            _ => voxel_to_ras.col(0), // i-column
        };
        let normal = glam::Vec3::new(col.x, col.y, col.z).normalize();

        let dims = self.dimensions();
        let ci = dims[0] as f32 / 2.0;
        let cj = dims[1] as f32 / 2.0;
        let ck = dims[2] as f32 / 2.0;
        let si = slice_index as f32;
        let center_voxel = match axis_index {
            0 => glam::Vec3::new(ci, cj, si),
            1 => glam::Vec3::new(ci, si, ck),
            _ => glam::Vec3::new(si, cj, ck),
        };
        (normal, voxel_to_ras.transform_point3(center_voxel))
    }

    /// Extract glyph instances and amplitudes for a single slice.
    ///
    /// `axis`: 0=i (sagittal), 1=j (coronal), 2=k (axial).
    /// `slice_idx`: the voxel-grid index along `axis`.
    /// `skip`: render every Nth voxel (1 = no skip).
    pub fn glyphs_for_slice(&self, axis: usize, slice_idx: u32, skip: u32) -> SliceGlyphData {
        self.glyphs_for_slice_with_detail(axis, slice_idx, skip, 3)
    }

    pub fn glyphs_for_slice_with_detail(
        &self,
        axis: usize,
        slice_idx: u32,
        skip: u32,
        sh_detail: u32,
    ) -> SliceGlyphData {
        let Some(source_kind) = self.glyph_source_kind() else {
            return SliceGlyphData::default();
        };
        let skip = skip.max(1);
        let scale = self.default_glyph_scale();
        let Some(slice_indices) = self
            .slice_compact_indices
            .get(axis)
            .and_then(|slices| slices.get(slice_idx as usize))
        else {
            return SliceGlyphData::default();
        };
        let visible_voxels = slice_indices.len();
        let nv = match source_kind {
            OdxGlyphSourceKind::Odf => self.odf_render_row_width().unwrap_or(0),
            OdxGlyphSourceKind::Sh => self.sh_render_row_width(sh_detail).unwrap_or(0),
        };
        if nv == 0 {
            return SliceGlyphData::default();
        }
        let mut instances = Vec::with_capacity(visible_voxels);
        let mut amplitudes = Vec::with_capacity(visible_voxels * nv);
        let mut count = 0u32;

        match source_kind {
            OdxGlyphSourceKind::Odf => {
                let source = self.odf_source.as_ref().expect("ODF source should exist");
                debug_assert_eq!(
                    source.ncols
                        * if source.sample_domain.is_hemisphere() {
                            2
                        } else {
                            1
                        },
                    nv
                );
                let odf_view = self
                    .dataset
                    .odf::<f32>(&source.name)
                    .expect("ODF source missing during slice materialization");
                for &compact_idx in slice_indices {
                    if skip > 1 && (count % skip) != 0 {
                        count += 1;
                        continue;
                    }
                    count += 1;
                    let amp_offset = amplitudes.len() as u32;
                    let row = odf_view.row(compact_idx);
                    if source.sample_domain.is_hemisphere() {
                        append_mirrored_hemisphere_row(row, &mut amplitudes);
                    } else {
                        amplitudes.extend_from_slice(row);
                    }
                    instances.push(GlyphInstance {
                        center: self.centers_ras[compact_idx],
                        scale,
                        amplitude_offset: amp_offset,
                        min_contacts: 0,
                        contact_count: 1,
                        _pad: 0,
                    });
                }
            }
            OdxGlyphSourceKind::Sh => {
                let source = self.sh_source.as_ref().expect("SH source should exist");
                let mesh = self
                    .sh_render_mesh(sh_detail)
                    .expect("SH render mesh should exist for glyphs");
                debug_assert_eq!(mrtrix_sh::ncoeffs_for_lmax(source.sh_order), source.ncoeffs);
                let sh_view = self
                    .dataset
                    .sh::<f32>(&source.name)
                    .expect("SH source missing during slice materialization");
                let mut sampled = vec![0.0f32; mesh.row_width()];
                for &compact_idx in slice_indices {
                    if skip > 1 && (count % skip) != 0 {
                        count += 1;
                        continue;
                    }
                    count += 1;
                    let amp_offset = amplitudes.len() as u32;
                    let row = sh_view.row(compact_idx);
                    mesh.sample_plan().apply_row_into(row, &mut sampled);
                    amplitudes.extend_from_slice(&sampled);
                    instances.push(GlyphInstance {
                        center: self.centers_ras[compact_idx],
                        scale,
                        amplitude_offset: amp_offset,
                        min_contacts: 0,
                        contact_count: 1,
                        _pad: 0,
                    });
                }
            }
        }

        SliceGlyphData {
            amp_norm: slice_amp_norm(&amplitudes),
            instances,
            amplitudes,
        }
    }

    fn build_odf_slice_metadata(
        &self,
        axis: usize,
        slice_idx: u32,
        rows_per_chunk: usize,
    ) -> Option<OdfSliceMetadata> {
        let source = self.odf_source.as_ref()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let odf_view = self.odf_view_f32()?;
        let scale = self.default_glyph_scale();
        let mut instances = Vec::with_capacity(slice_indices.len());
        let mut chunk_worklists: Vec<OdfChunkWorklist> = Vec::new();
        let mut max_amp = 0.0f32;

        for (output_row, &compact_idx) in slice_indices.iter().enumerate() {
            let row = odf_view.row(compact_idx);
            for &value in row {
                if value.is_finite() {
                    max_amp = max_amp.max(value);
                }
            }

            let chunk_index = compact_idx / rows_per_chunk;
            let work_item = OdfChunkWorkItem {
                local_row: (compact_idx % rows_per_chunk) as u32,
                output_row: output_row as u32,
            };
            match chunk_worklists.last_mut() {
                Some(chunk) if chunk.chunk_index == chunk_index => chunk.work_items.push(work_item),
                _ => chunk_worklists.push(OdfChunkWorklist {
                    chunk_index,
                    work_items: vec![work_item],
                }),
            }

            let amp_offset = output_row.saturating_mul(source.render_vertices.len()) as u32;
            instances.push(GlyphInstance {
                center: self.centers_ras[compact_idx],
                scale,
                amplitude_offset: amp_offset,
                min_contacts: 0,
                contact_count: 1,
                _pad: 0,
            });
        }

        Some(OdfSliceMetadata {
            instances,
            chunk_worklists,
            amp_norm: if max_amp > 0.0 { max_amp } else { 1.0 },
        })
    }

    /// All fixel instances for the entire volume (no slice filter).
    ///
    /// Use this to upload fixels once at load time. The 2D slice views use
    /// shader-side slab clipping to show only the fixels on the current slice,
    /// so the CPU never needs to re-filter on slice change.
    pub fn all_fixels(&self) -> Vec<FixelInstance> {
        self.all_fixels_with_scalars(None)
    }

    pub fn all_fixels_with_scalars(&self, scalars: Option<&[f32]>) -> Vec<FixelInstance> {
        let offsets = self.dataset.offsets();
        if offsets.len() <= 1 {
            return Vec::new();
        }
        let dirs = self.dataset.directions();
        let scale = self.default_glyph_scale();
        let mut fixels = Vec::new();
        for (compact_idx, &center) in self.centers_ras.iter().enumerate() {
            if compact_idx + 1 >= offsets.len() {
                break;
            }
            let start = offsets[compact_idx] as usize;
            let end = offsets[compact_idx + 1] as usize;
            for (j, &dir) in dirs[start..end].iter().enumerate() {
                let global = start + j;
                let s = scalars
                    .and_then(|arr| arr.get(global).copied())
                    .unwrap_or(0.0);
                fixels.push(FixelInstance {
                    center,
                    direction: dir,
                    length: scale,
                    scalar: s,
                });
            }
        }
        fixels
    }

    /// The glyph scale (≈ half the minimum voxel dimension in mm).
    ///
    /// Exposed so callers can use this as a slab half-width when computing
    /// which fixels to show in a given 2D slice view.
    pub fn glyph_scale(&self) -> f32 {
        self.default_glyph_scale()
    }

    /// Extract fixel (peak) instances for a single slice.
    pub fn fixels_for_slice(&self, axis: usize, slice_idx: u32) -> Vec<FixelInstance> {
        self.fixels_for_slice_with_scalars(axis, slice_idx, None)
    }

    pub fn fixels_for_slice_with_scalars(
        &self,
        axis: usize,
        slice_idx: u32,
        scalars: Option<&[f32]>,
    ) -> Vec<FixelInstance> {
        let offsets = self.dataset.offsets();
        if offsets.len() <= 1 {
            return Vec::new();
        }
        let Some(slice_indices) = self
            .slice_compact_indices
            .get(axis)
            .and_then(|slices| slices.get(slice_idx as usize))
        else {
            return Vec::new();
        };
        let dirs = self.dataset.directions();
        let scale = self.default_glyph_scale();

        let mut fixels = Vec::new();
        for &compact_idx in slice_indices {
            if compact_idx + 1 >= offsets.len() {
                break;
            }
            let start = offsets[compact_idx] as usize;
            let end = offsets[compact_idx + 1] as usize;
            for (j, &dir) in dirs[start..end].iter().enumerate() {
                let global = start + j;
                let s = scalars
                    .and_then(|arr| arr.get(global).copied())
                    .unwrap_or(0.0);
                fixels.push(FixelInstance {
                    center: self.centers_ras[compact_idx],
                    direction: dir,
                    length: scale,
                    scalar: s,
                });
            }
        }
        fixels
    }

    /// List available DPV (data-per-voxel) scalar array names.
    pub fn dpv_names(&self) -> Vec<&str> {
        self.dataset.dpv_names()
    }

    /// Convert a scalar DPV array into a dense `NiftiVolume` suitable for slice rendering.
    ///
    /// The compact (masked) values are scattered into a full 3D grid using the
    /// precomputed `ijk_lookup`. Voxels outside the mask are set to zero.
    pub fn dpv_to_volume(&self, name: &str) -> anyhow::Result<NiftiVolume> {
        let values = self.dataset.scalar_dpv_f32(name)?;
        let dims = self.dataset.header().dimensions;
        let ni = dims[0] as usize;
        let nj = dims[1] as usize;
        let nk = dims[2] as usize;

        // Find intensity range from non-zero masked values.
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in &values {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
        let range = max_val - min_val;

        // Build dense 3D grid in i-fastest (Fortran) order to match NiftiVolume convention.
        let mut data = vec![0.0f32; ni * nj * nk];
        for (compact_idx, ijk) in self.ijk_lookup.iter().enumerate() {
            let i = ijk[0] as usize;
            let j = ijk[1] as usize;
            let k = ijk[2] as usize;
            // Fortran order: i varies fastest → index = i + j*ni + k*ni*nj
            let flat = i + j * ni + k * ni * nj;
            let v = values[compact_idx];
            data[flat] = if range > 0.0 {
                ((v - min_val) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        // Convert the ODX affine ([[f64; 4]; 4] row-major) to glam Mat4 (column-major).
        let a = &self.dataset.header().voxel_to_rasmm;
        let voxel_to_ras = Mat4::from_cols(
            Vec4::new(a[0][0] as f32, a[1][0] as f32, a[2][0] as f32, 0.0),
            Vec4::new(a[0][1] as f32, a[1][1] as f32, a[2][1] as f32, 0.0),
            Vec4::new(a[0][2] as f32, a[1][2] as f32, a[2][2] as f32, 0.0),
            Vec4::new(a[0][3] as f32, a[1][3] as f32, a[2][3] as f32, 1.0),
        );

        Ok(NiftiVolume {
            data,
            dims: [ni, nj, nk],
            voxel_to_ras,
        })
    }

    /// Compute a reasonable default glyph scale from the voxel spacing.
    fn default_glyph_scale(&self) -> f32 {
        let affine = &self.dataset.header().voxel_to_rasmm;
        let dx = (affine[0][0] * affine[0][0]
            + affine[1][0] * affine[1][0]
            + affine[2][0] * affine[2][0])
            .sqrt() as f32;
        let dy = (affine[0][1] * affine[0][1]
            + affine[1][1] * affine[1][1]
            + affine[2][1] * affine[2][1])
            .sqrt() as f32;
        let dz = (affine[0][2] * affine[0][2]
            + affine[1][2] * affine[1][2]
            + affine[2][2] * affine[2][2])
            .sqrt() as f32;
        dx.min(dy).min(dz)
    }

    pub fn default_normalized_peak_length_mm(&self) -> f32 {
        self.default_glyph_scale() * 0.45
    }
}

impl OdxScene {
    /// All fixel directions flattened (parallel to the internal offsets CSR).
    pub fn directions(&self) -> &[[f32; 3]] {
        self.dataset.directions()
    }

    /// Per-voxel CSR offsets into `directions()`.
    pub fn offsets(&self) -> &[u32] {
        self.dataset.offsets()
    }

    pub fn dataset(&self) -> &OdxDataset {
        &self.dataset
    }

    /// Available DPF (data-per-fixel) array names.
    pub fn dpf_names(&self) -> Vec<&str> {
        self.dataset.dpf_names()
    }

    /// Load a scalar DPF array by name, expanded to one f32 per fixel.
    pub fn scalar_dpf_f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.dataset.scalar_dpf_f32(name)?)
    }
}

/// A field of fixel directions across an ODX volume. Produced by `OdxSource`
/// and consumed by Fixel display and `ColorByFixelScalars` nodes.
#[derive(Clone)]
pub struct FixelField {
    pub source_id: FileId,
    pub scene: Arc<OdxScene>,
    /// Scalars applied to each fixel. Defaults to directional RGB.
    pub scalars: FixelScalars,
    /// Shader colormap code (0=directional/none, 2=plasma, 3=viridis,
    /// 4=inferno, 5=blue-white-red). When 0 the shader falls back to
    /// `abs(direction)` and ignores `scalar_range`.
    pub colormap_code: u32,
    pub scalar_range: (f32, f32),
}

/// Per-fixel scalar values. Either RGB (directional colors) or a scalar field
/// to be colormapped downstream.
#[derive(Clone)]
pub enum FixelScalarValues {
    Rgb(Arc<Vec<[f32; 3]>>),
    Scalar(Arc<Vec<f32>>),
}

/// Data-per-fixel scalars. Parallels `SurfaceScalars` for fixels.
#[derive(Clone)]
pub struct FixelScalars {
    pub source_id: FileId,
    pub name: String,
    pub fixel_count: usize,
    pub values: FixelScalarValues,
    pub range: (f32, f32),
}

impl FixelScalars {
    /// Build directional-RGB "scalars" from a list of unit direction vectors.
    pub fn from_directions(source_id: FileId, dirs: &[[f32; 3]]) -> Self {
        let rgb: Vec<[f32; 3]> = dirs
            .iter()
            .map(|d| [d[0].abs(), d[1].abs(), d[2].abs()])
            .collect();
        let count = rgb.len();
        Self {
            source_id,
            name: "direction".to_string(),
            fixel_count: count,
            values: FixelScalarValues::Rgb(Arc::new(rgb)),
            range: (0.0, 1.0),
        }
    }

    /// Build scalar fixel values from a DPF array.
    pub fn from_scalar(source_id: FileId, name: String, values: Vec<f32>) -> Self {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &values {
            if v.is_finite() {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
        }
        if !hi.is_finite() {
            hi = 1.0;
        }
        Self {
            source_id,
            fixel_count: values.len(),
            name,
            values: FixelScalarValues::Scalar(Arc::new(values)),
            range: (lo, hi),
        }
    }
}

/// ODF/SH field handle. Carries the sphere geometry and lazy slice
/// materialization state for glyph rendering.
#[derive(Clone)]
pub struct OdfField {
    pub source_id: FileId,
    pub scene: Arc<OdxScene>,
}

/// Sample a `VolumeScalars` input at each glyph center.
///
/// Missing inputs and out-of-bounds samples are encoded as `NaN` so the shader
/// can treat them as "gate disabled" without needing a separate boolean path.
pub fn sample_volume_scalars_for_glyphs(
    instances: &[GlyphInstance],
    scalars: Option<&VolumeScalars>,
) -> Vec<f32> {
    instances
        .iter()
        .map(|instance| {
            scalars
                .and_then(|volume| volume.sample_ras(glam::Vec3::from(instance.center)))
                .unwrap_or(f32::NAN)
        })
        .collect()
}

/// Lazy catalog of available DPV and DPF arrays on an ODX dataset.
/// `OdxVolumeSelect` / `OdxFixelScalarSelect` nodes consume this and choose
/// one by name.
#[derive(Clone)]
pub struct OdxCatalog {
    pub source_id: FileId,
    pub scene: Arc<OdxScene>,
    pub dpv_names: Arc<Vec<String>>,
    pub dpf_names: Arc<Vec<String>>,
}

impl OdxCatalog {
    pub fn from_scene(source_id: FileId, scene: Arc<OdxScene>) -> Self {
        let dpv: Vec<String> = scene.dpv_names().iter().map(|s| s.to_string()).collect();
        let dpf: Vec<String> = scene.dpf_names().iter().map(|s| s.to_string()).collect();
        Self {
            source_id,
            scene,
            dpv_names: Arc::new(dpv),
            dpf_names: Arc::new(dpf),
        }
    }

    /// Materialize a DPV array as a dense `NiftiVolume`. Called by
    /// `OdxVolumeSelect` lazily when the user picks a name.
    pub fn materialize_dpv(&self, name: &str) -> anyhow::Result<NiftiVolume> {
        self.scene.dpv_to_volume(name)
    }
}

fn resolve_odf_source(dataset: &OdxDataset, warnings: &mut Vec<String>) -> Option<OdfGlyphSource> {
    let odf_view = dataset.odf::<f32>("amplitudes").ok()?;
    let ncols = odf_view.ncols();
    let header_domain = match dataset.header().odf_sample_domain.as_deref() {
        Some("hemisphere") => Some(OdfSampleDomain::Hemisphere),
        Some("full") => Some(OdfSampleDomain::FullSphere),
        Some(other) => {
            warnings.push(format!(
                "ODF glyphs disabled: unsupported odf_sample_domain '{other}' for {} columns.",
                ncols
            ));
            return None;
        }
        None => None,
    };

    if let (Some(vertices), Some(faces)) = (dataset.sphere_vertices(), dataset.sphere_faces()) {
        let expected_cols = match header_domain.unwrap_or(OdfSampleDomain::FullSphere) {
            OdfSampleDomain::Hemisphere => vertices.len() / 2,
            OdfSampleDomain::FullSphere => vertices.len(),
        };
        if ncols == expected_cols {
            let render_indices: Vec<u32> =
                faces.iter().flat_map(|face| face.iter().copied()).collect();
            return Some(OdfGlyphSource {
                name: "amplitudes".into(),
                sample_domain: header_domain.unwrap_or(OdfSampleDomain::FullSphere),
                ncols,
                render_vertices: vertices.to_vec(),
                render_indices,
            });
        }
    }

    let fallback_domain = match header_domain {
        Some(OdfSampleDomain::Hemisphere)
            if ncols == dsistudio_odf8::hemisphere_vertices_ras().len() =>
        {
            Some(OdfSampleDomain::Hemisphere)
        }
        Some(OdfSampleDomain::FullSphere) if ncols == dsistudio_odf8::full_vertices_ras().len() => {
            Some(OdfSampleDomain::FullSphere)
        }
        Some(_) => None,
        None if ncols == dsistudio_odf8::hemisphere_vertices_ras().len() => {
            Some(OdfSampleDomain::Hemisphere)
        }
        None if ncols == dsistudio_odf8::full_vertices_ras().len() => {
            Some(OdfSampleDomain::FullSphere)
        }
        None => None,
    };

    if let Some(sample_domain) = fallback_domain {
        let render_indices: Vec<u32> = dsistudio_odf8::faces()
            .iter()
            .flat_map(|face| face.iter().copied())
            .collect();
        return Some(OdfGlyphSource {
            name: "amplitudes".into(),
            sample_domain,
            ncols,
            render_vertices: dsistudio_odf8::full_vertices_ras().to_vec(),
            render_indices,
        });
    }

    warnings.push(format!(
        "ODF glyphs disabled: ODF row width {ncols} could not be matched to an explicit sphere-with-faces or built-in odf8."
    ));
    None
}

fn resolve_sh_source(dataset: &OdxDataset, warnings: &mut Vec<String>) -> Option<ShGlyphSource> {
    let sh_view = dataset.sh::<f32>("coefficients").ok()?;
    let ncoeffs = sh_view.ncols();
    let sh_order = match dataset.header().sh_order.map(|order| order as usize) {
        Some(order) => order,
        None => match mrtrix_sh::lmax_for_ncoeffs(ncoeffs) {
            Ok(order) => order,
            Err(err) => {
                warnings.push(format!(
                    "SH glyphs disabled: could not infer SH order from {ncoeffs} coefficients ({err})."
                ));
                return None;
            }
        },
    };
    Some(ShGlyphSource {
        name: "coefficients".into(),
        ncoeffs,
        sh_order,
    })
}

fn build_full_icosphere_mesh(detail: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut vertices = vec![
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];
    for vertex in &mut vertices {
        *vertex = Vec3::from_array(*vertex).normalize().to_array();
    }
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..detail {
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut subdivided = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint_index(&mut vertices, &mut midpoint_cache, a, b);
            let bc = midpoint_index(&mut vertices, &mut midpoint_cache, b, c);
            let ca = midpoint_index(&mut vertices, &mut midpoint_cache, c, a);
            subdivided.push([a, ab, ca]);
            subdivided.push([b, bc, ab]);
            subdivided.push([c, ca, bc]);
            subdivided.push([ab, bc, ca]);
        }
        faces = subdivided;
    }

    let indices = faces.into_iter().flat_map(|face| face).collect();
    (vertices, indices)
}

fn midpoint_index(
    vertices: &mut Vec<[f32; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let va = Vec3::from_array(vertices[a as usize]);
    let vb = Vec3::from_array(vertices[b as usize]);
    let idx = vertices.len() as u32;
    vertices.push((va + vb).normalize().to_array());
    cache.insert(key, idx);
    idx
}

fn append_mirrored_hemisphere_row(row: &[f32], out: &mut Vec<f32>) {
    out.extend_from_slice(row);
    out.extend_from_slice(row);
}

pub fn condition_odf_amplitudes_in_place(
    values: &mut [f32],
    conditioning: OdfAmplitudeConditioning,
    target_peak_length_mm: f32,
) {
    if values.is_empty() || (!conditioning.subtract_iso && !conditioning.norm_within_voxel) {
        return;
    }

    let min_amp = if conditioning.subtract_iso {
        values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .reduce(f32::min)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    if conditioning.subtract_iso {
        for value in values.iter_mut() {
            if value.is_finite() {
                *value -= min_amp;
            }
        }
    }

    if !conditioning.norm_within_voxel {
        return;
    }

    let peak = values.iter().copied().fold(0.0f32, |acc, value| {
        if value.is_finite() {
            acc.max(value)
        } else {
            acc
        }
    });
    if !peak.is_finite()
        || peak <= 0.0
        || !target_peak_length_mm.is_finite()
        || target_peak_length_mm <= 0.0
    {
        return;
    }

    let scale = target_peak_length_mm / peak;
    for value in values.iter().copied() {
        if !value.is_finite() {
            return;
        }
    }

    for value in values.iter_mut() {
        *value *= scale;
    }
}

fn promote_odf_slice_cache_key(cache: &mut OdfSliceMetadataCache, key: (usize, u32, usize)) {
    if let Some(position) = cache.order.iter().position(|entry| *entry == key) {
        cache.order.remove(position);
        cache.order.push_back(key);
    }
}

fn slice_amp_norm(values: &[f32]) -> f32 {
    let max_amp = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, f32::max);
    if max_amp > 0.0 { max_amp } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    use odx_rs::{DType, OdxBuilder};

    fn build_test_dataset_with_odf(hemisphere: bool) -> OdxDataset {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let hemi = dsistudio_odf8::hemisphere_vertices_ras();
        let dims = [1, 1, 2];
        let mask = vec![1u8, 1u8];
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dims,
            mask,
        );
        builder.set_sphere(full, faces);
        builder.push_voxel_peaks(&[]);
        builder.push_voxel_peaks(&[]);
        if hemisphere {
            builder.set_odf_sample_domain("hemisphere");
        }
        let ncols = if hemisphere {
            hemi.len()
        } else {
            dsistudio_odf8::full_vertices_ras().len()
        };
        let values: Vec<f32> = (0..(ncols * 2)).map(|idx| idx as f32 + 1.0).collect();
        builder.set_odf_data(
            "amplitudes",
            bytemuck::cast_slice(&values).to_vec(),
            ncols,
            DType::Float32,
        );
        builder.finalize().unwrap()
    }

    fn build_test_dataset_with_sh() -> OdxDataset {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let dims = [1, 1, 2];
        let mask = vec![1u8, 1u8];
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dims,
            mask,
        );
        builder.set_sphere(full, faces);
        builder.push_voxel_peaks(&[]);
        builder.push_voxel_peaks(&[]);
        builder.set_sh_info(2, "tournier07".into());
        let coeffs: Vec<f32> = vec![
            1.0, 0.1, 0.2, 0.3, 0.4, 0.5, //
            0.8, 0.2, 0.1, 0.0, -0.1, 0.3,
        ];
        builder.set_sh_data(
            "coefficients",
            bytemuck::cast_slice(&coeffs).to_vec(),
            6,
            DType::Float32,
        );
        builder.finalize().unwrap()
    }

    fn build_test_dataset_with_odf_width(
        ncols: usize,
        domain: Option<&str>,
        include_explicit_sphere: bool,
    ) -> OdxDataset {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let dims = [1, 1, 2];
        let mask = vec![1u8, 1u8];
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dims,
            mask,
        );
        if include_explicit_sphere {
            builder.set_sphere(full, faces);
        } else {
            builder.set_sphere_id("dsistudio_odf8");
        }
        builder.push_voxel_peaks(&[]);
        builder.push_voxel_peaks(&[]);
        if let Some(domain) = domain {
            builder.set_odf_sample_domain(domain);
        }
        let values: Vec<f32> = (0..(ncols * 2)).map(|idx| idx as f32 + 1.0).collect();
        builder.set_odf_data(
            "amplitudes",
            bytemuck::cast_slice(&values).to_vec(),
            ncols,
            DType::Float32,
        );
        builder.finalize().unwrap()
    }

    fn build_test_dataset_with_sh_and_odf(odf_cols: usize, odf_domain: Option<&str>) -> OdxDataset {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let dims = [1, 1, 2];
        let mask = vec![1u8, 1u8];
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dims,
            mask,
        );
        builder.set_sphere(full, faces);
        builder.push_voxel_peaks(&[]);
        builder.push_voxel_peaks(&[]);
        if let Some(domain) = odf_domain {
            builder.set_odf_sample_domain(domain);
        }
        let odf_values: Vec<f32> = (0..(odf_cols * 2)).map(|idx| idx as f32 + 1.0).collect();
        builder.set_odf_data(
            "amplitudes",
            bytemuck::cast_slice(&odf_values).to_vec(),
            odf_cols,
            DType::Float32,
        );
        builder.set_sh_info(2, "tournier07".into());
        let coeffs: Vec<f32> = vec![
            1.0, 0.1, 0.2, 0.3, 0.4, 0.5, //
            0.8, 0.2, 0.1, 0.0, -0.1, 0.3,
        ];
        builder.set_sh_data(
            "coefficients",
            bytemuck::cast_slice(&coeffs).to_vec(),
            6,
            DType::Float32,
        );
        builder.finalize().unwrap()
    }

    #[test]
    fn condition_odf_amplitudes_noop_when_disabled() {
        let mut values = vec![1.0, 3.0, 5.0];
        condition_odf_amplitudes_in_place(
            &mut values,
            OdfAmplitudeConditioning::new(false, false),
            1.0,
        );
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn condition_odf_amplitudes_subtracts_minimum() {
        let mut values = vec![2.0, 5.0, 7.0];
        condition_odf_amplitudes_in_place(
            &mut values,
            OdfAmplitudeConditioning::new(true, false),
            1.0,
        );
        assert_eq!(values, vec![0.0, 3.0, 5.0]);
    }

    #[test]
    fn condition_odf_amplitudes_scales_peak_only() {
        let mut values = vec![2.0, 3.0, 5.0];
        condition_odf_amplitudes_in_place(
            &mut values,
            OdfAmplitudeConditioning::new(false, true),
            4.5,
        );
        assert!((values[0] - 1.8).abs() < 1e-6);
        assert!((values[1] - 2.7).abs() < 1e-6);
        assert!((values[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn condition_odf_amplitudes_subtracts_then_scales_peak() {
        let mut values = vec![2.0, 5.0, 7.0];
        condition_odf_amplitudes_in_place(
            &mut values,
            OdfAmplitudeConditioning::new(true, true),
            4.5,
        );
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[1] - 2.7).abs() < 1e-6);
        assert!((values[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn condition_odf_amplitudes_skips_invalid_or_zero_peak_normalization() {
        let mut zero_sum = vec![4.0, 4.0, 4.0];
        condition_odf_amplitudes_in_place(
            &mut zero_sum,
            OdfAmplitudeConditioning::new(true, true),
            4.5,
        );
        assert_eq!(zero_sum, vec![0.0, 0.0, 0.0]);

        let mut invalid = vec![1.0, f32::NAN, 3.0];
        condition_odf_amplitudes_in_place(
            &mut invalid,
            OdfAmplitudeConditioning::new(false, true),
            4.5,
        );
        assert!(invalid[0] == 1.0 && invalid[2] == 3.0 && invalid[1].is_nan());
    }

    #[test]
    fn conditioned_hemisphere_slice_scales_peak_after_expansion() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(true)).unwrap();
        let values = scene
            .conditioned_odf_amplitudes_for_slice(2, 0, OdfAmplitudeConditioning::new(false, true))
            .unwrap();
        assert_eq!(values.len(), dsistudio_odf8::full_vertices_ras().len());
        let peak = values.iter().copied().fold(0.0f32, f32::max);
        assert!((peak - scene.default_normalized_peak_length_mm()).abs() < 1e-6);
        let hemi_len = dsistudio_odf8::hemisphere_vertices_ras().len();
        assert_eq!(&values[..hemi_len], &values[hemi_len..]);
    }

    #[test]
    fn lazy_odf_slice_matches_stored_row() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(false)).unwrap();
        let slice = scene.glyphs_for_slice(2, 0, 1);
        let full_bins = scene.odf_render_row_width().unwrap();
        assert_eq!(slice.instances.len(), 1);
        assert_eq!(slice.amplitudes.len(), full_bins);
        let expected: Vec<f32> = (1..=full_bins).map(|v| v as f32).collect();
        assert_eq!(slice.amplitudes, expected);
        assert_eq!(slice.amp_norm, full_bins as f32);
    }

    #[test]
    fn lazy_hemisphere_odf_slice_mirrors_row() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(true)).unwrap();
        let slice = scene.glyphs_for_slice(2, 0, 1);
        let hemi = dsistudio_odf8::hemisphere_vertices_ras().len();
        assert_eq!(slice.amplitudes.len(), hemi * 2);
        assert_eq!(&slice.amplitudes[..hemi], &slice.amplitudes[hemi..]);
    }

    #[test]
    fn odf_slice_metadata_matches_materialized_rows() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(false)).unwrap();
        let metadata = scene.odf_slice_metadata(2, 0, 1).unwrap();
        let expected = scene.odf_amplitudes_for_slice(2, 0).unwrap();
        let actual = reconstruct_odf_slice(&scene, metadata.as_ref(), 1);
        assert_eq!(actual, expected);
        assert_eq!(metadata.amp_norm, slice_amp_norm(&expected));
        assert_eq!(
            metadata.instances.len(),
            scene
                .glyph_instances_for_slice(2, 0, scene.odf_render_row_width().unwrap())
                .len()
        );
    }

    #[test]
    fn hemisphere_odf_slice_metadata_matches_materialized_rows() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(true)).unwrap();
        let metadata = scene.odf_slice_metadata(2, 0, 2).unwrap();
        let expected = scene.odf_amplitudes_for_slice(2, 0).unwrap();
        let actual = reconstruct_odf_slice(&scene, metadata.as_ref(), 2);
        assert_eq!(actual, expected);
        assert_eq!(metadata.amp_norm, slice_amp_norm(&expected));
    }

    #[test]
    fn lazy_sh_slice_matches_sampling_helper() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh()).unwrap();
        let slice = scene.glyphs_for_slice_with_detail(2, 0, 1, 3);
        let mesh = scene.sh_render_mesh(3).unwrap();
        let coeffs = scene
            .dataset()
            .sh::<f32>("coefficients")
            .unwrap()
            .row(0)
            .to_vec();
        let expected =
            mrtrix_sh::sample_rows_nonnegative(&coeffs, 1, mesh.vertices(), coeffs.len()).unwrap();
        assert_eq!(slice.instances.len(), 1);
        assert_eq!(slice.amplitudes, expected);
    }

    #[test]
    fn glyph_source_introspection_reports_odf() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(false)).unwrap();
        assert!(scene.has_glyph_field());
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Odf));
        assert_eq!(scene.compact_voxel_count(), 2);
    }

    #[test]
    fn glyph_source_introspection_reports_sh() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh()).unwrap();
        assert!(scene.has_glyph_field());
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Sh));
        assert_eq!(scene.compact_voxel_count(), 2);
    }

    #[test]
    fn glyph_source_introspection_reports_none_for_fixel_only() {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [1, 1, 1],
            vec![1u8],
        );
        builder.set_sphere(full, faces);
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
        let scene = OdxScene::from_dataset(builder.finalize().unwrap()).unwrap();
        assert!(!scene.has_glyph_field());
        assert_eq!(scene.glyph_source_kind(), None);
    }

    #[test]
    fn slice_plane_uses_oblique_odx_affine() {
        let full = dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = dsistudio_odf8::faces().to_vec();
        let affine = [
            [2.0, 0.0, 0.5, 10.0],
            [0.0, 3.0, 1.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut builder = OdxBuilder::new(affine, [4, 6, 8], vec![1u8; 4 * 6 * 8]);
        builder.set_sphere(full, faces);
        for _ in 0..(4 * 6 * 8) {
            builder.push_voxel_peaks(&[]);
        }
        let scene = OdxScene::from_dataset(builder.finalize().unwrap()).unwrap();

        let (normal, center) = scene.slice_plane(0, 5);
        let expected_normal = glam::Vec3::new(0.5, 1.0, 4.0).normalize();
        let expected_center = glam::Vec3::new(16.5, 34.0, 50.0);

        assert!((normal - expected_normal).length() < 1e-6);
        assert!((center - expected_center).length() < 1e-6);
    }

    #[test]
    fn sh_is_preferred_when_both_sh_and_valid_odf_are_present() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh_and_odf(642, None)).unwrap();
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Sh));
    }

    #[test]
    fn sh_is_preferred_when_odf_is_invalid() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh_and_odf(123, None)).unwrap();
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Sh));
        assert!(
            scene
                .glyph_warnings()
                .iter()
                .any(|warning| warning.contains("ODF glyphs disabled"))
        );
    }

    #[test]
    fn explicit_sphere_full_odf_uses_dataset_geometry() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(false)).unwrap();
        let (vertices, indices) = scene.odf_render_geometry().unwrap();
        assert_eq!(vertices.len(), dsistudio_odf8::full_vertices_ras().len());
        assert_eq!(indices.len(), dsistudio_odf8::faces().len() * 3);
    }

    #[test]
    fn explicit_sphere_hemisphere_odf_uses_dataset_geometry() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(true)).unwrap();
        let (vertices, _) = scene.odf_render_geometry().unwrap();
        assert_eq!(vertices.len(), dsistudio_odf8::full_vertices_ras().len());
        assert!(scene.glyph_source_is_hemisphere());
    }

    #[test]
    fn built_in_odf8_supports_hemisphere_without_explicit_sphere() {
        let scene =
            OdxScene::from_dataset(build_test_dataset_with_odf_width(321, None, false)).unwrap();
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Odf));
        assert!(scene.glyph_source_is_hemisphere());
        assert_eq!(scene.odf_render_row_width(), Some(642));
    }

    #[test]
    fn built_in_odf8_supports_full_sphere_without_explicit_sphere() {
        let scene =
            OdxScene::from_dataset(build_test_dataset_with_odf_width(642, None, false)).unwrap();
        assert_eq!(scene.glyph_source_kind(), Some(OdxGlyphSourceKind::Odf));
        assert!(!scene.glyph_source_is_hemisphere());
        assert_eq!(scene.odf_render_row_width(), Some(642));
    }

    #[test]
    fn conflicting_odf_domain_disables_odf_glyphs() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf_width(
            642,
            Some("hemisphere"),
            false,
        ))
        .unwrap();
        assert_eq!(scene.glyph_source_kind(), None);
        assert!(
            scene
                .glyph_warnings()
                .iter()
                .any(|warning| warning.contains("ODF glyphs disabled"))
        );
    }

    #[test]
    fn sh_detail_changes_render_mesh_resolution() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh()).unwrap();
        let detail_two = scene.sh_render_mesh(2).unwrap();
        let detail_three = scene.sh_render_mesh(3).unwrap();
        assert_ne!(detail_two.row_width(), detail_three.row_width());
        assert!(detail_three.row_width() > detail_two.row_width());
    }

    #[test]
    fn sh_detail_clamps_to_storage_limit() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_sh()).unwrap();
        let row_width_4 = scene.sh_render_row_width(4).unwrap();
        let row_width_5 = scene.sh_render_row_width(5).unwrap();
        let limit = row_width_4 * std::mem::size_of::<f32>();
        assert_eq!(scene.clamp_sh_detail_for_slice(2, 0, 5, limit), 4);
        assert_eq!(
            scene.clamp_sh_detail_for_slice(2, 0, 5, row_width_5 * std::mem::size_of::<f32>()),
            5
        );
    }

    fn reconstruct_odf_slice(
        scene: &OdxScene,
        metadata: &OdfSliceMetadata,
        rows_per_chunk: usize,
    ) -> Vec<f32> {
        let view = scene.odf_view_f32().unwrap();
        let source_bins = scene.odf_source_row_width().unwrap();
        let full_bins = scene.odf_render_row_width().unwrap();
        let mut out = vec![0.0f32; metadata.instances.len() * full_bins];
        for chunk in &metadata.chunk_worklists {
            for work_item in &chunk.work_items {
                let compact_idx = chunk.chunk_index * rows_per_chunk + work_item.local_row as usize;
                let row = view.row(compact_idx);
                let dst = &mut out[(work_item.output_row as usize * full_bins)
                    ..((work_item.output_row as usize + 1) * full_bins)];
                if scene.glyph_source_is_hemisphere() {
                    dst[..source_bins].copy_from_slice(row);
                    dst[source_bins..].copy_from_slice(row);
                } else {
                    dst.copy_from_slice(row);
                }
            }
        }
        out
    }
}

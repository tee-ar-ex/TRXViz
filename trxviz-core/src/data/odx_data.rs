use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec4};
use odx_rs::formats::dsistudio_odf8;
use odx_rs::typed_view::TypedView2D;
use odx_rs::{OdxDataset, mrtrix_sh};

use crate::data::cifti::VolumeScalars;
use crate::data::loaded_files::FileId;
use crate::data::nifti_data::NiftiVolume;
use crate::renderer::glyph_renderer::GlyphInstance;

/// Loaded ODX dataset with precomputed spatial lookups for slice-local rendering.
pub struct OdxScene {
    dataset: OdxDataset,
    /// `[i, j, k]` grid position for each compact (masked) voxel.
    ijk_lookup: Vec<[u32; 3]>,
    /// Compact voxel indices grouped by slice for each ODX axis (i, j, k).
    slice_compact_indices: [Vec<Vec<usize>>; 3],
    /// RAS+ center for each compact voxel.
    centers_ras: Vec<[f32; 3]>,
    glyph_source: Option<GlyphFieldSource>,
    /// Full-sphere vertex positions (unit vectors).
    pub sphere_vertices: Vec<[f32; 3]>,
    /// Triangle indices (flattened from `[[u32; 3]]`).
    pub sphere_indices: Vec<u32>,
    /// Number of sphere vertices (per voxel amplitude row width).
    pub nb_sphere_vertices: usize,
    odf_slice_cache: Mutex<OdfSliceMetadataCache>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OdxGlyphSourceKind {
    Odf,
    Sh,
}

#[derive(Debug)]
enum GlyphFieldSource {
    Odf {
        name: String,
        hemisphere: bool,
        ncols: usize,
    },
    Sh {
        name: String,
        hemisphere: bool,
        ncoeffs: usize,
        sh_order: usize,
        sample_plan: mrtrix_sh::RowSamplePlan,
    },
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

        // Resolve sphere mesh — prefer dataset's own, fall back to built-in odf8.
        let (full_verts, faces) = match (dataset.sphere_vertices(), dataset.sphere_faces()) {
            (Some(v), Some(f)) => (v.to_vec(), f.to_vec()),
            _ => (
                dsistudio_odf8::full_vertices_ras().to_vec(),
                dsistudio_odf8::faces().to_vec(),
            ),
        };
        let sphere_indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
        let nb_sphere_vertices = full_verts.len();

        let is_hemisphere = dataset.header().odf_sample_domain.as_deref() == Some("hemisphere");
        let glyph_source = if let Ok(odf_view) = dataset.odf::<f32>("amplitudes") {
            let expected_cols = if is_hemisphere {
                dsistudio_odf8::hemisphere_vertices_ras().len()
            } else {
                nb_sphere_vertices
            };
            if odf_view.ncols() != expected_cols {
                anyhow::bail!(
                    "ODF amplitudes have {} columns but expected {} for the active sphere",
                    odf_view.ncols(),
                    expected_cols
                );
            }
            Some(GlyphFieldSource::Odf {
                name: "amplitudes".into(),
                hemisphere: is_hemisphere,
                ncols: odf_view.ncols(),
            })
        } else if let Ok(sh_view) = dataset.sh::<f32>("coefficients") {
            let dirs: &[[f32; 3]] = if is_hemisphere {
                dsistudio_odf8::hemisphere_vertices_ras()
            } else {
                &full_verts
            };
            let ncoeffs = sh_view.ncols();
            let sh_order = dataset
                .header()
                .sh_order
                .map(|order| order as usize)
                .unwrap_or(mrtrix_sh::lmax_for_ncoeffs(ncoeffs)?);
            Some(GlyphFieldSource::Sh {
                name: "coefficients".into(),
                hemisphere: is_hemisphere,
                ncoeffs,
                sh_order,
                sample_plan: mrtrix_sh::RowSamplePlan::for_sh_rows_nonnegative(dirs, ncoeffs)?,
            })
        } else {
            None
        };

        Ok(Self {
            dataset,
            ijk_lookup,
            slice_compact_indices,
            centers_ras,
            glyph_source,
            sphere_vertices: full_verts,
            sphere_indices,
            nb_sphere_vertices,
            odf_slice_cache: Mutex::new(OdfSliceMetadataCache::default()),
        })
    }

    /// Number of full-volume masked voxels.
    pub fn nb_voxels(&self) -> usize {
        self.ijk_lookup.len()
    }

    pub fn compact_voxel_count(&self) -> usize {
        self.ijk_lookup.len()
    }

    pub fn has_glyph_field(&self) -> bool {
        self.glyph_source.is_some()
    }

    pub fn glyph_source_kind(&self) -> Option<OdxGlyphSourceKind> {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Odf { .. }) => Some(OdxGlyphSourceKind::Odf),
            Some(GlyphFieldSource::Sh { .. }) => Some(OdxGlyphSourceKind::Sh),
            None => None,
        }
    }

    pub fn glyph_row_width(&self) -> usize {
        self.nb_sphere_vertices
    }

    pub fn glyph_instances_full_volume(&self) -> Vec<GlyphInstance> {
        let scale = self.default_glyph_scale();
        self.centers_ras
            .iter()
            .enumerate()
            .map(|(compact_idx, &center)| GlyphInstance {
                center,
                scale,
                amplitude_offset: (compact_idx * self.nb_sphere_vertices) as u32,
                min_contacts: 0,
                contact_count: 1,
                _pad: 0,
            })
            .collect()
    }

    pub fn glyph_instances_for_slice(&self, axis: usize, slice_idx: u32) -> Vec<GlyphInstance> {
        let scale = self.default_glyph_scale();
        self.slice_compact_indices(axis, slice_idx)
            .iter()
            .enumerate()
            .map(|(local_idx, &compact_idx)| GlyphInstance {
                center: self.centers_ras[compact_idx],
                scale,
                amplitude_offset: (local_idx * self.nb_sphere_vertices) as u32,
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
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Odf { name, .. }) => self.dataset.odf::<f32>(name).ok(),
            _ => None,
        }
    }

    pub fn odf_amplitudes_full_sphere(&self) -> Option<Vec<f32>> {
        let GlyphFieldSource::Odf { hemisphere, .. } = self.glyph_source.as_ref()? else {
            return None;
        };
        let view = self.odf_view_f32()?;
        if !hemisphere {
            return Some(view.as_flat_slice().to_vec());
        }
        let mut out = Vec::with_capacity(view.nrows() * self.nb_sphere_vertices);
        for row in view.rows() {
            append_mirrored_hemisphere_row(row, &mut out);
        }
        Some(out)
    }

    pub fn odf_amplitudes_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        let GlyphFieldSource::Odf { hemisphere, .. } = self.glyph_source.as_ref()? else {
            return None;
        };
        let view = self.odf_view_f32()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let mut out = Vec::with_capacity(slice_indices.len() * self.nb_sphere_vertices);
        for &compact_idx in slice_indices {
            let row = view.row(compact_idx);
            if *hemisphere {
                append_mirrored_hemisphere_row(row, &mut out);
            } else {
                out.extend_from_slice(row);
            }
        }
        Some(out)
    }

    pub fn odf_source_row_width(&self) -> Option<usize> {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Odf { ncols, .. }) => Some(*ncols),
            _ => None,
        }
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
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Sh { name, .. }) => self.dataset.sh::<f32>(name).ok(),
            _ => None,
        }
    }

    pub fn sh_coefficients_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        let GlyphFieldSource::Sh { .. } = self.glyph_source.as_ref()? else {
            return None;
        };
        let view = self.sh_view_f32()?;
        let slice_indices = self.slice_compact_indices(axis, slice_idx);
        let mut out = Vec::with_capacity(slice_indices.len() * view.ncols());
        for &compact_idx in slice_indices {
            out.extend_from_slice(view.row(compact_idx));
        }
        Some(out)
    }

    pub fn glyph_amplitudes_for_slice(&self, axis: usize, slice_idx: u32) -> Option<Vec<f32>> {
        match self.glyph_source.as_ref()? {
            GlyphFieldSource::Odf { .. } => self.odf_amplitudes_for_slice(axis, slice_idx),
            GlyphFieldSource::Sh {
                hemisphere,
                sample_plan,
                ..
            } => {
                let view = self.sh_view_f32()?;
                let slice_indices = self.slice_compact_indices(axis, slice_idx);
                let mut out = Vec::with_capacity(slice_indices.len() * self.nb_sphere_vertices);
                let mut sampled = vec![0.0f32; sample_plan.ndir()];
                for &compact_idx in slice_indices {
                    sample_plan.apply_row_into(view.row(compact_idx), &mut sampled);
                    if *hemisphere {
                        append_mirrored_hemisphere_row(&sampled, &mut out);
                    } else {
                        out.extend_from_slice(&sampled);
                    }
                }
                Some(out)
            }
        }
    }

    pub fn sh_order(&self) -> Option<usize> {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Sh { sh_order, .. }) => Some(*sh_order),
            _ => None,
        }
    }

    pub fn sh_source_dir_count(&self) -> Option<usize> {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Sh { sample_plan, .. }) => Some(sample_plan.source_dir_count()),
            _ => None,
        }
    }

    pub fn sh_transform_flat(&self) -> Option<&[f32]> {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Sh { sample_plan, .. }) => Some(sample_plan.transform_flat()),
            _ => None,
        }
    }

    pub fn glyph_source_is_hemisphere(&self) -> bool {
        match self.glyph_source.as_ref() {
            Some(GlyphFieldSource::Odf { hemisphere, .. })
            | Some(GlyphFieldSource::Sh { hemisphere, .. }) => *hemisphere,
            None => false,
        }
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

    /// Extract glyph instances and amplitudes for a single slice.
    ///
    /// `axis`: 0=i (sagittal), 1=j (coronal), 2=k (axial).
    /// `slice_idx`: the voxel-grid index along `axis`.
    /// `skip`: render every Nth voxel (1 = no skip).
    pub fn glyphs_for_slice(&self, axis: usize, slice_idx: u32, skip: u32) -> SliceGlyphData {
        let Some(source) = &self.glyph_source else {
            return SliceGlyphData::default();
        };
        let skip = skip.max(1);
        let nv = self.nb_sphere_vertices;
        let scale = self.default_glyph_scale();
        let Some(slice_indices) = self
            .slice_compact_indices
            .get(axis)
            .and_then(|slices| slices.get(slice_idx as usize))
        else {
            return SliceGlyphData::default();
        };
        let visible_voxels = slice_indices.len();
        let mut instances = Vec::with_capacity(visible_voxels);
        let mut amplitudes = Vec::with_capacity(visible_voxels * nv);
        let mut count = 0u32;

        match source {
            GlyphFieldSource::Odf {
                name,
                hemisphere,
                ncols,
            } => {
                debug_assert_eq!(*ncols * if *hemisphere { 2 } else { 1 }, nv);
                let odf_view = self
                    .dataset
                    .odf::<f32>(name)
                    .expect("ODF source missing during slice materialization");
                for &compact_idx in slice_indices {
                    if skip > 1 && (count % skip) != 0 {
                        count += 1;
                        continue;
                    }
                    count += 1;
                    let amp_offset = amplitudes.len() as u32;
                    let row = odf_view.row(compact_idx);
                    if *hemisphere {
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
            GlyphFieldSource::Sh {
                name,
                hemisphere,
                ncoeffs,
                sh_order,
                sample_plan,
            } => {
                debug_assert_eq!(mrtrix_sh::ncoeffs_for_lmax(*sh_order), *ncoeffs);
                let sh_view = self
                    .dataset
                    .sh::<f32>(name)
                    .expect("SH source missing during slice materialization");
                let mut sampled = vec![0.0f32; sample_plan.ndir()];
                for &compact_idx in slice_indices {
                    if skip > 1 && (count % skip) != 0 {
                        count += 1;
                        continue;
                    }
                    count += 1;
                    let amp_offset = amplitudes.len() as u32;
                    let row = sh_view.row(compact_idx);
                    sample_plan.apply_row_into(row, &mut sampled);
                    if *hemisphere {
                        append_mirrored_hemisphere_row(&sampled, &mut amplitudes);
                    } else {
                        amplitudes.extend_from_slice(&sampled);
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
        let GlyphFieldSource::Odf { hemisphere, .. } = self.glyph_source.as_ref()? else {
            return None;
        };
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

            let amp_offset = output_row.saturating_mul(self.nb_sphere_vertices) as u32;
            let _ = hemisphere;
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

fn append_mirrored_hemisphere_row(row: &[f32], out: &mut Vec<f32>) {
    out.extend_from_slice(row);
    out.extend_from_slice(row);
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

    #[test]
    fn lazy_odf_slice_matches_stored_row() {
        let scene = OdxScene::from_dataset(build_test_dataset_with_odf(false)).unwrap();
        let slice = scene.glyphs_for_slice(2, 0, 1);
        assert_eq!(slice.instances.len(), 1);
        assert_eq!(slice.amplitudes.len(), scene.nb_sphere_vertices);
        let expected: Vec<f32> = (1..=scene.nb_sphere_vertices).map(|v| v as f32).collect();
        assert_eq!(slice.amplitudes, expected);
        assert_eq!(slice.amp_norm, scene.nb_sphere_vertices as f32);
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
        assert_eq!(metadata.instances.len(), scene.glyph_instances_for_slice(2, 0).len());
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
        let slice = scene.glyphs_for_slice(2, 0, 1);
        let coeffs = scene
            .dataset()
            .sh::<f32>("coefficients")
            .unwrap()
            .row(0)
            .to_vec();
        let expected =
            mrtrix_sh::sample_rows_nonnegative(&coeffs, 1, &scene.sphere_vertices, coeffs.len())
                .unwrap();
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

    fn reconstruct_odf_slice(
        scene: &OdxScene,
        metadata: &OdfSliceMetadata,
        rows_per_chunk: usize,
    ) -> Vec<f32> {
        let view = scene.odf_view_f32().unwrap();
        let source_bins = scene.odf_source_row_width().unwrap();
        let full_bins = scene.glyph_row_width();
        let mut out = vec![0.0f32; metadata.instances.len() * full_bins];
        for chunk in &metadata.chunk_worklists {
            for work_item in &chunk.work_items {
                let compact_idx =
                    chunk.chunk_index * rows_per_chunk + work_item.local_row as usize;
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

//! Input prep, output assembly, and filter helpers shared by the Dipy
//! probabilistic and PTT GPU paths.
//!
//! The two shader pipelines consume the same scene-derived inputs (SH
//! coefficients, B-matrix, voxel LUT, sphere vertices, voxel-space seeds,
//! affines); they differ only in which buffers they bind and which
//! shader they compile. `DipyGpuInputs` bundles the shared prep so each
//! path is `prepare + shader-specific-dispatch + assemble`.

use std::sync::Arc;

use glam::Mat4;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::data::loaded_files::StreamlineBacking;
use crate::data::trx_data::TrxGpuData;
use crate::error::{WorkflowError, WorkflowResult};
use crate::units::StreamlineIndex;
use crate::workflow::tracking_filters::{
    streamline_endpoint_in, streamline_hits_all_rois, streamline_passes_hausdorff,
    streamline_satisfies_end_masks,
};
use crate::workflow::{DipyTractographyPlan, PostFilter, StreamlineDataset, StreamlineFlow};

/// Maximum seeds dispatched in a single GPU batch.
pub(super) const BATCH_SIZE: u32 = 2048;

/// Bytes per f32.
pub(super) const F32_SIZE: u64 = 4;

/// Shared inputs for any Dipy-style GPU tracker. Built once per
/// `run_gpu_dipy*` call by `prepare_dipy_inputs`.
pub(super) struct DipyGpuInputs {
    pub(super) ncoeffs: u32,
    pub(super) n_dirs: u32,
    pub(super) nx: u32,
    pub(super) ny: u32,
    pub(super) nz: u32,
    pub(super) max_pts: u32,
    pub(super) min_pts: u32,
    pub(super) step_size_vox: f32,
    pub(super) max_angle_cos: f32,
    pub(super) vox_to_ras: Mat4,

    pub(super) seeds_flat: Vec<f32>,
    pub(super) total_seeds: u32,

    pub(super) sh_buf: wgpu::Buffer,
    pub(super) b_buf: wgpu::Buffer,
    pub(super) lut_buf: wgpu::Buffer,
    pub(super) sv_buf: wgpu::Buffer,
}

/// Build all the GPU-uniform inputs for a Dipy tracker. Returns
/// `Ok(None)` when the seed mask is empty (caller should use `empty_flow`).
/// Returns `Err` when the ODX lacks SH coefficients.
pub(super) fn prepare_dipy_inputs(
    plan: &DipyTractographyPlan,
    device: &wgpu::Device,
) -> WorkflowResult<Option<DipyGpuInputs>> {
    let scene = &plan.odx_scene;

    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation("ODX file has no SH coefficients. Re-derive with odx-rs.".into())
    })?;
    let ncoeffs = sh_view.ncols() as u32;
    let nb_voxels = sh_view.nrows();

    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for GPU tractography.".into())
    })?;
    let sphere_verts = mesh.vertices();
    let n_dirs = sphere_verts.len() as u32;

    let b_matrix_flat: Vec<f32> = mesh.transform_flat().to_vec();

    let sh_flat: Vec<f32> = (0..nb_voxels)
        .flat_map(|i| sh_view.row(i).iter().copied())
        .collect();

    let dims = scene.dimensions();
    let [nx, ny, nz] = [dims[0] as u32, dims[1] as u32, dims[2] as u32];
    let lut_len = (nx * ny * nz) as usize;
    let mut lut: Vec<u32> = vec![0xFFFF_FFFFu32; lut_len];
    for (compact_idx, &[ix, iy, iz]) in scene.ijk_lookup().iter().enumerate() {
        let flat =
            ix as usize * ny as usize * nz as usize + iy as usize * nz as usize + iz as usize;
        lut[flat] = compact_idx as u32;
    }

    let sphere_verts_flat: Vec<f32> = sphere_verts
        .iter()
        .flat_map(|v: &[f32; 3]| v.iter().copied())
        .collect();

    let vox_to_ras: Mat4 = scene.voxel_to_ras();
    let ras_to_vox: Mat4 = vox_to_ras.inverse();

    // Smallest voxel dimension (mm) used to convert step_size_mm → voxel
    // units. The shader works in voxel space because the LUT lives there.
    let smallest_vs = vox_to_ras
        .col(0)
        .truncate()
        .length()
        .min(vox_to_ras.col(1).truncate().length())
        .min(vox_to_ras.col(2).truncate().length())
        .max(1e-3);
    let step_size_vox = plan.step_size_mm / smallest_vs;
    let max_angle_cos = plan.max_angle_deg.to_radians().cos();

    // Seed points: wired mask if present, else whole-brain (every
    // voxel in the ODX compact mask). Matches Yeh's whole-brain default
    // when no mask is wired.
    let seeds_ras: Vec<[f32; 3]> = match plan.seed_mask.as_deref() {
        Some(mask) => mask.nonzero_voxel_centers_ras(),
        None => scene.centers_ras().to_vec(),
    };
    let seeds_vox: Vec<[f32; 3]> = seeds_ras
        .iter()
        .flat_map(|seed| {
            (0..plan.seeds_per_voxel).map(move |_| {
                let v = ras_to_vox.transform_point3(glam::Vec3::from_array(*seed));
                [v.x, v.y, v.z]
            })
        })
        .collect();
    let total_seeds = seeds_vox.len() as u32;
    if total_seeds == 0 {
        return Ok(None);
    }

    let seeds_flat: Vec<f32> = seeds_vox
        .iter()
        .flat_map(|s: &[f32; 3]| s.iter().copied())
        .collect();

    let max_pts = plan.max_points;
    let min_pts = (plan.min_len_mm / plan.step_size_mm).ceil() as u32;

    let sh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_sh_coeffs"),
        contents: bytemuck::cast_slice(&sh_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_b_matrix"),
        contents: bytemuck::cast_slice(&b_matrix_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_lut"),
        contents: bytemuck::cast_slice(&lut),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dipy_sphere_verts"),
        contents: bytemuck::cast_slice(&sphere_verts_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    Ok(Some(DipyGpuInputs {
        ncoeffs,
        n_dirs,
        nx,
        ny,
        nz,
        max_pts,
        min_pts,
        step_size_vox,
        max_angle_cos,
        vox_to_ras,
        seeds_flat,
        total_seeds,
        sh_buf,
        b_buf,
        lut_buf,
        sv_buf,
    }))
}

/// Apply the full post-hoc filter slate to a GPU-produced streamline.
/// Matches what `cpu_dipy` applies per attempt — ROI / end / no_end /
/// Hausdorff. Per-step masks (`limiting_mask`, `roa_mask`, `term_mask`)
/// are **not** enforced on the GPU path: the shader doesn't currently
/// do per-step mask lookups. Documented gap for PR 2b / 2c.
pub(super) fn keep_streamline_for_plan(
    plan: &DipyTractographyPlan,
    streamline: &[[f32; 3]],
) -> bool {
    if !plan.roi_masks.is_empty() && !streamline_hits_all_rois(streamline, &plan.roi_masks) {
        return false;
    }
    if let Some(ne) = plan.no_end_mask.as_deref()
        && streamline_endpoint_in(streamline, ne)
    {
        return false;
    }
    if !plan.end_masks.is_empty() && !streamline_satisfies_end_masks(streamline, &plan.end_masks) {
        return false;
    }
    if let Some(PostFilter::Hausdorff {
        reference_points_ras,
        max_mm,
    }) = plan.post_filter.as_ref()
        && !streamline_passes_hausdorff(streamline, reference_points_ras, *max_mm)
    {
        return false;
    }
    true
}

/// Wrap a positions+offsets pair into a `StreamlineFlow` ready to
/// return from a tracker.
pub(super) fn assemble_flow(
    plan: &DipyTractographyPlan,
    positions: Vec<[f32; 3]>,
    offsets: Vec<u32>,
) -> StreamlineFlow {
    let nb_streamlines = offsets.len() - 1;
    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(positions, offsets));
    let selected: Vec<StreamlineIndex> = (0..nb_streamlines as u32).map(StreamlineIndex).collect();
    let dataset = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
    });
    StreamlineFlow {
        dataset,
        selected_streamlines: selected,
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    }
}

/// Empty-flow result used by the prob/PTT paths when the seed mask is
/// empty. Distinct from a successful run that yielded zero streamlines
/// after filtering — this one skips the pipeline entirely.
pub(super) fn empty_flow(plan: &DipyTractographyPlan) -> WorkflowResult<StreamlineFlow> {
    let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(vec![], vec![0]));
    let dataset = Arc::new(StreamlineDataset {
        name: plan.label.clone(),
        gpu_data,
        backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
    });
    Ok(StreamlineFlow {
        dataset,
        selected_streamlines: vec![],
        color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
    })
}

/// Precompute FOD amplitudes evaluated at every sphere vertex for every
/// voxel: `amp[v, d] = max(0, Σ_k sh[v, k] · b_matrix[d, k])`.
///
/// Used by the PTT shader so the per-step FOD-at-direction lookup is a
/// single trilinear interp of one channel, instead of recomputing the
/// SH dot product (8 corners × ncoeffs ≈ 360 mults) on every call.
/// ~4.5× speedup per FOD call (the dominant inner-loop cost).
///
/// Output buffer size: nb_voxels × n_dirs × 4 bytes. For a typical
/// 50k-voxel × 162-dir volume, ~32 MB — fits comfortably in any GPU
/// memory budget. Cost: nb_voxels × n_dirs × ncoeffs mults on the CPU;
/// we parallelize the outer-voxel loop with rayon so the precompute
/// amortizes to ~hundreds of ms on a modern machine.
pub(super) fn precompute_fod_amplitudes(
    plan: &DipyTractographyPlan,
) -> WorkflowResult<(Vec<f32>, u32, u32, usize)> {
    let scene = &plan.odx_scene;
    let sh_view = scene.sh_view_f32().ok_or_else(|| {
        WorkflowError::Evaluation("ODX file has no SH coefficients. Re-derive with odx-rs.".into())
    })?;
    let mesh = scene.sh_render_mesh(2).ok_or_else(|| {
        WorkflowError::Evaluation("Could not build SH render mesh for GPU tractography.".into())
    })?;
    let n_dirs = mesh.vertices().len();
    let ncoeffs = sh_view.ncols();
    let nb_voxels = sh_view.nrows();
    let b_matrix = mesh.transform_flat();

    let mut amps = vec![0.0f32; nb_voxels * n_dirs];
    amps.par_chunks_mut(n_dirs)
        .enumerate()
        .for_each(|(v, voxel_slice)| {
            let sh_row = sh_view.row(v);
            for d in 0..n_dirs {
                let b_base = d * ncoeffs;
                let mut sum = 0.0f32;
                for k in 0..ncoeffs {
                    sum += sh_row[k] * b_matrix[b_base + k];
                }
                voxel_slice[d] = sum.max(0.0);
            }
        });

    Ok((amps, n_dirs as u32, ncoeffs as u32, nb_voxels))
}

// ── bind group layout entry helpers ──────────────────────────────────────

pub(super) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

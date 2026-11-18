//! Bridge from `trxviz-core` to the Python `mpat.trxviz_bridge` CLI.
//!
//! Handles:
//! 1. Resolving the Python interpreter path from:
//!    - Node-level UI text box override (`node.python_path`)
//!    - CLI flag override (`--python-env` / `--mpat-python`, stored via [`set_cli_python_path`])
//!    - Environment variable (`MPAT_PYTHON`)
//!    - `python3` on `PATH`
//!
//!    The interpreter's environment must have `mpat` installed (it brings `mittens-dmri`
//!    and `odx`); nothing is added to `PYTHONPATH`.
//! 2. Serializing `VoxelMask` instances to uncompressed NIfTI-1 `.nii` files in a staging folder.
//! 3. Spawning `python -m mpat.trxviz_bridge build_transitions <req.json>` and
//!    `python -m mpat.trxviz_bridge solve_bundle <req.json>`, streaming `PROGRESS <done> <total>`
//!    updates, live 3-D volume frames from `live_density.mmap` (`STREAM_FRAME`), and honoring [`CancelFlag`].
//! 4. Loading the resulting `.nii` scalar maps into `Arc<VolumeScalars>` and `.trx` streamlines
//!    into `StreamlineFlow`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use trx_rs::AnyTrxFile;

use crate::data::cifti::{ScalarKind, ScalarMetadata, VolumeScalars};
use crate::data::loaded_files::StreamlineBacking;
use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::{ColorMode, TrxGpuData};
use crate::error::{WorkflowError, WorkflowResult};
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::units::StreamlineIndex;

use super::eval_inputs::volume_scalars_from_nifti_volume;
use super::tracking::CancelFlag;
use super::types::{
    MpatTractographyJob, MpatTractographyOutput, MpatTransitionsHandle, MpatTransitionsJob,
    MpatTransitionsOutput, StreamlineDataset, StreamlineFlow, VoxelMask,
};

static CLI_PYTHON_PATH: OnceLock<String> = OnceLock::new();

/// Record the CLI `--python-env` / `--mpat-python` override for MPAT subprocesses.
pub fn set_cli_python_path(path: impl Into<String>) {
    let s = path.into().trim().to_string();
    if !s.is_empty() {
        let _ = CLI_PYTHON_PATH.set(s);
    }
}

/// Read the CLI Python interpreter path override, if one was passed at startup.
pub fn cli_python_path() -> Option<&'static str> {
    CLI_PYTHON_PATH.get().map(String::as_str)
}

/// Resolve the Python interpreter binary to use for `mpat.trxviz_bridge`.
pub fn resolve_mpat_python(node_override: &str) -> String {
    let trimmed = node_override.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    if let Some(cli) = cli_python_path() {
        if !cli.trim().is_empty() {
            return cli.trim().to_string();
        }
    }
    if let Ok(env_val) = std::env::var("MPAT_PYTHON") {
        let env_trimmed = env_val.trim();
        if !env_trimmed.is_empty() {
            return env_trimmed.to_string();
        }
    }
    "python3".to_string()
}

/// Write a `VoxelMask` to a minimal uncompressed 352-byte-header NIfTI-1 `.nii` file.
/// Returns `Ok(false)` if the mask is empty (so the caller can omit it from the request).
pub fn write_voxel_mask_nifti(mask: &VoxelMask, path: &Path) -> WorkflowResult<bool> {
    if mask.is_empty() || mask.dims.contains(&0) {
        return Ok(false);
    }
    let [nx, ny, nz] = mask.dims;
    let expected_len = (nx as usize) * (ny as usize) * (nz as usize);
    if mask.data.len() != expected_len {
        return Err(WorkflowError::Evaluation(format!(
            "VoxelMask data length {} does not match dims {:?}",
            mask.data.len(),
            mask.dims
        )));
    }

    let mut hdr = [0u8; 352];
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());
    let dims_i16: [i16; 8] = [3, nx as i16, ny as i16, nz as i16, 1, 1, 1, 1];
    for (idx, &d) in dims_i16.iter().enumerate() {
        let off = 40 + idx * 2;
        hdr[off..off + 2].copy_from_slice(&d.to_le_bytes());
    }
    hdr[70..72].copy_from_slice(&2i16.to_le_bytes());
    hdr[72..74].copy_from_slice(&8i16.to_le_bytes());

    let cols = mask.voxel_to_ras.to_cols_array_2d();
    let dx = (cols[0][0] * cols[0][0] + cols[0][1] * cols[0][1] + cols[0][2] * cols[0][2]).sqrt();
    let dy = (cols[1][0] * cols[1][0] + cols[1][1] * cols[1][1] + cols[1][2] * cols[1][2]).sqrt();
    let dz = (cols[2][0] * cols[2][0] + cols[2][1] * cols[2][1] + cols[2][2] * cols[2][2]).sqrt();
    let pixdim: [f32; 8] = [
        1.0,
        dx.max(1e-3),
        dy.max(1e-3),
        dz.max(1e-3),
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (idx, &p) in pixdim.iter().enumerate() {
        let off = 76 + idx * 4;
        hdr[off..off + 4].copy_from_slice(&p.to_le_bytes());
    }
    hdr[108..112].copy_from_slice(&352.0f32.to_le_bytes());
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes());
    hdr[116..120].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[123] = 2;
    hdr[252..254].copy_from_slice(&0i16.to_le_bytes());
    hdr[254..256].copy_from_slice(&1i16.to_le_bytes());

    let srows: [[f32; 4]; 3] = [
        [cols[0][0], cols[1][0], cols[2][0], cols[3][0]],
        [cols[0][1], cols[1][1], cols[2][1], cols[3][1]],
        [cols[0][2], cols[1][2], cols[2][2], cols[3][2]],
    ];
    for (r, row) in srows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            let off = 280 + (r * 4 + c) * 4;
            hdr[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    hdr[344..348].copy_from_slice(b"n+1\0");

    let mut file = std::fs::File::create(path).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to create NIfTI mask {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(&hdr).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to write NIfTI header {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(&mask.data).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to write NIfTI data {}: {e}",
            path.display()
        ))
    })?;
    Ok(true)
}

fn load_volume_scalars_from_path(
    path: &Path,
    map_name: &str,
) -> WorkflowResult<Arc<VolumeScalars>> {
    let nifti = NiftiVolume::load(path).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to load MPAT output NIfTI {} ({}): {e}",
            map_name,
            path.display()
        ))
    })?;
    Ok(Arc::new(volume_scalars_from_nifti_volume(
        &nifti,
        map_name.to_string(),
        0,
    )))
}

/// Read the active double-buffer slot from `live_density.mmap` written by `LiveDensityStreamer`.
pub fn read_live_density_mmap(mmap_path: &Path) -> Option<(Arc<VolumeScalars>, String)> {
    let mut file = std::fs::File::open(mmap_path).ok()?;
    let mut hdr = [0u8; 128];
    file.read_exact(&mut hdr).ok()?;
    let magic = u32::from_le_bytes(hdr[0..4].try_into().ok()?);
    let version = u32::from_le_bytes(hdr[4..8].try_into().ok()?);
    if magic != 0x5441504D || version != 1 {
        return None;
    }
    let seq = u32::from_le_bytes(hdr[8..12].try_into().ok()?);
    if seq == 0 {
        return None;
    }
    let phase = u32::from_le_bytes(hdr[12..16].try_into().ok()?);
    let step = u32::from_le_bytes(hdr[16..20].try_into().ok()?);
    let nx = u32::from_le_bytes(hdr[20..24].try_into().ok()?) as usize;
    let ny = u32::from_le_bytes(hdr[24..28].try_into().ok()?) as usize;
    let nz = u32::from_le_bytes(hdr[28..32].try_into().ok()?) as usize;
    let residual = f32::from_le_bytes(hdr[32..36].try_into().ok()?);
    let max_val = f32::from_le_bytes(hdr[36..40].try_into().ok()?);
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }

    let mut cols = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let off = 40 + (r * 4 + c) * 4;
            cols[c][r] = f32::from_le_bytes(hdr[off..off + 4].try_into().ok()?);
        }
    }
    let voxel_to_ras = glam::Mat4::from_cols_array_2d(&cols);

    let nvox = nx.checked_mul(ny)?.checked_mul(nz)?;
    let slot_bytes = nvox.checked_mul(4)?;
    let slot_idx = (seq as usize) % 2;
    let offset = 128u64 + (slot_idx as u64) * (slot_bytes as u64);
    file.seek(SeekFrom::Start(offset)).ok()?;

    let mut raw = vec![0u8; slot_bytes];
    file.read_exact(&mut raw).ok()?;
    let norm = if max_val.is_finite() && max_val > 1e-12 {
        1.0 / max_val
    } else {
        0.0
    };
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() && v > 0.0 {
                (v * norm).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();

    let phase_label = match phase {
        1 => "Phase 1/2 — Backward Power Iteration",
        2 => "Phase 2/2 — Forward Power Iteration",
        3 => "Converged Bundle Density",
        _ => "Power Iteration",
    };
    let summary = format!("{phase_label}: step t={step} (remaining mass {residual:.2e})");
    let vol = VolumeScalars {
        dims: [nx, ny, nz],
        voxel_to_ras,
        values,
        kind: ScalarKind::Continuous,
        metadata: ScalarMetadata {
            map_name: format!("Live {phase_label} #{seq}"),
            suggested_range: Some((0.0, 1.0)),
            series_index: Some(seq as usize),
            series_value: Some(step as f32),
            label_table: Vec::new(),
        },
    };
    Some((Arc::new(vol), summary))
}

fn spawn_mpat_bridge<F>(
    python_bin: &str,
    subcommand: &str,
    req_path: &Path,
    mmap_path: Option<PathBuf>,
    cancel: &CancelFlag,
    progress_cb: F,
) -> WorkflowResult<serde_json::Value>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let mut cmd = Command::new(python_bin);
    cmd.arg("-m")
        .arg("mpat.trxviz_bridge")
        .arg(subcommand)
        .arg(req_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to launch Python interpreter '{python_bin}' for MPAT ({subcommand}): {e}. \
             Set the Python path in the node inspector, via --python-env, or MPAT_PYTHON."
        ))
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| WorkflowError::Evaluation("Failed to capture MPAT stdout".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| WorkflowError::Evaluation("Failed to capture MPAT stderr".into()))?;

    let stderr_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut lines = Vec::new();
        for line in reader.lines().map_while(Result::ok) {
            lines.push(line);
        }
        lines.join("\n")
    });

    let cancel_for_stdout = cancel.clone();
    let stdout_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut result_json: Option<String> = None;
        for line in reader.lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("PROGRESS ") {
                let mut parts = rest.split_whitespace();
                if let (Some(d_str), Some(t_str)) = (parts.next(), parts.next()) {
                    if let (Ok(done), Ok(total)) = (d_str.parse::<u64>(), t_str.parse::<u64>()) {
                        progress_cb(done, total);
                    }
                }
            } else if trimmed.starts_with("STREAM_FRAME ") {
                if let Some(mp) = &mmap_path {
                    if let Some((vol, summary)) = read_live_density_mmap(mp) {
                        cancel_for_stdout.report_live_volume(vol, summary);
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("RESULT ") {
                result_json = Some(rest.to_string());
            }
        }
        result_json
    });

    loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(WorkflowError::Evaluation("cancelled by user".into()));
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let result_str = stdout_handle.join().unwrap_or(None);
                let stderr_str = stderr_handle.join().unwrap_or_default();
                if !status.success() {
                    let detail = if stderr_str.trim().is_empty() {
                        format!("exit status {status}")
                    } else {
                        stderr_str
                            .lines()
                            .rev()
                            .take(12)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    return Err(WorkflowError::Evaluation(format!(
                        "MPAT {subcommand} failed ({python_bin}): {detail}"
                    )));
                }
                if let Some(json_text) = result_str {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&json_text).map_err(|e| {
                            WorkflowError::Evaluation(format!(
                                "Failed to parse MPAT {subcommand} RESULT JSON: {e}"
                            ))
                        })?;
                    return Ok(parsed);
                } else {
                    return Err(WorkflowError::Evaluation(format!(
                        "MPAT {subcommand} exited cleanly but emitted no RESULT JSON. Stderr: {stderr_str}"
                    )));
                }
            }
            Ok(None) => {
                std::thread::sleep(Duration::from_millis(40));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(WorkflowError::Evaluation(format!(
                    "Error waiting on MPAT {subcommand} subprocess: {e}"
                )));
            }
        }
    }
}

/// Run the `MpatTransitionsOp` background job via `python -m mpat.trxviz_bridge build_transitions`.
pub fn run_mpat_transitions<F>(
    job: &MpatTransitionsJob,
    cancel: &CancelFlag,
    progress_cb: F,
) -> WorkflowResult<MpatTransitionsOutput>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let python_bin = resolve_mpat_python(&job.python_path);
    let staging_root = std::env::temp_dir().join("trxviz_mpat_cache");
    let out_dir = staging_root.join(format!(
        "transitions_node{}_{:016x}",
        job.node_uuid.0, job.fingerprint
    ));
    std::fs::create_dir_all(&out_dir).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to create MPAT transitions staging dir {}: {e}",
            out_dir.display()
        ))
    })?;

    let mut transient_mask_path: Option<String> = None;
    if let Some(mask) = &job.transient_mask {
        let p = out_dir.join("input_transient_mask.nii");
        if write_voxel_mask_nifti(mask, &p)? {
            transient_mask_path = Some(p.to_string_lossy().to_string());
        }
    }

    let mut absorb_mask_path: Option<String> = None;
    if let Some(mask) = &job.absorb_mask {
        let p = out_dir.join("input_absorb_mask.nii");
        if write_voxel_mask_nifti(mask, &p)? {
            absorb_mask_path = Some(p.to_string_lossy().to_string());
        }
    }

    let req_json = serde_json::json!({
        "out_dir": out_dir.to_string_lossy(),
        "cache_dir": staging_root.to_string_lossy(),
        "working_dir": if job.working_dir.trim().is_empty() { None } else { Some(job.working_dir.trim()) },
        "odx_path": job.odx_path.as_ref().map(|p| p.to_string_lossy().to_string()),
        "transient_mask_path": transient_mask_path,
        "absorb_mask_path": absorb_mask_path,
        "preset": job.preset.as_str(),
        "angle_deg": job.angle_deg,
        "step_vox": job.step_vox,
        "double_odf": job.double_odf,
        "fod_thresh": job.fod_thresh,
        "fod_power": job.fod_power,
        "stop_mass_tau": job.stop_mass_tau,
        "curv_sigma_deg": job.curv_sigma_deg,
        "probe_radius_vox": job.probe_radius_vox,
        "max_length_mm": job.max_length_mm,
        "wm_threshold": job.wm_threshold,
    });

    let req_path = out_dir.join("request_transitions.json");
    std::fs::write(&req_path, serde_json::to_string_pretty(&req_json).unwrap()).map_err(|e| {
        WorkflowError::Evaluation(format!("Failed to write {}: {e}", req_path.display()))
    })?;

    let result_val = spawn_mpat_bridge(
        &python_bin,
        "build_transitions",
        &req_path,
        None,
        cancel,
        progress_cb,
    )?;

    let state_path = PathBuf::from(
        result_val
            .get("state_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::Evaluation("Missing state_path in MPAT result".into()))?,
    );
    let summary = result_val
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap_or("MPAT transitions ready")
        .to_string();

    let dims_vec: Vec<u32> = result_val
        .get("dims")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();
    let dims = if dims_vec.len() == 3 {
        [dims_vec[0], dims_vec[1], dims_vec[2]]
    } else {
        [1, 1, 1]
    };

    let mut voxel_to_ras = glam::Mat4::IDENTITY;
    if let Some(rows) = result_val.get("voxel_to_ras").and_then(|v| v.as_array()) {
        if rows.len() == 4 {
            let mut cols = [[0.0f32; 4]; 4];
            for (r, row_val) in rows.iter().enumerate() {
                if let Some(row_arr) = row_val.as_array() {
                    for (c, val) in row_arr.iter().take(4).enumerate() {
                        cols[c][r] = val.as_f64().unwrap_or(0.0) as f32;
                    }
                }
            }
            voxel_to_ras = glam::Mat4::from_cols_array_2d(&cols);
        }
    }

    let vols_obj = result_val
        .get("volumes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| WorkflowError::Evaluation("Missing volumes object in MPAT result".into()))?;

    let load_vol = |key: &str, label: &str| -> WorkflowResult<Arc<VolumeScalars>> {
        let p_str = vols_obj.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
            WorkflowError::Evaluation(format!("Missing volume '{key}' in MPAT transitions result"))
        })?;
        load_volume_scalars_from_path(Path::new(p_str), label)
    };

    let codi_double = load_vol("codi_double", "CoDI (Double-ODF)")?;
    let codi_single = load_vol("codi_single", "CoDI (Single-ODF)")?;
    let coasy = load_vol("coasy", "CoAsy (Compositional Asymmetry)")?;
    let p_terminate_double = load_vol("p_terminate_double", "P(Terminate Double-ODF)")?;
    let p_terminate_single = load_vol("p_terminate_single", "P(Terminate Single-ODF)")?;
    let total_amplitude = load_vol("total_amplitude", "Total ODF Amplitude")?;
    let kl_single = load_vol("kl_single", "KL Divergence (Single-ODF)")?;
    let chain_dead = load_vol("chain_dead", "Chain Dead-State Prob")?;

    let handle = Arc::new(MpatTransitionsHandle {
        fingerprint: job.fingerprint,
        state_path,
        working_dir: if job.working_dir.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(job.working_dir.trim()))
        },
        dims,
        voxel_to_ras,
        summary: summary.clone(),
    });

    Ok(MpatTransitionsOutput {
        handle,
        codi_double,
        codi_single,
        coasy,
        p_terminate_double,
        p_terminate_single,
        total_amplitude,
        kl_single,
        chain_dead,
        summary,
    })
}

/// Run the `MpatTractographyOp` background job via `python -m mpat.trxviz_bridge solve_bundle`.
pub fn run_mpat_tractography<F>(
    job: &MpatTractographyJob,
    cancel: &CancelFlag,
    progress_cb: F,
) -> WorkflowResult<MpatTractographyOutput>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let python_bin = resolve_mpat_python(&job.python_path);
    let staging_root = std::env::temp_dir().join("trxviz_mpat_cache");
    let out_dir = staging_root.join(format!(
        "tractography_node{}_{:016x}",
        job.node_uuid.0, job.fingerprint
    ));
    std::fs::create_dir_all(&out_dir).map_err(|e| {
        WorkflowError::Evaluation(format!(
            "Failed to create MPAT tractography staging dir {}: {e}",
            out_dir.display()
        ))
    })?;

    let mut seed_mask_path: Option<String> = None;
    let seed_mask_ref = job.seed_mask.as_ref().or_else(|| {
        job.tracking_plan
            .as_ref()
            .and_then(|p| p.seed_mask.as_ref())
    });
    if let Some(mask) = seed_mask_ref {
        let p = out_dir.join("seed_mask.nii");
        if write_voxel_mask_nifti(mask, &p)? {
            seed_mask_path = Some(p.to_string_lossy().to_string());
        }
    }

    let mut include_mask_paths = Vec::new();
    let mut exclude_mask_path: Option<String> = None;
    let mut start_mask_path: Option<String> = None;
    let mut end_mask_path: Option<String> = None;
    let mut term_mask_path: Option<String> = None;
    let mut no_end_mask_path: Option<String> = None;
    let mut limiting_mask_path: Option<String> = None;

    if let Some(plan) = &job.tracking_plan {
        for (idx, roi) in plan.roi_masks.iter().enumerate() {
            let p = out_dir.join(format!("include_roi_{idx:02}.nii"));
            if write_voxel_mask_nifti(roi, &p)? {
                include_mask_paths.push(p.to_string_lossy().to_string());
            }
        }
        if let Some(roa) = &plan.roa_mask {
            let p = out_dir.join("exclude_roa.nii");
            if write_voxel_mask_nifti(roa, &p)? {
                exclude_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
        if let Some(start) = plan.end_masks.first() {
            let p = out_dir.join("start_endpoint.nii");
            if write_voxel_mask_nifti(start, &p)? {
                start_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
        if plan.end_masks.len() >= 2 {
            // Masks on other grids are resampled onto the first end mask's grid, as the
            // plan's own Add nodes do.
            let mut end_union = (*plan.end_masks[1]).clone();
            for extra in &plan.end_masks[2..] {
                end_union = crate::workflow::ops::union_masks(&end_union, extra);
            }
            let p = out_dir.join("end_endpoint.nii");
            if write_voxel_mask_nifti(&end_union, &p)? {
                end_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
        if let Some(term) = &plan.term_mask {
            let p = out_dir.join("term_absorption.nii");
            if write_voxel_mask_nifti(term, &p)? {
                term_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
        if let Some(no_end) = &plan.no_end_mask {
            let p = out_dir.join("no_end_mask.nii");
            if write_voxel_mask_nifti(no_end, &p)? {
                no_end_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
        if let Some(limiting) = &plan.limiting_mask {
            let p = out_dir.join("limiting_mask.nii");
            if write_voxel_mask_nifti(limiting, &p)? {
                limiting_mask_path = Some(p.to_string_lossy().to_string());
            }
        }
    }

    let bundle_name = job
        .tracking_plan
        .as_ref()
        .map(|p| p.label.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| job.label.clone());

    let mmap_path = out_dir.join("live_density.mmap");
    let req_json = serde_json::json!({
        "state_path": job.transitions.state_path.to_string_lossy(),
        "out_dir": out_dir.to_string_lossy(),
        "stream_mmap_path": mmap_path.to_string_lossy(),
        "bundle_name": bundle_name,
        "seed_mask_path": seed_mask_path,
        "include_mask_paths": include_mask_paths,
        "exclude_mask_path": exclude_mask_path,
        "start_mask_path": start_mask_path,
        "end_mask_path": end_mask_path,
        "term_mask_path": term_mask_path,
        "no_end_mask_path": no_end_mask_path,
        "limiting_mask_path": limiting_mask_path,
        "seal_26": job.seal_26,
        "midline_exclusion": job.midline_exclusion,
        "midline_tol_mm": job.midline_tol_mm,
        "solver_tol": job.solver_tol,
        "solver_accel": job.solver_accel.as_str(),
        "extract_streamlines": job.extract_streamlines,
        "extract_arbor": job.extract_arbor,
        "n_streamlines": job.n_streamlines,
        "n_arbor_branches": job.n_arbor_branches,
        "step_vox": job.step_vox,
        "min_length_mm": job.min_length_mm,
        "max_length_mm": job.max_length_mm,
    });

    let req_path = out_dir.join("request_solve.json");
    std::fs::write(&req_path, serde_json::to_string_pretty(&req_json).unwrap()).map_err(|e| {
        WorkflowError::Evaluation(format!("Failed to write {}: {e}", req_path.display()))
    })?;

    let result_val = spawn_mpat_bridge(
        &python_bin,
        "solve_bundle",
        &req_path,
        Some(mmap_path),
        cancel,
        progress_cb,
    )?;

    let summary = result_val
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap_or("MPAT tractography ready")
        .to_string();

    let trx_path = PathBuf::from(
        result_val
            .get("trx_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::Evaluation("Missing trx_path in MPAT result".into()))?,
    );

    let flow = if trx_path.exists() {
        let any = AnyTrxFile::load(&trx_path).map_err(|e| {
            WorkflowError::Evaluation(format!(
                "Failed to load MPAT output TRX {}: {e}",
                trx_path.display()
            ))
        })?;
        let mut gpu_data = TrxGpuData::from_any_trx(&any).map_err(|e| {
            WorkflowError::Evaluation(format!(
                "Failed to build GPU data from MPAT TRX {}: {e}",
                trx_path.display()
            ))
        })?;
        let color_mode = if gpu_data.dpv_names.iter().any(|n| n == "afd") {
            ColorMode::Dpv("afd".to_string())
        } else {
            ColorMode::DirectionRgb
        };
        gpu_data.recolor(&color_mode, None, SurfaceColormap::Inferno);
        let nb = gpu_data.nb_streamlines as u32;
        let dataset = Arc::new(StreamlineDataset {
            name: bundle_name,
            gpu_data: Arc::new(gpu_data),
            backing: StreamlineBacking::Native(Arc::new(any)),
        });
        StreamlineFlow {
            dataset,
            selected_streamlines: (0..nb).map(StreamlineIndex).collect(),
            color_mode,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            scalar_colormap: SurfaceColormap::Inferno,
        }
    } else {
        let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(Vec::new(), vec![0]));
        let dataset = Arc::new(StreamlineDataset {
            name: bundle_name,
            gpu_data,
            backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
        });
        StreamlineFlow {
            dataset,
            selected_streamlines: Vec::new(),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            scalar_colormap: SurfaceColormap::Inferno,
        }
    };

    let vols_obj = result_val
        .get("volumes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            WorkflowError::Evaluation("Missing volumes object in MPAT solve result".into())
        })?;

    let mut volumes = HashMap::new();
    for (key, label) in [
        ("afd", "AFD Map (Analytical SIFT2)"),
        ("density", "Visitation Density"),
        ("tdimm", "Physical TDI (mm)"),
        ("qharmonic", "Harmonic Coordinate (q)"),
        ("coherence", "Flux Coherence"),
        ("keff", "Branching K_eff"),
        ("endpoints", "Endpoints Map"),
    ] {
        if let Some(p_str) = vols_obj.get(key).and_then(|v| v.as_str()) {
            let vol = load_volume_scalars_from_path(Path::new(p_str), label)?;
            volumes.insert(key.to_string(), vol);
        }
    }

    Ok(MpatTractographyOutput {
        flow,
        afd: volumes
            .remove("afd")
            .ok_or_else(|| WorkflowError::Evaluation("Missing afd volume".into()))?,
        density: volumes
            .remove("density")
            .ok_or_else(|| WorkflowError::Evaluation("Missing density volume".into()))?,
        tdimm: volumes
            .remove("tdimm")
            .ok_or_else(|| WorkflowError::Evaluation("Missing tdimm volume".into()))?,
        qharmonic: volumes
            .remove("qharmonic")
            .ok_or_else(|| WorkflowError::Evaluation("Missing qharmonic volume".into()))?,
        coherence: volumes
            .remove("coherence")
            .ok_or_else(|| WorkflowError::Evaluation("Missing coherence volume".into()))?,
        keff: volumes
            .remove("keff")
            .ok_or_else(|| WorkflowError::Evaluation("Missing keff volume".into()))?,
        endpoints: volumes
            .remove("endpoints")
            .ok_or_else(|| WorkflowError::Evaluation("Missing endpoints volume".into()))?,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::op::WorkflowOp;
    use crate::workflow::ops::{MpatTractographyOp, MpatTransitionsOp};
    use crate::workflow::types::PortKind;

    #[test]
    fn mpat_nodes_expose_expected_visible_ports() {
        let trans = MpatTransitionsOp::default();
        assert_eq!(
            trans.input_ports().to_owned(),
            vec![PortKind::OdfField, PortKind::VoxelMask, PortKind::VoxelMask]
        );
        assert_eq!(
            trans.output_ports().to_owned(),
            vec![
                PortKind::MpatTransitions,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
            ]
        );

        let tract = MpatTractographyOp::default();
        assert_eq!(
            tract.input_ports().to_owned(),
            vec![
                PortKind::MpatTransitions,
                PortKind::VoxelMask,
                PortKind::TrackingPlan,
            ]
        );
        assert_eq!(
            tract.output_ports().to_owned(),
            vec![
                PortKind::Streamline,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
                PortKind::Volume,
            ]
        );
    }

    #[test]
    fn resolve_mpat_python_honors_node_and_cli_priority() {
        assert_eq!(
            resolve_mpat_python("/custom/node/python"),
            "/custom/node/python"
        );
        set_cli_python_path("/cli/flag/python");
        assert_eq!(resolve_mpat_python(""), "/cli/flag/python");
    }

    #[test]
    fn write_voxel_mask_nifti_round_trips_nonzero_mask() {
        let mut data = vec![0u8; 64];
        data[1 + 4 * (2 + 4 * 3)] = 1;
        let mask = VoxelMask {
            dims: [4, 4, 4],
            voxel_to_ras: glam::Mat4::IDENTITY,
            data,
            nonzero_centers: Default::default(),
            nonzero_indices: Default::default(),
        };
        let tmp = std::env::temp_dir().join("test_mpat_mask_roundtrip.nii");
        let written = write_voxel_mask_nifti(&mask, &tmp).unwrap();
        assert!(written);
        let vol = NiftiVolume::load(&tmp).unwrap();
        assert_eq!(vol.dims, [4, 4, 4]);
        let _ = std::fs::remove_file(&tmp);
    }
}

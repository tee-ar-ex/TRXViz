//! Build a `TrackingPlan` from a pyAFQ derivatives directory + bundle name.
//!
//! pyAFQ's `BundleDict.transform_rois` writes per-bundle warped subject-space
//! NIfTIs at `<base>/ROIs/<base>_space-{to_space}_desc-<BundleDesc><Role><N?>_mask.nii.gz`
//! (and `*_probseg.nii.gz` for the probability map), where
//! `BundleDesc = str_to_desc(bundle_name)` and `Role ∈ {Include, Exclude, Start, End}`.
//! See `pyAFQ/AFQ/api/bundle_dict.py:1489`.
//!
//! This module discovers those files for a single bundle, loads each one as a
//! `VoxelMask`, dilates it by the role-specific tolerance (matching pyAFQ's
//! `check_sls_with_inclusion` / `dist_to_atlas` semantics), and assembles a
//! `TrackingPlan` whose `roi_masks` (waypoint AND), `roa_mask` (exclusion
//! union), `end_masks` (start + end), and `post_filter` (prob map) wire
//! cleanly into the existing post-hoc filter pipeline.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::data::nifti_data::NiftiVolume;
use crate::workflow::ops::pyafq_bundles::{PyafqBundleSpec, str_to_desc};
use crate::workflow::{PostFilter, TrackingPlan, VoxelMask};

use super::mask_dilate::dilate_mask;

/// User-tunable parameters that mirror pyAFQ `segmentation_params`.
#[derive(Clone)]
pub struct PyafqPlanParams {
    pub to_space: String,
    pub dist_to_waypoint_mm: f32,
    pub dist_to_exclusion_mm: f32,
    pub dist_to_endpoint_mm: f32,
    pub prob_threshold: f32,
    pub override_min_len_mm: Option<f32>,
    pub override_max_len_mm: Option<f32>,
}

/// One per-role union mask plus the per-waypoint masks the plan filter needs.
pub struct PyafqPlanOutputs {
    pub plan: TrackingPlan,
    /// Union of all dilated `Include<N>` masks. For visualization only —
    /// the *individual* dilated waypoints live inside `plan.roi_masks` so
    /// AND-semantics are preserved.
    pub include_union: Arc<VoxelMask>,
    /// Union of all dilated `Exclude<N>` masks (== `plan.roa_mask` content).
    /// Empty mask when no exclusions exist.
    pub exclude_union: Arc<VoxelMask>,
    /// Dilated `Start` mask, or an empty mask when the bundle lacks one.
    pub start_mask: Arc<VoxelMask>,
    /// Dilated `End` mask, or an empty mask when the bundle lacks one.
    pub end_mask: Arc<VoxelMask>,
    /// Number of include / exclude / endpoint files actually loaded — used
    /// by the inspector to show "2 include / 0 exclude / 1 end / probmap loaded".
    pub n_include: usize,
    pub n_exclude: usize,
    pub has_start: bool,
    pub has_end: bool,
    pub has_prob_map: bool,
    /// Resolved file paths for the inspector / debug log.
    pub include_paths: Vec<PathBuf>,
    pub exclude_paths: Vec<PathBuf>,
    pub start_path: Option<PathBuf>,
    pub end_path: Option<PathBuf>,
    pub prob_map_path: Option<PathBuf>,
}

/// Files discovered for one bundle. Sorted: includes by ascending N,
/// excludes by ascending N. Start / End / probseg are single files (or
/// absent).
pub struct DiscoveredBundleFiles {
    pub includes: Vec<PathBuf>,
    pub excludes: Vec<PathBuf>,
    pub start: Option<PathBuf>,
    pub end: Option<PathBuf>,
    pub prob_map: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum PyafqPlanError {
    #[error(
        "no Include masks found for bundle '{bundle}' in {dir:?} (space={space}). \
         Spaces detected in this directory: {available_spaces:?}. \
         Older pyAFQ derivatives (e.g. HBN POD2) use space=T1w."
    )]
    NoIncludes {
        bundle: String,
        dir: PathBuf,
        space: String,
        available_spaces: Vec<String>,
    },
    #[error("failed to load NIfTI {path:?}: {source}")]
    Nifti {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
    #[error("waypoint masks have inconsistent grids: {0:?} vs {1:?}")]
    GridMismatch([u32; 3], [u32; 3]),
    #[error("I/O error walking {dir:?}: {source}")]
    Io {
        dir: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Walk `working_dir` recursively, collecting every `.nii.gz` that matches
/// either the modern pyAFQ ROI naming or the legacy (≤2022) naming for the
/// requested bundle. Returns paths grouped by role and sorted by `<N>` for
/// the indexed roles.
///
/// Modern: `*_space-{to_space}_desc-{BundleDesc}{Role}{N?}_mask.nii.gz`
///   where `BundleDesc = str_to_desc(display_name)`,
///   `Role ∈ {Include, Exclude, Start, End}`.
///
/// Legacy: `*_space-{to_space}_desc-{anything}_desc-ROI-{LegacyKey}-{N}-{role}.nii.gz`
///   where `LegacyKey ∈ bundle.legacy_keys`,
///   `role ∈ {include, exclude, start, end}`.
pub fn discover_bundle_files(
    working_dir: &Path,
    bundle: &PyafqBundleSpec,
    to_space: &str,
) -> Result<DiscoveredBundleFiles, PyafqPlanError> {
    let mut all_paths: Vec<PathBuf> = Vec::new();
    walk_collect(working_dir, &mut all_paths).map_err(|source| PyafqPlanError::Io {
        dir: working_dir.to_path_buf(),
        source,
    })?;

    let modern_desc = str_to_desc(bundle.display_name);
    let modern_token = format!("_space-{}_desc-{}", to_space, modern_desc);
    let space_prefix = format!("_space-{}_", to_space);

    let mut includes: Vec<(u32, PathBuf)> = Vec::new();
    let mut excludes: Vec<(u32, PathBuf)> = Vec::new();
    let mut start: Option<PathBuf> = None;
    let mut end: Option<PathBuf> = None;
    let mut prob_map: Option<PathBuf> = None;

    for path in all_paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".nii.gz") {
            continue;
        }

        // Try modern naming first.
        if let Some(desc_pos) = name.find(&modern_token) {
            let after_desc = &name[desc_pos + modern_token.len()..];
            if let Some(rest) = after_desc.strip_prefix("_probseg") {
                if rest == ".nii.gz" {
                    prob_map = Some(path);
                }
                continue;
            }
            if let Some(role_token) = after_desc.strip_suffix("_mask.nii.gz") {
                if let Some(idx_str) = role_token.strip_prefix("Include") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        includes.push((idx, path));
                        continue;
                    }
                } else if let Some(idx_str) = role_token.strip_prefix("Exclude") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        excludes.push((idx, path));
                        continue;
                    }
                } else if role_token == "Start" {
                    start = Some(path);
                    continue;
                } else if role_token == "End" {
                    end = Some(path);
                    continue;
                }
            }
            // Filename matched the modern space+desc prefix but not a known
            // suffix — skip without falling through to legacy parsing
            // (avoids misclassifying e.g. `_desc-{Bundle}_dwi.nii.gz`).
            continue;
        }

        // Fall back to legacy naming. Require the right `_space-{to_space}_`
        // tag so we don't pick up template-space files when the user asked
        // for subject space (or vice versa).
        if !name.contains(&space_prefix) {
            continue;
        }
        for legacy_key in bundle.legacy_keys {
            let legacy_token = format!("_desc-ROI-{}-", legacy_key);
            let Some(token_pos) = name.find(&legacy_token) else {
                continue;
            };
            let after = &name[token_pos + legacy_token.len()..];
            // Expect `<N>-<role>.nii.gz`.
            let Some(stem) = after.strip_suffix(".nii.gz") else {
                break;
            };
            let Some((idx_str, role_str)) = stem.split_once('-') else {
                break;
            };
            let Ok(idx) = idx_str.parse::<u32>() else {
                break;
            };
            match role_str {
                "include" => includes.push((idx, path.clone())),
                "exclude" => excludes.push((idx, path.clone())),
                "start" => start = Some(path.clone()),
                "end" => end = Some(path.clone()),
                _ => {}
            }
            break;
        }
    }

    includes.sort_by_key(|(i, _)| *i);
    excludes.sort_by_key(|(i, _)| *i);

    Ok(DiscoveredBundleFiles {
        includes: includes.into_iter().map(|(_, p)| p).collect(),
        excludes: excludes.into_iter().map(|(_, p)| p).collect(),
        start,
        end,
        prob_map,
    })
}

fn walk_collect(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk_collect(&path, out)?;
        } else if ft.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

/// Build a pyAFQ `TrackingPlan` for one bundle.
pub fn build_pyafq_plan(
    working_dir: &Path,
    bundle: &PyafqBundleSpec,
    label: String,
    params: &PyafqPlanParams,
) -> Result<PyafqPlanOutputs, PyafqPlanError> {
    let files = discover_bundle_files(working_dir, bundle, &params.to_space)?;

    if files.includes.is_empty() {
        return Err(PyafqPlanError::NoIncludes {
            bundle: bundle.display_name.to_string(),
            dir: working_dir.to_path_buf(),
            space: params.to_space.clone(),
            available_spaces: detect_spaces(working_dir),
        });
    }

    // Load every file we need. The first include defines the canonical grid;
    // every other mask is checked against it (pyAFQ writes them all in the
    // same subject space, so a mismatch means the user pointed at a
    // mixed-space directory).
    let first_include = load_mask(&files.includes[0])?;
    let dims = first_include.dims;
    let voxel_to_ras = first_include.voxel_to_ras;
    let min_vs = min_voxel_size_mm(&voxel_to_ras).max(1e-6);

    let waypoint_radius = (params.dist_to_waypoint_mm / min_vs).max(0.0);
    let exclusion_radius = (params.dist_to_exclusion_mm / min_vs).max(0.0);
    let endpoint_radius = (params.dist_to_endpoint_mm / min_vs).max(0.0);

    // Inclusion / waypoint masks (AND semantics in the plan).
    let mut roi_masks: Vec<Arc<VoxelMask>> = Vec::with_capacity(files.includes.len());
    let mut include_union_data = vec![0u8; first_include.data.len()];
    let dilated = dilate_mask(&first_include.data, dims, waypoint_radius);
    or_into(&mut include_union_data, &dilated);
    roi_masks.push(Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: dilated,
        ..Default::default()
    }));
    for path in &files.includes[1..] {
        let m = load_mask(path)?;
        if m.dims != dims {
            return Err(PyafqPlanError::GridMismatch(dims, m.dims));
        }
        let dilated = dilate_mask(&m.data, dims, waypoint_radius);
        or_into(&mut include_union_data, &dilated);
        roi_masks.push(Arc::new(VoxelMask {
            dims,
            voxel_to_ras,
            data: dilated,
            ..Default::default()
        }));
    }
    let include_union = Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: include_union_data,
        ..Default::default()
    });

    // Exclusion masks (OR-combined into a single roa_mask).
    let mut exclude_union_data = vec![0u8; first_include.data.len()];
    for path in &files.excludes {
        let m = load_mask(path)?;
        if m.dims != dims {
            return Err(PyafqPlanError::GridMismatch(dims, m.dims));
        }
        let dilated = dilate_mask(&m.data, dims, exclusion_radius);
        or_into(&mut exclude_union_data, &dilated);
    }
    let exclude_union = Arc::new(VoxelMask {
        dims,
        voxel_to_ras,
        data: exclude_union_data,
        ..Default::default()
    });
    let roa_mask = if files.excludes.is_empty() {
        None
    } else {
        Some(exclude_union.clone())
    };

    // Start / End endpoint masks.
    let (start_mask, has_start) = match &files.start {
        Some(p) => {
            let m = load_mask(p)?;
            if m.dims != dims {
                return Err(PyafqPlanError::GridMismatch(dims, m.dims));
            }
            let dilated = dilate_mask(&m.data, dims, endpoint_radius);
            (
                Arc::new(VoxelMask {
                    dims,
                    voxel_to_ras,
                    data: dilated,
                    ..Default::default()
                }),
                true,
            )
        }
        None => (
            Arc::new(VoxelMask {
                dims,
                voxel_to_ras,
                data: vec![0u8; first_include.data.len()],
                ..Default::default()
            }),
            false,
        ),
    };
    let (end_mask, has_end) = match &files.end {
        Some(p) => {
            let m = load_mask(p)?;
            if m.dims != dims {
                return Err(PyafqPlanError::GridMismatch(dims, m.dims));
            }
            let dilated = dilate_mask(&m.data, dims, endpoint_radius);
            (
                Arc::new(VoxelMask {
                    dims,
                    voxel_to_ras,
                    data: dilated,
                    ..Default::default()
                }),
                true,
            )
        }
        None => (
            Arc::new(VoxelMask {
                dims,
                voxel_to_ras,
                data: vec![0u8; first_include.data.len()],
                ..Default::default()
            }),
            false,
        ),
    };

    // Plan's `end_masks` carries only the masks that exist on disk.
    // pyAFQ semantics: a streamline must terminate inside *every* end mask
    // present (start AND end when both are defined).
    let mut end_masks: Vec<Arc<VoxelMask>> = Vec::new();
    if has_start {
        end_masks.push(start_mask.clone());
    }
    if has_end {
        end_masks.push(end_mask.clone());
    }

    // Probability map (binarized at any non-zero voxel — pyAFQ's
    // `prob_threshold` then filters by the *fraction* of streamline points
    // landing inside, see `streamline_passes_pyafq_prob`).
    let (prob_post_filter, has_prob_map) = match &files.prob_map {
        Some(p) => {
            let vol = NiftiVolume::load(p).map_err(|source| PyafqPlanError::Nifti {
                path: p.clone(),
                source,
            })?;
            if [vol.dims[0] as u32, vol.dims[1] as u32, vol.dims[2] as u32] != dims {
                return Err(PyafqPlanError::GridMismatch(
                    dims,
                    [vol.dims[0] as u32, vol.dims[1] as u32, vol.dims[2] as u32],
                ));
            }
            // NiftiVolume normalizes to [0,1]; any voxel > 0 marks the
            // bundle's probabilistic support region.
            let prob_data: Vec<u8> = vol
                .data
                .iter()
                .map(|&v| if v > 0.0 { 1u8 } else { 0u8 })
                .collect();
            let prob_mask = Arc::new(VoxelMask {
                dims,
                voxel_to_ras,
                data: prob_data,
                ..Default::default()
            });
            (
                Some(PostFilter::PyAFQProb {
                    prob_map: prob_mask,
                    threshold: params.prob_threshold,
                }),
                true,
            )
        }
        None => (None, false),
    };

    let min_len_mm = params.override_min_len_mm.or(bundle.min_len_mm);
    let max_len_mm = params.override_max_len_mm.or(bundle.max_len_mm);

    let plan = TrackingPlan {
        label,
        grid_dims: dims,
        voxel_to_ras,
        seed_mask: None,
        limiting_mask: None,
        roa_mask,
        term_mask: None,
        roi_masks,
        end_masks,
        no_end_mask: None,
        post_filter: prob_post_filter,
        min_len_mm,
        max_len_mm,
        max_angle_deg: None,
        step_size_mm: None,
        fixel_threshold: None,
        smooth_fraction: None,
        tolerance_mm: None,
        fixel_otsu: None,
    };

    Ok(PyafqPlanOutputs {
        plan,
        include_union,
        exclude_union,
        start_mask,
        end_mask,
        n_include: files.includes.len(),
        n_exclude: files.excludes.len(),
        has_start,
        has_end,
        has_prob_map,
        include_paths: files.includes,
        exclude_paths: files.excludes,
        start_path: files.start,
        end_path: files.end,
        prob_map_path: files.prob_map,
    })
}

/// Load a NIfTI as a binary `VoxelMask` (any non-zero voxel = inside).
fn load_mask(path: &Path) -> Result<VoxelMask, PyafqPlanError> {
    let vol = NiftiVolume::load(path).map_err(|source| PyafqPlanError::Nifti {
        path: path.to_path_buf(),
        source,
    })?;
    let dims = [vol.dims[0] as u32, vol.dims[1] as u32, vol.dims[2] as u32];
    let data: Vec<u8> = vol
        .data
        .iter()
        .map(|&v| if v > 0.0 { 1u8 } else { 0u8 })
        .collect();
    Ok(VoxelMask {
        dims,
        voxel_to_ras: vol.voxel_to_ras,
        data,
        ..Default::default()
    })
}

fn or_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Cheap directory scan: for every catalog bundle, return whether the
/// directory contains at least one matching ROI mask (modern OR legacy
/// naming, in the requested `to_space`). Lets the inspector gray out
/// bundles that aren't present in the user's working directory.
///
/// Single `read_dir` walk; quick string checks per file. Safe to call
/// every frame on a typical AFQ derivatives tree (~hundreds of files).
pub fn scan_available_bundles(
    working_dir: &Path,
    to_space: &str,
) -> std::collections::HashSet<&'static str> {
    use crate::workflow::ops::pyafq_bundles::PYAFQ_BUNDLES;

    let mut available: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    if !working_dir.exists() {
        return available;
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    if walk_collect(working_dir, &mut paths).is_err() {
        return available;
    }

    let space_prefix = format!("_space-{}_", to_space);
    for path in &paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".nii.gz") || !name.contains(&space_prefix) {
            continue;
        }
        for bundle in PYAFQ_BUNDLES {
            if available.contains(bundle.display_name) {
                continue;
            }
            let modern_token = format!(
                "_space-{}_desc-{}",
                to_space,
                str_to_desc(bundle.display_name)
            );
            if name.contains(&modern_token)
                && (name.ends_with("_mask.nii.gz") || name.contains("_probseg"))
            {
                available.insert(bundle.display_name);
                continue;
            }
            for legacy_key in bundle.legacy_keys {
                if name.contains(&format!("_desc-ROI-{}-", legacy_key)) {
                    available.insert(bundle.display_name);
                    break;
                }
            }
        }
    }
    available
}

/// Scan `dir` for any `.nii.gz` filenames containing a `_space-{X}_` token
/// and return the set of `X` values, sorted. Used to give a useful error
/// when the user's `to_space` doesn't match what's on disk.
fn detect_spaces(dir: &Path) -> Vec<String> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let _ = walk_collect(dir, &mut paths);
    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for path in paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".nii.gz") {
            continue;
        }
        let mut search = name;
        while let Some(start) = search.find("_space-") {
            let rest = &search[start + "_space-".len()..];
            let end = rest.find('_').unwrap_or(rest.len());
            let token = &rest[..end];
            if !token.is_empty() {
                found.insert(token.to_string());
            }
            search = &rest[end..];
        }
    }
    found.into_iter().collect()
}

fn min_voxel_size_mm(voxel_to_ras: &glam::Mat4) -> f32 {
    let x = voxel_to_ras.col(0).truncate().length();
    let y = voxel_to_ras.col(1).truncate().length();
    let z = voxel_to_ras.col(2).truncate().length();
    x.min(y).min(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::ops::pyafq_bundles::lookup;
    use std::fs::{self, File};

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        File::create(&p).unwrap();
        p
    }

    #[test]
    fn discovers_modern_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let roi_dir = tmp.path().join("ROIs");
        fs::create_dir_all(&roi_dir).unwrap();
        let base = "sub-X_ses-01_space-subject_desc-LeftCorticospinal";
        touch(&roi_dir, &format!("{base}Include0_mask.nii.gz"));
        touch(&roi_dir, &format!("{base}Include1_mask.nii.gz"));
        touch(&roi_dir, &format!("{base}End_mask.nii.gz"));
        touch(&roi_dir, &format!("{base}_probseg.nii.gz"));

        let bundle = lookup("Left Corticospinal").unwrap();
        let files = discover_bundle_files(tmp.path(), bundle, "subject").unwrap();
        assert_eq!(files.includes.len(), 2);
        assert!(files.start.is_none());
        assert!(files.end.is_some());
        assert!(files.prob_map.is_some());
    }

    #[test]
    fn discovers_legacy_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let roi_dir = tmp.path().join("ROIs");
        fs::create_dir_all(&roi_dir).unwrap();
        let base = "sub-X_ses-01_acq-64dir_space-T1w_desc-preproc_dwi_desc-ROI-CST_L";
        touch(&roi_dir, &format!("{base}-1-include.nii.gz"));
        touch(&roi_dir, &format!("{base}-2-include.nii.gz"));

        let bundle = lookup("Left Corticospinal").unwrap();
        let files = discover_bundle_files(tmp.path(), bundle, "T1w").unwrap();
        assert_eq!(files.includes.len(), 2);
        assert_eq!(files.excludes.len(), 0);
    }

    #[test]
    fn legacy_slf_with_exclude() {
        let tmp = tempfile::tempdir().unwrap();
        let roi_dir = tmp.path().join("ROIs");
        fs::create_dir_all(&roi_dir).unwrap();
        let base = "sub-X_ses-01_acq-64dir_space-T1w_desc-preproc_dwi_desc-ROI-SLF_L";
        touch(&roi_dir, &format!("{base}-1-include.nii.gz"));
        touch(&roi_dir, &format!("{base}-2-include.nii.gz"));
        touch(&roi_dir, &format!("{base}-3-exclude.nii.gz"));

        let bundle = lookup("Left SLF").unwrap();
        let files = discover_bundle_files(tmp.path(), bundle, "T1w").unwrap();
        assert_eq!(files.includes.len(), 2);
        assert_eq!(files.excludes.len(), 1);
    }

    /// Smoke-test against the real HBN POD2 derivatives at
    /// `../test_data/HBN_AFQ/`. Skipped by default (`--ignored`) since it
    /// requires local data; run manually after `aws s3 sync`.
    #[test]
    #[ignore]
    fn smoke_hbn_pod2_legacy() {
        // Resolve relative to this crate's manifest dir so the test works
        // regardless of where `cargo test` is invoked from.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = manifest.join("../../test_data/HBN_AFQ");
        if !dir.exists() {
            eprintln!("skipping: {dir:?} not present");
            return;
        }
        for name in [
            "Left Corticospinal",
            "Left Arcuate",
            "Left SLF",
            "Callosum Motor",
        ] {
            let b = lookup(name).expect("bundle in catalog");
            let files = discover_bundle_files(&dir, b, "T1w").expect("discovery");
            eprintln!(
                "{name}: {} include / {} exclude / start={} / end={} / probmap={}",
                files.includes.len(),
                files.excludes.len(),
                files.start.is_some(),
                files.end.is_some(),
                files.prob_map.is_some(),
            );
            assert!(
                !files.includes.is_empty(),
                "{name}: expected ≥1 include from HBN data",
            );
        }
    }

    #[test]
    fn space_mismatch_finds_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let roi_dir = tmp.path().join("ROIs");
        fs::create_dir_all(&roi_dir).unwrap();
        let base = "sub-X_ses-01_acq-64dir_space-T1w_desc-preproc_dwi_desc-ROI-CST_L";
        touch(&roi_dir, &format!("{base}-1-include.nii.gz"));

        let bundle = lookup("Left Corticospinal").unwrap();
        let files = discover_bundle_files(tmp.path(), bundle, "subject").unwrap();
        assert!(files.includes.is_empty());
    }
}

//! Static catalog of pyAFQ canonical bundles.
//!
//! Mirrors `default_bd()` (Optic Radiation + 18 default tracts),
//! `callosal_bd()` (8 callosal subdivisions), and `baby_bd()` (pediatric
//! bundles — adds Middle Longitudinal, Superior Longitudinal as a distinct
//! tract from Arcuate, and Forceps Major / Minor) from
//! `pyAFQ/AFQ/api/bundle_dict.py`. Used by the `Prepare pyAFQ Plan` op to
//! populate the bundle dropdown and to know per-bundle defaults that aren't
//! recoverable from the on-disk NIfTIs alone (length thresholds, which roles
//! to expect).
//!
//! Recobundles atlases (`reco_bd(16/80)`) are intentionally omitted — they
//! are tractogram-driven, not waypoint-ROI-driven, and don't fit this op's
//! shape.
//!
//! When pyAFQ writes its warped subject-space ROIs, the BIDS descriptor
//! fragment for a bundle is `str_to_desc(display_name)` — i.e. spaces, `-`,
//! and `_` stripped. See `pyAFQ/AFQ/tasks/utils.py:110`. We compute the same
//! transform at runtime when globbing.

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PyafqCategory {
    Default,
    Callosal,
    /// `baby_bd()` — pediatric bundles. Same display names as adult tracts
    /// for shared bundles, plus Middle Longitudinal, Superior Longitudinal
    /// (distinct from Arcuate), and Forceps Major / Minor.
    Pediatric,
}

#[derive(Clone, Copy, Debug)]
pub struct PyafqBundleSpec {
    pub display_name: &'static str,
    pub category: PyafqCategory,
    /// Bundle's `length.min_len` from `bundle_dict.py`. `None` when pyAFQ
    /// doesn't set one explicitly (we fall back to the global default).
    pub min_len_mm: Option<f32>,
    /// pyAFQ rarely sets `max_len`; included for completeness.
    pub max_len_mm: Option<f32>,
    pub has_start: bool,
    pub has_end: bool,
    pub has_prob_map: bool,
    /// Older pyAFQ (≤2022, e.g. HBN POD2 derivatives) used short bundle
    /// abbreviations and a different filename layout:
    /// `*_desc-ROI-{legacy_key}-{N}-{role}.nii.gz` instead of the modern
    /// `*_desc-{display_desc}{Role}{N}_mask.nii.gz`. List every legacy key
    /// that maps to this display name so the discoverer can read both
    /// layouts without forcing users to re-run pyAFQ.
    pub legacy_keys: &'static [&'static str],
}

/// pyAFQ's `segmentation_params["dist_to_waypoint"]` default. Used when the
/// op's slider is left at the canonical value.
pub const DEFAULT_DIST_TO_WAYPOINT_MM: f32 = 1.5;
/// pyAFQ's `segmentation_params["dist_to_atlas"]` default for endpoint masks.
pub const DEFAULT_DIST_TO_ENDPOINT_MM: f32 = 4.0;

/// Strip spaces, `-`, and `_`. Mirrors pyAFQ's `str_to_desc`
/// (`pyAFQ/AFQ/tasks/utils.py:110`).
pub fn str_to_desc(s: &str) -> String {
    s.chars()
        .filter(|c| *c != ' ' && *c != '-' && *c != '_')
        .collect()
}

pub const PYAFQ_BUNDLES: &[PyafqBundleSpec] = &[
    // OR_bd()
    PyafqBundleSpec {
        display_name: "Left Optic Radiation",
        category: PyafqCategory::Default,
        min_len_mm: None,
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Right Optic Radiation",
        category: PyafqCategory::Default,
        min_len_mm: None,
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    // default_bd() — anterior thalamic, cingulum, CST, IFOF, ILF, arcuate, uncinate, pARC, VOF
    PyafqBundleSpec {
        display_name: "Left Anterior Thalamic",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ATR_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Anterior Thalamic",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ATR_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Cingulum Cingulate",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: false,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["CGC_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Cingulum Cingulate",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: false,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["CGC_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Corticospinal",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: false,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["CST_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Corticospinal",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: false,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["CST_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Inferior Fronto-occipital",
        category: PyafqCategory::Default,
        min_len_mm: Some(80.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["IFO_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Inferior Fronto-occipital",
        category: PyafqCategory::Default,
        min_len_mm: Some(80.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["IFO_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Inferior Longitudinal",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ILF_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Inferior Longitudinal",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ILF_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Arcuate",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ARC_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Arcuate",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["ARC_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Uncinate",
        category: PyafqCategory::Default,
        min_len_mm: None,
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["UNC_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Uncinate",
        category: PyafqCategory::Default,
        min_len_mm: None,
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: true,
        legacy_keys: &["UNC_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Posterior Arcuate",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Right Posterior Arcuate",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Left Vertical Occipital",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Right Vertical Occipital",
        category: PyafqCategory::Default,
        min_len_mm: Some(30.0),
        max_len_mm: None,
        has_start: true,
        has_end: true,
        has_prob_map: false,
        legacy_keys: &[],
    },
    // Legacy-only: Superior Longitudinal Fasciculus. Modern pyAFQ folds
    // SLF into Arcuate; older AFQ tracked them separately.
    PyafqBundleSpec {
        display_name: "Left SLF",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["SLF_L"],
    },
    PyafqBundleSpec {
        display_name: "Right SLF",
        category: PyafqCategory::Default,
        min_len_mm: Some(40.0),
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["SLF_R"],
    },
    // callosal_bd() — three-waypoint includes (no start/end, no prob_map).
    // Legacy AFQ used the same short keys (no `Callosum` prefix) for these.
    PyafqBundleSpec {
        display_name: "Callosum Anterior Frontal",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["AntFrontal"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Motor",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["Motor"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Occipital",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["Occipital"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Orbital",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["Orbital"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Posterior Parietal",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["PostParietal"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Superior Frontal",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["SupFrontal"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Superior Parietal",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["SupParietal"],
    },
    PyafqBundleSpec {
        display_name: "Callosum Temporal",
        category: PyafqCategory::Callosal,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: false,
        legacy_keys: &["Temporal"],
    },
    // baby_bd() — pediatric-only bundles. Adult-shared pediatric tracts
    // (ATR, CGC, CST, IFO, ILF, ARC, UNC, OR, pARC, VOF) are already covered
    // above via their default-category entries — pediatric rewrites them with
    // a third include ROI, but the on-disk file naming follows the same
    // display name → `str_to_desc` rule, so the discoverer finds them with
    // the existing Default specs. Only bundles unique to pediatric AFQ get
    // their own entries here.
    PyafqBundleSpec {
        display_name: "Left Middle Longitudinal",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &["MdLF_L"],
    },
    PyafqBundleSpec {
        display_name: "Right Middle Longitudinal",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &["MdLF_R"],
    },
    PyafqBundleSpec {
        display_name: "Left Superior Longitudinal",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Right Superior Longitudinal",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &[],
    },
    PyafqBundleSpec {
        display_name: "Forceps Minor",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &["FA"],
    },
    PyafqBundleSpec {
        display_name: "Forceps Major",
        category: PyafqCategory::Pediatric,
        min_len_mm: None,
        max_len_mm: None,
        has_start: false,
        has_end: false,
        has_prob_map: true,
        legacy_keys: &["FP"],
    },
];

pub fn lookup(display_name: &str) -> Option<&'static PyafqBundleSpec> {
    PYAFQ_BUNDLES.iter().find(|b| b.display_name == display_name)
}

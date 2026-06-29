use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::WorkflowResult;

use super::graph::{GraphRect, WorkflowGraph};
use super::ops::WorkflowNodeKind;
use crate::data::bundle_mesh::BundleMesh;
use crate::data::cifti::{CiftiIntent, CiftiStructure, SurfaceScalars, VolumeScalars};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{FileId, StreamlineBacking, VolumeColormap};
use crate::data::orientation_field::{
    BoundaryContactField, BoundaryGlyphColorMode, BoundaryGlyphNormalization,
};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::lighting::WorkflowRender3D;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::units::{Millimeters, ParcelId, StreamlineIndex};
use trx_rs::DuplicateRemovalParams;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct WorkflowNodeUuid(pub u64);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowNode {
    pub uuid: WorkflowNodeUuid,
    pub op: WorkflowNodeKind,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GroupFilter {
    All,
    None,
    Selected(BTreeSet<String>),
}

impl Default for GroupFilter {
    fn default() -> Self {
        Self::All
    }
}

impl GroupFilter {
    pub fn from_csv(csv: &str) -> Self {
        if csv.trim() == "__none__" {
            return Self::None;
        }
        let labels = csv
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<BTreeSet<_>>();
        if labels.is_empty() {
            Self::All
        } else {
            Self::Selected(labels)
        }
    }

    pub fn to_csv(&self) -> String {
        match self {
            Self::All => String::new(),
            Self::None => "__none__".to_string(),
            Self::Selected(labels) => labels.iter().cloned().collect::<Vec<_>>().join(", "),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ParcelIdSet(pub BTreeSet<ParcelId>);

impl ParcelIdSet {
    pub fn from_csv(csv: &str) -> Self {
        Self(
            csv.split(',')
                .map(str::trim)
                .filter_map(|value| value.parse::<u32>().ok().map(ParcelId))
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn to_csv(&self) -> String {
        self.0
            .iter()
            .map(|label| label.0.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct DpvFieldName(pub String);

impl DpvFieldName {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl AsRef<str> for DpvFieldName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for DpvFieldName {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for DpvFieldName {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct DpsFieldName(pub String);

impl DpsFieldName {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl AsRef<str> for DpsFieldName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for DpsFieldName {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for DpsFieldName {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlyphColormap {
    #[default]
    Directional,
    Plasma,
    Viridis,
    Inferno,
    BlueWhiteRed,
}

/// Piecewise opacity curve driven by a voxel-scalar input.
/// Below `range.0` → `below`, above `range.1` → `above`, linear in between.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpacityGate {
    pub range: (f32, f32),
    pub below: f32,
    pub above: f32,
}

impl Default for OpacityGate {
    fn default() -> Self {
        Self {
            range: (0.0, 1.0),
            below: 0.0,
            above: 1.0,
        }
    }
}

/// Piecewise size-scaling curve driven by a voxel-scalar input.
/// Below `range.0` → `min_scale`, above `range.1` → `max_scale`, linear in between.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SizeGate {
    pub range: (f32, f32),
    pub min_scale: f32,
    pub max_scale: f32,
}

impl Default for SizeGate {
    fn default() -> Self {
        Self {
            range: (0.0, 1.0),
            min_scale: 0.5,
            max_scale: 1.5,
        }
    }
}

pub fn default_fixel_colormap() -> SurfaceColormap {
    SurfaceColormap::Inferno
}

pub fn default_fixel_line_width() -> f32 {
    0.006
}

pub fn default_fixel_length_scale() -> f32 {
    1.0
}

pub fn default_full_opacity() -> f32 {
    1.0
}

pub fn default_fundus_tube_radius() -> f32 {
    0.4
}

pub fn default_fixel_slab_thickness_mm() -> Millimeters {
    Millimeters(1.0)
}

pub fn default_odf_glyph_scale() -> f32 {
    3.25
}

pub fn default_odf_glyph_detail() -> u32 {
    3
}

pub fn default_true() -> bool {
    true
}

pub fn default_false() -> bool {
    false
}

pub fn default_workflow_slice_view_kind() -> WorkflowSliceViewKind {
    WorkflowSliceViewKind::Axial
}

pub fn default_enabled() -> bool {
    true
}

pub fn default_boundary_field_voxel_size_mm() -> Millimeters {
    Millimeters(3.0)
}

pub fn default_boundary_field_sphere_lod() -> u32 {
    12
}

pub fn default_boundary_field_normalization() -> BoundaryGlyphNormalization {
    BoundaryGlyphNormalization::GlobalPeak
}

pub fn default_boundary_glyph_scale() -> f32 {
    2.0
}

pub fn default_boundary_glyph_density_3d_step() -> usize {
    2
}

pub fn default_boundary_glyph_slice_density_step() -> usize {
    1
}

pub fn default_boundary_glyph_color_mode() -> BoundaryGlyphColorMode {
    BoundaryGlyphColorMode::DirectionRgb
}

pub fn default_boundary_glyph_min_contacts() -> u32 {
    1
}

pub fn default_bundle_surface_outline_thickness() -> f32 {
    1.15
}

pub fn default_bundle_surface_min_component_volume_mm3() -> Millimeters {
    Millimeters(0.0)
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub enum SurfaceDisplaySpace {
    #[default]
    Anatomical,
    Stage,
}

/// Serialize an f32 threshold: finite values as numbers, ±infinity as `null`.
/// `null` is also accepted on deserialization and maps back to ±infinity
/// (sign is determined by the default for each field).
fn serialize_f32_inf_as_null<S: serde::Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
    if v.is_finite() {
        s.serialize_f32(*v)
    } else {
        s.serialize_none()
    }
}

fn deserialize_threshold_min<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let opt: Option<f32> = serde::Deserialize::deserialize(d)?;
    Ok(opt.unwrap_or(f32::NEG_INFINITY))
}

fn deserialize_threshold_max<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let opt: Option<f32> = serde::Deserialize::deserialize(d)?;
    Ok(opt.unwrap_or(f32::INFINITY))
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceOverlayLayerConfig {
    pub enabled: bool,
    pub solid_color: [f32; 4],
    pub opacity: f32,
    pub colormap: SurfaceColormap,
    pub range_min: f32,
    pub range_max: f32,
    #[serde(
        serialize_with = "serialize_f32_inf_as_null",
        deserialize_with = "deserialize_threshold_min"
    )]
    pub threshold_min: f32,
    #[serde(
        serialize_with = "serialize_f32_inf_as_null",
        deserialize_with = "deserialize_threshold_max"
    )]
    pub threshold_max: f32,
    pub use_label_colors: bool,
    pub legend_label: String,
}

pub fn default_surface_overlay_layers() -> Vec<SurfaceOverlayLayerConfig> {
    let mut layers = Vec::with_capacity(5);
    layers.push(SurfaceOverlayLayerConfig {
        enabled: true,
        solid_color: DEFAULT_SURFACE_BASE_RGBA,
        opacity: 1.0,
        colormap: SurfaceColormap::Inferno,
        range_min: 0.0,
        range_max: 1.0,
        threshold_min: f32::NEG_INFINITY,
        threshold_max: f32::INFINITY,
        use_label_colors: false,
        legend_label: "Base".to_string(),
    });
    for index in 1..5 {
        layers.push(SurfaceOverlayLayerConfig {
            enabled: false,
            solid_color: DEFAULT_SURFACE_BASE_RGBA,
            opacity: 1.0,
            colormap: SurfaceColormap::Inferno,
            range_min: 0.0,
            range_max: 1.0,
            threshold_min: f32::NEG_INFINITY,
            threshold_max: f32::INFINITY,
            use_label_colors: true,
            legend_label: format!("Overlay {index}"),
        });
    }
    layers
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkflowDocument {
    #[serde(default = "default_next_node_uuid")]
    pub next_node_uuid: u64,
    pub graph: WorkflowGraph,
    pub assets: Vec<WorkflowAssetDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_3d: Option<WorkflowCamera3D>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_3d: Option<WorkflowRender3D>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_view_3d: Option<WorkflowSliceView3D>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_view_ui: Option<WorkflowSliceViewUi>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<WorkflowSelection>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkflowProject {
    pub version: u32,
    pub document: WorkflowDocument,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WorkflowAssetDocument {
    Streamlines {
        id: FileId,
        path: PathBuf,
        imported: bool,
    },
    Volume {
        id: FileId,
        path: PathBuf,
    },
    Cifti {
        id: FileId,
        path: PathBuf,
        intent: CiftiIntent,
    },
    Surface {
        id: FileId,
        path: PathBuf,
    },
    Parcellation {
        id: FileId,
        path: PathBuf,
        label_table_path: Option<PathBuf>,
    },
    Odx {
        id: FileId,
        path: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowCamera3D {
    pub target: [f32; 3],
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowSliceView3D {
    pub visible: [bool; 3],
    /// World-space slice positions in RAS mm: [axial_z, coronal_y, sagittal_x].
    pub positions_ras: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowSliceViewKind {
    Axial,
    Coronal,
    Sagittal,
}

impl WorkflowSliceViewKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
        }
    }

    pub fn viewport_index(self) -> usize {
        match self {
            Self::Axial => 0,
            Self::Coronal => 1,
            Self::Sagittal => 2,
        }
    }

    pub fn odx_axis(self) -> usize {
        match self {
            Self::Axial => 2,
            Self::Coronal => 1,
            Self::Sagittal => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowView2DMode {
    Slice,
    Ortho,
    Lightbox,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowOrthoSliceCamera {
    pub center: [f32; 2],
    pub half_extent: f32,
    pub rotation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowSliceViewUi {
    pub mode: WorkflowView2DMode,
    pub single_view: WorkflowSliceViewKind,
    pub lightbox_axis: WorkflowSliceViewKind,
    pub lightbox_rows: usize,
    pub lightbox_cols: usize,
    pub active_axis: usize,
    pub ortho_show_row: bool,
    pub slice_cameras: [WorkflowOrthoSliceCamera; 3],
}

#[derive(Clone, Debug, Default)]
pub struct NodeEvalState {
    pub summary: String,
    pub error: Option<String>,
    pub execution: Option<WorkflowExecutionStatus>,
    pub fingerprint: Option<u64>,
    pub last_result_summary: Option<String>,
    pub available_streamline_groups: Vec<String>,
    /// DPS / DPV field names present on the streamline output of this
    /// node's last evaluation. The inspector uses these to populate
    /// comboboxes in `ColorByDps` / `ColorByDpv` so the user can pick
    /// from the actual available fields rather than typing the name.
    pub available_dps_fields: Vec<String>,
    pub available_dpv_fields: Vec<String>,
    /// Names of op params whose value was overridden by a `TrackingPlan` on
    /// the last evaluation. The UI uses this to grey out the corresponding
    /// sliders and advertise that the plan is winning.
    pub overridden_fields: Vec<String>,
    /// For each overridden numeric field, the effective value. The UI binds
    /// this to the greyed-out slider so the user sees the plan's value in
    /// place of the op's own. Non-numeric overrides (e.g. `seed_mask`) only
    /// appear in `overridden_fields`.
    pub overridden_values: std::collections::BTreeMap<String, f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowEvalMode {
    Interactive,
    Settled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowExecutionStatus {
    NeverRun,
    Stale,
    Queued,
    Running,
    Ready,
    Failed(String),
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BundleSurfaceColorMode {
    #[default]
    Solid,
    BoundaryField,
    SourceColors,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BundleSurfaceBuildMode {
    #[default]
    MarchingCubes,
    Streamtubes,
}

impl BundleSurfaceColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Solid => "Solid",
            // Display label only — the enum variant is still named
            // `BoundaryField` internally so the data type
            // `BoundaryContactField` and `boundary_field_cache` keep
            // their names. The user-facing language is "direction
            // field" because the producer (StreamlineDirectionField)
            // and one of the consumers (Purifibre) aren't
            // boundary-glyph-specific.
            Self::BoundaryField => "Direction field",
            Self::SourceColors => "Source colors",
        }
    }
}

impl BundleSurfaceBuildMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MarchingCubes => "Marching Cubes",
            Self::Streamtubes => "Streamtubes",
        }
    }
}

impl WorkflowExecutionStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::NeverRun => "Run required",
            Self::Stale => "Stale",
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Ready => "Ready",
            Self::Failed(_) => "Failed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExpensiveNodeRunRecord {
    pub current_fingerprint: Option<u64>,
    pub last_success_fingerprint: Option<u64>,
    /// Fingerprint at which the user most recently cancelled this node.
    /// The GUI's `should_queue_expensive_job` treats a match here the
    /// same as a `last_success_fingerprint` match: don't auto-re-run.
    /// Cleared when the user clicks a Run button (so a second click
    /// retries at the same fingerprint) or when the fingerprint
    /// changes (because it won't match anymore).
    pub last_cancelled_fingerprint: Option<u64>,
    pub status: WorkflowExecutionStatus,
    pub last_result_summary: Option<String>,
}

impl Default for ExpensiveNodeRunRecord {
    fn default() -> Self {
        Self {
            current_fingerprint: None,
            last_success_fingerprint: None,
            last_cancelled_fingerprint: None,
            status: WorkflowExecutionStatus::NeverRun,
            last_result_summary: None,
        }
    }
}

#[derive(Clone)]
pub struct CachedSurfaceQuery {
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub struct CachedDerivedStreamline {
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub struct CachedSurfaceStreamlineMap {
    pub map: SurfaceScalars,
}

#[derive(Clone)]
pub struct CachedBoundaryField {
    pub fingerprint: u64,
    pub field: Arc<BoundaryContactField>,
}

#[derive(Clone)]
pub struct CachedTubeGeometry {
    pub fingerprint: u64,
    pub vertices: Vec<crate::data::trx_data::TubeMeshVertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone)]
pub struct CachedBundleSurfaceMeshes {
    pub fingerprint: u64,
    pub meshes: Vec<(BundleMesh, String)>,
}

#[derive(Clone)]
pub struct OdxDpvMaterialization {
    pub source_id: FileId,
    pub dpv_name: String,
    pub volume: Arc<crate::data::nifti_data::NiftiVolume>,
}

#[derive(Clone, Default)]
pub struct WorkflowExecutionCache {
    pub node_runs: HashMap<WorkflowNodeUuid, ExpensiveNodeRunRecord>,
    pub derived_streamline_cache: HashMap<WorkflowNodeUuid, CachedDerivedStreamline>,
    pub surface_query_cache: HashMap<WorkflowNodeUuid, CachedSurfaceQuery>,
    pub surface_streamline_map_cache: HashMap<WorkflowNodeUuid, CachedSurfaceStreamlineMap>,
    pub tube_geometry_cache: HashMap<WorkflowNodeUuid, CachedTubeGeometry>,
    pub bundle_surface_mesh_cache: HashMap<WorkflowNodeUuid, CachedBundleSurfaceMeshes>,
    pub boundary_field_cache: HashMap<WorkflowNodeUuid, CachedBoundaryField>,
    pub odx_dpv_materializations: HashMap<WorkflowNodeUuid, OdxDpvMaterialization>,
    pub dipy_tractography_results: HashMap<WorkflowNodeUuid, CachedTractographyResult>,
    pub yeh_tractography_results: HashMap<WorkflowNodeUuid, CachedTractographyResult>,
    pub voxel_mask_mesh_cache: HashMap<WorkflowNodeUuid, CachedVoxelMaskMesh>,
    pub hausdorff_plan_cache: HashMap<WorkflowNodeUuid, CachedHausdorffPlan>,
    pub pyafq_plan_cache: HashMap<WorkflowNodeUuid, CachedPyafqPlan>,
    pub tip_prune_cache: HashMap<WorkflowNodeUuid, CachedTipPrune>,
    pub purifibre_cache: HashMap<WorkflowNodeUuid, CachedPurifibre>,
    /// `StreamlineSourceOp` evaluates on every refresh; without
    /// caching it would `Arc::new` a fresh `StreamlineDataset`
    /// each time and downstream `Arc::as_ptr`-based fingerprints
    /// (see `hash_flow`) would rebuild caches on every eval.
    /// Reusing the cached Arc when the underlying `gpu_data`
    /// pointer is unchanged keeps fingerprints stable.
    pub streamline_source_datasets: HashMap<WorkflowNodeUuid, Arc<StreamlineDataset>>,
    /// Self-displaying `TriangleFundusOp` derives a *new*
    /// `StreamlineDataset` from its input each eval. Like
    /// `streamline_source_datasets`, a fresh `Arc::new` per frame
    /// would churn the `Arc::as_ptr`-keyed draw fingerprint and
    /// re-run the (expensive) tube-geometry job every frame — the
    /// cylinder GUI lockup. Cache the built dataset keyed by a
    /// content fingerprint (input flow + params) so the Arc — and
    /// thus the draw fingerprint — is stable until something
    /// genuinely changes.
    pub triangle_fundus_datasets:
        HashMap<WorkflowNodeUuid, (u64, Arc<StreamlineDataset>)>,
    /// `SampleVolumeAlongStreamline` derives a new `StreamlineDataset`
    /// each eval (clones `TrxGpuData`, trilinearly samples every vertex,
    /// writes a DPS field). A fresh `Arc::new` per frame would churn the
    /// `Arc::as_ptr`-keyed downstream `hash_flow` and invalidate
    /// tube-geometry / bundle-surface caches every render. Cache the
    /// built dataset keyed by a content fingerprint (input flow +
    /// dps_name + volume identity) so the Arc — and thus downstream
    /// fingerprints — stays stable until something genuinely changes.
    pub sample_volume_along_streamline_cache:
        HashMap<WorkflowNodeUuid, (u64, Arc<StreamlineDataset>)>,
    /// Memoized `VolumeScalars` view of each loaded NIfTI, keyed by
    /// `FileId`. Building a `VolumeScalars` from a `NiftiVolume`
    /// requires scanning every voxel for min/max and cloning the entire
    /// voxel buffer (see `volume_scalars_from_nifti_volume`). Without
    /// this cache, every Volume-consuming op (`OdfGlyphRenderer`,
    /// `SampleVolumeAlongStreamline`, `RoiFromVolume`,
    /// `VolumeOverlayStack`) repeated that work on every workflow
    /// evaluation — i.e. every frame during an Interactive slider
    /// drag — which made the editor visibly slow once a NIfTI was
    /// wired into any of those ops.
    pub volume_scalars_cache: HashMap<FileId, Arc<VolumeScalars>>,
}

#[derive(Clone)]
pub struct CachedHausdorffPlan {
    pub fingerprint: u64,
    pub plan: Arc<TrackingPlan>,
    pub seed_mask: Arc<VoxelMask>,
    pub limiting_mask: Arc<VoxelMask>,
    pub no_end_mask: Arc<VoxelMask>,
    pub summary: String,
}

/// pyAFQ plan cache entry. Mirrors `CachedHausdorffPlan` but exposes
/// per-role union masks (`include`, `exclude`, `start`, `end`) for the
/// VoxelMaskDisplay outputs. The plan's `roi_masks` field still carries
/// the *individual* dilated waypoints — the union mask here is for
/// visualization only.
#[derive(Clone)]
pub struct CachedPyafqPlan {
    pub fingerprint: u64,
    pub plan: Arc<TrackingPlan>,
    pub include_mask: Arc<VoxelMask>,
    pub exclude_mask: Arc<VoxelMask>,
    pub start_mask: Arc<VoxelMask>,
    pub end_mask: Arc<VoxelMask>,
    /// Continuous probability map output from `*_probseg.nii.gz`. None
    /// when the bundle lacks a prob map. Surfaces on the
    /// `PreparePyafqPlan` op's `VolumeScalars` output port.
    pub prob_map: Option<Arc<VolumeScalars>>,
    pub summary: String,
}

#[derive(Clone)]
pub struct CachedTipPrune {
    pub fingerprint: u64,
    pub selected: Vec<crate::units::StreamlineIndex>,
    pub summary: String,
}

/// Purifibre result cache entry.
///
/// The op produces *two* outputs — the scored passthrough (all input
/// streamlines, with FICO attached) and the filtered survivors — but
/// both share the same underlying `StreamlineDataset` (differing only
/// in the `selected_streamlines` list).
///
/// Cache shape splits "scoring" from "thresholding" so dragging the
/// `puri_fraction` slider doesn't trigger a full re-score:
///
/// - `score_fingerprint` covers params and inputs that affect the
///   per-streamline FICO computation (trim_fraction,
///   spherical_smoothing_deg, upstream streamlines + boundary field).
/// - `filter_fingerprint` adds `puri_fraction` on top.
///
/// On evaluation:
///   - score_fingerprint matches → reuse `scored_dataset` + raw
///     `fico_scores`; just re-threshold to get a new
///     `filtered_selection`. Cheap (sort + filter).
///   - score_fingerprint mismatches → re-score from scratch.
#[derive(Clone)]
pub struct CachedPurifibre {
    pub score_fingerprint: u64,
    pub filter_fingerprint: u64,
    /// Input streamlines dataset with the `"fico"` DPS field attached.
    /// Shared between the scored-passthrough and filtered outputs.
    pub scored_dataset: Arc<StreamlineDataset>,
    /// Selection for output 0 (scored passthrough) — equals the input
    /// `selected_streamlines`, unmodified.
    pub scored_selection: Vec<crate::units::StreamlineIndex>,
    /// Per-streamline FICO scores (length = `nb_streamlines`, NaN for
    /// unscored). Kept around so re-thresholding doesn't need to
    /// re-walk segments.
    pub fico_scores: Vec<f32>,
    /// Selection for output 1 (filtered) — streamlines that survived
    /// the puri_fraction cutoff.
    pub filtered_selection: Vec<crate::units::StreamlineIndex>,
    pub summary: String,
}

#[derive(Clone)]
pub struct CachedTractographyResult {
    pub fingerprint: u64,
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub struct SceneFramePlan {
    pub reactive_streamline_plans: Vec<ReactiveStreamlinePlan>,
    pub surface_query_plans: Vec<SurfaceQueryPlan>,
    pub surface_map_plans: Vec<SurfaceMapPlan>,
    pub streamline_draws: Vec<StreamlineDrawPlan>,
    // volume_draws now live in `draws` (the DrawPrimitive registry);
    // multiple VolumeDisplay draws are folded into one Composite.
    // Surface draws (Anatomical + Stage) now live in `draws`, discriminated
    // by SurfaceDrawPlan.space; query via SceneFramePlan::surface_draws.
    pub bundle_surface_plans: Vec<BundleSurfacePlan>,
    // BundleDrawPlan now lives in `draws` (the DrawPrimitive registry).
    pub boundary_field_plans: Vec<BoundaryFieldPlan>,
    /// Node UUIDs of `StreamlineDirectionField` (or other BoundaryField-
    /// producing) nodes whose cached field is being *consumed* by a
    /// downstream op for its own computation (not just for rendering).
    /// The GUI's post-render cache-pruning sweep keeps
    /// `boundary_field_cache` entries for any UUID in this set in
    /// addition to those referenced by `bundle_draws` /
    /// `boundary_glyph_draws`. Populated by inline ops like Purifibre
    /// that hold a `BoundaryField` input but don't emit a draw plan —
    /// without this, the retain() sweep would silently drop their
    /// upstream field and leave them forever stale.
    pub boundary_fields_in_use: std::collections::HashSet<WorkflowNodeUuid>,
    /// Generic draw-primitive registry (see [`super::draw`]). Fixel
    /// draws (3D and on-slice) live here instead of bespoke
    /// `fixel_*_draws` fields; other draw kinds are migrating over one at
    /// a time. Display ops push during evaluation; the render backends
    /// read back via `draws.of_type::<T>()`.
    pub draws: super::draw::DrawList,
    pub dipy_tractography_plans: Vec<DipyTractographyPlan>,
    pub yeh_tractography_plans: Vec<YehTractographyPlan>,
    // VoxelMaskMeshDrawPlan now lives in `draws` (the DrawPrimitive
    // registry) rather than a bespoke field.
    pub hausdorff_plan_jobs: Vec<HausdorffPlanJob>,
    pub pyafq_plan_jobs: Vec<PyafqPlanJob>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum VoxelMaskRenderStyle {
    /// Render every "on" voxel as a literal cube face. No smoothing, no
    /// marching cubes — what you see is exactly the mask data on disk.
    /// Default for new nodes; existing serialized workflows that omit the
    /// `style` field upgrade to this on load.
    #[default]
    VoxelAccurate,
    /// Gaussian-smoothed iso-surface via marching cubes plus Taubin
    /// smoothing. Stylized look originally designed for outlining messy
    /// streamline density volumes.
    SmoothMesh,
}

/// 2D slice-overlay rendering mode for voxel-accurate masks. Doesn't
/// affect the 3D mesh, so it isn't part of the mesh fingerprint.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum VoxelMaskSliceMode {
    /// Fill each voxel cell that the slice plane intersects.
    Filled,
    /// Outline only the perimeter of the masked region in the slice
    /// (edges between an on-voxel cell and an off-voxel cell are drawn;
    /// edges between two on-voxel cells are skipped). Composes cleanly
    /// when multiple masks overlap.
    #[default]
    Outline,
}

#[derive(Clone)]
pub struct VoxelMaskMeshDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub draw_id: FileId,
    pub label: String,
    pub fingerprint: u64,
    pub color: [f32; 4],
    pub opacity: f32,
    pub style: VoxelMaskRenderStyle,
    /// 2D-only — render mode for the slice overlay when `style ==
    /// VoxelAccurate`. Does not affect the 3D mesh.
    pub slice_mode: VoxelMaskSliceMode,
    /// The raw mask, kept on the draw plan so the 2D overlay path can
    /// look up dims / `voxel_to_ras` / data without a separate cache
    /// lookup keyed off node UUID. Cheap to clone (it's an `Arc`).
    pub voxel_mask: Arc<VoxelMask>,
}

#[derive(Clone)]
pub struct CachedVoxelMaskMesh {
    pub fingerprint: u64,
    pub mesh: crate::data::bundle_mesh::BundleMesh,
    pub draw_id: FileId,
}

/// Which viewport a [`FixelDrawPlan`] targets. Distinguishes the two
/// fixel display ops now that both share the registry instead of
/// separate `fixel_3d_draws` / `fixel_2d_draws` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixelView {
    /// Full 3D fixel arrows.
    ThreeD,
    /// Fixels clipped to a slice slab (2D overlay).
    TwoD,
}

#[derive(Clone)]
pub struct FixelDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    /// Viewport this draw targets; selects which GPU fixel resources the
    /// backend uploads into (3D vs on-slice).
    pub view: FixelView,
    pub field: crate::data::odx_data::FixelField,
    pub line_width: f32,
    pub length_scale: f32,
    pub opacity: f32,
    pub offset_from_slice: f32,
    pub slab_thickness_mm: Millimeters,
    pub visible: bool,
    pub colormap_code: u32,
    pub scalar_range: (f32, f32),
    /// Per-fixel opacity gate applied in-shader to the instance scalar.
    /// `OpacityGate::default()` (pass-through) leaves all fixels at full
    /// alpha. When auto-wired from `scene.default_fixel_otsu()`, fixels
    /// below the tracking-Otsu band fade to `below` alpha so the user
    /// sees which fixels feed tracking vs which are sub-threshold.
    pub opacity_gate: OpacityGate,
}

impl SceneFramePlan {
    /// The fixel draw that should drive `view`'s GPU upload: the first
    /// visible one of that view, else the first present. Centralizes what
    /// were three identical copies in the GUI viewer, the GUI job-sync
    /// loop, and the headless renderer. Kept next to the plan types it
    /// queries so the scene-plan API stays discoverable from `types.rs`.
    pub fn active_fixel_draw(&self, view: FixelView) -> Option<&FixelDrawPlan> {
        self.draws
            .of_type::<FixelDrawPlan>()
            .find(|p| p.view == view && p.visible)
            .or_else(|| self.draws.of_type::<FixelDrawPlan>().find(|p| p.view == view))
    }

    /// Whether any fixel draw was emitted this frame (either view).
    pub fn has_fixel_draws(&self) -> bool {
        self.draws.of_type::<FixelDrawPlan>().next().is_some()
    }

    /// Whether any emitted fixel draw is visible (either view).
    pub fn any_fixel_visible(&self) -> bool {
        self.draws.of_type::<FixelDrawPlan>().any(|p| p.visible)
    }

    /// The boundary-glyph draw that should drive the GPU upload: first
    /// visible, else first present (single-active, like fixel/ODF glyph).
    pub fn active_boundary_glyph_draw(&self) -> Option<&BoundaryGlyphDrawPlan> {
        self.draws
            .of_type::<BoundaryGlyphDrawPlan>()
            .find(|d| d.visible)
            .or_else(|| self.draws.of_type::<BoundaryGlyphDrawPlan>().next())
    }

    /// The ODF-glyph draw that should drive the GPU upload: first visible,
    /// else first present (single-active).
    pub fn active_odf_glyph_draw(&self) -> Option<&OdfGlyphDrawPlan> {
        self.draws
            .of_type::<OdfGlyphDrawPlan>()
            .find(|p| p.visible)
            .or_else(|| self.draws.of_type::<OdfGlyphDrawPlan>().next())
    }

    /// Surface draws for one display space (Anatomical vs Stage). After
    /// the registry merge, the `space` field — not a field name — is the
    /// sole discriminant between the two, so every consumer that used to
    /// pick a field now filters here.
    pub fn surface_draws(
        &self,
        space: SurfaceDisplaySpace,
    ) -> impl Iterator<Item = &SurfaceDrawPlan> {
        self.draws
            .of_type::<SurfaceDrawPlan>()
            .filter(move |d| d.space == space)
    }
}

#[derive(Clone)]
pub struct OdfGlyphDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub field: crate::data::odx_data::OdfField,
    pub scale: f32,
    pub subtract_iso: bool,
    pub norm_within_voxel: bool,
    pub opacity: f32,
    pub offset_from_slice: f32,
    pub gloss: f32,
    pub vertex_colormap: GlyphColormap,
    pub slice_axis: WorkflowSliceViewKind,
    pub opacity_gate: OpacityGate,
    pub size_gate: SizeGate,
    pub detail: u32,
    pub opacity_scalars: Option<Arc<VolumeScalars>>,
    pub size_scalars: Option<Arc<VolumeScalars>>,
    pub visible: bool,
}

impl Default for SceneFramePlan {
    fn default() -> Self {
        Self {
            reactive_streamline_plans: Vec::new(),
            surface_query_plans: Vec::new(),
            surface_map_plans: Vec::new(),
            streamline_draws: Vec::new(),
            bundle_surface_plans: Vec::new(),
            boundary_fields_in_use: std::collections::HashSet::new(),
            boundary_field_plans: Vec::new(),
            draws: super::draw::DrawList::default(),
            dipy_tractography_plans: Vec::new(),
            yeh_tractography_plans: Vec::new(),
            hausdorff_plan_jobs: Vec::new(),
            pyafq_plan_jobs: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct StreamlineFlow {
    pub dataset: Arc<StreamlineDataset>,
    pub selected_streamlines: Vec<StreamlineIndex>,
    pub color_mode: ColorMode,
    pub scalar_auto_range: bool,
    pub scalar_range_min: f32,
    pub scalar_range_max: f32,
    /// Colormap used when `color_mode` is a scalar mode
    /// (`Dps` / `Dpv`). Ignored for `DirectionRgb`, `Group`,
    /// `Uniform`. Defaults to `BlueWhiteRed`.
    pub scalar_colormap: SurfaceColormap,
}

#[derive(Clone)]
pub struct StreamlineDataset {
    pub name: String,
    pub gpu_data: Arc<TrxGpuData>,
    pub backing: StreamlineBacking,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct ParcellationAsset {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<ParcellationVolume>,
    pub label_table_path: Option<PathBuf>,
    pub visible: bool,
}

#[derive(Clone)]
pub struct LoadedParcellation {
    pub asset: ParcellationAsset,
}

#[derive(Clone)]
pub struct StreamlineDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub draw_id: FileId,
    pub label: String,
    pub visible: bool,
    pub flow: StreamlineFlow,
    pub render_style: RenderStyle,
    pub tube_radius_mm: Millimeters,
    pub tube_sides: u32,
    pub slab_half_width_mm: Millimeters,
    /// Per-display opacity multiplier in [0, 1]. Multiplied with the
    /// per-vertex color alpha by the streamline / tube fragment
    /// shader. 1.0 = fully opaque (default).
    pub opacity: f32,
}

#[derive(Clone)]
pub struct BundleDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub build_node_uuid: WorkflowNodeUuid,
    pub boundary_field_node_uuid: Option<WorkflowNodeUuid>,
    pub draw_id: FileId,
    pub label: String,
    pub flow: StreamlineFlow,
    pub per_group: bool,
    pub color_mode: BundleSurfaceColorMode,
    pub build_mode: BundleSurfaceBuildMode,
    pub voxel_size_mm: Millimeters,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub min_component_volume_mm3: Millimeters,
    pub tube_radius_mm: Millimeters,
    pub tube_sides: u32,
    pub opacity: f32,
    pub outline_thickness: f32,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct BoundaryGlyphDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub visible: bool,
    pub scale: f32,
    pub density_3d_step: usize,
    pub slice_density_step: usize,
    pub color_mode: BoundaryGlyphColorMode,
    pub min_contacts: u32,
}

/// Where a `VolumeDisplayOp` gets its voxel data from. `File` references a
/// loaded NIfTI on disk (or an ODX-DPV materialization keyed in
/// `execution_cache.odx_dpv_materializations`). `InMemory` carries the
/// `VolumeScalars` value produced upstream by another op (e.g. the pyAFQ
/// probability map output) along with a content-derived `handle` used to
/// key cached GPU slice resources — same handle ⇒ same data ⇒ no
/// re-upload, even though the value gets cloned through the workflow on
/// every evaluation.
#[derive(Clone)]
pub enum VolumeBacking {
    File(FileId),
    InMemory {
        handle: u64,
        scalars: Arc<VolumeScalars>,
    },
    /// A multi-layer composite produced by `VolumeOverlayStackOp`.
    /// The renderer composes the requested 2D slice on demand from the
    /// stack's layers; downstream consumers that need scalar voxel
    /// access reject this variant.
    Composite {
        handle: u64,
        stack: Arc<CompositeVolumeStack>,
    },
}

impl VolumeBacking {
    /// Stable `usize` key for indexing `AllSliceResources.entries`. For
    /// `File` it's the `FileId` counter; for `InMemory`/`Composite`
    /// it's a 64-bit content hash. Counter values and random hashes
    /// coexist in the same `usize` namespace without practical
    /// collision risk.
    pub fn slice_key(&self) -> usize {
        match self {
            Self::File(id) => *id,
            Self::InMemory { handle, .. } => *handle as usize,
            Self::Composite { handle, .. } => *handle as usize,
        }
    }

    pub fn from_scalars(scalars: VolumeScalars) -> Self {
        let handle = volume_scalars_handle(&scalars);
        Self::InMemory {
            handle,
            scalars: Arc::new(scalars),
        }
    }
}

/// Interpolation kernel for resampling a layer onto another grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Interp {
    Nearest,
    Trilinear,
}

impl Default for Interp {
    fn default() -> Self {
        Self::Trilinear
    }
}

/// Per-layer config for a `VolumeOverlayStackOp`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VolumeOverlayLayerConfig {
    pub enabled: bool,
    pub opacity: f32,
    pub colormap: crate::data::loaded_files::VolumeColormap,
    pub window_center: f32,
    pub window_width: f32,
    pub threshold_min: f32,
    pub threshold_max: f32,
    /// Resampling kernel used when this layer's grid differs from the
    /// stack's base grid. Defaults to `Trilinear`; the op overrides to
    /// `Nearest` on first eval if the layer's `ScalarKind` is `Label`.
    pub interpolation: Interp,
    pub legend_label: String,
}

impl Default for VolumeOverlayLayerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            opacity: 1.0,
            colormap: crate::data::loaded_files::VolumeColormap::Grayscale,
            window_center: 0.5,
            window_width: 1.0,
            threshold_min: 0.0,
            threshold_max: 1.0,
            interpolation: Interp::Trilinear,
            legend_label: String::new(),
        }
    }
}

/// Snapshot of a `VolumeOverlayStackOp` evaluation: the base layer's
/// target grid plus all enabled-or-disabled layers in order. The
/// renderer composes 2D slices from this on demand.
#[derive(Clone)]
pub struct CompositeVolumeStack {
    /// Target grid (layer 0's dims).
    pub dims: [usize; 3],
    /// Target grid (layer 0's voxel-to-RAS).
    pub voxel_to_ras: glam::Mat4,
    /// Layer 0 first; subsequent layers carry their own dims/affine
    /// inside their `VolumeScalars`.
    pub layers: Vec<(Arc<VolumeScalars>, VolumeOverlayLayerConfig)>,
}

impl CompositeVolumeStack {
    /// Content-derived handle: changes whenever the layer set, any
    /// underlying `VolumeScalars`, or any layer config changes. Used
    /// to key the GPU composite slice cache.
    pub fn handle(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.dims.hash(&mut h);
        for v in self.voxel_to_ras.to_cols_array() {
            v.to_bits().hash(&mut h);
        }
        for (scalars, cfg) in &self.layers {
            volume_scalars_handle(scalars).hash(&mut h);
            cfg.enabled.hash(&mut h);
            cfg.opacity.to_bits().hash(&mut h);
            cfg.colormap.as_u32().hash(&mut h);
            cfg.window_center.to_bits().hash(&mut h);
            cfg.window_width.to_bits().hash(&mut h);
            cfg.threshold_min.to_bits().hash(&mut h);
            cfg.threshold_max.to_bits().hash(&mut h);
            (cfg.interpolation as u8).hash(&mut h);
            cfg.legend_label.hash(&mut h);
        }
        h.finish()
    }
}

/// Four pre-allocated layer slots for a fresh `VolumeOverlayStack`.
/// Mirrors `default_surface_overlay_layers` — fixed count, only layer
/// 0 enabled by default.
pub fn default_volume_overlay_layers() -> Vec<VolumeOverlayLayerConfig> {
    let mut layers = vec![VolumeOverlayLayerConfig::default(); 4];
    layers[0].enabled = true;
    layers[0].legend_label = "Base".to_string();
    layers
}

fn volume_scalars_handle(s: &VolumeScalars) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.dims.hash(&mut h);
    for v in s.voxel_to_ras.to_cols_array() {
        v.to_bits().hash(&mut h);
    }
    s.values.len().hash(&mut h);
    let stride = (s.values.len() / 64).max(1);
    for i in (0..s.values.len()).step_by(stride) {
        s.values[i].to_bits().hash(&mut h);
    }
    h.finish()
}

#[derive(Clone)]
pub struct VolumeDrawPlan {
    pub source: VolumeBacking,
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub window_center: f32,
    pub window_width: f32,
}

#[derive(Clone)]
pub struct SurfaceDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub source_id: FileId,
    pub structure: Option<CiftiStructure>,
    pub color: [f32; 3],
    pub opacity: f32,
    pub outline_color: [f32; 3],
    pub outline_thickness: f32,
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub gloss: f32,
    pub projection_colormap: SurfaceColormap,
    pub range_min: f32,
    pub range_max: f32,
    pub projection_scalars: Option<Vec<f32>>,
    pub vertex_rgba: Vec<[f32; 4]>,
    pub space: SurfaceDisplaySpace,
    pub model_matrix: [[f32; 4]; 4],
}

pub const DEFAULT_SURFACE_COLOR: [f32; 3] = [0.72, 0.72, 0.72];
pub const DEFAULT_SURFACE_OPACITY: f32 = 1.0;
pub const DEFAULT_SURFACE_BASE_RGBA: [f32; 4] = [0.72, 0.72, 0.72, 1.0];

#[derive(Clone)]
pub struct ParcellationDrawPlan {
    pub source_id: FileId,
    pub labels: BTreeSet<ParcelId>,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct SurfaceQueryPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: Millimeters,
}

#[derive(Clone)]
pub struct SurfaceMapPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: Millimeters,
    pub dps_field: Option<DpsFieldName>,
}

#[derive(Clone)]
pub enum ReactiveStreamlineOp {
    Merge,
    RemoveDuplicates {
        params: DuplicateRemovalParams,
    },
    ParcelROI {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<ParcelId>,
    },
    ParcelROA {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<ParcelId>,
    },
    ParcelEnd {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<ParcelId>,
        endpoint_count: usize,
    },
    ParcelCrop {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<ParcelId>,
        keep_inside: bool,
    },
    AddGroupsFromParcellation {
        parcellation: Arc<ParcellationVolume>,
        parcellation_name: String,
    },
}

#[derive(Clone)]
pub struct ReactiveStreamlinePlan {
    pub node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub op: ReactiveStreamlineOp,
    pub left: StreamlineFlow,
    pub right: StreamlineFlow,
}

#[derive(Clone)]
pub struct BundleSurfacePlan {
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub flow: StreamlineFlow,
    pub per_group: bool,
    pub build_mode: BundleSurfaceBuildMode,
    pub voxel_size_mm: Millimeters,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub min_component_volume_mm3: Millimeters,
    pub tube_radius_mm: Millimeters,
    pub tube_sides: u32,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct BoundaryFieldPlan {
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub flow: StreamlineFlow,
    pub voxel_size_mm: Millimeters,
    pub sphere_lod: u32,
    pub normalization: BoundaryGlyphNormalization,
    pub binning_mode: crate::data::orientation_field::DirectionFieldBinningMode,
}

#[derive(Clone)]
pub struct WorkflowRuntime {
    pub scene_plan: SceneFramePlan,
    pub node_state: HashMap<WorkflowNodeUuid, NodeEvalState>,
    pub save_streamline_targets: HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    pub graph_error: Option<String>,
}

impl Default for WorkflowRuntime {
    fn default() -> Self {
        Self {
            scene_plan: SceneFramePlan::default(),
            node_state: HashMap::new(),
            save_streamline_targets: HashMap::new(),
            graph_error: None,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct SaveStreamlinePlan {
    pub node_uuid: WorkflowNodeUuid,
    pub output_path: PathBuf,
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub(crate) struct ParcelSelection {
    pub source_id: FileId,
    pub labels: BTreeSet<ParcelId>,
}

#[derive(Clone)]
pub(crate) enum WorkflowValue {
    Streamline(StreamlineFlow),
    Volume(VolumeBacking),
    Cifti(FileId),
    Surface(FileId),
    Parcellation(FileId),
    ParcelSelection(ParcelSelection),
    GroupSelection(GroupFilter),
    SurfaceScalars(SurfaceScalars),
    SurfaceAppearance(SurfaceAppearance),
    BundleSurface(BundleSurfacePlan),
    BoundaryField(BoundaryFieldPlan),
    Fixels(crate::data::odx_data::FixelField),
    FixelScalars(crate::data::odx_data::FixelScalars),
    OdfField(crate::data::odx_data::OdfField),
    OdxCatalog(crate::data::odx_data::OdxCatalog),
    VoxelMask(Arc<VoxelMask>),
    TrackingPlan(Arc<TrackingPlan>),
}

/// A voxel-space binary mask. Replaces the earlier point-cloud `SeedRoi`; now
/// fills every region role in a `TrackingPlan` (seed / limiting / roa / term /
/// end / no_end).
#[derive(Debug, Default)]
pub struct VoxelMask {
    pub dims: [u32; 3],
    pub voxel_to_ras: glam::Mat4,
    /// Row-major over (x, y, z): linear index = x + dims.x * (y + dims.y * z).
    /// One byte per voxel; non-zero = inside the mask.
    pub data: Vec<u8>,
    /// Lazily-computed RAS+mm centers of every non-zero voxel. Filled on
    /// first call to `nonzero_voxel_centers_ras` so per-attempt seed
    /// generation in cpu_dipy / gpu/dipy doesn't re-scan O(V) voxels each
    /// tractography re-run. Public so existing `VoxelMask { dims, … }`
    /// struct-literal constructions stay valid; pass
    /// `std::sync::OnceLock::new()` (the `Default`) to leave the cache
    /// empty.
    #[doc(hidden)]
    pub nonzero_centers: std::sync::OnceLock<Arc<Vec<[f32; 3]>>>,
    /// Lazily-computed list of (i, j, k) indices for every non-zero
    /// voxel. Used by the voxel-accurate slice overlay so it iterates
    /// only on-voxels per frame instead of scanning the full volume.
    /// Same caching pattern as `nonzero_centers`.
    #[doc(hidden)]
    pub nonzero_indices: std::sync::OnceLock<Arc<Vec<[u32; 3]>>>,
}

impl Clone for VoxelMask {
    fn clone(&self) -> Self {
        Self {
            dims: self.dims,
            voxel_to_ras: self.voxel_to_ras,
            data: self.data.clone(),
            // The cache is intentionally not shared across clones — the
            // common case is that the same mask is wrapped in `Arc<VoxelMask>`
            // and clones are vanishingly rare, so an empty cache here is
            // simpler than threading shared interior state through Clone.
            nonzero_centers: std::sync::OnceLock::new(),
            nonzero_indices: std::sync::OnceLock::new(),
        }
    }
}

impl VoxelMask {
    pub fn lin_idx(&self, x: u32, y: u32, z: u32) -> usize {
        (x as usize)
            + (self.dims[0] as usize) * ((y as usize) + (self.dims[1] as usize) * (z as usize))
    }

    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b != 0).count()
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&b| b == 0)
    }

    /// Enumerate RAS+mm centers of every non-zero voxel. Memoized per mask:
    /// the first call walks the volume and stores the result; subsequent
    /// calls return a clone of the cached `Arc`. Each tractography re-run
    /// reuses the same mask Arc, so seed generation no longer pays an O(V)
    /// scan per parameter sweep.
    /// Enumerate (i, j, k) indices of every non-zero voxel. Memoized
    /// per mask. Cheap to clone (returns an `Arc`).
    pub fn nonzero_voxel_indices(&self) -> Arc<Vec<[u32; 3]>> {
        self.nonzero_indices
            .get_or_init(|| {
                let [nx, ny, nz] = self.dims;
                let mut out = Vec::new();
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let idx = self.lin_idx(x, y, z);
                            if self.data[idx] != 0 {
                                out.push([x, y, z]);
                            }
                        }
                    }
                }
                Arc::new(out)
            })
            .clone()
    }

    pub fn nonzero_voxel_centers_ras(&self) -> Vec<[f32; 3]> {
        self.nonzero_centers
            .get_or_init(|| {
                let [nx, ny, nz] = self.dims;
                let mut out = Vec::new();
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let idx = self.lin_idx(x, y, z);
                            if self.data[idx] != 0 {
                                let p = self.voxel_to_ras.transform_point3(glam::Vec3::new(
                                    x as f32, y as f32, z as f32,
                                ));
                                out.push([p.x, p.y, p.z]);
                            }
                        }
                    }
                }
                Arc::new(out)
            })
            .as_ref()
            .clone()
    }
}

/// A plan bundling every spatial role a tractography method may consult.
/// Fields are optional; any subset may be populated. An unwired input to a
/// tracking op synthesizes a plan from the ODX/fixel whole-mask with no
/// constraints.
#[derive(Clone, Debug)]
pub struct TrackingPlan {
    pub label: String,
    pub grid_dims: [u32; 3],
    pub voxel_to_ras: glam::Mat4,
    // Per-step constraints (enforced during GPU propagation).
    pub seed_mask: Option<Arc<VoxelMask>>,
    pub limiting_mask: Option<Arc<VoxelMask>>,
    pub roa_mask: Option<Arc<VoxelMask>>,
    pub term_mask: Option<Arc<VoxelMask>>,
    // Post-hoc whole-streamline filters (applied after GPU readback).
    /// Waypoint regions — a streamline must pass through **every** mask in
    /// this list to be kept (AND semantics). Applied post-hoc.
    pub roi_masks: Vec<Arc<VoxelMask>>,
    pub end_masks: Vec<Arc<VoxelMask>>,
    pub no_end_mask: Option<Arc<VoxelMask>>,
    pub post_filter: Option<PostFilter>,
    // Optional per-parameter overrides. When `Some`, a consuming tracker
    // should use the plan's value instead of its own slider. Anything left
    // `None` falls back to the tracker's own setting. `tolerance_mm` is
    // carried for informational purposes and for future limiting-mask
    // enforcement.
    pub min_len_mm: Option<f32>,
    pub max_len_mm: Option<f32>,
    pub max_angle_deg: Option<f32>,
    pub step_size_mm: Option<f32>,
    pub fixel_threshold: Option<f32>,
    /// Yeh-specific direction-smoothing fraction. Other trackers ignore.
    pub smooth_fraction: Option<f32>,
    pub tolerance_mm: Option<f32>,
    /// Otsu threshold of the tracking-metric scalar, in its native
    /// units. When `Some`, consuming trackers scale their fixel-threshold
    /// sentinel randomization (`fixel_threshold <= 0`) to
    /// `[0.5·fixel_otsu, 0.7·fixel_otsu]`, matching DSI-Studio.
    pub fixel_otsu: Option<f32>,
}

#[derive(Clone, Debug)]
pub enum PostFilter {
    /// Reject a candidate streamline if its mean min-distance to the
    /// reference point cloud exceeds `max_mm`.
    Hausdorff {
        reference_points_ras: Arc<Vec<[f32; 3]>>,
        max_mm: f32,
    },
}

#[derive(Clone)]
pub(crate) struct EvaluatedValue {
    pub value: WorkflowValue,
    pub stale: bool,
}

impl From<WorkflowValue> for EvaluatedValue {
    fn from(value: WorkflowValue) -> Self {
        Self {
            value,
            stale: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowSelection {
    Node(WorkflowNodeUuid),
    Asset(FileId),
}

#[derive(Clone)]
pub struct StreamlineDisplayRuntime {
    pub draw_id: FileId,
    pub fingerprint: u64,
    pub bundle_fingerprint: Option<u64>,
    pub bundle_meshes_cpu: Vec<BundleMesh>,
}

impl Default for StreamlineDisplayRuntime {
    fn default() -> Self {
        Self {
            draw_id: 0,
            fingerprint: 0,
            bundle_fingerprint: None,
            bundle_meshes_cpu: Vec::new(),
        }
    }
}

/// Direction-getter variant used by the DIPY-style local tracker.
///
/// Mirrors DIPY's swappable direction-getter abstraction: every variant
/// shares the same outer `LocalTracking` skeleton (seed loop, forward/
/// backward branches, stitching, post-filters) and differs only in how
/// the per-step direction is chosen. Pure-data variants (`Probabilistic`)
/// are parameter-free here — they read what they need from the parent
/// `DipyTractographyPlan` (e.g. `relative_peak_threshold`). Variants
/// with their own knobs (`Ptt`) carry them inline.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DipyDirectionGetter {
    /// PMF-on-sphere sampling (the existing implementation in cpu_dipy.rs).
    Probabilistic,
    /// Parallel Transport Tractography (Aydogan & Shi 2021).
    ///
    /// Implementation lives in `gpu/dipy.rs::run_gpu_dipy_ptt` (GPU
    /// only — no CPU reference yet). See `docs/ptt-implementation-notes.md`
    /// for the algorithm overview and parameter mapping vs. nibrary/DIPY.
    Ptt {
        /// Length (mm) of each candidate arc evaluated per step.
        probe_length_mm: f32,
        /// Number of FOD samples taken along each candidate arc when
        /// scoring it. (DIPY parameter name: `probe_quality`.)
        probe_quality: u32,
        /// Radius (mm) of the circumferential probe samples around
        /// each arc point. `0.0` means a degenerate probe — single
        /// sample at the arc point itself.
        probe_radius_mm: f32,
        /// Number of circumferential samples around each arc point at
        /// `probe_radius_mm`. `1` collapses to a single on-axis sample.
        probe_count: u32,
        /// Maximum frame curvature (1/mm) considered when generating
        /// candidate arcs. Larger = sharper turns allowed.
        max_curvature_per_mm: f32,
        /// Exponent applied to the per-candidate data support (FOD
        /// integral). DIPY default 1.0; values > 1 sharpen the
        /// posterior toward the strongest candidates.
        data_support_exponent: f32,
        /// Floor on candidate data support. Candidates below this are
        /// rejected outright. DIPY default 0.05.
        min_data_support: f32,
        /// Maximum candidate evaluations per step before giving up
        /// the streamline. DIPY default 100.
        rejection_sampling_max_try: u32,
    },
}

impl Default for DipyDirectionGetter {
    fn default() -> Self {
        DipyDirectionGetter::Probabilistic
    }
}

impl DipyDirectionGetter {
    /// DIPY-equivalent default PTT parameters. Convenience constructor
    /// so callers don't need to remember every field.
    pub fn ptt_default() -> Self {
        DipyDirectionGetter::Ptt {
            probe_length_mm: 0.5,
            probe_quality: 4,
            probe_radius_mm: 0.0,
            probe_count: 1,
            max_curvature_per_mm: 1.0 / 3.0,
            data_support_exponent: 1.0,
            min_data_support: 0.05,
            rejection_sampling_max_try: 100,
        }
    }
}

/// Parameters and inputs for a GPU/CPU tractography run.
#[derive(Clone)]
pub struct DipyTractographyPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub label: String,
    /// Authoritative fingerprint for this plan — the hash the GUI keys
    /// the in-flight job and result cache on. Computed by the op
    /// during `evaluate()` from its config + upstream identity and
    /// persists unchanged through dispatch / completion / caching.
    /// Replaces the previous "GUI reads fingerprint from
    /// `node_runs.current_fingerprint` with an `unwrap_or(0)` fallback"
    /// pattern that produced PR 1's silent-discard race.
    pub fingerprint: crate::workflow::op::ContentHash,
    pub odx_source_id: FileId,
    pub odx_scene: Arc<crate::data::odx_data::OdxScene>,
    /// When `None`, seed from every voxel in the ODX mask (whole-brain,
    /// mirroring Yeh's default when no mask is wired).
    pub seed_mask: Option<Arc<VoxelMask>>,
    pub step_size_mm: f32,
    pub max_angle_deg: f32,
    pub min_len_mm: f32,
    pub max_len_mm: f32,
    pub fixel_threshold: f32,
    pub relative_peak_threshold: f32,
    pub seeds_per_voxel: u32,
    pub max_points: u32,
    pub rng_seed: u64,
    // Constraint masks populated from a wired `TrackingPlan`. Enforced by
    // the CPU tracker (per-step for limiting/roa/term; post-hoc for
    // roi/end/no_end/post_filter). The current GPU path ignores them.
    pub limiting_mask: Option<Arc<VoxelMask>>,
    pub roa_mask: Option<Arc<VoxelMask>>,
    pub term_mask: Option<Arc<VoxelMask>>,
    pub roi_masks: Vec<Arc<VoxelMask>>,
    pub end_masks: Vec<Arc<VoxelMask>>,
    pub no_end_mask: Option<Arc<VoxelMask>>,
    pub post_filter: Option<PostFilter>,
    /// Otsu threshold in the tracking metric's native units (see
    /// `TrackingPlan::fixel_otsu`). Drives the CPU tracker's
    /// `fixel_threshold <= 0` randomization base.
    pub fixel_otsu: Option<f32>,
    /// Which DIPY direction-getter to use. Defaults to `Probabilistic`
    /// (the only variant currently implemented on CPU).
    pub direction_getter: DipyDirectionGetter,
}

#[derive(Clone)]
pub struct YehTractographyPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub label: String,
    /// Authoritative fingerprint for this plan — see the note on
    /// `DipyTractographyPlan::fingerprint`. The GUI reads this field
    /// directly when queueing a job; no more `unwrap_or(0)` fallback
    /// through the cache.
    pub fingerprint: crate::workflow::op::ContentHash,
    pub odx_source_id: FileId,
    pub odx_scene: Arc<crate::data::odx_data::OdxScene>,
    /// When `None`, seed from every voxel with at least one fixel peak.
    pub seed_mask: Option<Arc<VoxelMask>>,
    /// Per-step: streamline terminates if it leaves this mask.
    pub limiting_mask: Option<Arc<VoxelMask>>,
    /// Per-step: streamline is rejected if it enters this mask.
    pub roa_mask: Option<Arc<VoxelMask>>,
    /// Per-step: streamline terminates cleanly if it enters this mask.
    pub term_mask: Option<Arc<VoxelMask>>,
    /// Post-hoc: streamline must pass through **every** mask in this list
    /// (AND-semantics waypoints).
    pub roi_masks: Vec<Arc<VoxelMask>>,
    /// Post-hoc: streamline must touch each of these end regions at an
    /// endpoint (DSI-Studio-style end_region logic, simplified to "at least
    /// one endpoint per end_mask").
    pub end_masks: Vec<Arc<VoxelMask>>,
    /// Post-hoc: streamline is rejected if either endpoint lies in this
    /// mask.
    pub no_end_mask: Option<Arc<VoxelMask>>,
    /// Post-hoc: additional filter (e.g. Hausdorff distance to reference).
    pub post_filter: Option<PostFilter>,
    /// Base step size (mm). Per-seed step ∈ [0.5, 1.5] × this value.
    pub step_size_mm: f32,
    /// Maximum turning angle (degrees). Per-seed angle sampled ∈
    /// [max_angle_deg/2, max_angle_deg].
    pub max_angle_deg: f32,
    pub min_len_mm: f32,
    pub max_len_mm: f32,
    /// Base fixel threshold. Per-seed threshold jittered ∈ [base − 0.1, base + 0.1].
    pub fixel_threshold: f32,
    /// Direction smoothing fraction. 0.0 = pure new peak, 0.95 = heavy
    /// carry-over, and the sentinel **1.0** triggers per-seed randomization
    /// in `[0.0, 0.95]` (matches DSI-Studio).
    pub smooth_fraction: f32,
    pub max_points: u32,
    /// Stop seeding once this many streamlines have been kept.
    pub target_streamlines: u32,
    /// Safety cap: stop even if `target_streamlines` hasn't been reached after
    /// this many random seed attempts.
    pub max_seed_attempts: u32,
    pub rng_seed: u64,
    /// Otsu threshold in the tracking metric's native units (see
    /// `TrackingPlan::fixel_otsu`). Used as the base value for the
    /// `fixel_threshold <= 0` sentinel randomization.
    pub fixel_otsu: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkflowJobKind {
    ReactiveStreamline,
    SurfaceQuery,
    SurfaceMap,
    TubeGeometry,
    BundleSurface,
    BoundaryField,
    DipyTractography,
    YehTractography,
    PrepareHausdorffPlan,
    PreparePyafqPlan,
}

#[derive(Clone)]
pub enum WorkflowJobPayload {
    ReactiveStreamline(ReactiveStreamlinePlan),
    SurfaceQuery(SurfaceQueryPlan),
    SurfaceMap(SurfaceMapPlan),
    TubeGeometry(StreamlineDrawPlan),
    BundleSurface {
        plan: BundleSurfacePlan,
        color_mode: BundleSurfaceColorMode,
        boundary_field: Option<Arc<BoundaryContactField>>,
    },
    BoundaryField {
        plan: BoundaryFieldPlan,
    },
    DipyTractography {
        plan: DipyTractographyPlan,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    },
    YehTractography {
        plan: YehTractographyPlan,
    },
    PrepareHausdorffPlan(HausdorffPlanJob),
    PreparePyafqPlan(PyafqPlanJob),
}

#[derive(Clone)]
pub enum WorkflowJobOutput {
    ReactiveStreamline(StreamlineFlow),
    SurfaceQuery(StreamlineFlow),
    SurfaceMap(SurfaceScalars),
    TubeGeometry {
        vertices: Vec<crate::data::trx_data::TubeMeshVertex>,
        indices: Vec<u32>,
    },
    BundleSurface {
        meshes: Vec<(BundleMesh, String)>,
    },
    BoundaryField {
        field: Option<Arc<BoundaryContactField>>,
    },
    DipyTractography {
        flow: StreamlineFlow,
    },
    YehTractography {
        flow: StreamlineFlow,
    },
    PrepareHausdorffPlan {
        plan: Arc<TrackingPlan>,
        seed_mask: Arc<VoxelMask>,
        limiting_mask: Arc<VoxelMask>,
        no_end_mask: Arc<VoxelMask>,
        summary: String,
    },
    PreparePyafqPlan {
        plan: Arc<TrackingPlan>,
        include_mask: Arc<VoxelMask>,
        exclude_mask: Arc<VoxelMask>,
        start_mask: Arc<VoxelMask>,
        end_mask: Arc<VoxelMask>,
        prob_map: Option<Arc<VolumeScalars>>,
        summary: String,
    },
}

/// All inputs needed to run a Hausdorff plan build off-thread. Cloned by the
/// op's `evaluate()` so the GUI thread doesn't block on EDT/voxelization.
#[derive(Clone)]
pub struct HausdorffPlanJob {
    pub node_uuid: WorkflowNodeUuid,
    pub fingerprint: u64,
    pub label: String,
    pub odx_scene: Arc<crate::data::odx_data::OdxScene>,
    pub gpu_data: Arc<crate::data::trx_data::TrxGpuData>,
    pub selected: Vec<u32>,
    pub params: crate::gpu::plan_prep::hausdorff::HausdorffPlanParams,
    pub fixel_otsu: odx_rs::qc::FixelOtsu,
}

/// All inputs needed to run a pyAFQ plan build off-thread.
#[derive(Clone)]
pub struct PyafqPlanJob {
    pub node_uuid: WorkflowNodeUuid,
    pub fingerprint: u64,
    pub label: String,
    pub working_dir: std::path::PathBuf,
    pub bundle_spec: &'static crate::workflow::ops::pyafq_bundles::PyafqBundleSpec,
    pub params: crate::gpu::plan_prep::pyafq::PyafqPlanParams,
}

pub enum WorkflowJobMessage {
    Started {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
    },
    /// Incremental progress from a long-running job. Trackers emit
    /// these every ~1024 attempts (CPU) or every batch (GPU) so the
    /// GUI can render a real progress bar rather than just a spinner.
    /// `done` and `total` are in whatever unit makes sense for the op
    /// (attempts for CPU tractography, seeds for GPU).
    Progress {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        done: u64,
        total: u64,
    },
    Finished {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        result: WorkflowResult<WorkflowJobOutput>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortKind {
    Streamline,
    Volume,
    Cifti,
    Surface,
    Parcellation,
    ParcelSelection,
    GroupSelection,
    SurfaceScalars,
    SurfaceAppearance,
    BundleSurface,
    BoundaryField,
    Fixels,
    FixelScalars,
    OdfField,
    OdxCatalog,
    VoxelMask,
    TrackingPlan,
}

pub struct SeededWorkflowBranch {
    pub bounds: GraphRect,
    pub primary_selection: WorkflowSelection,
}

pub fn default_next_node_uuid() -> u64 {
    1
}

pub fn default_document() -> WorkflowDocument {
    WorkflowDocument {
        next_node_uuid: default_next_node_uuid(),
        graph: WorkflowGraph::new(),
        assets: Vec::new(),
        camera_3d: None,
        render_3d: None,
        slice_view_3d: None,
        slice_view_ui: None,
        selection: None,
    }
}

impl Default for WorkflowProject {
    fn default() -> Self {
        Self {
            version: 1,
            document: default_document(),
        }
    }
}

/// Walk the document and assign UUIDs to any nodes that still carry
/// `WorkflowNodeUuid(0)`. Called after loading a project to ensure every
/// node has a stable identity before evaluation.
pub fn ensure_node_uuids(document: &mut WorkflowDocument) {
    let mut next = document.next_node_uuid.max(1);
    let zero_uuids: Vec<WorkflowNodeUuid> = document
        .graph
        .nodes()
        .filter_map(|(uuid, _)| (uuid.0 == 0).then_some(uuid))
        .collect();
    if !zero_uuids.is_empty() {
        log::warn!(
            "ensure_node_uuids: {} node(s) with uuid 0 in canonical graph; \
             will be repaired on next editor sync",
            zero_uuids.len()
        );
    }
    next = next.max(document.graph.max_uuid() + 1);
    document.next_node_uuid = next;
}

#[cfg(test)]
mod tests {
    use super::{
        PortKind, WorkflowCamera3D, WorkflowDocument, WorkflowNodeKind, WorkflowOrthoSliceCamera,
        WorkflowProject, WorkflowSliceView3D, WorkflowSliceViewKind, WorkflowSliceViewUi,
        WorkflowView2DMode, default_surface_overlay_layers,
    };
    use crate::lighting::{SceneLightingPreset, WorkflowBackground3D, WorkflowRender3D};

    #[test]
    fn workflow_document_camera_round_trips() {
        let mut document = super::default_document();
        document.camera_3d = Some(WorkflowCamera3D {
            target: [1.0, 2.0, 3.0],
            azimuth_deg: 45.0,
            elevation_deg: 25.0,
            distance: 180.0,
        });
        document.render_3d = Some(WorkflowRender3D {
            lighting_preset: SceneLightingPreset::Studio,
            background: WorkflowBackground3D::Solid {
                color: [0.1, 0.2, 0.3],
            },
            fog_enabled: true,
            fog_color: [0.2, 0.3, 0.4],
            fog_start_fraction: 0.6,
            fog_end_fraction: 0.95,
            vignette_strength: 0.2,
            exposure: 1.1,
            contrast: 1.05,
        });
        document.slice_view_3d = Some(WorkflowSliceView3D {
            visible: [true, false, true],
            positions_ras: [10.0, 20.0, 30.0],
        });
        document.slice_view_ui = Some(WorkflowSliceViewUi {
            mode: WorkflowView2DMode::Lightbox,
            single_view: WorkflowSliceViewKind::Axial,
            lightbox_axis: WorkflowSliceViewKind::Sagittal,
            lightbox_rows: 3,
            lightbox_cols: 4,
            active_axis: 2,
            ortho_show_row: false,
            slice_cameras: [WorkflowOrthoSliceCamera {
                center: [0.0, 1.0],
                half_extent: 42.0,
                rotation: 0.0,
            }; 3],
        });
        document.selection = Some(super::WorkflowSelection::Node(super::WorkflowNodeUuid(7)));

        let json = serde_json::to_string(&document).unwrap();
        let restored: WorkflowDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.camera_3d, document.camera_3d);
        assert_eq!(restored.render_3d, document.render_3d);
        assert_eq!(restored.slice_view_3d, document.slice_view_3d);
        assert_eq!(restored.slice_view_ui, document.slice_view_ui);
        assert_eq!(restored.selection, document.selection);
    }

    #[test]
    fn workflow_document_defaults_camera_when_missing() {
        let json = r#"{"next_node_uuid":1,"graph":{"nodes":{},"wires":[]},"assets":[]}"#;
        let restored: WorkflowDocument = serde_json::from_str(json).unwrap();
        assert!(restored.camera_3d.is_none());
        assert!(restored.render_3d.is_none());
        assert!(restored.slice_view_3d.is_none());
        assert!(restored.slice_view_ui.is_none());
        assert!(restored.selection.is_none());
    }

    #[test]
    fn odx_volume_select_exposes_volume_and_volume_scalars_outputs() {
        assert_eq!(
            WorkflowNodeKind::OdxVolumeSelect {
                dpv_name: String::new()
            }
            .outputs(),
            vec![PortKind::Volume]
        );
    }

    #[test]
    fn surface_overlay_stack_inputs_include_surface_and_layer_scalars() {
        let layers = default_surface_overlay_layers();
        let inputs = WorkflowNodeKind::SurfaceOverlayStack {
            layers: layers.clone(),
        }
        .inputs();
        assert_eq!(inputs.len(), layers.len() + 1);
        assert_eq!(inputs.first(), Some(&PortKind::Surface));
        assert!(
            inputs[1..]
                .iter()
                .all(|port| *port == PortKind::SurfaceScalars)
        );
    }

    #[test]
    fn workflow_document_defaults_next_node_uuid_when_missing() {
        let json = r#"{"graph":{"nodes":{},"wires":[]},"assets":[]}"#;
        let restored: WorkflowDocument = serde_json::from_str(json).unwrap();
        assert_eq!(restored.next_node_uuid, 1);
    }

    #[test]
    fn workflow_project_slice_view_ui_round_trips() {
        let project = WorkflowProject {
            version: 1,
            document: WorkflowDocument {
                slice_view_ui: Some(WorkflowSliceViewUi {
                    mode: WorkflowView2DMode::Lightbox,
                    single_view: WorkflowSliceViewKind::Axial,
                    lightbox_axis: WorkflowSliceViewKind::Sagittal,
                    lightbox_rows: 3,
                    lightbox_cols: 4,
                    active_axis: 2,
                    ortho_show_row: false,
                    slice_cameras: [WorkflowOrthoSliceCamera {
                        center: [0.0, 1.0],
                        half_extent: 42.0,
                        rotation: 0.0,
                    }; 3],
                }),
                selection: Some(super::WorkflowSelection::Asset(42)),
                ..super::default_document()
            },
        };

        let json = serde_json::to_string(&project).unwrap();
        let restored: WorkflowProject = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.document.slice_view_ui,
            project.document.slice_view_ui
        );
        assert_eq!(restored.document.selection, project.document.selection);
    }

    #[test]
    fn odf_glyph_renderer_detail_defaults_when_missing() {
        let json = r#"{
            "OdfGlyphRenderer": {
                "scale": 3.25,
                "opacity": 1.0,
                "offset_from_slice": 0.0,
                "gloss": 0.0,
                "vertex_colormap": "Directional",
                "slice_axis": "Axial",
                "opacity_gate": {"range":[0.0,1.0],"below":0.0,"above":1.0},
                "size_gate": {"range":[0.0,1.0],"min_scale":0.5,"max_scale":1.5},
                "visible": true
            }
        }"#;
        let restored: WorkflowNodeKind = serde_json::from_str(json).unwrap();
        match restored {
            WorkflowNodeKind::OdfGlyphRenderer {
                detail,
                subtract_iso,
                norm_within_voxel,
                ..
            } => {
                assert_eq!(detail, 3);
                assert!(subtract_iso);
                assert!(!norm_within_voxel);
            }
            _ => panic!("expected OdfGlyphRenderer"),
        }
    }
}
#[derive(Clone)]
pub struct SurfaceAppearance {
    pub source_id: FileId,
    pub structure: Option<CiftiStructure>,
    pub vertex_rgba: Vec<[f32; 4]>,
    pub legend_labels: Vec<String>,
}

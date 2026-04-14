use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use super::graph::{GraphRect, WorkflowGraph};
use crate::data::bundle_mesh::BundleMesh;
use crate::data::cifti::{
    CiftiIntent, CiftiStructure, SurfaceScalars, VolumeScalars,
};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{FileId, StreamlineBacking, VolumeColormap};
use crate::data::orientation_field::{
    BoundaryContactField, BoundaryGlyphColorMode, BoundaryGlyphNormalization,
};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::lighting::WorkflowRender3D;
use crate::renderer::mesh_renderer::SurfaceColormap;
use trx_rs::DuplicateRemovalParams;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct WorkflowNodeUuid(pub u64);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowNode {
    pub uuid: WorkflowNodeUuid,
    pub kind: WorkflowNodeKind,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowNodeKind {
    StreamlineSource {
        source_id: FileId,
    },
    VolumeSource {
        source_id: FileId,
    },
    CiftiSource {
        source_id: FileId,
    },
    SurfaceSource {
        source_id: FileId,
    },
    CiftiStructure {
        structure: CiftiStructure,
        map_index: usize,
    },
    ParcellationSource {
        source_id: FileId,
    },
    LimitStreamlines {
        limit: usize,
        randomize: bool,
        seed: u64,
    },
    GroupSelect {
        groups_csv: String,
    },
    RandomSubset {
        limit: usize,
        seed: u64,
    },
    SphereQuery {
        center: [f32; 3],
        radius_mm: f32,
    },
    SurfaceDepthQuery {
        depth_mm: f32,
    },
    RemoveDuplicates {
        params: DuplicateRemovalParams,
    },
    Merge,
    AddGroupsFromParcellation,
    ParcelSelect {
        labels_csv: String,
    },
    ParcelROI,
    ParcelROA,
    ParcelEnd {
        endpoint_count: usize,
    },
    ParcelLimiting,
    ParcelTerminative,
    ParcelSurfaceBuild,
    ColorByDirection,
    ColorByGroup,
    ColorByDPV {
        field: String,
    },
    ColorByDPS {
        field: String,
    },
    UniformColor {
        color: [f32; 4],
    },
    SurfaceProjectionDensity {
        depth_mm: f32,
    },
    SurfaceProjectionMeanDps {
        depth_mm: f32,
        field: String,
    },
    SurfaceOverlayStack {
        #[serde(default = "default_surface_overlay_layers")]
        layers: Vec<SurfaceOverlayLayerConfig>,
    },
    BundleSurfaceBuild {
        #[serde(default)]
        per_group: bool,
        build_mode: BundleSurfaceBuildMode,
        voxel_size_mm: f32,
        threshold: f32,
        smooth_sigma: f32,
        #[serde(default = "default_bundle_surface_min_component_volume_mm3")]
        min_component_volume_mm3: f32,
        tube_radius_mm: f32,
        tube_sides: u32,
        opacity: f32,
    },
    BoundaryFieldBuild {
        #[serde(default = "default_boundary_field_voxel_size_mm")]
        voxel_size_mm: f32,
        #[serde(default = "default_boundary_field_sphere_lod")]
        sphere_lod: u32,
        #[serde(default = "default_boundary_field_normalization")]
        normalization: BoundaryGlyphNormalization,
    },
    StreamlineDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        render_style: RenderStyle,
        tube_radius_mm: f32,
        tube_sides: u32,
        slab_half_width_mm: f32,
    },
    VolumeDisplay {
        colormap: VolumeColormap,
        opacity: f32,
        window_center: f32,
        window_width: f32,
    },
    SurfaceDisplay {
        color: [f32; 3],
        opacity: f32,
        outline_color: [f32; 3],
        outline_thickness: f32,
        show_projection_map: bool,
        map_opacity: f32,
        map_threshold: f32,
        gloss: f32,
        projection_colormap: SurfaceColormap,
        range_min: f32,
        range_max: f32,
        space: SurfaceDisplaySpace,
    },
    VolumeScalarsDisplay {
        colormap: VolumeColormap,
        opacity: f32,
    },
    BundleSurfaceDisplay {
        #[serde(default)]
        color_mode: BundleSurfaceColorMode,
        #[serde(default = "default_bundle_surface_outline_thickness")]
        outline_thickness: f32,
    },
    BoundaryGlyphDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        #[serde(default = "default_boundary_glyph_scale")]
        scale: f32,
        #[serde(default = "default_boundary_glyph_density_3d_step")]
        density_3d_step: usize,
        #[serde(default = "default_boundary_glyph_slice_density_step")]
        slice_density_step: usize,
        #[serde(default = "default_boundary_glyph_color_mode")]
        color_mode: BoundaryGlyphColorMode,
        #[serde(default = "default_boundary_glyph_min_contacts")]
        min_contacts: u32,
    },
    ParcellationDisplay {
        labels_csv: String,
        opacity: f32,
    },
    SaveStreamlines {
        output_path: String,
    },
}

pub fn default_enabled() -> bool {
    true
}

pub fn default_boundary_field_voxel_size_mm() -> f32 {
    3.0
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

pub fn default_bundle_surface_min_component_volume_mm3() -> f32 {
    0.0
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub enum SurfaceDisplaySpace {
    #[default]
    Anatomical,
    Stage,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceOverlayLayerConfig {
    pub enabled: bool,
    pub solid_color: [f32; 4],
    pub opacity: f32,
    pub colormap: SurfaceColormap,
    pub range_min: f32,
    pub range_max: f32,
    pub threshold_min: f32,
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
    // Backward compatibility for projects saved before slice positions were persisted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_visible_3d: Option<[bool; 3]>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkflowProject {
    pub version: u32,
    pub document: WorkflowDocument,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_view_ui: Option<WorkflowSliceViewUi>,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            Self::BoundaryField => "Boundary field",
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
    pub status: WorkflowExecutionStatus,
    pub last_result_summary: Option<String>,
}

impl Default for ExpensiveNodeRunRecord {
    fn default() -> Self {
        Self {
            current_fingerprint: None,
            last_success_fingerprint: None,
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

#[derive(Clone, Default)]
pub struct WorkflowExecutionCache {
    pub node_runs: HashMap<WorkflowNodeUuid, ExpensiveNodeRunRecord>,
    pub derived_streamline_cache: HashMap<WorkflowNodeUuid, CachedDerivedStreamline>,
    pub surface_query_cache: HashMap<WorkflowNodeUuid, CachedSurfaceQuery>,
    pub surface_streamline_map_cache: HashMap<WorkflowNodeUuid, CachedSurfaceStreamlineMap>,
    pub tube_geometry_cache: HashMap<WorkflowNodeUuid, CachedTubeGeometry>,
    pub bundle_surface_mesh_cache: HashMap<WorkflowNodeUuid, CachedBundleSurfaceMeshes>,
    pub boundary_field_cache: HashMap<WorkflowNodeUuid, CachedBoundaryField>,
}

#[derive(Clone)]
pub struct SceneFramePlan {
    pub reactive_streamline_plans: Vec<ReactiveStreamlinePlan>,
    pub surface_query_plans: Vec<SurfaceQueryPlan>,
    pub surface_map_plans: Vec<SurfaceMapPlan>,
    pub streamline_draws: Vec<StreamlineDrawPlan>,
    pub volume_draws: Vec<VolumeDrawPlan>,
    pub surface_draws: Vec<SurfaceDrawPlan>,
    pub stage_surface_draws: Vec<SurfaceDrawPlan>,
    pub volume_scalar_draws: Vec<VolumeScalarDrawPlan>,
    pub bundle_surface_plans: Vec<BundleSurfacePlan>,
    pub bundle_draws: Vec<BundleDrawPlan>,
    pub parcellation_draws: Vec<ParcellationDrawPlan>,
    pub boundary_field_plans: Vec<BoundaryFieldPlan>,
    pub boundary_glyph_draws: Vec<BoundaryGlyphDrawPlan>,
}

impl Default for SceneFramePlan {
    fn default() -> Self {
        Self {
            reactive_streamline_plans: Vec::new(),
            surface_query_plans: Vec::new(),
            surface_map_plans: Vec::new(),
            streamline_draws: Vec::new(),
            volume_draws: Vec::new(),
            surface_draws: Vec::new(),
            stage_surface_draws: Vec::new(),
            volume_scalar_draws: Vec::new(),
            bundle_surface_plans: Vec::new(),
            bundle_draws: Vec::new(),
            parcellation_draws: Vec::new(),
            boundary_field_plans: Vec::new(),
            boundary_glyph_draws: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct StreamlineFlow {
    pub dataset: Arc<StreamlineDataset>,
    pub selected_streamlines: Arc<Vec<u32>>,
    pub color_mode: ColorMode,
    pub scalar_auto_range: bool,
    pub scalar_range_min: f32,
    pub scalar_range_max: f32,
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
    pub tube_radius_mm: f32,
    pub tube_sides: u32,
    pub slab_half_width_mm: f32,
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
    pub voxel_size_mm: f32,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub min_component_volume_mm3: f32,
    pub tube_radius_mm: f32,
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

#[derive(Clone, Copy)]
pub struct VolumeDrawPlan {
    pub source_id: FileId,
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
    pub labels: BTreeSet<u32>,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct SurfaceQueryPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: f32,
}

#[derive(Clone)]
pub struct SurfaceMapPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: f32,
    pub dps_field: Option<String>,
}

#[derive(Clone)]
pub enum ReactiveStreamlineOp {
    Merge,
    RemoveDuplicates {
        params: DuplicateRemovalParams,
    },
    ParcelROI {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<u32>,
    },
    ParcelROA {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<u32>,
    },
    ParcelEnd {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<u32>,
        endpoint_count: usize,
    },
    ParcelCrop {
        parcellation: Arc<ParcellationVolume>,
        labels: BTreeSet<u32>,
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
    pub voxel_size_mm: f32,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub min_component_volume_mm3: f32,
    pub tube_radius_mm: f32,
    pub tube_sides: u32,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct BoundaryFieldPlan {
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub flow: StreamlineFlow,
    pub voxel_size_mm: f32,
    pub sphere_lod: u32,
    pub normalization: BoundaryGlyphNormalization,
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
pub(super) struct ParcelSelection {
    pub source_id: FileId,
    pub labels: BTreeSet<u32>,
}

#[derive(Clone)]
pub(super) enum WorkflowValue {
    Streamline(StreamlineFlow),
    Volume(FileId),
    Cifti(FileId),
    Surface(FileId),
    Parcellation(FileId),
    ParcelSelection(ParcelSelection),
    SurfaceScalars(SurfaceScalars),
    VolumeScalars(VolumeScalars),
    SurfaceAppearance(SurfaceAppearance),
    BundleSurface(BundleSurfacePlan),
    BoundaryField(BoundaryFieldPlan),
}

#[derive(Clone)]
pub(super) struct EvaluatedValue {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkflowJobKind {
    ReactiveStreamline,
    SurfaceQuery,
    SurfaceMap,
    TubeGeometry,
    BundleSurface,
    BoundaryField,
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
}

#[derive(Clone)]
pub enum WorkflowJobMessage {
    Started {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
    },
    Finished {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        result: Result<WorkflowJobOutput, String>,
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
    SurfaceScalars,
    VolumeScalars,
    SurfaceAppearance,
    BundleSurface,
    BoundaryField,
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
        slice_visible_3d: None,
    }
}

impl Default for WorkflowProject {
    fn default() -> Self {
        Self {
            version: 1,
            document: default_document(),
            slice_view_ui: None,
        }
    }
}

impl WorkflowNodeKind {
    pub fn title(&self) -> &'static str {
        match self {
            Self::StreamlineSource { .. } => "Streamline Source",
            Self::VolumeSource { .. } => "Volume Source",
            Self::CiftiSource { .. } => "CIFTI Source",
            Self::SurfaceSource { .. } => "Surface Source",
            Self::CiftiStructure { structure, .. } => match structure {
                CiftiStructure::CortexLeft => "CIFTI Left Cortex",
                CiftiStructure::CortexRight => "CIFTI Right Cortex",
                CiftiStructure::Subcortical => "CIFTI Subcortex",
            },
            Self::ParcellationSource { .. } => "Parcellation Source",
            Self::LimitStreamlines { .. } => "Limit Streamlines",
            Self::GroupSelect { .. } => "Group Select",
            Self::RandomSubset { .. } => "Random Subset",
            Self::SphereQuery { .. } => "Sphere Query",
            Self::SurfaceDepthQuery { .. } => "Surface Depth Query",
            Self::RemoveDuplicates { .. } => "Remove Duplicates",
            Self::Merge => "Merge",
            Self::AddGroupsFromParcellation => "Add Groups From Parcellation",
            Self::ParcelSelect { .. } => "Parcel Select",
            Self::ParcelROI => "Parcel ROI",
            Self::ParcelROA => "Parcel ROA",
            Self::ParcelEnd { .. } => "Parcel End",
            Self::ParcelLimiting => "Parcel Limiting",
            Self::ParcelTerminative => "Parcel Terminative",
            Self::ParcelSurfaceBuild => "Parcel Surface Build",
            Self::ColorByDirection => "Color By Direction",
            Self::ColorByGroup => "Color By Group",
            Self::ColorByDPV { .. } => "Color By DPV",
            Self::ColorByDPS { .. } => "Color By DPS",
            Self::UniformColor { .. } => "Uniform Color",
            Self::SurfaceProjectionDensity { .. } => "Map Streamlines to Surface",
            Self::SurfaceProjectionMeanDps { .. } => "Map Streamlines to Surface (Mean DPS)",
            Self::SurfaceOverlayStack { .. } => "Surface Overlay Stack",
            Self::BundleSurfaceBuild { .. } => "Bundle Surface Build",
            Self::BoundaryFieldBuild { .. } => "Boundary Field Build",
            Self::StreamlineDisplay { .. } => "Streamline Display",
            Self::VolumeDisplay { .. } => "Volume Display",
            Self::VolumeScalarsDisplay { .. } => "Volume Scalars Display",
            Self::SurfaceDisplay { .. } => "Surface Display",
            Self::BundleSurfaceDisplay { .. } => "Bundle Surface Display",
            Self::BoundaryGlyphDisplay { .. } => "Boundary Glyph Display",
            Self::ParcellationDisplay { .. } => "Parcellation Display",
            Self::SaveStreamlines { .. } => "Save Streamlines",
        }
    }

    pub fn inputs(&self) -> Vec<PortKind> {
        match self {
            Self::StreamlineSource { .. }
            | Self::VolumeSource { .. }
            | Self::CiftiSource { .. }
            | Self::SurfaceSource { .. }
            | Self::ParcellationSource { .. } => Vec::new(),
            Self::LimitStreamlines { .. }
            | Self::GroupSelect { .. }
            | Self::RandomSubset { .. }
            | Self::SphereQuery { .. }
            | Self::RemoveDuplicates { .. }
            | Self::ColorByDirection
            | Self::ColorByGroup
            | Self::ColorByDPV { .. }
            | Self::ColorByDPS { .. }
            | Self::UniformColor { .. }
            | Self::StreamlineDisplay { .. }
            | Self::SaveStreamlines { .. } => vec![PortKind::Streamline],
            Self::BundleSurfaceBuild { .. } => vec![PortKind::Streamline],
            Self::BoundaryFieldBuild { .. } => vec![PortKind::Streamline],
            Self::BundleSurfaceDisplay { .. } => {
                vec![PortKind::BundleSurface, PortKind::BoundaryField]
            }
            Self::BoundaryGlyphDisplay { .. } => vec![PortKind::BoundaryField],
            Self::SurfaceDepthQuery { .. } => vec![PortKind::Streamline, PortKind::Surface],
            Self::CiftiStructure { structure, .. } => match structure {
                CiftiStructure::Subcortical => vec![PortKind::Cifti],
                _ => vec![PortKind::Cifti],
            },
            Self::Merge => {
                vec![PortKind::Streamline, PortKind::Streamline]
            }
            Self::AddGroupsFromParcellation => vec![PortKind::Streamline, PortKind::Parcellation],
            Self::ParcelSelect { .. } | Self::ParcellationDisplay { .. } => {
                vec![PortKind::Parcellation]
            }
            Self::ParcelROI
            | Self::ParcelROA
            | Self::ParcelEnd { .. }
            | Self::ParcelLimiting
            | Self::ParcelTerminative => {
                vec![PortKind::Streamline, PortKind::ParcelSelection]
            }
            Self::ParcelSurfaceBuild => vec![PortKind::ParcelSelection],
            Self::SurfaceProjectionDensity { .. } | Self::SurfaceProjectionMeanDps { .. } => {
                vec![PortKind::Streamline, PortKind::Surface]
            }
            Self::VolumeDisplay { .. } => vec![PortKind::Volume],
            Self::VolumeScalarsDisplay { .. } => vec![PortKind::VolumeScalars],
            Self::SurfaceOverlayStack { layers } => {
                let mut ports = vec![PortKind::Surface];
                ports.extend(std::iter::repeat_n(PortKind::SurfaceScalars, layers.len()));
                ports
            }
            Self::SurfaceDisplay { .. } => vec![PortKind::SurfaceAppearance],
        }
    }

    pub fn outputs(&self) -> Vec<PortKind> {
        match self {
            Self::StreamlineSource { .. }
            | Self::LimitStreamlines { .. }
            | Self::GroupSelect { .. }
            | Self::RandomSubset { .. }
            | Self::SphereQuery { .. }
            | Self::SurfaceDepthQuery { .. }
            | Self::RemoveDuplicates { .. }
            | Self::Merge
            | Self::AddGroupsFromParcellation
            | Self::ParcelROI
            | Self::ParcelROA
            | Self::ParcelEnd { .. }
            | Self::ParcelLimiting
            | Self::ParcelTerminative
            | Self::ColorByDirection
            | Self::ColorByGroup
            | Self::ColorByDPV { .. }
            | Self::ColorByDPS { .. }
            | Self::UniformColor { .. } => vec![PortKind::Streamline],
            Self::VolumeSource { .. } => vec![PortKind::Volume],
            Self::CiftiSource { .. } => vec![PortKind::Cifti],
            Self::SurfaceSource { .. } => vec![PortKind::Surface],
            Self::ParcellationSource { .. } => vec![PortKind::Parcellation],
            Self::ParcelSelect { .. } => vec![PortKind::ParcelSelection],
            Self::SurfaceProjectionDensity { .. } | Self::SurfaceProjectionMeanDps { .. } => {
                vec![PortKind::SurfaceScalars]
            }
            Self::CiftiStructure { structure, .. } => match structure {
                CiftiStructure::Subcortical => vec![PortKind::VolumeScalars],
                _ => vec![PortKind::SurfaceScalars],
            },
            Self::SurfaceOverlayStack { .. } => vec![PortKind::SurfaceAppearance],
            Self::BundleSurfaceBuild { .. } => vec![PortKind::BundleSurface],
            Self::BoundaryFieldBuild { .. } => vec![PortKind::BoundaryField],
            Self::ParcelSurfaceBuild
            | Self::StreamlineDisplay { .. }
            | Self::VolumeDisplay { .. }
            | Self::VolumeScalarsDisplay { .. }
            | Self::SurfaceDisplay { .. }
            | Self::BoundaryGlyphDisplay { .. }
            | Self::ParcellationDisplay { .. }
            | Self::BundleSurfaceDisplay { .. }
            | Self::SaveStreamlines { .. } => Vec::new(),
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
        WorkflowCamera3D, WorkflowDocument, WorkflowOrthoSliceCamera, WorkflowProject,
        WorkflowSliceView3D, WorkflowSliceViewKind, WorkflowSliceViewUi, WorkflowView2DMode,
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
        document.slice_visible_3d = Some([true, false, true]);

        let json = serde_json::to_string(&document).unwrap();
        let restored: WorkflowDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.camera_3d, document.camera_3d);
        assert_eq!(restored.render_3d, document.render_3d);
        assert_eq!(restored.slice_view_3d, document.slice_view_3d);
        assert_eq!(restored.slice_visible_3d, document.slice_visible_3d);
    }

    #[test]
    fn workflow_document_defaults_camera_when_missing() {
        let json = r#"{"next_node_uuid":1,"graph":{"nodes":{},"wires":[]},"assets":[]}"#;
        let restored: WorkflowDocument = serde_json::from_str(json).unwrap();
        assert!(restored.camera_3d.is_none());
        assert!(restored.render_3d.is_none());
        assert!(restored.slice_view_3d.is_none());
        assert!(restored.slice_visible_3d.is_none());
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
            document: super::default_document(),
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
        };

        let json = serde_json::to_string(&project).unwrap();
        let restored: WorkflowProject = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.slice_view_ui, project.slice_view_ui);
    }
}
#[derive(Clone)]
pub struct SurfaceAppearance {
    pub source_id: FileId,
    pub structure: Option<CiftiStructure>,
    pub vertex_rgba: Vec<[f32; 4]>,
    pub legend_labels: Vec<String>,
}

#[derive(Clone, Copy)]
pub struct VolumeScalarDrawPlan {
    pub dims: [usize; 3],
    pub voxel_to_ras: [[f32; 4]; 4],
    pub colormap: VolumeColormap,
    pub opacity: f32,
}

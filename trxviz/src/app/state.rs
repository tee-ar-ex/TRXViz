use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use egui::Rect;
use egui_snarl::Snarl;
use glam::Vec3;
use trxviz_core::data::gifti_data::GiftiSurfaceData;
use trxviz_core::data::loaded_files::{FileId, LoadedCifti, LoadedNifti};
use trxviz_core::data::nifti_data::NiftiVolume;
use trxviz_core::data::odx_data::OdxScene;
use trxviz_core::data::orientation_field::BoundaryContactField;
pub use trxviz_core::lighting::{SceneLightingParams, WorkflowBackground3D, WorkflowRender3D};
use trxviz_core::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use trxviz_core::renderer::glyph_renderer::OdxGlyphResourceKey;
use trxviz_core::renderer::slice_renderer::SliceAxis;
pub use trxviz_core::scene::{
    HeadlessScene as SceneState, LoadedGiftiSurface, LoadedParcellationSource,
    LoadedStreamlineSource,
};

use crate::app::workflow::{
    StreamlineDisplayRuntime, WorkflowCamera3D, WorkflowDocument, WorkflowExecutionCache,
    WorkflowJobKind, WorkflowJobMessage, WorkflowNode, WorkflowNodeUuid, WorkflowRuntime,
    WorkflowSelection, WorkflowSliceView3D, WorkspacePane, default_document,
    default_workspace_tree, snarl_from_graph,
};
use egui_tiles::Tree;
use trx_rs::{Format, VtkCoordinateMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiMode {
    Simple,
    Advanced,
}

impl UiMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Advanced => "Advanced",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SliceViewKind {
    Axial,
    Coronal,
    Sagittal,
}

impl SliceViewKind {
    pub const ALL: [Self; 3] = [Self::Axial, Self::Coronal, Self::Sagittal];

    pub fn label(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
        }
    }

    pub fn slice_axis_index(self) -> Option<usize> {
        match self {
            Self::Axial => Some(0),
            Self::Coronal => Some(1),
            Self::Sagittal => Some(2),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum View2DMode {
    Slice,
    Ortho,
    Lightbox,
}

impl View2DMode {
    pub const ALL: [Self; 3] = [Self::Slice, Self::Ortho, Self::Lightbox];

    pub fn label(self) -> &'static str {
        match self {
            Self::Slice => "Slice",
            Self::Ortho => "Ortho",
            Self::Lightbox => "Lightbox",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportTarget {
    View3D,
    View2D,
    InflatedStage,
}

impl ExportTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::View3D => "3D View",
            Self::View2D => "2D View",
            Self::InflatedStage => "Inflated Stage",
        }
    }
}

pub struct PendingExportRequest {
    pub target: ExportTarget,
    pub path: PathBuf,
    pub scale: u32,
    pub requested_screenshot: bool,
}

#[derive(Clone)]
pub struct ExportDialogState {
    pub open: bool,
    pub target: ExportTarget,
    pub scale: u32,
}

impl Default for ExportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            target: ExportTarget::View3D,
            scale: 2,
        }
    }
}

pub struct View2DState {
    pub window_open: bool,
    pub mode: View2DMode,
    pub single_view: SliceViewKind,
    pub lightbox_axis: SliceViewKind,
    pub lightbox_rows: usize,
    pub lightbox_cols: usize,
    pub active_axis: usize,
    pub ortho_show_row: bool,
}

impl Default for View2DState {
    fn default() -> Self {
        Self {
            window_open: false,
            mode: View2DMode::Ortho,
            single_view: SliceViewKind::Axial,
            lightbox_axis: SliceViewKind::Axial,
            lightbox_rows: 3,
            lightbox_cols: 4,
            active_axis: 0,
            ortho_show_row: true,
        }
    }
}

pub struct PendingFileLoad {
    pub job_id: u64,
    pub label: String,
}

#[derive(Clone, Default)]
pub struct ReferenceAffineDialogState {
    pub open: bool,
    pub source_path: Option<PathBuf>,
    pub reference_path: Option<PathBuf>,
    pub error_msg: Option<String>,
}

impl ReferenceAffineDialogState {
    pub fn open_for_source(&mut self, path: PathBuf) {
        self.open = true;
        self.source_path = Some(path);
        self.reference_path = None;
        self.error_msg = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.source_path = None;
        self.reference_path = None;
        self.error_msg = None;
    }
}

#[derive(Clone)]
pub struct ImportDialogState {
    pub open: bool,
    pub source_path: Option<PathBuf>,
    pub detected_format: Option<Format>,
    pub reference_path: Option<PathBuf>,
    pub vtk_coordinate_mode: VtkCoordinateMode,
    pub error_msg: Option<String>,
}

impl Default for ImportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            source_path: None,
            detected_format: None,
            reference_path: None,
            vtk_coordinate_mode: VtkCoordinateMode::HeaderOrWarn,
            error_msg: None,
        }
    }
}

impl ImportDialogState {
    pub fn open_with_path(&mut self, path: Option<PathBuf>, format: Option<Format>) {
        self.open = true;
        self.source_path = path;
        self.detected_format = format;
        self.reference_path = None;
        self.vtk_coordinate_mode = VtkCoordinateMode::HeaderOrWarn;
        self.error_msg = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.error_msg = None;
    }
}

#[derive(Clone)]
pub struct MergeStreamlineRowState {
    pub source_path: Option<PathBuf>,
    pub detected_format: Option<Format>,
    pub reference_path: Option<PathBuf>,
    pub vtk_coordinate_mode: VtkCoordinateMode,
    pub group_name: String,
}

impl Default for MergeStreamlineRowState {
    fn default() -> Self {
        Self {
            source_path: None,
            detected_format: None,
            reference_path: None,
            vtk_coordinate_mode: VtkCoordinateMode::HeaderOrWarn,
            group_name: String::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct MergeStreamlinesDialogState {
    pub open: bool,
    pub rows: Vec<MergeStreamlineRowState>,
    pub output_path: Option<PathBuf>,
    pub delete_dps: bool,
    pub delete_dpv: bool,
    pub delete_groups: bool,
    pub positions_dtype: Option<FormatlessDType>,
    pub error_msg: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatlessDType {
    Float16,
    Float32,
    Float64,
}

impl FormatlessDType {
    pub const ALL: [Self; 3] = [Self::Float16, Self::Float32, Self::Float64];

    pub fn label(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
}

impl From<FormatlessDType> for trx_rs::DType {
    fn from(value: FormatlessDType) -> Self {
        match value {
            FormatlessDType::Float16 => trx_rs::DType::Float16,
            FormatlessDType::Float32 => trx_rs::DType::Float32,
            FormatlessDType::Float64 => trx_rs::DType::Float64,
        }
    }
}

impl MergeStreamlinesDialogState {
    pub fn open(&mut self) {
        self.open = true;
        self.error_msg = None;
        if self.rows.len() < 2 {
            self.rows.resize_with(2, MergeStreamlineRowState::default);
        }
    }

    pub fn close(&mut self) {
        self.open = false;
        self.error_msg = None;
    }
}

pub struct ViewportState {
    pub camera_3d: OrbitCamera,
    pub slice_cameras: [OrthoSliceCamera; 3],
    pub slice_indices: [usize; 3],
    pub slices_dirty: bool,
    pub volume_center: Vec3,
    pub volume_extent: f32,
    pub slice_visible: [bool; 3],
    pub slice_world_offsets: [f32; 3],
    pub render_3d: WorkflowRender3D,
    pub boundary_field: Option<Arc<BoundaryContactField>>,
    pub boundary_field_revision: u64,
    pub window_3d_open: bool,
    pub window_3d_size: [f32; 2],
    pub inflated_stage_open: bool,
    pub inflated_stage_size: [f32; 2],
    pub inflated_stage_camera: OrbitCamera,
    pub view_2d: View2DState,
    pub window_2d_size: [f32; 2],
    pub export_dialog: ExportDialogState,
    pub pending_export: Option<PendingExportRequest>,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            camera_3d: OrbitCamera::new(Vec3::ZERO, 200.0),
            slice_cameras: [
                OrthoSliceCamera::new(SliceAxis::Axial, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Coronal, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Sagittal, Vec3::ZERO, 200.0),
            ],
            slice_indices: [0; 3],
            slices_dirty: false,
            volume_center: Vec3::ZERO,
            volume_extent: 200.0,
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            render_3d: WorkflowRender3D::default(),
            boundary_field: None,
            boundary_field_revision: 0,
            window_3d_open: false,
            window_3d_size: [1200.0, 900.0],
            inflated_stage_open: false,
            inflated_stage_size: [1200.0, 900.0],
            inflated_stage_camera: OrbitCamera::new(Vec3::ZERO, 250.0),
            view_2d: View2DState::default(),
            window_2d_size: [1400.0, 900.0],
            export_dialog: ExportDialogState::default(),
            pending_export: None,
        }
    }
}

impl ViewportState {
    pub fn workflow_camera_3d(&self) -> WorkflowCamera3D {
        WorkflowCamera3D {
            target: self.viewport_target().to_array(),
            azimuth_deg: self.camera_3d.yaw.to_degrees(),
            elevation_deg: self.camera_3d.pitch.to_degrees(),
            distance: self.camera_3d.distance,
        }
    }

    pub fn apply_workflow_camera_3d(&mut self, camera: WorkflowCamera3D) {
        self.camera_3d.center = Vec3::from_array(camera.target);
        self.camera_3d.yaw = camera.azimuth_deg.to_radians();
        self.camera_3d.pitch = camera.elevation_deg.to_radians();
        self.camera_3d.distance = camera.distance.max(0.1);
    }

    pub fn workflow_render_3d(&self) -> WorkflowRender3D {
        self.render_3d.clone().sanitized()
    }

    pub fn apply_workflow_render_3d(&mut self, render_3d: WorkflowRender3D) {
        self.render_3d = render_3d.sanitized();
    }

    pub fn scene_lighting(&self) -> SceneLightingParams {
        self.render_3d.scene_lighting()
    }

    pub fn workflow_slice_view_3d(&self, nifti_files: &[LoadedNifti]) -> WorkflowSliceView3D {
        WorkflowSliceView3D {
            visible: self.slice_visible,
            positions_ras: [
                self.slice_world_position(nifti_files, 0),
                self.slice_world_position(nifti_files, 1),
                self.slice_world_position(nifti_files, 2),
            ],
        }
    }

    pub fn apply_workflow_slice_view_3d(
        &mut self,
        slice_view: WorkflowSliceView3D,
        nifti_files: &[LoadedNifti],
    ) {
        self.slice_visible = slice_view.visible;
        self.slice_world_offsets = slice_view.positions_ras;
        if let Some(nf) = nifti_files.first() {
            self.slice_indices = [
                nf.volume
                    .nearest_slice_index(0, slice_view.positions_ras[0]),
                nf.volume
                    .nearest_slice_index(1, slice_view.positions_ras[1]),
                nf.volume
                    .nearest_slice_index(2, slice_view.positions_ras[2]),
            ];
            self.slices_dirty = true;
        }
    }

    fn viewport_target(&self) -> Vec3 {
        self.camera_3d.center
    }

    fn gifti_axis_bounds(
        &self,
        gifti_surfaces: &[LoadedGiftiSurface],
        axis_index: usize,
    ) -> Option<(f32, f32)> {
        let mut min_pos = f32::INFINITY;
        let mut max_pos = f32::NEG_INFINITY;

        for surface in gifti_surfaces {
            let (surface_min, surface_max) = match axis_index {
                0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
                1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
                _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
            };
            min_pos = min_pos.min(surface_min);
            max_pos = max_pos.max(surface_max);
        }

        if min_pos.is_finite() && max_pos.is_finite() {
            Some((min_pos, max_pos))
        } else {
            None
        }
    }

    pub fn slice_world_position_for_index(
        &self,
        nifti_files: &[LoadedNifti],
        axis_index: usize,
        index: usize,
    ) -> f32 {
        if let Some(nf) = nifti_files.first() {
            let vol = &nf.volume;
            let idx = index as f32;
            let world = match axis_index {
                0 => vol.voxel_to_world(Vec3::new(0.0, 0.0, idx)),
                1 => vol.voxel_to_world(Vec3::new(0.0, idx, 0.0)),
                2 => vol.voxel_to_world(Vec3::new(idx, 0.0, 0.0)),
                _ => Vec3::ZERO,
            };
            match axis_index {
                0 => world.z,
                1 => world.y,
                2 => world.x,
                _ => 0.0,
            }
        } else {
            self.slice_world_offsets[axis_index]
        }
    }

    pub fn slice_world_position(&self, nifti_files: &[LoadedNifti], axis_index: usize) -> f32 {
        self.slice_world_position_for_index(nifti_files, axis_index, self.slice_indices[axis_index])
    }

    pub fn step_slice(
        &mut self,
        nifti_files: &[LoadedNifti],
        gifti_surfaces: &[LoadedGiftiSurface],
        odx_dims: Option<[u64; 3]>,
        odx_voxel_to_ras: Option<glam::Mat4>,
        axis_index: usize,
        delta: isize,
    ) -> bool {
        if let Some(nf) = nifti_files.first() {
            let vol = &nf.volume;
            let max_idx = match axis_index {
                0 => vol.dims[2].saturating_sub(1),
                1 => vol.dims[1].saturating_sub(1),
                _ => vol.dims[0].saturating_sub(1),
            };
            let new_idx = (self.slice_indices[axis_index] as isize + delta)
                .clamp(0, max_idx as isize) as usize;
            if new_idx != self.slice_indices[axis_index] {
                self.slice_indices[axis_index] = new_idx;
                self.slices_dirty = true;
                return true;
            }
            return false;
        }

        // ODX voxel-grid stepping (same logic as NIfTI but uses ODX dimensions).
        if let Some(dims) = odx_dims {
            let max_idx = match axis_index {
                0 => dims[2].saturating_sub(1) as usize,
                1 => dims[1].saturating_sub(1) as usize,
                _ => dims[0].saturating_sub(1) as usize,
            };
            let new_idx = (self.slice_indices[axis_index] as isize + delta)
                .clamp(0, max_idx as isize) as usize;
            if new_idx != self.slice_indices[axis_index] {
                self.slice_indices[axis_index] = new_idx;
                if let Some(affine) = odx_voxel_to_ras {
                    let v = match axis_index {
                        0 => glam::Vec3::new(0.0, 0.0, new_idx as f32),
                        1 => glam::Vec3::new(0.0, new_idx as f32, 0.0),
                        _ => glam::Vec3::new(new_idx as f32, 0.0, 0.0),
                    };
                    let world = affine.transform_point3(v);
                    self.slice_world_offsets[axis_index] = match axis_index {
                        0 => world.z,
                        1 => world.y,
                        _ => world.x,
                    };
                }
                self.slices_dirty = true;
                return true;
            }
            return false;
        }

        let Some(field) = self.boundary_field.as_ref() else {
            let Some((min_pos, max_pos)) = self.gifti_axis_bounds(gifti_surfaces, axis_index)
            else {
                return false;
            };
            let span = (max_pos - min_pos).abs();
            let step = (span / 256.0).max(0.5);
            let new_pos = (self.slice_world_offsets[axis_index] + delta as f32 * step)
                .clamp(min_pos, max_pos);
            if (new_pos - self.slice_world_offsets[axis_index]).abs() > f32::EPSILON {
                self.slice_world_offsets[axis_index] = new_pos;
                return true;
            }
            return false;
        };

        let voxel = field.grid.voxel_size_mm.max(0.5);
        let dims = field.grid.dims;
        let min_pos = match axis_index {
            0 => field.grid.origin_ras.z + 0.5 * voxel,
            1 => field.grid.origin_ras.y + 0.5 * voxel,
            _ => field.grid.origin_ras.x + 0.5 * voxel,
        };
        let max_pos = match axis_index {
            0 => field.grid.origin_ras.z + (dims[2] as f32 - 0.5) * voxel,
            1 => field.grid.origin_ras.y + (dims[1] as f32 - 0.5) * voxel,
            _ => field.grid.origin_ras.x + (dims[0] as f32 - 0.5) * voxel,
        };
        let new_pos =
            (self.slice_world_offsets[axis_index] + delta as f32 * voxel).clamp(min_pos, max_pos);
        if (new_pos - self.slice_world_offsets[axis_index]).abs() > f32::EPSILON {
            self.slice_world_offsets[axis_index] = new_pos;
            return true;
        }
        false
    }
}

pub struct WorkflowState {
    pub document: WorkflowDocument,
    pub editor_snarl: Snarl<WorkflowNode>,
    pub workspace: Tree<WorkspacePane>,
    pub runtime: WorkflowRuntime,
    pub selection: Option<WorkflowSelection>,
    pub graph_focus_request: Option<Rect>,
    pub display_runtimes: HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    pub next_draw_id: FileId,
    pub project_path: Option<PathBuf>,
    pub node_feedback: HashMap<WorkflowNodeUuid, String>,
    pub execution_cache: WorkflowExecutionCache,
    pub run_expensive_requested: bool,
    pub run_session_active: bool,
    /// Set when a render-only parameter changes (color, opacity, visibility toggle).
    /// Triggers an immediate graph re-evaluation without incrementing document_revision,
    /// so no fingerprints become stale and no expensive jobs are triggered.
    pub render_only_changed: bool,
    /// Set when a background job completes (poll_workflow_job_messages saw a Finished message).
    /// Triggers an immediate interactive re-evaluation so downstream synchronous nodes
    /// (e.g. SurfaceOverlayStack) pick up the new cached data without requiring a second
    /// button press.
    pub pending_job_completion: bool,
    pub document_revision: u64,
    pub last_interactive_revision: u64,
    pub last_settled_revision: u64,
    pub last_runtime_revision: u64,
    pub last_resource_sync_revision: u64,
    pub uploaded_dpv_by_source: HashMap<FileId, (WorkflowNodeUuid, String)>,
    pub uploaded_odx_glyph_resource_key: Option<OdxGlyphResourceKey>,
    pub uploaded_fixel_3d_fingerprint: u64,
    pub uploaded_fixel_2d_fingerprint: u64,
    pub editor_interaction_active: bool,
    pub last_semantic_edit_at: f64,
    pub job_tx: mpsc::Sender<WorkflowJobMessage>,
    pub job_rx: mpsc::Receiver<WorkflowJobMessage>,
    pub jobs_in_flight: HashMap<WorkflowNodeUuid, (WorkflowJobKind, u64)>,
}

impl WorkflowState {
    pub fn new(
        job_tx: mpsc::Sender<WorkflowJobMessage>,
        job_rx: mpsc::Receiver<WorkflowJobMessage>,
    ) -> Self {
        let document = default_document();
        Self {
            editor_snarl: snarl_from_graph(&document.graph),
            document,
            workspace: default_workspace_tree(),
            runtime: WorkflowRuntime::default(),
            selection: None,
            graph_focus_request: None,
            display_runtimes: HashMap::new(),
            next_draw_id: 1_000_000,
            project_path: None,
            node_feedback: HashMap::new(),
            execution_cache: WorkflowExecutionCache::default(),
            run_expensive_requested: false,
            run_session_active: false,
            render_only_changed: false,
            pending_job_completion: false,
            document_revision: 1,
            last_interactive_revision: 0,
            last_settled_revision: 0,
            last_runtime_revision: 0,
            last_resource_sync_revision: 0,
            uploaded_dpv_by_source: HashMap::new(),
            uploaded_odx_glyph_resource_key: None,
            uploaded_fixel_3d_fingerprint: 0,
            uploaded_fixel_2d_fingerprint: 0,
            editor_interaction_active: false,
            last_semantic_edit_at: 0.0,
            job_tx,
            job_rx,
            jobs_in_flight: HashMap::new(),
        }
    }
}

pub enum WorkerMessage {
    TrxLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    ImportedStreamlinesLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    MergedStreamlinesCreated {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    NiftiLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<NiftiVolume, String>,
    },
    CiftiLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedCifti, String>,
    },
    GiftiLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<GiftiSurfaceData, String>,
    },
    ParcellationLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedParcellationSource, String>,
    },
    OdxLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<OdxScene, String>,
    },
}

pub type WorkerSender = mpsc::Sender<WorkerMessage>;
pub type WorkerReceiver = mpsc::Receiver<WorkerMessage>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_dialog_open_with_path_sets_state() {
        let mut state = ImportDialogState::default();
        let path = PathBuf::from("sample.tck.gz");
        state.open_with_path(Some(path.clone()), Some(Format::Tck));
        assert!(state.open);
        assert_eq!(state.source_path.as_deref(), Some(path.as_path()));
        assert_eq!(state.detected_format, Some(Format::Tck));
        assert!(state.reference_path.is_none());
        assert_eq!(state.vtk_coordinate_mode, VtkCoordinateMode::HeaderOrWarn);
        assert!(state.error_msg.is_none());
    }

    #[test]
    fn reference_affine_dialog_open_records_source_and_resets_errors() {
        let mut state = ReferenceAffineDialogState {
            open: false,
            source_path: None,
            reference_path: Some(PathBuf::from("old.nii.gz")),
            error_msg: Some("old error".into()),
        };
        let path = PathBuf::from("sample.fib.gz");
        state.open_for_source(path.clone());
        assert!(state.open);
        assert_eq!(state.source_path.as_deref(), Some(path.as_path()));
        assert!(state.reference_path.is_none());
        assert!(state.error_msg.is_none());
    }

    #[test]
    fn reference_affine_dialog_close_clears_state() {
        let mut state = ReferenceAffineDialogState {
            open: true,
            source_path: Some(PathBuf::from("sample.fib.gz")),
            reference_path: Some(PathBuf::from("ref.nii.gz")),
            error_msg: Some("bad".into()),
        };
        state.close();
        assert!(!state.open);
        assert!(state.source_path.is_none());
        assert!(state.reference_path.is_none());
        assert!(state.error_msg.is_none());
    }

    #[test]
    fn merge_dialog_open_initializes_two_rows() {
        let mut state = MergeStreamlinesDialogState::default();
        state.open();
        assert!(state.open);
        assert_eq!(state.rows.len(), 2);
        assert!(state.output_path.is_none());
        assert!(state.error_msg.is_none());
    }
}

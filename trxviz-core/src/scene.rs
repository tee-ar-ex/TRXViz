//! Shared scene and loaded-asset state used by both the GUI and headless paths.

use std::path::PathBuf;
use std::sync::Arc;

use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{FileId, LoadedNifti, LoadedTrx, StreamlineBacking};
use crate::data::orientation_field::BoundaryContactField;
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::TrxGpuData;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::workflow::{
    LoadedParcellation, StreamlineDisplayRuntime, WorkflowDocument, WorkflowExecutionCache,
    WorkflowNodeUuid, WorkflowRuntime, default_document,
};
use glam::Vec3;
use trx_rs::Format;

#[derive(Clone)]
pub struct LoadedGiftiSurface {
    pub id: usize,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<GiftiSurfaceData>,
    pub visible: bool,
    pub opacity: f32,
    pub color: [f32; 3],
    pub outline_color: [f32; 3],
    pub outline_thickness: f32,
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub surface_gloss: f32,
    pub projection_colormap: SurfaceColormap,
    pub auto_range: bool,
    pub range_min: f32,
    pub range_max: f32,
}

pub struct LoadedStreamlineSource {
    pub data: TrxGpuData,
    pub backing: StreamlineBacking,
}

pub struct LoadedParcellationSource {
    pub data: ParcellationVolume,
    pub label_table_path: Option<PathBuf>,
}

pub struct HeadlessScene {
    pub trx_files: Vec<LoadedTrx>,
    pub nifti_files: Vec<LoadedNifti>,
    pub gifti_surfaces: Vec<LoadedGiftiSurface>,
    pub parcellations: Vec<LoadedParcellation>,
    pub next_file_id: FileId,
    pub volume_center: Vec3,
    pub volume_extent: f32,
    pub slice_indices: [usize; 3],
    pub slice_world_offsets: [f32; 3],
    pub slice_visible: [bool; 3],
    pub boundary_field: Option<Arc<BoundaryContactField>>,
    pub boundary_field_revision: u64,
}

impl Default for HeadlessScene {
    fn default() -> Self {
        Self {
            trx_files: Vec::new(),
            nifti_files: Vec::new(),
            gifti_surfaces: Vec::new(),
            parcellations: Vec::new(),
            next_file_id: 0,
            volume_center: Vec3::ZERO,
            volume_extent: 1.0,
            slice_indices: [0; 3],
            slice_world_offsets: [0.0; 3],
            slice_visible: [true; 3],
            boundary_field: None,
            boundary_field_revision: 0,
        }
    }
}

pub struct HeadlessWorkflowState {
    pub document: WorkflowDocument,
    pub runtime: WorkflowRuntime,
    pub display_runtimes: std::collections::HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    pub next_draw_id: FileId,
    pub execution_cache: WorkflowExecutionCache,
    pub project_path: Option<PathBuf>,
}

impl Default for HeadlessWorkflowState {
    fn default() -> Self {
        Self {
            document: default_document(),
            runtime: WorkflowRuntime::default(),
            display_runtimes: std::collections::HashMap::new(),
            next_draw_id: 1_000_000,
            execution_cache: WorkflowExecutionCache::default(),
            project_path: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct ImportDialogState {
    pub open: bool,
    pub source_path: Option<PathBuf>,
    pub detected_format: Option<Format>,
    pub reference_path: Option<PathBuf>,
    pub error_msg: Option<String>,
}

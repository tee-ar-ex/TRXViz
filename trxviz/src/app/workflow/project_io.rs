use std::path::Path;

use egui_tiles::{Container, Linear, LinearDir, Tile, Tiles, Tree};

use crate::app::state::{SliceViewKind, View2DMode, ViewportState};

use super::{
    WorkflowDocument, WorkflowOrthoSliceCamera, WorkflowProject, WorkflowSliceViewKind,
    WorkflowSliceViewUi, WorkflowView2DMode, load_workflow_project_from_path,
    relativized_document, resolve_document_asset_paths,
};

/// The four panes that make up the advanced-mode workspace layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkspacePane {
    Assets,
    Preview,
    Graph,
    Inspector,
}

/// Default 4-pane layout: Assets | (Preview / Graph) | Inspector.
pub fn default_workspace_tree() -> Tree<WorkspacePane> {
    let mut tiles = Tiles::default();
    let assets = tiles.insert_pane(WorkspacePane::Assets);
    let preview = tiles.insert_pane(WorkspacePane::Preview);
    let graph = tiles.insert_pane(WorkspacePane::Graph);
    let inspector = tiles.insert_pane(WorkspacePane::Inspector);
    let mut center = Linear::new(LinearDir::Vertical, vec![preview, graph]);
    center.shares[preview] = 2.0;
    center.shares[graph] = 1.0;
    let center = tiles.insert_new(Tile::Container(Container::Linear(center)));
    let mut root = Linear::new(LinearDir::Horizontal, vec![assets, center, inspector]);
    root.shares[assets] = 0.85;
    root.shares[center] = 2.5;
    root.shares[inspector] = 0.95;
    let root = tiles.insert_new(Tile::Container(Container::Linear(root)));
    Tree::new("workflow_workspace", root, tiles)
}

/// On-disk project format. `workspace` is at the top level (not nested inside
/// `document`) so that `WorkflowDocument` stays free of GUI dependencies.
#[derive(serde::Serialize, serde::Deserialize)]
struct GuiWorkflowProject {
    #[serde(flatten)]
    project: WorkflowProject,
    #[serde(default = "default_workspace_tree")]
    workspace: Tree<WorkspacePane>,
}

fn into_core_slice_kind(kind: SliceViewKind) -> WorkflowSliceViewKind {
    match kind {
        SliceViewKind::Axial => WorkflowSliceViewKind::Axial,
        SliceViewKind::Coronal => WorkflowSliceViewKind::Coronal,
        SliceViewKind::Sagittal => WorkflowSliceViewKind::Sagittal,
    }
}

fn from_core_slice_kind(kind: WorkflowSliceViewKind) -> SliceViewKind {
    match kind {
        WorkflowSliceViewKind::Axial => SliceViewKind::Axial,
        WorkflowSliceViewKind::Coronal => SliceViewKind::Coronal,
        WorkflowSliceViewKind::Sagittal => SliceViewKind::Sagittal,
    }
}

fn into_core_view_mode(mode: View2DMode) -> WorkflowView2DMode {
    match mode {
        View2DMode::Slice => WorkflowView2DMode::Slice,
        View2DMode::Ortho => WorkflowView2DMode::Ortho,
        View2DMode::Lightbox => WorkflowView2DMode::Lightbox,
    }
}

fn from_core_view_mode(mode: WorkflowView2DMode) -> View2DMode {
    match mode {
        WorkflowView2DMode::Slice => View2DMode::Slice,
        WorkflowView2DMode::Ortho => View2DMode::Ortho,
        WorkflowView2DMode::Lightbox => View2DMode::Lightbox,
    }
}

pub fn capture_gui_slice_view_state(viewport: &ViewportState) -> WorkflowSliceViewUi {
    WorkflowSliceViewUi {
        mode: into_core_view_mode(viewport.view_2d.mode),
        single_view: into_core_slice_kind(viewport.view_2d.single_view),
        lightbox_axis: into_core_slice_kind(viewport.view_2d.lightbox_axis),
        lightbox_rows: viewport.view_2d.lightbox_rows,
        lightbox_cols: viewport.view_2d.lightbox_cols,
        active_axis: viewport.view_2d.active_axis,
        ortho_show_row: viewport.view_2d.ortho_show_row,
        slice_cameras: std::array::from_fn(|idx| {
            let camera = &viewport.slice_cameras[idx];
            WorkflowOrthoSliceCamera {
                center: camera.center,
                half_extent: camera.half_extent,
                rotation: camera.rotation,
            }
        }),
    }
}

pub fn apply_gui_slice_view_state(viewport: &mut ViewportState, state: WorkflowSliceViewUi) {
    viewport.view_2d.mode = from_core_view_mode(state.mode);
    viewport.view_2d.single_view = from_core_slice_kind(state.single_view);
    viewport.view_2d.lightbox_axis = from_core_slice_kind(state.lightbox_axis);
    viewport.view_2d.lightbox_rows = state.lightbox_rows.max(1);
    viewport.view_2d.lightbox_cols = state.lightbox_cols.max(1);
    viewport.view_2d.active_axis = state.active_axis.min(2);
    viewport.view_2d.ortho_show_row = state.ortho_show_row;
    for (camera, saved) in viewport.slice_cameras.iter_mut().zip(state.slice_cameras) {
        camera.center = saved.center;
        camera.half_extent = saved.half_extent.max(0.001);
        camera.rotation = saved.rotation;
    }
}

/// Load a project file. Returns `(WorkflowProject, workspace_tree)`.
/// Old files that have `workspace` nested inside `document` are handled by
/// trying the new format first, then falling back.
pub fn gui_load_project(
    path: &Path,
) -> Result<(WorkflowProject, Tree<WorkspacePane>, Option<WorkflowSliceViewUi>), String> {
    let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;

    // New format: workspace at top level.
    if let Ok(gui) = serde_json::from_str::<GuiWorkflowProject>(&contents) {
        let mut project = gui.project;
        resolve_document_asset_paths(&mut project.document, path);
        return Ok((project.clone(), gui.workspace, project.slice_view_ui));
    }

    // Fall back to core loader (handles bare WorkflowDocument too).
    let mut project = load_workflow_project_from_path(path)?;
    resolve_document_asset_paths(&mut project.document, path);
    Ok((project, default_workspace_tree(), None))
}

/// Save a project file with the workspace tree at the top level.
pub fn gui_save_project(
    document: &WorkflowDocument,
    workspace: &Tree<WorkspacePane>,
    slice_view_ui: WorkflowSliceViewUi,
    path: &Path,
) -> Result<(), String> {
    let document = relativized_document(document, path);
    // Serialize via the core helper (which wraps in WorkflowProject), then
    // inject the workspace into the resulting JSON value.
    let project = serde_json::to_value(WorkflowProject {
        version: 1,
        document,
        slice_view_ui: Some(slice_view_ui),
    })
    .map_err(|e| e.to_string())?;
    let workspace_val = serde_json::to_value(workspace).map_err(|e| e.to_string())?;

    let mut obj = match project {
        serde_json::Value::Object(map) => map,
        _ => return Err("unexpected serialization shape".to_string()),
    };
    obj.insert("workspace".to_string(), workspace_val);

    let json = serde_json::to_string_pretty(&obj).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

use std::path::Path;

use egui_tiles::{Container, Linear, LinearDir, Tile, Tiles, Tree};

use super::{
    WorkflowDocument, WorkflowProject, load_workflow_project_from_path,
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

/// Load a project file. Returns `(WorkflowProject, workspace_tree)`.
/// Old files that have `workspace` nested inside `document` are handled by
/// trying the new format first, then falling back.
pub fn gui_load_project(
    path: &Path,
) -> Result<(WorkflowProject, Tree<WorkspacePane>), String> {
    let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;

    // New format: workspace at top level.
    if let Ok(gui) = serde_json::from_str::<GuiWorkflowProject>(&contents) {
        let mut project = gui.project;
        resolve_document_asset_paths(&mut project.document, path);
        return Ok((project, gui.workspace));
    }

    // Fall back to core loader (handles bare WorkflowDocument too).
    let mut project = load_workflow_project_from_path(path)?;
    resolve_document_asset_paths(&mut project.document, path);
    Ok((project, default_workspace_tree()))
}

/// Save a project file with the workspace tree at the top level.
pub fn gui_save_project(
    document: &WorkflowDocument,
    workspace: &Tree<WorkspacePane>,
    path: &Path,
) -> Result<(), String> {
    let document = relativized_document(document, path);
    // Serialize via the core helper (which wraps in WorkflowProject), then
    // inject the workspace into the resulting JSON value.
    let project = serde_json::to_value(WorkflowProject {
        version: 1,
        document,
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

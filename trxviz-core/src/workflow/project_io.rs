use std::path::{Path, PathBuf};

use super::*;

pub fn save_workflow_project_to_path(
    document: &WorkflowDocument,
    path: &Path,
) -> Result<(), String> {
    let project = WorkflowProject {
        version: 1,
        document: document.clone(),
        slice_view_ui: None,
    };
    let json = serde_json::to_string_pretty(&project).map_err(|err| err.to_string())?;
    std::fs::write(path, json).map_err(|err| err.to_string())
}

pub fn load_workflow_project_from_path(path: &Path) -> Result<WorkflowProject, String> {
    let contents = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    // Preferred: current project format.
    if let Ok(project) = serde_json::from_str::<WorkflowProject>(&contents) {
        return Ok(project);
    }
    // Bare (unversioned) document in the current format.
    serde_json::from_str::<WorkflowDocument>(&contents)
        .map(|document| WorkflowProject {
            version: 1,
            document,
            slice_view_ui: None,
        })
        .map_err(|err| err.to_string())
}

fn asset_path_mut(asset: &mut WorkflowAssetDocument) -> &mut PathBuf {
    match asset {
        WorkflowAssetDocument::Streamlines { path, .. }
        | WorkflowAssetDocument::Volume { path, .. }
        | WorkflowAssetDocument::Cifti { path, .. }
        | WorkflowAssetDocument::Surface { path, .. }
        | WorkflowAssetDocument::Parcellation { path, .. } => path,
    }
}

pub fn relativized_document(document: &WorkflowDocument, project_path: &Path) -> WorkflowDocument {
    let mut document = document.clone();
    let Some(base_dir) = project_path.parent() else {
        return document;
    };
    for asset in &mut document.assets {
        let path = asset_path_mut(asset);
        if path.is_absolute()
            && let Ok(relative) = path.strip_prefix(base_dir)
        {
            *path = relative.to_path_buf();
        }
    }
    document
}

pub fn resolve_document_asset_paths(document: &mut WorkflowDocument, project_path: &Path) {
    let Some(base_dir) = project_path.parent() else {
        return;
    };
    for asset in &mut document.assets {
        let path = asset_path_mut(asset);
        if path.is_relative() {
            *path = base_dir.join(&*path);
        }
    }
}

use std::path::{Path, PathBuf};

use super::*;

pub fn save_workflow_project_to_path(
    document: &WorkflowDocument,
    path: &Path,
) -> WorkflowResult<()> {
    let project = WorkflowProject {
        version: 1,
        document: document.clone(),
        slice_view_ui: None,
    };
    let json = serde_json::to_string_pretty(&project)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_workflow_project_from_path(path: &Path) -> WorkflowResult<WorkflowProject> {
    let contents = std::fs::read_to_string(path)?;
    let value = serde_json::from_str::<serde_json::Value>(&contents)?;
    load_workflow_project_from_value(value)
}

fn load_workflow_project_from_value(value: serde_json::Value) -> WorkflowResult<WorkflowProject> {
    if let Ok(mut project) = serde_json::from_value::<WorkflowProject>(value.clone()) {
        ensure_node_uuids(&mut project.document);
        return Ok(project);
    }

    if let Some(project_value) = value.get("project")
        && let Ok(mut project) = serde_json::from_value::<WorkflowProject>(project_value.clone())
    {
        ensure_node_uuids(&mut project.document);
        return Ok(project);
    }

    let mut document = serde_json::from_value::<WorkflowDocument>(value)?;
    ensure_node_uuids(&mut document);
    Ok(WorkflowProject {
        version: 1,
        document,
        slice_view_ui: None,
    })
}

fn asset_path_mut(asset: &mut WorkflowAssetDocument) -> &mut PathBuf {
    match asset {
        WorkflowAssetDocument::Streamlines { path, .. }
        | WorkflowAssetDocument::Volume { path, .. }
        | WorkflowAssetDocument::Cifti { path, .. }
        | WorkflowAssetDocument::Surface { path, .. }
        | WorkflowAssetDocument::Parcellation { path, .. }
        | WorkflowAssetDocument::Odx { path, .. } => path,
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

#[cfg(test)]
mod tests {
    use super::load_workflow_project_from_value;

    #[test]
    fn load_workflow_project_accepts_nested_gui_wrapper() {
        let value = serde_json::json!({
            "project": {
                "version": 1,
                "document": {
                    "graph": { "nodes": {}, "wires": [] },
                    "assets": []
                }
            },
            "workspace": {}
        });

        let project = load_workflow_project_from_value(value).unwrap();
        assert_eq!(project.version, 1);
        assert_eq!(project.document.next_node_uuid, 1);
        assert!(project.document.assets.is_empty());
    }
}

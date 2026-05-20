use std::collections::{HashMap, HashSet};

use crate::data::loaded_files::FileId;
use crate::error::{WorkflowError, WorkflowResult};

use super::{WorkflowAssetDocument, WorkflowDocument, WorkflowNodeKind, WorkflowNodeUuid};

#[derive(Clone, Debug, Default)]
pub struct SimpleWorkflowBindings {
    pub streamline: HashMap<FileId, SimpleStreamlineBinding>,
    pub volume: HashMap<FileId, SimpleDisplayBinding>,
    pub surface: HashMap<FileId, SimpleSurfaceBinding>,
    pub parcellation: HashMap<FileId, SimpleDisplayBinding>,
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleStreamlineBinding {
    pub display: WorkflowNodeUuid,
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleDisplayBinding {
    pub display: WorkflowNodeUuid,
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleSurfaceBinding {
    pub display: WorkflowNodeUuid,
    pub overlay_stack: Option<WorkflowNodeUuid>,
}

#[derive(Clone, Debug, Default)]
pub struct WorkflowEditability {
    pub bindings: SimpleWorkflowBindings,
    pub read_only_reasons: HashMap<FileId, String>,
}

impl WorkflowEditability {
    pub fn has_read_only_assets(&self) -> bool {
        !self.read_only_reasons.is_empty()
    }

    pub fn first_reason(&self) -> Option<&str> {
        self.read_only_reasons.values().next().map(String::as_str)
    }

    pub fn reason_for(&self, asset_id: FileId) -> Option<&str> {
        self.read_only_reasons.get(&asset_id).map(String::as_str)
    }
}

pub fn classify_workflow_editability(document: &WorkflowDocument) -> WorkflowEditability {
    let mut outgoing = HashMap::<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>::new();
    let mut kinds = HashMap::<WorkflowNodeUuid, &WorkflowNodeKind>::new();

    for (uuid, node) in document.graph.nodes() {
        kinds.insert(uuid, &node.op);
    }

    for wire in document.graph.wires() {
        outgoing
            .entry(wire.from.node)
            .or_default()
            .push(wire.to.node);
    }

    let mut editability = WorkflowEditability::default();

    for asset in &document.assets {
        let result = match asset {
            WorkflowAssetDocument::Streamlines { id, .. } => {
                match_streamline_asset(*id, &kinds, &outgoing).map(|binding| {
                    editability.bindings.streamline.insert(*id, binding);
                })
            }
            WorkflowAssetDocument::Volume { id, .. } => match_simple_display_asset(
                *id,
                &kinds,
                &outgoing,
                is_volume_source,
                is_volume_display,
            )
            .map(|binding| {
                editability.bindings.volume.insert(*id, binding);
            }),
            WorkflowAssetDocument::Cifti { .. } => Err(WorkflowError::Evaluation(
                "CIFTI workflow branches are only editable in Advanced mode.".to_string(),
            )),
            WorkflowAssetDocument::Surface { id, .. } => {
                match_surface_asset(*id, &kinds, &outgoing).map(|binding| {
                    editability.bindings.surface.insert(*id, binding);
                })
            }
            WorkflowAssetDocument::Parcellation { id, .. } => match_simple_display_asset(
                *id,
                &kinds,
                &outgoing,
                is_parcellation_source,
                is_parcellation_display,
            )
            .map(|binding| {
                editability.bindings.parcellation.insert(*id, binding);
            }),
            WorkflowAssetDocument::Odx { .. } => Err(WorkflowError::Evaluation(
                "ODX workflow branches are only editable in Advanced mode.".to_string(),
            )),
        };

        if let Err(reason) = result {
            editability
                .read_only_reasons
                .insert(workflow_asset_id(asset), reason.to_string());
        }
    }

    editability
}

fn match_streamline_asset(
    asset_id: FileId,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
) -> WorkflowResult<SimpleStreamlineBinding> {
    let source = find_single_node(kinds, |kind| {
        matches!(
            kind,
            WorkflowNodeKind::StreamlineSource { source_id } if *source_id == asset_id
        )
    })
    .ok_or_else(|| {
        WorkflowError::Evaluation(format!(
            "Streamline asset {asset_id} is missing its source node."
        ))
    })?;
    let display = find_unique_reachable_display(source, kinds, outgoing, |kind| {
        matches!(kind, WorkflowNodeKind::StreamlineDisplay { .. })
    })?;

    Ok(SimpleStreamlineBinding { display })
}

fn match_simple_display_asset(
    asset_id: FileId,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    is_source: impl Fn(&WorkflowNodeKind, FileId) -> bool,
    is_display: impl Fn(&WorkflowNodeKind) -> bool,
) -> WorkflowResult<SimpleDisplayBinding> {
    let source = find_single_node(kinds, |kind| is_source(kind, asset_id)).ok_or_else(|| {
        WorkflowError::Evaluation(format!("Asset {asset_id} is missing its source node."))
    })?;
    let display = find_unique_reachable_display(source, kinds, outgoing, is_display)?;

    Ok(SimpleDisplayBinding { display })
}

fn match_surface_asset(
    asset_id: FileId,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
) -> WorkflowResult<SimpleSurfaceBinding> {
    let source =
        find_single_node(kinds, |kind| is_surface_source(kind, asset_id)).ok_or_else(|| {
            WorkflowError::Evaluation(format!("Asset {asset_id} is missing its source node."))
        })?;
    let display = find_unique_reachable_display(source, kinds, outgoing, is_surface_display)?;
    let overlay_stack = find_unique_reachable_optional_node(source, kinds, outgoing, |kind| {
        matches!(kind, WorkflowNodeKind::SurfaceOverlayStack { .. })
    })?;

    Ok(SimpleSurfaceBinding {
        display,
        overlay_stack,
    })
}

fn find_single_node(
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    predicate: impl Fn(&WorkflowNodeKind) -> bool,
) -> Option<WorkflowNodeUuid> {
    let mut found = None;
    for (&uuid, kind) in kinds {
        if predicate(kind) {
            if found.is_some() {
                return None;
            }
            found = Some(uuid);
        }
    }
    found
}

fn find_unique_reachable_display(
    source: WorkflowNodeUuid,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    predicate: impl Fn(&WorkflowNodeKind) -> bool,
) -> WorkflowResult<WorkflowNodeUuid> {
    let mut stack = vec![source];
    let mut visited = HashSet::from([source]);
    let mut matches = Vec::new();

    while let Some(current) = stack.pop() {
        let Some(children) = outgoing.get(&current) else {
            continue;
        };
        for &child in children {
            if !visited.insert(child) {
                continue;
            }
            if let Some(kind) = kinds.get(&child) {
                if predicate(kind) {
                    if outgoing
                        .get(&child)
                        .is_some_and(|children| !children.is_empty())
                    {
                        return Err(WorkflowError::Evaluation(
                            "This project routes a display node into additional workflow nodes, which requires Advanced mode."
                                .to_string(),
                        ));
                    }
                    matches.push(child);
                }
                stack.push(child);
            }
        }
    }

    match matches.len() {
        1 => Ok(matches[0]),
        0 => Err(WorkflowError::Evaluation(
            "This project does not expose a unique terminal display node for this asset in Simple mode."
                .to_string(),
        )),
        _ => Err(WorkflowError::Evaluation(
            "This asset feeds multiple display branches; edit it in Advanced mode."
                .to_string(),
        )),
    }
}

fn find_unique_reachable_optional_node(
    source: WorkflowNodeUuid,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    predicate: impl Fn(&WorkflowNodeKind) -> bool,
) -> WorkflowResult<Option<WorkflowNodeUuid>> {
    let mut stack = vec![source];
    let mut visited = HashSet::from([source]);
    let mut matches = Vec::new();

    while let Some(current) = stack.pop() {
        let Some(children) = outgoing.get(&current) else {
            continue;
        };
        for &child in children {
            if !visited.insert(child) {
                continue;
            }
            if let Some(kind) = kinds.get(&child) {
                if predicate(kind) {
                    matches.push(child);
                }
                stack.push(child);
            }
        }
    }

    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches[0])),
        _ => Err(WorkflowError::Evaluation(
            "This asset feeds multiple surface appearance branches; edit it in Advanced mode."
                .to_string(),
        )),
    }
}

fn workflow_asset_id(asset: &WorkflowAssetDocument) -> FileId {
    match asset {
        WorkflowAssetDocument::Streamlines { id, .. }
        | WorkflowAssetDocument::Volume { id, .. }
        | WorkflowAssetDocument::Cifti { id, .. }
        | WorkflowAssetDocument::Surface { id, .. }
        | WorkflowAssetDocument::Parcellation { id, .. }
        | WorkflowAssetDocument::Odx { id, .. } => *id,
    }
}

fn is_volume_source(kind: &WorkflowNodeKind, asset_id: FileId) -> bool {
    matches!(
        kind,
        WorkflowNodeKind::VolumeSource { source_id } if *source_id == asset_id
    )
}

fn is_volume_display(kind: &WorkflowNodeKind) -> bool {
    matches!(kind, WorkflowNodeKind::VolumeDisplay { .. })
}

fn is_surface_source(kind: &WorkflowNodeKind, asset_id: FileId) -> bool {
    matches!(
        kind,
        WorkflowNodeKind::SurfaceSource { source_id } if *source_id == asset_id
    )
}

fn is_surface_display(kind: &WorkflowNodeKind) -> bool {
    matches!(kind, WorkflowNodeKind::SurfaceDisplay { .. })
}

fn is_parcellation_source(kind: &WorkflowNodeKind, asset_id: FileId) -> bool {
    matches!(
        kind,
        WorkflowNodeKind::ParcellationSource { source_id } if *source_id == asset_id
    )
}

fn is_parcellation_display(kind: &WorkflowNodeKind) -> bool {
    matches!(kind, WorkflowNodeKind::ParcellationDisplay { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{
        GraphPos, InPort, OutPort, add_default_nodes_for_asset, default_document, make_node,
    };
    use std::path::PathBuf;

    #[test]
    fn default_document_is_simple_editable() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Streamlines {
            id: 1,
            path: PathBuf::from("sample.trx"),
            imported: false,
        };
        document.assets.push(asset.clone());
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, Some(10_000), false);

        let editability = classify_workflow_editability(&document);
        let binding = editability.bindings.streamline.get(&1).copied();
        assert!(binding.is_some());
        assert!(!editability.has_read_only_assets());
    }

    #[test]
    fn default_surface_branch_is_simple_editable() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Surface {
            id: 7,
            path: PathBuf::from("surface.gii"),
        };
        document.assets.push(asset.clone());
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None, false);

        let editability = classify_workflow_editability(&document);
        let binding = editability.bindings.surface.get(&7).copied();
        assert!(binding.is_some());
        assert!(!editability.has_read_only_assets());
    }

    #[test]
    fn mixed_project_keeps_simple_bindings_for_compatible_assets() {
        let mut document = default_document();
        let streamline = WorkflowAssetDocument::Streamlines {
            id: 1,
            path: PathBuf::from("sample.trx"),
            imported: false,
        };
        let cifti = WorkflowAssetDocument::Cifti {
            id: 2,
            path: PathBuf::from("sample.dscalar.nii"),
            intent: crate::data::cifti::CiftiIntent::DenseScalar,
        };
        document.assets.push(streamline.clone());
        document.assets.push(cifti);
        add_default_nodes_for_asset(&mut document, &streamline, GraphPos::ZERO, Some(10_000), false);

        let editability = classify_workflow_editability(&document);
        assert!(editability.bindings.streamline.contains_key(&1));
        assert_eq!(
            editability.reason_for(2),
            Some("CIFTI workflow branches are only editable in Advanced mode.")
        );
    }

    #[test]
    fn multiple_display_endpoints_make_asset_read_only() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Volume {
            id: 3,
            path: PathBuf::from("volume.nii.gz"),
        };
        document.assets.push(asset.clone());
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None, false);

        let source = document
            .graph
            .nodes()
            .find_map(|(id, node)| {
                matches!(
                    node.op,
                    WorkflowNodeKind::VolumeSource { source_id } if source_id == 3
                )
                .then_some(id)
            })
            .unwrap();
        let extra_display = make_node(
            &mut document,
            WorkflowNodeKind::VolumeDisplay {
                colormap: crate::data::loaded_files::VolumeColormap::Hot,
                opacity: 0.5,
                window_center: 0.5,
                window_width: 1.0,
            },
            GraphPos::new(200.0, 200.0),
        );
        document.graph.connect(
            OutPort {
                node: source,
                output: 0,
            },
            InPort {
                node: extra_display,
                input: 0,
            },
        );

        let editability = classify_workflow_editability(&document);
        assert!(!editability.bindings.volume.contains_key(&3));
        assert_eq!(
            editability.reason_for(3),
            Some("This asset feeds multiple display branches; edit it in Advanced mode.")
        );
    }
}

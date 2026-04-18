use std::collections::{HashMap, HashSet};

use crate::data::loaded_files::FileId;

use super::{WorkflowAssetDocument, WorkflowDocument, WorkflowNodeKind, WorkflowNodeUuid};

#[derive(Clone, Debug, Default)]
pub struct SimpleWorkflowBindings {
    pub streamline: HashMap<FileId, SimpleStreamlineBinding>,
    pub volume: HashMap<FileId, SimpleDisplayBinding>,
    pub surface: HashMap<FileId, SimpleDisplayBinding>,
    pub parcellation: HashMap<FileId, SimpleDisplayBinding>,
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleStreamlineBinding {
    pub source: WorkflowNodeUuid,
    pub group_select: WorkflowNodeUuid,
    pub limit: WorkflowNodeUuid,
    pub color: WorkflowNodeUuid,
    pub display: WorkflowNodeUuid,
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleDisplayBinding {
    pub display: WorkflowNodeUuid,
}

#[derive(Clone, Debug)]
pub enum WorkflowEditability {
    Simple(SimpleWorkflowBindings),
    AdvancedOnly { reason: String },
}

impl WorkflowEditability {
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Simple(_) => None,
            Self::AdvancedOnly { reason } => Some(reason),
        }
    }
}

pub fn classify_workflow_editability(document: &WorkflowDocument) -> WorkflowEditability {
    let mut incoming = HashMap::<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>::new();
    let mut outgoing = HashMap::<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>::new();
    let mut kinds = HashMap::<WorkflowNodeUuid, &WorkflowNodeKind>::new();

    for (uuid, node) in document.graph.nodes() {
        kinds.insert(uuid, &node.kind);
    }

    for wire in document.graph.wires() {
        outgoing
            .entry(wire.from.node)
            .or_default()
            .push(wire.to.node);
        incoming
            .entry(wire.to.node)
            .or_default()
            .push(wire.from.node);
    }

    let mut bindings = SimpleWorkflowBindings::default();
    let mut used = HashSet::<WorkflowNodeUuid>::new();

    for asset in &document.assets {
        let result = match asset {
            WorkflowAssetDocument::Streamlines { id, .. } => match_streamline_asset(
                *id, &kinds, &incoming, &outgoing, &mut used,
            )
            .map(|binding| {
                bindings.streamline.insert(*id, binding);
            }),
            WorkflowAssetDocument::Volume { id, .. } => match_simple_display_asset(
                *id,
                &kinds,
                &incoming,
                &outgoing,
                &mut used,
                is_volume_source,
                is_volume_display,
            )
            .map(|binding| {
                bindings.volume.insert(*id, binding);
            }),
            WorkflowAssetDocument::Cifti { .. } => {
                Err("CIFTI workflow branches are only editable in Advanced mode.".to_string())
            }
            WorkflowAssetDocument::Surface { id, .. } => match_simple_display_asset(
                *id,
                &kinds,
                &incoming,
                &outgoing,
                &mut used,
                is_surface_source,
                is_surface_display,
            )
            .map(|binding| {
                bindings.surface.insert(*id, binding);
            }),
            WorkflowAssetDocument::Parcellation { id, .. } => match_simple_display_asset(
                *id,
                &kinds,
                &incoming,
                &outgoing,
                &mut used,
                is_parcellation_source,
                is_parcellation_display,
            )
            .map(|binding| {
                bindings.parcellation.insert(*id, binding);
            }),
            WorkflowAssetDocument::Odx { .. } => {
                Err("ODX workflow branches are only editable in Advanced mode.".to_string())
            }
        };

        if let Err(reason) = result {
            return WorkflowEditability::AdvancedOnly { reason };
        }
    }

    if used.len() != kinds.len() {
        return WorkflowEditability::AdvancedOnly {
            reason: "This project contains extra workflow nodes that are only editable in Advanced mode.".to_string(),
        };
    }

    WorkflowEditability::Simple(bindings)
}

fn match_streamline_asset(
    asset_id: FileId,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    incoming: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    used: &mut HashSet<WorkflowNodeUuid>,
) -> Result<SimpleStreamlineBinding, String> {
    let source = find_single_node(kinds, |kind| {
        matches!(
            kind,
            WorkflowNodeKind::StreamlineSource { source_id } if *source_id == asset_id
        )
    })
    .ok_or_else(|| format!("Streamline asset {asset_id} is missing its source node."))?;

    let group_select = expect_next(source, outgoing, incoming, kinds, |kind| {
        matches!(kind, WorkflowNodeKind::GroupSelect { .. })
    })
    .ok_or_else(|| {
        "This project does not match the default Simple streamline chain (source -> group -> limit -> color -> display).".to_string()
    })?;

    let limit = expect_next(group_select, outgoing, incoming, kinds, |kind| {
        matches!(kind, WorkflowNodeKind::LimitStreamlines { .. })
    })
    .ok_or_else(|| {
        "This project does not match the default Simple streamline chain (source -> group -> limit -> color -> display).".to_string()
    })?;

    let color = expect_next(limit, outgoing, incoming, kinds, is_simple_streamline_color)
        .ok_or_else(|| {
            "This project does not match the default Simple streamline chain (source -> group -> limit -> color -> display).".to_string()
        })?;

    let display = expect_next(color, outgoing, incoming, kinds, |kind| {
        matches!(kind, WorkflowNodeKind::StreamlineDisplay { .. })
    })
    .ok_or_else(|| {
        "This project does not match the default Simple streamline chain (source -> group -> limit -> color -> display).".to_string()
    })?;

    if !incoming.get(&source).map_or(true, Vec::is_empty)
        || !outgoing.get(&display).map_or(true, Vec::is_empty)
    {
        return Err(
            "This project branches or feeds the default streamline view in a way that requires Advanced mode."
                .to_string(),
        );
    }

    used.extend([source, group_select, limit, color, display]);

    Ok(SimpleStreamlineBinding {
        source,
        group_select,
        limit,
        color,
        display,
    })
}

fn match_simple_display_asset(
    asset_id: FileId,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    incoming: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    used: &mut HashSet<WorkflowNodeUuid>,
    is_source: impl Fn(&WorkflowNodeKind, FileId) -> bool,
    is_display: impl Fn(&WorkflowNodeKind) -> bool,
) -> Result<SimpleDisplayBinding, String> {
    let source = find_single_node(kinds, |kind| is_source(kind, asset_id))
        .ok_or_else(|| format!("Asset {asset_id} is missing its source node."))?;
    let display = expect_next(source, outgoing, incoming, kinds, is_display).ok_or_else(|| {
        "This project does not match the default Simple asset chain (source -> display)."
            .to_string()
    })?;

    if !incoming.get(&source).map_or(true, Vec::is_empty)
        || !outgoing.get(&display).map_or(true, Vec::is_empty)
    {
        return Err(
            "This project branches or feeds a default asset view in a way that requires Advanced mode."
                .to_string(),
        );
    }

    used.extend([source, display]);

    Ok(SimpleDisplayBinding { display })
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

fn expect_next(
    current: WorkflowNodeUuid,
    outgoing: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    incoming: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    kinds: &HashMap<WorkflowNodeUuid, &WorkflowNodeKind>,
    predicate: impl Fn(&WorkflowNodeKind) -> bool,
) -> Option<WorkflowNodeUuid> {
    let next = match outgoing.get(&current) {
        Some(children) if children.len() == 1 => children[0],
        _ => return None,
    };
    if incoming.get(&next).map_or(0, Vec::len) != 1 {
        return None;
    }
    predicate(kinds.get(&next)?).then_some(next)
}

fn is_simple_streamline_color(kind: &WorkflowNodeKind) -> bool {
    matches!(
        kind,
        WorkflowNodeKind::ColorByDirection
            | WorkflowNodeKind::ColorByGroup
            | WorkflowNodeKind::UniformColor { .. }
    )
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
    use crate::workflow::{GraphPos, add_default_nodes_for_asset, default_document};
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
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, Some(10_000));

        let editability = classify_workflow_editability(&document);
        assert!(matches!(editability, WorkflowEditability::Simple(_)));
    }
}

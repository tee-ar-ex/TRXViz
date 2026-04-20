use crate::data::loaded_files::FileId;

use super::seed::{make_node, relayout_connected_component};
use super::*;

pub fn set_default_odx_fixel_3d_visibility(
    document: &mut WorkflowDocument,
    asset_id: FileId,
    visible: bool,
) -> bool {
    let source_nodes: Vec<_> = document
        .graph
        .nodes()
        .filter_map(|(uuid, node)| match node.op {
            WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
            _ => None,
        })
        .collect();

    let targets: Vec<_> = document
        .graph
        .wires()
        .filter(|wire| source_nodes.contains(&wire.from.node))
        .map(|wire| wire.to.node)
        .collect();

    let mut changed = false;
    for uuid in targets {
        let Some(node) = document.graph.get_mut(uuid) else {
            continue;
        };
        let WorkflowNodeKind::Fixel3DDisplay {
            visible: node_visible,
            ..
        } = &mut node.op
        else {
            continue;
        };
        if *node_visible != visible {
            *node_visible = visible;
            changed = true;
        }
    }

    changed
}

const ODX_PREFERRED_DPF_NAMES: &[&str] = &["amplitude", "afd", "fd", "qa"];
const ODX_PREFERRED_DPV_NAMES: &[&str] = &["anisotropic_power", "gfa", "dti_fa", "fd", "afd", "qa"];

fn choose_default_odx_dpf_name(dpf_names: &[String]) -> Option<String> {
    if dpf_names.is_empty() {
        return None;
    }
    for preferred in ODX_PREFERRED_DPF_NAMES {
        if dpf_names.iter().any(|name| name == preferred) {
            return Some((*preferred).to_string());
        }
    }
    let mut sorted = dpf_names.to_vec();
    sorted.sort();
    sorted.into_iter().next()
}

fn choose_default_odx_dpv_name(dpv_names: &[String]) -> Option<String> {
    if dpv_names.is_empty() {
        return None;
    }
    for preferred in ODX_PREFERRED_DPV_NAMES {
        if dpv_names.iter().any(|name| name == preferred) {
            return Some((*preferred).to_string());
        }
    }
    let mut sorted = dpv_names.to_vec();
    sorted.sort();
    sorted.into_iter().next()
}

pub fn set_default_odx_volume_dpv(
    document: &mut WorkflowDocument,
    asset_id: FileId,
    dpv_names: &[String],
) -> bool {
    let Some(default_name) = choose_default_odx_dpv_name(dpv_names) else {
        return false;
    };

    let source_nodes: Vec<_> = document
        .graph
        .nodes()
        .filter_map(|(uuid, node)| match node.op {
            WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
            _ => None,
        })
        .collect();

    let targets: Vec<_> = document
        .graph
        .wires()
        .filter(|wire| source_nodes.contains(&wire.from.node) && wire.from.output == 2)
        .map(|wire| wire.to.node)
        .collect();

    let mut changed = false;
    for uuid in targets {
        let Some(node) = document.graph.get_mut(uuid) else {
            continue;
        };
        let WorkflowNodeKind::OdxVolumeSelect { dpv_name } = &mut node.op else {
            continue;
        };
        if dpv_name.is_empty() {
            *dpv_name = default_name.clone();
            changed = true;
        }
    }

    changed
}

pub fn set_default_odx_fixel_dpf(
    document: &mut WorkflowDocument,
    asset_id: FileId,
    dpf_names: &[String],
) -> bool {
    let Some(default_name) = choose_default_odx_dpf_name(dpf_names) else {
        return false;
    };

    let Some(source_uuid) = document
        .graph
        .nodes()
        .find_map(|(uuid, node)| match node.op {
            WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
            _ => None,
        })
    else {
        return false;
    };

    let mut changed = false;
    changed |=
        ensure_default_odx_fixel_scalar_branch(document, source_uuid, default_name.clone(), true);
    changed |= ensure_default_odx_fixel_scalar_branch(document, source_uuid, default_name, false);
    changed
}

fn ensure_default_odx_fixel_scalar_branch(
    document: &mut WorkflowDocument,
    source_uuid: WorkflowNodeUuid,
    dpf_name: String,
    is_3d: bool,
) -> bool {
    let display_uuid = document.graph.wires().find_map(|wire| {
        if wire.from.node != source_uuid || wire.from.output != 0 || wire.to.input != 0 {
            return None;
        }
        match document.graph.get(wire.to.node).map(|node| &node.op) {
            Some(WorkflowNodeKind::Fixel3DDisplay { .. }) if is_3d => Some(wire.to.node),
            Some(WorkflowNodeKind::Fixel2DDisplay { .. }) if !is_3d => Some(wire.to.node),
            _ => None,
        }
    });
    let Some(display_uuid) = display_uuid else {
        return false;
    };

    let selector_uuid = make_node(
        document,
        OdxFixelScalarSelectOp {
            dpf_name: dpf_name.clone(),
        }
        .into(),
        document.graph.pos(display_uuid).unwrap_or(GraphPos::ZERO),
    );
    let color_uuid = make_node(
        document,
        ColorByFixelScalarsOp::default().into(),
        document.graph.pos(display_uuid).unwrap_or(GraphPos::ZERO),
    );
    document.graph.connect(
        OutPort {
            node: source_uuid,
            output: 2,
        },
        InPort {
            node: selector_uuid,
            input: 0,
        },
    );
    document.graph.connect(
        OutPort {
            node: source_uuid,
            output: 0,
        },
        InPort {
            node: color_uuid,
            input: 0,
        },
    );
    document.graph.connect(
        OutPort {
            node: selector_uuid,
            output: 0,
        },
        InPort {
            node: color_uuid,
            input: 1,
        },
    );
    document.graph.connect(
        OutPort {
            node: color_uuid,
            output: 0,
        },
        InPort {
            node: display_uuid,
            input: 0,
        },
    );
    let _ = relayout_connected_component(document, source_uuid);
    true
}

#[cfg(test)]
mod tests {
    use super::{
        choose_default_odx_dpf_name, choose_default_odx_dpv_name,
        set_default_odx_fixel_3d_visibility, set_default_odx_fixel_dpf, set_default_odx_volume_dpv,
    };
    use crate::workflow::seed::assert_graph_has_no_overlaps;
    use crate::workflow::{
        GraphPos, WorkflowAssetDocument, WorkflowNodeKind, add_default_nodes_for_asset,
        default_document,
    };
    use std::path::PathBuf;

    #[test]
    fn hides_default_odx_fixel_3d_display_when_requested() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 7,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);
        assert_graph_has_no_overlaps(&document);

        let changed = set_default_odx_fixel_3d_visibility(&mut document, 7, false);

        assert!(changed);
        assert_eq!(odx_fixel_3d_visibility(&document, 7), Some(false));
    }

    #[test]
    fn keeps_default_odx_fixel_3d_display_visible_without_glyph_field() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 9,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);
        assert_graph_has_no_overlaps(&document);

        let changed = set_default_odx_fixel_3d_visibility(&mut document, 9, true);

        assert!(!changed);
        assert_eq!(odx_fixel_3d_visibility(&document, 9), Some(true));
    }

    #[test]
    fn chooses_preferred_odx_dpv_name() {
        let selected = choose_default_odx_dpv_name(&[
            "qa".to_string(),
            "afd".to_string(),
            "anisotropic_power".to_string(),
        ]);
        assert_eq!(selected.as_deref(), Some("anisotropic_power"));
    }

    #[test]
    fn chooses_lexicographic_odx_dpv_name_when_no_preferred_exists() {
        let selected = choose_default_odx_dpv_name(&[
            "zeta".to_string(),
            "beta".to_string(),
            "alpha".to_string(),
        ]);
        assert_eq!(selected.as_deref(), Some("alpha"));
    }

    #[test]
    fn chooses_preferred_odx_dpf_name() {
        let selected = choose_default_odx_dpf_name(&[
            "qa".to_string(),
            "afd".to_string(),
            "amplitude".to_string(),
        ]);
        assert_eq!(selected.as_deref(), Some("amplitude"));
    }

    #[test]
    fn falls_back_to_lexicographic_odx_dpf_name() {
        let selected = choose_default_odx_dpf_name(&["zeta".to_string(), "alpha".to_string()]);
        assert_eq!(selected.as_deref(), Some("alpha"));
    }

    #[test]
    fn inserts_default_odx_scalar_branches_when_dpf_available() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 10,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

        let changed = set_default_odx_fixel_dpf(
            &mut document,
            10,
            &["qa".to_string(), "amplitude".to_string()],
        );

        assert!(changed);
        let selector_names: Vec<_> = document
            .graph
            .nodes()
            .filter_map(|(_, node)| match &node.op {
                WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => Some(dpf_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            selector_names,
            vec!["amplitude".to_string(), "amplitude".to_string()]
        );
        assert_eq!(
            document
                .graph
                .nodes()
                .filter(|(_, node)| matches!(
                    node.op,
                    WorkflowNodeKind::ColorByFixelScalars { .. }
                ))
                .count(),
            2
        );
        assert_graph_has_no_overlaps(&document);
    }

    #[test]
    fn leaves_default_odx_directional_branches_when_no_dpf_available() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 15,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

        let changed = set_default_odx_fixel_dpf(&mut document, 15, &[]);

        assert!(!changed);
        assert_eq!(
            document
                .graph
                .nodes()
                .filter(|(_, node)| matches!(
                    node.op,
                    WorkflowNodeKind::OdxFixelScalarSelect { .. }
                ))
                .count(),
            0
        );
        assert_graph_has_no_overlaps(&document);
    }

    #[test]
    fn sets_default_odx_volume_dpv_from_preference_list() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 11,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

        let changed = set_default_odx_volume_dpv(
            &mut document,
            11,
            &["fd".to_string(), "qa".to_string(), "gfa".to_string()],
        );

        assert!(changed);
        assert_eq!(odx_volume_dpv_name(&document, 11).as_deref(), Some("gfa"));
    }

    #[test]
    fn falls_back_to_lexicographic_odx_volume_dpv() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 12,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

        let changed = set_default_odx_volume_dpv(
            &mut document,
            12,
            &["zeta".to_string(), "alpha".to_string(), "beta".to_string()],
        );

        assert!(changed);
        assert_eq!(odx_volume_dpv_name(&document, 12).as_deref(), Some("alpha"));
    }

    #[test]
    fn preserves_existing_odx_volume_dpv_selection() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 13,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);
        let changed = set_default_odx_volume_dpv(&mut document, 13, &["qa".to_string()]);
        assert!(changed);
        assert_eq!(odx_volume_dpv_name(&document, 13).as_deref(), Some("qa"));

        let changed = set_default_odx_volume_dpv(
            &mut document,
            13,
            &["anisotropic_power".to_string(), "qa".to_string()],
        );

        assert!(!changed);
        assert_eq!(odx_volume_dpv_name(&document, 13).as_deref(), Some("qa"));
    }

    #[test]
    fn leaves_odx_volume_dpv_empty_when_no_dpvs_available() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 14,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

        let changed = set_default_odx_volume_dpv(&mut document, 14, &[]);

        assert!(!changed);
        assert_eq!(odx_volume_dpv_name(&document, 14).as_deref(), Some(""));
    }

    fn odx_fixel_3d_visibility(
        document: &crate::workflow::WorkflowDocument,
        asset_id: crate::data::loaded_files::FileId,
    ) -> Option<bool> {
        let source = document
            .graph
            .nodes()
            .find_map(|(uuid, node)| match node.op {
                WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
                _ => None,
            })?;

        document
            .graph
            .wires()
            .find(|wire| wire.from.node == source)
            .and_then(|wire| document.graph.get(wire.to.node))
            .and_then(|node| match &node.op {
                WorkflowNodeKind::Fixel3DDisplay { visible, .. } => Some(*visible),
                _ => None,
            })
    }

    fn odx_volume_dpv_name(
        document: &crate::workflow::WorkflowDocument,
        asset_id: crate::data::loaded_files::FileId,
    ) -> Option<String> {
        let source = document
            .graph
            .nodes()
            .find_map(|(uuid, node)| match node.op {
                WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
                _ => None,
            })?;

        document
            .graph
            .wires()
            .find(|wire| wire.from.node == source && wire.from.output == 2)
            .and_then(|wire| document.graph.get(wire.to.node))
            .and_then(|node| match &node.op {
                WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(dpv_name.clone()),
                _ => None,
            })
    }
}

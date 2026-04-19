use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::data::cifti::CiftiStructure;
use crate::data::loaded_files::{FileId, VolumeColormap};
use crate::data::trx_data::RenderStyle;
use crate::renderer::mesh_renderer::SurfaceColormap;

use super::graph::{GraphPos, GraphRect, InPort, OutPort};
use super::*;

/// Allocate a fresh UUID and insert a new node at the given canvas position.
pub fn make_node(
    document: &mut WorkflowDocument,
    kind: WorkflowNodeKind,
    pos: GraphPos,
) -> WorkflowNodeUuid {
    let uuid = WorkflowNodeUuid(document.next_node_uuid);
    document.next_node_uuid += 1;
    document.graph.insert_node(
        WorkflowNode {
            uuid,
            label: kind.title().to_string(),
            kind,
        },
        pos,
    );
    uuid
}

pub fn suggest_asset_branch_origin(document: &WorkflowDocument) -> GraphPos {
    let mut min_x = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (pos, _) in document.graph.nodes_pos() {
        min_x = min_x.min(pos.x);
        max_y = max_y.max(pos.y);
    }

    if min_x.is_finite() && max_y.is_finite() {
        GraphPos::new(min_x, max_y + 170.0)
    } else {
        GraphPos::new(40.0, 80.0)
    }
}

fn branch_bounds(document: &WorkflowDocument, nodes: &[WorkflowNodeUuid]) -> GraphRect {
    let mut bounds = GraphRect::EMPTY;
    for uuid in nodes {
        if let Some(pos) = document.graph.pos(*uuid) {
            bounds.extend(pos);
        }
    }
    if bounds.is_finite() {
        bounds.expanded(220.0, 120.0)
    } else {
        GraphRect {
            min: GraphPos::ZERO,
            max: GraphPos::new(640.0, 240.0),
        }
    }
}

fn offset(base: GraphPos, dx: f32, dy: f32) -> GraphPos {
    GraphPos::new(base.x + dx, base.y + dy)
}

fn connect_chain(document: &mut WorkflowDocument, from: WorkflowNodeUuid, to: WorkflowNodeUuid) {
    document.graph.connect(
        OutPort {
            node: from,
            output: 0,
        },
        InPort { node: to, input: 0 },
    );
}

pub fn add_default_nodes_for_asset(
    document: &mut WorkflowDocument,
    asset: &WorkflowAssetDocument,
    pos: GraphPos,
    streamline_limit: Option<usize>,
) -> SeededWorkflowBranch {
    match asset {
        WorkflowAssetDocument::Streamlines { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::StreamlineSource { source_id: *id },
                pos,
            );
            let group = make_node(
                document,
                WorkflowNodeKind::GroupSelect {
                    groups_csv: String::new(),
                },
                offset(pos, 220.0, 0.0),
            );
            let limit = make_node(
                document,
                WorkflowNodeKind::LimitStreamlines {
                    limit: streamline_limit.unwrap_or(30_000).max(1),
                    randomize: false,
                    seed: 1,
                },
                offset(pos, 440.0, 0.0),
            );
            let color = make_node(
                document,
                WorkflowNodeKind::ColorByDirection,
                offset(pos, 660.0, 0.0),
            );
            let display = make_node(
                document,
                WorkflowNodeKind::StreamlineDisplay {
                    enabled: true,
                    render_style: RenderStyle::Flat,
                    tube_radius_mm: 0.4,
                    tube_sides: 8,
                    slab_half_width_mm: 5.0,
                },
                offset(pos, 880.0, 0.0),
            );
            connect_chain(document, source, group);
            connect_chain(document, group, limit);
            connect_chain(document, limit, color);
            connect_chain(document, color, display);
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, group, limit, color, display]),
                primary_selection: WorkflowSelection::Node(limit),
            }
        }
        WorkflowAssetDocument::Volume { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::VolumeSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
                offset(pos, 220.0, 0.0),
            );
            connect_chain(document, source, display);
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(source),
            }
        }
        WorkflowAssetDocument::Cifti { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::CiftiSource { source_id: *id },
                pos,
            );
            let left = make_node(
                document,
                WorkflowNodeKind::CiftiStructure {
                    structure: CiftiStructure::CortexLeft,
                    map_index: 0,
                },
                offset(pos, 240.0, -80.0),
            );
            let right = make_node(
                document,
                WorkflowNodeKind::CiftiStructure {
                    structure: CiftiStructure::CortexRight,
                    map_index: 0,
                },
                offset(pos, 240.0, 0.0),
            );
            let subcortical = make_node(
                document,
                WorkflowNodeKind::CiftiStructure {
                    structure: CiftiStructure::Subcortical,
                    map_index: 0,
                },
                offset(pos, 240.0, 80.0),
            );
            document.graph.connect(
                OutPort {
                    node: source,
                    output: 0,
                },
                InPort {
                    node: left,
                    input: 0,
                },
            );
            document.graph.connect(
                OutPort {
                    node: source,
                    output: 0,
                },
                InPort {
                    node: right,
                    input: 0,
                },
            );
            document.graph.connect(
                OutPort {
                    node: source,
                    output: 0,
                },
                InPort {
                    node: subcortical,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, left, right, subcortical]),
                primary_selection: WorkflowSelection::Node(source),
            }
        }
        WorkflowAssetDocument::Surface { id, path } => {
            let default_surface_space = if guess_non_anatomical_surface(path) {
                SurfaceDisplaySpace::Stage
            } else {
                SurfaceDisplaySpace::Anatomical
            };
            let source = make_node(
                document,
                WorkflowNodeKind::SurfaceSource { source_id: *id },
                pos,
            );
            let overlay = make_node(
                document,
                WorkflowNodeKind::SurfaceOverlayStack {
                    layers: default_surface_overlay_layers(),
                },
                offset(pos, 220.0, 0.0),
            );
            let display = make_node(
                document,
                WorkflowNodeKind::SurfaceDisplay {
                    color: DEFAULT_SURFACE_COLOR,
                    opacity: DEFAULT_SURFACE_OPACITY,
                    outline_color: DEFAULT_SURFACE_COLOR,
                    outline_thickness: 1.25,
                    show_projection_map: false,
                    map_opacity: 1.0,
                    map_threshold: 0.0,
                    gloss: 0.45,
                    projection_colormap: SurfaceColormap::Inferno,
                    range_min: 0.0,
                    range_max: 1.0,
                    space: default_surface_space,
                },
                offset(pos, 440.0, 0.0),
            );
            connect_chain(document, source, overlay);
            connect_chain(document, overlay, display);
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, overlay, display]),
                primary_selection: WorkflowSelection::Node(source),
            }
        }
        WorkflowAssetDocument::Parcellation { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::ParcellationSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::ParcellationDisplay {
                    labels_csv: String::new(),
                    opacity: 0.9,
                },
                offset(pos, 240.0, 0.0),
            );
            connect_chain(document, source, display);
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(source),
            }
        }
        WorkflowAssetDocument::Odx { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::OdxSource { source_id: *id },
                pos,
            );
            let fixel_3d = make_node(
                document,
                WorkflowNodeKind::Fixel3DDisplay {
                    line_width: default_fixel_line_width(),
                    length_scale: default_fixel_length_scale(),
                    opacity: default_full_opacity(),
                    offset_from_slice: 0.0,
                    visible: true,
                },
                offset(pos, 260.0, -180.0),
            );
            let fixel_2d = make_node(
                document,
                WorkflowNodeKind::Fixel2DDisplay {
                    line_width: default_fixel_line_width(),
                    opacity: default_full_opacity(),
                    slab_thickness_mm: default_fixel_slab_thickness_mm(),
                    length_scale: default_fixel_length_scale(),
                    visible: true,
                },
                offset(pos, 260.0, -80.0),
            );
            let glyph = make_node(
                document,
                WorkflowNodeKind::OdfGlyphRenderer {
                    scale: default_odf_glyph_scale(),
                    opacity: default_full_opacity(),
                    offset_from_slice: 0.0,
                    gloss: 0.0,
                    vertex_colormap: GlyphColormap::default(),
                    slice_axis: WorkflowSliceViewKind::Axial,
                    opacity_gate: OpacityGate::default(),
                    size_gate: SizeGate::default(),
                    visible: true,
                },
                offset(pos, 260.0, 40.0),
            );
            let dpv_select = make_node(
                document,
                WorkflowNodeKind::OdxVolumeSelect {
                    dpv_name: String::new(),
                },
                offset(pos, 260.0, 140.0),
            );
            let volume_display = make_node(
                document,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
                offset(pos, 520.0, 140.0),
            );
            connect_chain(document, source, fixel_3d);
            connect_chain(document, source, fixel_2d);
            document.graph.connect(
                OutPort {
                    node: source,
                    output: 1,
                },
                InPort {
                    node: glyph,
                    input: 0,
                },
            );
            document.graph.connect(
                OutPort {
                    node: source,
                    output: 2,
                },
                InPort {
                    node: dpv_select,
                    input: 0,
                },
            );
            connect_chain(document, dpv_select, volume_display);
            SeededWorkflowBranch {
                bounds: branch_bounds(
                    document,
                    &[
                        source,
                        fixel_3d,
                        fixel_2d,
                        glyph,
                        dpv_select,
                        volume_display,
                    ],
                ),
                primary_selection: WorkflowSelection::Node(source),
            }
        }
    }
}

pub fn set_default_odx_fixel_3d_visibility(
    document: &mut WorkflowDocument,
    asset_id: FileId,
    visible: bool,
) -> bool {
    let source_nodes: Vec<_> = document
        .graph
        .nodes()
        .filter_map(|(uuid, node)| match node.kind {
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
        } = &mut node.kind
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
        .filter_map(|(uuid, node)| match node.kind {
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
        let WorkflowNodeKind::OdxVolumeSelect { dpv_name } = &mut node.kind else {
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
        .find_map(|(uuid, node)| match node.kind {
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
        match document.graph.get(wire.to.node).map(|node| &node.kind) {
            Some(WorkflowNodeKind::Fixel3DDisplay { .. }) if is_3d => Some(wire.to.node),
            Some(WorkflowNodeKind::Fixel2DDisplay { .. }) if !is_3d => Some(wire.to.node),
            _ => None,
        }
    });
    let Some(display_uuid) = display_uuid else {
        return false;
    };

    let source_pos = document.graph.pos(source_uuid).unwrap_or(GraphPos::ZERO);
    let display_pos = document.graph.pos(display_uuid).unwrap_or(source_pos);
    let selector_pos = GraphPos::new(source_pos.x + 240.0, display_pos.y - 40.0);
    let color_pos = GraphPos::new(source_pos.x + 500.0, display_pos.y - 20.0);
    let selector_uuid = make_node(
        document,
        WorkflowNodeKind::OdxFixelScalarSelect {
            dpf_name: dpf_name.clone(),
        },
        selector_pos,
    );
    let color_uuid = make_node(
        document,
        WorkflowNodeKind::ColorByFixelScalars {
            colormap: default_fixel_colormap(),
            range: None,
            length_scale_by_scalar: false,
        },
        color_pos,
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
    true
}

fn guess_non_anatomical_surface(path: &Path) -> bool {
    let file_name = match path.file_name() {
        Some(name) => name.to_string_lossy(),
        None => return false,
    };
    non_anatomical_surface_regex().is_match(&file_name)
}

fn non_anatomical_surface_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)([-_.](sphere|inflated)[-_.])").expect("valid non-anatomical regex")
    })
}

#[cfg(test)]
mod tests {
    use super::{
        choose_default_odx_dpf_name, choose_default_odx_dpv_name, guess_non_anatomical_surface,
        set_default_odx_fixel_3d_visibility, set_default_odx_fixel_dpf, set_default_odx_volume_dpv,
    };
    use crate::workflow::{
        GraphPos, WorkflowAssetDocument, WorkflowNodeKind, add_default_nodes_for_asset,
        default_document,
    };
    use std::path::Path;
    use std::path::PathBuf;

    #[test]
    fn guesses_inflated_surface_as_non_anatomical() {
        assert!(guess_non_anatomical_surface(Path::new(
            "100307.L.inflated.164k_fs_LR.surf.gii"
        )));
    }

    #[test]
    fn guesses_sphere_surface_as_non_anatomical() {
        assert!(guess_non_anatomical_surface(Path::new(
            "subject-L_sphere.surf.gii"
        )));
    }

    #[test]
    fn anatomical_surface_name_is_not_promoted_to_stage() {
        assert!(!guess_non_anatomical_surface(Path::new(
            "subject.L.midthickness.surf.gii"
        )));
    }

    #[test]
    fn hides_default_odx_fixel_3d_display_when_requested() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Odx {
            id: 7,
            path: PathBuf::from("subject.odx"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);

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
            .filter_map(|(_, node)| match &node.kind {
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
                    node.kind,
                    WorkflowNodeKind::ColorByFixelScalars { .. }
                ))
                .count(),
            2
        );
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
                    node.kind,
                    WorkflowNodeKind::OdxFixelScalarSelect { .. }
                ))
                .count(),
            0
        );
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
            .find_map(|(uuid, node)| match node.kind {
                WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
                _ => None,
            })?;

        document
            .graph
            .wires()
            .find(|wire| wire.from.node == source)
            .and_then(|wire| document.graph.get(wire.to.node))
            .and_then(|node| match &node.kind {
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
            .find_map(|(uuid, node)| match node.kind {
                WorkflowNodeKind::OdxSource { source_id } if source_id == asset_id => Some(uuid),
                _ => None,
            })?;

        document
            .graph
            .wires()
            .find(|wire| wire.from.node == source && wire.from.output == 2)
            .and_then(|wire| document.graph.get(wire.to.node))
            .and_then(|node| match &node.kind {
                WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(dpv_name.clone()),
                _ => None,
            })
    }
}

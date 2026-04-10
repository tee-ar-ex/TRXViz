use crate::data::loaded_files::VolumeColormap;
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
        WorkflowAssetDocument::Surface { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::SurfaceSource { source_id: *id },
                pos,
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
                },
                offset(pos, 220.0, 0.0),
            );
            connect_chain(document, source, display);
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
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
    }
}

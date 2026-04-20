use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::data::cifti::CiftiStructure;

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

fn connect_chain(document: &mut WorkflowDocument, from: WorkflowNodeUuid, to: WorkflowNodeUuid) {
    document.graph.connect(
        OutPort {
            node: from,
            output: 0,
        },
        InPort { node: to, input: 0 },
    );
}

fn finalize_seeded_branch(
    document: &mut WorkflowDocument,
    nodes: &[WorkflowNodeUuid],
    primary_selection: WorkflowSelection,
    anchor: GraphPos,
) -> SeededWorkflowBranch {
    let sizes = estimated_workflow_node_sizes(&document.graph);
    let layout = layout_workflow_graph_subset(
        &document.graph,
        &sizes,
        nodes,
        Some(anchor),
        &WorkflowLayoutOptions::default(),
    );
    apply_workflow_layout(&mut document.graph, &layout);
    SeededWorkflowBranch {
        bounds: layout.bounds,
        primary_selection,
    }
}

pub(crate) fn relayout_connected_component(
    document: &mut WorkflowDocument,
    seed: WorkflowNodeUuid,
) -> Option<GraphRect> {
    let nodes = weakly_connected_closure(&document.graph, &[seed]);
    if nodes.is_empty() {
        return None;
    }
    let sizes = estimated_workflow_node_sizes(&document.graph);
    let layout = layout_workflow_graph_subset(
        &document.graph,
        &sizes,
        &nodes,
        None,
        &WorkflowLayoutOptions::default(),
    );
    apply_workflow_layout(&mut document.graph, &layout);
    Some(layout.bounds)
}

pub fn add_default_nodes_for_asset(
    document: &mut WorkflowDocument,
    asset: &WorkflowAssetDocument,
    pos: GraphPos,
    streamline_limit: Option<usize>,
) -> SeededWorkflowBranch {
    match asset {
        WorkflowAssetDocument::Streamlines { id, .. } => {
            let source = make_node(document, StreamlineSourceOp { source_id: *id }.into(), pos);
            let group = make_node(document, GroupSelectOp::default().into(), pos);
            let limit = make_node(
                document,
                LimitStreamlinesOp {
                    limit: streamline_limit.unwrap_or(30_000).max(1),
                    ..Default::default()
                }
                .into(),
                pos,
            );
            let color = make_node(document, ColorByDirectionOp.into(), pos);
            let display = make_node(document, StreamlineDisplayOp::default().into(), pos);
            connect_chain(document, source, group);
            connect_chain(document, group, limit);
            connect_chain(document, limit, color);
            connect_chain(document, color, display);
            finalize_seeded_branch(
                document,
                &[source, group, limit, color, display],
                WorkflowSelection::Node(limit),
                pos,
            )
        }
        WorkflowAssetDocument::Volume { id, .. } => {
            let source = make_node(document, VolumeSourceOp { source_id: *id }.into(), pos);
            let display = make_node(document, VolumeDisplayOp::default().into(), pos);
            connect_chain(document, source, display);
            finalize_seeded_branch(
                document,
                &[source, display],
                WorkflowSelection::Node(source),
                pos,
            )
        }
        WorkflowAssetDocument::Cifti { id, .. } => {
            let source = make_node(document, CiftiSourceOp { source_id: *id }.into(), pos);
            let left = make_node(
                document,
                CiftiStructureOp {
                    structure: CiftiStructure::CortexLeft,
                    map_index: 0,
                }
                .into(),
                pos,
            );
            let right = make_node(
                document,
                CiftiStructureOp {
                    structure: CiftiStructure::CortexRight,
                    map_index: 0,
                }
                .into(),
                pos,
            );
            let subcortical = make_node(
                document,
                CiftiStructureOp {
                    structure: CiftiStructure::Subcortical,
                    map_index: 0,
                }
                .into(),
                pos,
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
            finalize_seeded_branch(
                document,
                &[source, left, right, subcortical],
                WorkflowSelection::Node(source),
                pos,
            )
        }
        WorkflowAssetDocument::Surface { id, path } => {
            let default_surface_space = if guess_non_anatomical_surface(path) {
                SurfaceDisplaySpace::Stage
            } else {
                SurfaceDisplaySpace::Anatomical
            };
            let source = make_node(document, SurfaceSourceOp { source_id: *id }.into(), pos);
            let overlay = make_node(document, SurfaceOverlayStackOp::default().into(), pos);
            let display = make_node(
                document,
                SurfaceDisplayOp {
                    space: default_surface_space,
                    ..Default::default()
                }
                .into(),
                pos,
            );
            connect_chain(document, source, overlay);
            connect_chain(document, overlay, display);
            finalize_seeded_branch(
                document,
                &[source, overlay, display],
                WorkflowSelection::Node(source),
                pos,
            )
        }
        WorkflowAssetDocument::Parcellation { id, .. } => {
            let source = make_node(
                document,
                ParcellationSourceOp { source_id: *id }.into(),
                pos,
            );
            let display = make_node(document, ParcellationDisplayOp::default().into(), pos);
            connect_chain(document, source, display);
            finalize_seeded_branch(
                document,
                &[source, display],
                WorkflowSelection::Node(source),
                pos,
            )
        }
        WorkflowAssetDocument::Odx { id, .. } => {
            let source = make_node(document, OdxSourceOp { source_id: *id }.into(), pos);
            let fixel_3d = make_node(document, Fixel3DDisplayOp::default().into(), pos);
            let fixel_2d = make_node(document, Fixel2DDisplayOp::default().into(), pos);
            let glyph = make_node(document, OdfGlyphRendererOp::default().into(), pos);
            let dpv_select = make_node(document, OdxVolumeSelectOp::default().into(), pos);
            let volume_display = make_node(document, VolumeDisplayOp::default().into(), pos);
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
            finalize_seeded_branch(
                document,
                &[
                    source,
                    fixel_3d,
                    fixel_2d,
                    glyph,
                    dpv_select,
                    volume_display,
                ],
                WorkflowSelection::Node(source),
                pos,
            )
        }
    }
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
pub(crate) fn assert_graph_has_no_overlaps(document: &crate::workflow::WorkflowDocument) {
    let sizes = estimated_workflow_node_sizes(&document.graph);
    let rects: Vec<_> = document
        .graph
        .nodes()
        .filter_map(|(uuid, _)| {
            let pos = document.graph.pos(uuid)?;
            let size = sizes.get(&uuid).copied()?;
            Some((
                uuid,
                crate::workflow::GraphRect {
                    min: pos,
                    max: crate::workflow::GraphPos::new(pos.x + size.width, pos.y + size.height),
                },
            ))
        })
        .collect();
    for (idx, (_, left)) in rects.iter().enumerate() {
        for (_, right) in rects.iter().skip(idx + 1) {
            let overlaps = left.min.x < right.max.x
                && left.max.x > right.min.x
                && left.min.y < right.max.y
                && left.max.y > right.min.y;
            assert!(!overlaps, "workflow nodes should not overlap");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        add_default_nodes_for_asset, assert_graph_has_no_overlaps, guess_non_anatomical_surface,
    };
    use crate::data::cifti::CiftiIntent;
    use crate::workflow::{GraphPos, WorkflowAssetDocument, default_document};
    use std::path::{Path, PathBuf};

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
    fn seeded_streamline_branch_is_non_overlapping() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Streamlines {
            id: 1,
            path: PathBuf::from("tracks.trx"),
            imported: false,
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, Some(10_000));
        assert_graph_has_no_overlaps(&document);
    }

    #[test]
    fn seeded_surface_branch_is_non_overlapping() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Surface {
            id: 2,
            path: PathBuf::from("subject.L.midthickness.surf.gii"),
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);
        assert_graph_has_no_overlaps(&document);
    }

    #[test]
    fn seeded_cifti_branch_is_non_overlapping() {
        let mut document = default_document();
        let asset = WorkflowAssetDocument::Cifti {
            id: 3,
            path: PathBuf::from("subject.dscalar.nii"),
            intent: CiftiIntent::DenseScalar,
        };
        add_default_nodes_for_asset(&mut document, &asset, GraphPos::ZERO, None);
        assert_graph_has_no_overlaps(&document);
    }
}

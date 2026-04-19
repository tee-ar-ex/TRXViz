use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use petgraph::Directed;
use petgraph::algo::kosaraju_scc;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;

use super::{GraphPos, GraphRect, WorkflowGraph, WorkflowNode, WorkflowNodeUuid};

const APPROX_CHAR_WIDTH: f32 = 7.2;
const TITLE_CHAR_WIDTH: f32 = 7.8;
const NODE_MIN_WIDTH: f32 = 170.0;
const NODE_FRAME_HORIZONTAL_PADDING: f32 = 72.0;
const NODE_HEADER_HEIGHT: f32 = 28.0;
const NODE_BODY_HEIGHT: f32 = 24.0;
const NODE_ROW_HEIGHT: f32 = 24.0;
const NODE_BOTTOM_PADDING: f32 = 18.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NodeSize {
    pub width: f32,
    pub height: f32,
}

impl NodeSize {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkflowLayoutOptions {
    pub column_gap: f32,
    pub row_gap: f32,
    pub component_gap: f32,
    pub outer_margin: f32,
}

impl Default for WorkflowLayoutOptions {
    fn default() -> Self {
        Self {
            column_gap: 96.0,
            row_gap: 36.0,
            component_gap: 120.0,
            outer_margin: 24.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowLayoutResult {
    pub positions: HashMap<WorkflowNodeUuid, GraphPos>,
    pub bounds: GraphRect,
}

impl Default for WorkflowLayoutResult {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            bounds: GraphRect::EMPTY,
        }
    }
}

pub fn estimate_workflow_node_size(node: &WorkflowNode) -> NodeSize {
    let title_width = node.kind.title().chars().count() as f32 * TITLE_CHAR_WIDTH;
    let max_input_label = node
        .kind
        .inputs()
        .iter()
        .map(|port| port_kind_label(*port).chars().count() as f32 * APPROX_CHAR_WIDTH)
        .fold(0.0, f32::max);
    let max_output_label = node
        .kind
        .outputs()
        .iter()
        .map(|port| port_kind_label(*port).chars().count() as f32 * APPROX_CHAR_WIDTH)
        .fold(0.0, f32::max);
    let rows = node.kind.inputs().len().max(node.kind.outputs().len()) as f32;
    let width = (title_width.max(max_input_label + max_output_label)
        + NODE_FRAME_HORIZONTAL_PADDING)
        .max(NODE_MIN_WIDTH);
    let height =
        NODE_HEADER_HEIGHT + NODE_BODY_HEIGHT + rows * NODE_ROW_HEIGHT + NODE_BOTTOM_PADDING;
    NodeSize::new(width, height)
}

pub fn estimated_workflow_node_sizes(graph: &WorkflowGraph) -> HashMap<WorkflowNodeUuid, NodeSize> {
    graph
        .nodes()
        .map(|(uuid, node)| (uuid, estimate_workflow_node_size(node)))
        .collect()
}

pub fn weakly_connected_closure(
    graph: &WorkflowGraph,
    seeds: &[WorkflowNodeUuid],
) -> Vec<WorkflowNodeUuid> {
    if seeds.is_empty() {
        return Vec::new();
    }
    let mut adjacency: HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>> = HashMap::new();
    for (uuid, _) in graph.nodes() {
        adjacency.entry(uuid).or_default();
    }
    for wire in graph.wires() {
        adjacency
            .entry(wire.from.node)
            .or_default()
            .push(wire.to.node);
        adjacency
            .entry(wire.to.node)
            .or_default()
            .push(wire.from.node);
    }
    let mut queue: VecDeque<WorkflowNodeUuid> = seeds.iter().copied().collect();
    let mut visited = BTreeSet::new();
    while let Some(uuid) = queue.pop_front() {
        if !visited.insert(uuid) {
            continue;
        }
        for neighbor in adjacency.get(&uuid).into_iter().flatten() {
            if !visited.contains(neighbor) {
                queue.push_back(*neighbor);
            }
        }
    }
    visited.into_iter().collect()
}

pub fn layout_workflow_graph(
    graph: &WorkflowGraph,
    node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    options: &WorkflowLayoutOptions,
) -> WorkflowLayoutResult {
    let nodes: Vec<_> = graph.nodes().map(|(uuid, _)| uuid).collect();
    layout_workflow_graph_subset(graph, node_sizes, &nodes, None, options)
}

pub fn layout_workflow_graph_subset(
    graph: &WorkflowGraph,
    node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    nodes: &[WorkflowNodeUuid],
    anchor: Option<GraphPos>,
    options: &WorkflowLayoutOptions,
) -> WorkflowLayoutResult {
    if nodes.is_empty() {
        return WorkflowLayoutResult::default();
    }

    let selected: BTreeSet<_> = nodes.iter().copied().collect();
    let anchor = anchor.unwrap_or_else(|| {
        selected
            .iter()
            .filter_map(|uuid| graph.pos(*uuid))
            .fold(GraphRect::EMPTY, |mut rect, pos| {
                rect.extend(pos);
                rect
            })
            .min
    });

    let mut components = weak_components(graph, &selected);
    components.sort_by(|left, right| {
        component_sort_key(graph, left).cmp(&component_sort_key(graph, right))
    });

    let mut positions = HashMap::new();
    let mut stacked_bounds = GraphRect::EMPTY;
    let mut cursor_y = options.outer_margin;

    for component in components {
        let component_layout = layout_single_component(graph, node_sizes, &component, options);
        let component_height = component_layout.bounds.max.y - component_layout.bounds.min.y;
        for (uuid, pos) in component_layout.positions {
            let translated = GraphPos::new(pos.x, pos.y + cursor_y - component_layout.bounds.min.y);
            extend_bounds_with_node(
                &mut stacked_bounds,
                translated,
                node_sizes_for(node_sizes, graph, uuid),
            );
            positions.insert(uuid, translated);
        }
        cursor_y += component_height + options.component_gap;
    }

    if !stacked_bounds.is_finite() {
        return WorkflowLayoutResult::default();
    }

    let dx = anchor.x - stacked_bounds.min.x;
    let dy = anchor.y - stacked_bounds.min.y;
    for pos in positions.values_mut() {
        pos.x += dx;
        pos.y += dy;
    }
    let bounds = translate_rect(stacked_bounds, dx, dy);

    WorkflowLayoutResult { positions, bounds }
}

pub fn apply_workflow_layout(graph: &mut WorkflowGraph, layout: &WorkflowLayoutResult) {
    for (uuid, pos) in &layout.positions {
        graph.set_pos(*uuid, *pos);
    }
}

fn layout_single_component(
    graph: &WorkflowGraph,
    node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    nodes: &[WorkflowNodeUuid],
    options: &WorkflowLayoutOptions,
) -> WorkflowLayoutResult {
    let component_nodes: BTreeSet<_> = nodes.iter().copied().collect();
    let (ranks, edges) = rank_component(graph, &component_nodes);
    let mut columns: BTreeMap<usize, Vec<WorkflowNodeUuid>> = BTreeMap::new();
    for uuid in nodes {
        columns
            .entry(*ranks.get(uuid).unwrap_or(&0))
            .or_default()
            .push(*uuid);
    }

    let current_y: HashMap<WorkflowNodeUuid, f32> = nodes
        .iter()
        .map(|uuid| (*uuid, graph.pos(*uuid).map(|pos| pos.y).unwrap_or(0.0)))
        .collect();

    for column in columns.values_mut() {
        column.sort_by(|left, right| {
            current_y[left]
                .total_cmp(&current_y[right])
                .then_with(|| left.0.cmp(&right.0))
        });
    }

    for _ in 0..4 {
        reorder_columns(&mut columns, &edges, false, &current_y);
        reorder_columns(&mut columns, &edges, true, &current_y);
    }

    let mut positions = HashMap::new();
    let mut bounds = GraphRect::EMPTY;
    let mut x_cursor = options.outer_margin;
    for nodes_in_column in columns.values() {
        let column_width = nodes_in_column
            .iter()
            .map(|uuid| node_sizes_for(node_sizes, graph, *uuid).width)
            .fold(0.0, f32::max);
        let mut y_cursor = options.outer_margin;
        for uuid in nodes_in_column {
            let pos = GraphPos::new(x_cursor, y_cursor);
            let size = node_sizes_for(node_sizes, graph, *uuid);
            extend_bounds_with_node(&mut bounds, pos, size);
            positions.insert(*uuid, pos);
            y_cursor += size.height + options.row_gap;
        }
        x_cursor += column_width + options.column_gap;
    }

    WorkflowLayoutResult { positions, bounds }
}

fn rank_component(
    graph: &WorkflowGraph,
    nodes: &BTreeSet<WorkflowNodeUuid>,
) -> (
    HashMap<WorkflowNodeUuid, usize>,
    HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
) {
    let mut dag = StableGraph::<WorkflowNodeUuid, (), Directed>::default();
    let mut graph_idx = HashMap::new();
    for uuid in nodes {
        graph_idx.insert(*uuid, dag.add_node(*uuid));
    }
    for wire in graph.wires() {
        if nodes.contains(&wire.from.node) && nodes.contains(&wire.to.node) {
            dag.add_edge(graph_idx[&wire.from.node], graph_idx[&wire.to.node], ());
        }
    }

    let sccs = kosaraju_scc(&dag);
    let mut node_to_scc = HashMap::new();
    for (scc_idx, members) in sccs.iter().enumerate() {
        for member in members {
            node_to_scc.insert(dag[*member], scc_idx);
        }
    }

    let mut cond = StableGraph::<usize, (), Directed>::default();
    let cond_nodes: Vec<_> = (0..sccs.len()).map(|idx| cond.add_node(idx)).collect();
    let mut seen_edges = HashSet::new();
    for edge in dag.edge_references() {
        let from = node_to_scc[&dag[edge.source()]];
        let to = node_to_scc[&dag[edge.target()]];
        if from != to && seen_edges.insert((from, to)) {
            cond.add_edge(cond_nodes[from], cond_nodes[to], ());
        }
    }

    let topo = petgraph::algo::toposort(&cond, None).unwrap_or_default();
    let mut scc_rank = HashMap::<usize, usize>::new();
    for idx in topo {
        let scc_idx = cond[idx];
        let mut rank = 0usize;
        for incoming in cond.neighbors_directed(idx, petgraph::Incoming) {
            let parent = cond[incoming];
            rank = rank.max(scc_rank.get(&parent).copied().unwrap_or(0) + 1);
        }
        scc_rank.insert(scc_idx, rank);
    }

    let mut ranks = HashMap::new();
    for uuid in nodes {
        ranks.insert(
            *uuid,
            scc_rank.get(&node_to_scc[uuid]).copied().unwrap_or(0),
        );
    }

    let mut edges = HashMap::<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>::new();
    for uuid in nodes {
        edges.entry(*uuid).or_default();
    }
    for wire in graph.wires() {
        if nodes.contains(&wire.from.node) && nodes.contains(&wire.to.node) {
            edges.entry(wire.from.node).or_default().push(wire.to.node);
            edges.entry(wire.to.node).or_default().push(wire.from.node);
        }
    }

    (ranks, edges)
}

fn reorder_columns(
    columns: &mut BTreeMap<usize, Vec<WorkflowNodeUuid>>,
    neighbors: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
    reverse: bool,
    current_y: &HashMap<WorkflowNodeUuid, f32>,
) {
    let column_keys: Vec<_> = if reverse {
        columns.keys().copied().rev().collect()
    } else {
        columns.keys().copied().collect()
    };
    let order_index = build_order_index(columns);
    for key in column_keys {
        let Some(column) = columns.get_mut(&key) else {
            continue;
        };
        column.sort_by(|left, right| {
            barycenter(neighbors.get(left), &order_index)
                .total_cmp(&barycenter(neighbors.get(right), &order_index))
                .then_with(|| current_y[left].total_cmp(&current_y[right]))
                .then_with(|| left.0.cmp(&right.0))
        });
    }
}

fn build_order_index(
    columns: &BTreeMap<usize, Vec<WorkflowNodeUuid>>,
) -> HashMap<WorkflowNodeUuid, usize> {
    let mut order = HashMap::new();
    for nodes in columns.values() {
        for (idx, uuid) in nodes.iter().enumerate() {
            order.insert(*uuid, idx);
        }
    }
    order
}

fn barycenter(
    neighbors: Option<&Vec<WorkflowNodeUuid>>,
    order_index: &HashMap<WorkflowNodeUuid, usize>,
) -> f32 {
    let Some(neighbors) = neighbors else {
        return f32::INFINITY;
    };
    let mut count = 0.0f32;
    let mut sum = 0.0f32;
    for neighbor in neighbors {
        if let Some(index) = order_index.get(neighbor) {
            sum += *index as f32;
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        f32::INFINITY
    }
}

fn weak_components(
    graph: &WorkflowGraph,
    nodes: &BTreeSet<WorkflowNodeUuid>,
) -> Vec<Vec<WorkflowNodeUuid>> {
    let mut adjacency: HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>> = HashMap::new();
    for uuid in nodes {
        adjacency.insert(*uuid, Vec::new());
    }
    for wire in graph.wires() {
        if nodes.contains(&wire.from.node) && nodes.contains(&wire.to.node) {
            adjacency
                .entry(wire.from.node)
                .or_default()
                .push(wire.to.node);
            adjacency
                .entry(wire.to.node)
                .or_default()
                .push(wire.from.node);
        }
    }

    let mut components = Vec::new();
    let mut visited = BTreeSet::new();
    for uuid in nodes {
        if visited.contains(uuid) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([*uuid]);
        while let Some(next) = queue.pop_front() {
            if !visited.insert(next) {
                continue;
            }
            component.push(next);
            for neighbor in adjacency.get(&next).into_iter().flatten() {
                if !visited.contains(neighbor) {
                    queue.push_back(*neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

fn component_sort_key(graph: &WorkflowGraph, component: &[WorkflowNodeUuid]) -> (i32, i32, u64) {
    let mut bounds = GraphRect::EMPTY;
    let mut min_uuid = u64::MAX;
    for uuid in component {
        if let Some(pos) = graph.pos(*uuid) {
            bounds.extend(pos);
        }
        min_uuid = min_uuid.min(uuid.0);
    }
    (
        bounds.min.y.round() as i32,
        bounds.min.x.round() as i32,
        min_uuid,
    )
}

fn node_sizes_for(
    node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    graph: &WorkflowGraph,
    uuid: WorkflowNodeUuid,
) -> NodeSize {
    node_sizes
        .get(&uuid)
        .copied()
        .or_else(|| graph.get(uuid).map(estimate_workflow_node_size))
        .unwrap_or_else(|| NodeSize::new(NODE_MIN_WIDTH, NODE_HEADER_HEIGHT + NODE_BODY_HEIGHT))
}

fn extend_bounds_with_node(bounds: &mut GraphRect, pos: GraphPos, size: NodeSize) {
    bounds.extend(pos);
    bounds.extend(GraphPos::new(pos.x + size.width, pos.y + size.height));
}

fn translate_rect(rect: GraphRect, dx: f32, dy: f32) -> GraphRect {
    GraphRect {
        min: GraphPos::new(rect.min.x + dx, rect.min.y + dy),
        max: GraphPos::new(rect.max.x + dx, rect.max.y + dy),
    }
}

fn port_kind_label(port: super::PortKind) -> &'static str {
    match port {
        super::PortKind::Streamline => "Streamline",
        super::PortKind::Volume => "Volume",
        super::PortKind::Surface => "Surface",
        super::PortKind::Parcellation => "Parcellation",
        super::PortKind::ParcelSelection => "Parcel Set",
        super::PortKind::Cifti => "CIFTI",
        super::PortKind::SurfaceScalars => "Surface Scalars",
        super::PortKind::VolumeScalars => "Volume Scalars",
        super::PortKind::SurfaceAppearance => "Surface Appearance",
        super::PortKind::BundleSurface => "Bundle Surface",
        super::PortKind::BoundaryField => "Boundary Field",
        super::PortKind::Fixels => "Fixels",
        super::PortKind::FixelScalars => "Fixel Scalars",
        super::PortKind::OdfField => "ODF Field",
        super::PortKind::OdxCatalog => "ODX Catalog",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{InPort, OutPort, WorkflowNode, WorkflowNodeKind};

    fn make_node(uuid: u64, x: f32, y: f32, kind: WorkflowNodeKind) -> WorkflowNodeEntry {
        WorkflowNodeEntry {
            node: WorkflowNode {
                uuid: WorkflowNodeUuid(uuid),
                label: kind.title().to_string(),
                kind,
            },
            pos: GraphPos::new(x, y),
        }
    }

    use crate::workflow::WorkflowNodeEntry;

    fn synthetic_graph() -> WorkflowGraph {
        let mut graph = WorkflowGraph::new();
        graph.insert_node(
            make_node(1, 0.0, 0.0, WorkflowNodeKind::OdxSource { source_id: 1 }).node,
            GraphPos::new(0.0, 0.0),
        );
        graph.insert_node(
            make_node(
                2,
                0.0,
                0.0,
                WorkflowNodeKind::OdxFixelScalarSelect {
                    dpf_name: "very_long_scalar_name".into(),
                },
            )
            .node,
            GraphPos::new(0.0, 0.0),
        );
        graph.insert_node(
            make_node(
                3,
                0.0,
                0.0,
                WorkflowNodeKind::ColorByFixelScalars {
                    colormap: crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
                    range: None,
                    length_scale_by_scalar: false,
                },
            )
            .node,
            GraphPos::new(0.0, 0.0),
        );
        graph.insert_node(
            make_node(
                4,
                0.0,
                0.0,
                WorkflowNodeKind::Fixel3DDisplay {
                    line_width: 0.1,
                    length_scale: 1.0,
                    opacity: 1.0,
                    offset_from_slice: 0.0,
                    visible: true,
                },
            )
            .node,
            GraphPos::new(0.0, 0.0),
        );
        graph.connect(
            OutPort {
                node: WorkflowNodeUuid(1),
                output: 2,
            },
            InPort {
                node: WorkflowNodeUuid(2),
                input: 0,
            },
        );
        graph.connect(
            OutPort {
                node: WorkflowNodeUuid(1),
                output: 0,
            },
            InPort {
                node: WorkflowNodeUuid(3),
                input: 0,
            },
        );
        graph.connect(
            OutPort {
                node: WorkflowNodeUuid(2),
                output: 0,
            },
            InPort {
                node: WorkflowNodeUuid(3),
                input: 1,
            },
        );
        graph.connect(
            OutPort {
                node: WorkflowNodeUuid(3),
                output: 0,
            },
            InPort {
                node: WorkflowNodeUuid(4),
                input: 0,
            },
        );
        graph
    }

    fn overlaps(
        layout: &WorkflowLayoutResult,
        sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    ) -> bool {
        let rects: Vec<_> = layout
            .positions
            .iter()
            .map(|(uuid, pos)| {
                let size = sizes[uuid];
                (
                    *uuid,
                    GraphRect {
                        min: *pos,
                        max: GraphPos::new(pos.x + size.width, pos.y + size.height),
                    },
                )
            })
            .collect();
        for (idx, (_, left)) in rects.iter().enumerate() {
            for (_, right) in rects.iter().skip(idx + 1) {
                if left.min.x < right.max.x
                    && left.max.x > right.min.x
                    && left.min.y < right.max.y
                    && left.max.y > right.min.y
                {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn wide_synthetic_nodes_do_not_overlap() {
        let graph = synthetic_graph();
        let mut sizes = estimated_workflow_node_sizes(&graph);
        sizes.insert(WorkflowNodeUuid(2), NodeSize::new(340.0, 120.0));
        sizes.insert(WorkflowNodeUuid(3), NodeSize::new(320.0, 120.0));

        let layout = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());

        assert!(!overlaps(&layout, &sizes));
    }

    #[test]
    fn disconnected_components_stack_vertically() {
        let mut graph = synthetic_graph();
        graph.insert_node(
            WorkflowNode {
                uuid: WorkflowNodeUuid(10),
                label: "Display".into(),
                kind: WorkflowNodeKind::VolumeDisplay {
                    colormap: crate::data::loaded_files::VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
            },
            GraphPos::new(500.0, 400.0),
        );
        let sizes = estimated_workflow_node_sizes(&graph);

        let layout = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());
        let y1 = layout.positions[&WorkflowNodeUuid(1)].y;
        let y10 = layout.positions[&WorkflowNodeUuid(10)].y;

        assert!(y10 > y1);
    }

    #[test]
    fn cycles_still_receive_finite_positions() {
        let mut graph = synthetic_graph();
        graph.connect(
            OutPort {
                node: WorkflowNodeUuid(4),
                output: 0,
            },
            InPort {
                node: WorkflowNodeUuid(2),
                input: 0,
            },
        );
        let sizes = estimated_workflow_node_sizes(&graph);

        let layout = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());

        assert!(
            layout
                .positions
                .values()
                .all(|pos| pos.x.is_finite() && pos.y.is_finite())
        );
    }

    #[test]
    fn layout_is_stable_given_same_graph_and_sizes() {
        let graph = synthetic_graph();
        let sizes = estimated_workflow_node_sizes(&graph);

        let first = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());
        let second = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());

        assert_eq!(first, second);
    }

    #[test]
    fn applying_layout_moves_nodes_without_changing_graph_shape() {
        let mut graph = synthetic_graph();
        let original = graph.clone();
        let sizes = estimated_workflow_node_sizes(&graph);

        let layout = layout_workflow_graph(&graph, &sizes, &WorkflowLayoutOptions::default());
        apply_workflow_layout(&mut graph, &layout);

        assert_eq!(
            original
                .wires()
                .map(|wire| (
                    wire.from.node,
                    wire.from.output,
                    wire.to.node,
                    wire.to.input
                ))
                .collect::<Vec<_>>(),
            graph
                .wires()
                .map(|wire| (
                    wire.from.node,
                    wire.from.output,
                    wire.to.node,
                    wire.to.input
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            original
                .nodes()
                .map(|(uuid, node)| (uuid, node.kind.clone()))
                .collect::<Vec<_>>(),
            graph
                .nodes()
                .map(|(uuid, node)| (uuid, node.kind.clone()))
                .collect::<Vec<_>>()
        );
        assert!(
            original
                .nodes()
                .any(|(uuid, _)| original.pos(uuid) != graph.pos(uuid))
        );
    }
}

//! Hand-rolled layered-graph layout and SVG emission for workflow
//! diagrams in the docs. Zero external dependencies: strings in, string
//! out.
//!
//! Algorithm summary:
//! - Longest-path layering (Kahn's algorithm over reversed edges): each
//!   node's layer is `1 + max(layer of predecessor)`; sources land at 0.
//! - Within a layer, nodes are ordered by `WorkflowNodeUuid` for
//!   stability — snapshot tests don't churn as unrelated nodes are
//!   added. No median-heuristic crossing reduction yet; small example
//!   graphs look fine without it.
//! - Nodes are fixed-width rounded rects. Height scales with
//!   `max(inputs, outputs)`.
//! - Ports sit flush on the left/right edges at equal vertical spacing.
//! - Wires are cubic beziers with horizontal control handles — matches
//!   the egui-snarl look closely enough for static docs.
//!
//! The port color table duplicates `graph_viewer::port_color` in the
//! GUI crate; see the comment next to `port_fill` for why we don't
//! share it.
//!
//! Only `render_workflow_svg` is exported — everything else is an
//! internal layout detail.
use std::collections::{BTreeMap, HashMap};

use trxviz_core::workflow::port_labels::{input_port_label, output_port_label};
use trxviz_core::workflow::{
    GraphPos, PortKind, Wire, WorkflowDocument, WorkflowGraph, WorkflowNodeKind, WorkflowNodeUuid,
};

const NODE_WIDTH: f32 = 180.0;
const PORT_ROW_HEIGHT: f32 = 18.0;
const NODE_MIN_HEIGHT: f32 = 56.0;
const NODE_HEADER_HEIGHT: f32 = 28.0;
const LAYER_PITCH_X: f32 = 260.0;
const ROW_PITCH_Y: f32 = 28.0; // extra gutter between nodes in the same layer
const MARGIN: f32 = 24.0;
const PORT_RADIUS: f32 = 5.0;

/// Render a workflow document as a standalone SVG string. The viewport
/// is sized to fit the laid-out nodes plus a small margin.
pub fn render_workflow_svg(doc: &WorkflowDocument) -> String {
    let laid = layout(&doc.graph);
    emit_svg(&laid, &doc.graph)
}

struct LaidOutNode {
    x: f32,
    y: f32,
    height: f32,
    title: String,
    kind: WorkflowNodeKind,
    inputs: Vec<PortKind>,
    outputs: Vec<PortKind>,
}

struct Layout {
    nodes: BTreeMap<WorkflowNodeUuid, LaidOutNode>,
    width: f32,
    height: f32,
}

fn node_height(inputs: usize, outputs: usize) -> f32 {
    let ports = inputs.max(outputs).max(1) as f32;
    NODE_MIN_HEIGHT.max(NODE_HEADER_HEIGHT + ports * PORT_ROW_HEIGHT + 12.0)
}

fn port_y(node_y: f32, index: usize, total: usize) -> f32 {
    // Evenly distribute `total` ports inside the body of the node
    // (below the header). The first port sits one half-row down so the
    // stack is visually centered.
    let body_top = node_y + NODE_HEADER_HEIGHT;
    let body_bottom_padding = 12.0;
    let available = (node_height_of(total) - NODE_HEADER_HEIGHT - body_bottom_padding).max(0.0);
    let step = available / total.max(1) as f32;
    body_top + step * index as f32 + step * 0.5
}

fn node_height_of(ports: usize) -> f32 {
    node_height(ports, ports)
}

fn layout(graph: &WorkflowGraph) -> Layout {
    // 1. Collect (uuid, kind, inputs, outputs) for each node.
    struct NodeInfo {
        title: String,
        kind: WorkflowNodeKind,
        inputs: Vec<PortKind>,
        outputs: Vec<PortKind>,
    }
    let mut info: BTreeMap<WorkflowNodeUuid, NodeInfo> = BTreeMap::new();
    for (uuid, node) in graph.nodes() {
        info.insert(
            uuid,
            NodeInfo {
                title: node.op.title().to_string(),
                kind: node.op.clone(),
                inputs: node.op.inputs(),
                outputs: node.op.outputs(),
            },
        );
    }

    // 2. Longest-path layering.
    let preds = predecessors(graph);
    let layer_of = longest_path_layers(&info.keys().copied().collect::<Vec<_>>(), &preds);

    // 3. Bucket by layer. Within a layer, prefer the editor's y-position
    //    when available (so hand-crafted examples keep their ordering);
    //    fall back to uuid for determinism.
    let mut by_layer: BTreeMap<u32, Vec<WorkflowNodeUuid>> = BTreeMap::new();
    for (&uuid, &layer) in &layer_of {
        by_layer.entry(layer).or_default().push(uuid);
    }
    let positions: HashMap<WorkflowNodeUuid, GraphPos> =
        graph.nodes_pos().map(|(pos, uuid)| (uuid, pos)).collect();
    for bucket in by_layer.values_mut() {
        bucket.sort_by(|a, b| {
            let ya = positions.get(a).map(|p| p.y).unwrap_or(0.0);
            let yb = positions.get(b).map(|p| p.y).unwrap_or(0.0);
            ya.partial_cmp(&yb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
    }

    // 4. Assign (x, y) to each node.
    let mut laid: BTreeMap<WorkflowNodeUuid, LaidOutNode> = BTreeMap::new();
    let mut canvas_width: f32 = 0.0;
    let mut canvas_height: f32 = 0.0;
    for (&layer, bucket) in &by_layer {
        let mut y = MARGIN;
        for &uuid in bucket {
            let ni = info.get(&uuid).expect("layered node missing from info");
            let h = node_height(ni.inputs.len(), ni.outputs.len());
            let x = MARGIN + layer as f32 * LAYER_PITCH_X;
            laid.insert(
                uuid,
                LaidOutNode {
                    x,
                    y,
                    height: h,
                    title: ni.title.clone(),
                    kind: ni.kind.clone(),
                    inputs: ni.inputs.clone(),
                    outputs: ni.outputs.clone(),
                },
            );
            y += h + ROW_PITCH_Y;
            canvas_width = canvas_width.max(x + NODE_WIDTH + MARGIN);
            canvas_height = canvas_height.max(y - ROW_PITCH_Y + MARGIN);
        }
    }

    Layout {
        nodes: laid,
        width: canvas_width.max(MARGIN * 2.0),
        height: canvas_height.max(MARGIN * 2.0),
    }
}

fn predecessors(graph: &WorkflowGraph) -> HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>> {
    let mut preds: HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>> = HashMap::new();
    for (uuid, _) in graph.nodes() {
        preds.insert(uuid, Vec::new());
    }
    for Wire { from, to } in graph.wires() {
        preds.entry(to.node).or_default().push(from.node);
    }
    preds
}

fn longest_path_layers(
    ordered_uuids: &[WorkflowNodeUuid],
    preds: &HashMap<WorkflowNodeUuid, Vec<WorkflowNodeUuid>>,
) -> HashMap<WorkflowNodeUuid, u32> {
    // Iterative relaxation. Terminates in O(V) passes for DAGs; cycles
    // (which shouldn't exist in a valid workflow) are capped at V
    // iterations so we can't loop forever.
    let mut layer: HashMap<WorkflowNodeUuid, u32> =
        ordered_uuids.iter().map(|u| (*u, 0u32)).collect();
    let cap = ordered_uuids.len().max(1);
    for _ in 0..cap {
        let mut changed = false;
        for &uuid in ordered_uuids {
            if let Some(parents) = preds.get(&uuid) {
                let best = parents
                    .iter()
                    .map(|p| layer.get(p).copied().unwrap_or(0) + 1)
                    .max()
                    .unwrap_or(0);
                let entry = layer.entry(uuid).or_insert(0);
                if *entry < best {
                    *entry = best;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    layer
}

fn emit_svg(layout: &Layout, graph: &WorkflowGraph) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" \
         width=\"{w}\" height=\"{h}\" font-family=\"IBM Plex Sans, system-ui, sans-serif\">\n",
        w = layout.width.ceil() as u32,
        h = layout.height.ceil() as u32,
    ));
    push_background(&mut s, layout);

    // Wires first so they render below nodes.
    for Wire { from, to } in graph.wires() {
        let (Some(src), Some(dst)) = (layout.nodes.get(&from.node), layout.nodes.get(&to.node))
        else {
            continue;
        };
        let sx = src.x + NODE_WIDTH;
        let sy = port_y(src.y, from.output, src.outputs.len());
        let dx = dst.x;
        let dy = port_y(dst.y, to.input, dst.inputs.len());
        let dist = (dx - sx).abs().max(40.0);
        let c1x = sx + dist * 0.5;
        let c2x = dx - dist * 0.5;
        let kind = src
            .outputs
            .get(from.output)
            .copied()
            .unwrap_or(PortKind::Streamline);
        s.push_str(&format!(
            "  <path d=\"M {sx:.1} {sy:.1} C {c1x:.1} {sy:.1}, {c2x:.1} {dy:.1}, {dx:.1} {dy:.1}\" \
             stroke=\"{color}\" stroke-width=\"2\" fill=\"none\" opacity=\"0.85\"/>\n",
            color = port_fill(kind),
        ));
    }

    for (_uuid, node) in &layout.nodes {
        push_node(&mut s, node);
    }

    s.push_str("</svg>\n");
    s
}

fn push_background(s: &mut String, layout: &Layout) {
    s.push_str(&format!(
        "  <rect x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" fill=\"#1e2129\"/>\n",
        w = layout.width.ceil() as u32,
        h = layout.height.ceil() as u32,
    ));
}

fn push_node(s: &mut String, node: &LaidOutNode) {
    s.push_str(&format!(
        "  <g>\n    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
         rx=\"6\" ry=\"6\" fill=\"#2b303b\" stroke=\"#4a5165\" stroke-width=\"1\"/>\n",
        x = node.x,
        y = node.y,
        w = NODE_WIDTH,
        h = node.height,
    ));
    // Header strip.
    s.push_str(&format!(
        "    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{hh:.1}\" \
         rx=\"6\" ry=\"6\" fill=\"#343b4a\"/>\n",
        x = node.x,
        y = node.y,
        w = NODE_WIDTH,
        hh = NODE_HEADER_HEIGHT,
    ));
    s.push_str(&format!(
        "    <text x=\"{tx:.1}\" y=\"{ty:.1}\" fill=\"#e7eaf0\" font-size=\"13\" \
         font-weight=\"600\">{title}</text>\n",
        tx = node.x + 10.0,
        ty = node.y + NODE_HEADER_HEIGHT - 9.0,
        title = escape_xml(&truncate(&node.title, 24)),
    ));
    for (i, kind) in node.inputs.iter().enumerate() {
        let cy = port_y(node.y, i, node.inputs.len());
        s.push_str(&format!(
            "    <circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r}\" fill=\"{fill}\" \
             stroke=\"#1e2129\" stroke-width=\"1\"/>\n",
            cx = node.x,
            r = PORT_RADIUS,
            fill = port_fill(*kind),
        ));
        let label = input_port_label(&node.kind, i, *kind);
        s.push_str(&format!(
            "    <text x=\"{tx:.1}\" y=\"{ty:.1}\" fill=\"#b7bdcc\" font-size=\"10\" \
             text-anchor=\"start\" dominant-baseline=\"middle\">{label}</text>\n",
            tx = node.x + PORT_RADIUS + 6.0,
            ty = cy,
            label = escape_xml(&label),
        ));
    }
    for (i, kind) in node.outputs.iter().enumerate() {
        let cy = port_y(node.y, i, node.outputs.len());
        s.push_str(&format!(
            "    <circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r}\" fill=\"{fill}\" \
             stroke=\"#1e2129\" stroke-width=\"1\"/>\n",
            cx = node.x + NODE_WIDTH,
            r = PORT_RADIUS,
            fill = port_fill(*kind),
        ));
        let label = output_port_label(&node.kind, i, *kind);
        s.push_str(&format!(
            "    <text x=\"{tx:.1}\" y=\"{ty:.1}\" fill=\"#b7bdcc\" font-size=\"10\" \
             text-anchor=\"end\" dominant-baseline=\"middle\">{label}</text>\n",
            tx = node.x + NODE_WIDTH - PORT_RADIUS - 6.0,
            ty = cy,
            label = escape_xml(&label),
        ));
    }
    s.push_str("  </g>\n");
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(max_chars - 1).collect();
        out.push('…');
        out
    }
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Duplicated from `graph_viewer::port_color` in the GUI crate rather
/// than shared because docgen doesn't depend on egui (the GUI crate
/// pulls in wgpu; dragging that dependency chain into a docs generator
/// is not worth the DRY win for 17 RGB tuples).
fn port_fill(kind: PortKind) -> &'static str {
    match kind {
        PortKind::Streamline => "#52b5ff",
        PortKind::Volume => "#ffb14f",
        PortKind::Surface => "#91ffa1",
        PortKind::Parcellation => "#ff6c91",
        PortKind::ParcelSelection => "#ffd94f",
        PortKind::Cifti => "#78b0ff",
        PortKind::SurfaceScalars => "#d68bff",
        PortKind::VolumeScalars => "#ff9170",
        PortKind::SurfaceAppearance => "#aae291",
        PortKind::BundleSurface => "#8fe0c9",
        PortKind::BoundaryField => "#ffa060",
        PortKind::Fixels => "#ff7070",
        PortKind::FixelScalars => "#e870b4",
        PortKind::OdfField => "#c470e8",
        PortKind::OdxCatalog => "#a078dc",
        PortKind::VoxelMask => "#70dca0",
        PortKind::TrackingPlan => "#c8b460",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trxviz_core::workflow::{
        InPort, OutPort, WorkflowNode, WorkflowNodeKind, default_document,
    };

    fn tiny_doc() -> WorkflowDocument {
        let mut doc = default_document();
        let a = WorkflowNodeUuid(1);
        let b = WorkflowNodeUuid(2);
        doc.graph.insert_node(
            WorkflowNode {
                uuid: a,
                op: WorkflowNodeKind::StreamlineSource { source_id: 0 },
                label: "src".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph.insert_node(
            WorkflowNode {
                uuid: b,
                op: WorkflowNodeKind::SaveStreamlines {
                    output_path: String::new(),
                },
                label: "sink".into(),
            },
            GraphPos::new(0.0, 0.0),
        );
        doc.graph
            .connect(OutPort { node: a, output: 0 }, InPort { node: b, input: 0 });
        doc.next_node_uuid = 3;
        doc
    }

    #[test]
    fn svg_contains_a_node_rect_per_graph_node_and_one_wire_path() {
        let svg = render_workflow_svg(&tiny_doc());
        assert!(svg.contains("<svg"));
        // two node groups → at least two <rect> body elements plus two
        // header strips; we just check the title text is present.
        assert!(svg.contains("Streamline Source"));
        assert!(svg.contains("Save Streamlines"));
        // one connection → one <path>
        assert_eq!(svg.matches("<path").count(), 1);
    }

    #[test]
    fn layering_places_sink_to_the_right_of_source() {
        let doc = tiny_doc();
        let laid = layout(&doc.graph);
        let src = laid.nodes.get(&WorkflowNodeUuid(1)).unwrap();
        let sink = laid.nodes.get(&WorkflowNodeUuid(2)).unwrap();
        assert!(
            sink.x > src.x,
            "sink x={} should be right of source x={}",
            sink.x,
            src.x,
        );
    }
}

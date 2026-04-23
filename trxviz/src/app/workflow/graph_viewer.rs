//! Egui graph editor adapter.
//!
//! The canonical workflow graph lives in `WorkflowGraph` and is pure serde
//! (no egui dependency). The GUI editor, however, is built on `egui-snarl`,
//! which owns its own node-and-wire data structure. We bridge the two by
//! rebuilding a transient `Snarl<WorkflowNode>` view from the canonical
//! graph each frame, running the editor against that view, and syncing any
//! edits back into the canonical graph.
//!
//! This keeps Snarl out of the on-disk format and the evaluator while still
//! letting us reuse the existing editor widget.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;

use egui::emath::TSTransform;
use egui::{Pos2, Rect};
use egui_snarl::{
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlViewer},
};
use regex::Regex;

use trxviz_core::workflow::PortKind;

use super::*;

/// Build a fresh `Snarl<WorkflowNode>` view from the canonical graph. The
/// returned Snarl can be reused as persistent editor state.
pub fn snarl_from_graph(graph: &WorkflowGraph) -> Snarl<WorkflowNode> {
    let mut snarl = Snarl::<WorkflowNode>::new();
    let mut uuid_to_node_id: HashMap<WorkflowNodeUuid, NodeId> = HashMap::new();

    for (uuid, entry) in graph.entries() {
        let node_id = snarl.insert_node(Pos2::new(entry.pos.x, entry.pos.y), entry.node.clone());
        uuid_to_node_id.insert(*uuid, node_id);
    }

    for wire in graph.wires() {
        let (Some(from_node), Some(to_node)) = (
            uuid_to_node_id.get(&wire.from.node),
            uuid_to_node_id.get(&wire.to.node),
        ) else {
            continue;
        };
        snarl.connect(
            OutPinId {
                node: *from_node,
                output: wire.from.output,
            },
            InPinId {
                node: *to_node,
                input: wire.to.input,
            },
        );
    }

    snarl
}

/// Sync an edited Snarl editor view back into the document's canonical graph.
///
/// This assigns fresh UUIDs to any nodes that were added via the Snarl context
/// menu (they start with `WorkflowNodeUuid(0)`), rewrites the canonical
/// `WorkflowGraph` in-place, and updates `document.next_node_uuid`.
pub fn sync_graph_from_snarl(
    snarl: &mut Snarl<WorkflowNode>,
    document: &mut WorkflowDocument,
) -> GraphEditSummary {
    let old_graph = document.graph.clone();

    // Pass 1: assign UUIDs to any newly added (uuid == 0) nodes.
    let mut next = document.next_node_uuid.max(1);
    let node_ids: Vec<NodeId> = snarl.node_ids().map(|(id, _)| id).collect();
    for node_id in &node_ids {
        let info = match snarl.get_node_info_mut(*node_id) {
            Some(info) => info,
            None => continue,
        };
        if info.value.uuid.0 == 0 {
            info.value.uuid = WorkflowNodeUuid(next);
            next += 1;
        } else if info.value.uuid.0 >= next {
            next = info.value.uuid.0 + 1;
        }
    }

    // Pass 2: rebuild the canonical graph from the (now-uuid'd) snarl state.
    let mut graph = WorkflowGraph::new();
    for node_id in &node_ids {
        let Some(info) = snarl.get_node_info(*node_id) else {
            continue;
        };
        let pos = GraphPos::new(info.pos.x, info.pos.y);
        graph.insert_node(info.value.clone(), pos);
    }
    for (out_pin, in_pin) in snarl.wires() {
        let Some(from_info) = snarl.get_node_info(out_pin.node) else {
            continue;
        };
        let Some(to_info) = snarl.get_node_info(in_pin.node) else {
            continue;
        };
        graph.connect(
            OutPort {
                node: from_info.value.uuid,
                output: out_pin.output,
            },
            InPort {
                node: to_info.value.uuid,
                input: in_pin.input,
            },
        );
    }

    document.graph = graph;
    document.next_node_uuid = next;

    diff_graphs(&old_graph, &document.graph)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GraphEditSummary {
    pub topology_changed: bool,
    pub node_params_changed: bool,
    pub node_positions_changed: bool,
    pub selection_changed: bool,
}

impl GraphEditSummary {
    pub fn semantic_changed(self) -> bool {
        self.topology_changed || self.node_params_changed
    }
}

fn diff_graphs(before: &WorkflowGraph, after: &WorkflowGraph) -> GraphEditSummary {
    let before_nodes: HashMap<WorkflowNodeUuid, (&WorkflowNode, GraphPos)> = before
        .entries()
        .map(|(uuid, entry)| (*uuid, (&entry.node, entry.pos)))
        .collect();
    let after_nodes: HashMap<WorkflowNodeUuid, (&WorkflowNode, GraphPos)> = after
        .entries()
        .map(|(uuid, entry)| (*uuid, (&entry.node, entry.pos)))
        .collect();

    let before_ids: BTreeSet<WorkflowNodeUuid> = before_nodes.keys().copied().collect();
    let after_ids: BTreeSet<WorkflowNodeUuid> = after_nodes.keys().copied().collect();

    let mut summary = GraphEditSummary {
        topology_changed: before_ids != after_ids || wire_set(before) != wire_set(after),
        ..Default::default()
    };

    for uuid in before_ids.intersection(&after_ids) {
        let Some((before_node, before_pos)) = before_nodes.get(uuid).copied() else {
            continue;
        };
        let Some((after_node, after_pos)) = after_nodes.get(uuid).copied() else {
            continue;
        };
        if before_node != after_node {
            summary.node_params_changed = true;
        }
        if before_pos != after_pos {
            summary.node_positions_changed = true;
        }
    }

    summary
}

fn wire_set(graph: &WorkflowGraph) -> BTreeSet<(WorkflowNodeUuid, usize, WorkflowNodeUuid, usize)> {
    graph
        .wires()
        .map(|wire| {
            (
                wire.from.node,
                wire.from.output,
                wire.to.node,
                wire.to.input,
            )
        })
        .collect()
}

pub struct WorkflowGraphViewer<'a> {
    pub selected: &'a mut Option<WorkflowSelection>,
    pub focus_bounds: &'a mut Option<Rect>,
    pub viewport_rect: Rect,
    pub node_state: &'a HashMap<WorkflowNodeUuid, NodeEvalState>,
    pub assets: &'a [WorkflowAssetDocument],
    pub measured_node_sizes: &'a mut HashMap<WorkflowNodeUuid, NodeSize>,
    pub layout_reflow_nodes: &'a mut BTreeSet<WorkflowNodeUuid>,
}

impl SnarlViewer<WorkflowNode> for WorkflowGraphViewer<'_> {
    fn title(&mut self, node: &WorkflowNode) -> String {
        let base = if node.label.is_empty() {
            node.op.title().to_string()
        } else {
            node.label.clone()
        };
        match &node.op {
            WorkflowNodeKind::SurfaceSource { source_id } => {
                let guess = self.assets.iter().find_map(|asset| match asset {
                    WorkflowAssetDocument::Surface { id, path } if id == source_id => {
                        guess_surface_hemisphere(path)
                    }
                    _ => None,
                });
                match guess {
                    Some(HemisphereGuess::Left) => format!("{base} (Left)"),
                    Some(HemisphereGuess::Right) => format!("{base} (Right)"),
                    None => base,
                }
            }
            _ => base,
        }
    }

    fn inputs(&mut self, node: &WorkflowNode) -> usize {
        node.op.inputs().len()
    }

    fn outputs(&mut self, node: &WorkflowNode) -> usize {
        node.op.outputs().len()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node_op = &snarl[pin.id.node].op;
        let port = node_op.inputs()[pin.id.input];
        ui.label(input_port_label(node_op, pin.id.input, port));
        pin_info_for_port(port)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let port = snarl[pin.id.node].op.outputs()[pin.id.output];
        ui.horizontal(|ui| {
            ui.label(output_port_label(
                &snarl[pin.id.node].op,
                pin.id.output,
                port,
            ));
            // Reserve space so the pin circle doesn't overlap the trailing label glyphs.
            ui.add_space(18.0);
        });
        pin_info_for_port(port)
    }

    fn has_body(&mut self, _node: &WorkflowNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) {
        ui.small(match &snarl[node].op {
            WorkflowNodeKind::LimitStreamlines {
                limit,
                randomize,
                seed,
            } => {
                if *randomize {
                    format!("Keep {limit} streamlines, random seed {seed}")
                } else {
                    format!("Keep first {limit} streamlines")
                }
            }
            WorkflowNodeKind::GroupSelect { groups } => {
                let groups_csv = groups.to_csv();
                if groups_csv.trim().is_empty() {
                    "All groups".to_string()
                } else if groups_csv.trim() == "__none__" {
                    "No groups".to_string()
                } else {
                    format!("Groups: {groups_csv}")
                }
            }
            WorkflowNodeKind::RandomSubset { limit, seed } => {
                format!("Keep {limit} streamlines, seed {seed}")
            }
            WorkflowNodeKind::StreamlineDisplay { enabled, .. } => {
                if *enabled {
                    "Visible".to_string()
                } else {
                    "Hidden".to_string()
                }
            }
            WorkflowNodeKind::SphereQuery { center, radius_mm } => {
                format!(
                    "center=({:.1}, {:.1}, {:.1}) r={radius_mm:.1} mm",
                    center[0], center[1], center[2]
                )
            }
            WorkflowNodeKind::ParcelSelect { labels } => {
                let labels_csv = labels.to_csv();
                if labels_csv.trim().is_empty() {
                    "Labels: all nonzero".to_string()
                } else {
                    format!("Labels: {labels_csv}")
                }
            }
            WorkflowNodeKind::SaveStreamlines { output_path } => {
                if output_path.is_empty() {
                    "No output path".to_string()
                } else {
                    output_path.clone()
                }
            }
            other => other.title().to_string(),
        });
        if let Some(state) = self.node_state.get(&snarl[node].uuid)
            && let Some(execution) = &state.execution
        {
            let color = match execution {
                WorkflowExecutionStatus::Ready => egui::Color32::from_rgb(96, 210, 128),
                WorkflowExecutionStatus::NeverRun | WorkflowExecutionStatus::Stale => {
                    egui::Color32::from_rgb(255, 196, 96)
                }
                WorkflowExecutionStatus::Queued => egui::Color32::from_rgb(156, 168, 255),
                WorkflowExecutionStatus::Running => egui::Color32::from_rgb(110, 180, 255),
                WorkflowExecutionStatus::Failed(_) => egui::Color32::from_rgb(255, 112, 112),
            };
            ui.colored_label(color, execution.label());
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<WorkflowNode>) {
        let Some(out_kind) = snarl[from.id.node]
            .op
            .outputs()
            .get(from.id.output)
            .copied()
        else {
            return;
        };
        let Some(in_kind) = snarl[to.id.node].op.inputs().get(to.id.input).copied() else {
            return;
        };
        if out_kind != in_kind {
            return;
        }
        for &remote in &to.remotes {
            snarl.disconnect(remote, to.id);
        }
        snarl.connect(from.id, to.id);
    }

    fn final_node_rect(
        &mut self,
        node: NodeId,
        rect: egui::Rect,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) {
        // `rect` is in canvas (graph) coordinates. The snarl canvas is created
        // as a sublayer, so `rect_contains_pointer`'s `layer_id_at` check always
        // fails (it returns the parent layer ID, not the sublayer). We apply the
        // transform manually instead and compare screen-space positions directly.
        let to_global = ui
            .ctx()
            .layer_transform_to_global(ui.layer_id())
            .unwrap_or_default();
        let screen_rect = to_global * rect;
        if ui.input(|i| {
            i.pointer.primary_clicked()
                && i.pointer
                    .interact_pos()
                    .map_or(false, |p| screen_rect.contains(p))
        }) {
            *self.selected = Some(WorkflowSelection::Node(snarl[node].uuid));
        }

        let uuid = snarl[node].uuid;
        if uuid.0 == 0 {
            return;
        }
        let (_, size_changed) =
            store_node_measurement(self.measured_node_sizes, uuid, rect.size().x, rect.size().y);
        if size_changed && node_overlaps_snarl(snarl, node, self.measured_node_sizes) {
            self.layout_reflow_nodes.insert(uuid);
        }
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<WorkflowNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<WorkflowNode>) {
        let measured_node_sizes = &*self.measured_node_sizes;
        ui.menu_button("Streamline Filters", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                LimitStreamlinesOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                GroupSelectOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                RandomSubsetOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SphereQueryOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SurfaceDepthQueryOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                RemoveDuplicatesOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                TipPruneOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                PurifibreOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(ui, snarl, pos, MergeOp.into(), measured_node_sizes);
            add_node_button(
                ui,
                snarl,
                pos,
                AddGroupsFromParcellationOp.into(),
                measured_node_sizes,
            );
        });

        ui.menu_button("Parcellation", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                ParcelSelectOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(ui, snarl, pos, ParcelRoiOp.into(), measured_node_sizes);
            add_node_button(ui, snarl, pos, ParcelRoaOp.into(), measured_node_sizes);
            add_node_button(
                ui,
                snarl,
                pos,
                ParcelEndOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ParcelCropOp { keep_inside: true }.into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ParcelCropOp { keep_inside: false }.into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ParcelSurfaceBuildOp.into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ParcellationDisplayOp::default().into(),
                measured_node_sizes,
            );
        });

        ui.menu_button("Styling", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                ColorByDirectionOp.into(),
                measured_node_sizes,
            );
            add_node_button(ui, snarl, pos, ColorByGroupOp.into(), measured_node_sizes);
            add_node_button(
                ui,
                snarl,
                pos,
                ColorByDpvOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ColorByDpsOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                UniformColorOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SurfaceProjectionDensityOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SurfaceProjectionMeanDpsOp::default().into(),
                measured_node_sizes,
            );
        });

        ui.menu_button("Rendering", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                StreamlineDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                VolumeDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SurfaceDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                BundleSurfaceBuildOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                BundleSurfaceDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                StreamlineDirectionFieldOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                BoundaryGlyphDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                SaveStreamlinesOp::default().into(),
                measured_node_sizes,
            );
        });

        ui.menu_button("ODX", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                OdxFixelScalarSelectOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                ColorByFixelScalarsOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                OdxVolumeSelectOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                Fixel3DDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                Fixel2DDisplayOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                OdfGlyphRendererOp::default().into(),
                measured_node_sizes,
            );
        });

        ui.menu_button("Tractography", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                DipyTractographyOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                YehTractographyOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                RoiFromParcelOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                RoiFromVolumeOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                RoiFromShapeOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                PrepareHausdorffPlanOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                PrepareSimplePlanOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddRoiOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddRoaOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddEndRegionOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddNoEndOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddLimitingOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                AddTermOp::default().into(),
                measured_node_sizes,
            );
            add_node_button(
                ui,
                snarl,
                pos,
                VoxelMaskDisplayOp::default().into(),
                measured_node_sizes,
            );
        });
    }

    fn current_transform(&mut self, to_global: &mut TSTransform, _snarl: &mut Snarl<WorkflowNode>) {
        let Some(bounds) = self.focus_bounds.take() else {
            return;
        };

        let padded = bounds.expand2(egui::vec2(180.0, 120.0));
        let size = padded.size();
        let fit_scale_x = if size.x > 1.0 {
            self.viewport_rect.width() / size.x
        } else {
            2.0
        };
        let fit_scale_y = if size.y > 1.0 {
            self.viewport_rect.height() / size.y
        } else {
            2.0
        };
        let scaling = fit_scale_x.min(fit_scale_y).clamp(0.2, 2.0);
        to_global.scaling = scaling;
        to_global.translation =
            self.viewport_rect.center().to_vec2() - padded.center().to_vec2() * scaling;
    }

    fn has_node_menu(&mut self, _node: &WorkflowNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) {
        if ui.button("Delete").clicked() {
            snarl.remove_node(node);
            ui.close();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HemisphereGuess {
    Left,
    Right,
}

fn guess_surface_hemisphere(path: &Path) -> Option<HemisphereGuess> {
    let file_name = path.file_name()?.to_string_lossy();
    if left_hemisphere_regex().is_match(&file_name) {
        return Some(HemisphereGuess::Left);
    }
    if right_hemisphere_regex().is_match(&file_name) {
        return Some(HemisphereGuess::Right);
    }
    None
}

fn left_hemisphere_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?i)(left|[-_.]l[-_.])").expect("valid left regex"))
}

fn right_hemisphere_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?i)(right|[-_.]r[-_.])").expect("valid right regex"))
}

fn port_name(port: PortKind) -> &'static str {
    match port {
        PortKind::Streamline => "Streamline",
        PortKind::Volume => "Volume",
        PortKind::Surface => "Surface",
        PortKind::Parcellation => "Parcellation",
        PortKind::ParcelSelection => "Parcel Set",
        PortKind::Cifti => "CIFTI",
        PortKind::SurfaceScalars => "Surface Scalars",
        PortKind::VolumeScalars => "Volume Scalars",
        PortKind::SurfaceAppearance => "Surface Appearance",
        PortKind::BundleSurface => "Bundle Surface",
        PortKind::BoundaryField => "Boundary Field",
        PortKind::Fixels => "Fixels",
        PortKind::FixelScalars => "Fixel Scalars",
        PortKind::OdfField => "ODF Field",
        PortKind::OdxCatalog => "ODX Catalog",
        PortKind::VoxelMask => "Voxel Mask",
        PortKind::TrackingPlan => "Tracking Plan",
    }
}

fn input_port_label(node_kind: &WorkflowNodeKind, input_index: usize, port: PortKind) -> String {
    match node_kind {
        WorkflowNodeKind::OdfGlyphRenderer { .. } => match input_index {
            0 => "ODF Field".to_string(),
            1 => "Opacity Scalars".to_string(),
            2 => "Size Scalars".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::Purifibre { .. } => match input_index {
            0 => "Streamlines".to_string(),
            1 => "Direction field".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::SurfaceOverlayStack { layers } => {
            if input_index == 0 {
                "Surface".to_string()
            } else {
                let layer_index = input_index - 1;
                let layer_name = if layer_index == 0 {
                    "Layer 0: Base".to_string()
                } else {
                    format!("Layer {layer_index}")
                };
                let legend = layers
                    .get(layer_index)
                    .map(|layer| layer.legend_label.trim())
                    .filter(|legend| !legend.is_empty())
                    .unwrap_or("");
                if legend.is_empty() {
                    format!("{layer_name} Scalars")
                } else {
                    format!("{layer_name} Scalars ({legend})")
                }
            }
        }
        _ => port_name(port).to_string(),
    }
}

fn output_port_label(node_kind: &WorkflowNodeKind, output_index: usize, port: PortKind) -> String {
    match node_kind {
        WorkflowNodeKind::OdxVolumeSelect { .. } => match output_index {
            0 => "Volume".to_string(),
            1 => "Volume Scalars".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::PrepareHausdorffPlan { .. } => match output_index {
            0 => "Plan".to_string(),
            1 => "Seed Mask".to_string(),
            2 => "Limiting Mask".to_string(),
            3 => "No-End Mask".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::PrepareSimplePlan { .. } => match output_index {
            0 => "Plan".to_string(),
            _ => port_name(port).to_string(),
        },
        WorkflowNodeKind::Purifibre { .. } => match output_index {
            // Output 0 is the input streamlines passthrough with the
            // FICO DPS field attached — useful for visualizing the
            // score distribution before any filtering happens.
            0 => "Scored (all)".to_string(),
            // Output 1 has only streamlines that survived the
            // discard-fraction cutoff.
            1 => "Filtered".to_string(),
            _ => port_name(port).to_string(),
        },
        _ => port_name(port).to_string(),
    }
}

fn add_node_button(
    ui: &mut egui::Ui,
    snarl: &mut Snarl<WorkflowNode>,
    pos: Pos2,
    op: WorkflowNodeKind,
    measured_node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
) {
    if ui.button(op.title()).clicked() {
        let node = WorkflowNode {
            uuid: WorkflowNodeUuid(0),
            label: op.title().to_string(),
            op,
        };
        let insert_pos = find_nearest_free_node_position(snarl, pos, &node, measured_node_sizes);
        snarl.insert_node(insert_pos, node);
        ui.close();
    }
}

fn find_nearest_free_node_position(
    snarl: &Snarl<WorkflowNode>,
    desired_pos: Pos2,
    node: &WorkflowNode,
    measured_node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
) -> Pos2 {
    let size = estimate_workflow_node_size(node);
    if !snarl_position_overlaps(snarl, desired_pos, size, measured_node_sizes, None) {
        return desired_pos;
    }

    const GRID_STEP: f32 = 40.0;
    for radius in 1..=20 {
        let radius = radius as f32;
        for dy in -((radius) as i32)..=((radius) as i32) {
            for dx in -((radius) as i32)..=((radius) as i32) {
                if dx.abs() != radius as i32 && dy.abs() != radius as i32 {
                    continue;
                }
                let candidate = Pos2::new(
                    desired_pos.x + dx as f32 * GRID_STEP,
                    desired_pos.y + dy as f32 * GRID_STEP,
                );
                if !snarl_position_overlaps(snarl, candidate, size, measured_node_sizes, None) {
                    return candidate;
                }
            }
        }
    }

    desired_pos
}

fn store_node_measurement(
    measured_node_sizes: &mut HashMap<WorkflowNodeUuid, NodeSize>,
    uuid: WorkflowNodeUuid,
    width: f32,
    height: f32,
) -> (NodeSize, bool) {
    let measured = NodeSize::new(width, height);
    let prior = measured_node_sizes.insert(uuid, measured);
    let size_changed = prior.map_or(true, |size| {
        (size.width - measured.width).abs() > 8.0 || (size.height - measured.height).abs() > 8.0
    });
    (measured, size_changed)
}

fn node_overlaps_snarl(
    snarl: &Snarl<WorkflowNode>,
    node_id: NodeId,
    measured_node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
) -> bool {
    let Some(info) = snarl.get_node_info(node_id) else {
        return false;
    };
    let size = measured_node_sizes
        .get(&info.value.uuid)
        .copied()
        .unwrap_or_else(|| estimate_workflow_node_size(&info.value));
    snarl_position_overlaps(snarl, info.pos, size, measured_node_sizes, Some(node_id))
}

fn snarl_position_overlaps(
    snarl: &Snarl<WorkflowNode>,
    pos: Pos2,
    size: NodeSize,
    measured_node_sizes: &HashMap<WorkflowNodeUuid, NodeSize>,
    ignore: Option<NodeId>,
) -> bool {
    let candidate = expanded_node_rect(pos, size);
    snarl.node_ids().any(|(other_id, _)| {
        if Some(other_id) == ignore {
            return false;
        }
        let Some(other_node) = snarl.get_node_info(other_id) else {
            return false;
        };
        let other_size = measured_node_sizes
            .get(&other_node.value.uuid)
            .copied()
            .unwrap_or_else(|| estimate_workflow_node_size(&other_node.value));
        candidate.intersects(expanded_node_rect(other_node.pos, other_size))
    })
}

fn expanded_node_rect(pos: Pos2, size: NodeSize) -> Rect {
    Rect::from_min_size(pos, egui::vec2(size.width, size.height)).expand(20.0)
}

fn pin_info_for_port(port: PortKind) -> PinInfo {
    let color = match port {
        PortKind::Streamline => egui::Color32::from_rgb(82, 181, 255),
        PortKind::Volume => egui::Color32::from_rgb(255, 177, 79),
        PortKind::Surface => egui::Color32::from_rgb(145, 255, 161),
        PortKind::Parcellation => egui::Color32::from_rgb(255, 108, 145),
        PortKind::ParcelSelection => egui::Color32::from_rgb(255, 217, 79),
        PortKind::Cifti => egui::Color32::from_rgb(120, 176, 255),
        PortKind::SurfaceScalars => egui::Color32::from_rgb(214, 139, 255),
        PortKind::VolumeScalars => egui::Color32::from_rgb(255, 145, 112),
        PortKind::SurfaceAppearance => egui::Color32::from_rgb(170, 226, 145),
        PortKind::BundleSurface => egui::Color32::from_rgb(143, 224, 201),
        PortKind::BoundaryField => egui::Color32::from_rgb(255, 160, 96),
        PortKind::Fixels => egui::Color32::from_rgb(255, 112, 112),
        PortKind::FixelScalars => egui::Color32::from_rgb(232, 112, 180),
        PortKind::OdfField => egui::Color32::from_rgb(196, 112, 232),
        PortKind::OdxCatalog => egui::Color32::from_rgb(160, 120, 220),
        PortKind::VoxelMask => egui::Color32::from_rgb(112, 220, 160),
        PortKind::TrackingPlan => egui::Color32::from_rgb(200, 180, 96),
    };
    PinInfo::circle().with_fill(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn moving_node_only_reports_position_change() {
        let mut document = default_document();
        make_node(
            &mut document,
            WorkflowNodeKind::LimitStreamlines {
                limit: 10,
                randomize: false,
                seed: 1,
            },
            GraphPos::new(10.0, 20.0),
        );
        let mut snarl = snarl_from_graph(&document.graph);
        let node_id = snarl
            .node_ids()
            .map(|(id, _)| id)
            .next()
            .expect("node inserted into snarl");
        let info = snarl
            .get_node_info_mut(node_id)
            .expect("node info must exist");
        info.pos.x += 50.0;

        let summary = sync_graph_from_snarl(&mut snarl, &mut document);

        assert!(summary.node_positions_changed);
        assert!(!summary.topology_changed);
        assert!(!summary.node_params_changed);
        assert!(!summary.semantic_changed());
    }

    #[test]
    fn hemisphere_guess_finds_left_word() {
        assert_eq!(
            guess_surface_hemisphere(Path::new("subject_left.surf.gii")),
            Some(HemisphereGuess::Left)
        );
    }

    #[test]
    fn hemisphere_guess_finds_right_token() {
        assert_eq!(
            guess_surface_hemisphere(Path::new("subject.R.inflated.surf.gii")),
            Some(HemisphereGuess::Right)
        );
    }

    #[test]
    fn hemisphere_guess_ignores_unmatched_names() {
        assert_eq!(
            guess_surface_hemisphere(Path::new("subject_inflated.surf.gii")),
            None
        );
    }

    #[test]
    fn hemisphere_guess_matches_mixed_separator_tokens() {
        assert_eq!(
            guess_surface_hemisphere(Path::new("subject-L_inflated.surf.gii")),
            Some(HemisphereGuess::Left)
        );
        assert_eq!(
            guess_surface_hemisphere(Path::new("subject.R-inflated.surf.gii")),
            Some(HemisphereGuess::Right)
        );
    }

    #[test]
    fn manual_insert_avoids_existing_node_overlap() {
        let mut snarl = Snarl::new();
        snarl.insert_node(
            Pos2::new(0.0, 0.0),
            WorkflowNode {
                uuid: WorkflowNodeUuid(1),
                label: "Existing".into(),
                op: WorkflowNodeKind::LimitStreamlines {
                    limit: 10,
                    randomize: false,
                    seed: 1,
                },
            },
        );
        let new_node = WorkflowNode {
            uuid: WorkflowNodeUuid(0),
            label: "Display".into(),
            op: WorkflowNodeKind::StreamlineDisplay {
                enabled: true,
                render_style: trxviz_core::data::trx_data::RenderStyle::Flat,
                tube_radius_mm: trxviz_core::units::Millimeters(0.4),
                tube_sides: 8,
                slab_half_width_mm: trxviz_core::units::Millimeters(5.0),
            },
        };

        let pos = find_nearest_free_node_position(&snarl, Pos2::ZERO, &new_node, &HashMap::new());

        assert_ne!(pos, Pos2::ZERO);
        assert!(!snarl_position_overlaps(
            &snarl,
            pos,
            estimate_workflow_node_size(&new_node),
            &HashMap::new(),
            None
        ));
    }

    #[test]
    fn store_node_measurement_updates_cache() {
        let mut cache = HashMap::new();

        let (size, changed) = store_node_measurement(&mut cache, WorkflowNodeUuid(9), 240.0, 120.0);

        assert_eq!(size, NodeSize::new(240.0, 120.0));
        assert!(changed);
        assert_eq!(
            cache.get(&WorkflowNodeUuid(9)),
            Some(&NodeSize::new(240.0, 120.0))
        );
    }

    #[test]
    fn odf_glyph_renderer_volume_inputs_have_specific_labels() {
        let node: WorkflowNodeKind = OdfGlyphRendererOp::default().into();
        assert_eq!(
            input_port_label(&node, 1, PortKind::VolumeScalars),
            "Opacity Scalars"
        );
        assert_eq!(
            input_port_label(&node, 2, PortKind::VolumeScalars),
            "Size Scalars"
        );
    }

    #[test]
    fn odx_volume_select_outputs_have_specific_labels() {
        let node: WorkflowNodeKind = OdxVolumeSelectOp::default().into();
        assert_eq!(output_port_label(&node, 0, PortKind::Volume), "Volume");
        assert_eq!(
            output_port_label(&node, 1, PortKind::VolumeScalars),
            "Volume Scalars"
        );
    }
}

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

use egui::emath::TSTransform;
use egui::{Pos2, Rect};
use egui_snarl::{
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlViewer},
};

use trxviz_core::data::loaded_files::VolumeColormap;
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::renderer::mesh_renderer::SurfaceColormap;
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
    let before_nodes: HashMap<WorkflowNodeUuid, (&WorkflowNode, GraphPos)> =
        before.entries().map(|(uuid, entry)| (*uuid, (&entry.node, entry.pos))).collect();
    let after_nodes: HashMap<WorkflowNodeUuid, (&WorkflowNode, GraphPos)> =
        after.entries().map(|(uuid, entry)| (*uuid, (&entry.node, entry.pos))).collect();

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
}

impl SnarlViewer<WorkflowNode> for WorkflowGraphViewer<'_> {
    fn title(&mut self, node: &WorkflowNode) -> String {
        if node.label.is_empty() {
            node.kind.title().to_string()
        } else {
            node.label.clone()
        }
    }

    fn inputs(&mut self, node: &WorkflowNode) -> usize {
        node.kind.inputs().len()
    }

    fn outputs(&mut self, node: &WorkflowNode) -> usize {
        node.kind.outputs().len()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        ui.label(port_name(snarl[pin.id.node].kind.inputs()[pin.id.input]));
        pin_info_for_port(snarl[pin.id.node].kind.inputs()[pin.id.input])
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        ui.label(port_name(snarl[pin.id.node].kind.outputs()[pin.id.output]));
        pin_info_for_port(snarl[pin.id.node].kind.outputs()[pin.id.output])
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
        ui.small(match &snarl[node].kind {
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
            WorkflowNodeKind::GroupSelect { groups_csv } => {
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
            WorkflowNodeKind::ParcelSelect { labels_csv } => {
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
            .kind
            .outputs()
            .get(from.id.output)
            .copied()
        else {
            return;
        };
        let Some(in_kind) = snarl[to.id.node].kind.inputs().get(to.id.input).copied() else {
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
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<WorkflowNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<WorkflowNode>) {
        ui.menu_button("Streamline Filters", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::LimitStreamlines {
                    limit: 30_000,
                    randomize: false,
                    seed: 1,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::GroupSelect {
                    groups_csv: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::RandomSubset {
                    limit: 10_000,
                    seed: 1,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SphereQuery {
                    center: [0.0, 0.0, 0.0],
                    radius_mm: 10.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceDepthQuery { depth_mm: 2.0 },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::RemoveDuplicates);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::Merge);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::AddGroupsFromParcellation);
        });

        ui.menu_button("Parcellation", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcelSelect {
                    labels_csv: String::new(),
                },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelROI);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelROA);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcelEnd { endpoint_count: 1 },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelLimiting);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelTerminative);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelSurfaceBuild);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcellationDisplay {
                    labels_csv: String::new(),
                    opacity: 0.9,
                },
            );
        });

        ui.menu_button("Styling", |ui| {
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ColorByDirection);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ColorByGroup);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ColorByDPV {
                    field: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ColorByDPS {
                    field: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::UniformColor {
                    color: [0.95, 0.8, 0.2, 1.0],
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceProjectionDensity { depth_mm: 2.0 },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceProjectionMeanDps {
                    depth_mm: 2.0,
                    field: String::new(),
                },
            );
        });

        ui.menu_button("Rendering", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::StreamlineDisplay {
                    enabled: true,
                    render_style: RenderStyle::Flat,
                    tube_radius_mm: 0.4,
                    tube_sides: 8,
                    slab_half_width_mm: 5.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
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
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BundleSurfaceBuild {
                    per_group: false,
                    voxel_size_mm: 2.0,
                    threshold: 3.0,
                    smooth_sigma: 0.5,
                    min_component_volume_mm3: 0.0,
                    opacity: 0.5,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BundleSurfaceDisplay {
                    color_mode: BundleSurfaceColorMode::Solid,
                    outline_thickness: 1.15,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BoundaryFieldBuild {
                    voxel_size_mm: default_boundary_field_voxel_size_mm(),
                    sphere_lod: default_boundary_field_sphere_lod(),
                    normalization: default_boundary_field_normalization(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BoundaryGlyphDisplay {
                    enabled: default_enabled(),
                    scale: default_boundary_glyph_scale(),
                    density_3d_step: default_boundary_glyph_density_3d_step(),
                    slice_density_step: default_boundary_glyph_slice_density_step(),
                    color_mode: default_boundary_glyph_color_mode(),
                    min_contacts: default_boundary_glyph_min_contacts(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SaveStreamlines {
                    output_path: String::new(),
                },
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

fn port_name(port: PortKind) -> &'static str {
    match port {
        PortKind::Streamline => "Streamline",
        PortKind::Volume => "Volume",
        PortKind::Surface => "Surface",
        PortKind::Parcellation => "Parcellation",
        PortKind::ParcelSelection => "Parcel Set",
        PortKind::SurfaceMap => "Surface Map",
        PortKind::BundleSurface => "Bundle Surface",
        PortKind::BoundaryField => "Boundary Field",
    }
}

fn add_node_button(
    ui: &mut egui::Ui,
    snarl: &mut Snarl<WorkflowNode>,
    pos: Pos2,
    kind: WorkflowNodeKind,
) {
    if ui.button(kind.title()).clicked() {
        snarl.insert_node(
            pos,
            WorkflowNode {
                uuid: WorkflowNodeUuid(0),
                label: kind.title().to_string(),
                kind,
            },
        );
        ui.close();
    }
}

fn pin_info_for_port(port: PortKind) -> PinInfo {
    let color = match port {
        PortKind::Streamline => egui::Color32::from_rgb(82, 181, 255),
        PortKind::Volume => egui::Color32::from_rgb(255, 177, 79),
        PortKind::Surface => egui::Color32::from_rgb(145, 255, 161),
        PortKind::Parcellation => egui::Color32::from_rgb(255, 108, 145),
        PortKind::ParcelSelection => egui::Color32::from_rgb(255, 217, 79),
        PortKind::SurfaceMap => egui::Color32::from_rgb(214, 139, 255),
        PortKind::BundleSurface => egui::Color32::from_rgb(143, 224, 201),
        PortKind::BoundaryField => egui::Color32::from_rgb(255, 160, 96),
    };
    PinInfo::circle().with_fill(color)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

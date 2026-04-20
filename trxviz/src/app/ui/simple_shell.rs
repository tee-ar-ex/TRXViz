use crate::app::workflow::{
    self, SimpleDisplayBinding, SimpleStreamlineBinding, SimpleSurfaceBinding,
    WorkflowAssetDocument, WorkflowNodeKind, WorkflowNodeUuid, classify_workflow_editability,
};
use trxviz_core::data::loaded_files::VolumeColormap;

impl super::super::TrxVizApp {
    pub(in crate::app) fn show_simple_shell(&mut self, ctx: &egui::Context) -> bool {
        let editability = classify_workflow_editability(&self.workflow.document);
        let assets = self.workflow.document.assets.clone();
        let mut open_files_requested = false;

        egui::SidePanel::left("simple_sidebar")
            .default_width(320.0)
            .min_width(280.0)
            .max_width(360.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Data & Display");
                    ui.separator();
                    if ui.button("Open Files...").clicked() {
                        open_files_requested = true;
                    }
                    ui.add_space(8.0);
                    self.show_messages(ui);

                    if editability.has_read_only_assets() {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.strong("Some workflow branches are Advanced-only");
                            if let Some(reason) = editability.first_reason() {
                                ui.small(reason);
                            }
                            if ui.button("Switch to Advanced").clicked() {
                                self.ui_mode = crate::app::state::UiMode::Advanced;
                            }
                        });
                        ui.add_space(8.0);
                    }

                    if assets.is_empty() {
                        ui.small("Drop tractograms, volumes, surfaces, or parcellations to start.");
                        return;
                    }

                    for asset in assets {
                        match asset {
                            WorkflowAssetDocument::Streamlines { id, .. } => {
                                if let Some(binding) =
                                    editability.bindings.streamline.get(&id).copied()
                                {
                                    self.show_simple_streamline_asset(ui, id, binding, None);
                                } else {
                                    self.show_simple_streamline_asset(
                                        ui,
                                        id,
                                        self.fallback_streamline_binding(id),
                                        editability.reason_for(id),
                                    );
                                }
                            }
                            WorkflowAssetDocument::Volume { id, .. } => {
                                if let Some(binding) = editability.bindings.volume.get(&id).copied()
                                {
                                    self.show_simple_volume_asset(ui, id, binding, None);
                                } else {
                                    self.show_simple_volume_asset(
                                        ui,
                                        id,
                                        self.fallback_display_binding(id),
                                        editability.reason_for(id),
                                    );
                                }
                            }
                            WorkflowAssetDocument::Surface { id, .. } => {
                                if let Some(binding) =
                                    editability.bindings.surface.get(&id).copied()
                                {
                                    self.show_simple_surface_asset(ui, id, binding, None);
                                } else {
                                    self.show_simple_surface_asset(
                                        ui,
                                        id,
                                        self.fallback_surface_binding(id),
                                        editability.reason_for(id),
                                    );
                                }
                            }
                            WorkflowAssetDocument::Parcellation { id, .. } => {
                                if let Some(binding) =
                                    editability.bindings.parcellation.get(&id).copied()
                                {
                                    self.show_simple_parcellation_asset(ui, id, binding, None);
                                } else {
                                    self.show_simple_parcellation_asset(
                                        ui,
                                        id,
                                        self.fallback_display_binding(id),
                                        editability.reason_for(id),
                                    );
                                }
                            }
                            WorkflowAssetDocument::Cifti { id, path, .. } => {
                                ui.group(|ui| {
                                    ui.strong(format!("CIFTI {}", id));
                                    ui.small(path.display().to_string());
                                    show_advanced_only_reason(ui, editability.reason_for(id));
                                });
                            }
                            WorkflowAssetDocument::Odx { id, path, .. } => {
                                ui.group(|ui| {
                                    ui.strong(format!("ODX {}", id));
                                    ui.small(path.display().to_string());
                                    show_advanced_only_reason(ui, editability.reason_for(id));
                                });
                            }
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.scene_is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.vertical(|ui| {
                        ui.heading("Drop files to start");
                        ui.small("TRX, TRK, TCK, VTK, TinyTrack, NIfTI, GIFTI, CIFTI (.dscalar/.dtseries/.dlabel/.pscalar), parcellations, and ODX/fib.gz/fz/pam5/mif are supported.");
                        ui.add_space(12.0);
                        if ui.button("Open Files...").clicked() {
                            open_files_requested = true;
                        }
                    });
                });
                return;
            }

            self.show_embedded_preview(ui);
        });

        open_files_requested
    }

    fn show_messages(&mut self, ui: &mut egui::Ui) {
        if let Some(message) = self.status_msg.clone() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(96, 210, 128), &message);
                    if ui.button("Clear").clicked() {
                        self.status_msg = None;
                    }
                });
            });
            ui.add_space(8.0);
        }
        if !self
            .scene
            .trx_files
            .iter()
            .all(|trx| trx.import_warnings.is_empty())
        {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 214, 102),
                    "Imported streamline warnings",
                );
                for trx in self
                    .scene
                    .trx_files
                    .iter()
                    .filter(|trx| !trx.import_warnings.is_empty())
                {
                    ui.small(format!("{}:", trx.name));
                    for warning in &trx.import_warnings {
                        ui.label(warning);
                    }
                }
            });
            ui.add_space(8.0);
        }
        if let Some(message) = self.error_msg.clone() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(255, 110, 110), &message);
                    if ui.button("Dismiss").clicked() {
                        self.error_msg = None;
                    }
                });
            });
            ui.add_space(8.0);
        }
    }

    fn show_simple_streamline_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleStreamlineBinding,
        read_only_reason: Option<&str>,
    ) {
        let Some(trx) = self.scene.trx_files.iter().find(|asset| asset.id == id) else {
            return;
        };
        let title = truncate_simple_label(&trx.name, 36);
        let path = truncate_simple_label(&trx.path.display().to_string(), 52);
        let nb_streamlines = trx.data.nb_streamlines;
        let nb_vertices = trx.data.nb_vertices;
        let group_count = trx.data.groups.len();
        let import_warnings = trx.import_warnings.clone();

        egui::CollapsingHeader::new(title)
            .id_salt(("simple_streamline_asset", id))
            .default_open(true)
            .show(ui, |ui| {
                ui.small(path);
                ui.small(format!(
                    "{nb_streamlines} streamlines, {nb_vertices} vertices, {group_count} groups"
                ));
                for warning in &import_warnings {
                    ui.colored_label(egui::Color32::from_rgb(255, 214, 102), warning);
                }
                if let Some(reason) = read_only_reason {
                    show_advanced_only_reason(ui, Some(reason));
                    return;
                }

                let original_op = self.workflow_node_op(binding.display).cloned();
                if let Some(WorkflowNodeKind::StreamlineDisplay {
                    enabled,
                    slab_half_width_mm,
                    ..
                }) = self.workflow_node_op_mut(binding.display)
                {
                    ui.checkbox(enabled, "Visible");
                    ui.add(
                        egui::Slider::new(&mut slab_half_width_mm.0, 0.0..=50.0)
                            .text("Slice slab half-width"),
                    );
                }
                if let Some(original_op) = original_op {
                    self.finish_simple_render_only_edit(binding.display, original_op);
                }
            });
    }

    fn show_simple_volume_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleDisplayBinding,
        read_only_reason: Option<&str>,
    ) {
        let Some(volume) = self.scene.nifti_files.iter().find(|asset| asset.id == id) else {
            return;
        };
        let name = truncate_simple_label(&volume.name, 36);
        let dims = volume.volume.dims;
        egui::CollapsingHeader::new(name)
            .id_salt(("simple_volume_asset", id))
            .default_open(true)
            .show(ui, |ui| {
                ui.small(format!("{} x {} x {}", dims[0], dims[1], dims[2]));
                if let Some(reason) = read_only_reason {
                    show_advanced_only_reason(ui, Some(reason));
                    return;
                }

                let original_op = self.workflow_node_op(binding.display).cloned();
                if let Some(WorkflowNodeKind::VolumeDisplay {
                    colormap,
                    opacity,
                    window_center,
                    window_width,
                }) = self.workflow_node_op_mut(binding.display)
                {
                    opacity_checkbox(ui, opacity, "Visible");
                    egui::ComboBox::from_id_salt(("simple_volume_colormap", binding.display.0))
                        .selected_text(colormap.label())
                        .show_ui(ui, |ui| {
                            for choice in VolumeColormap::ALL {
                                ui.selectable_value(colormap, *choice, choice.label());
                            }
                        });
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(egui::Slider::new(window_center, 0.0..=1.0).text("Window center"));
                    ui.add(egui::Slider::new(window_width, 0.01..=2.0).text("Window width"));
                }
                if let Some(original_op) = original_op {
                    self.finish_simple_render_only_edit(binding.display, original_op);
                }
            });
    }

    fn show_simple_surface_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleSurfaceBinding,
        read_only_reason: Option<&str>,
    ) {
        let Some(surface) = self
            .scene
            .gifti_surfaces
            .iter()
            .find(|asset| asset.id == id)
        else {
            return;
        };
        let name = truncate_simple_label(&surface.name, 36);
        let path = truncate_simple_label(&surface.path.display().to_string(), 52);
        let vertex_count = surface.data.vertices.len();
        let triangle_count = surface.data.indices.len() / 3;
        egui::CollapsingHeader::new(name)
            .id_salt(("simple_surface_asset", id))
            .default_open(false)
            .show(ui, |ui| {
                ui.small(path.clone());
                ui.small(format!(
                    "{vertex_count} vertices, {triangle_count} triangles"
                ));
                if let Some(reason) = read_only_reason {
                    show_advanced_only_reason(ui, Some(reason));
                    return;
                }

                let original_display = self.workflow_node_op(binding.display).cloned();
                let original_overlay = binding
                    .overlay_stack
                    .and_then(|uuid| self.workflow_node_op(uuid).cloned());
                if let Some(WorkflowNodeKind::SurfaceDisplay { opacity, .. }) =
                    self.workflow_node_op_mut(binding.display)
                {
                    ui.label("Shared");
                    opacity_checkbox(ui, opacity, "Visible");
                }
                let mesh_style_uses_overlay = if let Some(overlay_uuid) = binding.overlay_stack {
                    if let Some(WorkflowNodeKind::SurfaceOverlayStack { layers }) =
                        self.workflow_node_op_mut(overlay_uuid)
                    {
                        if let Some(base) = layers.first_mut() {
                            ui.separator();
                            ui.label("3D Mesh");
                            let mut mesh_rgb = [
                                base.solid_color[0],
                                base.solid_color[1],
                                base.solid_color[2],
                            ];
                            if ui.color_edit_button_rgb(&mut mesh_rgb).changed() {
                                base.solid_color[0] = mesh_rgb[0];
                                base.solid_color[1] = mesh_rgb[1];
                                base.solid_color[2] = mesh_rgb[2];
                            }
                            ui.add(
                                egui::Slider::new(&mut base.opacity, 0.0..=1.0)
                                    .text("Mesh opacity"),
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !mesh_style_uses_overlay
                    && let Some(WorkflowNodeKind::SurfaceDisplay { color, opacity, .. }) =
                        self.workflow_node_op_mut(binding.display)
                {
                    ui.separator();
                    ui.label("3D Mesh");
                    ui.color_edit_button_rgb(color);
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Mesh opacity"));
                }
                if let Some(WorkflowNodeKind::SurfaceDisplay {
                    outline_color,
                    outline_thickness,
                    opacity,
                    ..
                }) = self.workflow_node_op_mut(binding.display)
                {
                    ui.separator();
                    ui.label("2D Slice Outline");
                    ui.color_edit_button_rgb(outline_color);
                    ui.add(
                        egui::Slider::new(outline_thickness, 0.25..=8.0).text("Outline thickness"),
                    );
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Outline opacity"));
                }
                if let Some(original_display) = original_display {
                    self.finish_simple_render_only_edit(binding.display, original_display);
                }
                if let (Some(overlay_uuid), Some(original_overlay)) =
                    (binding.overlay_stack, original_overlay)
                {
                    self.finish_simple_render_only_edit(overlay_uuid, original_overlay);
                }
            });
    }

    fn show_simple_parcellation_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleDisplayBinding,
        read_only_reason: Option<&str>,
    ) {
        let Some(parcel) = self
            .scene
            .parcellations
            .iter()
            .find(|asset| asset.asset.id == id)
        else {
            return;
        };
        let name = truncate_simple_label(&parcel.asset.name, 36);
        let path = truncate_simple_label(&parcel.asset.path.display().to_string(), 52);
        let label_count = parcel
            .asset
            .data
            .label_table
            .keys()
            .copied()
            .filter(|label| label.0 != 0)
            .count();
        egui::CollapsingHeader::new(name)
            .id_salt(("simple_parcellation_asset", id))
            .default_open(false)
            .show(ui, |ui| {
                ui.small(path.clone());
                ui.small(format!("{label_count} labels"));
                if let Some(reason) = read_only_reason {
                    show_advanced_only_reason(ui, Some(reason));
                    return;
                }

                let original_op = self.workflow_node_op(binding.display).cloned();
                if let Some(WorkflowNodeKind::ParcellationDisplay { opacity, .. }) =
                    self.workflow_node_op_mut(binding.display)
                {
                    opacity_checkbox(ui, opacity, "Visible");
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                }
                if let Some(original_op) = original_op {
                    self.finish_simple_render_only_edit(binding.display, original_op);
                }
            });
    }

    fn workflow_node_op_mut(&mut self, uuid: WorkflowNodeUuid) -> Option<&mut WorkflowNodeKind> {
        self.workflow
            .document
            .graph
            .get_mut(uuid)
            .map(|node| &mut node.op)
    }

    fn workflow_node_op(&self, uuid: WorkflowNodeUuid) -> Option<&WorkflowNodeKind> {
        self.workflow.document.graph.get(uuid).map(|node| &node.op)
    }

    fn sync_editor_node_from_document(&mut self, node_uuid: WorkflowNodeUuid) {
        let Some(node_copy) = self.workflow.document.graph.get(node_uuid).cloned() else {
            return;
        };
        if let Some(node_id) = self
            .workflow
            .editor_snarl
            .nodes_ids_data()
            .find_map(|(id, value)| (value.value.uuid == node_uuid).then_some(id))
            && let Some(info) = self.workflow.editor_snarl.get_node_info_mut(node_id)
        {
            info.value = node_copy;
        }
    }

    fn finish_simple_render_only_edit(
        &mut self,
        node_uuid: WorkflowNodeUuid,
        original_op: WorkflowNodeKind,
    ) {
        let Some(current_op) = self.workflow_node_op(node_uuid).cloned() else {
            return;
        };
        if current_op == original_op {
            return;
        }
        if !workflow::is_render_only_change(&original_op, &current_op) {
            return;
        }
        self.sync_editor_node_from_document(node_uuid);
        self.mark_render_only_edit();
    }

    fn fallback_streamline_binding(&self, _id: usize) -> SimpleStreamlineBinding {
        SimpleStreamlineBinding {
            display: WorkflowNodeUuid(0),
        }
    }

    fn fallback_display_binding(&self, _id: usize) -> SimpleDisplayBinding {
        SimpleDisplayBinding {
            display: WorkflowNodeUuid(0),
        }
    }

    fn fallback_surface_binding(&self, _id: usize) -> SimpleSurfaceBinding {
        SimpleSurfaceBinding {
            display: WorkflowNodeUuid(0),
            overlay_stack: None,
        }
    }
}

fn show_advanced_only_reason(ui: &mut egui::Ui, reason: Option<&str>) {
    ui.small(reason.unwrap_or("Switch to Advanced mode to edit this workflow."));
}

fn opacity_checkbox(ui: &mut egui::Ui, opacity: &mut f32, label: &str) -> bool {
    let mut visible = *opacity > 0.0;
    if ui.checkbox(&mut visible, label).changed() {
        *opacity = if visible { (*opacity).max(0.75) } else { 0.0 };
        return true;
    }
    false
}

fn truncate_simple_label(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }

    let keep = (max_chars.saturating_sub(3)) / 2;
    let prefix: String = value.chars().take(keep).collect();
    let suffix: String = value
        .chars()
        .skip(char_count.saturating_sub(keep))
        .collect();
    format!("{prefix}...{suffix}")
}

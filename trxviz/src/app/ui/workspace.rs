use egui_tiles::{Behavior, Tree, UiResponse};
use trxviz_core::lighting::SceneLightingPreset;

use crate::app::workflow::{self, WorkflowGraphViewer, WorkflowSelection, WorkspacePane};

impl super::super::TrxVizApp {
    pub(in crate::app) fn show_workspace(
        &mut self,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
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
            let mut tree = std::mem::replace(
                &mut self.workflow.workspace,
                Tree::empty("workflow_workspace"),
            );
            let mut behavior = WorkspaceBehavior { app: self, frame };
            tree.ui(&mut behavior, ui);
            self.workflow.workspace = tree;
        });
    }

    fn show_assets_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Assets");
        ui.separator();

        if self.workflow.document.assets.is_empty() {
            ui.small("Open files to populate the graph.");
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for asset in &self.workflow.document.assets {
                match asset {
                    workflow::WorkflowAssetDocument::Streamlines { id, path, imported } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        let label = if *imported {
                            format!("Streamlines (imported)\n{}", path.display())
                        } else {
                            format!("Streamlines\n{}", path.display())
                        };
                        if ui.selectable_label(selected, label).clicked() {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                    workflow::WorkflowAssetDocument::Volume { id, path } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Volume\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                    workflow::WorkflowAssetDocument::Cifti { id, path, .. } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("CIFTI\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                    workflow::WorkflowAssetDocument::Surface { id, path } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Surface\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                    workflow::WorkflowAssetDocument::Parcellation { id, path, .. } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Parcellation\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                    workflow::WorkflowAssetDocument::Odx { id, path, .. } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("ODX\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                            self.workflow.document.selection = self.workflow.selection;
                        }
                    }
                }
                ui.add_space(6.0);
            }
        });
    }

    fn show_graph_pane(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Run Expensive Nodes").clicked() {
                self.workflow.run_expensive_requested = true;
                ui.ctx().request_repaint();
            }
            if ui.button("Arrange Graph").clicked() {
                if self.arrange_workflow_graph().is_some() {
                    self.mark_workflow_nonsemantic_edit();
                    ui.ctx().request_repaint();
                }
            }
            if self.workflow.run_expensive_requested {
                ui.small("Will run on the next graph refresh.");
            }
        });
        ui.separator();

        let prior_selection = self.workflow.selection;
        let mut viewer = WorkflowGraphViewer {
            selected: &mut self.workflow.selection,
            focus_bounds: &mut self.workflow.graph_focus_request,
            viewport_rect: ui.max_rect(),
            node_state: &self.workflow.runtime.node_state,
            assets: &self.workflow.document.assets,
            measured_node_sizes: &mut self.workflow.measured_node_sizes,
            layout_reflow_nodes: &mut self.workflow.layout_reflow_nodes,
        };
        let response = egui_snarl::ui::SnarlWidget::new()
            .id(egui::Id::new("workflow_graph"))
            .show(&mut self.workflow.editor_snarl, &mut viewer, ui);

        let mut summary: workflow::GraphEditSummary = workflow::sync_graph_from_snarl(
            &mut self.workflow.editor_snarl,
            &mut self.workflow.document,
        );
        summary.selection_changed = self.workflow.selection != prior_selection;
        if summary.selection_changed {
            self.workflow.document.selection = self.workflow.selection;
        }
        if summary.semantic_changed() {
            self.mark_workflow_semantic_edit(ui.ctx().input(|input| input.time));
        } else if summary.node_positions_changed || summary.selection_changed {
            self.mark_workflow_nonsemantic_edit();
        }
        self.workflow.editor_interaction_active =
            response.hovered() && ui.ctx().input(|input| input.pointer.any_down());
        self.workflow.layout_reflow_pending = !self.workflow.layout_reflow_nodes.is_empty();
        if !self.workflow.editor_interaction_active && self.apply_pending_workflow_layout_reflow() {
            self.mark_workflow_nonsemantic_edit();
            ui.ctx().request_repaint();
        }
    }

    fn show_inspector_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        ui.separator();

        let render_changed = self.show_render_settings_section(ui);
        if render_changed {
            self.workflow.document.render_3d = Some(self.capture_document_render_3d());
            self.mark_workflow_semantic_edit(ui.ctx().input(|input| input.time));
        }

        ui.separator();

        match self.workflow.selection {
            Some(WorkflowSelection::Asset(asset_id)) => self.show_asset_inspector(ui, asset_id),
            Some(WorkflowSelection::Node(node_uuid)) => self.show_node_inspector(ui, node_uuid),
            None => {
                ui.small("Select an asset or node.");
                if let Some(error) = &self.workflow.runtime.graph_error {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, error);
                }
            }
        }
    }

    fn show_render_settings_section(&mut self, ui: &mut egui::Ui) -> bool {
        let original = self.capture_document_render_3d();
        let render = self.viewport.render_3d_mut();

        ui.collapsing("Render", |ui| {
            egui::ComboBox::from_id_salt("scene_lighting_preset")
                .selected_text(render.lighting_preset.label())
                .show_ui(ui, |ui| {
                    for preset in SceneLightingPreset::ALL {
                        ui.selectable_value(&mut render.lighting_preset, preset, preset.label());
                    }
                });

            let mut gradient = matches!(
                render.background,
                crate::app::state::WorkflowBackground3D::VerticalGradient { .. }
            );
            ui.checkbox(&mut gradient, "Vertical gradient");
            if gradient {
                let (mut top, mut bottom) = match render.background {
                    crate::app::state::WorkflowBackground3D::Solid { color } => (color, color),
                    crate::app::state::WorkflowBackground3D::VerticalGradient { top, bottom } => {
                        (top, bottom)
                    }
                };
                ui.label("Top color");
                ui.color_edit_button_rgb(&mut top);
                ui.label("Bottom color");
                ui.color_edit_button_rgb(&mut bottom);
                render.background =
                    crate::app::state::WorkflowBackground3D::VerticalGradient { top, bottom };
            } else {
                let mut color = match render.background {
                    crate::app::state::WorkflowBackground3D::Solid { color } => color,
                    crate::app::state::WorkflowBackground3D::VerticalGradient {
                        bottom, ..
                    } => bottom,
                };
                ui.label("Background color");
                ui.color_edit_button_rgb(&mut color);
                render.background = crate::app::state::WorkflowBackground3D::Solid { color };
            }

            ui.add(egui::Slider::new(&mut render.vignette_strength, 0.0..=0.5).text("Vignette"));
            ui.add(egui::Slider::new(&mut render.exposure, 0.5..=1.5).text("Exposure"));
            ui.add(egui::Slider::new(&mut render.contrast, 0.75..=1.5).text("Contrast"));

            ui.separator();
            ui.collapsing("Advanced", |ui| {
                ui.checkbox(&mut render.fog_enabled, "Depth fade");
                ui.add_enabled_ui(render.fog_enabled, |ui| {
                    ui.label("Fade color");
                    ui.color_edit_button_rgb(&mut render.fog_color);
                    ui.add(
                        egui::Slider::new(&mut render.fog_start_fraction, 0.0..=0.95)
                            .text("Fade near"),
                    );
                    ui.add(
                        egui::Slider::new(&mut render.fog_end_fraction, 0.05..=1.0)
                            .text("Fade far"),
                    );
                });
            });
        });

        let current = self.capture_document_render_3d();
        self.viewport.set_render_3d(current.clone());
        current != original
    }

    fn show_asset_inspector(&mut self, ui: &mut egui::Ui, asset_id: usize) {
        if let Some(trx) = self
            .scene
            .trx_files
            .iter()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&trx.name);
            ui.label(trx.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "{} streamlines, {} vertices, {} groups",
                trx.data.nb_streamlines,
                trx.data.nb_vertices,
                trx.data.groups.len()
            ));
            if !trx.import_warnings.is_empty() {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(255, 214, 102), "Import warnings");
                for warning in &trx.import_warnings {
                    ui.label(warning);
                }
            }
            return;
        }
        if let Some(volume) = self
            .scene
            .nifti_files
            .iter()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&volume.name);
            ui.label(format!(
                "Dims: {} x {} x {}",
                volume.volume.dims[0], volume.volume.dims[1], volume.volume.dims[2]
            ));
            return;
        }
        if let Some(surface) = self
            .scene
            .gifti_surfaces
            .iter_mut()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&surface.name);
            ui.label(surface.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "{} vertices, {} triangles",
                surface.data.vertices.len(),
                surface.data.indices.len() / 3
            ));
            return;
        }
        if let Some(parcel) = self
            .scene
            .parcellations
            .iter()
            .find(|asset| asset.asset.id == asset_id)
        {
            ui.strong(&parcel.asset.name);
            ui.label(parcel.asset.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "Label volume: {} x {} x {}",
                parcel.asset.data.dims[0], parcel.asset.data.dims[1], parcel.asset.data.dims[2]
            ));
            ui.label(format!(
                "{} nonzero parcel labels",
                parcel
                    .asset
                    .data
                    .labels
                    .iter()
                    .copied()
                    .filter(|label| label.0 != 0)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            ));
            return;
        }
        if let Some(odx) = self
            .scene
            .odx_files
            .iter()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&odx.name);
            ui.label(odx.path.display().to_string());
            ui.separator();
            let dims = odx.scene.dimensions();
            ui.label(format!("Dims: {} x {} x {}", dims[0], dims[1], dims[2]));
            ui.label(format!("{} masked voxels", odx.scene.compact_voxel_count()));
            if !odx.warnings.is_empty() {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(255, 214, 102), "Glyph warnings");
                for warning in &odx.warnings {
                    ui.label(warning);
                }
            }
        }
    }

    fn show_node_inspector(&mut self, ui: &mut egui::Ui, node_uuid: workflow::WorkflowNodeUuid) {
        let Some(original_node) = self.workflow.document.graph.get(node_uuid).cloned() else {
            ui.small("Selected node is no longer present.");
            return;
        };

        // Snapshot any ODX names for the selected node before taking a mutable graph borrow.
        let odx_selector_names = match &original_node.op {
            workflow::WorkflowNodeKind::OdxFixelScalarSelect { .. }
            | workflow::WorkflowNodeKind::OdxVolumeSelect { .. } => {
                self.resolve_odx_selector_names(node_uuid)
            }
            _ => None,
        };
        let sh_detail_limit = match &original_node.op {
            workflow::WorkflowNodeKind::OdfGlyphRenderer { .. } => {
                self.max_safe_sh_detail_for_node(node_uuid)
            }
            _ => None,
        };
        let available_groups = self
            .workflow
            .runtime
            .node_state
            .get(&node_uuid)
            .map(|state| state.available_streamline_groups.clone())
            .unwrap_or_default();
        let save_ready = self
            .workflow
            .runtime
            .save_streamline_targets
            .contains_key(&node_uuid);

        let (save_now, run_requested, node_changed) = {
            let node = self
                .workflow
                .document
                .graph
                .get_mut(node_uuid)
                .expect("node must still exist while inspector is open");
            ui.text_edit_singleline(&mut node.label);
            ui.separator();
            let (overridden_fields, overridden_values): (
                Vec<String>,
                std::collections::BTreeMap<String, f32>,
            ) = self
                .workflow
                .runtime
                .node_state
                .get(&node_uuid)
                .map(|s| (s.overridden_fields.clone(), s.overridden_values.clone()))
                .unwrap_or_default();
            let (available_dps_fields, available_dpv_fields): (Vec<String>, Vec<String>) = self
                .workflow
                .runtime
                .node_state
                .get(&node_uuid)
                .map(|s| {
                    (
                        s.available_dps_fields.clone(),
                        s.available_dpv_fields.clone(),
                    )
                })
                .unwrap_or_default();
            let edit_result = workflow::edit_node_op(
                ui,
                node_uuid,
                &mut node.op,
                workflow::NodeEditorContext {
                    available_groups: &available_groups,
                    available_dps_fields: &available_dps_fields,
                    available_dpv_fields: &available_dpv_fields,
                    odx_selector_names: odx_selector_names.as_ref(),
                    sh_detail_limit,
                    save_ready,
                    overridden_fields: &overridden_fields,
                    overridden_values: &overridden_values,
                    gpu_available: self.gpu_device.is_some(),
                },
            );
            (
                edit_result.save_now,
                edit_result.run_expensive_requested,
                *node != original_node,
            )
        };

        if run_requested {
            self.workflow.run_expensive_requested = true;
        }

        if let Some(state) = self.workflow.runtime.node_state.get(&node_uuid) {
            ui.separator();
            ui.small(&state.summary);
            if let Some(execution) = &state.execution {
                ui.label(format!("Status: {}", execution.label()));
            }
            if let Some(fingerprint) = state.fingerprint {
                ui.small(format!("Fingerprint: {fingerprint:016x}"));
            }
            if let Some(result_summary) = &state.last_result_summary {
                ui.small(result_summary);
            }
            if let Some(error) = &state.error {
                ui.colored_label(egui::Color32::RED, error);
            }
        }

        if let Some(message) = self.workflow.node_feedback.get(&node_uuid) {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(96, 210, 128), message);
        }

        if save_now {
            self.save_streamline_node(node_uuid);
        }

        if node_changed {
            let node_copy = self
                .workflow
                .document
                .graph
                .get(node_uuid)
                .cloned()
                .expect("node must still exist after inspector edit");
            if let Some(node_id) = self
                .workflow
                .editor_snarl
                .node_ids()
                .find_map(|(id, value)| (value.uuid == node_uuid).then_some(id))
                && let Some(info) = self.workflow.editor_snarl.get_node_info_mut(node_id)
            {
                info.value = node_copy.clone();
            }
            if workflow::is_render_only_change(&original_node.op, &node_copy.op) {
                self.mark_render_only_edit();
            } else {
                self.mark_workflow_semantic_edit(ui.ctx().input(|input| input.time));
            }
        }
    }

    fn resolve_odx_selector_names(
        &self,
        node_uuid: workflow::WorkflowNodeUuid,
    ) -> Option<workflow::OdxSelectorNames> {
        let source_id = self.workflow.document.graph.wires().find_map(|wire| {
            if wire.to.node != node_uuid || wire.to.input != 0 {
                return None;
            }
            match self
                .workflow
                .document
                .graph
                .get(wire.from.node)
                .map(|node| &node.op)
            {
                Some(workflow::WorkflowNodeKind::OdxSource { source_id }) => Some(*source_id),
                _ => None,
            }
        })?;
        let loaded = self
            .scene
            .odx_files
            .iter()
            .find(|asset| asset.id == source_id)?;
        Some(workflow::OdxSelectorNames {
            dpv_names: loaded
                .scene
                .dpv_names()
                .iter()
                .map(|name| name.to_string())
                .collect(),
            dpf_names: loaded
                .scene
                .dataset()
                .dpf_names()
                .iter()
                .map(|name| name.to_string())
                .collect(),
        })
    }

    fn max_safe_sh_detail_for_node(&self, node_uuid: workflow::WorkflowNodeUuid) -> Option<u32> {
        let limit = self.max_storage_buffer_binding_size?;
        let plan = self
            .workflow
            .runtime
            .scene_plan
            .odf_glyph_draws
            .iter()
            .find(|plan| plan.node_uuid == node_uuid)?;
        if plan.field.scene.glyph_source_kind()
            != Some(trxviz_core::data::odx_data::OdxGlyphSourceKind::Sh)
        {
            return None;
        }
        let viewport_index = plan.slice_axis.viewport_index();
        let axis = plan.slice_axis.odx_axis();
        let slice_idx = self.viewport.slice_index(viewport_index) as u32;
        Some(
            plan.field
                .scene
                .max_sh_detail_for_slice(axis, slice_idx, limit, 6)
                .max(1),
        )
    }

    fn show_preview_pane(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.show_embedded_preview(ui);
    }
}

struct WorkspaceBehavior<'a> {
    app: &'a mut super::super::TrxVizApp,
    frame: &'a mut eframe::Frame,
}

impl Behavior<WorkspacePane> for WorkspaceBehavior<'_> {
    fn tab_title_for_pane(&mut self, pane: &WorkspacePane) -> egui::WidgetText {
        match pane {
            WorkspacePane::Assets => "Assets".into(),
            WorkspacePane::Preview => "Preview".into(),
            WorkspacePane::Graph => "Workflow".into(),
            WorkspacePane::Inspector => "Inspector".into(),
        }
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut WorkspacePane,
    ) -> UiResponse {
        match pane {
            WorkspacePane::Assets => self.app.show_assets_pane(ui),
            WorkspacePane::Preview => self.app.show_preview_pane(ui, self.frame),
            WorkspacePane::Graph => self.app.show_graph_pane(ui),
            WorkspacePane::Inspector => self.app.show_inspector_pane(ui),
        }
        UiResponse::None
    }
}

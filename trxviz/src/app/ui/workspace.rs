use std::collections::BTreeSet;

use egui_tiles::{Behavior, Tree, UiResponse};
use trxviz_core::data::loaded_files::VolumeColormap;
use trxviz_core::data::orientation_field::{BoundaryGlyphColorMode, BoundaryGlyphNormalization};
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::lighting::SceneLightingPreset;
use trxviz_core::renderer::mesh_renderer::SurfaceColormap;

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
        };
        let response = egui_snarl::ui::SnarlWidget::new()
            .id(egui::Id::new("workflow_graph"))
            .show(&mut self.workflow.editor_snarl, &mut viewer, ui);

        let mut summary: workflow::GraphEditSummary = workflow::sync_graph_from_snarl(
            &mut self.workflow.editor_snarl,
            &mut self.workflow.document,
        );
        summary.selection_changed = self.workflow.selection != prior_selection;
        if summary.semantic_changed() {
            self.mark_workflow_semantic_edit(ui.ctx().input(|input| input.time));
        } else if summary.node_positions_changed || summary.selection_changed {
            self.mark_workflow_nonsemantic_edit();
        }
        self.workflow.editor_interaction_active =
            response.hovered() && ui.ctx().input(|input| input.pointer.any_down());
    }

    fn show_inspector_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        ui.separator();

        let render_changed = self.show_render_settings_section(ui);
        if render_changed {
            self.workflow.document.render_3d = Some(self.viewport.workflow_render_3d());
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
        let original = self.viewport.workflow_render_3d();
        let render = &mut self.viewport.render_3d;

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

        self.viewport.render_3d = self.viewport.workflow_render_3d();
        self.viewport.workflow_render_3d() != original
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
                    .filter(|label| *label != 0)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            ));
        }
    }

    fn show_node_inspector(&mut self, ui: &mut egui::Ui, node_uuid: workflow::WorkflowNodeUuid) {
        let Some(original_node) = self.workflow.document.graph.get(node_uuid).cloned() else {
            ui.small("Selected node is no longer present.");
            return;
        };

        // Snapshot any ODX names for the selected node before taking a mutable graph borrow.
        let odx_selector_names = match &original_node.kind {
            workflow::WorkflowNodeKind::OdxFixelScalarSelect { .. }
            | workflow::WorkflowNodeKind::OdxVolumeSelect { .. } => {
                self.resolve_odx_selector_names(node_uuid)
            }
            _ => None,
        };

        let mut save_now = false;
        let node_changed = {
            let node = self
                .workflow
                .document
                .graph
                .get_mut(node_uuid)
                .expect("node must still exist while inspector is open");
            ui.text_edit_singleline(&mut node.label);
            ui.separator();

            match &mut node.kind {
                workflow::WorkflowNodeKind::LimitStreamlines {
                    limit,
                    randomize,
                    seed,
                } => {
                    ui.add(egui::Slider::new(limit, 1..=1_000_000).text("Limit"));
                    ui.checkbox(randomize, "Randomize before limiting");
                    if *randomize {
                        ui.add(egui::DragValue::new(seed).speed(1.0).prefix("Seed "));
                    }
                }
                workflow::WorkflowNodeKind::GroupSelect { groups_csv } => {
                    ui.label("Comma-separated group names");
                    let available_groups = self
                        .workflow
                        .runtime
                        .node_state
                        .get(&node_uuid)
                        .map(|state| state.available_streamline_groups.as_slice())
                        .unwrap_or(&[]);
                    show_group_select_editor(ui, groups_csv, available_groups);
                }
                workflow::WorkflowNodeKind::RandomSubset { limit, seed } => {
                    ui.add(egui::Slider::new(limit, 1..=1_000_000).text("Limit"));
                    ui.add(egui::DragValue::new(seed).speed(1.0).prefix("Seed "));
                }
                workflow::WorkflowNodeKind::SphereQuery { center, radius_mm } => {
                    ui.label("Center (RAS+ mm)");
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut center[0]).speed(0.5).prefix("X "));
                        ui.add(egui::DragValue::new(&mut center[1]).speed(0.5).prefix("Y "));
                        ui.add(egui::DragValue::new(&mut center[2]).speed(0.5).prefix("Z "));
                    });
                    ui.add(egui::DragValue::new(radius_mm).speed(0.5).prefix("Radius "));
                }
                workflow::WorkflowNodeKind::SurfaceDepthQuery { depth_mm }
                | workflow::WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
                    ui.add(egui::DragValue::new(depth_mm).speed(0.25).prefix("Depth "));
                }
                workflow::WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
                    ui.add(egui::DragValue::new(depth_mm).speed(0.25).prefix("Depth "));
                    ui.text_edit_singleline(field);
                }
                workflow::WorkflowNodeKind::ParcelSelect { labels_csv } => {
                    ui.label("Comma-separated label IDs");
                    ui.small("Leave empty to use every nonzero parcel label.");
                    ui.text_edit_multiline(labels_csv);
                }
                workflow::WorkflowNodeKind::ParcelEnd { endpoint_count } => {
                    ui.add(egui::Slider::new(endpoint_count, 1..=2).text("Matching endpoints"));
                }
                workflow::WorkflowNodeKind::ColorByDPV { field }
                | workflow::WorkflowNodeKind::ColorByDPS { field } => {
                    ui.text_edit_singleline(field);
                }
                workflow::WorkflowNodeKind::UniformColor { color } => {
                    ui.color_edit_button_rgba_unmultiplied(color);
                }
                workflow::WorkflowNodeKind::RemoveDuplicates { params } => {
                    egui::ComboBox::from_id_salt(format!("duplicate_mode_{}", node_uuid.0))
                        .selected_text(match params.mode {
                            trx_rs::DuplicateRemovalMode::Exact => "Exact",
                            trx_rs::DuplicateRemovalMode::Near => "Near",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut params.mode,
                                trx_rs::DuplicateRemovalMode::Exact,
                                "Exact",
                            );
                            ui.selectable_value(
                                &mut params.mode,
                                trx_rs::DuplicateRemovalMode::Near,
                                "Near",
                            );
                        });
                    if matches!(params.mode, trx_rs::DuplicateRemovalMode::Near) {
                        ui.add(
                            egui::DragValue::new(&mut params.tolerance_mm)
                                .speed(0.05)
                                .range(0.05..=100.0)
                                .prefix("Tolerance "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut params.endpoint_tolerance_mm)
                                .speed(0.05)
                                .range(0.05..=100.0)
                                .prefix("Endpoint tol "),
                        );
                        ui.add(
                            egui::Slider::new(&mut params.min_shared_voxel_fraction, 0.0..=1.0)
                                .text("Shared voxels"),
                        );
                    }
                }
                workflow::WorkflowNodeKind::StreamlineDisplay {
                    enabled,
                    render_style,
                    tube_radius_mm,
                    tube_sides,
                    slab_half_width_mm,
                } => {
                    ui.checkbox(enabled, "Visible");
                    egui::ComboBox::from_id_salt(format!("render_style_{}", node_uuid.0))
                        .selected_text(format!("{render_style:?}"))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(render_style, RenderStyle::Flat, "Flat");
                            ui.selectable_value(
                                render_style,
                                RenderStyle::Illuminated,
                                "Illuminated",
                            );
                            ui.selectable_value(render_style, RenderStyle::DepthCue, "Depth Cue");
                            ui.selectable_value(render_style, RenderStyle::Tubes, "Tubes");
                        });
                    ui.add(
                        egui::DragValue::new(tube_radius_mm)
                            .speed(0.1)
                            .prefix("Tube radius "),
                    );
                    ui.add(
                        egui::DragValue::new(tube_sides)
                            .speed(1.0)
                            .prefix("Tube sides "),
                    );
                    ui.add(
                        egui::DragValue::new(slab_half_width_mm)
                            .speed(0.5)
                            .prefix("Slice slab "),
                    );
                }
                workflow::WorkflowNodeKind::VolumeDisplay {
                    colormap,
                    opacity,
                    window_center,
                    window_width,
                } => {
                    egui::ComboBox::from_id_salt(format!("volume_colormap_{}", node_uuid.0))
                        .selected_text(colormap.label())
                        .show_ui(ui, |ui| {
                            for value in VolumeColormap::ALL {
                                ui.selectable_value(colormap, *value, value.label());
                            }
                        });
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(egui::Slider::new(window_center, 0.0..=1.0).text("Window center"));
                    ui.add(egui::Slider::new(window_width, 0.01..=2.0).text("Window width"));
                }
                workflow::WorkflowNodeKind::SurfaceOverlayStack { layers } => {
                    ui.small("Ordered surface appearance layers. Layer 0 provides the fallback base color and also styles the first connected scalar input.");
                    ui.separator();
                    for (layer_index, layer) in layers.iter_mut().enumerate() {
                        let is_base_layer = layer_index == 0;
                        let title = if is_base_layer {
                            "Layer 0: Base".to_string()
                        } else {
                            format!("Layer {layer_index}")
                        };
                        ui.collapsing(title, |ui| {
                            ui.checkbox(&mut layer.enabled, "Enabled");
                            ui.horizontal(|ui| {
                                ui.label("Legend");
                                ui.text_edit_singleline(&mut layer.legend_label);
                            });
                            if is_base_layer {
                                ui.label("Fallback base color");
                                ui.color_edit_button_rgba_unmultiplied(&mut layer.solid_color);
                            }
                            ui.add(
                                egui::Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity"),
                            );
                            ui.checkbox(&mut layer.use_label_colors, "Use label table colors");
                            if layer.use_label_colors {
                                ui.small(
                                    "Label overlays use the attached label-table RGBA colors.",
                                );
                            }
                            ui.add_enabled_ui(!layer.use_label_colors, |ui| {
                                show_surface_colormap_picker(
                                    ui,
                                    format!(
                                        "surface_overlay_colormap_{}_{}",
                                        node_uuid.0, layer_index
                                    ),
                                    &mut layer.colormap,
                                );
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut layer.range_min)
                                            .speed(0.1)
                                            .prefix("Min "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut layer.range_max)
                                            .speed(0.1)
                                            .prefix("Max "),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut layer.threshold_min)
                                            .speed(0.1)
                                            .prefix("Thresh min "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut layer.threshold_max)
                                            .speed(0.1)
                                            .prefix("Thresh max "),
                                    );
                                });
                            });
                        });
                    }
                }
                workflow::WorkflowNodeKind::SurfaceDisplay {
                    color,
                    opacity,
                    outline_color,
                    outline_thickness,
                    show_projection_map,
                    map_opacity,
                    map_threshold,
                    gloss,
                    projection_colormap,
                    range_min,
                    range_max,
                    space,
                } => {
                    ui.label("Surface");
                    ui.color_edit_button_rgb(color);
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    egui::ComboBox::from_id_salt(format!("surface_space_{}", node_uuid.0))
                        .selected_text(match space {
                            workflow::SurfaceDisplaySpace::Anatomical => "Anatomical",
                            workflow::SurfaceDisplaySpace::Stage => "Stage",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                space,
                                workflow::SurfaceDisplaySpace::Anatomical,
                                "Anatomical",
                            );
                            ui.selectable_value(
                                space,
                                workflow::SurfaceDisplaySpace::Stage,
                                "Stage",
                            );
                        });
                    ui.separator();
                    ui.label("Slice outline");
                    ui.color_edit_button_rgb(outline_color);
                    ui.add(egui::Slider::new(outline_thickness, 0.25..=8.0).text("Thickness"));
                    ui.separator();
                    ui.checkbox(show_projection_map, "Show surface map");
                    ui.add(egui::Slider::new(map_opacity, 0.0..=1.0).text("Map opacity"));
                    ui.add(egui::Slider::new(map_threshold, 0.0..=1.0).text("Map threshold"));
                    ui.add(egui::Slider::new(gloss, 0.0..=1.0).text("Gloss"));
                    egui::ComboBox::from_id_salt(format!("surface_colormap_{}", node_uuid.0))
                        .selected_text(format!("{projection_colormap:?}"))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                projection_colormap,
                                SurfaceColormap::BlueWhiteRed,
                                "Blue-White-Red",
                            );
                            ui.selectable_value(
                                projection_colormap,
                                SurfaceColormap::Viridis,
                                "Viridis",
                            );
                            ui.selectable_value(
                                projection_colormap,
                                SurfaceColormap::Inferno,
                                "Inferno",
                            );
                        });
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(range_min).speed(0.1).prefix("Min "));
                        ui.add(egui::DragValue::new(range_max).speed(0.1).prefix("Max "));
                    });
                }
                workflow::WorkflowNodeKind::BundleSurfaceBuild {
                    per_group,
                    build_mode,
                    voxel_size_mm,
                    threshold,
                    smooth_sigma,
                    min_component_volume_mm3,
                    tube_radius_mm,
                    tube_sides,
                    opacity,
                } => {
                    ui.checkbox(per_group, "Per group");
                    egui::ComboBox::from_id_salt(format!("bundle_build_mode_{}", node_uuid.0))
                        .selected_text(build_mode.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                build_mode,
                                workflow::BundleSurfaceBuildMode::MarchingCubes,
                                workflow::BundleSurfaceBuildMode::MarchingCubes.label(),
                            );
                            ui.selectable_value(
                                build_mode,
                                workflow::BundleSurfaceBuildMode::Streamtubes,
                                workflow::BundleSurfaceBuildMode::Streamtubes.label(),
                            );
                        });
                    if matches!(*build_mode, workflow::BundleSurfaceBuildMode::MarchingCubes) {
                        ui.add(
                            egui::DragValue::new(voxel_size_mm)
                                .speed(0.1)
                                .prefix("Voxel "),
                        );
                        ui.add(
                            egui::DragValue::new(threshold)
                                .speed(0.1)
                                .prefix("Threshold "),
                        );
                        ui.add(
                            egui::DragValue::new(smooth_sigma)
                                .speed(0.05)
                                .prefix("Smooth "),
                        );
                        ui.add(
                            egui::DragValue::new(min_component_volume_mm3)
                                .speed(1.0)
                                .range(0.0..=1_000_000.0)
                                .prefix("Min component mm^3 "),
                        );
                    } else {
                        ui.add(
                            egui::DragValue::new(tube_radius_mm)
                                .speed(0.05)
                                .range(0.01..=20.0)
                                .prefix("Tube radius "),
                        );
                        ui.add(
                            egui::DragValue::new(tube_sides)
                                .speed(1.0)
                                .range(3..=64)
                                .prefix("Tube sides "),
                        );
                    }
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                }
                workflow::WorkflowNodeKind::BundleSurfaceDisplay {
                    color_mode,
                    outline_thickness,
                } => {
                    egui::ComboBox::from_id_salt(format!(
                        "bundle_surface_color_mode_{}",
                        node_uuid.0
                    ))
                    .selected_text(color_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            color_mode,
                            workflow::BundleSurfaceColorMode::Solid,
                            workflow::BundleSurfaceColorMode::Solid.label(),
                        );
                        ui.selectable_value(
                            color_mode,
                            workflow::BundleSurfaceColorMode::BoundaryField,
                            workflow::BundleSurfaceColorMode::BoundaryField.label(),
                        );
                        ui.selectable_value(
                            color_mode,
                            workflow::BundleSurfaceColorMode::SourceColors,
                            workflow::BundleSurfaceColorMode::SourceColors.label(),
                        );
                    });
                    ui.separator();
                    ui.label("Slice outline");
                    ui.add(egui::Slider::new(outline_thickness, 0.25..=8.0).text("Thickness"));
                }
                workflow::WorkflowNodeKind::BoundaryFieldBuild {
                    voxel_size_mm,
                    sphere_lod,
                    normalization,
                } => {
                    ui.add(
                        egui::DragValue::new(voxel_size_mm)
                            .speed(0.1)
                            .range(0.5..=100.0)
                            .prefix("Voxel "),
                    );
                    ui.add(
                        egui::DragValue::new(sphere_lod)
                            .speed(1.0)
                            .range(4..=64)
                            .prefix("Sphere LOD "),
                    );
                    egui::ComboBox::from_id_salt(format!(
                        "boundary_field_normalization_{}",
                        node_uuid.0
                    ))
                    .selected_text(normalization.label())
                    .show_ui(ui, |ui| {
                        for value in BoundaryGlyphNormalization::ALL {
                            ui.selectable_value(normalization, value, value.label());
                        }
                    });
                }
                workflow::WorkflowNodeKind::BoundaryGlyphDisplay {
                    enabled,
                    scale,
                    density_3d_step,
                    slice_density_step,
                    color_mode,
                    min_contacts,
                } => {
                    ui.checkbox(enabled, "Visible");
                    ui.add(egui::DragValue::new(scale).speed(0.1).prefix("Scale "));
                    ui.add(
                        egui::DragValue::new(density_3d_step)
                            .speed(1.0)
                            .range(1..=64)
                            .prefix("3D step "),
                    );
                    ui.add(
                        egui::DragValue::new(slice_density_step)
                            .speed(1.0)
                            .range(1..=64)
                            .prefix("Slice step "),
                    );
                    ui.add(
                        egui::DragValue::new(min_contacts)
                            .speed(1.0)
                            .range(1..=1_000_000)
                            .prefix("Min contacts "),
                    );
                    egui::ComboBox::from_id_salt(format!(
                        "boundary_glyph_color_mode_{}",
                        node_uuid.0
                    ))
                    .selected_text(color_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            color_mode,
                            BoundaryGlyphColorMode::DirectionRgb,
                            BoundaryGlyphColorMode::DirectionRgb.label(),
                        );
                        ui.selectable_value(
                            color_mode,
                            BoundaryGlyphColorMode::Monochrome,
                            BoundaryGlyphColorMode::Monochrome.label(),
                        );
                    });
                }
                workflow::WorkflowNodeKind::ParcellationDisplay {
                    labels_csv,
                    opacity,
                } => {
                    ui.label("Comma-separated label IDs");
                    ui.small("Leave empty to use every nonzero parcel label.");
                    ui.text_edit_multiline(labels_csv);
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                }
                workflow::WorkflowNodeKind::SaveStreamlines { output_path } => {
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(output_path);
                        if ui.button("Browse...").clicked()
                            && let Some(path) = rfd::FileDialog::new()
                                .set_file_name("streamlines.trx")
                                .save_file()
                        {
                            *output_path = path.display().to_string();
                        }
                    });
                    let ready = self
                        .workflow
                        .runtime
                        .save_streamline_targets
                        .contains_key(&node_uuid);
                    if ui
                        .add_enabled(ready, egui::Button::new("Save Now"))
                        .clicked()
                    {
                        save_now = true;
                    }
                    if !ready {
                        ui.small("Connect a streamline input to enable export.");
                    }
                }
                workflow::WorkflowNodeKind::OdfGlyphRenderer {
                    scale,
                    opacity,
                    offset_from_slice,
                    gloss,
                    vertex_colormap,
                    slice_axis,
                    opacity_gate,
                    size_gate,
                    visible,
                } => {
                    ui.checkbox(visible, "Visible");
                    ui.add(egui::Slider::new(scale, 0.1..=5.0).text("Scale"));
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(egui::Slider::new(gloss, 0.0..=1.0).text("Gloss"));
                    ui.add(
                        egui::DragValue::new(offset_from_slice)
                            .speed(0.25)
                            .prefix("Slice offset "),
                    );
                    egui::ComboBox::from_id_salt(format!("glyph_colormap_{}", node_uuid.0))
                        .selected_text(format!("{vertex_colormap:?}"))
                        .show_ui(ui, |ui| {
                            for value in [
                                workflow::GlyphColormap::Directional,
                                workflow::GlyphColormap::Plasma,
                                workflow::GlyphColormap::Viridis,
                                workflow::GlyphColormap::Inferno,
                                workflow::GlyphColormap::BlueWhiteRed,
                            ] {
                                ui.selectable_value(vertex_colormap, value, format!("{value:?}"));
                            }
                        });
                    egui::ComboBox::from_id_salt(format!("glyph_slice_axis_{}", node_uuid.0))
                        .selected_text(slice_axis.label())
                        .show_ui(ui, |ui| {
                            for value in [
                                workflow::WorkflowSliceViewKind::Axial,
                                workflow::WorkflowSliceViewKind::Coronal,
                                workflow::WorkflowSliceViewKind::Sagittal,
                            ] {
                                ui.selectable_value(slice_axis, value, value.label());
                            }
                        });
                    ui.collapsing("Opacity gate (VolumeScalars input)", |ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut opacity_gate.range.0)
                                    .speed(0.01)
                                    .prefix("Min "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut opacity_gate.range.1)
                                    .speed(0.01)
                                    .prefix("Max "),
                            );
                        });
                        ui.add(egui::Slider::new(&mut opacity_gate.below, 0.0..=1.0).text("Below"));
                        ui.add(egui::Slider::new(&mut opacity_gate.above, 0.0..=1.0).text("Above"));
                    });
                    ui.collapsing("Size gate (VolumeScalars input)", |ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut size_gate.range.0)
                                    .speed(0.01)
                                    .prefix("Min "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut size_gate.range.1)
                                    .speed(0.01)
                                    .prefix("Max "),
                            );
                        });
                        ui.add(
                            egui::Slider::new(&mut size_gate.min_scale, 0.0..=5.0)
                                .text("Below scale"),
                        );
                        ui.add(
                            egui::Slider::new(&mut size_gate.max_scale, 0.0..=5.0)
                                .text("Above scale"),
                        );
                    });
                }
                workflow::WorkflowNodeKind::Fixel3DDisplay {
                    line_width,
                    length_scale,
                    opacity,
                    offset_from_slice,
                    visible,
                } => {
                    ui.checkbox(visible, "Visible");
                    ui.add(egui::Slider::new(line_width, 0.001..=0.05).text("Line width"));
                    ui.add(egui::Slider::new(length_scale, 0.1..=5.0).text("Length scale"));
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(
                        egui::DragValue::new(offset_from_slice)
                            .speed(0.25)
                            .prefix("Slice offset "),
                    );
                }
                workflow::WorkflowNodeKind::Fixel2DDisplay {
                    line_width,
                    opacity,
                    slab_thickness_mm,
                    length_scale,
                    visible,
                } => {
                    ui.checkbox(visible, "Visible");
                    ui.add(egui::Slider::new(line_width, 0.001..=0.05).text("Line width"));
                    ui.add(egui::Slider::new(length_scale, 0.1..=5.0).text("Length scale"));
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(
                        egui::Slider::new(slab_thickness_mm, 0.1..=20.0)
                            .text("Slab thickness (mm)"),
                    );
                }
                workflow::WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => {
                    ui.label("DPF");
                    let label = if dpf_name.is_empty() {
                        "<none>".to_string()
                    } else {
                        dpf_name.clone()
                    };
                    egui::ComboBox::from_id_salt(format!("dpf_select_{}", node_uuid.0))
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            if let Some(odx_names) = odx_selector_names.as_ref() {
                                if odx_names.dpf_names.is_empty() {
                                    ui.small("This ODX asset has no scalar DPFs.");
                                } else {
                                    for name in &odx_names.dpf_names {
                                        ui.selectable_value(dpf_name, name.clone(), name);
                                    }
                                }
                            } else {
                                ui.small("Connect this node to an ODX Source catalog output.");
                            }
                        });
                }
                workflow::WorkflowNodeKind::OdxVolumeSelect { dpv_name } => {
                    ui.label("DPV");
                    let label = if dpv_name.is_empty() {
                        "<none>".to_string()
                    } else {
                        dpv_name.clone()
                    };
                    egui::ComboBox::from_id_salt(format!("dpv_select_{}", node_uuid.0))
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            if let Some(odx_names) = odx_selector_names.as_ref() {
                                if odx_names.dpv_names.is_empty() {
                                    ui.small("This ODX asset has no scalar DPVs.");
                                } else {
                                    for name in &odx_names.dpv_names {
                                        ui.selectable_value(dpv_name, name.clone(), name);
                                    }
                                }
                            } else {
                                ui.small("Connect this node to an ODX Source catalog output.");
                            }
                        });
                }
                workflow::WorkflowNodeKind::ColorByFixelScalars {
                    colormap,
                    range,
                    length_scale_by_scalar: _,
                } => {
                    egui::ComboBox::from_id_salt(("fixel_cmap", node_uuid))
                        .selected_text(surface_colormap_label(*colormap))
                        .show_ui(ui, |ui| {
                            for cm in [
                                SurfaceColormap::BlueWhiteRed,
                                SurfaceColormap::Viridis,
                                SurfaceColormap::Inferno,
                            ] {
                                ui.selectable_value(colormap, cm, surface_colormap_label(cm));
                            }
                        });
                    let mut custom_range = range.is_some();
                    if ui
                        .checkbox(&mut custom_range, "Override scalar range")
                        .changed()
                    {
                        *range = if custom_range { Some((0.0, 1.0)) } else { None };
                    }
                    if let Some((lo, hi)) = range.as_mut() {
                        ui.horizontal(|ui| {
                            ui.label("min");
                            ui.add(egui::DragValue::new(lo).speed(0.01));
                            ui.label("max");
                            ui.add(egui::DragValue::new(hi).speed(0.01));
                        });
                    }
                }
                _ => {
                    ui.small("This node has no editable parameters yet.");
                }
            }
            *node != original_node
        };

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
            if workflow::is_render_only_change(&original_node.kind, &node_copy.kind) {
                self.mark_render_only_edit();
            } else {
                self.mark_workflow_semantic_edit(ui.ctx().input(|input| input.time));
            }
        }
    }

    fn resolve_odx_selector_names(
        &self,
        node_uuid: workflow::WorkflowNodeUuid,
    ) -> Option<OdxSelectorNames> {
        let source_id = self.workflow.document.graph.wires().find_map(|wire| {
            if wire.to.node != node_uuid || wire.to.input != 0 {
                return None;
            }
            match self
                .workflow
                .document
                .graph
                .get(wire.from.node)
                .map(|node| &node.kind)
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
        Some(OdxSelectorNames {
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

    fn show_preview_pane(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.show_embedded_preview(ui);
    }
}

struct OdxSelectorNames {
    dpv_names: Vec<String>,
    dpf_names: Vec<String>,
}

fn show_group_select_editor(
    ui: &mut egui::Ui,
    groups_csv: &mut String,
    available_groups: &[String],
) {
    let output = egui::TextEdit::singleline(groups_csv)
        .hint_text("Type a group name")
        .desired_width(f32::INFINITY)
        .show(ui);
    let response = output.response.clone();

    if available_groups.is_empty() {
        ui.small("Autocomplete appears when the input streamline data exposes group names.");
        return;
    }

    ui.small(format!("{} groups available", available_groups.len()));

    let selected = parse_group_csv(groups_csv);
    let current_fragment = current_group_fragment(groups_csv);
    let current_fragment_lower = current_fragment.to_ascii_lowercase();
    let has_current_fragment = !current_fragment.is_empty();
    let suggestion_state_id = response.id.with("group_select_suggestions");
    let mut suggestions_open = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(suggestion_state_id))
        .unwrap_or(false);

    if response.has_focus() {
        suggestions_open = true;
    }

    let suggestions = available_groups
        .iter()
        .filter(|name| {
            !selected.contains(name.as_str())
                && (current_fragment_lower.is_empty()
                    || name.to_ascii_lowercase().contains(&current_fragment_lower))
        })
        .take(8)
        .cloned()
        .collect::<Vec<_>>();

    if suggestions.is_empty() {
        ui.ctx().data_mut(|d| d.remove::<bool>(suggestion_state_id));
        return;
    }

    if !suggestions_open {
        return;
    }

    let mut picked_suggestion = false;
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_max_height(140.0);
        egui::ScrollArea::vertical()
            .max_height(140.0)
            .show(ui, |ui| {
                for suggestion in suggestions {
                    if ui.selectable_label(false, &suggestion).clicked() {
                        replace_group_fragment(groups_csv, &suggestion);
                        let mut state = output.state.clone();
                        let cursor = egui::text::CCursor::new(groups_csv.chars().count());
                        state
                            .cursor
                            .set_char_range(Some(egui::text::CCursorRange::two(cursor, cursor)));
                        egui::TextEdit::store_state(ui.ctx(), response.id, state);
                        response.request_focus();
                        picked_suggestion = true;
                        ui.ctx().request_repaint();
                    }
                }
            });
    });

    if picked_suggestion {
        ui.ctx()
            .data_mut(|d| d.insert_temp(suggestion_state_id, false));
        return;
    }

    let clicked_elsewhere = ui.input(|i| i.pointer.any_click()) && !response.hovered();
    let keep_open = response.has_focus() || (!clicked_elsewhere && has_current_fragment);
    ui.ctx()
        .data_mut(|d| d.insert_temp(suggestion_state_id, keep_open));
}

fn parse_group_csv(groups_csv: &str) -> BTreeSet<&str> {
    groups_csv
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect()
}

fn current_group_fragment(groups_csv: &str) -> &str {
    groups_csv
        .rsplit(',')
        .next()
        .map(str::trim)
        .unwrap_or_default()
}

fn replace_group_fragment(groups_csv: &mut String, group_name: &str) {
    let prefix_end = groups_csv.rfind(',').map(|idx| idx + 1).unwrap_or(0);
    let prefix = groups_csv[..prefix_end].trim_end().to_string();
    if prefix.is_empty() {
        *groups_csv = format!("{group_name}, ");
    } else {
        *groups_csv = format!("{prefix} {group_name}, ");
    }
}

fn show_surface_colormap_picker(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    colormap: &mut SurfaceColormap,
) {
    egui::ComboBox::from_id_salt(id_salt)
        .selected_text(surface_colormap_label(*colormap))
        .show_ui(ui, |ui| {
            ui.selectable_value(
                colormap,
                SurfaceColormap::BlueWhiteRed,
                surface_colormap_label(SurfaceColormap::BlueWhiteRed),
            );
            ui.selectable_value(
                colormap,
                SurfaceColormap::Viridis,
                surface_colormap_label(SurfaceColormap::Viridis),
            );
            ui.selectable_value(
                colormap,
                SurfaceColormap::Inferno,
                surface_colormap_label(SurfaceColormap::Inferno),
            );
        });
}

fn surface_colormap_label(colormap: SurfaceColormap) -> &'static str {
    match colormap {
        SurfaceColormap::BlueWhiteRed => "Blue-White-Red",
        SurfaceColormap::Viridis => "Viridis",
        SurfaceColormap::Inferno => "Inferno",
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

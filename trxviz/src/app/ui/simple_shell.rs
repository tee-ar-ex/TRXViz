use std::collections::BTreeSet;

use crate::app::workflow::{
    SimpleDisplayBinding, SimpleStreamlineBinding, WorkflowAssetDocument, WorkflowEditability,
    WorkflowNodeKind, WorkflowNodeUuid, classify_workflow_editability,
};
use trxviz_core::data::loaded_files::VolumeColormap;
use trxviz_core::data::trx_data::RenderStyle;

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

                    if let Some(reason) = editability.reason() {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.strong("Advanced workflow loaded");
                            ui.small(reason);
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
                        match (&editability, asset) {
                            (
                                WorkflowEditability::Simple(bindings),
                                WorkflowAssetDocument::Streamlines { id, .. },
                            ) => {
                                if let Some(binding) = bindings.streamline.get(&id).copied() {
                                    self.show_simple_streamline_asset(ui, id, binding, false);
                                } else {
                                    self.show_simple_streamline_summary(ui, id);
                                }
                            }
                            (
                                WorkflowEditability::Simple(bindings),
                                WorkflowAssetDocument::Volume { id, .. },
                            ) => {
                                if let Some(binding) = bindings.volume.get(&id).copied() {
                                    self.show_simple_volume_asset(ui, id, binding, false);
                                } else {
                                    self.show_simple_volume_summary(ui, id);
                                }
                            }
                            (
                                WorkflowEditability::Simple(bindings),
                                WorkflowAssetDocument::Surface { id, .. },
                            ) => {
                                if let Some(binding) = bindings.surface.get(&id).copied() {
                                    self.show_simple_surface_asset(ui, id, binding, false);
                                } else {
                                    self.show_simple_surface_summary(ui, id);
                                }
                            }
                            (
                                WorkflowEditability::Simple(bindings),
                                WorkflowAssetDocument::Parcellation { id, .. },
                            ) => {
                                if let Some(binding) = bindings.parcellation.get(&id).copied() {
                                    self.show_simple_parcellation_asset(ui, id, binding, false);
                                } else {
                                    self.show_simple_parcellation_summary(ui, id);
                                }
                            }
                            (_, WorkflowAssetDocument::Cifti { id, path, .. }) => {
                                ui.group(|ui| {
                                    ui.strong(format!("CIFTI {}", id));
                                    ui.small(path.display().to_string());
                                    ui.small("CIFTI assets are available in Advanced mode.");
                                });
                            }
                            (_, WorkflowAssetDocument::Streamlines { id, .. }) => {
                                self.show_simple_streamline_asset(
                                    ui,
                                    id,
                                    self.fallback_streamline_binding(id),
                                    true,
                                );
                            }
                            (_, WorkflowAssetDocument::Volume { id, .. }) => {
                                self.show_simple_volume_asset(
                                    ui,
                                    id,
                                    self.fallback_display_binding(id),
                                    true,
                                );
                            }
                            (_, WorkflowAssetDocument::Surface { id, .. }) => {
                                self.show_simple_surface_asset(
                                    ui,
                                    id,
                                    self.fallback_display_binding(id),
                                    true,
                                );
                            }
                            (_, WorkflowAssetDocument::Parcellation { id, .. }) => {
                                self.show_simple_parcellation_asset(
                                    ui,
                                    id,
                                    self.fallback_display_binding(id),
                                    true,
                                );
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
                        ui.small("TRX, TRK, TCK, VTK, TinyTrack, NIfTI, GIFTI, CIFTI (.dscalar/.dtseries/.dlabel/.pscalar), and parcellations are supported.");
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
        read_only: bool,
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
        let available_groups = self
            .workflow
            .runtime
            .node_state
            .get(&binding.source)
            .map(|state| state.available_streamline_groups.clone())
            .unwrap_or_default();

        egui::CollapsingHeader::new(title)
            .default_open(true)
            .show(ui, |ui| {
                ui.small(path);
                ui.small(format!(
                    "{nb_streamlines} streamlines, {nb_vertices} vertices, {group_count} groups"
                ));
                for warning in &import_warnings {
                    ui.colored_label(egui::Color32::from_rgb(255, 214, 102), warning);
                }
                if read_only {
                    ui.small("Switch to Advanced mode to edit this workflow.");
                    return;
                }

                if let Some(WorkflowNodeKind::StreamlineDisplay {
                    enabled,
                    render_style,
                    tube_radius_mm: _,
                    tube_sides: _,
                    slab_half_width_mm: _,
                }) = self.workflow_node_kind_mut(binding.display)
                {
                    ui.checkbox(enabled, "Visible");
                    render_style_picker(ui, render_style, "simple_render_style", binding.display);
                }

                if let Some(WorkflowNodeKind::LimitStreamlines { limit, .. }) =
                    self.workflow_node_kind_mut(binding.limit)
                {
                    ui.add(egui::Slider::new(limit, 1..=nb_streamlines.max(1)).text("Limit"));
                }

                if let Some(kind) = self.workflow_node_kind_mut(binding.color) {
                    let mut mode = simple_color_mode(kind);
                    egui::ComboBox::from_id_salt(("simple_color_mode", binding.color.0))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for choice in SimpleColorMode::ALL {
                                ui.selectable_value(&mut mode, choice, choice.label());
                            }
                        });
                    apply_simple_color_mode(kind, mode);
                    if let WorkflowNodeKind::UniformColor { color } = kind {
                        ui.color_edit_button_rgba_unmultiplied(color);
                    }
                }

                if !available_groups.is_empty() {
                    ui.separator();
                    ui.label("Groups");
                    if let Some(WorkflowNodeKind::GroupSelect { groups_csv }) =
                        self.workflow_node_kind_mut(binding.group_select)
                    {
                        show_group_toggle_list(ui, groups_csv, &available_groups);
                    }
                }
            });
    }

    fn show_simple_volume_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleDisplayBinding,
        read_only: bool,
    ) {
        let Some(volume) = self.scene.nifti_files.iter().find(|asset| asset.id == id) else {
            return;
        };
        let name = truncate_simple_label(&volume.name, 36);
        let dims = volume.volume.dims;
        egui::CollapsingHeader::new(name)
            .default_open(true)
            .show(ui, |ui| {
                ui.small(format!("{} x {} x {}", dims[0], dims[1], dims[2]));
                if read_only {
                    ui.small("Switch to Advanced mode to edit this workflow.");
                    return;
                }

                if let Some(WorkflowNodeKind::VolumeDisplay {
                    colormap,
                    opacity,
                    window_center,
                    window_width,
                }) = self.workflow_node_kind_mut(binding.display)
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
            });
    }

    fn show_simple_surface_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleDisplayBinding,
        read_only: bool,
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
            .default_open(false)
            .show(ui, |ui| {
                ui.small(path.clone());
                ui.small(format!(
                    "{vertex_count} vertices, {triangle_count} triangles"
                ));
                if read_only {
                    ui.small("Switch to Advanced mode to edit this workflow.");
                    return;
                }

                if let Some(WorkflowNodeKind::SurfaceDisplay { color, opacity, .. }) =
                    self.workflow_node_kind_mut(binding.display)
                {
                    opacity_checkbox(ui, opacity, "Visible");
                    ui.color_edit_button_rgb(color);
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                }
            });
    }

    fn show_simple_parcellation_asset(
        &mut self,
        ui: &mut egui::Ui,
        id: usize,
        binding: SimpleDisplayBinding,
        read_only: bool,
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
            .filter(|label| *label != 0)
            .count();
        egui::CollapsingHeader::new(name)
            .default_open(false)
            .show(ui, |ui| {
                ui.small(path.clone());
                ui.small(format!("{label_count} labels"));
                if read_only {
                    ui.small("Switch to Advanced mode to edit this workflow.");
                    return;
                }

                if let Some(WorkflowNodeKind::ParcellationDisplay { opacity, .. }) =
                    self.workflow_node_kind_mut(binding.display)
                {
                    opacity_checkbox(ui, opacity, "Visible");
                    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                }
            });
    }

    fn show_simple_streamline_summary(&self, ui: &mut egui::Ui, id: usize) {
        if let Some(trx) = self.scene.trx_files.iter().find(|asset| asset.id == id) {
            ui.label(truncate_simple_label(&trx.name, 36));
        }
    }

    fn show_simple_volume_summary(&self, ui: &mut egui::Ui, id: usize) {
        if let Some(volume) = self.scene.nifti_files.iter().find(|asset| asset.id == id) {
            ui.label(truncate_simple_label(&volume.name, 36));
        }
    }

    fn show_simple_surface_summary(&self, ui: &mut egui::Ui, id: usize) {
        if let Some(surface) = self
            .scene
            .gifti_surfaces
            .iter()
            .find(|asset| asset.id == id)
        {
            ui.label(truncate_simple_label(&surface.name, 36));
        }
    }

    fn show_simple_parcellation_summary(&self, ui: &mut egui::Ui, id: usize) {
        if let Some(parcel) = self
            .scene
            .parcellations
            .iter()
            .find(|asset| asset.asset.id == id)
        {
            ui.label(truncate_simple_label(&parcel.asset.name, 36));
        }
    }

    fn workflow_node_kind_mut(&mut self, uuid: WorkflowNodeUuid) -> Option<&mut WorkflowNodeKind> {
        self.workflow
            .document
            .graph
            .get_mut(uuid)
            .map(|node| &mut node.kind)
    }

    fn fallback_streamline_binding(&self, _id: usize) -> SimpleStreamlineBinding {
        SimpleStreamlineBinding {
            source: WorkflowNodeUuid(0),
            group_select: WorkflowNodeUuid(0),
            limit: WorkflowNodeUuid(0),
            color: WorkflowNodeUuid(0),
            display: WorkflowNodeUuid(0),
        }
    }

    fn fallback_display_binding(&self, _id: usize) -> SimpleDisplayBinding {
        SimpleDisplayBinding {
            display: WorkflowNodeUuid(0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimpleColorMode {
    Direction,
    Group,
    Uniform,
}

impl SimpleColorMode {
    const ALL: [Self; 3] = [Self::Direction, Self::Group, Self::Uniform];

    fn label(self) -> &'static str {
        match self {
            Self::Direction => "Direction",
            Self::Group => "Group",
            Self::Uniform => "Uniform",
        }
    }
}

fn simple_color_mode(kind: &WorkflowNodeKind) -> SimpleColorMode {
    match kind {
        WorkflowNodeKind::ColorByGroup => SimpleColorMode::Group,
        WorkflowNodeKind::UniformColor { .. } => SimpleColorMode::Uniform,
        _ => SimpleColorMode::Direction,
    }
}

fn apply_simple_color_mode(kind: &mut WorkflowNodeKind, mode: SimpleColorMode) {
    let previous_uniform = match kind {
        WorkflowNodeKind::UniformColor { color } => *color,
        _ => [0.95, 0.75, 0.25, 1.0],
    };
    *kind = match mode {
        SimpleColorMode::Direction => WorkflowNodeKind::ColorByDirection,
        SimpleColorMode::Group => WorkflowNodeKind::ColorByGroup,
        SimpleColorMode::Uniform => WorkflowNodeKind::UniformColor {
            color: previous_uniform,
        },
    };
}

fn render_style_picker(
    ui: &mut egui::Ui,
    render_style: &mut RenderStyle,
    id_salt: &'static str,
    node_uuid: WorkflowNodeUuid,
) {
    egui::ComboBox::from_id_salt((id_salt, node_uuid.0))
        .selected_text(render_style_label(*render_style))
        .show_ui(ui, |ui| {
            for choice in [
                RenderStyle::Flat,
                RenderStyle::Illuminated,
                RenderStyle::DepthCue,
                RenderStyle::Tubes,
            ] {
                ui.selectable_value(render_style, choice, render_style_label(choice));
            }
        });
}

fn render_style_label(style: RenderStyle) -> &'static str {
    match style {
        RenderStyle::Flat => "Flat",
        RenderStyle::Illuminated => "Illuminated",
        RenderStyle::Tubes => "Tubes",
        RenderStyle::DepthCue => "Depth cue",
    }
}

fn opacity_checkbox(ui: &mut egui::Ui, opacity: &mut f32, label: &str) {
    let mut visible = *opacity > 0.0;
    if ui.checkbox(&mut visible, label).changed() {
        *opacity = if visible { (*opacity).max(0.75) } else { 0.0 };
    }
}

fn show_group_toggle_list(ui: &mut egui::Ui, groups_csv: &mut String, available_groups: &[String]) {
    let mut selected = parse_groups(groups_csv, available_groups);
    let mut changed = false;

    ui.horizontal(|ui| {
        if ui.button("All").clicked() {
            selected = available_groups.iter().cloned().collect();
            changed = true;
        }
        if ui.button("Hide all").clicked() {
            selected.clear();
            changed = true;
        }
    });

    for group in available_groups {
        let mut enabled = selected.contains(group);
        if ui
            .checkbox(&mut enabled, truncate_simple_label(group, 32))
            .changed()
        {
            changed = true;
            if enabled {
                selected.insert(group.clone());
            } else {
                selected.remove(group);
            }
        }
    }

    if changed {
        if selected.len() == available_groups.len() {
            groups_csv.clear();
        } else if selected.is_empty() {
            *groups_csv = "__none__".to_string();
        } else {
            *groups_csv = selected.into_iter().collect::<Vec<_>>().join(", ");
        }
    }
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

fn parse_groups(groups_csv: &str, available_groups: &[String]) -> BTreeSet<String> {
    if groups_csv.trim() == "__none__" {
        return BTreeSet::new();
    }
    let selected = groups_csv
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>();
    if selected.is_empty() {
        available_groups.iter().cloned().collect()
    } else {
        selected
    }
}

use std::path::Path;

use crate::app::callbacks::{self, BundleDrawInfo, StreamlineDrawInfo, VolumeDrawInfo};
use crate::app::state::{ExportTarget, SliceViewKind, View2DMode};
use glam::{Mat4, Vec3};
use trxviz_core::data::cifti::CiftiStructure;
use trxviz_core::data::orientation_field::BoundaryGlyphColorMode;
use trxviz_core::headless::HeadlessView;
use trxviz_core::renderer::camera::OrbitCamera;
use trxviz_core::renderer::mesh_renderer::MeshDrawStyle;

fn viewport_3d_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("trxviz_3d_window")
}

fn viewport_2d_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("trxviz_2d_window")
}

fn viewport_stage_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("trxviz_stage_window")
}

fn export_viewport_id(target: ExportTarget) -> egui::ViewportId {
    match target {
        ExportTarget::View3D => egui::ViewportId::from_hash_of("trxviz_export_3d"),
        ExportTarget::View2D => egui::ViewportId::from_hash_of("trxviz_export_2d"),
        ExportTarget::InflatedStage => egui::ViewportId::from_hash_of("trxviz_export_stage"),
    }
}

struct ViewerRenderData {
    surface_draws: Vec<(usize, usize, MeshDrawStyle)>,
    volume_draws: Vec<VolumeDrawInfo>,
    streamline_draws: Vec<StreamlineDrawInfo>,
    bundle_draws: Vec<BundleDrawInfo>,
    any_visible_streamlines: bool,
    glyph_visible: bool,
    glyph_color_mode: BoundaryGlyphColorMode,
    glyph_density_3d_step: u32,
    glyph_slice_density_step: u32,
}

impl super::super::TrxVizApp {
    pub(in crate::app) fn show_viewports(&mut self, ctx: &egui::Context) {
        self.show_export_dialog(ctx);
        self.show_3d_window(ctx);
        self.show_inflated_stage_window(ctx);
        self.show_2d_window(ctx);
        self.show_export_viewport(ctx);
    }

    pub(in crate::app) fn show_embedded_preview(&mut self, ui: &mut egui::Ui) {
        if self.scene_is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("Open files or load a project to populate the preview.");
            });
            return;
        }

        let render_data = self.build_viewer_render_data();
        self.show_embedded_preview_toolbar(ui);
        ui.add_space(8.0);

        let available = ui.available_size();
        let any_slice_visible = self.viewport.slice_visible.iter().any(|&visible| visible);
        let spacing_y = ui.spacing().item_spacing.y;
        let top_height = if any_slice_visible {
            (available.y * 0.58).max(180.0)
        } else {
            available.y
        };
        let bottom_height = if any_slice_visible {
            (available.y - top_height - spacing_y).max(140.0)
        } else {
            0.0
        };

        let (rect_3d, response_3d) = ui.allocate_exact_size(
            egui::vec2(available.x, top_height.max(120.0)),
            egui::Sense::click_and_drag(),
        );
        self.viewport.window_3d_size = [rect_3d.width().max(320.0), rect_3d.height().max(240.0)];
        self.draw_scene3d_rect(ui, rect_3d, Some(&response_3d), &render_data);

        if any_slice_visible && bottom_height > 0.0 {
            ui.add_space(spacing_y);
            let size = egui::vec2(available.x, bottom_height);
            self.viewport.window_2d_size = [size.x.max(320.0), size.y.max(240.0)];
            ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
                match self.viewport.view_2d.mode {
                    View2DMode::Slice => self.show_2d_slice_mode(ui, &render_data, true),
                    View2DMode::Ortho => self.show_2d_ortho_mode(ui, &render_data, true),
                    View2DMode::Lightbox => self.show_2d_lightbox_mode(ui, &render_data, true),
                }
            });
        }
    }

    fn show_embedded_preview_toolbar(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Preview");
                ui.separator();
                if ui.button("Pop Out 3D").clicked() {
                    self.viewport.window_3d_open = true;
                }
                if ui.button("Open Inflated Stage").clicked() {
                    self.viewport.inflated_stage_open = true;
                    self.reset_inflated_stage_camera();
                }
                if ui.button("Pop Out 2D").clicked() {
                    self.viewport.view_2d.window_open = true;
                }
                if ui.button("Copy 3D Camera").clicked() {
                    self.copy_camera_3d_json(ui.ctx());
                }
                ui.separator();
                ui.label("Slices");
                ui.checkbox(&mut self.viewport.slice_visible[0], "Axial");
                ui.checkbox(&mut self.viewport.slice_visible[1], "Coronal");
                ui.checkbox(&mut self.viewport.slice_visible[2], "Sagittal");
                ui.separator();
                ui.label("2D");
                egui::ComboBox::from_id_salt("embedded_mode_2d")
                    .selected_text(self.viewport.view_2d.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in View2DMode::ALL {
                            ui.selectable_value(
                                &mut self.viewport.view_2d.mode,
                                mode,
                                mode.label(),
                            );
                        }
                    });

                match self.viewport.view_2d.mode {
                    View2DMode::Slice => {
                        slice_kind_picker(
                            ui,
                            &mut self.viewport.view_2d.single_view,
                            "embedded_slice_axis",
                        );
                    }
                    View2DMode::Ortho => {
                        ui.checkbox(&mut self.viewport.view_2d.ortho_show_row, "Row layout");
                    }
                    View2DMode::Lightbox => {
                        slice_kind_picker(
                            ui,
                            &mut self.viewport.view_2d.lightbox_axis,
                            "embedded_lightbox_axis",
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.viewport.view_2d.lightbox_rows)
                                .range(1..=8)
                                .prefix("Rows "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.viewport.view_2d.lightbox_cols)
                                .range(1..=8)
                                .prefix("Cols "),
                        );
                    }
                }
            });
        });
    }

    fn show_3d_window(&mut self, ctx: &egui::Context) {
        if !self.viewport.window_3d_open {
            return;
        }

        let builder = egui::ViewportBuilder::default()
            .with_title("TRXViz: 3D")
            .with_inner_size(self.viewport.window_3d_size);
        ctx.show_viewport_immediate(viewport_3d_id(), builder, |ctx, class| {
            if ctx.input(|i| i.viewport().close_requested()) {
                self.viewport.window_3d_open = false;
                return;
            }

            if class == egui::ViewportClass::Embedded {
                let mut open = self.viewport.window_3d_open;
                egui::Window::new("3D View")
                    .open(&mut open)
                    .default_size(self.viewport.window_3d_size)
                    .show(ctx, |ui| self.show_3d_contents(ui.ctx(), true));
                self.viewport.window_3d_open = open;
            } else {
                self.show_3d_contents(ctx, true);
            }
        });
    }

    fn show_2d_window(&mut self, ctx: &egui::Context) {
        if !self.viewport.view_2d.window_open {
            return;
        }

        let builder = egui::ViewportBuilder::default()
            .with_title("TRXViz: 2D")
            .with_inner_size(self.viewport.window_2d_size);
        ctx.show_viewport_immediate(viewport_2d_id(), builder, |ctx, class| {
            if ctx.input(|i| i.viewport().close_requested()) {
                self.viewport.view_2d.window_open = false;
                return;
            }

            if class == egui::ViewportClass::Embedded {
                let mut open = self.viewport.view_2d.window_open;
                egui::Window::new("2D View")
                    .open(&mut open)
                    .default_size(self.viewport.window_2d_size)
                    .show(ctx, |ui| self.show_2d_contents(ui.ctx(), true));
                self.viewport.view_2d.window_open = open;
            } else {
                self.show_2d_contents(ctx, true);
            }
        });
    }

    fn show_inflated_stage_window(&mut self, ctx: &egui::Context) {
        if !self.viewport.inflated_stage_open {
            return;
        }
        let builder = egui::ViewportBuilder::default()
            .with_title("TRXViz: Inflated Stage")
            .with_inner_size(self.viewport.inflated_stage_size)
            .with_resizable(true);
        ctx.show_viewport_immediate(viewport_stage_id(), builder, |ctx, class| {
            if ctx.input(|i| i.viewport().close_requested()) {
                self.viewport.inflated_stage_open = false;
                return;
            }

            if class == egui::ViewportClass::Embedded {
                let mut open = self.viewport.inflated_stage_open;
                egui::Window::new("Inflated Stage")
                    .open(&mut open)
                    .default_size(self.viewport.inflated_stage_size)
                    .resizable(true)
                    .show(ctx, |ui| self.show_inflated_stage_contents(ui.ctx(), true));
                self.viewport.inflated_stage_open = open;
            } else {
                self.show_inflated_stage_contents(ctx, true);
            }
        });
    }

    fn show_export_dialog(&mut self, ctx: &egui::Context) {
        if !self.viewport.export_dialog.open {
            return;
        }

        let mut open = self.viewport.export_dialog.open;
        let mut start_export = false;
        egui::Window::new("Export View")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(self.viewport.export_dialog.target.label());
                ui.add(
                    egui::Slider::new(&mut self.viewport.export_dialog.scale, 1..=8).text("Scale"),
                );
                ui.small("Scale multiplies the current viewer window resolution.");
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.viewport.export_dialog.open = false;
                    }
                    if ui.button("Export").clicked() {
                        start_export = true;
                    }
                });
            });
        self.viewport.export_dialog.open = open && self.viewport.export_dialog.open;

        if !start_export {
            return;
        }

        let default_name = match self.viewport.export_dialog.target {
            ExportTarget::View3D => "trxviz-3d.png",
            ExportTarget::View2D => "trxviz-2d.png",
            ExportTarget::InflatedStage => "trxviz-stage.png",
        };
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .set_file_name(default_name)
            .save_file()
        {
            self.viewport.pending_export = Some(crate::app::state::PendingExportRequest {
                target: self.viewport.export_dialog.target,
                path,
                scale: self.viewport.export_dialog.scale.max(1),
                requested_screenshot: false,
            });
            self.viewport.export_dialog.open = false;
            ctx.request_repaint();
        }
    }

    fn show_export_viewport(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.viewport.pending_export.as_ref() else {
            return;
        };

        let base_size = match pending.target {
            ExportTarget::View3D => self.viewport.window_3d_size,
            ExportTarget::View2D => self.viewport.window_2d_size,
            ExportTarget::InflatedStage => self.viewport.inflated_stage_size,
        };
        let export_size = [
            (base_size[0] * pending.scale as f32).max(256.0),
            (base_size[1] * pending.scale as f32).max(256.0),
        ];
        let builder = egui::ViewportBuilder::default()
            .with_title(format!("Export {}", pending.target.label()))
            .with_inner_size(export_size)
            .with_visible(false)
            .with_decorations(false)
            .with_resizable(false)
            .with_taskbar(false);
        let target = pending.target;
        ctx.show_viewport_immediate(
            export_viewport_id(target),
            builder,
            |ctx, _class| match target {
                ExportTarget::View3D => self.show_3d_contents(ctx, false),
                ExportTarget::View2D => self.show_2d_contents(ctx, false),
                ExportTarget::InflatedStage => self.show_inflated_stage_contents(ctx, false),
            },
        );
    }

    fn show_3d_contents(&mut self, ctx: &egui::Context, interactive: bool) {
        if interactive {
            let size = ctx.input(|i| i.content_rect().size());
            self.viewport.window_3d_size = [size.x.max(320.0), size.y.max(240.0)];
        }

        let render_data = self.build_viewer_render_data();
        egui::TopBottomPanel::top("window_3d_toolbar").show_animated(ctx, interactive, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.small("3D window");
                ui.separator();
                ui.label("Slice quads");
                ui.checkbox(&mut self.viewport.slice_visible[0], "Axial");
                ui.checkbox(&mut self.viewport.slice_visible[1], "Coronal");
                ui.checkbox(&mut self.viewport.slice_visible[2], "Sagittal");
                ui.separator();
                ui.small("Drag orbit");
                ui.small("Shift-drag or middle-drag pan");
                ui.small("Scroll zoom");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.scene_is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open files or load a project to populate the 3D view.");
                });
                return;
            }

            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            self.draw_scene3d_rect(ui, rect, interactive.then_some(&response), &render_data);
        });

        self.finish_export_if_ready(ctx, ExportTarget::View3D);
    }

    fn show_inflated_stage_contents(&mut self, ctx: &egui::Context, interactive: bool) {
        if interactive {
            let size = ctx.input(|i| i.content_rect().size());
            self.viewport.inflated_stage_size = [size.x.max(320.0), size.y.max(240.0)];
        }

        egui::TopBottomPanel::top("inflated_stage_toolbar").show_animated(ctx, interactive, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.small("Inflated Stage");
                ui.separator();
                if ui.button("Reset camera").clicked() {
                    self.reset_inflated_stage_camera();
                }
                ui.menu_button("Export", |ui| {
                    if ui.button("PNG").clicked() {
                        self.viewport.export_dialog.target = ExportTarget::InflatedStage;
                        self.viewport.export_dialog.open = true;
                        ui.close();
                    }
                    if ui.button("Blender (GLB)").clicked() {
                        self.export_to_blender(HeadlessView::InflatedStage);
                        ui.close();
                    }
                });
                ui.separator();
                ui.small("Drag orbit");
                ui.small("Shift-drag or middle-drag pan");
                ui.small("Scroll zoom");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let render_data = self.build_stage_render_data();
            if render_data.surface_draws.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No surfaces are routed to Stage.");
                });
                return;
            }
            let available = ui.available_size();
            let (rect, response) =
                ui.allocate_exact_size(available, egui::Sense::click_and_drag());
            self.draw_stage_scene3d_rect(ui, rect, interactive.then_some(&response), &render_data);
        });

        self.finish_export_if_ready(ctx, ExportTarget::InflatedStage);
    }

    fn reset_inflated_stage_camera(&mut self) {
        let Some((center, span)) = self.inflated_stage_bounds() else {
            let mut camera = OrbitCamera::new(Vec3::ZERO, 250.0);
            camera.yaw = 0.0;
            camera.pitch = 0.0;
            self.viewport.inflated_stage_camera = camera;
            return;
        };
        let mut camera = OrbitCamera::new(center, (span * 1.15).max(50.0));
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        self.viewport.inflated_stage_camera = camera;
    }

    fn inflated_stage_bounds(&self) -> Option<(Vec3, f32)> {
        let mut bounds_min = Vec3::splat(f32::INFINITY);
        let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;

        for (surface_id, _, style) in self.stage_surface_draw_instances() {
            let Some(surface) = self
                .scene
                .gifti_surfaces
                .iter()
                .find(|surface| surface.id == surface_id)
            else {
                continue;
            };
            let model = Mat4::from_cols_array_2d(&style.model_matrix);
            for corner in bbox_corners(surface.data.bbox_min, surface.data.bbox_max) {
                let point = model.transform_point3(corner);
                bounds_min = bounds_min.min(point);
                bounds_max = bounds_max.max(point);
                any = true;
            }
        }

        if !any {
            return None;
        }

        let center = (bounds_min + bounds_max) * 0.5;
        let extents = bounds_max - bounds_min;
        let span = extents.x.abs().max(extents.y.abs()).max(extents.z.abs()).max(1.0);
        Some((center, span))
    }

    fn stage_surface_draw_instances(&self) -> Vec<(usize, usize, MeshDrawStyle)> {
        let mut surface_draws = Vec::new();
        for draw in &self.workflow.runtime.scene_plan.stage_surface_draws {
            let Some(surface) = self
                .scene
                .gifti_surfaces
                .iter()
                .find(|surface| surface.id == draw.source_id)
            else {
                continue;
            };
            for (uniform_slot, model_matrix) in stage_instance_model_matrices(
                draw.structure,
                surface.data.bbox_min,
                surface.data.bbox_max,
            )
            .into_iter()
            .enumerate()
            {
                surface_draws.push((
                    draw.source_id,
                    uniform_slot,
                    MeshDrawStyle {
                        color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                        scalar_min: draw.range_min,
                        scalar_max: draw.range_max,
                        scalar_enabled: draw.show_projection_map,
                        vertex_color_enabled: !draw.vertex_rgba.is_empty(),
                        colormap: draw.projection_colormap,
                        gloss: draw.gloss,
                        map_opacity: draw.map_opacity,
                        map_threshold: draw.map_threshold,
                        model_matrix: model_matrix.to_cols_array_2d(),
                    },
                ));
            }
        }
        surface_draws
    }

    fn show_2d_contents(&mut self, ctx: &egui::Context, interactive: bool) {
        if interactive {
            let size = ctx.input(|i| i.content_rect().size());
            self.viewport.window_2d_size = [size.x.max(320.0), size.y.max(240.0)];
        }

        let render_data = self.build_viewer_render_data();
        egui::TopBottomPanel::top("window_2d_toolbar").show_animated(ctx, interactive, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Mode");
                egui::ComboBox::from_id_salt("mode_2d")
                    .selected_text(self.viewport.view_2d.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in View2DMode::ALL {
                            ui.selectable_value(
                                &mut self.viewport.view_2d.mode,
                                mode,
                                mode.label(),
                            );
                        }
                    });

                match self.viewport.view_2d.mode {
                    View2DMode::Slice => {
                        ui.separator();
                        slice_kind_picker(ui, &mut self.viewport.view_2d.single_view, "slice_axis");
                    }
                    View2DMode::Ortho => {
                        ui.separator();
                        ui.checkbox(&mut self.viewport.view_2d.ortho_show_row, "Row layout");
                    }
                    View2DMode::Lightbox => {
                        ui.separator();
                        slice_kind_picker(
                            ui,
                            &mut self.viewport.view_2d.lightbox_axis,
                            "lightbox_axis",
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.viewport.view_2d.lightbox_rows)
                                .range(1..=8)
                                .prefix("Rows "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.viewport.view_2d.lightbox_cols)
                                .range(1..=8)
                                .prefix("Cols "),
                        );
                    }
                }

                ui.separator();
                ui.small("Pan: drag");
                ui.small("Move slice: scroll");
                ui.small("Zoom: pinch");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.scene_is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open files or load a project to populate the 2D view.");
                });
                return;
            }

            match self.viewport.view_2d.mode {
                View2DMode::Slice => self.show_2d_slice_mode(ui, &render_data, interactive),
                View2DMode::Ortho => self.show_2d_ortho_mode(ui, &render_data, interactive),
                View2DMode::Lightbox => self.show_2d_lightbox_mode(ui, &render_data, interactive),
            }
        });

        self.finish_export_if_ready(ctx, ExportTarget::View2D);
    }

    fn show_2d_slice_mode(
        &mut self,
        ui: &mut egui::Ui,
        render_data: &ViewerRenderData,
        interactive: bool,
    ) {
        let axis_index = self
            .viewport
            .view_2d
            .single_view
            .slice_axis_index()
            .unwrap_or(0);
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        self.draw_slice_rect(
            ui,
            rect,
            interactive.then_some(&response),
            axis_index,
            self.viewport
                .slice_world_position(&self.scene.nifti_files, axis_index),
            render_data,
            true,
        );
    }

    fn show_2d_ortho_mode(
        &mut self,
        ui: &mut egui::Ui,
        render_data: &ViewerRenderData,
        interactive: bool,
    ) {
        let available = ui.available_size();
        let spacing = ui.spacing().item_spacing;
        if self.viewport.view_2d.ortho_show_row {
            let width = ((available.x - 2.0 * spacing.x) / 3.0).max(80.0);
            let height = available.y.max(80.0);
            ui.horizontal(|ui| {
                for axis_index in 0..3 {
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(width, height),
                        egui::Sense::click_and_drag(),
                    );
                    self.draw_slice_rect(
                        ui,
                        rect,
                        interactive.then_some(&response),
                        axis_index,
                        self.viewport
                            .slice_world_position(&self.scene.nifti_files, axis_index),
                        render_data,
                        true,
                    );
                }
            });
        } else {
            let width = ((available.x - spacing.x) / 2.0).max(80.0);
            let height = ((available.y - spacing.y) / 2.0).max(80.0);
            ui.horizontal(|ui| {
                let (rect0, response0) = ui
                    .allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
                self.draw_slice_rect(
                    ui,
                    rect0,
                    interactive.then_some(&response0),
                    0,
                    self.viewport
                        .slice_world_position(&self.scene.nifti_files, 0),
                    render_data,
                    true,
                );
                ui.vertical(|ui| {
                    let (rect1, response1) = ui.allocate_exact_size(
                        egui::vec2(width, height),
                        egui::Sense::click_and_drag(),
                    );
                    self.draw_slice_rect(
                        ui,
                        rect1,
                        interactive.then_some(&response1),
                        1,
                        self.viewport
                            .slice_world_position(&self.scene.nifti_files, 1),
                        render_data,
                        true,
                    );
                    let (rect2, response2) = ui.allocate_exact_size(
                        egui::vec2(width, height),
                        egui::Sense::click_and_drag(),
                    );
                    self.draw_slice_rect(
                        ui,
                        rect2,
                        interactive.then_some(&response2),
                        2,
                        self.viewport
                            .slice_world_position(&self.scene.nifti_files, 2),
                        render_data,
                        true,
                    );
                });
            });
        }
    }

    fn show_2d_lightbox_mode(
        &mut self,
        ui: &mut egui::Ui,
        render_data: &ViewerRenderData,
        interactive: bool,
    ) {
        let axis_index = self
            .viewport
            .view_2d
            .lightbox_axis
            .slice_axis_index()
            .unwrap_or(0);
        let rows = self.viewport.view_2d.lightbox_rows.max(1);
        let cols = self.viewport.view_2d.lightbox_cols.max(1);
        let total = rows * cols;
        let center_tile = total / 2;
        let available = ui.available_size();
        let spacing = ui.spacing().item_spacing;
        let tile_width =
            ((available.x - spacing.x * (cols.saturating_sub(1) as f32)) / cols as f32).max(60.0);
        let tile_height =
            ((available.y - spacing.y * (rows.saturating_sub(1) as f32)) / rows as f32).max(60.0);
        let center_index = self.viewport.slice_indices[axis_index];
        let max_index = self.max_slice_index(axis_index);

        for row in 0..rows {
            ui.horizontal(|ui| {
                for col in 0..cols {
                    let tile = row * cols + col;
                    let delta = tile as isize - center_tile as isize;
                    let index = center_index.saturating_add_signed(delta).min(max_index);
                    let slice_pos = self.viewport.slice_world_position_for_index(
                        &self.scene.nifti_files,
                        axis_index,
                        index,
                    );
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(tile_width, tile_height),
                        egui::Sense::click_and_drag(),
                    );

                    if interactive && response.clicked() {
                        self.viewport.slice_indices[axis_index] = index;
                        self.viewport.slices_dirty = true;
                    }

                    self.draw_slice_rect(
                        ui,
                        rect,
                        interactive.then_some(&response),
                        axis_index,
                        slice_pos,
                        render_data,
                        tile == center_tile,
                    );
                }
            });
        }
    }

    fn draw_scene3d_rect(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: Option<&egui::Response>,
        render_data: &ViewerRenderData,
    ) {
        if let Some(response) = response {
            let modifiers = ui.input(|i| i.modifiers);
            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = ui.input(|i| i.pointer.delta());
                if modifiers.shift {
                    self.viewport.camera_3d.pan_screen(delta.x, delta.y);
                } else {
                    self.viewport.camera_3d.handle_drag(delta.x, delta.y);
                }
            }
            if response.dragged_by(egui::PointerButton::Middle) {
                let delta = ui.input(|i| i.pointer.delta());
                self.viewport.camera_3d.pan_screen(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    self.viewport.camera_3d.handle_scroll(scroll * 0.01);
                }
            }
        }

        let aspect = rect.width() / rect.height().max(1.0);
        let vp = self.viewport.camera_3d.view_projection(aspect);
        let fog_span = (self.viewport.camera_3d.distance + self.viewport.volume_extent).max(1.0);
        let render_3d = self.viewport.workflow_render_3d();
        let fog_near = fog_span * render_3d.fog_start_fraction;
        let fog_far = fog_span * render_3d.fog_end_fraction;
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            callbacks::Scene3DCallback {
                view_proj: vp,
                camera_pos: self.viewport.camera_3d.eye(),
                camera_dir: self.viewport.camera_3d.view_direction(),
                streamline_draws: render_data.streamline_draws.clone(),
                show_streamlines: render_data.any_visible_streamlines,
                volume_draws: render_data.volume_draws.clone(),
                slice_visible: self.viewport.slice_visible,
                surface_draws: render_data.surface_draws.clone(),
                bundle_draws: render_data.bundle_draws.clone(),
                show_boundary_glyphs: render_data.glyph_visible,
                boundary_glyph_color_mode: render_data.glyph_color_mode,
                boundary_glyph_draw_step: render_data.glyph_density_3d_step,
                scene_lighting: self.viewport.scene_lighting(),
                render_3d,
                fog_near,
                fog_far,
            },
        ));
        self.draw_3d_axes(ui, rect, vp);
    }

    fn draw_stage_scene3d_rect(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: Option<&egui::Response>,
        render_data: &ViewerRenderData,
    ) {
        if let Some(response) = response {
            let modifiers = ui.input(|i| i.modifiers);
            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = ui.input(|i| i.pointer.delta());
                if modifiers.shift {
                    self.viewport.inflated_stage_camera.pan_screen(delta.x, delta.y);
                } else {
                    self.viewport.inflated_stage_camera.handle_drag(delta.x, delta.y);
                }
            }
            if response.dragged_by(egui::PointerButton::Middle) {
                let delta = ui.input(|i| i.pointer.delta());
                self.viewport.inflated_stage_camera.pan_screen(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    self.viewport.inflated_stage_camera.handle_scroll(scroll * 0.01);
                }
            }
        }

        let aspect = rect.width() / rect.height().max(1.0);
        let vp = self.viewport.inflated_stage_camera.view_projection(aspect);
        let fog_span =
            (self.viewport.inflated_stage_camera.distance + self.viewport.volume_extent).max(1.0);
        let render_3d = self.viewport.workflow_render_3d();
        let fog_near = fog_span * render_3d.fog_start_fraction;
        let fog_far = fog_span * render_3d.fog_end_fraction;
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            callbacks::Scene3DCallback {
                view_proj: vp,
                camera_pos: self.viewport.inflated_stage_camera.eye(),
                camera_dir: self.viewport.inflated_stage_camera.view_direction(),
                streamline_draws: Vec::new(),
                show_streamlines: false,
                volume_draws: Vec::new(),
                slice_visible: [false; 3],
                surface_draws: render_data.surface_draws.clone(),
                bundle_draws: Vec::new(),
                show_boundary_glyphs: false,
                boundary_glyph_color_mode: BoundaryGlyphColorMode::DirectionRgb,
                boundary_glyph_draw_step: 1,
                scene_lighting: self.viewport.scene_lighting(),
                render_3d,
                fog_near,
                fog_far,
            },
        ));
    }

    fn draw_slice_rect(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: Option<&egui::Response>,
        axis_index: usize,
        slice_pos: f32,
        render_data: &ViewerRenderData,
        show_crosshairs: bool,
    ) {
        if let Some(response) = response {
            if response.clicked() {
                self.viewport.view_2d.active_axis = axis_index;
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = ui.input(|i| i.pointer.delta());
                self.viewport.slice_cameras[axis_index].pan_screen(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let step = if scroll > 0.0 { 1 } else { -1 };
                    self.viewport.step_slice(
                        &self.scene.nifti_files,
                        &self.scene.gifti_surfaces,
                        axis_index,
                        step,
                    );
                }

                let zoom_delta = ui.input(|i| i.zoom_delta());
                if (zoom_delta - 1.0).abs() > 0.001 {
                    let zoom_amount = (zoom_delta - 1.0) * 10.0;
                    self.viewport.slice_cameras[axis_index].zoom(zoom_amount);
                }
            }
        }

        let aspect = rect.width() / rect.height().max(1.0);
        let vp = self.viewport.slice_cameras[axis_index].view_projection(aspect, slice_pos);
        let glyph_slab_half_width = self
            .viewport
            .boundary_field
            .as_ref()
            .map(|field| 0.5 * field.grid.voxel_size_mm)
            .unwrap_or(0.0);
        let slab_axis = match axis_index {
            0 => 2u32,
            1 => 1u32,
            _ => 0u32,
        };

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            callbacks::SliceViewCallback {
                view_proj: vp,
                quad_index: axis_index,
                bind_group_index: axis_index + 1,
                volume_draws: render_data.volume_draws.clone(),
                streamline_draws: render_data.streamline_draws.clone(),
                show_streamlines: render_data.any_visible_streamlines,
                slab_axis,
                slab_min: slice_pos - glyph_slab_half_width,
                slab_max: slice_pos + glyph_slab_half_width,
                show_boundary_glyphs: render_data.glyph_visible,
                boundary_glyph_color_mode: render_data.glyph_color_mode,
                boundary_glyph_draw_step: render_data.glyph_slice_density_step,
                scene_lighting: self.viewport.scene_lighting(),
            },
        ));

        if show_crosshairs {
            self.draw_crosshairs(ui, rect, axis_index, vp);
        }
        self.draw_orientation_labels(ui, rect, axis_index, vp);
        self.draw_mesh_intersections(ui, rect, axis_index, vp, slice_pos);
        self.draw_bundle_mesh_intersections(ui, rect, axis_index, vp, slice_pos);
        self.draw_parcellation_intersections(ui, rect, axis_index, vp, slice_pos);
    }

    fn build_viewer_render_data(&self) -> ViewerRenderData {
        let surface_draws = self
            .workflow
            .runtime
            .scene_plan
            .surface_draws
            .iter()
            .map(|draw| {
                (
                    draw.source_id,
                    0,
                    MeshDrawStyle {
                        color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                        scalar_min: draw.range_min,
                        scalar_max: draw.range_max,
                        scalar_enabled: draw.show_projection_map,
                        vertex_color_enabled: !draw.vertex_rgba.is_empty(),
                        colormap: draw.projection_colormap,
                        gloss: draw.gloss,
                        map_opacity: draw.map_opacity,
                        map_threshold: draw.map_threshold,
                        model_matrix: draw.model_matrix,
                    },
                )
            })
            .collect();

        let volume_draws = self
            .workflow
            .runtime
            .scene_plan
            .volume_draws
            .iter()
            .map(|draw| VolumeDrawInfo {
                file_id: draw.source_id,
                window_center: draw.window_center,
                window_width: draw.window_width,
                colormap: draw.colormap.as_u32(),
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>();

        let streamline_draws = self
            .workflow
            .runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| StreamlineDrawInfo {
                file_id: draw.draw_id,
                visible: draw.visible,
                render_style: draw.render_style,
                tube_radius: draw.tube_radius_mm,
                slab_half_width: draw.slab_half_width_mm,
            })
            .collect::<Vec<_>>();

        let bundle_draws = self
            .workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| BundleDrawInfo {
                file_id: draw.draw_id,
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>();

        let glyph_draw = self
            .workflow
            .runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible);

        ViewerRenderData {
            any_visible_streamlines: streamline_draws.iter().any(|draw| draw.visible),
            surface_draws,
            volume_draws,
            streamline_draws,
            bundle_draws,
            glyph_visible: glyph_draw.is_some() && self.viewport.boundary_field.is_some(),
            glyph_color_mode: glyph_draw
                .map(|draw| draw.color_mode)
                .unwrap_or(BoundaryGlyphColorMode::DirectionRgb),
            glyph_density_3d_step: glyph_draw
                .map(|draw| draw.density_3d_step as u32)
                .unwrap_or(1),
            glyph_slice_density_step: glyph_draw
                .map(|draw| draw.slice_density_step as u32)
                .unwrap_or(1),
        }
    }

    fn build_stage_render_data(&self) -> ViewerRenderData {
        let surface_draws = self.stage_surface_draw_instances();
        ViewerRenderData {
            surface_draws,
            volume_draws: Vec::new(),
            streamline_draws: Vec::new(),
            bundle_draws: Vec::new(),
            any_visible_streamlines: false,
            glyph_visible: false,
            glyph_color_mode: BoundaryGlyphColorMode::DirectionRgb,
            glyph_density_3d_step: 1,
            glyph_slice_density_step: 1,
        }
    }

    fn finish_export_if_ready(&mut self, ctx: &egui::Context, target: ExportTarget) {
        let Some(pending) = self.viewport.pending_export.as_mut() else {
            return;
        };
        if pending.target != target {
            return;
        }

        if !pending.requested_screenshot {
            pending.requested_screenshot = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            ctx.request_repaint();
            return;
        }

        let screenshot = ctx.input(|i| {
            i.events.iter().find_map(|event| match event {
                egui::Event::Screenshot {
                    viewport_id, image, ..
                } if *viewport_id == ctx.viewport_id() => Some(image.clone()),
                _ => None,
            })
        });

        let Some(image) = screenshot else {
            ctx.request_repaint();
            return;
        };

        let path = pending.path.clone();
        match save_color_image(image.as_ref(), &path) {
            Ok(()) => {
                self.status_msg = Some(format!("Saved PNG to {}", path.display()));
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to export PNG: {err}"));
            }
        }
        self.viewport.pending_export = None;
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    pub(in crate::app) fn scene_is_empty(&self) -> bool {
        self.scene.trx_files.is_empty()
            && self.scene.nifti_files.is_empty()
            && self.scene.gifti_surfaces.is_empty()
            && self.scene.parcellations.is_empty()
    }

    fn max_slice_index(&self, axis_index: usize) -> usize {
        self.scene
            .nifti_files
            .first()
            .map(|nf| match axis_index {
                0 => nf.volume.dims[2].saturating_sub(1),
                1 => nf.volume.dims[1].saturating_sub(1),
                _ => nf.volume.dims[0].saturating_sub(1),
            })
            .unwrap_or(self.viewport.slice_indices[axis_index].saturating_add(128))
    }
}

fn slice_kind_picker(ui: &mut egui::Ui, value: &mut SliceViewKind, id_salt: &'static str) {
    egui::ComboBox::from_id_salt(id_salt)
        .selected_text(value.label())
        .show_ui(ui, |ui| {
            for choice in SliceViewKind::ALL {
                ui.selectable_value(value, choice, choice.label());
            }
        });
}

fn save_color_image(image: &egui::ColorImage, path: &Path) -> anyhow::Result<()> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for pixel in &image.pixels {
        rgba.extend_from_slice(&pixel.to_array());
    }
    image::save_buffer(
        path,
        &rgba,
        image.size[0] as u32,
        image.size[1] as u32,
        image::ColorType::Rgba8,
    )?;
    Ok(())
}

fn bbox_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}

fn stage_instance_model_matrices(
    structure: Option<CiftiStructure>,
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> Vec<Mat4> {
    let center = (bbox_min + bbox_max) * 0.5;
    let extents = bbox_max - bbox_min;
    let span = extents.x.abs().max(extents.y.abs()).max(extents.z.abs()).max(1.0);
    let separation = span * 0.55;
    let lateral_row_z = span * 0.42;
    let medial_row_z = -span * 0.42;
    let center_transform = Mat4::from_translation(-center);

    match structure {
        Some(CiftiStructure::CortexLeft) => vec![
            stage_panel_transform(center_transform, separation, lateral_row_z, 90.0),
            stage_panel_transform(center_transform, separation, medial_row_z, -90.0),
        ],
        Some(CiftiStructure::CortexRight) => vec![
            stage_panel_transform(center_transform, -separation, lateral_row_z, -90.0),
            stage_panel_transform(center_transform, -separation, medial_row_z, 90.0),
        ],
        _ => vec![Mat4::IDENTITY],
    }
}

fn stage_panel_transform(
    center_transform: Mat4,
    x_shift: f32,
    z_shift: f32,
    turn_deg: f32,
) -> Mat4 {
    Mat4::from_translation(Vec3::new(x_shift, 0.0, z_shift))
        * Mat4::from_rotation_z(turn_deg.to_radians())
        * center_transform
}

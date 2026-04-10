use glam::Vec3;

use crate::app::helpers::{intersect_edge_with_slice, tri_axis_value};

impl crate::app::TrxVizApp {
    /// Draw anatomical orientation labels (R/L/A/P/S/I) on a slice view.
    pub(super) fn draw_orientation_labels(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        _axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        let center = self.viewport.volume_center;
        let axis_len = (self.viewport.volume_extent * 0.2).max(10.0);

        // Define the 6 anatomical directions as offsets from center.
        let directions: &[(Vec3, &str)] = &[
            (Vec3::X * axis_len, "R"),  // +X = Right
            (-Vec3::X * axis_len, "L"), // -X = Left
            (Vec3::Y * axis_len, "A"),  // +Y = Anterior
            (-Vec3::Y * axis_len, "P"), // -Y = Posterior
            (Vec3::Z * axis_len, "S"),  // +Z = Superior
            (-Vec3::Z * axis_len, "I"), // -Z = Inferior
        ];

        let project = |world: Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            if clip.w.abs() < 1e-6 {
                return rect.center();
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        let painter = ui.painter_at(rect);
        let label_color = egui::Color32::from_rgb(220, 220, 220);
        let font = egui::FontId::proportional(14.0);
        let margin = 16.0;
        let center_screen = project(center);

        // Place labels by projecting a small offset and extending from center to the viewport edge.
        for &(offset, label) in directions {
            let p = project(center + offset);
            let delta = egui::vec2(p.x - center_screen.x, p.y - center_screen.y);
            let len2 = delta.length_sq();
            // Skip look-axis directions that collapse to the center in this view.
            if len2 < 1e-6 {
                continue;
            }
            let dir = delta / len2.sqrt();
            let tx = if dir.x.abs() > 1e-6 {
                ((rect.width() * 0.5 - margin) / dir.x.abs()).abs()
            } else {
                f32::INFINITY
            };
            let ty = if dir.y.abs() > 1e-6 {
                ((rect.height() * 0.5 - margin) / dir.y.abs()).abs()
            } else {
                f32::INFINITY
            };
            let t = tx.min(ty);
            let label_pos = egui::pos2(center_screen.x + dir.x * t, center_screen.y + dir.y * t);

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                label,
                font.clone(),
                label_color,
            );
        }
    }

    /// Draw 3D orientation axes in the corner of the 3D viewport.
    pub(super) fn draw_3d_axes(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);

        // Place axes in bottom-left corner
        let origin_screen = egui::pos2(rect.left() + 50.0, rect.bottom() - 50.0);
        let axis_length = 30.0;

        let axes = [
            (Vec3::X, "R", egui::Color32::RED),
            (Vec3::Y, "A", egui::Color32::GREEN),
            (Vec3::Z, "S", egui::Color32::from_rgb(80, 120, 255)),
        ];

        for (dir, label, color) in axes {
            // Project the direction vector (just the rotation, no translation)
            let clip0 = view_proj * Vec3::ZERO.extend(1.0);
            let clip1 = view_proj * dir.extend(1.0);
            // Direction in NDC
            let ndc0 = egui::vec2(clip0.x / clip0.w, clip0.y / clip0.w);
            let ndc1 = egui::vec2(clip1.x / clip1.w, clip1.y / clip1.w);
            let dir_ndc = ndc1 - ndc0;
            let dir_screen = egui::vec2(dir_ndc.x, -dir_ndc.y); // flip Y
            let dir_norm = if dir_screen.length() > 0.001 {
                dir_screen / dir_screen.length()
            } else {
                egui::vec2(0.0, 0.0)
            };

            let end = origin_screen + dir_norm * axis_length;
            painter.line_segment([origin_screen, end], egui::Stroke::new(2.0, color));
            painter.text(
                end + dir_norm * 10.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }

    /// Draw crosshair lines on a 2D slice view showing the other two slice positions.
    pub(super) fn draw_crosshairs(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        // Get the world-space positions of the other two slices
        let (other_a, other_b) = match axis_index {
            // Axial view: show coronal (Y) and sagittal (X) positions
            0 => (
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 2),
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 1),
            ),
            // Coronal view: show sagittal (X) and axial (Z) positions
            1 => (
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 2),
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 0),
            ),
            // Sagittal view: show coronal (Y) and axial (Z) positions
            _ => (
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 1),
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 0),
            ),
        };

        let slice_pos = self
            .viewport
            .slice_world_position(&self.scene.nifti_files, axis_index);

        // Create world-space points on the crosshair lines and project them
        // For each crosshair line, we create two points at the extremes of the view
        let far = 10000.0;
        let (h_p1, h_p2, v_p1, v_p2) = match axis_index {
            0 => {
                // Axial: horizontal = coronal(Y), vertical = sagittal(X)
                let y = other_b; // coronal Y position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, y, slice_pos),
                    glam::Vec3::new(far, y, slice_pos),
                    glam::Vec3::new(x, -far, slice_pos),
                    glam::Vec3::new(x, far, slice_pos),
                )
            }
            1 => {
                // Coronal: horizontal = axial(Z), vertical = sagittal(X)
                let z = other_b; // axial Z position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, slice_pos, z),
                    glam::Vec3::new(far, slice_pos, z),
                    glam::Vec3::new(x, slice_pos, -far),
                    glam::Vec3::new(x, slice_pos, far),
                )
            }
            _ => {
                // Sagittal: horizontal = axial(Z), vertical = coronal(Y)
                let z = other_b; // axial Z position
                let y = other_a; // coronal Y position
                (
                    glam::Vec3::new(slice_pos, -far, z),
                    glam::Vec3::new(slice_pos, far, z),
                    glam::Vec3::new(slice_pos, y, -far),
                    glam::Vec3::new(slice_pos, y, far),
                )
            }
        };

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // NDC [-1,1] → screen rect
            let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
            let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(); // flip Y
            egui::pos2(sx, sy)
        };

        let crosshair_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 128);
        let stroke = egui::Stroke::new(1.0, crosshair_color);
        let painter = ui.painter_at(rect);

        // Horizontal line (clipped to rect)
        painter.line_segment([project(h_p1), project(h_p2)], stroke);
        // Vertical line (clipped to rect)
        painter.line_segment([project(v_p1), project(v_p2)], stroke);
    }

    pub(super) fn draw_mesh_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        if self.scene.gifti_surfaces.is_empty() {
            return;
        }
        let painter = ui.painter_at(rect);
        let eps = 1e-4f32;

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        for draw in &self.workflow.runtime.scene_plan.surface_draws {
            if draw.opacity <= 0.01 {
                continue;
            }
            let Some(surface) = self
                .scene
                .gifti_surfaces
                .iter()
                .find(|surface| surface.id == draw.source_id)
            else {
                continue;
            };

            // Surface-level early out by axis-aligned bounds.
            let (smin, smax) = match axis_index {
                0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
                1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
                _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
            };
            if slice_pos < smin - eps || slice_pos > smax + eps {
                continue;
            }

            let color = egui::Color32::from_rgba_unmultiplied(
                (draw.outline_color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.outline_color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.outline_color[2].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.opacity.clamp(0.0, 1.0) * 255.0) as u8,
            );
            let stroke = egui::Stroke::new(draw.outline_thickness.clamp(0.25, 8.0), color);

            for tri in surface.data.indices.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                let a = glam::Vec3::from(surface.data.vertices[ia]);
                let b = glam::Vec3::from(surface.data.vertices[ib]);
                let c = glam::Vec3::from(surface.data.vertices[ic]);

                let tmin = tri_axis_value(a, axis_index)
                    .min(tri_axis_value(b, axis_index))
                    .min(tri_axis_value(c, axis_index));
                let tmax = tri_axis_value(a, axis_index)
                    .max(tri_axis_value(b, axis_index))
                    .max(tri_axis_value(c, axis_index));
                if slice_pos < tmin - eps || slice_pos > tmax + eps {
                    continue;
                }

                let mut pts = Vec::with_capacity(3);
                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                    if let Some(p) = intersect_edge_with_slice(p0, p1, axis_index, slice_pos, eps) {
                        if !pts
                            .iter()
                            .any(|q: &glam::Vec3| (*q - p).length_squared() <= eps * eps)
                        {
                            pts.push(p);
                        }
                    }
                }
                if pts.len() < 2 {
                    continue;
                }
                // For rare 3-point cases (vertex on plane), keep the longest segment.
                let (p0, p1) = if pts.len() == 2 {
                    (pts[0], pts[1])
                } else {
                    let mut best = (pts[0], pts[1]);
                    let mut best_d2 = (pts[1] - pts[0]).length_squared();
                    for i in 0..pts.len() {
                        for j in (i + 1)..pts.len() {
                            let d2 = (pts[j] - pts[i]).length_squared();
                            if d2 > best_d2 {
                                best = (pts[i], pts[j]);
                                best_d2 = d2;
                            }
                        }
                    }
                    best
                };

                painter.line_segment([project(p0), project(p1)], stroke);
            }
        }
    }

    pub(super) fn draw_bundle_mesh_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        if self.workflow.runtime.scene_plan.bundle_draws.is_empty() {
            return;
        }

        let painter = ui.painter_at(rect);
        let eps = 1e-4f32;

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        for draw in &self.workflow.runtime.scene_plan.bundle_draws {
            if draw.opacity <= 0.01 {
                continue;
            }
            let Some(runtime) = self.workflow.display_runtimes.get(&draw.node_uuid) else {
                continue;
            };
            if runtime.bundle_meshes_cpu.is_empty() {
                continue;
            }

            for mesh in &runtime.bundle_meshes_cpu {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }

                let mut bbox_min = glam::Vec3::splat(f32::INFINITY);
                let mut bbox_max = glam::Vec3::splat(f32::NEG_INFINITY);
                for vertex in &mesh.vertices {
                    let pos = glam::Vec3::from(vertex.position);
                    bbox_min = bbox_min.min(pos);
                    bbox_max = bbox_max.max(pos);
                }

                let (smin, smax) = match axis_index {
                    0 => (bbox_min.z, bbox_max.z),
                    1 => (bbox_min.y, bbox_max.y),
                    _ => (bbox_min.x, bbox_max.x),
                };
                if slice_pos < smin - eps || slice_pos > smax + eps {
                    continue;
                }

                for tri in mesh.indices.chunks_exact(3) {
                    let ia = tri[0] as usize;
                    let ib = tri[1] as usize;
                    let ic = tri[2] as usize;
                    let a = glam::Vec3::from(mesh.vertices[ia].position);
                    let b = glam::Vec3::from(mesh.vertices[ib].position);
                    let c = glam::Vec3::from(mesh.vertices[ic].position);

                    let tmin = tri_axis_value(a, axis_index)
                        .min(tri_axis_value(b, axis_index))
                        .min(tri_axis_value(c, axis_index));
                    let tmax = tri_axis_value(a, axis_index)
                        .max(tri_axis_value(b, axis_index))
                        .max(tri_axis_value(c, axis_index));
                    if slice_pos < tmin - eps || slice_pos > tmax + eps {
                        continue;
                    }

                    let rgb = [
                        (mesh.vertices[ia].color[0]
                            + mesh.vertices[ib].color[0]
                            + mesh.vertices[ic].color[0])
                            / 3.0,
                        (mesh.vertices[ia].color[1]
                            + mesh.vertices[ib].color[1]
                            + mesh.vertices[ic].color[1])
                            / 3.0,
                        (mesh.vertices[ia].color[2]
                            + mesh.vertices[ib].color[2]
                            + mesh.vertices[ic].color[2])
                            / 3.0,
                    ];
                    let color = egui::Color32::from_rgba_unmultiplied(
                        (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
                        (draw.opacity.clamp(0.0, 1.0) * 255.0) as u8,
                    );
                    let stroke = egui::Stroke::new(draw.outline_thickness.clamp(0.25, 8.0), color);

                    let mut pts = Vec::with_capacity(3);
                    for (p0, p1) in [(a, b), (b, c), (c, a)] {
                        if let Some(p) =
                            intersect_edge_with_slice(p0, p1, axis_index, slice_pos, eps)
                        {
                            if !pts
                                .iter()
                                .any(|q: &glam::Vec3| (*q - p).length_squared() <= eps * eps)
                            {
                                pts.push(p);
                            }
                        }
                    }
                    if pts.len() < 2 {
                        continue;
                    }

                    let (p0, p1) = if pts.len() == 2 {
                        (pts[0], pts[1])
                    } else {
                        let mut best = (pts[0], pts[1]);
                        let mut best_d2 = (pts[1] - pts[0]).length_squared();
                        for i in 0..pts.len() {
                            for j in (i + 1)..pts.len() {
                                let d2 = (pts[j] - pts[i]).length_squared();
                                if d2 > best_d2 {
                                    best = (pts[i], pts[j]);
                                    best_d2 = d2;
                                }
                            }
                        }
                        best
                    };

                    painter.line_segment([project(p0), project(p1)], stroke);
                }
            }
        }
    }

    pub(super) fn draw_parcellation_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        if self
            .workflow
            .runtime
            .scene_plan
            .parcellation_draws
            .is_empty()
        {
            return;
        }

        let painter = ui.painter_at(rect);
        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        for draw in &self.workflow.runtime.scene_plan.parcellation_draws {
            let Some(parcellation) = self
                .scene
                .parcellations
                .iter()
                .find(|asset| asset.asset.id == draw.source_id)
            else {
                continue;
            };
            let Some(slice_index) = parcellation.asset.data.nearest_slice_index(
                axis_index,
                slice_pos,
                self.viewport.volume_center,
            ) else {
                continue;
            };
            let labels = if draw.labels.is_empty() {
                parcellation
                    .asset
                    .data
                    .label_table
                    .keys()
                    .copied()
                    .filter(|label| *label != 0)
                    .collect()
            } else {
                draw.labels.clone()
            };
            for (segment, color) in
                parcellation
                    .asset
                    .data
                    .slice_contour_segments(axis_index, slice_index, &labels)
            {
                let stroke = egui::Stroke::new(
                    1.2,
                    egui::Color32::from_rgba_unmultiplied(
                        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
                        (draw.opacity.clamp(0.0, 1.0) * 255.0) as u8,
                    ),
                );
                painter.line_segment([project(segment[0]), project(segment[1])], stroke);
            }
        }
    }
}

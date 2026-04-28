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
        let center = self.viewport.volume_center();
        let axis_len = (self.viewport.volume_extent() * 0.2).max(10.0);

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

    /// Draw 3D orientation axes in the corner of the main 3D viewport.
    pub(super) fn draw_3d_axes(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);
        let origin_screen = egui::pos2(rect.left() + 50.0, rect.bottom() - 50.0);
        let axis_length = 30.0;

        let axes = [
            (Vec3::X, "R", egui::Color32::RED),
            (Vec3::Y, "A", egui::Color32::GREEN),
            (Vec3::Z, "S", egui::Color32::from_rgb(80, 120, 255)),
        ];

        for (dir, label, color) in axes {
            let clip0 = view_proj * Vec3::ZERO.extend(1.0);
            let clip1 = view_proj * dir.extend(1.0);
            let ndc0 = egui::vec2(clip0.x / clip0.w, clip0.y / clip0.w);
            let ndc1 = egui::vec2(clip1.x / clip1.w, clip1.y / clip1.w);
            let dir_ndc = ndc1 - ndc0;
            let dir_screen = egui::vec2(dir_ndc.x, -dir_ndc.y);
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

    pub(super) fn draw_voxel_mask_mesh_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        use trxviz_core::workflow::VoxelMaskRenderStyle;
        // VoxelMaskSliceMode is also used inside draw_voxel_accurate_overlay
        // below — kept in scope via fully-qualified path on the helper.

        if self
            .workflow
            .runtime
            .scene_plan
            .voxel_mask_mesh_draws
            .is_empty()
        {
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

        // Slice-axis world coordinate the viewer cuts through. Layout in
        // egui-overlay code:
        //   axis_index 0 → sagittal slab (constant world Z plane)
        //   axis_index 1 → coronal  slab (constant world Y plane)
        //   axis_index 2 → axial    slab (constant world X plane)
        // (matching the bbox-axis check below.)
        let world_axis = match axis_index {
            0 => 2usize, // Z
            1 => 1usize, // Y
            _ => 0usize, // X
        };

        for draw in &self.workflow.runtime.scene_plan.voxel_mask_mesh_draws {
            if draw.opacity <= 0.01 {
                continue;
            }

            let color = egui::Color32::from_rgba_unmultiplied(
                (draw.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.color[2].clamp(0.0, 1.0) * 255.0) as u8,
                (draw.opacity.clamp(0.0, 1.0) * 255.0) as u8,
            );

            if matches!(draw.style, VoxelMaskRenderStyle::VoxelAccurate) {
                draw_voxel_accurate_overlay(
                    &painter,
                    &project,
                    &draw.voxel_mask,
                    world_axis,
                    slice_pos,
                    color,
                    draw.slice_mode,
                );
                continue;
            }

            let Some(cache) = self
                .workflow
                .execution_cache
                .voxel_mask_mesh_cache
                .get(&draw.node_uuid)
                .filter(|c| c.fingerprint == draw.fingerprint)
            else {
                continue;
            };
            let mesh = &cache.mesh;
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

            let stroke = egui::Stroke::new(1.5, color);

            for tri in mesh.indices.chunks_exact(3) {
                let a = glam::Vec3::from(mesh.vertices[tri[0] as usize].position);
                let b = glam::Vec3::from(mesh.vertices[tri[1] as usize].position);
                let c = glam::Vec3::from(mesh.vertices[tri[2] as usize].position);

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
                self.viewport.volume_center(),
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
                    .filter(|label| label.0 != 0)
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

/// Voxel-accurate 2D overlay: for every "on" voxel whose world-space
/// cube straddles the slice plane, fill the polygon where the cube meets
/// the plane. For axis-aligned affines this collapses to a perfect grid
/// of voxel rectangles; for oblique affines it produces correctly
/// tilted/clipped polygons.
fn draw_voxel_accurate_overlay(
    painter: &egui::Painter,
    project: &dyn Fn(glam::Vec3) -> egui::Pos2,
    mask: &trxviz_core::workflow::VoxelMask,
    world_axis: usize,
    slice_pos: f32,
    color: egui::Color32,
    slice_mode: trxviz_core::workflow::VoxelMaskSliceMode,
) {
    use trxviz_core::workflow::VoxelMaskSliceMode;

    let dims = mask.dims;
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    if nx == 0 || ny == 0 || nz == 0 || mask.data.len() != nx * ny * nz {
        return;
    }
    let aff = mask.voxel_to_ras;
    // Bounding-box reject the whole mask before iterating.
    let mut bbox_min = glam::Vec3::splat(f32::INFINITY);
    let mut bbox_max = glam::Vec3::splat(f32::NEG_INFINITY);
    for &dx in &[0.0f32, nx as f32] {
        for &dy in &[0.0f32, ny as f32] {
            for &dz in &[0.0f32, nz as f32] {
                let p = aff.transform_point3(glam::Vec3::new(dx, dy, dz));
                bbox_min = bbox_min.min(p);
                bbox_max = bbox_max.max(p);
            }
        }
    }
    let eps = 1e-4f32;
    if slice_pos < bbox_min[world_axis] - eps || slice_pos > bbox_max[world_axis] + eps {
        return;
    }

    // Slice plane in voxel coords: n_v · v = d_v, where n_v = M^T e_world.
    let m3 = glam::Mat3::from_mat4(aff);
    let n_v = match world_axis {
        0 => m3.transpose().x_axis,
        1 => m3.transpose().y_axis,
        _ => m3.transpose().z_axis,
    };
    let aff_translation = glam::Vec3::new(aff.w_axis.x, aff.w_axis.y, aff.w_axis.z);
    let d_v = slice_pos - aff_translation[world_axis];

    // Outline mode draws the perimeter of the masked region via
    // `draw_voxel_outline_contour`. The slice snaps to the voxel layer
    // along the dominant voxel axis, matching how the anatomical texture
    // and parcellation contours behave for oblique affines. Falls
    // through to filled only for the degenerate case where `n_v == 0`
    // (impossible for an invertible affine, but guarded for safety).
    if matches!(slice_mode, VoxelMaskSliceMode::Outline) {
        if let Some(normal_axis) = dominant_voxel_axis(n_v) {
            draw_voxel_outline_contour(painter, project, mask, aff, n_v, d_v, normal_axis, color);
            return;
        }
    }

    let cube_half_extent = 0.5 * (n_v.x.abs() + n_v.y.abs() + n_v.z.abs());

    // Iterate only on-voxels (cached on the mask). For typical ROIs of
    // ~10K voxels in a 256³ volume this is ~1600× cheaper than
    // scanning the full grid each frame.
    let on_voxels = mask.nonzero_voxel_indices();
    let mut offset_buf: Vec<glam::Vec3> = Vec::with_capacity(8);
    let mut polygon_buf: Vec<egui::Pos2> = Vec::with_capacity(8);

    for [i, j, k] in on_voxels.iter() {
        // Center value of n_v · v for the voxel center (i+0.5, j+0.5, k+0.5).
        let center_v =
            n_v.x * (*i as f32 + 0.5) + n_v.y * (*j as f32 + 0.5) + n_v.z * (*k as f32 + 0.5);
        if (d_v - center_v).abs() > cube_half_extent + eps {
            continue;
        }
        let voxel_origin = glam::Vec3::new(*i as f32, *j as f32, *k as f32);
        cube_slice_polygon_voxel_offsets(aff, voxel_origin, world_axis, slice_pos, &mut offset_buf);
        if offset_buf.len() < 3 {
            continue;
        }

        polygon_buf.clear();
        for o in &offset_buf {
            polygon_buf.push(project(aff.transform_point3(voxel_origin + *o)));
        }
        painter.add(egui::Shape::convex_polygon(
            polygon_buf.clone(),
            color,
            egui::Stroke::NONE,
        ));
    }
}

/// Return the voxel-space axis index (0/1/2) whose `n_v` component is
/// largest in absolute value — i.e., the voxel axis most perpendicular
/// to the slice plane. Used by outline mode to pick which voxel-layer
/// the slice should snap to.
///
/// For axis-aligned / 90°-rotated affines this is the only non-zero
/// component. For oblique affines it's an approximation: outline edges
/// are emitted on a single layer of voxels along this axis and projected
/// through `voxel_to_ras` to world space, matching the parcellation
/// contour overlay's behavior on oblique data
/// ([parcellation_data.rs:181](trxviz-core/src/data/parcellation_data.rs:181)).
/// Both overlays therefore track the same voxel-aligned slab as the
/// underlying anatomical texture.
///
/// Returns `None` only for the degenerate `n_v == 0` case (impossible
/// for an invertible affine).
fn dominant_voxel_axis(n_v: glam::Vec3) -> Option<usize> {
    let abs = [n_v.x.abs(), n_v.y.abs(), n_v.z.abs()];
    let mut best = 0usize;
    for i in 1..3 {
        if abs[i] > abs[best] {
            best = i;
        }
    }
    if abs[best] <= 1e-9 {
        return None;
    }
    Some(best)
}

/// Outline mode for voxel masks.
///
/// The slice plane snaps to one voxel layer along `normal_axis` (the
/// voxel axis most aligned with the world slice axis). For every
/// on-voxel in that layer, walks its 4 in-plane face neighbors; where
/// the neighbor is off (or out of bounds), emits the line segment where
/// the shared face meets the slice plane. Endpoints are projected
/// through `voxel_to_ras` to world space and orthographically projected
/// to screen.
///
/// For axis-aligned and 90°-rotated affines the outline is exact. For
/// oblique affines the outline matches the parcellation contour
/// convention: it shows the voxel-aligned slab's perimeter projected
/// through the affine, which lines up with the voxel-aligned anatomical
/// texture under orthographic slice projection.
fn draw_voxel_outline_contour(
    painter: &egui::Painter,
    project: &dyn Fn(glam::Vec3) -> egui::Pos2,
    mask: &trxviz_core::workflow::VoxelMask,
    aff: glam::Mat4,
    n_v: glam::Vec3,
    d_v: f32,
    normal_axis: usize,
    color: egui::Color32,
) {
    let stroke = egui::Stroke::new(1.5, color);
    voxel_outline_edges(mask, aff, n_v, d_v, normal_axis, |a, b| {
        painter.line_segment([project(a), project(b)], stroke);
    });
}

/// Compute outline-mode edge segments and hand each one to `emit` as a
/// `[world_a, world_b]` pair. Pure logic — no `egui::Painter` — so the
/// algorithm is unit-testable in isolation. Both `draw_voxel_outline_contour`
/// and the test suite go through this entry point.
fn voxel_outline_edges<F>(
    mask: &trxviz_core::workflow::VoxelMask,
    aff: glam::Mat4,
    n_v: glam::Vec3,
    d_v: f32,
    normal_axis: usize,
    mut emit: F,
) where
    F: FnMut(glam::Vec3, glam::Vec3),
{
    let n_normal = match normal_axis {
        0 => n_v.x,
        1 => n_v.y,
        _ => n_v.z,
    };
    if n_normal.abs() < 1e-9 {
        return;
    }
    // Snap the slice to the voxel-axis layer it crosses *at the volume
    // center*, matching the parcellation contour overlay's convention
    // ([parcellation_data.rs:155](trxviz-core/src/data/parcellation_data.rs:155)).
    // For axis-aligned data n_v has zero off-axis terms and this
    // collapses to `d_v / n_normal`. For oblique data the slice plane
    // crosses the voxel grid at a layer that varies across the volume;
    // we pick the central one so the outline lines up with what the
    // anatomical texture and parcellation contours show in the same
    // viewport.
    let center = glam::Vec3::new(
        mask.dims[0] as f32 * 0.5,
        mask.dims[1] as f32 * 0.5,
        mask.dims[2] as f32 * 0.5,
    );
    let off_axis_sum = match normal_axis {
        0 => n_v.y * center.y + n_v.z * center.z,
        1 => n_v.x * center.x + n_v.z * center.z,
        _ => n_v.x * center.x + n_v.y * center.y,
    };
    let slice_pos_v = (d_v - off_axis_sum) / n_normal;
    let dim_normal = mask.dims[normal_axis] as f32;
    if slice_pos_v < 0.0 || slice_pos_v > dim_normal {
        return;
    }

    // Voxel layer the slice cuts through. `floor` puts a slice at the
    // integer face into the lower voxel; clamp covers `slice_pos_v ==
    // dims` (top boundary).
    let dim_normal_i = mask.dims[normal_axis] as i32;
    let layer = (slice_pos_v.floor() as i32).clamp(0, dim_normal_i - 1);

    let in_plane: [usize; 2] = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };

    let dims_i = [
        mask.dims[0] as i32,
        mask.dims[1] as i32,
        mask.dims[2] as i32,
    ];
    let dims_us = [
        mask.dims[0] as usize,
        mask.dims[1] as usize,
        mask.dims[2] as usize,
    ];
    let lin = |i: i32, j: i32, k: i32| -> usize {
        (i as usize) + dims_us[0] * ((j as usize) + dims_us[1] * (k as usize))
    };

    let on_voxels = mask.nonzero_voxel_indices();

    for [i, j, k] in on_voxels.iter() {
        let coords = [*i as i32, *j as i32, *k as i32];
        if coords[normal_axis] != layer {
            continue;
        }
        for &axis in &in_plane {
            let other_axis = if axis == in_plane[0] {
                in_plane[1]
            } else {
                in_plane[0]
            };
            for &dir in &[-1i32, 1i32] {
                let mut neighbor = coords;
                neighbor[axis] += dir;
                let neighbor_on = neighbor[0] >= 0
                    && neighbor[1] >= 0
                    && neighbor[2] >= 0
                    && neighbor[0] < dims_i[0]
                    && neighbor[1] < dims_i[1]
                    && neighbor[2] < dims_i[2]
                    && mask.data[lin(neighbor[0], neighbor[1], neighbor[2])] != 0;
                if neighbor_on {
                    continue;
                }
                let mut p0 = [coords[0] as f32, coords[1] as f32, coords[2] as f32];
                p0[axis] += if dir > 0 { 1.0 } else { 0.0 };
                p0[normal_axis] = slice_pos_v;
                let mut p1 = p0;
                p1[other_axis] += 1.0;

                let pa = aff.transform_point3(glam::Vec3::from(p0));
                let pb = aff.transform_point3(glam::Vec3::from(p1));
                emit(pa, pb);
            }
        }
    }
}

/// Compute, in **voxel offset space** (each coordinate in `[0, 1]`), the
/// convex polygon where the unit cube `[0,1]³` intersects the world-space
/// slice plane defined by transforming `voxel_to_ras` and slicing at
/// `world_pos[world_axis] == slice_pos`. Vertices are CCW-ordered around
/// their centroid in the slice plane. The output buffer is cleared.
///
/// Returns offsets (not absolute voxel positions) so the caller can reuse
/// them to identify which cube face an edge midpoint lies on, regardless
/// of which voxel index this cube belongs to.
fn cube_slice_polygon_voxel_offsets(
    voxel_to_ras: glam::Mat4,
    voxel_origin: glam::Vec3,
    world_axis: usize,
    slice_pos: f32,
    out: &mut Vec<glam::Vec3>,
) {
    out.clear();
    // 8 cube corners in voxel space (offsets).
    const CORNERS: [glam::Vec3; 8] = [
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 1.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 0.0),
        glam::Vec3::new(0.0, 0.0, 1.0),
        glam::Vec3::new(1.0, 0.0, 1.0),
        glam::Vec3::new(0.0, 1.0, 1.0),
        glam::Vec3::new(1.0, 1.0, 1.0),
    ];
    let mut sgn: [f32; 8] = [0.0; 8];
    for (idx, off) in CORNERS.iter().enumerate() {
        let p = voxel_to_ras.transform_point3(voxel_origin + *off);
        sgn[idx] = p[world_axis] - slice_pos;
    }
    // The 12 cube edges (pairs of corner indices).
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ];
    let eps = 1e-6f32;
    let mut offsets: Vec<glam::Vec3> = Vec::with_capacity(8);
    for (a, b) in EDGES.iter() {
        let oa = CORNERS[*a];
        let ob = CORNERS[*b];
        let ca = sgn[*a];
        let cb = sgn[*b];
        let pa_on = ca.abs() <= eps;
        let pb_on = cb.abs() <= eps;
        if pa_on {
            offsets.push(oa);
        }
        if pb_on {
            offsets.push(ob);
        }
        if pa_on || pb_on {
            continue;
        }
        if ca * cb >= 0.0 {
            continue;
        }
        let t = ca / (ca - cb);
        offsets.push(oa.lerp(ob, t));
    }
    if offsets.len() < 3 {
        return;
    }
    // Dedup near-duplicates.
    let dup_eps2 = 1e-8f32;
    let mut deduped: Vec<glam::Vec3> = Vec::with_capacity(offsets.len());
    for p in offsets {
        if !deduped
            .iter()
            .any(|q| (*q - p).length_squared() <= dup_eps2)
        {
            deduped.push(p);
        }
    }
    if deduped.len() < 3 {
        return;
    }
    // Sort CCW around centroid in the slice plane (in world space, since
    // the slice plane is axis-aligned in world).
    let world_basis = match world_axis {
        0 => (glam::Vec3::Y, glam::Vec3::Z),
        1 => (glam::Vec3::X, glam::Vec3::Z),
        _ => (glam::Vec3::X, glam::Vec3::Y),
    };
    let world: Vec<glam::Vec3> = deduped
        .iter()
        .map(|o| voxel_to_ras.transform_point3(voxel_origin + *o))
        .collect();
    let centroid_w =
        world.iter().copied().fold(glam::Vec3::ZERO, |a, b| a + b) / (world.len() as f32);
    let mut indexed: Vec<(usize, f32)> = world
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let d = *w - centroid_w;
            (i, d.dot(world_basis.1).atan2(d.dot(world_basis.0)))
        })
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, _) in indexed {
        out.push(deduped[idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::{cube_slice_polygon_voxel_offsets, dominant_voxel_axis, voxel_outline_edges};
    use trxviz_core::workflow::VoxelMask;

    #[test]
    fn cube_slice_axis_aligned_axial_returns_unit_square() {
        // Identity affine, voxel at origin, slice at z = 0.5 (mid-cube).
        let mut out = Vec::new();
        cube_slice_polygon_voxel_offsets(glam::Mat4::IDENTITY, glam::Vec3::ZERO, 2, 0.5, &mut out);
        assert_eq!(out.len(), 4, "axial mid-slice ⇒ 4-corner quad");
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        for p in &out {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
        }
        assert!((x_min - 0.0).abs() < 1e-5);
        assert!((x_max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cube_slice_through_face_emits_face_polygon() {
        let mut out = Vec::new();
        cube_slice_polygon_voxel_offsets(glam::Mat4::IDENTITY, glam::Vec3::ZERO, 2, 0.0, &mut out);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn cube_slice_outside_returns_empty() {
        let mut out = Vec::new();
        cube_slice_polygon_voxel_offsets(glam::Mat4::IDENTITY, glam::Vec3::ZERO, 2, 5.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn cube_slice_oblique_diagonal_emits_hexagon() {
        let rot = glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let mut out = Vec::new();
        cube_slice_polygon_voxel_offsets(rot, glam::Vec3::ZERO, 2, 0.0, &mut out);
        assert!(out.len() >= 3);
    }

    #[test]
    fn dominant_voxel_axis_picks_largest_component() {
        assert_eq!(dominant_voxel_axis(glam::Vec3::new(0.0, 0.0, 2.0)), Some(2));
        assert_eq!(dominant_voxel_axis(glam::Vec3::new(1.5, 0.0, 0.0)), Some(0));
    }

    #[test]
    fn dominant_voxel_axis_tolerates_floating_point_noise() {
        // Sub-1% off-axis components: still axis-aligned.
        assert_eq!(
            dominant_voxel_axis(glam::Vec3::new(1e-5, 1e-5, 1.0)),
            Some(2)
        );
    }

    #[test]
    fn dominant_voxel_axis_picks_largest_for_oblique() {
        // 30° YZ rotation — Z still dominates.
        assert_eq!(
            dominant_voxel_axis(glam::Vec3::new(0.0, 0.5, 0.866)),
            Some(2)
        );
        // 45° XZ rotation — exact tie. Tie-breaking is arbitrary; either
        // axis is a valid choice and produces a coherent outline.
        let axis = dominant_voxel_axis(glam::Vec3::new(0.707, 0.0, 0.707)).unwrap();
        assert!(axis == 0 || axis == 2);
    }

    #[test]
    fn dominant_voxel_axis_handles_zero_normal() {
        assert_eq!(dominant_voxel_axis(glam::Vec3::ZERO), None);
    }

    /// Build a 3×3×3 mask with the given on-voxel indices and affine.
    fn mask_with(on: &[[u32; 3]], voxel_to_ras: glam::Mat4) -> VoxelMask {
        let mut data = vec![0u8; 27];
        for [i, j, k] in on {
            let idx = *i as usize + 3 * (*j as usize + 3 * *k as usize);
            data[idx] = 1;
        }
        VoxelMask {
            dims: [3, 3, 3],
            voxel_to_ras,
            data,
            ..Default::default()
        }
    }

    /// Compute the slice-plane normal in voxel space (`n_v`) and offset
    /// (`d_v`), the same way `draw_voxel_accurate_overlay` does.
    fn slice_plane(aff: glam::Mat4, world_axis: usize, slice_pos: f32) -> (glam::Vec3, f32) {
        let m3 = glam::Mat3::from_mat4(aff);
        let n_v = match world_axis {
            0 => m3.transpose().x_axis,
            1 => m3.transpose().y_axis,
            _ => m3.transpose().z_axis,
        };
        let t = glam::Vec3::new(aff.w_axis.x, aff.w_axis.y, aff.w_axis.z);
        let d_v = slice_pos - t[world_axis];
        (n_v, d_v)
    }

    /// `world_axis` here matches the production convention: 0=X, 1=Y,
    /// 2=Z (literal world axes).
    const AXIAL_WORLD_AXIS: usize = 2;

    #[test]
    fn outline_axis_aligned_single_voxel_emits_four_edges() {
        // Identity affine, single on-voxel at (1,1,1), axial slice at
        // world z=1.5. Expect 4 edges (in-plane perimeter of the lone
        // voxel).
        let mask = mask_with(&[[1, 1, 1]], glam::Mat4::IDENTITY);
        let (n_v, d_v) = slice_plane(glam::Mat4::IDENTITY, AXIAL_WORLD_AXIS, 1.5);
        let normal_axis = dominant_voxel_axis(n_v).unwrap();
        assert_eq!(normal_axis, 2);

        let mut edges: Vec<[glam::Vec3; 2]> = Vec::new();
        voxel_outline_edges(
            &mask,
            glam::Mat4::IDENTITY,
            n_v,
            d_v,
            normal_axis,
            |a, b| {
                edges.push([a, b]);
            },
        );
        assert_eq!(edges.len(), 4, "isolated voxel should emit 4 edges");
        for [a, b] in &edges {
            assert!((a.z - 1.5).abs() < 1e-5);
            assert!((b.z - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn outline_oblique_30deg_emits_edges_on_dominant_layer() {
        // 30° rotation about Y, single on-voxel at (1,1,1). Axial slice
        // through that voxel's center. World Z still aligns more with
        // voxel Z than with voxel X (cos 30° > sin 30°), so the
        // dominant voxel axis is still Z. Outline snaps to that layer.
        let aff = glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_6);
        let mask = mask_with(&[[1, 1, 1]], aff);

        let center_world = aff.transform_point3(glam::Vec3::new(1.5, 1.5, 1.5));
        let slice_pos = center_world.z;

        let (n_v, d_v) = slice_plane(aff, AXIAL_WORLD_AXIS, slice_pos);
        let normal_axis = dominant_voxel_axis(n_v).unwrap();
        assert_eq!(normal_axis, 2);

        let mut edges: Vec<[glam::Vec3; 2]> = Vec::new();
        voxel_outline_edges(&mask, aff, n_v, d_v, normal_axis, |a, b| {
            edges.push([a, b]);
        });
        assert_eq!(edges.len(), 4);

        // Each endpoint, after world_to_voxel, must lie on the chosen
        // layer's plane (voxel-z = slice_pos_v) and on the unit-cube
        // perimeter of voxel (1,1,*). The expected slice_pos_v matches
        // the production formula: snap to the layer the slice plane
        // crosses at the volume center.
        let inv = aff.inverse();
        let center = glam::Vec3::new(1.5, 1.5, 1.5);
        let off_axis_sum = n_v.x * center.x + n_v.y * center.y;
        let slice_pos_v = (d_v - off_axis_sum) / n_v.z;
        for [a, b] in &edges {
            for endpoint in [a, b] {
                let v = inv.transform_point3(*endpoint);
                assert!(
                    (v.z - slice_pos_v).abs() < 1e-4,
                    "endpoint not on layer: voxel-z={}, expected {}",
                    v.z,
                    slice_pos_v
                );
                let x_ok = (v.x - 1.0).abs() < 1e-4 || (v.x - 2.0).abs() < 1e-4;
                let y_ok = (v.y - 1.0).abs() < 1e-4 || (v.y - 2.0).abs() < 1e-4;
                assert!(x_ok && y_ok, "endpoint voxel-coords out of bounds: {:?}", v);
            }
        }
    }

    #[test]
    fn outline_skips_internal_edges() {
        // 2-voxel rod along voxel X. The shared face between (0,0,0)
        // and (1,0,0) must not be drawn. The perimeter of a 2×1 block
        // in the XY plane is 6 unit segments.
        let mask = mask_with(&[[0, 0, 0], [1, 0, 0]], glam::Mat4::IDENTITY);
        let (n_v, d_v) = slice_plane(glam::Mat4::IDENTITY, AXIAL_WORLD_AXIS, 0.5);
        let normal_axis = dominant_voxel_axis(n_v).unwrap();

        let mut edges: Vec<[glam::Vec3; 2]> = Vec::new();
        voxel_outline_edges(
            &mask,
            glam::Mat4::IDENTITY,
            n_v,
            d_v,
            normal_axis,
            |a, b| {
                edges.push([a, b]);
            },
        );
        assert_eq!(
            edges.len(),
            6,
            "2-voxel rod should emit 6 perimeter edges, got {}",
            edges.len()
        );
    }
}

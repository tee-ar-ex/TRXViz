use std::collections::BTreeSet;

use trxviz_core::data::loaded_files::VolumeColormap;
use trxviz_core::data::orientation_field::{BoundaryGlyphColorMode, BoundaryGlyphNormalization};
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::renderer::mesh_renderer::SurfaceColormap;

use crate::app::workflow;

#[derive(Clone, Debug)]
pub(crate) struct OdxSelectorNames {
    pub(crate) dpv_names: Vec<String>,
    pub(crate) dpf_names: Vec<String>,
}

pub(crate) struct NodeEditorContext<'a> {
    pub(crate) available_groups: &'a [String],
    pub(crate) odx_selector_names: Option<&'a OdxSelectorNames>,
    pub(crate) sh_detail_limit: Option<u32>,
    pub(crate) save_ready: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeEditorResult {
    pub(crate) save_now: bool,
}

pub(crate) fn edit_node_op(
    ui: &mut egui::Ui,
    node_uuid: workflow::WorkflowNodeUuid,
    op: &mut workflow::WorkflowNodeKind,
    ctx: NodeEditorContext<'_>,
) -> NodeEditorResult {
    let mut result = NodeEditorResult::default();

    match op {
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
        workflow::WorkflowNodeKind::GroupSelect { groups } => {
            ui.label("Comma-separated group names");
            show_group_select_editor(ui, groups, ctx.available_groups);
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
            ui.add(
                egui::DragValue::new(&mut radius_mm.0)
                    .speed(0.5)
                    .prefix("Radius "),
            );
        }
        workflow::WorkflowNodeKind::SurfaceDepthQuery { depth_mm }
        | workflow::WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
            ui.add(
                egui::DragValue::new(&mut depth_mm.0)
                    .speed(0.25)
                    .prefix("Depth "),
            );
        }
        workflow::WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
            ui.add(
                egui::DragValue::new(&mut depth_mm.0)
                    .speed(0.25)
                    .prefix("Depth "),
            );
            edit_dps_field(ui, field);
        }
        workflow::WorkflowNodeKind::ParcelSelect { labels } => {
            ui.label("Comma-separated label IDs");
            ui.small("Leave empty to use every nonzero parcel label.");
            edit_parcel_id_set(ui, labels);
        }
        workflow::WorkflowNodeKind::ParcelEnd { endpoint_count } => {
            ui.add(egui::Slider::new(endpoint_count, 1..=2).text("Matching endpoints"));
        }
        workflow::WorkflowNodeKind::ColorByDPV { field } => {
            edit_field_name(ui, field);
        }
        workflow::WorkflowNodeKind::ColorByDPS { field } => {
            edit_field_name(ui, field);
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
                    ui.selectable_value(render_style, RenderStyle::Illuminated, "Illuminated");
                    ui.selectable_value(render_style, RenderStyle::DepthCue, "Depth Cue");
                    ui.selectable_value(render_style, RenderStyle::Tubes, "Tubes");
                });
            ui.add(
                egui::DragValue::new(&mut tube_radius_mm.0)
                    .speed(0.1)
                    .prefix("Tube radius "),
            );
            ui.add(
                egui::DragValue::new(tube_sides)
                    .speed(1.0)
                    .prefix("Tube sides "),
            );
            ui.add(
                egui::DragValue::new(&mut slab_half_width_mm.0)
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
                    ui.add(egui::Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity"));
                    ui.checkbox(&mut layer.use_label_colors, "Use label table colors");
                    if layer.use_label_colors {
                        ui.small("Label overlays use the attached label-table RGBA colors.");
                    }
                    ui.add_enabled_ui(!layer.use_label_colors, |ui| {
                        show_surface_colormap_picker(
                            ui,
                            format!("surface_overlay_colormap_{}_{}", node_uuid.0, layer_index),
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
                    ui.selectable_value(space, workflow::SurfaceDisplaySpace::Stage, "Stage");
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
                    ui.selectable_value(projection_colormap, SurfaceColormap::Viridis, "Viridis");
                    ui.selectable_value(projection_colormap, SurfaceColormap::Inferno, "Inferno");
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
                    egui::DragValue::new(&mut voxel_size_mm.0)
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
                    egui::DragValue::new(&mut min_component_volume_mm3.0)
                        .speed(1.0)
                        .range(0.0..=1_000_000.0)
                        .prefix("Min component mm^3 "),
                );
            } else {
                ui.add(
                    egui::DragValue::new(&mut tube_radius_mm.0)
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
            egui::ComboBox::from_id_salt(format!("bundle_surface_color_mode_{}", node_uuid.0))
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
                egui::DragValue::new(&mut voxel_size_mm.0)
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
            egui::ComboBox::from_id_salt(format!("boundary_field_normalization_{}", node_uuid.0))
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
            egui::ComboBox::from_id_salt(format!("boundary_glyph_color_mode_{}", node_uuid.0))
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
        workflow::WorkflowNodeKind::ParcellationDisplay { labels, opacity } => {
            ui.label("Comma-separated label IDs");
            ui.small("Leave empty to use every nonzero parcel label.");
            edit_parcel_id_set(ui, labels);
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
            if ui
                .add_enabled(ctx.save_ready, egui::Button::new("Save Now"))
                .clicked()
            {
                result.save_now = true;
            }
            if !ctx.save_ready {
                ui.small("Connect a streamline input to enable export.");
            }
        }
        workflow::WorkflowNodeKind::OdfGlyphRenderer {
            scale,
            subtract_iso,
            norm_within_voxel,
            opacity,
            offset_from_slice,
            gloss,
            vertex_colormap,
            slice_axis,
            opacity_gate,
            size_gate,
            detail,
            visible,
        } => {
            let max_safe_detail = ctx.sh_detail_limit.unwrap_or(6);
            if *detail > max_safe_detail {
                *detail = max_safe_detail;
            }
            ui.checkbox(visible, "Visible");
            ui.add(egui::Slider::new(scale, 0.1..=5.0).text("Scale"));
            ui.checkbox(subtract_iso, "Subtract iso");
            ui.checkbox(norm_within_voxel, "Normalize within voxel");
            ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(gloss, 0.0..=1.0).text("Gloss"));
            ui.add(egui::Slider::new(detail, 1..=max_safe_detail).text("SH detail"));
            if max_safe_detail < 6 {
                ui.small(format!(
                    "GPU storage limit caps SH detail at {max_safe_detail} for the current slice."
                ));
            }
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
                ui.add(egui::Slider::new(&mut size_gate.min_scale, 0.0..=5.0).text("Below scale"));
                ui.add(egui::Slider::new(&mut size_gate.max_scale, 0.0..=5.0).text("Above scale"));
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
                egui::Slider::new(&mut slab_thickness_mm.0, 0.1..=20.0).text("Slab thickness (mm)"),
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
                    if let Some(odx_names) = ctx.odx_selector_names {
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
                    if let Some(odx_names) = ctx.odx_selector_names {
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

    result
}

fn edit_field_name<T>(ui: &mut egui::Ui, field: &mut T)
where
    T: From<String> + AsRef<str>,
{
    let mut value = field.as_ref().to_string();
    if ui.text_edit_singleline(&mut value).changed() {
        *field = T::from(value);
    }
}

fn edit_dps_field(ui: &mut egui::Ui, field: &mut workflow::DpsFieldName) {
    let mut value = field.as_str().to_string();
    if ui.text_edit_singleline(&mut value).changed() {
        *field = workflow::DpsFieldName::from(value);
    }
}

fn edit_parcel_id_set(ui: &mut egui::Ui, labels: &mut workflow::ParcelIdSet) {
    let mut csv = labels.to_csv();
    if ui.text_edit_multiline(&mut csv).changed() {
        *labels = workflow::ParcelIdSet::from_csv(&csv);
    }
}

fn show_group_select_editor(
    ui: &mut egui::Ui,
    groups: &mut workflow::GroupFilter,
    available_groups: &[String],
) {
    let buffer_id = ui.id().with("group_select_buffer");
    let mut groups_csv = ui
        .ctx()
        .data(|d| d.get_temp::<String>(buffer_id))
        .unwrap_or_else(|| groups.to_csv());
    let output = egui::TextEdit::singleline(&mut groups_csv)
        .hint_text("Type a group name")
        .desired_width(f32::INFINITY)
        .show(ui);
    let response = output.response.clone();
    if output.response.changed() {
        *groups = workflow::GroupFilter::from_csv(&groups_csv);
    }

    if available_groups.is_empty() {
        sync_group_select_buffer(ui.ctx(), buffer_id, &groups_csv, response.has_focus());
        ui.small("Autocomplete appears when the input streamline data exposes group names.");
        return;
    }

    ui.small(format!("{} groups available", available_groups.len()));

    let selected = parse_group_csv(&groups_csv);
    let current_fragment = current_group_fragment(&groups_csv);
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
        sync_group_select_buffer(ui.ctx(), buffer_id, &groups_csv, response.has_focus());
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
                        replace_group_fragment(&mut groups_csv, &suggestion);
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
        *groups = workflow::GroupFilter::from_csv(&groups_csv);
        sync_group_select_buffer(ui.ctx(), buffer_id, &groups_csv, true);
        ui.ctx()
            .data_mut(|d| d.insert_temp(suggestion_state_id, false));
        return;
    }

    let clicked_elsewhere = ui.input(|i| i.pointer.any_click()) && !response.hovered();
    let keep_open = response.has_focus() || (!clicked_elsewhere && has_current_fragment);
    ui.ctx()
        .data_mut(|d| d.insert_temp(suggestion_state_id, keep_open));
    sync_group_select_buffer(ui.ctx(), buffer_id, &groups_csv, response.has_focus());
}

fn sync_group_select_buffer(
    ctx: &egui::Context,
    buffer_id: egui::Id,
    groups_csv: &str,
    keep_buffer: bool,
) {
    if keep_buffer {
        ctx.data_mut(|d| d.insert_temp(buffer_id, groups_csv.to_owned()));
    } else {
        ctx.data_mut(|d| d.remove::<String>(buffer_id));
    }
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

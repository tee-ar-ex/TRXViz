use std::collections::BTreeSet;

use trxviz_core::data::loaded_files::VolumeColormap;
use trxviz_core::data::orientation_field::{
    BoundaryGlyphColorMode, BoundaryGlyphNormalization, DirectionFieldBinningMode,
};
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::renderer::mesh_renderer::SurfaceColormap;

use crate::app::workflow;

mod dipy_tractography;
mod yeh_tractography;

#[derive(Clone, Debug)]
pub(crate) struct OdxSelectorNames {
    pub(crate) dpv_names: Vec<String>,
    pub(crate) dpf_names: Vec<String>,
}

pub(crate) struct NodeEditorContext<'a> {
    pub(crate) available_groups: &'a [String],
    /// DPS / DPV field names present on this node's last evaluation
    /// (output dataset). The inspector uses these to populate
    /// comboboxes for `ColorByDps` / `ColorByDpv` rather than making
    /// the user type field names by hand.
    pub(crate) available_dps_fields: &'a [String],
    pub(crate) available_dpv_fields: &'a [String],
    pub(crate) odx_selector_names: Option<&'a OdxSelectorNames>,
    pub(crate) sh_detail_limit: Option<u32>,
    pub(crate) save_ready: bool,
    /// Names of this node's op params that a wired `TrackingPlan` is
    /// overriding on the most recent evaluation. Editor panels should
    /// disable the corresponding sliders so the user can see which values
    /// the tracker will actually use.
    pub(crate) overridden_fields: &'a [String],
    /// For each overridden numeric field, the effective value from the plan.
    /// Editor panels bind this to the greyed-out slider so the user sees
    /// the plan's number instead of the op's own.
    pub(crate) overridden_values: &'a std::collections::BTreeMap<String, f32>,
    /// Is a wgpu device available in this session? Passed to
    /// `WorkflowOp::validate` so GPU-only op variants (currently PTT)
    /// can surface an inline error when there's no GPU.
    pub(crate) gpu_available: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeEditorResult {
    pub(crate) save_now: bool,
    pub(crate) run_expensive_requested: bool,
}

pub(crate) fn edit_node_op(
    ui: &mut egui::Ui,
    node_uuid: workflow::WorkflowNodeUuid,
    op: &mut workflow::WorkflowNodeKind,
    ctx: NodeEditorContext<'_>,
) -> NodeEditorResult {
    let mut result = NodeEditorResult::default();

    // Pre-dispatch diagnostics (e.g. "PTT requires a GPU; none
    // available"). Rendered at the top of the inspector so the user
    // sees the problem before scrolling through the knobs. Errors are
    // advisory for now — the GUI still allows "Run" to be clicked,
    // and the worker returns a descriptive failure.
    let diagnostics = workflow::validate_op(
        op,
        &workflow::ValidateCtx {
            gpu_available: ctx.gpu_available,
        },
    );
    for diag in &diagnostics {
        let color = match diag.severity {
            workflow::DiagnosticSeverity::Error => egui::Color32::from_rgb(230, 100, 100),
            workflow::DiagnosticSeverity::Warning => egui::Color32::from_rgb(220, 180, 96),
        };
        let prefix = match diag.severity {
            workflow::DiagnosticSeverity::Error => "⛔",
            workflow::DiagnosticSeverity::Warning => "⚠",
        };
        ui.colored_label(color, format!("{prefix} {}", diag.message));
    }
    if !diagnostics.is_empty() {
        ui.separator();
    }

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
        workflow::WorkflowNodeKind::ColorByDPV { field, colormap } => {
            edit_picker_field(
                ui,
                field,
                ctx.available_dpv_fields,
                node_uuid,
                "color_by_dpv_field",
            );
            edit_colormap(ui, colormap, node_uuid, "color_by_dpv_colormap");
        }
        workflow::WorkflowNodeKind::ColorByDPS { field, colormap } => {
            edit_picker_field(
                ui,
                field,
                ctx.available_dps_fields,
                node_uuid,
                "color_by_dps_field",
            );
            edit_colormap(ui, colormap, node_uuid, "color_by_dps_colormap");
        }
        workflow::WorkflowNodeKind::UniformColor { color } => {
            ui.color_edit_button_rgba_unmultiplied(color);
        }
        workflow::WorkflowNodeKind::TipPrune {
            voxel_size_mm,
            iterations,
            min_support,
            max_unsupported_fraction,
        } => {
            ui.add(
                egui::Slider::new(voxel_size_mm, 0.25..=4.0)
                    .text("Voxel size (mm)")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(iterations, 1..=64).text("Iterations"));
            ui.add(
                egui::DragValue::new(min_support)
                    .range(0..=10)
                    .prefix("Min support "),
            );
            ui.add(
                egui::Slider::new(max_unsupported_fraction, 0.0..=1.0)
                    .text("Max unsupported fraction"),
            );
            ui.small("0.0 = strict DSI-Studio parity; 1.0 = passthrough");
        }
        workflow::WorkflowNodeKind::Purifibre {
            trim_fraction,
            puri_fraction,
            spherical_smoothing_deg,
        } => {
            ui.add(
                egui::Slider::new(trim_fraction, 0.0..=0.5)
                    .text("Trim fraction")
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                    .custom_parser(|s| {
                        s.trim_end_matches('%')
                            .parse::<f64>()
                            .ok()
                            .map(|v| v / 100.0)
                    }),
            );
            ui.add(
                egui::Slider::new(puri_fraction, 0.0..=0.9)
                    .text("Discard fraction")
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                    .custom_parser(|s| {
                        s.trim_end_matches('%')
                            .parse::<f64>()
                            .ok()
                            .map(|v| v / 100.0)
                    }),
            );
            ui.add(
                egui::Slider::new(spherical_smoothing_deg, 0.0..=45.0)
                    .text("Spherical smoothing (°)"),
            );
            ui.small(
                "Output 0 = input + FICO DPS; Output 1 = filtered survivors. \
                 Needs a BoundaryField upstream on input 1.",
            );
        }
        workflow::WorkflowNodeKind::SampleVolumeAlongStreamline { dps_name } => {
            ui.horizontal(|ui| {
                ui.label("DPS name");
                ui.text_edit_singleline(dps_name);
            });
            ui.small(
                "Trilinearly samples the input VolumeScalars at every \
                 streamline vertex, then attaches the per-streamline mean \
                 as a DPS field. Wire `Color By DPS` downstream.",
            );
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
            opacity,
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
            ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
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
        workflow::WorkflowNodeKind::VolumeOverlayStack { layers } => {
            ui.small(
                "Layer 0 sets the target grid; later layers are resampled per slice. \
                 Each layer is windowed, thresholded, colormapped and alpha-over composited \
                 atop the previous one.",
            );
            ui.separator();
            for (layer_index, layer) in layers.iter_mut().enumerate() {
                let is_base = layer_index == 0;
                let title = if is_base {
                    format!("Layer 0: Base — {}", layer.legend_label)
                } else {
                    format!("Layer {layer_index} — {}", layer.legend_label)
                };
                ui.collapsing(title, |ui| {
                    ui.checkbox(&mut layer.enabled, "Enabled");
                    ui.horizontal(|ui| {
                        ui.label("Legend");
                        ui.text_edit_singleline(&mut layer.legend_label);
                    });
                    egui::ComboBox::from_id_salt(format!(
                        "volume_overlay_colormap_{}_{}",
                        node_uuid.0, layer_index
                    ))
                    .selected_text(layer.colormap.label())
                    .show_ui(ui, |ui| {
                        for value in VolumeColormap::ALL {
                            ui.selectable_value(&mut layer.colormap, *value, value.label());
                        }
                    });
                    ui.add(egui::Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity"));
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut layer.window_center)
                                .speed(0.01)
                                .prefix("Window center "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut layer.window_width)
                                .speed(0.01)
                                .prefix("Window width "),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut layer.threshold_min)
                                .speed(0.01)
                                .prefix("Thresh min "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut layer.threshold_max)
                                .speed(0.01)
                                .prefix("Thresh max "),
                        );
                    });
                    if !is_base {
                        egui::ComboBox::from_id_salt(format!(
                            "volume_overlay_interp_{}_{}",
                            node_uuid.0, layer_index
                        ))
                        .selected_text(match layer.interpolation {
                            trxviz_core::workflow::Interp::Trilinear => "Trilinear",
                            trxviz_core::workflow::Interp::Nearest => "Nearest",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut layer.interpolation,
                                trxviz_core::workflow::Interp::Trilinear,
                                "Trilinear",
                            );
                            ui.selectable_value(
                                &mut layer.interpolation,
                                trxviz_core::workflow::Interp::Nearest,
                                "Nearest",
                            );
                        });
                    }
                });
            }
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
        workflow::WorkflowNodeKind::StreamlineDirectionField {
            voxel_size_mm,
            sphere_lod,
            normalization,
            binning_mode,
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
            egui::ComboBox::from_id_salt(format!("direction_field_binning_{}", node_uuid.0))
                .selected_text(binning_mode.label())
                .show_ui(ui, |ui| {
                    for value in DirectionFieldBinningMode::ALL {
                        ui.selectable_value(binning_mode, value, value.label());
                    }
                });
            egui::ComboBox::from_id_salt(format!("direction_field_normalization_{}", node_uuid.0))
                .selected_text(normalization.label())
                .show_ui(ui, |ui| {
                    for value in BoundaryGlyphNormalization::ALL {
                        ui.selectable_value(normalization, value, value.label());
                    }
                });
            ui.small(
                "Per-voxel histogram of streamline tangent directions \
                 (sTODI). Consumed by Boundary Glyph and Purifibre.",
            );
            ui.small(match binning_mode {
                DirectionFieldBinningMode::WithinVoxelTangent => {
                    "Within-voxel tangent: symmetric, length-weighted. \
                     Recommended for Purifibre."
                }
                DirectionFieldBinningMode::BoundaryCrossings => {
                    "Boundary crossings: asymmetric, count-weighted. \
                     Original boundary-glyph behavior."
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
            auto_gate_from_otsu,
            opacity_gate,
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
            fixel_opacity_gate_editor(ui, auto_gate_from_otsu, opacity_gate);
        }
        workflow::WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
            auto_gate_from_otsu,
            opacity_gate,
        } => {
            ui.checkbox(visible, "Visible");
            ui.add(egui::Slider::new(line_width, 0.001..=0.05).text("Line width"));
            ui.add(egui::Slider::new(length_scale, 0.1..=5.0).text("Length scale"));
            ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            ui.add(
                egui::Slider::new(&mut slab_thickness_mm.0, 0.1..=20.0).text("Slab thickness (mm)"),
            );
            fixel_opacity_gate_editor(ui, auto_gate_from_otsu, opacity_gate);
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
        workflow::WorkflowNodeKind::RoiFromParcel { labels } => {
            ui.label("Parcel labels (comma-separated IDs):");
            edit_parcel_id_set(ui, labels);
        }
        workflow::WorkflowNodeKind::RoiFromVolume { threshold } => {
            ui.add(egui::Slider::new(threshold, 0.0..=1.0).text("Threshold"));
        }
        workflow::WorkflowNodeKind::RoiFromShape {
            center_ras,
            radius_or_half_extent_mm,
            shape,
        } => {
            ui.label("Center (RAS+ mm)");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut center_ras[0])
                        .speed(0.5)
                        .prefix("X "),
                );
                ui.add(
                    egui::DragValue::new(&mut center_ras[1])
                        .speed(0.5)
                        .prefix("Y "),
                );
                ui.add(
                    egui::DragValue::new(&mut center_ras[2])
                        .speed(0.5)
                        .prefix("Z "),
                );
            });
            ui.add(
                egui::DragValue::new(&mut radius_or_half_extent_mm.0)
                    .speed(0.5)
                    .prefix("Radius/half-extent "),
            );
            egui::ComboBox::from_id_salt(("roi_shape", node_uuid))
                .selected_text(match shape {
                    workflow::RoiShape::Sphere => "Sphere",
                    workflow::RoiShape::Box => "Box",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(shape, workflow::RoiShape::Sphere, "Sphere");
                    ui.selectable_value(shape, workflow::RoiShape::Box, "Box");
                });
        }
        workflow::WorkflowNodeKind::DipyTractography {
            step_size_mm,
            max_angle_deg,
            min_len_mm,
            max_len_mm,
            fixel_threshold,
            relative_peak_threshold,
            seeds_per_voxel,
            max_points,
            rng_seed,
            direction_getter,
        } => {
            dipy_tractography::draw(
                ui,
                node_uuid,
                step_size_mm,
                max_angle_deg,
                min_len_mm,
                max_len_mm,
                fixel_threshold,
                relative_peak_threshold,
                seeds_per_voxel,
                max_points,
                rng_seed,
                direction_getter,
                &ctx,
                &mut result,
            );
        }
        workflow::WorkflowNodeKind::YehTractography {
            step_size_mm,
            max_angle_deg,
            min_len_mm,
            max_len_mm,
            fixel_threshold,
            smooth_fraction,
            max_points,
            target_streamlines,
            max_seed_attempts,
            rng_seed,
        } => {
            yeh_tractography::draw(
                ui,
                step_size_mm,
                max_angle_deg,
                min_len_mm,
                max_len_mm,
                fixel_threshold,
                smooth_fraction,
                max_points,
                target_streamlines,
                max_seed_attempts,
                rng_seed,
                &ctx,
                &mut result,
            );
        }
        workflow::WorkflowNodeKind::VoxelMaskDisplay {
            color,
            opacity,
            smooth_sigma,
            min_component_volume_mm3,
            style,
            slice_mode,
        } => {
            use trxviz_core::workflow::{VoxelMaskRenderStyle, VoxelMaskSliceMode};
            ui.horizontal(|ui| {
                ui.label("Color");
                let mut rgb = [color[0], color[1], color[2]];
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    color[0] = rgb[0];
                    color[1] = rgb[1];
                    color[2] = rgb[2];
                }
            });
            ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            ui.horizontal(|ui| {
                ui.label("Style");
                ui.selectable_value(style, VoxelMaskRenderStyle::VoxelAccurate, "Voxel-accurate");
                ui.selectable_value(style, VoxelMaskRenderStyle::SmoothMesh, "Smooth mesh");
            });
            if matches!(style, VoxelMaskRenderStyle::VoxelAccurate) {
                ui.horizontal(|ui| {
                    ui.label("Slice fill");
                    ui.selectable_value(slice_mode, VoxelMaskSliceMode::Filled, "Filled");
                    ui.selectable_value(slice_mode, VoxelMaskSliceMode::Outline, "Outline");
                });
            } else {
                ui.add(egui::Slider::new(smooth_sigma, 0.0..=3.0).text("Smooth σ (voxels)"));
                ui.add(
                    egui::Slider::new(&mut min_component_volume_mm3.0, 0.0..=1000.0)
                        .text("Min component vol (mm³)"),
                );
            }
        }
        workflow::WorkflowNodeKind::PrepareSimplePlan {
            override_step,
            step_size_mm,
            override_angle,
            max_angle_deg,
            override_min_len,
            min_len_mm,
            override_max_len,
            max_len_mm,
            override_fixel_threshold,
            fixel_threshold,
            override_smooth,
            smooth_fraction,
            override_fixel_otsu,
            fixel_otsu,
        } => {
            ui.small("Each override, when enabled, replaces the tracker's slider value.");
            let row = |ui: &mut egui::Ui,
                       enabled: &mut bool,
                       value: &mut f32,
                       range: std::ops::RangeInclusive<f32>,
                       label: &str| {
                ui.horizontal(|ui| {
                    ui.checkbox(enabled, "");
                    ui.add_enabled(*enabled, egui::Slider::new(value, range).text(label));
                });
            };
            row(
                ui,
                override_step,
                step_size_mm,
                0.25..=2.0,
                "Step size (mm)",
            );
            row(
                ui,
                override_angle,
                max_angle_deg,
                30.0..=90.0,
                "Max angle (°)",
            );
            row(
                ui,
                override_min_len,
                min_len_mm,
                5.0..=100.0,
                "Min length (mm)",
            );
            row(
                ui,
                override_max_len,
                max_len_mm,
                20.0..=500.0,
                "Max length (mm)",
            );
            row(
                ui,
                override_fixel_threshold,
                fixel_threshold,
                0.0..=0.5,
                "Fixel threshold",
            );
            row(
                ui,
                override_smooth,
                smooth_fraction,
                0.0..=0.95,
                "Smoothing",
            );
            row(ui, override_fixel_otsu, fixel_otsu, 0.0..=1.0, "Fixel Otsu");
        }
        workflow::WorkflowNodeKind::PrepareHausdorffPlan {
            tolerance_mm,
            seed_tolerance_mm,
            tracking_metric,
            otsu_scope,
            seed_fixel_otsu_factor,
            not_end_fixel_otsu_factor,
            max_reference_points,
        } => {
            use trxviz_core::data::odx_data::OtsuScope;
            ui.add(egui::Slider::new(tolerance_mm, 0.5..=20.0).text("Tolerance (mm)"));
            ui.small(
                "DSI-Studio tolerance: limiting-mask dilation, post-filter distance, \
                and ±2·tol on min/max length.",
            );
            ui.add(
                egui::Slider::new(seed_tolerance_mm, 0.0..=*tolerance_mm)
                    .text("Seed tolerance (mm)"),
            );
            ui.small("Small seed tolerance keeps seeds near the reference bundle.");
            ui.horizontal(|ui| {
                ui.label("Metric");
                let current = tracking_metric.clone().unwrap_or_else(|| "auto".into());
                egui::ComboBox::from_id_salt(("hausdorff_metric", node_uuid))
                    .selected_text(&current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(tracking_metric, None, "auto");
                        if let Some(names) = ctx.odx_selector_names {
                            for name in &names.dpf_names {
                                ui.selectable_value(tracking_metric, Some(name.clone()), name);
                            }
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Otsu scope");
                ui.selectable_value(otsu_scope, OtsuScope::AllFixels, "All fixels");
                ui.selectable_value(otsu_scope, OtsuScope::PrimaryPeak, "Primary peak");
            });
            ui.add(egui::Slider::new(seed_fixel_otsu_factor, 0.0..=2.0).text("Seed factor × Otsu"));
            ui.add(
                egui::Slider::new(not_end_fixel_otsu_factor, 0.0..=2.0)
                    .text("No-end factor × Otsu"),
            );
            ui.add(
                egui::Slider::new(max_reference_points, 1_000..=50_000)
                    .text("Max reference points"),
            );
        }
        workflow::WorkflowNodeKind::PreparePyafqPlan {
            working_dir,
            bundle_name,
            to_space,
            dist_to_waypoint_mm,
            dist_to_exclusion_mm,
            dist_to_endpoint_mm,
            override_min_len_mm,
            override_max_len_mm,
        } => {
            use trxviz_core::workflow::pyafq_bundles::{PYAFQ_BUNDLES, PyafqCategory};

            // Working directory picker.
            ui.horizontal(|ui| {
                ui.label("pyAFQ derivatives dir");
                if ui.button("📂 Browse…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        *working_dir = dir.to_string_lossy().into_owned();
                    }
                }
            });
            if working_dir.is_empty() {
                ui.small("(no directory selected)");
            } else {
                ui.small(working_dir.as_str());
            }

            // Auto-detect `to_space` from the working dir when blank. Saves
            // the user from having to know which space token (T1w, subject,
            // …) the dataset uses.
            let mut auto_filled_space: Option<String> = None;
            let mut auto_detect_failed = false;
            if to_space.is_empty() && !working_dir.is_empty() {
                match trxviz_core::gpu::plan_prep::pyafq::auto_pick_to_space(
                    std::path::Path::new(working_dir.as_str()),
                ) {
                    Some(picked) => {
                        *to_space = picked.clone();
                        auto_filled_space = Some(picked);
                    }
                    None => {
                        auto_detect_failed = true;
                    }
                }
            }

            // Bundle dropdown, grouped by category. Bundles whose ROI files
            // aren't present in the chosen working dir (under the chosen
            // `to_space`) are grayed out so the user can see at a glance
            // what this dataset actually contains.
            let available: std::collections::HashSet<&'static str> = if working_dir.is_empty() {
                PYAFQ_BUNDLES.iter().map(|s| s.display_name).collect()
            } else {
                trxviz_core::gpu::plan_prep::pyafq::scan_available_bundles(
                    std::path::Path::new(working_dir.as_str()),
                    to_space,
                )
            };
            ui.horizontal(|ui| {
                ui.label("Bundle");
                let label = if bundle_name.is_empty() {
                    "(pick a bundle)"
                } else {
                    bundle_name.as_str()
                };
                egui::ComboBox::from_id_salt(("pyafq_bundle", node_uuid))
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.label("Default");
                        for spec in PYAFQ_BUNDLES
                            .iter()
                            .filter(|s| s.category == PyafqCategory::Default)
                        {
                            let enabled = available.contains(spec.display_name);
                            ui.add_enabled_ui(enabled, |ui| {
                                ui.selectable_value(
                                    bundle_name,
                                    spec.display_name.to_string(),
                                    spec.display_name,
                                );
                            });
                        }
                        ui.separator();
                        ui.label("Callosal");
                        for spec in PYAFQ_BUNDLES
                            .iter()
                            .filter(|s| s.category == PyafqCategory::Callosal)
                        {
                            let enabled = available.contains(spec.display_name);
                            ui.add_enabled_ui(enabled, |ui| {
                                ui.selectable_value(
                                    bundle_name,
                                    spec.display_name.to_string(),
                                    spec.display_name,
                                );
                            });
                        }
                        ui.separator();
                        ui.label("Pediatric");
                        for spec in PYAFQ_BUNDLES
                            .iter()
                            .filter(|s| s.category == PyafqCategory::Pediatric)
                        {
                            let enabled = available.contains(spec.display_name);
                            ui.add_enabled_ui(enabled, |ui| {
                                ui.selectable_value(
                                    bundle_name,
                                    spec.display_name.to_string(),
                                    spec.display_name,
                                );
                            });
                        }
                    });
            });

            // Space token (rarely changed; keep as a simple text field).
            // Auto-detection above fills it in when blank; show a caption
            // so the user knows where the value came from.
            ui.horizontal(|ui| {
                ui.label("Space");
                ui.text_edit_singleline(to_space);
                if let Some(picked) = &auto_filled_space {
                    ui.small(format!("(auto-detected: {picked})"));
                } else if auto_detect_failed {
                    ui.small("(detect failed — type a space token)");
                }
            });

            // Distance tolerances.
            ui.add(
                egui::Slider::new(dist_to_waypoint_mm, 0.0..=10.0).text("Waypoint tolerance (mm)"),
            );
            ui.add(
                egui::Slider::new(dist_to_exclusion_mm, 0.0..=5.0).text("Exclusion tolerance (mm)"),
            );
            ui.add(
                egui::Slider::new(dist_to_endpoint_mm, 0.0..=10.0).text("Endpoint tolerance (mm)"),
            );

            // Length overrides.
            let bundle_spec = trxviz_core::workflow::pyafq_bundles::lookup(bundle_name);
            ui.horizontal(|ui| {
                let mut on = override_min_len_mm.is_some();
                let default_min = bundle_spec.and_then(|s| s.min_len_mm);
                if ui.checkbox(&mut on, "Override min len").changed() {
                    *override_min_len_mm = if on { default_min.or(Some(20.0)) } else { None };
                }
                if let Some(v) = override_min_len_mm.as_mut() {
                    ui.add(egui::DragValue::new(v).range(0.0..=300.0).suffix(" mm"));
                } else if let Some(d) = default_min {
                    ui.small(format!("(bundle default: {d:.0} mm)"));
                } else {
                    ui.small("(bundle default: tracker)");
                }
            });
            ui.horizontal(|ui| {
                let mut on = override_max_len_mm.is_some();
                let default_max = bundle_spec.and_then(|s| s.max_len_mm);
                if ui.checkbox(&mut on, "Override max len").changed() {
                    *override_max_len_mm = if on {
                        default_max.or(Some(250.0))
                    } else {
                        None
                    };
                }
                if let Some(v) = override_max_len_mm.as_mut() {
                    ui.add(egui::DragValue::new(v).range(0.0..=500.0).suffix(" mm"));
                } else if let Some(d) = default_max {
                    ui.small(format!("(bundle default: {d:.0} mm)"));
                } else {
                    ui.small("(bundle default: tracker)");
                }
            });
        }
        _ => {
            ui.small("This node has no editable parameters yet.");
        }
    }

    result
}

/// Shared edit widget for `auto_gate_from_otsu` + `opacity_gate` on fixel
/// display ops. When auto is on, the gate is hidden (the scene's
/// `default_fixel_otsu()` drives it at eval time); when off, expose the
/// four gate parameters explicitly.
fn fixel_opacity_gate_editor(ui: &mut egui::Ui, auto: &mut bool, gate: &mut workflow::OpacityGate) {
    ui.separator();
    ui.checkbox(auto, "Auto-gate from tracking Otsu");
    if !*auto {
        ui.add(egui::Slider::new(&mut gate.range.0, 0.0..=1.0).text("Gate range min"));
        ui.add(egui::Slider::new(&mut gate.range.1, 0.0..=1.0).text("Gate range max"));
        ui.add(egui::Slider::new(&mut gate.below, 0.0..=1.0).text("Alpha below"));
        ui.add(egui::Slider::new(&mut gate.above, 0.0..=1.0).text("Alpha above"));
    } else {
        ui.small("Sub-threshold fixels ghost to 10 % alpha.");
    }
}

#[allow(dead_code)]
fn edit_field_name<T>(ui: &mut egui::Ui, field: &mut T)
where
    T: From<String> + AsRef<str>,
{
    let mut value = field.as_ref().to_string();
    if ui.text_edit_singleline(&mut value).changed() {
        *field = T::from(value);
    }
}

/// DPS / DPV field picker. Renders a combobox of names available on
/// the upstream's last evaluation; falls back to a free-text input
/// when no names are known yet (upstream hasn't been built, or the
/// dataset has no scalar fields). The free-text fallback also kicks
/// in when the current field name isn't in the available list — the
/// user keeps the old value visible and can either pick a known name
/// from the dropdown or edit the text directly.
fn edit_picker_field<T>(
    ui: &mut egui::Ui,
    field: &mut T,
    available: &[String],
    node_uuid: workflow::WorkflowNodeUuid,
    salt: &str,
) where
    T: From<String> + AsRef<str>,
{
    let current = field.as_ref().to_string();

    if available.is_empty() {
        // No upstream fields known — text input with a hint so the
        // user understands why there's no dropdown.
        let mut value = current;
        if ui.text_edit_singleline(&mut value).changed() {
            *field = T::from(value);
        }
        ui.small("(no fields advertised by upstream — type one manually)");
        return;
    }

    let combo_id = format!("{salt}_{}", node_uuid.0);
    egui::ComboBox::from_id_salt(combo_id)
        .selected_text(if current.is_empty() {
            "(pick a field)"
        } else {
            current.as_str()
        })
        .show_ui(ui, |ui| {
            for name in available {
                if ui
                    .selectable_label(name.as_str() == current.as_str(), name.as_str())
                    .clicked()
                {
                    *field = T::from(name.clone());
                }
            }
        });

    // If the current field name isn't in the available list (upstream
    // changed under us, or the user typed it before the field
    // existed), surface that explicitly so the user isn't confused
    // when the combobox shows e.g. "fico" while the rendering shows
    // gray.
    if !current.is_empty() && !available.iter().any(|n| n == &current) {
        ui.small(format!(
            "⚠ \"{current}\" is not in the upstream's field list"
        ));
    }
}

/// Combobox for picking a `SurfaceColormap` (the scalar colormap
/// used by `ColorByDps` / `ColorByDpv`).
fn edit_colormap(
    ui: &mut egui::Ui,
    colormap: &mut SurfaceColormap,
    node_uuid: workflow::WorkflowNodeUuid,
    salt: &str,
) {
    let combo_id = format!("{salt}_{}", node_uuid.0);
    egui::ComboBox::from_id_salt(combo_id)
        .selected_text(colormap.label())
        .show_ui(ui, |ui| {
            for value in SurfaceColormap::ALL {
                ui.selectable_value(colormap, value, value.label());
            }
        });
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

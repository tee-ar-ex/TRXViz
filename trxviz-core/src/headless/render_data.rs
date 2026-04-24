use glam::{Mat4, Vec3};

use super::gpu_context::{active_fixel_draw_2d, active_fixel_draw_3d};
use super::{
    BoundaryGlyphColorMode, BundleDrawInfo, HeadlessRenderData, HeadlessRenderOptions,
    HeadlessView, SceneBounds, StreamlineDrawInfo, VolumeDrawInfo,
};
use crate::data::cifti::CiftiStructure;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::mesh_renderer::MeshDrawStyle;
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::units::Millimeters;
use crate::workflow::WorkflowCamera3D;

pub(super) fn build_render_data(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
    view: HeadlessView,
) -> HeadlessRenderData {
    let surface_draws = match view {
        HeadlessView::InflatedStage => stage_surface_draw_instances(scene, workflow),
        _ => workflow
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
            .collect(),
    };
    let volume_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
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
            .collect::<Vec<_>>()
    };
    let streamline_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
            .runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| StreamlineDrawInfo {
                file_id: draw.draw_id,
                visible: draw.visible,
                render_style: draw.render_style,
                tube_radius: draw.tube_radius_mm.0,
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>()
    };
    let bundle_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        let mut out = workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| BundleDrawInfo {
                file_id: draw.draw_id,
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>();
        // Voxel-mask meshes reuse the bundle-surface renderer via their own
        // draw_ids; they share the opaque pass.
        out.extend(
            workflow
                .runtime
                .scene_plan
                .voxel_mask_mesh_draws
                .iter()
                .map(|draw| BundleDrawInfo {
                    file_id: draw.draw_id,
                    opacity: draw.opacity,
                }),
        );
        out
    };
    let glyph_draw = if view == HeadlessView::InflatedStage {
        None
    } else {
        workflow
            .runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible)
    };
    let odx_has_glyph_field = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|p| p.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
        .map(|plan| plan.field.scene.has_glyph_field())
        .or_else(|| scene.odx_scene.as_ref().map(|odx| odx.has_glyph_field()))
        .unwrap_or(false);
    let odx_glyphs_active = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|p| p.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
        .map(|p| p.visible)
        .unwrap_or(scene.odx_scene.is_some());
    let fixel_3d_draw = active_fixel_draw_3d(workflow);
    let fixel_2d_draw = active_fixel_draw_2d(workflow);

    HeadlessRenderData {
        any_visible_streamlines: streamline_draws.iter().any(|draw| draw.visible),
        surface_draws,
        volume_draws,
        streamline_draws,
        bundle_draws,
        glyph_visible: glyph_draw.is_some()
            && !workflow.execution_cache.boundary_field_cache.is_empty(),
        glyph_color_mode: glyph_draw
            .map(|draw| draw.color_mode)
            .unwrap_or(BoundaryGlyphColorMode::DirectionRgb),
        glyph_density_3d_step: glyph_draw
            .map(|draw| draw.density_3d_step as u32)
            .unwrap_or(1),
        glyph_slice_density_step: glyph_draw
            .map(|draw| draw.slice_density_step as u32)
            .unwrap_or(1),
        odx_visible: scene.odx_scene.is_some()
            || !workflow.runtime.scene_plan.odf_glyph_draws.is_empty()
            || !workflow.runtime.scene_plan.fixel_3d_draws.is_empty()
            || !workflow.runtime.scene_plan.fixel_2d_draws.is_empty(),
        odx_fixel_3d_visible: fixel_3d_draw.map(|p| p.visible).unwrap_or(true)
            && !(odx_has_glyph_field && odx_glyphs_active),
        odx_fixel_2d_visible: fixel_2d_draw
            .map(|p| p.visible)
            .unwrap_or(scene.odx_scene.is_some()),
        fixel_3d_line_width: fixel_3d_draw.map(|p| p.line_width).unwrap_or(0.006),
        fixel_3d_opacity: fixel_3d_draw.map(|p| p.opacity).unwrap_or(1.0),
        fixel_3d_colormap_code: fixel_3d_draw.map(|p| p.colormap_code).unwrap_or(0),
        fixel_3d_scalar_range: fixel_3d_draw
            .map(|p| [p.scalar_range.0, p.scalar_range.1])
            .unwrap_or([0.0, 1.0]),
        fixel_3d_opacity_gate: fixel_3d_draw
            .map(|p| {
                [
                    p.opacity_gate.range.0,
                    p.opacity_gate.range.1,
                    p.opacity_gate.below,
                    p.opacity_gate.above,
                ]
            })
            .unwrap_or([0.0, 0.0, 1.0, 1.0]),
        fixel_2d_line_width: fixel_2d_draw.map(|p| p.line_width).unwrap_or(0.006),
        fixel_2d_slab_half_width_mm: fixel_2d_draw
            .map(|p| (p.slab_thickness_mm * 0.5).max(Millimeters(0.0)))
            .unwrap_or(Millimeters(1.0)),
        fixel_2d_opacity: fixel_2d_draw.map(|p| p.opacity).unwrap_or(1.0),
        fixel_2d_colormap_code: fixel_2d_draw.map(|p| p.colormap_code).unwrap_or(0),
        fixel_2d_scalar_range: fixel_2d_draw
            .map(|p| [p.scalar_range.0, p.scalar_range.1])
            .unwrap_or([0.0, 1.0]),
        fixel_2d_opacity_gate: fixel_2d_draw
            .map(|p| {
                [
                    p.opacity_gate.range.0,
                    p.opacity_gate.range.1,
                    p.opacity_gate.below,
                    p.opacity_gate.above,
                ]
            })
            .unwrap_or([0.0, 0.0, 1.0, 1.0]),
        odf_glyph_opacity: workflow
            .runtime
            .scene_plan
            .odf_glyph_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
            .map(|p| p.opacity)
            .unwrap_or(1.0),
        odf_glyph_gloss: workflow
            .runtime
            .scene_plan
            .odf_glyph_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
            .map(|p| p.gloss)
            .unwrap_or(0.0),
    }
}

pub(super) fn compute_render_bounds(
    scene: &HeadlessScene,
    render_data: &HeadlessRenderData,
) -> SceneBounds {
    let mut bounds_min = Vec3::splat(f32::INFINITY);
    let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;

    for (surface_id, _, style) in &render_data.surface_draws {
        let Some(surface) = scene
            .gifti_surfaces
            .iter()
            .find(|surface| surface.id == *surface_id)
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
        return SceneBounds {
            min: scene.volume_center - Vec3::splat(scene.volume_extent.max(1.0) * 0.5),
            max: scene.volume_center + Vec3::splat(scene.volume_extent.max(1.0) * 0.5),
        };
    }

    SceneBounds {
        min: bounds_min,
        max: bounds_max,
    }
}

pub(super) fn build_camera(
    bounds: &SceneBounds,
    saved_camera: Option<WorkflowCamera3D>,
    options: &HeadlessRenderOptions,
    aspect: f32,
) -> OrbitCamera {
    let saved_target = saved_camera.map(|camera| Vec3::from_array(camera.target));
    let center = options
        .target
        .or(saved_target)
        .unwrap_or((bounds.min + bounds.max) * 0.5);
    let radius = ((bounds.max - bounds.min) * 0.5).length().max(1.0);
    let mut camera = OrbitCamera::new(center, fit_distance(radius, aspect));
    camera.yaw = options
        .azimuth_deg
        .or(saved_camera.map(|camera| camera.azimuth_deg))
        .unwrap_or(45.0)
        .to_radians();
    camera.pitch = options
        .elevation_deg
        .or(saved_camera.map(|camera| camera.elevation_deg))
        .unwrap_or(25.0)
        .to_radians();
    camera.distance = options
        .distance
        .or(saved_camera.map(|camera| camera.distance))
        .unwrap_or(camera.distance)
        .max(0.1);
    camera
}

pub(super) fn stage_instance_model_matrices(
    structure: Option<CiftiStructure>,
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> Vec<Mat4> {
    let center = (bbox_min + bbox_max) * 0.5;
    let extents = bbox_max - bbox_min;
    let span = extents
        .x
        .abs()
        .max(extents.y.abs())
        .max(extents.z.abs())
        .max(1.0);
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

fn stage_surface_draw_instances(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
) -> Vec<(usize, usize, MeshDrawStyle)> {
    let mut draws = Vec::new();
    for draw in &workflow.runtime.scene_plan.stage_surface_draws {
        let Some(surface) = scene
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
            draws.push((
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
    draws
}

fn fit_distance(radius: f32, aspect: f32) -> f32 {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let half_y = fov_y * 0.5;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let limiting_half_angle = half_y.min(half_x).max(0.1);
    (radius / limiting_half_angle.sin()) * 1.1
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

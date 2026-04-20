use std::path::Path;

use glam::{Mat4, Vec3};

use super::gpu_context::{
    GpuSceneResources, TARGET_FORMAT, active_fixel_draw_2d, active_fixel_draw_3d,
};
use super::readback::readback_texture_to_png;
use super::{
    BoundaryGlyphColorMode, BundleDrawInfo, HeadlessRenderData, HeadlessRenderOptions,
    HeadlessView, SceneBounds, StreamlineDrawInfo, VolumeDrawInfo,
};
use crate::data::cifti::CiftiStructure;
use crate::data::trx_data::RenderStyle;
use crate::lighting::WorkflowRender3D;
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
            })
            .collect::<Vec<_>>()
    };
    let bundle_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| BundleDrawInfo {
                file_id: draw.draw_id,
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>()
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
        fixel_2d_line_width: fixel_2d_draw.map(|p| p.line_width).unwrap_or(0.006),
        fixel_2d_slab_half_width_mm: fixel_2d_draw
            .map(|p| (p.slab_thickness_mm * 0.5).max(Millimeters(0.0)))
            .unwrap_or(Millimeters(1.0)),
        fixel_2d_opacity: fixel_2d_draw.map(|p| p.opacity).unwrap_or(1.0),
        fixel_2d_colormap_code: fixel_2d_draw.map(|p| p.colormap_code).unwrap_or(0),
        fixel_2d_scalar_range: fixel_2d_draw
            .map(|p| [p.scalar_range.0, p.scalar_range.1])
            .unwrap_or([0.0, 1.0]),
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

#[cfg(feature = "png-export")]
pub(super) fn render_scene3d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
    render_3d: &WorkflowRender3D,
    slice_visible: [bool; 3],
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let aspect = width as f32 / height.max(1) as f32;
    let view_proj = camera.view_projection(aspect);
    let camera_pos = camera.eye();
    let camera_dir = camera.view_direction();
    let lighting = render_3d.scene_lighting();
    let bounds_radius = ((resources.bounds.max - resources.bounds.min) * 0.5)
        .length()
        .max(1.0);
    let fog_span = (camera.distance + bounds_radius).max(1.0);
    let fog_near = fog_span * render_3d.fog_start_fraction;
    let fog_far = fog_span * render_3d.fog_end_fraction;
    resources.background.update(
        queue,
        &render_3d.background,
        render_3d.exposure,
        render_3d.contrast,
        render_3d.vignette_strength,
    );

    for volume in &render_data.volume_draws {
        if let Some((_, slice)) = resources
            .slices
            .entries
            .iter()
            .find(|(id, _)| *id == volume.file_id)
        {
            slice.update_uniforms(
                queue,
                0,
                view_proj,
                volume.window_center,
                volume.window_width,
                volume.colormap,
                volume.opacity,
            );
        }
    }
    for streamline in &render_data.streamline_draws {
        if !streamline.visible {
            continue;
        }
        if let Some((_, resource)) = resources
            .streamlines
            .entries
            .iter()
            .find(|(id, _)| *id == streamline.file_id)
        {
            let aux = if streamline.render_style == RenderStyle::DepthCue {
                300.0
            } else {
                streamline.tube_radius
            };
            resource.update_uniforms(
                queue,
                0,
                view_proj,
                camera_pos,
                streamline.render_style as u32,
                glam::Vec3::Z,
                glam::Vec3::ZERO,
                0.0,
                aux,
                lighting,
                render_3d,
                fog_near,
                fog_far,
            );
        }
    }
    for (surface_id, uniform_slot, style) in &render_data.surface_draws {
        resources.meshes.update_surface_uniforms(
            queue,
            *surface_id,
            *uniform_slot,
            view_proj,
            style,
            camera_pos,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    for bundle in &render_data.bundle_draws {
        resources.meshes.update_bundle_uniforms(
            bundle.file_id,
            queue,
            view_proj,
            camera_pos,
            bundle.opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.glyph_visible || render_data.odx_visible {
        resources.glyphs.update_uniforms(
            queue,
            0,
            view_proj,
            camera_pos,
            glam::Vec3::Z,
            glam::Vec3::ZERO,
            0.0,
            render_data.glyph_color_mode,
            render_data.glyph_density_3d_step,
            render_data.odf_glyph_opacity,
            render_data.odf_glyph_gloss,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.odx_visible && render_data.odx_fixel_3d_visible {
        resources.fixels_3d.update_uniforms(
            queue,
            0,
            view_proj,
            camera_pos,
            glam::Vec3::Z,
            glam::Vec3::ZERO,
            0.0,
            1,
            render_data.fixel_3d_line_width,
            render_data.fixel_3d_opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
        resources.fixels_3d.update_colormap(
            queue,
            0,
            render_data.fixel_3d_colormap_code,
            (
                render_data.fixel_3d_scalar_range[0],
                render_data.fixel_3d_scalar_range[1],
            ),
        );
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trxviz_headless_encoder"),
    });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: render_3d.background.bottom_color()[0] as f64,
                        g: render_3d.background.bottom_color()[1] as f64,
                        b: render_3d.background.bottom_color()[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let render_pass: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut render_pass) };

        render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        resources.background.paint(render_pass);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources
                .slices
                .entries
                .iter()
                .find(|(id, _)| *id == volume.file_id)
            {
                render_pass.set_pipeline(&slice.pipeline);
                render_pass.set_bind_group(0, &slice.bind_groups[0], &[]);
                render_pass
                    .set_index_buffer(slice.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                for i in 0..3 {
                    if !slice_visible[i] {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, slice.quad_buffers[i].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }

        if render_data.any_visible_streamlines {
            for streamline in &render_data.streamline_draws {
                if !streamline.visible {
                    continue;
                }
                if let Some((_, resource)) = resources
                    .streamlines
                    .entries
                    .iter()
                    .find(|(id, _)| *id == streamline.file_id)
                {
                    render_pass.set_bind_group(0, &resource.bind_groups[0], &[]);
                    if streamline.render_style == RenderStyle::Tubes {
                        if let (Some(vertices), Some(indices)) =
                            (&resource.tube_vertex_buffer, &resource.tube_index_buffer)
                        {
                            render_pass.set_pipeline(&resource.tube_pipeline);
                            render_pass.set_vertex_buffer(0, vertices.slice(..));
                            render_pass
                                .set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..resource.num_tube_indices, 0, 0..1);
                        }
                    } else {
                        render_pass.set_pipeline(&resource.pipeline);
                        render_pass.set_vertex_buffer(0, resource.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, resource.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, resource.tangent_buffer.slice(..));
                        render_pass.set_index_buffer(
                            resource.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..resource.num_indices, 0, 0..1);
                    }
                }
            }
        }

        if !render_data.surface_draws.is_empty() {
            resources
                .meshes
                .paint_opaque(render_pass, &render_data.surface_draws);
        }
        if !render_data.bundle_draws.is_empty() {
            let bundle_draws = render_data
                .bundle_draws
                .iter()
                .map(|draw| (draw.file_id, draw.opacity))
                .collect::<Vec<_>>();
            resources
                .meshes
                .paint_bundle_opaque(render_pass, &bundle_draws);
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &bundle_draws,
                camera_pos,
                camera_dir,
            );
        } else if !render_data.surface_draws.is_empty() {
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &[],
                camera_pos,
                camera_dir,
            );
        }
        if render_data.glyph_visible || render_data.odx_visible {
            resources.glyphs.paint(render_pass, 0, false);
        }
        if render_data.odx_visible && render_data.odx_fixel_3d_visible {
            resources.fixels_3d.paint(render_pass, 0, false);
        }
    }

    readback_texture_to_png(device, queue, encoder, &texture, width, height, output_path)
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

fn stage_panel_transform(center_transform: Mat4, x_shift: f32, z_shift: f32, turn_deg: f32) -> Mat4 {
    Mat4::from_translation(Vec3::new(x_shift, 0.0, z_shift))
        * Mat4::from_rotation_z(turn_deg.to_radians())
        * center_transform
}

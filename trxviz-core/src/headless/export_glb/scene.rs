use anyhow::Context;
use glam::{Mat4, Vec3};
use serde_json::{Map, Value, json};

use super::bundle_mesh::add_bundle_mesh_to_glb;
use super::slice_plane::add_slice_plane_to_glb;
use super::{GlbBuilder, add_lighting_rig_to_glb, gltf_point, gltf_transform, gltf_vector};
use crate::headless::render_data::{compute_render_bounds, stage_instance_model_matrices};
use crate::headless::{HeadlessRenderData, HeadlessSceneExportOptions, HeadlessView, SceneBounds};
use crate::lighting::WorkflowRender3D;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::colormap::gloss_to_roughness;
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::workflow::{workflow_bundle_display_fingerprint, workflow_streamline_fingerprint};

pub(crate) fn compute_scene_bounds(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
) -> SceneBounds {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    let mut expand = |point: Vec3| {
        min = min.min(point);
        max = max.max(point);
    };

    for nifti in &scene.nifti_files {
        for x in [0.0, nifti.volume.dims[0] as f32] {
            for y in [0.0, nifti.volume.dims[1] as f32] {
                for z in [0.0, nifti.volume.dims[2] as f32] {
                    expand(nifti.volume.voxel_to_world(Vec3::new(x, y, z)));
                }
            }
        }
    }

    for surface in &scene.gifti_surfaces {
        expand(surface.data.bbox_min);
        expand(surface.data.bbox_max);
    }

    for draw in &workflow.runtime.scene_plan.streamline_draws {
        if !draw.visible {
            continue;
        }
        let subset = crate::workflow::materialize_flow_gpu(draw.flow.clone());
        for position in &subset.positions {
            expand(Vec3::from(*position));
        }
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        if let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        {
            for (mesh, _) in &cache.meshes {
                for vertex in &mesh.vertices {
                    expand(Vec3::from(vertex.position));
                }
            }
        }
    }

    if min.is_finite() && max.is_finite() {
        SceneBounds { min, max }
    } else {
        let half = Vec3::splat((scene.volume_extent * 0.5).max(1.0));
        SceneBounds {
            min: scene.volume_center - half,
            max: scene.volume_center + half,
        }
    }
}

pub(crate) fn build_glb_scene(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
    render_3d: &WorkflowRender3D,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<Vec<u8>> {
    let mut builder = GlbBuilder::new();
    let scene_bounds = if options.view == HeadlessView::InflatedStage {
        compute_render_bounds(scene, render_data)
    } else {
        compute_scene_bounds(scene, workflow)
    };
    let scene_center = (scene_bounds.min + scene_bounds.max) * 0.5;
    let scene_radius = ((scene_bounds.max - scene_bounds.min) * 0.5)
        .length()
        .max(1.0);

    match options.view {
        HeadlessView::InflatedStage => {
            for (draw_index, draw) in workflow
                .runtime
                .scene_plan
                .stage_surface_draws
                .iter()
                .enumerate()
            {
                let Some(surface) = scene
                    .gifti_surfaces
                    .iter()
                    .find(|surface| surface.id == draw.source_id)
                else {
                    continue;
                };
                let colors = super::super::bake::surface_vertex_colors_for_export(
                    surface.data.as_ref(),
                    draw,
                );
                let positions = surface
                    .data
                    .vertices
                    .iter()
                    .map(|position| gltf_point(*position))
                    .collect::<Vec<_>>();
                let normals = surface
                    .data
                    .normals
                    .iter()
                    .map(|normal| gltf_vector(*normal))
                    .collect::<Vec<_>>();
                let material = builder.add_unlit_vertex_color_material(
                    format!("stage_surface_material_{draw_index}"),
                    draw.opacity,
                    false,
                );
                let mesh = builder.add_mesh(
                    format!("stage_surface_mesh_{}", surface.name),
                    &positions,
                    Some(&normals),
                    Some(&colors),
                    None,
                    &surface.data.indices,
                    material,
                    false,
                )?;
                for (panel_index, model_matrix) in stage_instance_model_matrices(
                    draw.structure,
                    surface.data.bbox_min,
                    surface.data.bbox_max,
                )
                .into_iter()
                .enumerate()
                {
                    builder.add_mesh_node(
                        format!(
                            "stage_surface_{}_{}_{}",
                            surface.name, draw_index, panel_index
                        ),
                        mesh,
                        gltf_transform(model_matrix),
                    );
                }
            }
        }
        _ => {
            for (draw_index, draw) in workflow.runtime.scene_plan.surface_draws.iter().enumerate() {
                let Some(surface) = scene
                    .gifti_surfaces
                    .iter()
                    .find(|surface| surface.id == draw.source_id)
                else {
                    continue;
                };
                let colors = super::super::bake::surface_vertex_colors_for_export(
                    surface.data.as_ref(),
                    draw,
                );
                let positions = surface
                    .data
                    .vertices
                    .iter()
                    .map(|position| gltf_point(*position))
                    .collect::<Vec<_>>();
                let normals = surface
                    .data
                    .normals
                    .iter()
                    .map(|normal| gltf_vector(*normal))
                    .collect::<Vec<_>>();
                let material = builder.add_vertex_color_material(
                    format!("surface_material_{draw_index}"),
                    draw.opacity,
                    false,
                    gloss_to_roughness(draw.gloss).max(0.22),
                    if draw.opacity < 0.999 { 0.12 } else { 0.08 },
                );
                let mesh = builder.add_mesh(
                    format!("surface_mesh_{}", surface.name),
                    &positions,
                    Some(&normals),
                    Some(&colors),
                    None,
                    &surface.data.indices,
                    material,
                    false,
                )?;
                builder.add_mesh_node(
                    format!("surface_{}_{}", surface.name, draw_index),
                    mesh,
                    gltf_transform(Mat4::from_cols_array_2d(&draw.model_matrix)),
                );
            }
        }
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        else {
            continue;
        };
        for (component_index, (mesh, label)) in cache.meshes.iter().enumerate() {
            add_bundle_mesh_to_glb(&mut builder, draw, mesh, label, component_index)?;
        }
    }

    for draw in &workflow.runtime.scene_plan.streamline_draws {
        if !draw.visible {
            continue;
        }
        let fingerprint = workflow_streamline_fingerprint(draw);
        let Some(cache) = workflow
            .execution_cache
            .tube_geometry_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        else {
            continue;
        };
        let positions = cache
            .vertices
            .iter()
            .map(|vertex| gltf_point(vertex.position))
            .collect::<Vec<_>>();
        let normals = cache
            .vertices
            .iter()
            .map(|vertex| gltf_vector(vertex.normal))
            .collect::<Vec<_>>();
        let colors = cache
            .vertices
            .iter()
            .map(|vertex| vertex.color)
            .collect::<Vec<_>>();
        let alpha = colors
            .iter()
            .fold(1.0f32, |acc, color| acc.min(color[3]))
            .clamp(0.0, 1.0);
        let material = builder.add_vertex_color_material(
            format!("streamline_material_{}", draw.draw_id),
            alpha,
            false,
            0.32,
            0.16,
        );
        let mesh = builder.add_mesh(
            format!("streamline_mesh_{}", draw.label),
            &positions,
            Some(&normals),
            Some(&colors),
            None,
            &cache.indices,
            material,
            false,
        )?;
        builder.add_mesh_node(
            format!("streamlines_{}", draw.label),
            mesh,
            glam::Mat4::IDENTITY,
        );
    }

    if options.include_slices && options.view != HeadlessView::InflatedStage {
        for volume in &render_data.volume_draws {
            if volume.opacity <= 0.001 {
                continue;
            }
            // Try a file-backed match first (NIfTI scene asset).
            if let Some(nifti) = scene
                .nifti_files
                .iter()
                .find(|nifti| nifti.id == volume.slice_key)
            {
                for axis_index in 0..3 {
                    if !scene.slice_visible[axis_index] {
                        continue;
                    }
                    add_slice_plane_to_glb(
                        &mut builder,
                        &nifti.volume,
                        volume,
                        axis_index,
                        scene.slice_indices[axis_index],
                        nifti.name.as_str(),
                    )?;
                }
                continue;
            }
            // Otherwise look for a Composite stack matching this draw's
            // handle. (InMemory scalar volumes still have no GLB
            // export path — that's a separate gap.)
            if let Some(stack) = workflow
                .runtime
                .scene_plan
                .volume_draws
                .iter()
                .find_map(|d| match &d.source {
                    crate::workflow::VolumeBacking::Composite { handle, stack }
                        if (*handle as usize) == volume.slice_key =>
                    {
                        Some(stack.clone())
                    }
                    _ => None,
                })
            {
                for axis_index in 0..3 {
                    if !scene.slice_visible[axis_index] {
                        continue;
                    }
                    super::composite_slice_plane::add_composite_slice_plane_to_glb(
                        &mut builder,
                        &stack,
                        volume,
                        axis_index,
                        scene.slice_indices[axis_index],
                    )?;
                }
            }
        }
    }

    if options.include_lights {
        add_lighting_rig_to_glb(&mut builder, render_3d, camera, scene_center, scene_radius);
    }

    if options.include_camera {
        let aspect = options.width as f32 / options.height.max(1) as f32;
        builder.add_camera_node("scene_camera".to_string(), camera, aspect);
    }

    let mut extras = Map::new();
    extras.insert(
        "trxviz_background".to_string(),
        match &render_3d.background {
            crate::lighting::WorkflowBackground3D::Solid { color } => {
                json!({ "mode": "solid", "color": color })
            }
            crate::lighting::WorkflowBackground3D::VerticalGradient { top, bottom } => {
                json!({ "mode": "vertical_gradient", "top": top, "bottom": bottom })
            }
        },
    );
    builder.scene_extras = Some(Value::Object(extras));
    builder.finish().context("finishing GLB")
}

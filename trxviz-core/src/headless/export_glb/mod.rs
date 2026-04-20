use std::collections::BTreeSet;

use anyhow::Context;
use glam::{Mat4, Vec3};
use serde_json::{Map, Value, json};

use super::bake::{bake_slice_png, surface_vertex_colors_for_export};
use super::render_3d::{compute_render_bounds, stage_instance_model_matrices};
use super::{
    HeadlessRenderData, HeadlessSceneExportOptions, HeadlessView, SceneBounds, VolumeDrawInfo,
};
use crate::data::bundle_mesh::BundleMesh;
use crate::data::nifti_data::NiftiVolume;
use crate::lighting::WorkflowRender3D;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::colormap::gloss_to_roughness;
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::workflow::{
    workflow_bundle_display_fingerprint, workflow_streamline_fingerprint,
};

const GLTF_AXIS_CONVERSION: glam::Mat3 = glam::Mat3::from_cols_array(&[
    1.0, 0.0, 0.0, //
    0.0, 0.0, -1.0, //
    0.0, 1.0, 0.0, //
]);

pub(super) fn compute_scene_bounds(
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

pub(super) fn build_glb_scene(
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
                let colors = surface_vertex_colors_for_export(surface.data.as_ref(), draw);
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
                let colors = surface_vertex_colors_for_export(surface.data.as_ref(), draw);
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
            let Some(nifti) = scene
                .nifti_files
                .iter()
                .find(|nifti| nifti.id == volume.file_id)
            else {
                continue;
            };
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

fn add_bundle_mesh_to_glb(
    builder: &mut GlbBuilder,
    draw: &crate::workflow::BundleDrawPlan,
    mesh: &BundleMesh,
    label: &str,
    component_index: usize,
) -> anyhow::Result<()> {
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| gltf_point(vertex.position))
        .collect::<Vec<_>>();
    let normals = mesh
        .vertices
        .iter()
        .map(|vertex| gltf_vector(vertex.normal))
        .collect::<Vec<_>>();
    let colors = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.color)
        .collect::<Vec<_>>();
    let material = if matches!(
        draw.build_mode,
        crate::workflow::BundleSurfaceBuildMode::Streamtubes
    ) {
        builder.add_unlit_vertex_color_material(
            format!("bundle_material_{}_{}", draw.draw_id, component_index),
            draw.opacity,
            true,
        )
    } else {
        builder.add_vertex_color_material(
            format!("bundle_material_{}_{}", draw.draw_id, component_index),
            draw.opacity,
            true,
            0.38,
            0.10,
        )
    };
    let mesh_index = builder.add_mesh(
        format!("bundle_mesh_{}_{}", draw.label, component_index),
        &positions,
        Some(&normals),
        Some(&colors),
        None,
        &mesh.indices,
        material,
        true,
    )?;
    builder.add_mesh_node(
        format!("bundle_{}_{}", label, component_index),
        mesh_index,
        glam::Mat4::IDENTITY,
    );
    Ok(())
}

fn add_slice_plane_to_glb(
    builder: &mut GlbBuilder,
    volume: &NiftiVolume,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
    volume_name: &str,
) -> anyhow::Result<()> {
    let corners = match axis_index {
        0 => volume.axial_slice_corners(slice_index),
        1 => volume.coronal_slice_corners(slice_index),
        _ => volume.sagittal_slice_corners(slice_index),
    };
    let positions = corners
        .into_iter()
        .map(|corner| gltf_point(corner.to_array()))
        .collect::<Vec<_>>();
    let normal = gltf_vector(slice_plane_normal(axis_index).to_array());
    let normals = vec![normal; 4];
    let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let png = bake_slice_png(volume, draw, axis_index, slice_index)?;
    let texture = builder.add_png_texture(
        format!(
            "{}_slice_texture_{}_{}",
            volume_name, axis_index, slice_index
        ),
        &png,
    );
    let material = builder.add_textured_material(
        format!(
            "{}_slice_material_{}_{}",
            volume_name, axis_index, slice_index
        ),
        draw.opacity,
        true,
        true,
        texture,
    );
    let mesh = builder.add_mesh(
        format!("{}_slice_mesh_{}_{}", volume_name, axis_index, slice_index),
        &positions,
        Some(&normals),
        None,
        Some(&texcoords),
        &indices,
        material,
        true,
    )?;
    builder.add_mesh_node(
        format!("{}_slice_{}_{}", volume_name, axis_index, slice_index),
        mesh,
        glam::Mat4::IDENTITY,
    );
    Ok(())
}

fn slice_plane_normal(axis_index: usize) -> Vec3 {
    match axis_index {
        0 => Vec3::Z,
        1 => Vec3::Y,
        _ => Vec3::X,
    }
}

fn gltf_point(point: [f32; 3]) -> [f32; 3] {
    (gltf_axis_conversion() * Vec3::from(point)).to_array()
}

fn gltf_vector(vector: [f32; 3]) -> [f32; 3] {
    (gltf_axis_conversion() * Vec3::from(vector))
        .normalize_or_zero()
        .to_array()
}

fn gltf_axis_conversion() -> glam::Mat3 {
    GLTF_AXIS_CONVERSION
}

fn gltf_transform(transform: Mat4) -> Mat4 {
    let basis = Mat4::from_mat3(gltf_axis_conversion());
    basis * transform * basis.inverse()
}

fn add_lighting_rig_to_glb(
    builder: &mut GlbBuilder,
    render_3d: &WorkflowRender3D,
    camera: &OrbitCamera,
    scene_center: Vec3,
    scene_radius: f32,
) {
    use crate::lighting::SceneLightingPreset;

    let eye = camera.eye();
    let forward = (camera.center - eye).normalize_or_zero();
    let right = forward.cross(Vec3::Z).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let rig_distance = scene_radius * 2.2;

    let (headlight_power, key_power, fill_power, rim_power, overhead_power, backfill_power) =
        match render_3d.lighting_preset {
            SceneLightingPreset::Flat => (9000.0, 4200.0, 2800.0, 0.0, 1800.0, 1200.0),
            SceneLightingPreset::Soft => (13000.0, 6500.0, 4200.0, 2200.0, 2800.0, 1800.0),
            SceneLightingPreset::Studio => (16500.0, 9000.0, 5200.0, 3200.0, 3600.0, 2400.0),
        };

    builder.add_spot_light(
        "camera_headlight".to_string(),
        eye,
        camera.center,
        55_f32.to_radians(),
        38_f32.to_radians(),
        headlight_power,
    );

    let key_pos = scene_center + right * rig_distance * 0.9 + up * rig_distance * 0.7
        - forward * rig_distance * 0.55;
    builder.add_spot_light(
        "key_light".to_string(),
        key_pos,
        scene_center,
        70_f32.to_radians(),
        48_f32.to_radians(),
        key_power,
    );

    let fill_pos = scene_center - right * rig_distance * 1.1 + up * rig_distance * 0.35
        - forward * rig_distance * 0.25;
    builder.add_point_light("fill_light".to_string(), fill_pos, fill_power);

    if rim_power > 0.0 {
        let rim_pos = scene_center - right * rig_distance * 0.8
            + up * rig_distance * 0.55
            + forward * rig_distance * 0.95;
        builder.add_spot_light(
            "rim_light".to_string(),
            rim_pos,
            scene_center,
            65_f32.to_radians(),
            42_f32.to_radians(),
            rim_power,
        );
    }

    let overhead_pos = scene_center + Vec3::Z * rig_distance * 1.5;
    builder.add_point_light("overhead_fill".to_string(), overhead_pos, overhead_power);

    let backfill_pos = scene_center + forward * rig_distance * 1.2 + up * rig_distance * 0.15;
    builder.add_point_light("back_fill".to_string(), backfill_pos, backfill_power);
}

struct GlbBuilder {
    bin: Vec<u8>,
    accessors: Vec<Value>,
    buffer_views: Vec<Value>,
    materials: Vec<Value>,
    meshes: Vec<Value>,
    nodes: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    scene_nodes: Vec<usize>,
    scene_extras: Option<Value>,
    extensions_used: BTreeSet<String>,
    extensions_required: BTreeSet<String>,
}

impl GlbBuilder {
    fn new() -> Self {
        Self {
            bin: Vec::new(),
            accessors: Vec::new(),
            buffer_views: Vec::new(),
            materials: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            cameras: Vec::new(),
            lights: Vec::new(),
            scene_nodes: Vec::new(),
            scene_extras: None,
            extensions_used: BTreeSet::new(),
            extensions_required: BTreeSet::new(),
        }
    }

    fn add_vertex_color_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
        roughness: f32,
        emissive_strength: f32,
    ) -> usize {
        let alpha_mode = if alpha < 0.999 { "BLEND" } else { "OPAQUE" };
        let material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": alpha_mode,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
            "emissiveFactor": [emissive_strength, emissive_strength, emissive_strength],
        });
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_unlit_vertex_color_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
    ) -> usize {
        self.extensions_used
            .insert("KHR_materials_unlit".to_string());
        self.extensions_required
            .insert("KHR_materials_unlit".to_string());

        let material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": if alpha < 0.999 { "BLEND" } else { "OPAQUE" },
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "extensions": {
                "KHR_materials_unlit": {}
            }
        });
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_textured_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
        unlit: bool,
        texture_index: usize,
    ) -> usize {
        let mut material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": if alpha < 0.999 { "BLEND" } else { "OPAQUE" },
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "baseColorTexture": { "index": texture_index },
                "metallicFactor": 0.0,
                "roughnessFactor": if unlit { 1.0 } else { 0.8 },
            }
        });
        if unlit {
            self.extensions_used
                .insert("KHR_materials_unlit".to_string());
            self.extensions_required
                .insert("KHR_materials_unlit".to_string());
            material["extensions"] = json!({ "KHR_materials_unlit": {} });
        }
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_png_texture(&mut self, name: String, png_bytes: &[u8]) -> usize {
        let buffer_view = self.push_bytes(png_bytes, None);
        self.images.push(json!({
            "name": name,
            "bufferView": buffer_view,
            "mimeType": "image/png",
        }));
        self.textures.push(json!({
            "source": self.images.len() - 1,
        }));
        self.textures.len() - 1
    }

    fn add_mesh(
        &mut self,
        name: String,
        positions: &[[f32; 3]],
        normals: Option<&[[f32; 3]]>,
        colors: Option<&[[f32; 4]]>,
        texcoords: Option<&[[f32; 2]]>,
        indices: &[u32],
        material: usize,
        double_sided: bool,
    ) -> anyhow::Result<usize> {
        let mut attributes = Map::new();
        attributes.insert(
            "POSITION".to_string(),
            Value::from(self.add_accessor_vec3_f32(positions, Some(34962), true)),
        );
        if let Some(normals) = normals {
            attributes.insert(
                "NORMAL".to_string(),
                Value::from(self.add_accessor_vec3_f32(normals, Some(34962), false)),
            );
        }
        if let Some(colors) = colors {
            attributes.insert(
                "COLOR_0".to_string(),
                Value::from(self.add_accessor_vec4_f32(colors, Some(34962))),
            );
        }
        if let Some(texcoords) = texcoords {
            attributes.insert(
                "TEXCOORD_0".to_string(),
                Value::from(self.add_accessor_vec2_f32(texcoords, Some(34962))),
            );
        }
        let indices_accessor = self.add_accessor_u32(indices, Some(34963));
        self.meshes.push(json!({
            "name": name,
            "primitives": [{
                "attributes": Value::Object(attributes),
                "indices": indices_accessor,
                "material": material,
                "mode": 4
            }]
        }));
        let _ = double_sided;
        Ok(self.meshes.len() - 1)
    }

    fn add_mesh_node(&mut self, name: String, mesh_index: usize, transform: glam::Mat4) {
        self.nodes.push(json!({
            "name": name,
            "mesh": mesh_index,
            "matrix": transform.to_cols_array(),
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_camera_node(&mut self, name: String, camera: &OrbitCamera, aspect: f32) {
        self.cameras.push(json!({
            "name": name,
            "type": "perspective",
            "perspective": {
                "aspectRatio": aspect,
                "yfov": camera.fov_y,
                "znear": camera.near,
                "zfar": camera.far,
            }
        }));
        let transform = camera_node_transform(camera.eye(), camera.center, Vec3::Z);
        self.nodes.push(json!({
            "name": name,
            "camera": self.cameras.len() - 1,
            "matrix": transform.to_cols_array(),
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_point_light(&mut self, name: String, position: Vec3, intensity: f32) {
        self.extensions_used
            .insert("KHR_lights_punctual".to_string());
        self.lights.push(json!({
            "name": name,
            "type": "point",
            "intensity": intensity,
            "color": [1.0, 1.0, 1.0],
            "range": 0.0,
        }));
        let position = gltf_axis_conversion() * position;
        self.nodes.push(json!({
            "name": name,
            "translation": position.to_array(),
            "extensions": {
                "KHR_lights_punctual": {
                    "light": self.lights.len() - 1
                }
            }
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_spot_light(
        &mut self,
        name: String,
        position: Vec3,
        target: Vec3,
        outer_cone_angle: f32,
        inner_cone_angle: f32,
        intensity: f32,
    ) {
        self.extensions_used
            .insert("KHR_lights_punctual".to_string());
        self.lights.push(json!({
            "name": name,
            "type": "spot",
            "intensity": intensity,
            "color": [1.0, 1.0, 1.0],
            "range": 0.0,
            "spot": {
                "innerConeAngle": inner_cone_angle,
                "outerConeAngle": outer_cone_angle,
            }
        }));
        let transform = camera_node_transform(position, target, Vec3::Z);
        self.nodes.push(json!({
            "name": name,
            "matrix": transform.to_cols_array(),
            "extensions": {
                "KHR_lights_punctual": {
                    "light": self.lights.len() - 1
                }
            }
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_accessor_vec3_f32(
        &mut self,
        data: &[[f32; 3]],
        target: Option<u32>,
        include_bounds: bool,
    ) -> usize {
        let bytes = bytemuck::cast_slice(data);
        let buffer_view = self.push_bytes(bytes, target);
        let mut accessor = json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC3",
        });
        if include_bounds && !data.is_empty() {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for value in data {
                for axis in 0..3 {
                    min[axis] = min[axis].min(value[axis]);
                    max[axis] = max[axis].max(value[axis]);
                }
            }
            accessor["min"] = json!(min);
            accessor["max"] = json!(max);
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn add_accessor_vec4_f32(&mut self, data: &[[f32; 4]], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC4",
        }));
        self.accessors.len() - 1
    }

    fn add_accessor_vec2_f32(&mut self, data: &[[f32; 2]], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC2",
        }));
        self.accessors.len() - 1
    }

    fn add_accessor_u32(&mut self, data: &[u32], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5125,
            "count": data.len(),
            "type": "SCALAR",
        }));
        self.accessors.len() - 1
    }

    fn push_bytes(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let mut buffer_view = json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
        });
        if let Some(target) = target {
            buffer_view["target"] = Value::from(target);
        }
        self.buffer_views.push(buffer_view);
        self.buffer_views.len() - 1
    }

    fn finish(self) -> anyhow::Result<Vec<u8>> {
        let mut root = json!({
            "asset": {
                "version": "2.0",
                "generator": "trxviz-cli",
            },
            "scene": 0,
            "scenes": [{
                "nodes": self.scene_nodes,
            }],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{
                "byteLength": self.bin.len(),
            }],
        });
        if let Some(extras) = self.scene_extras {
            root["scenes"][0]["extras"] = extras;
        }
        if !self.images.is_empty() {
            root["images"] = Value::Array(self.images);
        }
        if !self.textures.is_empty() {
            root["textures"] = Value::Array(self.textures);
        }
        if !self.cameras.is_empty() {
            root["cameras"] = Value::Array(self.cameras);
        }
        if !self.lights.is_empty() {
            root["extensions"] = json!({
                "KHR_lights_punctual": {
                    "lights": self.lights
                }
            });
        }
        if !self.extensions_used.is_empty() {
            root["extensionsUsed"] =
                Value::Array(self.extensions_used.into_iter().map(Value::from).collect());
        }
        if !self.extensions_required.is_empty() {
            root["extensionsRequired"] = Value::Array(
                self.extensions_required
                    .into_iter()
                    .map(Value::from)
                    .collect(),
            );
        }

        let mut json_bytes = serde_json::to_vec(&root)?;
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let mut bin = self.bin;
        while bin.len() % 4 != 0 {
            bin.push(0);
        }

        let total_length = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        Ok(glb)
    }
}

fn camera_node_transform(eye: Vec3, target: Vec3, up: Vec3) -> glam::Mat4 {
    let eye = gltf_axis_conversion() * eye;
    let target = gltf_axis_conversion() * target;
    let up = (gltf_axis_conversion() * up).normalize_or_zero();
    let forward = (target - eye).normalize_or_zero();
    let right = forward.cross(up).normalize_or_zero();
    let corrected_up = (-forward).cross(right).normalize_or_zero();
    glam::Mat4::from_cols(
        right.extend(0.0),
        corrected_up.extend(0.0),
        (-forward).extend(0.0),
        eye.extend(1.0),
    )
}

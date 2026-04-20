use anyhow::Result;

use super::{GlbBuilder, gltf_point, gltf_vector};
use crate::data::bundle_mesh::BundleMesh;

pub(super) fn add_bundle_mesh_to_glb(
    builder: &mut GlbBuilder,
    draw: &crate::workflow::BundleDrawPlan,
    mesh: &BundleMesh,
    label: &str,
    component_index: usize,
) -> Result<()> {
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

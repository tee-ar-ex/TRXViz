use anyhow::Result;
use glam::Vec3;

use super::{GlbBuilder, gltf_point, gltf_vector};
use crate::headless::VolumeDrawInfo;
use crate::workflow::CompositeVolumeStack;

pub(super) fn add_composite_slice_plane_to_glb(
    builder: &mut GlbBuilder,
    stack: &CompositeVolumeStack,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
) -> Result<()> {
    // Slice corners — same math as the scalar path, transformed
    // through the stack's base voxel_to_ras (layer 0 grid).
    let corners = slice_corners(stack.dims, stack.voxel_to_ras, axis_index, slice_index);
    let positions = corners
        .into_iter()
        .map(|corner| gltf_point(corner.to_array()))
        .collect::<Vec<_>>();
    let normal = gltf_vector(slice_plane_normal(axis_index).to_array());
    let normals = vec![normal; 4];
    let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = [0u32, 1, 2, 0, 2, 3];

    let png = super::super::bake::bake_composite_slice_png(stack, axis_index, slice_index)?;
    let texture =
        builder.add_png_texture(format!("composite_tex_{axis_index}_{slice_index}"), &png);
    let material = builder.add_textured_material(
        format!("composite_mat_{axis_index}_{slice_index}"),
        draw.opacity,
        true,
        true,
        texture,
    );
    let mesh = builder.add_mesh(
        format!("composite_mesh_{axis_index}_{slice_index}"),
        &positions,
        Some(&normals),
        None,
        Some(&texcoords),
        &indices,
        material,
        true,
    )?;
    builder.add_mesh_node(
        format!("composite_slice_{axis_index}_{slice_index}"),
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

fn slice_corners(
    dims: [usize; 3],
    voxel_to_ras: glam::Mat4,
    axis_index: usize,
    slice_index: usize,
) -> [Vec3; 4] {
    let to_world = |v: Vec3| voxel_to_ras.transform_point3(v);
    match axis_index {
        0 => {
            let kf = slice_index as f32;
            let i0 = -0.5;
            let i1 = dims[0] as f32 - 0.5;
            let j0 = -0.5;
            let j1 = dims[1] as f32 - 0.5;
            [
                to_world(Vec3::new(i0, j0, kf)),
                to_world(Vec3::new(i1, j0, kf)),
                to_world(Vec3::new(i1, j1, kf)),
                to_world(Vec3::new(i0, j1, kf)),
            ]
        }
        1 => {
            let jf = slice_index as f32;
            let i0 = -0.5;
            let i1 = dims[0] as f32 - 0.5;
            let k0 = -0.5;
            let k1 = dims[2] as f32 - 0.5;
            [
                to_world(Vec3::new(i0, jf, k0)),
                to_world(Vec3::new(i1, jf, k0)),
                to_world(Vec3::new(i1, jf, k1)),
                to_world(Vec3::new(i0, jf, k1)),
            ]
        }
        _ => {
            let if_ = slice_index as f32;
            let j0 = -0.5;
            let j1 = dims[1] as f32 - 0.5;
            let k0 = -0.5;
            let k1 = dims[2] as f32 - 0.5;
            [
                to_world(Vec3::new(if_, j0, k0)),
                to_world(Vec3::new(if_, j1, k0)),
                to_world(Vec3::new(if_, j1, k1)),
                to_world(Vec3::new(if_, j0, k1)),
            ]
        }
    }
}

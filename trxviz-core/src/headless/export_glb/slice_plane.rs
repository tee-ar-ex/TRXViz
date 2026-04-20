use anyhow::Result;
use glam::Vec3;

use super::{GlbBuilder, gltf_point, gltf_vector};
use crate::data::nifti_data::NiftiVolume;
use crate::headless::VolumeDrawInfo;

pub(super) fn add_slice_plane_to_glb(
    builder: &mut GlbBuilder,
    volume: &NiftiVolume,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
    volume_name: &str,
) -> Result<()> {
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
    let png = super::super::bake::bake_slice_png(volume, draw, axis_index, slice_index)?;
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

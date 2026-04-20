use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::nifti_data::NiftiVolume;
use crate::renderer::colormap::{float_channel, surface_colormap_rgb, volume_colormap_rgb};
use image::ImageEncoder;

use super::VolumeDrawInfo;

pub(super) fn surface_vertex_colors_for_export(
    surface: &GiftiSurfaceData,
    draw: &crate::workflow::SurfaceDrawPlan,
) -> Vec<[f32; 4]> {
    if !draw.vertex_rgba.is_empty() {
        return draw.vertex_rgba.clone();
    }
    bake_surface_vertex_colors(surface, draw)
}

#[cfg(feature = "png-export")]
pub(super) fn bake_surface_vertex_colors(
    surface: &GiftiSurfaceData,
    draw: &crate::workflow::SurfaceDrawPlan,
) -> Vec<[f32; 4]> {
    let default = [draw.color[0], draw.color[1], draw.color[2], 1.0];
    let Some(scalars) = &draw.projection_scalars else {
        return vec![default; surface.vertices.len()];
    };

    scalars
        .iter()
        .map(|scalar| {
            let denom = (draw.range_max - draw.range_min).max(1e-6);
            let t = ((*scalar - draw.range_min) / denom).clamp(0.0, 1.0);
            let map_alpha = draw.map_opacity * if t >= draw.map_threshold { 1.0 } else { 0.0 };
            let map_rgb = surface_colormap_rgb(t, draw.projection_colormap);
            [
                draw.color[0] * (1.0 - map_alpha) + map_rgb[0] * map_alpha,
                draw.color[1] * (1.0 - map_alpha) + map_rgb[1] * map_alpha,
                draw.color[2] * (1.0 - map_alpha) + map_rgb[2] * map_alpha,
                1.0,
            ]
        })
        .collect()
}

#[cfg(feature = "png-export")]
pub(super) fn bake_slice_png(
    volume: &NiftiVolume,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
) -> anyhow::Result<Vec<u8>> {
    let (width, height) = match axis_index {
        0 => (volume.dims[0] as u32, volume.dims[1] as u32),
        1 => (volume.dims[0] as u32, volume.dims[2] as u32),
        _ => (volume.dims[1] as u32, volume.dims[2] as u32),
    };
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    let lo = draw.window_center - draw.window_width * 0.5;
    let hi = draw.window_center + draw.window_width * 0.5;

    for row in 0..height as usize {
        for col in 0..width as usize {
            let value = match axis_index {
                0 => {
                    volume.data
                        [col + row * volume.dims[0] + slice_index * volume.dims[0] * volume.dims[1]]
                }
                1 => {
                    volume.data
                        [col + slice_index * volume.dims[0] + row * volume.dims[0] * volume.dims[1]]
                }
                _ => {
                    volume.data
                        [slice_index + col * volume.dims[0] + row * volume.dims[0] * volume.dims[1]]
                }
            };
            let t = ((value - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
            let rgb = volume_colormap_rgb(t, draw.colormap);
            let dst = (row * width as usize + col) * 4;
            rgba[dst] = float_channel(rgb[0]);
            rgba[dst + 1] = float_channel(rgb[1]);
            rgba[dst + 2] = float_channel(rgb[2]);
            rgba[dst + 3] = float_channel(draw.opacity);
        }
    }

    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png).write_image(
        &rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(png)
}

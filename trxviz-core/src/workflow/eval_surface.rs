use crate::data::cifti::{CiftiStructure, ScalarKind, ScalarMetadata, SurfaceScalars};
use crate::data::loaded_files::FileId;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::scene::LoadedGiftiSurface;

use super::*;

pub(crate) fn compose_surface_appearance(
    surface_id: FileId,
    surface: &LoadedGiftiSurface,
    layers: &[SurfaceOverlayLayerConfig],
    scalar_inputs: &[Option<EvaluatedValue>],
) -> WorkflowResult<SurfaceAppearance> {
    let mut vertex_rgba = vec![DEFAULT_SURFACE_BASE_RGBA; surface.data.vertices.len()];
    let mut appearance_structure = None;
    if let Some(base) = layers.first() {
        for color in &mut vertex_rgba {
            *color = base.solid_color;
            color[3] = base.opacity.clamp(0.0, 1.0);
        }
    }
    let mut legend_labels = Vec::new();
    for (layer_index, layer) in layers.iter().enumerate() {
        if !layer.enabled {
            continue;
        }
        let Some(Some(EvaluatedValue {
            value: WorkflowValue::SurfaceScalars(scalars),
            ..
        })) = scalar_inputs.get(layer_index)
        else {
            if !layer.legend_label.trim().is_empty() {
                legend_labels.push(layer.legend_label.clone());
            }
            continue;
        };
        validate_surface_scalars(surface_id, surface, scalars)?;
        if appearance_structure.is_none() {
            appearance_structure = scalars.structure;
        }
        overlay_surface_scalars(&mut vertex_rgba, scalars, layer);
        if !layer.legend_label.trim().is_empty() {
            legend_labels.push(layer.legend_label.clone());
        } else if !scalars.metadata.map_name.trim().is_empty() {
            legend_labels.push(scalars.metadata.map_name.clone());
        }
    }
    Ok(SurfaceAppearance {
        source_id: surface_id,
        structure: appearance_structure,
        vertex_rgba,
        legend_labels,
    })
}

pub(crate) fn surface_display_model_matrix(
    surface: &LoadedGiftiSurface,
    structure: Option<CiftiStructure>,
    space: SurfaceDisplaySpace,
) -> glam::Mat4 {
    if space == SurfaceDisplaySpace::Anatomical {
        return glam::Mat4::IDENTITY;
    }
    let center = (surface.data.bbox_min + surface.data.bbox_max) * 0.5;
    let extents = surface.data.bbox_max - surface.data.bbox_min;
    let span = extents
        .x
        .abs()
        .max(extents.y.abs())
        .max(extents.z.abs())
        .max(1.0);
    let separation = span * 0.8;
    let (x_shift, turn_deg): (f32, f32) = match structure {
        Some(CiftiStructure::CortexLeft) => (separation, -90.0),
        Some(CiftiStructure::CortexRight) => (-separation, 90.0),
        _ => (0.0, 0.0),
    };
    glam::Mat4::from_translation(glam::Vec3::new(x_shift, 0.0, 0.0))
        * glam::Mat4::from_rotation_z(turn_deg.to_radians())
        * glam::Mat4::from_translation(-center)
}

fn validate_surface_scalars(
    surface_id: FileId,
    surface: &LoadedGiftiSurface,
    scalars: &SurfaceScalars,
) -> WorkflowResult<()> {
    if scalars.vertex_count != surface.data.vertices.len() {
        return Err(WorkflowError::Evaluation(format!(
            "Surface scalars have {} vertices but surface {} has {}",
            scalars.vertex_count,
            surface_id,
            surface.data.vertices.len()
        )));
    }
    if let Some(bound_surface_id) = scalars.source_surface_id
        && bound_surface_id != surface_id
    {
        return Err(WorkflowError::Evaluation(format!(
            "Surface scalars are bound to surface {} and cannot be applied to surface {}",
            bound_surface_id, surface_id
        )));
    }
    Ok(())
}

fn overlay_surface_scalars(
    vertex_rgba: &mut [[f32; 4]],
    scalars: &SurfaceScalars,
    layer: &SurfaceOverlayLayerConfig,
) {
    let (range_min, range_max) = scalars
        .metadata
        .suggested_range
        .unwrap_or((layer.range_min, layer.range_max));
    let denom = (range_max - range_min).max(1e-6);
    for (dst, scalar) in vertex_rgba.iter_mut().zip(scalars.values.iter()) {
        if !scalar.is_finite() {
            continue;
        }
        let src = match scalars.kind {
            ScalarKind::Label if layer.use_label_colors => {
                label_rgba(*scalar as i32, &scalars.metadata)
            }
            _ => {
                if *scalar < layer.threshold_min || *scalar > layer.threshold_max {
                    continue;
                }
                let t = ((*scalar - range_min) / denom).clamp(0.0, 1.0);
                let rgb = surface_colormap_rgb(t, layer.colormap);
                [rgb[0], rgb[1], rgb[2], layer.opacity.clamp(0.0, 1.0)]
            }
        };
        alpha_blend(dst, src);
    }
}

fn label_rgba(label: i32, metadata: &ScalarMetadata) -> [f32; 4] {
    metadata
        .label_table
        .iter()
        .find(|entry| entry.key == label)
        .map(|entry| entry.rgba)
        .unwrap_or([0.0, 0.0, 0.0, 0.0])
}

fn alpha_blend(dst: &mut [f32; 4], src: [f32; 4]) {
    let src_a = src[3].clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return;
    }
    let inv = 1.0 - src_a;
    dst[0] = dst[0] * inv + src[0] * src_a;
    dst[1] = dst[1] * inv + src[1] * src_a;
    dst[2] = dst[2] * inv + src[2] * src_a;
    dst[3] = (dst[3] + src_a).clamp(0.0, 1.0);
}

fn surface_colormap_rgb(t: f32, colormap: SurfaceColormap) -> [f32; 3] {
    match colormap {
        SurfaceColormap::BlueWhiteRed => {
            if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0]
            } else {
                let s = (1.0 - t) * 2.0;
                [1.0, s, s]
            }
        }
        SurfaceColormap::Viridis => {
            let anchors = [
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
            ];
            lerp_colormap(&anchors, t)
        }
        SurfaceColormap::Inferno => {
            let anchors = [
                [0.001, 0.000, 0.014],
                [0.125, 0.047, 0.290],
                [0.302, 0.073, 0.488],
                [0.511, 0.121, 0.561],
                [0.709, 0.212, 0.486],
                [0.865, 0.316, 0.347],
                [0.962, 0.471, 0.212],
                [0.988, 0.683, 0.139],
                [0.978, 0.893, 0.306],
            ];
            lerp_colormap(&anchors, t)
        }
    }
}

fn lerp_colormap(anchors: &[[f32; 3]], t: f32) -> [f32; 3] {
    if anchors.len() == 1 {
        return anchors[0];
    }
    let x = t.clamp(0.0, 1.0) * (anchors.len() as f32 - 1.0);
    let i = x.floor() as usize;
    let j = (i + 1).min(anchors.len() - 1);
    let f = x - i as f32;
    [
        anchors[i][0] * (1.0 - f) + anchors[j][0] * f,
        anchors[i][1] * (1.0 - f) + anchors[j][1] * f,
        anchors[i][2] * (1.0 - f) + anchors[j][2] * f,
    ]
}

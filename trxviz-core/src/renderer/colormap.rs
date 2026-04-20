use crate::renderer::mesh_renderer::SurfaceColormap;

pub fn surface_colormap_rgb(t: f32, colormap: SurfaceColormap) -> [f32; 3] {
    let t = clamp01(t);
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
        SurfaceColormap::Viridis => interpolate_colormap(
            t,
            &[
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
            ],
        ),
        SurfaceColormap::Inferno => interpolate_colormap(
            t,
            &[
                [0.001, 0.000, 0.014],
                [0.125, 0.047, 0.290],
                [0.302, 0.073, 0.488],
                [0.511, 0.121, 0.561],
                [0.709, 0.212, 0.486],
                [0.865, 0.316, 0.347],
                [0.962, 0.471, 0.212],
                [0.988, 0.683, 0.139],
                [0.978, 0.893, 0.306],
            ],
        ),
    }
}

pub fn interpolate_colormap(t: f32, colors: &[[f32; 3]]) -> [f32; 3] {
    if colors.len() == 1 {
        return colors[0];
    }
    let x = clamp01(t) * (colors.len() - 1) as f32;
    let i = x.floor().clamp(0.0, (colors.len() - 2) as f32) as usize;
    let f = x.fract();
    [
        colors[i][0] + (colors[i + 1][0] - colors[i][0]) * f,
        colors[i][1] + (colors[i + 1][1] - colors[i][1]) * f,
        colors[i][2] + (colors[i + 1][2] - colors[i][2]) * f,
    ]
}

pub fn volume_colormap_rgb(t: f32, colormap: u32) -> [f32; 3] {
    match colormap {
        1 => [
            clamp01(t * 2.5),
            clamp01(t * 2.5 - 1.0),
            clamp01(t * 5.0 - 4.0),
        ],
        2 => [t, 1.0 - t, 1.0],
        3 => [1.0, t, 0.0],
        4 => [0.0, t, 1.0],
        _ => [t, t, t],
    }
}

pub fn gloss_to_roughness(gloss: f32) -> f32 {
    (1.0 - clamp01(gloss) * 0.9).clamp(0.05, 1.0)
}

pub fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn float_channel(value: f32) -> u8 {
    (clamp01(value) * 255.0).round() as u8
}

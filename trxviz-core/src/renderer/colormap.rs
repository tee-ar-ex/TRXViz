use crate::data::loaded_files::VolumeColormap;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::workflow::GlyphColormap;

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
        SurfaceColormap::ColdHot => cold_hot_rgb(t),
        SurfaceColormap::ColdWhiteHot => cold_white_hot_rgb(t),
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
        5 => cold_hot_rgb(t),
        6 => cold_white_hot_rgb(t),
        _ => [t, t, t],
    }
}

// nilearn's stat-map diverging pair, built exactly from matplotlib's
// piecewise-linear `hot` (R ramps over [0, 0.365079], G over [0.365079,
// 0.746032], B over [0.746032, 1], with hot's 0.0416 red offset at black)
// and its channel-rotated "cold" twin (black → blue → cyan → white).
fn hot_ramp(x: f32) -> [f32; 3] {
    let c = clamp01(x);
    let r = if c < 0.365079 {
        0.0416 + c * (1.0 - 0.0416) / 0.365079
    } else {
        1.0
    };
    let g = clamp01((c - 0.365079) / (0.746032 - 0.365079));
    let b = clamp01((c - 0.746032) / (1.0 - 0.746032));
    [r, g, b]
}

fn cold_ramp(x: f32) -> [f32; 3] {
    let [r, g, b] = hot_ramp(x);
    [b, g, r]
}

/// nilearn `cold_hot`: white/cyan/blue for negatives → black at the
/// midpoint → red/yellow/white for positives. The default stat-map
/// colormap on dark backgrounds.
pub fn cold_hot_rgb(t: f32) -> [f32; 3] {
    let x = clamp01(t);
    if x < 0.5 {
        cold_ramp(1.0 - 2.0 * x)
    } else {
        hot_ramp(2.0 * x - 1.0)
    }
}

/// nilearn `cold_white_hot`: the white-centered variant for light
/// backgrounds — dark blue → cyan → white at the midpoint → orange → dark
/// red.
pub fn cold_white_hot_rgb(t: f32) -> [f32; 3] {
    let x = clamp01(t);
    if x < 0.5 {
        cold_ramp(2.0 * x)
    } else {
        hot_ramp(2.0 - 2.0 * x)
    }
}

/// Palette codes shared with `fixel.wgsl` / `glyph.wgsl` `sample_palette`:
/// 0 = directional (not a LUT), 1 = solid (glyph-only), 2 = plasma,
/// 3 = viridis, 4 = inferno, 5 = grayscale, 6 = blue-white-red,
/// 7 = cold-hot, 8 = cold-white-hot. Unknown codes fall back to
/// blue-white-red, matching the shaders.
pub fn palette_code_rgb(code: u32, t: f32) -> [f32; 3] {
    let x = clamp01(t);
    match code {
        2 => palette_plasma(x),
        3 => palette_viridis(x),
        4 => palette_inferno(x),
        5 => [x, x, x],
        7 => cold_hot_rgb(x),
        8 => cold_white_hot_rgb(x),
        _ => palette_bwr(x),
    }
}

/// Palette code for a [`SurfaceColormap`] applied to fixels/glyphs — the one
/// home for this mapping (BlueWhiteRed must be 6; 5 is grayscale in the
/// shaders).
pub fn surface_colormap_to_palette_code(colormap: SurfaceColormap) -> u32 {
    match colormap {
        SurfaceColormap::BlueWhiteRed => 6,
        SurfaceColormap::Viridis => 3,
        SurfaceColormap::Inferno => 4,
        SurfaceColormap::ColdHot => 7,
        SurfaceColormap::ColdWhiteHot => 8,
    }
}

/// Palette code for a [`GlyphColormap`], matching `glyph.wgsl` `color_mode`.
pub fn glyph_colormap_code(colormap: GlyphColormap) -> u32 {
    match colormap {
        GlyphColormap::Directional => 0,
        GlyphColormap::Plasma => 2,
        GlyphColormap::Viridis => 3,
        GlyphColormap::Inferno => 4,
        GlyphColormap::BlueWhiteRed => 6,
        GlyphColormap::ColdHot => 7,
        GlyphColormap::ColdWhiteHot => 8,
    }
}

/// Reference to any colormap the renderers use, so UI code (colorbars) can
/// sample the exact ramp a layer was drawn with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColormapRef {
    /// `fixel.wgsl` / `glyph.wgsl` `sample_palette` code.
    FixelGlyph(u32),
    Volume(VolumeColormap),
    Surface(SurfaceColormap),
}

pub fn colormap_ref_rgb(cmap: ColormapRef, t: f32) -> [f32; 3] {
    match cmap {
        ColormapRef::FixelGlyph(code) => palette_code_rgb(code, t),
        ColormapRef::Volume(v) => volume_colormap_rgb(t, v.as_u32()),
        ColormapRef::Surface(s) => surface_colormap_rgb(t, s),
    }
}

// CPU mirrors of the polynomial palettes in fixel.wgsl / glyph.wgsl. The
// coefficients must stay bit-identical to the shaders so colorbars match the
// rendered pixels.
fn horner(x: f32, c: [f32; 6]) -> f32 {
    (c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))))).clamp(0.0, 1.0)
}

fn palette_plasma(x: f32) -> [f32; 3] {
    [
        horner(
            x,
            [
                0.05873234, 2.176514, -2.68946, 6.130348, -11.10743, 5.436053,
            ],
        ),
        horner(
            x,
            [
                0.0233367, 0.2383834, 0.8639469, -1.930449, 2.137246, -0.6826481,
            ],
        ),
        horner(
            x,
            [
                0.5433248, 1.465357, -4.20353, 4.693453, -2.372373, 0.4475437,
            ],
        ),
    ]
}

fn palette_viridis(x: f32) -> [f32; 3] {
    [
        horner(
            x,
            [0.2777273, -0.4872, 4.61385, -14.1862, 14.7864, -4.07578],
        ),
        horner(
            x,
            [0.0054173, 1.40495, 0.33305, -3.02716, 2.69389, -0.90487],
        ),
        horner(
            x,
            [0.3340998, 0.73916, -2.94589, 5.48795, -5.52477, 2.17525],
        ),
    ]
}

fn palette_inferno(x: f32) -> [f32; 3] {
    [
        horner(
            x,
            [0.0002189, 0.1065, 11.60249, -41.7039, 55.0215, -24.00824],
        ),
        horner(x, [0.001652, -0.69594, 4.2625, -8.4392, 7.7773, -2.74041]),
        horner(x, [-0.019480, 3.9314, -15.9347, 44.3551, -55.2813, 22.9486]),
    ]
}

fn palette_bwr(x: f32) -> [f32; 3] {
    if x < 0.5 {
        let s = x * 2.0;
        [s, s, 1.0]
    } else {
        let s = (x - 0.5) * 2.0;
        [1.0, 1.0 - s, 1.0 - s]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rgb_close(a: [f32; 3], b: [f32; 3], tol: f32) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "channel {i}: {:?} vs {:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn palette_code_endpoints() {
        // Blue-white-red: blue at 0, white at midpoint, red at 1.
        assert_rgb_close(palette_code_rgb(6, 0.0), [0.0, 0.0, 1.0], 1e-6);
        assert_rgb_close(palette_code_rgb(6, 0.5), [1.0, 1.0, 1.0], 1e-6);
        assert_rgb_close(palette_code_rgb(6, 1.0), [1.0, 0.0, 0.0], 1e-6);
        // Grayscale is the identity ramp.
        assert_rgb_close(palette_code_rgb(5, 0.25), [0.25, 0.25, 0.25], 1e-6);
        // Polynomial palettes: dark-to-light endpoints from the WGSL constants.
        assert_rgb_close(
            palette_code_rgb(3, 0.0),
            [0.2777273, 0.0054173, 0.3340998],
            1e-4,
        );
        // The polynomial fit, not true viridis — must match what the shader renders.
        assert_rgb_close(
            palette_code_rgb(3, 1.0),
            [0.9287973, 0.5052773, 0.2657998],
            1e-4,
        );
        let inferno_lo = palette_code_rgb(4, 0.0);
        assert!(inferno_lo.iter().all(|&c| c < 0.05));
        let plasma_lo = palette_code_rgb(2, 0.0);
        assert!(plasma_lo[2] > 0.5 && plasma_lo[1] < 0.1);
        // Unknown codes fall back to blue-white-red, matching the shaders.
        assert_rgb_close(palette_code_rgb(99, 0.0), palette_code_rgb(6, 0.0), 0.0);
        // Inputs outside [0,1] clamp.
        assert_rgb_close(palette_code_rgb(5, 2.0), [1.0, 1.0, 1.0], 1e-6);
        assert_rgb_close(palette_code_rgb(5, -1.0), [0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn cold_hot_matches_nilearn_shape() {
        // Extremes are white, midpoint is black (hot's 0.0416 red seed),
        // negatives are blue/cyan, positives red/orange — the nilearn
        // stat-map look.
        assert_rgb_close(cold_hot_rgb(0.0), [1.0, 1.0, 1.0], 1e-6);
        assert_rgb_close(cold_hot_rgb(1.0), [1.0, 1.0, 1.0], 1e-6);
        assert_rgb_close(cold_hot_rgb(0.5), [0.0416, 0.0, 0.0], 1e-4);
        let neg = cold_hot_rgb(0.25); // cold(0.5): saturated blue
        assert!(
            neg[2] > 0.99 && neg[0] < 0.01 && neg[1] > 0.2 && neg[1] < 0.5,
            "{neg:?}"
        );
        let pos = cold_hot_rgb(0.75); // hot(0.5): saturated red-orange
        assert!(
            pos[0] > 0.99 && pos[2] < 0.01 && pos[1] > 0.2 && pos[1] < 0.5,
            "{pos:?}"
        );
        // The two maps are mirror pairs: cold_white_hot has white center,
        // dark extremes.
        assert_rgb_close(cold_white_hot_rgb(0.5), [1.0, 1.0, 1.0], 1e-6);
        assert_rgb_close(cold_white_hot_rgb(0.0), [0.0, 0.0, 0.0416], 1e-4);
        assert_rgb_close(cold_white_hot_rgb(1.0), [0.0416, 0.0, 0.0], 1e-4);
        // Code plumbing: palette 7/8 and volume 5/6 reach the same ramps.
        assert_rgb_close(palette_code_rgb(7, 0.25), cold_hot_rgb(0.25), 0.0);
        assert_rgb_close(palette_code_rgb(8, 0.25), cold_white_hot_rgb(0.25), 0.0);
        assert_rgb_close(volume_colormap_rgb(0.25, 5), cold_hot_rgb(0.25), 0.0);
        assert_rgb_close(volume_colormap_rgb(0.25, 6), cold_white_hot_rgb(0.25), 0.0);
        assert_rgb_close(
            surface_colormap_rgb(0.25, SurfaceColormap::ColdHot),
            cold_hot_rgb(0.25),
            0.0,
        );
    }

    #[test]
    fn surface_and_glyph_code_mappings() {
        // BlueWhiteRed must be 6: code 5 renders grayscale in fixel.wgsl.
        assert_eq!(
            surface_colormap_to_palette_code(SurfaceColormap::BlueWhiteRed),
            6
        );
        assert_eq!(
            surface_colormap_to_palette_code(SurfaceColormap::Viridis),
            3
        );
        assert_eq!(
            surface_colormap_to_palette_code(SurfaceColormap::Inferno),
            4
        );
        assert_eq!(glyph_colormap_code(GlyphColormap::Directional), 0);
        assert_eq!(glyph_colormap_code(GlyphColormap::BlueWhiteRed), 6);
    }

    #[test]
    fn colormap_ref_dispatch() {
        assert_rgb_close(
            colormap_ref_rgb(ColormapRef::FixelGlyph(5), 0.5),
            [0.5, 0.5, 0.5],
            1e-6,
        );
        assert_rgb_close(
            colormap_ref_rgb(ColormapRef::Volume(VolumeColormap::Grayscale), 0.5),
            [0.5, 0.5, 0.5],
            1e-6,
        );
        assert_rgb_close(
            colormap_ref_rgb(ColormapRef::Surface(SurfaceColormap::BlueWhiteRed), 0.0),
            [0.0, 0.0, 1.0],
            1e-6,
        );
    }
}

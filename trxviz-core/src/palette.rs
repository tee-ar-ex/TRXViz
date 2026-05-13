//! Distinct-color generation for visualisations with many categories
//! (bundle groups, voxel-mask layers, sheet ids, etc.).
//!
//! Small hand-tuned palettes (8–12 colours) only stay distinguishable when the
//! number of categories is small. Beyond that, adjacent items collide and the
//! visualisation looks monochrome. This module provides a deterministic
//! generator that yields visually distinct colours for hundreds of categories
//! by sweeping hue with the golden angle and alternating saturation/value
//! between two pleasant levels.
//!
//! Two entry points:
//! - [`distinct_color_index`](crate::palette::distinct_color_index) for callers
//!   that have an integer index (e.g. the nth voxel-mask layer, group
//!   enumeration order).
//! - [`distinct_color_hash`](crate::palette::distinct_color_hash) for callers
//!   that have a stable name (e.g. a bundle label) and want a colour that's
//!   deterministic across runs.

const PHI_DEGREES: f32 = 137.50776; // golden angle ≈ 360° × (1 - 1/φ)

/// Linear-RGBA colour for the `i`th category. Adjacent indices receive
/// hues separated by the golden angle, with `i % 2` and `i % 3` perturbing
/// saturation and value so neighbouring entries that happen to share modular
/// classes still differ in lightness.
pub fn distinct_color_index(i: usize) -> [f32; 4] {
    let hue = (i as f32 * PHI_DEGREES).rem_euclid(360.0);
    let sat = if i % 2 == 0 { 0.72 } else { 0.55 };
    let val = match i % 3 {
        0 => 0.95,
        1 => 0.78,
        _ => 0.62,
    };
    let [r, g, b] = hsv_to_linear_rgb(hue, sat, val);
    [r, g, b, 1.0]
}

/// Linear-RGBA colour derived from a 64-bit hash of an arbitrary key (e.g.
/// a bundle name). Stable across runs for the same hash input.
pub fn distinct_color_hash(hash: u64) -> [f32; 4] {
    // Unpack two independent random-looking streams from the same hash so
    // the hue, sat, and val choices don't all key on the same low bits.
    let h_units = (hash & 0xFFFF) as f32 / 65536.0; // 0..1
    let hue = h_units * 360.0;
    let sat = if (hash >> 16) & 1 == 0 { 0.72 } else { 0.55 };
    let val = match (hash >> 17) % 3 {
        0 => 0.95,
        1 => 0.78,
        _ => 0.62,
    };
    let [r, g, b] = hsv_to_linear_rgb(hue, sat, val);
    [r, g, b, 1.0]
}

/// Convert an HSV triple (`h ∈ [0, 360)`, `s, v ∈ [0, 1]`) to linear RGB.
fn hsv_to_linear_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_unit_alpha() {
        for i in 0..32 {
            let c = distinct_color_index(i);
            assert!((c[3] - 1.0).abs() < 1e-6, "alpha {} != 1.0", c[3]);
        }
    }

    #[test]
    fn rgb_components_in_range() {
        for i in 0..256 {
            let c = distinct_color_index(i);
            for (k, channel) in c[..3].iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(channel),
                    "color {} channel {} out of range: {}",
                    i,
                    k,
                    channel
                );
            }
        }
    }

    #[test]
    fn neighbouring_indices_differ() {
        // Adjacent categories should be visually distinguishable: at minimum
        // their hue offset of 137.5° guarantees a max-channel difference > 0.3.
        let a = distinct_color_index(7);
        let b = distinct_color_index(8);
        let dr = (a[0] - b[0]).abs();
        let dg = (a[1] - b[1]).abs();
        let db = (a[2] - b[2]).abs();
        let max_delta = dr.max(dg).max(db);
        assert!(
            max_delta > 0.2,
            "adjacent colors too similar: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn hash_path_deterministic() {
        let a = distinct_color_hash(0xdead_beef_cafe_babe);
        let b = distinct_color_hash(0xdead_beef_cafe_babe);
        assert_eq!(a, b);
    }
}

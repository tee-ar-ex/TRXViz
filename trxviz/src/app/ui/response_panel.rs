//! "Continuous-b response" panel.
//!
//! Visualizes the global dense-b tissue response functions carried in a cs-cbmt
//! ODX header (WM zonal SH `R_l(b)`, plus isotropic GM/CSF `R_0(b)`). The
//! response is global to the scan (one convolution kernel for every voxel), so
//! it is not tied to the cursor. Two views, both driven by a `b` slider:
//!
//!  1. a 3D glyph of the WM response's angular profile at the chosen b
//!     (axially symmetric about the fibre axis; sharpens with b), and
//!  2. the `R_l(b)` coefficient curves with a vertical b-marker.
//!
//! The glyph is software-rendered directly into the egui `Painter` (no wgpu
//! plumbing): the WM response is zonal SH, so amplitude is a function of the
//! polar angle only and evaluates from even Legendre polynomials. No-op for
//! datasets without a continuous-b response.

use egui::{Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, pos2, vec2};
use glam::{Mat3, Vec3};
use trxviz_core::data::odx_data::{ContbResponse, OdxScene};

/// Persistent UI state for the response panel.
pub struct ResponsePanelState {
    /// Currently scrubbed b-value (s/mm²).
    pub b: f64,
    /// Glyph camera rotation (radians), driven by dragging on the glyph.
    yaw: f32,
    pitch: f32,
    initialized: bool,
}

impl Default for ResponsePanelState {
    fn default() -> Self {
        Self { b: 1000.0, yaw: 0.6, pitch: 0.35, initialized: false }
    }
}

const WM_COLORS: [Color32; 5] = [
    Color32::from_rgb(0xd6, 0x27, 0x28), // r0
    Color32::from_rgb(0xff, 0x7f, 0x0e), // r2
    Color32::from_rgb(0x2c, 0xa0, 0x2c), // r4
    Color32::from_rgb(0x17, 0x9e, 0xc9), // r6
    Color32::from_rgb(0x94, 0x67, 0xbd), // r8
];
const GM_COLOR: Color32 = Color32::from_rgb(0x8c, 0x56, 0x4b);
const CSF_COLOR: Color32 = Color32::from_rgb(0x7f, 0x7f, 0x7f);

/// `√((2l+1)/4π)` for l = 0,2,4,6,8 (the zonal SH normalization).
const YL0: [f32; 5] = [0.282_094_8, 0.630_783_1, 0.846_284_4, 1.017_107_2, 1.163_106_6];

/// Even Legendre polynomials P_{0,2,4,6,8}(x).
fn legendre_even(x: f32) -> [f32; 5] {
    let x2 = x * x;
    let (x4, x6, x8) = (x2 * x2, x2 * x2 * x2, x2 * x2 * x2 * x2);
    [
        1.0,
        (3.0 * x2 - 1.0) * 0.5,
        (35.0 * x4 - 30.0 * x2 + 3.0) / 8.0,
        (231.0 * x6 - 315.0 * x4 + 105.0 * x2 - 5.0) / 16.0,
        (6435.0 * x8 - 12012.0 * x6 + 6930.0 * x4 - 1260.0 * x2 + 35.0) / 128.0,
    ]
}

/// WM response amplitude along a direction whose angle from the fibre axis has
/// cosine `cos` — `Σ_l r_l · Y_{l,0}(cos)`.
fn wm_amplitude(coeffs: &[f32], cos: f32) -> f32 {
    let p = legendre_even(cos);
    (0..coeffs.len().min(5)).map(|j| coeffs[j] * YL0[j] * p[j]).sum()
}

/// A UV (lat/long) unit sphere: vertices + triangle index triples.
fn uv_sphere(n_lat: usize, n_lon: usize) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    use std::f32::consts::PI;
    let mut verts = Vec::with_capacity((n_lat + 1) * n_lon);
    for i in 0..=n_lat {
        let theta = PI * i as f32 / n_lat as f32;
        let (st, ct) = theta.sin_cos();
        for j in 0..n_lon {
            let phi = 2.0 * PI * j as f32 / n_lon as f32;
            let (sp, cp) = phi.sin_cos();
            verts.push(Vec3::new(st * cp, st * sp, ct));
        }
    }
    let idx = |i: usize, j: usize| i * n_lon + (j % n_lon);
    let mut faces = Vec::with_capacity(n_lat * n_lon * 2);
    for i in 0..n_lat {
        for j in 0..n_lon {
            let (a, b, c, d) = (idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1));
            faces.push([a, b, d]);
            faces.push([a, d, c]);
        }
    }
    (verts, faces)
}

/// Draw the response panel for the active ODX scene, if it carries a
/// continuous-b response.
pub fn show_response_panel(
    ctx: &egui::Context,
    state: &mut ResponsePanelState,
    scene: Option<&OdxScene>,
) {
    let Some(resp) = scene.and_then(|s| s.contb_response()) else {
        return;
    };
    let bmax = resp.b_max().max(resp.step);
    if !state.initialized {
        state.b = (bmax * 0.3).clamp(resp.step, bmax);
        state.initialized = true;
    }
    state.b = state.b.clamp(0.0, bmax);

    egui::Window::new("Continuous-b response")
        .default_width(720.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("b");
                ui.add(
                    egui::Slider::new(&mut state.b, 0.0..=bmax)
                        .suffix(" s/mm\u{b2}")
                        .step_by(resp.step),
                );
            });

            let wm = resp.wm_at(state.b);
            let coeff_str = wm
                .iter()
                .enumerate()
                .map(|(j, v)| format!("r{}={:+.3}", 2 * j, v))
                .collect::<Vec<_>>()
                .join("  ");
            ui.monospace(format!("WM   {coeff_str}"));
            ui.monospace(format!(
                "GM   {:+.3}      CSF {:+.3}      (coeff = S/S0 \u{b7} \u{221a}4\u{03c0})",
                resp.gm_at(state.b),
                resp.csf_at(state.b),
            ));
            ui.horizontal_wrapped(|ui| {
                for j in 0..resp.n_wm_orders() {
                    ui.colored_label(WM_COLORS[j.min(WM_COLORS.len() - 1)], format!("WM r{}", 2 * j));
                }
                ui.colored_label(GM_COLOR, "GM");
                ui.colored_label(CSF_COLOR, "CSF");
            });
            ui.separator();

            ui.horizontal_top(|ui| {
                draw_glyph(ui, &wm, state.b, state);
                draw_plot(ui, resp, state.b, bmax);
            });
        });
}

/// Software-render the WM response as a shaded 3D glyph (drag to rotate).
fn draw_glyph(ui: &mut egui::Ui, wm_coeffs: &[f32], b: f64, state: &mut ResponsePanelState) {
    let side = 300.0;
    let (rect, drag) = ui.allocate_exact_size(vec2(side, side), Sense::drag());
    if drag.dragged() {
        let d = drag.drag_delta();
        state.yaw += d.x * 0.01;
        state.pitch = (state.pitch + d.y * 0.01).clamp(-1.55, 1.55);
    }
    let painter = ui.painter_at(rect);
    // Opaque backdrop: gives the glyph a viewport-like frame and prevents the
    // 3D scene behind a translucent window from bleeding through.
    painter.rect_filled(rect, egui::CornerRadius::ZERO, ui.visuals().extreme_bg_color);
    let center = rect.center();

    let (verts, faces) = uv_sphere(28, 56);
    let amps: Vec<f32> = verts.iter().map(|v| wm_amplitude(wm_coeffs, v.z).max(0.0)).collect();
    let max_r = amps.iter().cloned().fold(1e-6_f32, f32::max);
    let dscale = (side * 0.42) / max_r;

    let rot = Mat3::from_rotation_x(state.pitch) * Mat3::from_rotation_y(state.yaw);
    let view: Vec<Vec3> = verts.iter().zip(&amps).map(|(v, &a)| rot * (*v * a)).collect();
    let light = Vec3::new(0.35, 0.45, 1.0).normalize();

    let project = |p: Vec3| pos2(center.x + p.x * dscale, center.y - p.y * dscale);
    let mut tris: Vec<(f32, [Pos2; 3], Color32)> = Vec::with_capacity(faces.len());
    for f in &faces {
        let (p0, p1, p2) = (view[f[0]], view[f[1]], view[f[2]]);
        let nrm = (p1 - p0).cross(p2 - p0);
        let nl = nrm.length();
        if nl < 1e-9 {
            continue; // drop the degenerate pole-cap triangles
        }
        let mut n = nrm / nl;
        if n.z < 0.0 {
            n = -n;
        }
        let shade = n.dot(light).max(0.0) * 0.8 + 0.2;
        // Direction-encoded (DEC) base color from the undeformed surface point.
        let dir = (verts[f[0]] + verts[f[1]] + verts[f[2]]).normalize_or_zero();
        let col = Color32::from_rgb(
            (dir.x.abs() * 255.0 * shade) as u8,
            (dir.y.abs() * 255.0 * shade) as u8,
            (dir.z.abs() * 255.0 * shade) as u8,
        );
        let depth = (p0.z + p1.z + p2.z) / 3.0;
        tris.push((depth, [project(p0), project(p1), project(p2)], col));
    }
    // Painter's algorithm: far (smaller view-z) first.
    tris.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Emit ONE mesh rather than N anti-aliased polygons — individually feathered
    // triangles darken at every shared edge and read as spurious lines; a single
    // raw-triangle mesh has no internal seams.
    let mut mesh = egui::epaint::Mesh::default();
    for (_, pts, col) in &tris {
        let base = mesh.vertices.len() as u32;
        for p in pts {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: *p,
                uv: egui::epaint::WHITE_UV,
                color: *col,
            });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    painter.add(Shape::mesh(mesh));

    let font = FontId::proportional(11.0);
    let faint = ui.visuals().weak_text_color();
    painter.text(
        rect.left_top() + vec2(4.0, 3.0),
        Align2::LEFT_TOP,
        format!("WM response @ b={b:.0}"),
        font.clone(),
        ui.visuals().text_color(),
    );
    painter.text(
        rect.left_bottom() + vec2(4.0, -3.0),
        Align2::LEFT_BOTTOM,
        "drag to rotate \u{2022} fibre axis = z",
        FontId::proportional(9.0),
        faint,
    );
}

/// Paint the `R_l(b)` curves + a vertical marker at `b_cursor`.
fn draw_plot(ui: &mut egui::Ui, resp: &ContbResponse, b_cursor: f64, bmax: f64) {
    let width = ui.available_width().max(200.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, 300.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let plot = egui::Rect::from_min_max(
        pos2(rect.left() + 48.0, rect.top() + 8.0),
        pos2(rect.right() - 8.0, rect.bottom() - 22.0),
    );

    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut acc = |v: f32| {
        ymin = ymin.min(v);
        ymax = ymax.max(v);
    };
    for row in &resp.wm {
        for &v in row {
            acc(v);
        }
    }
    resp.gm.iter().for_each(|&v| acc(v));
    resp.csf.iter().for_each(|&v| acc(v));
    if !ymin.is_finite() || !ymax.is_finite() {
        return;
    }
    let span = (ymax - ymin).max(1e-6);
    ymin -= 0.05 * span;
    ymax += 0.05 * span;
    let x_max = bmax.max(1.0);

    let to_screen = |b: f64, y: f64| -> Pos2 {
        let fx = (b / x_max).clamp(0.0, 1.0) as f32;
        let fy = ((y - ymin as f64) / (ymax - ymin) as f64).clamp(0.0, 1.0) as f32;
        pos2(plot.left() + fx * plot.width(), plot.bottom() - fy * plot.height())
    };

    let axis = ui.visuals().widgets.noninteractive.fg_stroke.color.gamma_multiply(0.5);
    let grid = Stroke::new(1.0, axis.gamma_multiply(0.4));
    let font = FontId::proportional(10.0);
    let text_col = ui.visuals().text_color();

    painter.line_segment([plot.left_top(), plot.left_bottom()], Stroke::new(1.0, axis));
    painter.line_segment([plot.left_bottom(), plot.right_bottom()], Stroke::new(1.0, axis));
    if ymin < 0.0 && ymax > 0.0 {
        let y0 = to_screen(0.0, 0.0).y;
        painter.line_segment(
            [pos2(plot.left(), y0), pos2(plot.right(), y0)],
            Stroke::new(1.0, axis.gamma_multiply(0.7)),
        );
    }
    let mut bt = 0.0;
    while bt <= x_max + 1.0 {
        let x = to_screen(bt, ymin as f64).x;
        painter.line_segment([pos2(x, plot.bottom()), pos2(x, plot.bottom() + 3.0)], Stroke::new(1.0, axis));
        painter.text(pos2(x, plot.bottom() + 4.0), Align2::CENTER_TOP, format!("{bt:.0}"), font.clone(), text_col);
        painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], grid);
        bt += 1000.0;
    }
    for yv in [ymin as f64, 0.0, ymax as f64] {
        if yv < ymin as f64 || yv > ymax as f64 {
            continue;
        }
        let p = to_screen(0.0, yv);
        painter.text(pos2(plot.left() - 4.0, p.y), Align2::RIGHT_CENTER, format!("{yv:.2}"), font.clone(), text_col);
    }
    painter.text(pos2(plot.center().x, rect.bottom() - 10.0), Align2::CENTER_CENTER, "b (s/mm\u{b2})", font.clone(), text_col);

    let step = resp.step;
    let line = |vals: &dyn Fn(usize) -> f64, n: usize, color: Color32| {
        if n < 2 {
            return;
        }
        let pts: Vec<Pos2> = (0..n).map(|k| to_screen(k as f64 * step, vals(k))).collect();
        painter.add(Shape::line(pts, Stroke::new(1.6, color)));
    };
    for j in 0..resp.n_wm_orders() {
        let color = WM_COLORS[j.min(WM_COLORS.len() - 1)];
        line(&|k| resp.wm[k].get(j).copied().unwrap_or(0.0) as f64, resp.wm.len(), color);
    }
    line(&|k| resp.gm[k] as f64, resp.gm.len(), GM_COLOR);
    line(&|k| resp.csf[k] as f64, resp.csf.len(), CSF_COLOR);

    let xc = to_screen(b_cursor, 0.0).x;
    painter.line_segment(
        [pos2(xc, plot.top()), pos2(xc, plot.bottom())],
        Stroke::new(1.5, Color32::from_rgb(0xff, 0xd0, 0x20)),
    );
    painter.text(pos2(xc + 3.0, plot.top() + 2.0), Align2::LEFT_TOP, format!("b={b_cursor:.0}"), font, text_col);
}

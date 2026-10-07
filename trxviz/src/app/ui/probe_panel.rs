//! "Probe" — quantitative inspection of the voxel under the cursor.
//!
//! Hovering a viewport shows a live readout (world mm, voxel ijk, volume
//! values in real units, fixel summary); clicking pins the voxel and opens
//! a detail window with the full fixel table, SH coefficients + evaluated
//! peak amplitude, and a software-rendered ODF/SH profile glyph (same
//! painter technique as the continuous-b response panel). Named "Probe"
//! because `WorkspacePane::Inspector` is already the node-parameter
//! inspector. Streamline probing is future work.

use egui::{Align2, Color32, FontId, Pos2, Sense, Shape, pos2, vec2};
use glam::{Mat3, Mat4, Vec3};
use trxviz_core::data::loaded_files::LoadedNifti;
use trxviz_core::data::odx_data::OdxScene;
use trxviz_core::data::probe::{OdxProbeReport, probe_odx};
use trxviz_core::renderer::picking::{pick_ray, ray_plane_intersect};
use trxviz_core::workflow::{SceneFramePlan, VolumeBacking, VolumeDrawPlan};

/// SH render-mesh detail for the pinned glyph (cached per scene).
const PROBE_SH_DETAIL: u32 = 3;

/// A hover hit: the world position under the cursor on a slice plane.
#[derive(Clone, Copy, Debug)]
pub struct ProbeHover {
    pub world: Vec3,
}

/// Persistent probe state on the app.
pub struct ProbeState {
    /// Pinned world position (set by clicking a viewport).
    pub pinned: Option<Vec3>,
    glyph_yaw: f32,
    glyph_pitch: f32,
    /// Full report for the pinned voxel, keyed by (scene ptr, ijk) so the
    /// SH evaluation runs once per pin, not per frame.
    cached: Option<(ProbeCacheKey, OdxProbeReport)>,
}

type ProbeCacheKey = (usize, [i64; 3]);

impl Default for ProbeState {
    fn default() -> Self {
        Self {
            pinned: None,
            glyph_yaw: 0.6,
            glyph_pitch: 0.35,
            cached: None,
        }
    }
}

/// A slice plane the cursor ray can land on.
#[derive(Clone, Copy, Debug)]
pub struct PlaneCandidate {
    pub normal: Vec3,
    pub center: Vec3,
}

/// Cursor position → world position on the nearest candidate plane.
/// `bounds` (a volume's `voxel_to_ras` + dims) culls hits outside the
/// volume so the 3D view doesn't probe empty space past the slice quads.
pub fn pick_at_pointer(
    rect: egui::Rect,
    pointer: Pos2,
    view_proj: Mat4,
    planes: &[PlaneCandidate],
    bounds: Option<(Mat4, [usize; 3])>,
) -> Option<ProbeHover> {
    if rect.width() < 1.0 || rect.height() < 1.0 {
        return None;
    }
    let ndc = glam::Vec2::new(
        (pointer.x - rect.left()) / rect.width() * 2.0 - 1.0,
        1.0 - (pointer.y - rect.top()) / rect.height() * 2.0,
    );
    let ray = pick_ray(view_proj, ndc)?;
    let mut best: Option<(f32, ProbeHover)> = None;
    for plane in planes {
        let Some(world) = ray_plane_intersect(&ray, plane.normal, plane.center) else {
            continue;
        };
        if let Some((voxel_to_ras, dims)) = bounds {
            let v = voxel_to_ras.inverse() * world.extend(1.0);
            let inside = [v.x, v.y, v.z]
                .iter()
                .zip(dims.iter())
                .all(|(&c, &d)| c > -0.5 && c < d as f32 - 0.5);
            if !inside {
                continue;
            }
        }
        let t = (world - ray.origin).length();
        if best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, ProbeHover { world }));
        }
    }
    best.map(|(_, hover)| hover)
}

/// Volume values under a world point, in real units: `(label, value)` for
/// each visible file volume and each in-memory / composite layer.
fn volume_values_at(
    world: Vec3,
    nifti_files: &[LoadedNifti],
    plan: &SceneFramePlan,
) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for nf in nifti_files.iter().filter(|nf| nf.visible) {
        if let Some(v) = nf.volume.sample_world_real(world) {
            out.push((nf.name.clone(), v));
        }
    }
    for draw in plan.draws.of_type::<VolumeDrawPlan>() {
        match &draw.source {
            VolumeBacking::File(_) => {}
            VolumeBacking::InMemory { scalars, .. } => {
                if let Some(v) = scalars.sample_ras(world) {
                    let name = if scalars.metadata.map_name.is_empty() {
                        "volume".to_string()
                    } else {
                        scalars.metadata.map_name.clone()
                    };
                    out.push((name, v));
                }
            }
            VolumeBacking::Composite { stack, .. } => {
                for (scalars, cfg) in stack.layers.iter().filter(|(_, cfg)| cfg.enabled) {
                    if let Some(v) = scalars.sample_ras(world) {
                        let name = if !cfg.legend_label.is_empty() {
                            cfg.legend_label.clone()
                        } else if !scalars.metadata.map_name.is_empty() {
                            scalars.metadata.map_name.clone()
                        } else {
                            "layer".to_string()
                        };
                        out.push((name, v));
                    }
                }
            }
        }
    }
    out
}

/// Paint the live hover readout into the hovered viewport's corner.
pub fn paint_hover_readout(
    ui: &egui::Ui,
    rect: egui::Rect,
    hover: &ProbeHover,
    scene: Option<&OdxScene>,
    nifti_files: &[LoadedNifti],
    plan: &SceneFramePlan,
) {
    let w = hover.world;
    let mut lines = vec![format!("RAS {:+.1} {:+.1} {:+.1} mm", w.x, w.y, w.z)];

    // Cheap ODX probe: no SH/ODF evaluation on hover.
    let report = scene.and_then(|s| probe_odx(s, w, PROBE_SH_DETAIL, false));
    if let Some(report) = &report {
        match report.compact_row {
            Some(row) => lines.push(format!(
                "voxel [{} {} {}]  (row {row})",
                report.ijk[0], report.ijk[1], report.ijk[2]
            )),
            None => lines.push(format!(
                "voxel [{} {} {}]  (outside mask)",
                report.ijk[0], report.ijk[1], report.ijk[2]
            )),
        }
        for (name, value) in &report.dpv {
            lines.push(format!("{name}: {value:.4}"));
        }
        if !report.fixels.is_empty() {
            let summary = report
                .primary_fixel_metric
                .as_deref()
                .map(|metric| {
                    let vals: Vec<String> = report
                        .fixels
                        .iter()
                        .filter_map(|f| {
                            f.scalars
                                .iter()
                                .find(|(n, _)| n == metric)
                                .map(|(_, v)| format!("{v:.3}"))
                        })
                        .collect();
                    format!("  {metric} {}", vals.join(" "))
                })
                .unwrap_or_default();
            lines.push(format!("{} fixel(s){summary}", report.fixels.len()));
        }
    }
    for (name, value) in volume_values_at(w, nifti_files, plan) {
        lines.push(format!("{name}: {value:.4}"));
    }
    lines.push("click to pin".to_string());

    let font = FontId::monospace(10.0);
    let painter = ui.painter_at(rect);
    let line_height = 13.0;
    let pad = 5.0;
    let widest = lines.iter().map(|l| l.len()).max().unwrap_or(0) as f32 * 6.1;
    let size = vec2(
        widest + 2.0 * pad,
        lines.len() as f32 * line_height + 2.0 * pad,
    );
    let origin = rect.left_bottom() + vec2(6.0, -6.0 - size.y);
    let bg = egui::Rect::from_min_size(origin, size);
    painter.rect_filled(
        bg,
        egui::CornerRadius::same(3),
        Color32::from_black_alpha(160),
    );
    for (i, line) in lines.iter().enumerate() {
        let color = if i + 1 == lines.len() {
            Color32::from_gray(150)
        } else {
            Color32::from_gray(230)
        };
        painter.text(
            bg.left_top() + vec2(pad, pad + i as f32 * line_height),
            Align2::LEFT_TOP,
            line,
            font.clone(),
            color,
        );
    }
}

/// The pinned detail window. Self-hiding: no-op until a click pins a voxel.
pub fn show_probe_panel(
    ctx: &egui::Context,
    state: &mut ProbeState,
    scene: Option<&OdxScene>,
    nifti_files: &[LoadedNifti],
    plan: &SceneFramePlan,
) {
    let Some(world) = state.pinned else {
        return;
    };

    // Compute (or reuse) the full report for the pinned voxel.
    let report = scene.and_then(|s| {
        let lookup = s.source_lookup();
        let v = lookup.world_to_voxel([world.x as f64, world.y as f64, world.z as f64]);
        let ijk = [
            v[0].round() as i64,
            v[1].round() as i64,
            v[2].round() as i64,
        ];
        let key: ProbeCacheKey = (s as *const OdxScene as usize, ijk);
        if state.cached.as_ref().map(|(k, _)| *k) != Some(key) {
            let report = probe_odx(s, world, PROBE_SH_DETAIL, true)?;
            state.cached = Some((key, report));
        }
        state.cached.as_ref().map(|(_, r)| r.clone())
    });

    let mut open = true;
    egui::Window::new("Probe")
        .default_width(420.0)
        .resizable(true)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.monospace(format!(
                "RAS {:+.2} {:+.2} {:+.2} mm",
                world.x, world.y, world.z
            ));

            let volumes = volume_values_at(world, nifti_files, plan);
            if !volumes.is_empty() {
                ui.separator();
                ui.strong("Volumes");
                egui::Grid::new("probe_volumes")
                    .striped(true)
                    .show(ui, |ui| {
                        for (name, value) in &volumes {
                            ui.label(name);
                            ui.monospace(format!("{value:.4}"));
                            ui.end_row();
                        }
                    });
            }

            let Some(report) = report else {
                if scene.is_some() {
                    ui.separator();
                    ui.weak("Pinned point is outside the ODX volume.");
                }
                return;
            };

            ui.separator();
            match report.compact_row {
                Some(row) => ui.monospace(format!(
                    "ODX voxel [{} {} {}]  compact row {row}",
                    report.ijk[0], report.ijk[1], report.ijk[2]
                )),
                None => ui.monospace(format!(
                    "ODX voxel [{} {} {}]  — outside mask",
                    report.ijk[0], report.ijk[1], report.ijk[2]
                )),
            };

            if !report.dpv.is_empty() {
                ui.separator();
                ui.strong("Per-voxel scalars");
                egui::Grid::new("probe_dpv").striped(true).show(ui, |ui| {
                    for (name, value) in &report.dpv {
                        ui.label(name);
                        ui.monospace(format!("{value:.5}"));
                        ui.end_row();
                    }
                });
            }

            if !report.fixels.is_empty() {
                ui.separator();
                ui.strong(format!("Fixels ({})", report.fixels.len()));
                show_fixel_table(ui, &report, plan);
            }

            if let Some(sh) = &report.sh {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong("SH coefficients");
                    ui.weak(format!("basis: {}", sh.basis));
                    if sh.clamped_nonnegative {
                        ui.weak("(amplitudes clamped ≥ 0)");
                    }
                });
                if let Some(peak) = sh.peak_amplitude {
                    ui.monospace(format!("peak amplitude {peak:.5}"));
                }
                ui.collapsing(format!("{} coefficients", sh.coeffs.len()), |ui| {
                    let text = sh
                        .coeffs
                        .chunks(6)
                        .map(|chunk| {
                            chunk
                                .iter()
                                .map(|c| format!("{c:+.4}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    ui.monospace(text);
                });
            }

            // Profile glyph: SH render mesh when SH data exists, else the
            // stored ODF samples on the file's sphere.
            if let Some(scene) = scene {
                draw_profile_glyph(ui, state, scene, &report);
            }
        });
    if !open {
        state.pinned = None;
        state.cached = None;
    }
}

fn show_fixel_table(ui: &mut egui::Ui, report: &OdxProbeReport, plan: &SceneFramePlan) {
    use trxviz_core::workflow::FixelView;
    // The draw whose scalar/gate currently drives rendering (3D first).
    let active = plan
        .active_fixel_draw(FixelView::ThreeD)
        .or_else(|| plan.active_fixel_draw(FixelView::TwoD));
    let driving_name = active.map(|d| d.field.scalars.name.as_str());
    let gate = active.map(|d| (d.field.scalars.name.clone(), d.opacity_gate));

    let scalar_names: Vec<&str> = report
        .fixels
        .first()
        .map(|f| f.scalars.iter().map(|(n, _)| n.as_str()).collect())
        .unwrap_or_default();
    egui::Grid::new("probe_fixels")
        .striped(true)
        .show(ui, |ui| {
            ui.label("direction");
            for name in &scalar_names {
                if Some(*name) == driving_name {
                    ui.strong(format!("{name} •"));
                } else {
                    ui.label(*name);
                }
            }
            ui.end_row();
            for fixel in &report.fixels {
                // Dim fixels the active opacity gate fades out.
                let gated = gate
                    .as_ref()
                    .and_then(|(name, gate)| {
                        let value = fixel.scalars.iter().find(|(n, _)| n == name)?.1;
                        let (lo, hi) = gate.range;
                        let alpha = if value <= lo {
                            gate.below
                        } else if value >= hi {
                            gate.above
                        } else {
                            let t = (value - lo) / (hi - lo).max(1e-6);
                            gate.below + (gate.above - gate.below) * t
                        };
                        Some(alpha <= 0.1)
                    })
                    .unwrap_or(false);
                let paint = |ui: &mut egui::Ui, text: String| {
                    if gated {
                        ui.weak(text);
                    } else {
                        ui.monospace(text);
                    }
                };
                let [x, y, z] = fixel.direction;
                paint(ui, format!("{x:+.3} {y:+.3} {z:+.3}"));
                for name in &scalar_names {
                    let value = fixel
                        .scalars
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, v)| *v);
                    match value {
                        Some(v) => paint(ui, format!("{v:.4}")),
                        None => paint(ui, "—".to_string()),
                    }
                }
                ui.end_row();
            }
        });
}

/// The pinned voxel's angular profile, software-rendered like the
/// continuous-b response glyph (drag to rotate).
fn draw_profile_glyph(
    ui: &mut egui::Ui,
    state: &mut ProbeState,
    scene: &OdxScene,
    report: &OdxProbeReport,
) {
    // SH first (evaluated on the render icosphere); else stored ODF samples.
    let (verts, indices, amps, label) = if let (Some(sh), Some(mesh)) =
        (&report.sh, scene.sh_render_mesh(PROBE_SH_DETAIL))
    {
        let amps = mesh.evaluator().apply_row(&sh.coeffs);
        (
            mesh.vertices().to_vec(),
            mesh.indices().to_vec(),
            amps,
            "SH profile".to_string(),
        )
    } else if let (Some(odf), Some((verts, indices))) = (&report.odf, scene.odf_render_geometry()) {
        // Hemisphere files store half the sphere; the render geometry is
        // the mirrored full set, so the row duplicates onto the mirror.
        let mut amps = odf.amplitudes.clone();
        if odf.hemisphere && verts.len() == 2 * amps.len() {
            amps.extend_from_within(..);
        }
        if amps.len() != verts.len() {
            return;
        }
        let label = if odf.hemisphere {
            "ODF profile (hemisphere data, mirrored)".to_string()
        } else {
            "ODF profile".to_string()
        };
        (verts.to_vec(), indices.to_vec(), amps, label)
    } else {
        return;
    };

    ui.separator();
    ui.strong(&label);
    draw_amplitude_glyph(
        ui,
        &verts,
        &indices,
        &amps,
        &mut state.glyph_yaw,
        &mut state.glyph_pitch,
    );
}

/// Software-render an amplitude-deformed sphere mesh into the painter.
/// Same technique as `response_panel::draw_glyph`: painter's-algorithm
/// depth sort, direction-encoded colors, one seam-free `epaint::Mesh`.
fn draw_amplitude_glyph(
    ui: &mut egui::Ui,
    verts: &[[f32; 3]],
    indices: &[u32],
    amps: &[f32],
    yaw: &mut f32,
    pitch: &mut f32,
) {
    let side = 260.0;
    let (rect, drag) = ui.allocate_exact_size(vec2(side, side), Sense::drag());
    if drag.dragged() {
        let d = drag.drag_delta();
        *yaw += d.x * 0.01;
        *pitch = (*pitch + d.y * 0.01).clamp(-1.55, 1.55);
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(
        rect,
        egui::CornerRadius::ZERO,
        ui.visuals().extreme_bg_color,
    );
    let center = rect.center();

    let max_r = amps.iter().cloned().fold(1e-6_f32, f32::max);
    let dscale = (side * 0.42) / max_r;
    let rot = Mat3::from_rotation_x(*pitch) * Mat3::from_rotation_y(*yaw);
    let view: Vec<Vec3> = verts
        .iter()
        .zip(amps)
        .map(|(v, &a)| rot * (Vec3::from_array(*v) * a.max(0.0)))
        .collect();
    let light = Vec3::new(0.35, 0.45, 1.0).normalize();
    let project = |p: Vec3| pos2(center.x + p.x * dscale, center.y - p.y * dscale);

    let mut tris: Vec<(f32, [Pos2; 3], Color32)> = Vec::with_capacity(indices.len() / 3);
    for f in indices.chunks_exact(3) {
        let (i0, i1, i2) = (f[0] as usize, f[1] as usize, f[2] as usize);
        let (p0, p1, p2) = (view[i0], view[i1], view[i2]);
        let nrm = (p1 - p0).cross(p2 - p0);
        let nl = nrm.length();
        if nl < 1e-9 {
            continue;
        }
        let mut n = nrm / nl;
        if n.z < 0.0 {
            n = -n;
        }
        let shade = n.dot(light).max(0.0) * 0.8 + 0.2;
        let dir = (Vec3::from_array(verts[i0])
            + Vec3::from_array(verts[i1])
            + Vec3::from_array(verts[i2]))
        .normalize_or_zero();
        let col = Color32::from_rgb(
            (dir.x.abs() * 255.0 * shade) as u8,
            (dir.y.abs() * 255.0 * shade) as u8,
            (dir.z.abs() * 255.0 * shade) as u8,
        );
        let depth = (p0.z + p1.z + p2.z) / 3.0;
        tris.push((depth, [project(p0), project(p1), project(p2)], col));
    }
    tris.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
    painter.text(
        rect.left_bottom() + vec2(4.0, -3.0),
        Align2::LEFT_BOTTOM,
        "drag to rotate",
        FontId::proportional(9.0),
        ui.visuals().weak_text_color(),
    );
}

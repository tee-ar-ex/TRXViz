//! Colorbar model: which color-mapped quantities are on screen and what
//! range they span. This module is egui-free — the GUI paints the bars,
//! and a headless rasterizer can consume the same specs later.
//!
//! Bars show the raw colormap ramp. The 3D shaders apply post-processing
//! (exposure / contrast / depth fade) *after* colormapping, so on-screen
//! pixels can differ slightly from the bar that labels them.

use std::collections::HashMap;

use crate::data::loaded_files::FileId;
use crate::data::trx_data::ColorMode;
use crate::renderer::colormap::{ColormapRef, glyph_colormap_code};
use crate::workflow::{
    FixelView, GlyphColormap, SceneFramePlan, SurfaceDrawPlan, VolumeBacking, VolumeDrawPlan,
    WorkflowNodeUuid,
};

/// Which viewport the legend is for. The collector mirrors the renderer's
/// per-view selection: the 3D viewport draws `active_fixel_draw(ThreeD)`
/// plus glyphs and surfaces; slice viewports draw `active_fixel_draw(TwoD)`
/// and no glyphs/surfaces. Volumes and streamlines render in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegendView {
    ThreeD,
    Slice,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColorbarSpec {
    pub title: String,
    pub cmap: ColormapRef,
    /// Value at the bottom of the ramp, in the quantity's real units.
    pub min_value: f32,
    /// Value at the top of the ramp.
    pub max_value: f32,
    /// Tick positions as `(t in [0,1], label)`, bottom to top.
    pub ticks: Vec<(f32, String)>,
    /// Extra context, e.g. "normalized".
    pub note: Option<String>,
}

/// A loaded file-backed volume the collector can label: its draw-plan id,
/// display name, and real intensity range (`NiftiVolume::value_range`).
#[derive(Clone, Debug)]
pub struct FileVolumeInfo {
    pub id: FileId,
    pub name: String,
    pub value_range: (f32, f32),
}

/// Borrowed inputs for [`collect_colorbar_specs`], assembled by the caller
/// (the GUI today; headless later).
pub struct ColorbarInputs<'a> {
    pub plan: &'a SceneFramePlan,
    /// Which viewport this legend labels; bars only appear for layers that
    /// viewport actually renders.
    pub view: LegendView,
    pub file_volumes: &'a [FileVolumeInfo],
    /// Real amplitude normalization used by the glyph LUT this frame.
    /// 1.0 means amplitudes reach the shader already normalized.
    pub glyph_amp_norm: f32,
    /// Effective range per streamline node whose flow uses
    /// `scalar_auto_range` (2–98 percentile, memoized by the caller).
    /// Missing entries fall back to the flow's explicit range.
    pub streamline_auto_ranges: &'a HashMap<WorkflowNodeUuid, (f32, f32)>,
}

/// Everything the legend overlay should draw: full colorbars plus
/// name-only entries (layers whose colors are baked per-vertex and have no
/// single ramp, e.g. surface overlay stacks).
#[derive(Clone, Debug, Default)]
pub struct LegendModel {
    pub bars: Vec<ColorbarSpec>,
    pub titles: Vec<String>,
}

pub fn collect_colorbar_specs(inputs: &ColorbarInputs<'_>) -> LegendModel {
    let mut model = LegendModel::default();
    let plan = inputs.plan;

    // Fixels: exactly the draw this viewport renders — the same
    // single-active selection the GPU upload path uses (3D viewport uses
    // the ThreeD draw, slice viewports the TwoD draw).
    let fixel_view = match inputs.view {
        LegendView::ThreeD => FixelView::ThreeD,
        LegendView::Slice => FixelView::TwoD,
    };
    if let Some(draw) = plan.active_fixel_draw(fixel_view)
        && draw.visible
        && draw.colormap_code != 0
    {
        let (lo, hi) = draw.scalar_range;
        push_spec(
            &mut model.bars,
            ColorbarSpec {
                title: draw.field.scalars.name.clone(),
                cmap: ColormapRef::FixelGlyph(draw.colormap_code),
                min_value: lo,
                max_value: hi,
                ticks: nice_ticks(lo, hi, 4),
                note: None,
            },
        );
    }

    // ODF/SH glyphs: amplitude ramp, 3D viewport only (the slice-view
    // callback passes `show_odx_glyphs: false`). `Directional` has no
    // ramp; there is no user-set range — the LUT maps amp / amp_norm.
    if inputs.view == LegendView::ThreeD
        && let Some(draw) = plan.active_odf_glyph_draw()
        && draw.visible
        && draw.vertex_colormap != GlyphColormap::Directional
    {
        let amp_norm = if inputs.glyph_amp_norm > 0.0 {
            inputs.glyph_amp_norm
        } else {
            1.0
        };
        push_spec(
            &mut model.bars,
            ColorbarSpec {
                title: "ODF amplitude".to_string(),
                cmap: ColormapRef::FixelGlyph(glyph_colormap_code(draw.vertex_colormap)),
                min_value: 0.0,
                max_value: amp_norm,
                ticks: nice_ticks(0.0, amp_norm, 4),
                note: (amp_norm == 1.0).then(|| "normalized".to_string()),
            },
        );
    }

    // Volumes. File-backed slices are normalized [0,1] on the GPU, so the
    // window maps back through the file's real range; in-memory and
    // composite layers window raw values.
    for draw in plan.draws.of_type::<VolumeDrawPlan>() {
        match &draw.source {
            VolumeBacking::File(id) => {
                let Some(info) = inputs.file_volumes.iter().find(|v| v.id == *id) else {
                    continue;
                };
                let (vlo, vhi) = info.value_range;
                let span = vhi - vlo;
                let lo = vlo + (draw.window_center - draw.window_width * 0.5) * span;
                let hi = vlo + (draw.window_center + draw.window_width * 0.5) * span;
                push_spec(
                    &mut model.bars,
                    ColorbarSpec {
                        title: info.name.clone(),
                        cmap: ColormapRef::Volume(draw.colormap),
                        min_value: lo,
                        max_value: hi,
                        ticks: nice_ticks(lo, hi, 4),
                        note: None,
                    },
                );
            }
            VolumeBacking::InMemory { scalars, .. } => {
                let lo = draw.window_center - draw.window_width * 0.5;
                let hi = draw.window_center + draw.window_width * 0.5;
                let title = if scalars.metadata.map_name.is_empty() {
                    "Volume".to_string()
                } else {
                    scalars.metadata.map_name.clone()
                };
                push_spec(
                    &mut model.bars,
                    ColorbarSpec {
                        title,
                        cmap: ColormapRef::Volume(draw.colormap),
                        min_value: lo,
                        max_value: hi,
                        ticks: nice_ticks(lo, hi, 4),
                        note: None,
                    },
                );
            }
            VolumeBacking::Composite { stack, .. } => {
                for (scalars, cfg) in &stack.layers {
                    if !cfg.enabled {
                        continue;
                    }
                    let lo = cfg.window_center - cfg.window_width * 0.5;
                    let hi = cfg.window_center + cfg.window_width * 0.5;
                    let title = if !cfg.legend_label.is_empty() {
                        cfg.legend_label.clone()
                    } else if !scalars.metadata.map_name.is_empty() {
                        scalars.metadata.map_name.clone()
                    } else {
                        "Layer".to_string()
                    };
                    push_spec(
                        &mut model.bars,
                        ColorbarSpec {
                            title,
                            cmap: ColormapRef::Volume(cfg.colormap),
                            min_value: lo,
                            max_value: hi,
                            ticks: nice_ticks(lo, hi, 4),
                            note: None,
                        },
                    );
                }
            }
        }
    }

    // Surfaces: projection maps get a ramp; vertex-RGBA overlay stacks
    // contribute name-only legend entries. 3D viewport only — slice views
    // draw surfaces as intersection outlines with no colormap.
    let surface_draws = (inputs.view == LegendView::ThreeD)
        .then(|| plan.draws.of_type::<SurfaceDrawPlan>())
        .into_iter()
        .flatten();
    for draw in surface_draws {
        if draw.show_projection_map {
            let title = draw
                .legend_labels
                .iter()
                .find(|l| !l.is_empty())
                .cloned()
                .unwrap_or_else(|| "Surface map".to_string());
            push_spec(
                &mut model.bars,
                ColorbarSpec {
                    title,
                    cmap: ColormapRef::Surface(draw.projection_colormap),
                    min_value: draw.range_min,
                    max_value: draw.range_max,
                    ticks: nice_ticks(draw.range_min, draw.range_max, 4),
                    note: None,
                },
            );
        } else {
            for label in &draw.legend_labels {
                if !label.is_empty() && !model.titles.contains(label) {
                    model.titles.push(label.clone());
                }
            }
        }
    }

    // Streamlines: scalar color modes only (directional RGB, group, and
    // uniform colors carry no quantitative ramp).
    for draw in &plan.streamline_draws {
        if !draw.visible {
            continue;
        }
        let title = match &draw.flow.color_mode {
            ColorMode::Dpv(name) | ColorMode::DpvGlow(name) | ColorMode::Dps(name) => name.clone(),
            _ => continue,
        };
        let (lo, hi) = if draw.flow.scalar_auto_range {
            inputs
                .streamline_auto_ranges
                .get(&draw.node_uuid)
                .copied()
                .unwrap_or((draw.flow.scalar_range_min, draw.flow.scalar_range_max))
        } else {
            (draw.flow.scalar_range_min, draw.flow.scalar_range_max)
        };
        push_spec(
            &mut model.bars,
            ColorbarSpec {
                title,
                cmap: ColormapRef::Surface(draw.flow.scalar_colormap),
                min_value: lo,
                max_value: hi,
                ticks: nice_ticks(lo, hi, 4),
                note: draw
                    .flow
                    .scalar_auto_range
                    .then(|| "auto range".to_string()),
            },
        );
    }

    model
}

fn push_spec(bars: &mut Vec<ColorbarSpec>, spec: ColorbarSpec) {
    if !bars.contains(&spec) {
        bars.push(spec);
    }
}

/// "Nice" tick values covering `[min, max]` with roughly `target` ticks,
/// using 1-2-5 stepping. Returns `(t, label)` pairs with `t` normalized to
/// `[0, 1]`, in increasing order. Degenerate or non-finite spans get the
/// two endpoints only.
pub fn nice_ticks(min: f32, max: f32, target: usize) -> Vec<(f32, String)> {
    let span = max - min;
    if !span.is_finite() || span <= 0.0 {
        return vec![
            (0.0, format_value(min, span)),
            (1.0, format_value(max, span)),
        ];
    }
    let target = target.max(2) as f32;
    let raw_step = span / target;
    let mag = 10f32.powf(raw_step.log10().floor());
    let norm = raw_step / mag;
    let step = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    } * mag;

    let mut ticks = Vec::new();
    let first = (min / step).ceil() * step;
    let mut v = first;
    // Guard the loop count: fp accumulation on tiny steps must not spin.
    for _ in 0..64 {
        if v > max + step * 1e-3 {
            break;
        }
        let t = ((v - min) / span).clamp(0.0, 1.0);
        ticks.push((t, format_value(v, step)));
        v += step;
    }
    if ticks.len() < 2 {
        return vec![
            (0.0, format_value(min, span)),
            (1.0, format_value(max, span)),
        ];
    }
    ticks
}

/// Format a tick value with precision matched to the step size.
fn format_value(v: f32, step: f32) -> String {
    if !v.is_finite() {
        return "—".to_string();
    }
    // Snap tiny fp residue (e.g. 1.0000001e-7 from step accumulation) to 0.
    let v = if v.abs() < step.abs() * 1e-3 { 0.0 } else { v };
    let a = v.abs().max(step.abs());
    if a >= 10_000.0 || (a < 0.01 && v != 0.0) {
        format!("{v:.1e}")
    } else {
        let decimals = if step.abs() >= 1.0 {
            0
        } else {
            (-step.abs().log10().floor() as i32).clamp(0, 4) as usize
        };
        format!("{v:.decimals$}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_ticks_simple_span() {
        let ticks = nice_ticks(0.0, 1.0, 4);
        assert!(ticks.len() >= 3, "{ticks:?}");
        assert_eq!(ticks.first().unwrap().1, "0.0");
        assert_eq!(ticks.last().unwrap().1, "1.0");
        for w in ticks.windows(2) {
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn nice_ticks_crossing_zero() {
        let ticks = nice_ticks(-2.5, 7.5, 4);
        assert!(ticks.iter().any(|(_, l)| l == "0"), "{ticks:?}");
        assert!(ticks.iter().all(|(t, _)| (0.0..=1.0).contains(t)));
    }

    #[test]
    fn nice_ticks_degenerate_and_tiny() {
        let ticks = nice_ticks(3.0, 3.0, 4);
        assert_eq!(ticks.len(), 2);
        let ticks = nice_ticks(0.0, 1e-6, 4);
        assert!(ticks.len() >= 2, "{ticks:?}");
        let ticks = nice_ticks(0.0, 250_000.0, 4);
        assert!(ticks.len() >= 3, "{ticks:?}");
    }

    #[test]
    fn collector_volume_bars() {
        use crate::data::cifti::{ScalarKind, ScalarMetadata, VolumeScalars};
        use crate::data::loaded_files::VolumeColormap;
        use crate::workflow::{CompositeVolumeStack, VolumeOverlayLayerConfig};
        use std::sync::Arc;

        let scalars = |name: &str| {
            Arc::new(VolumeScalars {
                dims: [2, 2, 2],
                voxel_to_ras: glam::Mat4::IDENTITY,
                values: vec![0.0; 8],
                kind: ScalarKind::Continuous,
                metadata: ScalarMetadata {
                    map_name: name.to_string(),
                    suggested_range: None,
                    series_index: None,
                    series_value: None,
                    label_table: Vec::new(),
                },
            })
        };

        let mut plan = SceneFramePlan::default();
        // File-backed draw: window maps through the file's real range.
        plan.draws.push(VolumeDrawPlan {
            source: VolumeBacking::File(7),
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            window_center: 0.5,
            window_width: 1.0,
        });
        // Composite with one enabled + one disabled layer.
        let mut on_layer = VolumeOverlayLayerConfig::default();
        on_layer.enabled = true;
        on_layer.legend_label = "FA".to_string();
        on_layer.window_center = 0.4;
        on_layer.window_width = 0.8;
        on_layer.colormap = VolumeColormap::Hot;
        let off_layer = VolumeOverlayLayerConfig::default();
        plan.draws.push(VolumeDrawPlan {
            source: VolumeBacking::Composite {
                handle: 1,
                stack: Arc::new(CompositeVolumeStack {
                    dims: [2, 2, 2],
                    voxel_to_ras: glam::Mat4::IDENTITY,
                    layers: vec![(scalars("fa"), on_layer), (scalars("md"), off_layer)],
                }),
            },
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            window_center: 0.5,
            window_width: 1.0,
        });

        let file_volumes = [FileVolumeInfo {
            id: 7,
            name: "T1w".to_string(),
            value_range: (0.0, 2000.0),
        }];
        let model = collect_colorbar_specs(&ColorbarInputs {
            plan: &plan,
            view: LegendView::ThreeD,
            file_volumes: &file_volumes,
            glyph_amp_norm: 1.0,
            streamline_auto_ranges: &HashMap::new(),
        });

        assert_eq!(model.bars.len(), 2, "{:?}", model.bars);
        let t1 = &model.bars[0];
        assert_eq!(t1.title, "T1w");
        assert_eq!(t1.min_value, 0.0);
        assert_eq!(t1.max_value, 2000.0);
        let fa = &model.bars[1];
        assert_eq!(fa.title, "FA");
        assert_eq!(fa.cmap, ColormapRef::Volume(VolumeColormap::Hot));
        assert!((fa.min_value - 0.0).abs() < 1e-6);
        assert!((fa.max_value - 0.8).abs() < 1e-6);
        assert!(model.titles.is_empty());

        // Volume bars appear in slice viewports too.
        let slice_model = collect_colorbar_specs(&ColorbarInputs {
            plan: &plan,
            view: LegendView::Slice,
            file_volumes: &file_volumes,
            glyph_amp_norm: 1.0,
            streamline_auto_ranges: &HashMap::new(),
        });
        assert_eq!(slice_model.bars.len(), 2);
    }

    /// Regression: 2D and 3D fixel displays with different colormaps must
    /// produce different bars per viewport — the 3D legend shows only the
    /// ThreeD draw's bar and the slice legend only the TwoD draw's,
    /// mirroring `active_fixel_draw`.
    #[test]
    fn collector_fixel_bars_are_per_view() {
        use crate::data::odx_data::{FixelField, FixelScalars, OdxScene};
        use crate::units::Millimeters;
        use crate::workflow::{FixelDrawPlan, OpacityGate};
        use odx_rs::OdxBuilder;
        use std::sync::Arc;

        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [1, 1, 1],
            vec![1u8],
        );
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
        let scene = Arc::new(OdxScene::from_dataset(builder.finalize().unwrap()).unwrap());

        let field = |code: u32, name: &str| {
            let mut scalars = FixelScalars::from_directions(0, &[[1.0, 0.0, 0.0]]);
            scalars.name = name.to_string();
            FixelField {
                source_id: 0,
                scene: scene.clone(),
                scalars,
                colormap_code: code,
                scalar_range: (0.0, 1.0),
            }
        };
        let draw = |view: FixelView, code: u32, name: &str| FixelDrawPlan {
            node_uuid: WorkflowNodeUuid(code as u64),
            view,
            field: field(code, name),
            line_width: 0.006,
            length_scale: 1.0,
            opacity: 1.0,
            offset_from_slice: 0.0,
            slab_thickness_mm: Millimeters(1.0),
            visible: true,
            colormap_code: code,
            scalar_range: (0.0, 1.0),
            opacity_gate: OpacityGate::default(),
        };

        let mut plan = SceneFramePlan::default();
        plan.draws.push(draw(FixelView::TwoD, 6, "afd")); // BWR, pushed first
        plan.draws.push(draw(FixelView::ThreeD, 3, "afd")); // Viridis

        let empty_ranges = HashMap::new();
        let inputs = |view| ColorbarInputs {
            plan: &plan,
            view,
            file_volumes: &[],
            glyph_amp_norm: 1.0,
            streamline_auto_ranges: &empty_ranges,
        };
        let three_d = collect_colorbar_specs(&inputs(LegendView::ThreeD));
        assert_eq!(three_d.bars.len(), 1, "{:?}", three_d.bars);
        assert_eq!(three_d.bars[0].cmap, ColormapRef::FixelGlyph(3));
        let slice = collect_colorbar_specs(&inputs(LegendView::Slice));
        assert_eq!(slice.bars.len(), 1, "{:?}", slice.bars);
        assert_eq!(slice.bars[0].cmap, ColormapRef::FixelGlyph(6));
    }
}

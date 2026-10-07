//! Viewport colorbar/legend overlay: paints the `LegendModel` computed by
//! `trxviz_core::renderer::colorbar` along the right edge of a viewport.

use std::collections::HashMap;

use egui::{Align2, Color32, FontId, Rect, pos2, vec2};
use trxviz_core::data::loaded_files::LoadedNifti;
use trxviz_core::data::trx_data::{ColorMode, scalar_auto_range};
use trxviz_core::renderer::colorbar::{
    ColorbarInputs, FileVolumeInfo, LegendModel, LegendView, collect_colorbar_specs,
};
use trxviz_core::renderer::colormap::colormap_ref_rgb;
use trxviz_core::workflow::{SceneFramePlan, WorkflowNodeUuid};

/// Assemble the frame's legend model. `auto_ranges` memoizes the 2–98
/// percentile computation per streamline node (it sorts every scalar), so
/// it only reruns when the underlying data or field changes.
pub fn build_legend_model(
    plan: &SceneFramePlan,
    view: LegendView,
    nifti_files: &[LoadedNifti],
    glyph_amp_norm: f32,
    auto_ranges: &mut HashMap<WorkflowNodeUuid, (u64, (f32, f32))>,
) -> LegendModel {
    let file_volumes: Vec<FileVolumeInfo> = nifti_files
        .iter()
        .map(|nf| FileVolumeInfo {
            id: nf.id,
            name: nf.name.clone(),
            value_range: nf.volume.value_range,
        })
        .collect();

    let mut resolved: HashMap<WorkflowNodeUuid, (f32, f32)> = HashMap::new();
    for draw in &plan.streamline_draws {
        if !draw.visible || !draw.flow.scalar_auto_range {
            continue;
        }
        let gpu = &draw.flow.dataset.gpu_data;
        let (name, data) = match &draw.flow.color_mode {
            ColorMode::Dpv(name) => (
                name,
                gpu.dpv_data.iter().find(|(n, _)| n == name).map(|(_, d)| d),
            ),
            ColorMode::Dps(name) => (
                name,
                gpu.dps_data.iter().find(|(n, _)| n == name).map(|(_, d)| d),
            ),
            _ => continue,
        };
        let Some(data) = data else { continue };
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (std::sync::Arc::as_ptr(gpu) as usize).hash(&mut h);
            name.hash(&mut h);
            h.finish()
        };
        let range = match auto_ranges.get(&draw.node_uuid) {
            Some((fp, range)) if *fp == fingerprint => *range,
            _ => {
                let range = scalar_auto_range(data);
                auto_ranges.insert(draw.node_uuid, (fingerprint, range));
                range
            }
        };
        resolved.insert(draw.node_uuid, range);
    }

    collect_colorbar_specs(&ColorbarInputs {
        plan,
        view,
        file_volumes: &file_volumes,
        glyph_amp_norm,
        streamline_auto_ranges: &resolved,
    })
}

/// Paint the legend along the right edge of `rect`, top-down. Bars that
/// would overflow the rect are dropped with a "+N more" note.
pub fn paint_colorbars(ui: &egui::Ui, rect: Rect, model: &LegendModel) {
    if model.bars.is_empty() && model.titles.is_empty() {
        return;
    }
    let painter = ui.painter_at(rect);
    let bar_width = 16.0;
    let bar_height = 110.0;
    let margin = 8.0;
    let title_h = 13.0;
    let note_h = 11.0;
    let spacing = 10.0;
    let title_font = FontId::proportional(10.5);
    let tick_font = FontId::monospace(9.0);
    let text_col = Color32::from_gray(235);
    let faint = Color32::from_gray(180);

    // No backdrop box — a 1px dark shadow under each glyph keeps labels
    // readable over bright scene content without occluding it.
    let shadowed_text =
        |pos: egui::Pos2, align: Align2, text: &str, font: FontId, color: Color32| {
            painter.text(
                pos + vec2(1.0, 1.0),
                align,
                text,
                font.clone(),
                Color32::from_black_alpha(200),
            );
            painter.text(pos, align, text, font, color);
        };

    let mut y = rect.top() + margin;
    let bar_x = rect.right() - margin - bar_width;
    let mut dropped = 0usize;

    for spec in &model.bars {
        let note_lines = spec.note.is_some() as usize;
        let block_h = title_h + bar_height + note_lines as f32 * note_h + spacing;
        if y + block_h > rect.bottom() - margin {
            dropped += 1;
            continue;
        }

        // Title above the bar, right-aligned.
        shadowed_text(
            pos2(rect.right() - margin, y),
            Align2::RIGHT_TOP,
            &spec.title,
            title_font.clone(),
            text_col,
        );
        let bar_top = y + title_h;
        let bar = Rect::from_min_size(pos2(bar_x, bar_top), vec2(bar_width, bar_height));

        // Ramp: stacked flat-color quads (t = 1 at the top).
        let steps = 48;
        let mut mesh = egui::epaint::Mesh::default();
        for s in 0..steps {
            let t0 = s as f32 / steps as f32;
            let t1 = (s + 1) as f32 / steps as f32;
            let [r, g, b] = colormap_ref_rgb(spec.cmap, (t0 + t1) * 0.5);
            let col = Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
            let y0 = bar.bottom() - t0 * bar.height();
            let y1 = bar.bottom() - t1 * bar.height();
            let base = mesh.vertices.len() as u32;
            for pos in [
                pos2(bar.left(), y1),
                pos2(bar.right(), y1),
                pos2(bar.right(), y0),
                pos2(bar.left(), y0),
            ] {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos,
                    uv: egui::epaint::WHITE_UV,
                    color: col,
                });
            }
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        painter.add(egui::Shape::mesh(mesh));
        painter.rect_stroke(
            bar,
            egui::CornerRadius::ZERO,
            egui::Stroke::new(1.0, Color32::from_gray(90)),
            egui::StrokeKind::Outside,
        );

        // Ticks to the left of the bar.
        for (t, label) in &spec.ticks {
            let ty = bar.bottom() - t * bar.height();
            painter.line_segment(
                [pos2(bar.left() - 3.0, ty), pos2(bar.left(), ty)],
                egui::Stroke::new(1.0, faint),
            );
            shadowed_text(
                pos2(bar.left() - 5.0, ty),
                Align2::RIGHT_CENTER,
                label,
                tick_font.clone(),
                text_col,
            );
        }

        if let Some(note) = &spec.note {
            shadowed_text(
                pos2(rect.right() - margin, bar.bottom() + 2.0),
                Align2::RIGHT_TOP,
                note,
                FontId::proportional(9.0),
                faint,
            );
        }
        y += block_h;
    }

    // Name-only legend entries (e.g. surface overlay layers).
    for title in &model.titles {
        if y + note_h > rect.bottom() - margin {
            dropped += 1;
            continue;
        }
        shadowed_text(
            pos2(rect.right() - margin, y),
            Align2::RIGHT_TOP,
            title,
            FontId::proportional(9.5),
            faint,
        );
        y += note_h;
    }

    if dropped > 0 {
        shadowed_text(
            pos2(rect.right() - margin, rect.bottom() - margin),
            Align2::RIGHT_BOTTOM,
            &format!("+{dropped} more"),
            FontId::proportional(9.0),
            faint,
        );
    }
}

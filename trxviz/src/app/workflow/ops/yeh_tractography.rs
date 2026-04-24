//! Inspector panel for the Yeh (DSI-Studio) fixel tractography op.
//!
//! Extracted from the giant `edit_node_op` match in `ops/mod.rs` so
//! Yeh-specific slider logic lives next to the Yeh op. The
//! `override_slider` closure handles the "if wired TrackingPlan
//! overrides this field, show a greyed-out slider with the plan's
//! value" pattern — Yeh-specific because the DSI-Studio autotrack
//! sentinels (`0` = random) make this UX particularly important.

use super::{NodeEditorContext, NodeEditorResult};

pub(super) fn draw(
    ui: &mut egui::Ui,
    step_size_mm: &mut f32,
    max_angle_deg: &mut f32,
    min_len_mm: &mut f32,
    max_len_mm: &mut f32,
    fixel_threshold: &mut f32,
    smooth_fraction: &mut f32,
    max_points: &mut u32,
    target_streamlines: &mut u32,
    max_seed_attempts: &mut u32,
    rng_seed: &mut u64,
    ctx: &NodeEditorContext<'_>,
    result: &mut NodeEditorResult,
) {
    if !ctx.overridden_fields.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(220, 180, 96),
            "⚠ Greyed sliders are overridden by the wired TrackingPlan; \
             their displayed value comes from the plan.",
        );
    }

    // For each slider: if the plan overrides this field, show the
    // plan's value in a greyed-out slider; otherwise show the op's
    // own editable slider.
    let override_slider = |ui: &mut egui::Ui,
                           field: &str,
                           value_if_live: &mut f32,
                           range: std::ops::RangeInclusive<f32>,
                           text: &str| {
        if let Some(&plan_value) = ctx.overridden_values.get(field) {
            let mut displayed = plan_value;
            // Extend the slider range if the plan's value falls
            // outside the op's slider bounds, so the thumb is
            // visible rather than clamped to an endpoint.
            let lo = range.start().min(plan_value);
            let hi = range.end().max(plan_value);
            ui.add_enabled(false, egui::Slider::new(&mut displayed, lo..=hi).text(text));
        } else {
            ui.add(egui::Slider::new(value_if_live, range).text(text));
        }
    };

    override_slider(
        ui,
        "step_size_mm",
        step_size_mm,
        0.0..=2.0,
        "Step size mm (0 = random)",
    );
    override_slider(
        ui,
        "max_angle_deg",
        max_angle_deg,
        0.0..=90.0,
        "Max angle ° (0 = random)",
    );
    override_slider(ui, "min_len_mm", min_len_mm, 5.0..=100.0, "Min length (mm)");
    override_slider(
        ui,
        "max_len_mm",
        max_len_mm,
        20.0..=500.0,
        "Max length (mm)",
    );
    override_slider(
        ui,
        "fixel_threshold",
        fixel_threshold,
        0.0..=0.5,
        "Fixel threshold (0 = random)",
    );
    if let Some(&v) = ctx.overridden_values.get("fixel_otsu") {
        ui.small(format!(
            "plan fixel_otsu = {:.4} (random threshold centered on 0.6·this)",
            v
        ));
    }
    override_slider(
        ui,
        "smooth_fraction",
        smooth_fraction,
        0.0..=1.0,
        "Smoothing (1 = random)",
    );
    ui.add(egui::Slider::new(max_points, 50..=2000).text("Max points per streamline"));
    ui.add(egui::Slider::new(target_streamlines, 1_000..=1_000_000).text("Target streamlines"));
    ui.add(egui::Slider::new(max_seed_attempts, 100_000..=10_000_000).text("Max seed attempts"));
    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(rng_seed)
                .speed(1.0)
                .prefix("RNG seed "),
        );
        if ui.button("Randomize").clicked() {
            *rng_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42);
            result.run_expensive_requested = true;
        }
    });
    if ui.button("Run Yeh Tracking").clicked() {
        result.run_expensive_requested = true;
    }
}

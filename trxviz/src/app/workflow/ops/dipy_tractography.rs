//! Inspector panel for the Dipy/GPUStreamlines tractography op.
//!
//! Extracted out of `ops/mod.rs` (Stage 3 of `refactor_gui.md`, scoped
//! to the tractography ops) so the Dipy-specific inspector lives next
//! to the Dipy op. The exhaustive match on `DipyDirectionGetter` inside
//! `draw` means adding a new DG variant is a compile error — the
//! forcing-function that prevents regression of PR 1's bug #1, where
//! `direction_getter: _` silently discarded PTT config.

use crate::app::workflow;

use super::{NodeEditorContext, NodeEditorResult};

pub(super) fn draw(
    ui: &mut egui::Ui,
    node_uuid: workflow::WorkflowNodeUuid,
    step_size_mm: &mut f32,
    max_angle_deg: &mut f32,
    min_len_mm: &mut f32,
    max_len_mm: &mut f32,
    fixel_threshold: &mut f32,
    relative_peak_threshold: &mut f32,
    seeds_per_voxel: &mut u32,
    max_points: &mut u32,
    rng_seed: &mut u64,
    direction_getter: &mut workflow::DipyDirectionGetter,
    ctx: &NodeEditorContext<'_>,
    result: &mut NodeEditorResult,
) {
    // Direction-getter variant picker. We compare the discriminant via
    // a bool rather than selectable_value on the full enum because the
    // PTT variant carries its own inline parameters which we don't want
    // to recreate on every combo-box redraw.
    let is_ptt = matches!(direction_getter, workflow::DipyDirectionGetter::Ptt { .. });
    let mut new_is_ptt = is_ptt;
    egui::ComboBox::from_id_salt(format!("dipy_dg_{}", node_uuid.0))
        .selected_text(if is_ptt {
            "PTT (GPU only)"
        } else {
            "Probabilistic"
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut new_is_ptt, false, "Probabilistic");
            ui.selectable_value(&mut new_is_ptt, true, "PTT (GPU only)");
        });
    if new_is_ptt != is_ptt {
        *direction_getter = if new_is_ptt {
            workflow::DipyDirectionGetter::ptt_default()
        } else {
            workflow::DipyDirectionGetter::Probabilistic
        };
    }

    ui.add(egui::Slider::new(step_size_mm, 0.1..=2.0).text("Step size (mm)"));
    ui.add(egui::Slider::new(max_angle_deg, 10.0..=90.0).text("Max angle (°)"));
    ui.add(egui::Slider::new(min_len_mm, 5.0..=100.0).text("Min length (mm)"));
    ui.add(egui::Slider::new(max_len_mm, 20.0..=500.0).text("Max length (mm)"));
    ui.add(egui::Slider::new(fixel_threshold, 0.0..=0.5).text("Fixel threshold"));
    if let Some(&v) = ctx.overridden_values.get("fixel_otsu") {
        ui.small(format!("plan fixel_otsu = {:.4}", v));
    }
    ui.add(egui::Slider::new(relative_peak_threshold, 0.0..=1.0).text("Relative peak threshold"));
    ui.add(egui::Slider::new(seeds_per_voxel, 1..=10).text("Seeds per voxel"));
    ui.add(egui::Slider::new(max_points, 50..=2000).text("Max points"));
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

    // PTT-specific knobs. Exhaustive match on the enum so adding a new
    // variant is a compile error — this is the structural replacement
    // for PR 1's `direction_getter: _` band-aid.
    match direction_getter {
        workflow::DipyDirectionGetter::Probabilistic => {}
        workflow::DipyDirectionGetter::Ptt {
            probe_length_mm,
            probe_quality,
            probe_radius_mm,
            probe_count,
            max_curvature_per_mm,
            data_support_exponent,
            min_data_support,
            rejection_sampling_max_try,
        } => {
            ui.separator();
            ui.label("PTT probe parameters");
            ui.add(egui::Slider::new(probe_length_mm, 0.1..=2.0).text("Probe length (mm)"));
            ui.add(egui::Slider::new(probe_quality, 1..=16).text("Probe quality"));
            ui.add(egui::Slider::new(probe_radius_mm, 0.0..=2.0).text("Probe radius (mm)"));
            ui.add(egui::Slider::new(probe_count, 1..=8).text("Probe count"));
            ui.add(egui::Slider::new(max_curvature_per_mm, 0.0..=2.0).text("Max curvature (1/mm)"));
            ui.add(
                egui::Slider::new(data_support_exponent, 0.25..=4.0).text("Data support exponent"),
            );
            ui.add(egui::Slider::new(min_data_support, 0.0..=1.0).text("Min data support"));
            ui.add(
                egui::Slider::new(rejection_sampling_max_try, 10..=500)
                    .text("Rejection sampling max try"),
            );
        }
    }

    if ui.button("Run Tractography").clicked() {
        result.run_expensive_requested = true;
    }
}

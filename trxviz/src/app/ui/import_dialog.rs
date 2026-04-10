use std::path::PathBuf;

use trx_rs::{Format, VtkCoordinateMode};

use crate::app::state::ImportDialogState;

pub struct ImportDialogAction {
    pub import_requested: bool,
}

pub fn show_import_dialog(
    ctx: &egui::Context,
    state: &mut ImportDialogState,
) -> ImportDialogAction {
    let mut action = ImportDialogAction {
        import_requested: false,
    };

    if !state.open {
        return action;
    }

    let mut open = state.open;
    let mut close_after = false;

    egui::Window::new("Import Streamlines")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(480.0);
            ui.label("Import a foreign streamline format directly into the viewer.");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Source:");
                let source_label = state
                    .source_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "Select a streamline file".to_string());
                ui.monospace(source_label);
                if ui.button("Browse...").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Streamline files", &["trk", "tck", "vtk", "tt", "gz"])
                        .pick_file()
                {
                    state.source_path = Some(path.clone());
                    state.detected_format = trx_rs::detect_format(&path).ok();
                    state.error_msg = None;
                }
            });

            let format = state.detected_format;
            if let Some(format) = format {
                ui.label(format_summary(format));
            } else if state.source_path.is_some() {
                ui.colored_label(egui::Color32::YELLOW, "Unsupported or unrecognized streamline format.");
            } else {
                ui.label("Choose a `.trk`, `.trk.gz`, `.tck`, `.tck.gz`, `.vtk`, or `.tt.gz` file.");
            }

            if matches!(format, Some(Format::Tck | Format::Vtk)) {
                ui.separator();
                ui.label("Optional NIfTI reference");
                ui.small("Reserved for formats that may need external spatial metadata in future workflows.");
                ui.horizontal(|ui| {
                    let reference_label = state
                        .reference_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "No reference selected".to_string());
                    ui.monospace(reference_label);
                    if ui.button("Choose reference...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                            .pick_file()
                    {
                        state.reference_path = Some(path);
                    }
                    if state.reference_path.is_some() && ui.button("Clear").clicked() {
                        state.reference_path = None;
                    }
                });
            }

            if matches!(format, Some(Format::Vtk)) {
                ui.separator();
                ui.label("VTK coordinate system");
                egui::ComboBox::from_id_salt("vtk_coordinate_mode")
                    .selected_text(vtk_coordinate_mode_label(state.vtk_coordinate_mode))
                    .show_ui(ui, |ui| {
                        for mode in [
                            VtkCoordinateMode::HeaderOrWarn,
                            VtkCoordinateMode::AssumeRas,
                            VtkCoordinateMode::AssumeLps,
                        ] {
                            ui.selectable_value(
                                &mut state.vtk_coordinate_mode,
                                mode,
                                vtk_coordinate_mode_label(mode),
                            );
                        }
                    });
                ui.small(vtk_coordinate_mode_description(state.vtk_coordinate_mode));
                if let Some(path) = &state.source_path
                    && let Ok(warnings) = trx_rs::vtk_import_warnings(path, state.vtk_coordinate_mode)
                {
                    for warning in warnings {
                        ui.colored_label(egui::Color32::YELLOW, warning);
                    }
                }
            }

            ui.separator();
            ui.collapsing("Future metadata attachments", |ui| {
                ui.add_enabled_ui(false, |ui| {
                    ui.label("External text/CSV DPS and DPV attachment will live here.");
                    ui.horizontal(|ui| {
                        ui.label("Per-streamline table:");
                        ui.text_edit_singleline(&mut String::new());
                    });
                    ui.horizontal(|ui| {
                        ui.label("Per-vertex table:");
                        ui.text_edit_singleline(&mut String::new());
                    });
                });
                ui.small("Coming later.");
            });

            if let Some(msg) = &state.error_msg {
                ui.separator();
                ui.colored_label(egui::Color32::RED, msg);
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    close_after = true;
                }
                let can_import = matches!(
                    state.detected_format,
                    Some(Format::Trk | Format::Tck | Format::Vtk | Format::TinyTrack)
                );
                if ui
                    .add_enabled(can_import, egui::Button::new("Import"))
                    .clicked()
                {
                    action.import_requested = true;
                }
            });
        });

    state.open = open && !close_after;
    if close_after {
        state.error_msg = None;
    }
    action
}

fn format_summary(format: Format) -> &'static str {
    match format {
        Format::Trk => {
            "TrackVis TRK import. Direct viewing is intentionally second-class; convert to `.trx` first if you care about full handling."
        }
        Format::Tck => "MRtrix TCK import. Gzipped `.tck.gz` is supported.",
        Format::Vtk => "VTK PolyData streamline import.",
        Format::TinyTrack => {
            "DSI Studio Tiny Track import. Embedded TT metadata and groups will be preserved."
        }
        Format::Trx => "TRX files should be opened directly with File > Open TRX.",
    }
}

fn vtk_coordinate_mode_label(mode: VtkCoordinateMode) -> &'static str {
    match mode {
        VtkCoordinateMode::HeaderOrWarn => "Use header, else warn + assume LPS",
        VtkCoordinateMode::AssumeRas => "Force RAS",
        VtkCoordinateMode::AssumeLps => "Force LPS",
    }
}

fn vtk_coordinate_mode_description(mode: VtkCoordinateMode) -> &'static str {
    match mode {
        VtkCoordinateMode::HeaderOrWarn => {
            "Reads `SPACE=RAS` or `SPACE=LPS` when present. If the file is silent, trx-rs warns and assumes LPS."
        }
        VtkCoordinateMode::AssumeRas => "Treat the stored VTK coordinates as already being in RAS.",
        VtkCoordinateMode::AssumeLps => {
            "Treat the stored VTK coordinates as LPS and flip them into RAS."
        }
    }
}

#[allow(dead_code)]
fn _display_path(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .map(|value| value.display().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{format_summary, vtk_coordinate_mode_description, vtk_coordinate_mode_label};
    use trx_rs::{Format, VtkCoordinateMode};

    #[test]
    fn trk_summary_recommends_trx_conversion() {
        let summary = format_summary(Format::Trk);
        assert!(summary.contains("TrackVis"));
        assert!(summary.contains("convert"));
        assert!(summary.contains(".trx"));
    }

    #[test]
    fn vtk_mode_labels_are_stable() {
        assert!(vtk_coordinate_mode_label(VtkCoordinateMode::HeaderOrWarn).contains("assume LPS"));
        assert!(vtk_coordinate_mode_description(VtkCoordinateMode::AssumeRas).contains("RAS"));
    }
}

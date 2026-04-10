use std::path::PathBuf;

use trx_rs::Format;

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
                        .add_filter("Streamline files", &["tck", "vtk", "tt", "gz"])
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
                ui.label("Choose a `.tck`, `.tck.gz`, `.vtk`, or `.tt.gz` file.");
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
                    Some(Format::Tck | Format::Vtk | Format::TinyTrack)
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
        Format::Tck => "MRtrix TCK import. Gzipped `.tck.gz` is supported.",
        Format::Vtk => "VTK PolyData streamline import.",
        Format::TinyTrack => {
            "DSI Studio Tiny Track import. Embedded TT metadata and groups will be preserved."
        }
        Format::Trx => "TRX files should be opened directly with File > Open TRX.",
    }
}

#[allow(dead_code)]
fn _display_path(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .map(|value| value.display().to_string())
        .unwrap_or_default()
}

use crate::app::state::ReferenceAffineDialogState;

pub struct ReferenceAffineDialogAction {
    pub retry_requested: bool,
}

pub fn show_reference_affine_dialog(
    ctx: &egui::Context,
    state: &mut ReferenceAffineDialogState,
) -> ReferenceAffineDialogAction {
    let mut action = ReferenceAffineDialogAction {
        retry_requested: false,
    };

    if !state.open {
        return action;
    }

    let mut open = state.open;
    let mut close_after = false;

    egui::Window::new("Select Reference Affine")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(520.0);
            ui.label("This DSI Studio file is missing a `trans` matrix.");
            ui.small("Choose a NIfTI image so TRXViz can use its affine to convert the file for visualization.");
            ui.separator();

            ui.label("DSI Studio file:");
            let source_label = state
                .source_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "No file selected".to_string());
            ui.monospace(source_label);

            ui.separator();
            ui.label("Reference NIfTI:");
            ui.horizontal(|ui| {
                let reference_label = state
                    .reference_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "No reference selected".to_string());
                ui.monospace(reference_label);
                if ui.button("Browse...").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                        .pick_file()
                {
                    state.reference_path = Some(path);
                    state.error_msg = None;
                }
                if state.reference_path.is_some() && ui.button("Clear").clicked() {
                    state.reference_path = None;
                }
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
                if ui.button("Retry Load").clicked() {
                    action.retry_requested = true;
                }
            });
        });

    state.open = open && !close_after;
    if close_after {
        state.close();
    }

    action
}

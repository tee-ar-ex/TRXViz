use trx_rs::Format;

use crate::app::state::{FormatlessDType, MergeStreamlineRowState, MergeStreamlinesDialogState};

const PATH_LABEL_WIDTH: f32 = 210.0;
const GROUP_NAME_WIDTH: f32 = 130.0;

pub struct MergeStreamlinesDialogAction {
    pub merge_requested: bool,
}

pub fn show_merge_streamlines_dialog(
    ctx: &egui::Context,
    state: &mut MergeStreamlinesDialogState,
) -> MergeStreamlinesDialogAction {
    let mut action = MergeStreamlinesDialogAction {
        merge_requested: false,
    };

    if !state.open {
        return action;
    }

    let mut open = state.open;
    let mut close_after = false;

    egui::Window::new("Create Streamline File From Merge")
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(840.0)
        .show(ctx, |ui| {
            ui.label("Merge multiple streamline files into a new TRX file in the listed order.");
            ui.small("Optional references are stored per input row for formats that may need them in future import workflows.");
            ui.separator();

            ui.horizontal(|ui| {
                ui.add_sized([PATH_LABEL_WIDTH, 20.0], egui::Label::new(egui::RichText::new("Streamline File").strong()));
                ui.add_space(ui.spacing().item_spacing.x + 64.0);
                ui.add_sized([PATH_LABEL_WIDTH, 20.0], egui::Label::new(egui::RichText::new("Reference NIfTI").strong()));
                ui.add_space(ui.spacing().item_spacing.x + 108.0);
                ui.add_sized([GROUP_NAME_WIDTH, 20.0], egui::Label::new(egui::RichText::new("Group name").strong()));
            });

            let mut remove_idx = None;
            let mut move_up_idx = None;
            let mut move_down_idx = None;

            let row_count = state.rows.len();
            for (idx, row) in state.rows.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    show_compact_path_label(ui, row.source_path.as_deref(), "Select streamline file");
                    if ui.button("Browse...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("Streamline files", &["trx", "tck", "vtk", "tt", "gz"])
                            .pick_file()
                    {
                        row.source_path = Some(path.clone());
                        row.detected_format = trx_rs::detect_format(&path).ok();
                        state.error_msg = None;
                    }

                    show_compact_path_label(ui, row.reference_path.as_deref(), "Optional reference");
                    if ui.button("Ref...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                            .pick_file()
                    {
                        row.reference_path = Some(path);
                    }
                    if ui
                        .add_enabled(row.reference_path.is_some(), egui::Button::new("Clear Ref"))
                        .clicked()
                    {
                        row.reference_path = None;
                    }

                    ui.add_sized(
                        [GROUP_NAME_WIDTH, 20.0],
                        egui::TextEdit::singleline(&mut row.group_name).hint_text("Optional"),
                    );

                    if ui
                        .add_enabled(idx > 0, egui::Button::new("Up"))
                        .clicked()
                    {
                        move_up_idx = Some(idx);
                    }
                    if ui
                        .add_enabled(idx + 1 < row_count, egui::Button::new("Down"))
                        .clicked()
                    {
                        move_down_idx = Some(idx);
                    }
                    if ui
                        .add_enabled(row_count > 2, egui::Button::new("Remove"))
                        .clicked()
                    {
                        remove_idx = Some(idx);
                    }
                });

                match row.detected_format {
                    Some(format) => {
                        ui.small(format_summary(format));
                    }
                    None if row.source_path.is_some() => {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "Unsupported or unrecognized streamline format.",
                        );
                    }
                    None => {
                        ui.small("Choose a `.trx`, `.tck`, `.tck.gz`, `.vtk`, or `.tt.gz` file.");
                    }
                };

                ui.separator();
            }

            if let Some(idx) = move_up_idx {
                state.rows.swap(idx - 1, idx);
            }
            if let Some(idx) = move_down_idx {
                state.rows.swap(idx, idx + 1);
            }
            if let Some(idx) = remove_idx {
                state.rows.remove(idx);
            }

            if ui.button("+ Add Input").clicked() {
                state.rows.push(MergeStreamlineRowState::default());
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Output TRX:");
                show_compact_path_label(ui, state.output_path.as_deref(), "Choose output path");
                if ui.button("Browse...").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("TRX files", &["trx"])
                        .set_file_name("merged.trx")
                        .save_file()
                {
                    state.output_path = Some(path);
                    state.error_msg = None;
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.delete_dps, "Delete DPS");
                ui.checkbox(&mut state.delete_dpv, "Delete DPV");
                ui.checkbox(&mut state.delete_groups, "Delete Groups");
            });
            ui.horizontal(|ui| {
                ui.label("Positions dtype:");
                egui::ComboBox::from_id_salt("merge_positions_dtype")
                    .selected_text(
                        state
                            .positions_dtype
                            .map(FormatlessDType::label)
                            .unwrap_or("Use first input"),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.positions_dtype,
                            None,
                            "Use first input",
                        );
                        for dtype in FormatlessDType::ALL {
                            ui.selectable_value(
                                &mut state.positions_dtype,
                                Some(dtype),
                                dtype.label(),
                            );
                        }
                    });
            });

            if let Some(msg) = &state.error_msg {
                ui.separator();
                ui.colored_label(egui::Color32::RED, msg);
            }

            let valid_rows = state
                .rows
                .iter()
                .filter(|row| matches!(row.detected_format, Some(Format::Trx | Format::Tck | Format::Vtk | Format::TinyTrack)))
                .count();
            let can_merge = valid_rows >= 2 && state.output_path.is_some();

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    close_after = true;
                }
                if ui
                    .add_enabled(can_merge, egui::Button::new("Create Merged File"))
                    .clicked()
                {
                    action.merge_requested = true;
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
        Format::Trx => "Native TRX input.",
        Format::Tck => "MRtrix TCK input. Gzipped `.tck.gz` is supported.",
        Format::Vtk => "VTK PolyData streamline input.",
        Format::TinyTrack => "DSI Studio Tiny Track input.",
    }
}

fn show_compact_path_label(ui: &mut egui::Ui, path: Option<&std::path::Path>, empty_label: &str) {
    match path {
        Some(path) => {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(empty_label);
            let parent = path
                .parent()
                .map(shorten_path)
                .unwrap_or_else(|| ".".to_string());

            ui.allocate_ui_with_layout(
                egui::vec2(PATH_LABEL_WIDTH, 34.0),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_max_width(PATH_LABEL_WIDTH);
                    let response = ui
                        .add(egui::Label::new(egui::RichText::new(file_name).monospace()).truncate())
                        .on_hover_text(path.display().to_string());
                    ui.small(parent).on_hover_text(path.display().to_string());
                    response.context_menu(|ui| {
                        ui.label(path.display().to_string());
                    });
                },
            );
        }
        None => {
            ui.add_sized(
                [PATH_LABEL_WIDTH, 34.0],
                egui::Label::new(egui::RichText::new(empty_label).monospace()).truncate(),
            );
        }
    }
}

fn shorten_path(path: &std::path::Path) -> String {
    let components = path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    if components.len() <= 3 {
        return path.display().to_string();
    }

    format!(
        ".../{}/{}/{}",
        components[components.len() - 3],
        components[components.len() - 2],
        components[components.len() - 1]
    )
}

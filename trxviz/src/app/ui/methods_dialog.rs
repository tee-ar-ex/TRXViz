//! Read-only "Generate methods…" modal.
//!
//! The dialog is purely a viewer/exporter over the pair of strings in
//! [`MethodsDialogState`]. All assembly happens up-stream in
//! [`trxviz_core::workflow::methods::generate_methods_report`]; this
//! module is only here to surface the output in the GUI, let the user
//! copy it to the clipboard, and write `methods.md` + `references.bib`
//! to a chosen folder.

use std::path::Path;

use crate::app::state::MethodsDialogState;

pub struct MethodsDialogAction {
    /// Set when the user asks to export both files. The caller handles
    /// the actual file dialog + write, because that path interacts
    /// with native dialogs + the app's status reporting machinery.
    pub export_requested: bool,
}

pub fn show_methods_dialog(
    ctx: &egui::Context,
    state: &mut MethodsDialogState,
) -> MethodsDialogAction {
    let mut action = MethodsDialogAction {
        export_requested: false,
    };
    if !state.open {
        return action;
    }

    let mut open = state.open;
    let mut close_after = false;

    egui::Window::new("Generate methods")
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(820.0)
        .default_height(560.0)
        .show(ctx, |ui| {
            ui.label(
                "Methods section for the currently-loaded workflow. Re-open the \
                 dialog after edits to regenerate.",
            );
            ui.small(
                "Output uses Pandoc citation syntax (`[@key]`). Run the markdown \
                 through `pandoc --citeproc --bibliography=references.bib` to \
                 resolve citations.",
            );
            ui.separator();

            // Action row — kept above the scroll area so buttons stay
            // reachable even with long bodies.
            ui.horizontal(|ui| {
                if ui.button("Copy markdown").clicked() {
                    ctx.copy_text(state.body_markdown.clone());
                    state.status = Some("Copied methods markdown to clipboard.".into());
                }
                if ui.button("Copy BibTeX").clicked() {
                    ctx.copy_text(state.bibtex.clone());
                    state.status = Some("Copied BibTeX to clipboard.".into());
                }
                if ui.button("Export to folder…").clicked() {
                    action.export_requested = true;
                }
                if ui.button("Close").clicked() {
                    close_after = true;
                }
            });

            if let Some(status) = &state.status {
                ui.colored_label(egui::Color32::LIGHT_GREEN, status);
            }

            ui.separator();

            // Two-column split: markdown body above, BibTeX below. A
            // vertical split keeps the proportions easy to resize by
            // the user; egui doesn't ship a split-pane widget so we use
            // two independent ScrollAreas sized to half the window.
            egui::ScrollArea::vertical()
                .id_salt("methods_md_scroll")
                .max_height(280.0)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("methods.md").strong());
                    ui.add(
                        egui::TextEdit::multiline(&mut state.body_markdown.as_str())
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(14),
                    );
                });

            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("methods_bib_scroll")
                .max_height(220.0)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("references.bib").strong());
                    ui.add(
                        egui::TextEdit::multiline(&mut state.bibtex.as_str())
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(10),
                    );
                });
        });

    state.open = open && !close_after;
    if close_after {
        state.status = None;
    }
    action
}

/// Convenience that writes `methods.md` and `references.bib` into the
/// given directory using the currently-buffered strings. Returns a
/// human-readable status line on success; callers surface it via the
/// usual app status banner.
pub fn export_to_directory(
    state: &MethodsDialogState,
    dir: &Path,
) -> std::io::Result<String> {
    let md_path = dir.join("methods.md");
    let bib_path = dir.join("references.bib");
    std::fs::write(&md_path, &state.body_markdown)?;
    std::fs::write(&bib_path, &state.bibtex)?;
    Ok(format!(
        "Wrote methods prose to {} and bibliography to {}.",
        md_path.display(),
        bib_path.display()
    ))
}

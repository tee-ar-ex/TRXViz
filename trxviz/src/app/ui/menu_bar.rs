use crate::app::state::UiMode;

pub struct MenuAction {
    pub open_files: bool,
    pub new_workflow_project: bool,
    pub open_workflow_project: bool,
    pub save_workflow_project: bool,
    pub save_workflow_project_as: bool,
    pub open_trx: bool,
    pub import_streamlines: bool,
    pub create_streamline_merge: bool,
    pub open_nifti: bool,
    pub open_gifti: bool,
    pub open_parcellation: bool,
    pub open_3d_window: bool,
    pub open_2d_window: bool,
    pub export_3d_view: bool,
    pub export_2d_view: bool,
    pub switch_mode: Option<UiMode>,
}

pub fn show_menu_bar(ctx: &egui::Context, ui_mode: UiMode) -> MenuAction {
    let mut action = MenuAction {
        open_files: false,
        new_workflow_project: false,
        open_workflow_project: false,
        save_workflow_project: false,
        save_workflow_project_as: false,
        open_trx: false,
        import_streamlines: false,
        create_streamline_merge: false,
        open_nifti: false,
        open_gifti: false,
        open_parcellation: false,
        open_3d_window: false,
        open_2d_window: false,
        export_3d_view: false,
        export_2d_view: false,
        switch_mode: None,
    };
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open Files...").clicked() {
                    action.open_files = true;
                    ui.close();
                }
                if ui.button("Open Project...").clicked() {
                    action.open_workflow_project = true;
                    ui.close();
                }
                if ui_mode == UiMode::Advanced {
                    ui.separator();
                    if ui.button("New Workflow Project").clicked() {
                        action.new_workflow_project = true;
                        ui.close();
                    }
                    if ui.button("Save Workflow Project").clicked() {
                        action.save_workflow_project = true;
                        ui.close();
                    }
                    if ui.button("Save Workflow Project As...").clicked() {
                        action.save_workflow_project_as = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Open TRX...").clicked() {
                        action.open_trx = true;
                        ui.close();
                    }
                    if ui.button("Import Streamlines...").clicked() {
                        action.import_streamlines = true;
                        ui.close();
                    }
                    if ui.button("Create Streamline File From Merge...").clicked() {
                        action.create_streamline_merge = true;
                        ui.close();
                    }
                    if ui.button("Open NIfTI...").clicked() {
                        action.open_nifti = true;
                        ui.close();
                    }
                    if ui.button("Open GIFTI Surface...").clicked() {
                        action.open_gifti = true;
                        ui.close();
                    }
                    if ui.button("Open Parcellation...").clicked() {
                        action.open_parcellation = true;
                        ui.close();
                    }
                }
            });
            ui.menu_button("View", |ui| {
                if ui
                    .selectable_label(ui_mode == UiMode::Simple, "Simple Viewer")
                    .clicked()
                {
                    action.switch_mode = Some(UiMode::Simple);
                    ui.close();
                }
                if ui
                    .selectable_label(ui_mode == UiMode::Advanced, "Advanced Workspace")
                    .clicked()
                {
                    action.switch_mode = Some(UiMode::Advanced);
                    ui.close();
                }
                ui.separator();
                if ui.button("Open 3D Window").clicked() {
                    action.open_3d_window = true;
                    ui.close();
                }
                if ui.button("Open 2D Window").clicked() {
                    action.open_2d_window = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Export 3D View...").clicked() {
                    action.export_3d_view = true;
                    ui.close();
                }
                if ui.button("Export 2D View...").clicked() {
                    action.export_2d_view = true;
                    ui.close();
                }
            });
            ui.separator();
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let advanced = ui.selectable_label(
                    ui_mode == UiMode::Advanced,
                    UiMode::Advanced.label(),
                );
                let simple =
                    ui.selectable_label(ui_mode == UiMode::Simple, UiMode::Simple.label());
                if advanced.clicked() {
                    action.switch_mode = Some(UiMode::Advanced);
                } else if simple.clicked() {
                    action.switch_mode = Some(UiMode::Simple);
                }
            });
        });
    });
    action
}

mod callbacks;
mod file_loading;
mod helpers;
mod state;
mod ui;
mod workflow;

use std::path::PathBuf;

use state::{
    ImportDialogState, MergeStreamlinesDialogState, PendingFileLoad, SceneState, UiMode,
    ViewportState, WorkerMessage, WorkerReceiver, WorkerSender, WorkflowState,
};
use trxviz_core::renderer::slice_renderer::{AllSliceResources, SliceAxis};
use workflow::workflow_job_kind_title;

/// Main application state.
pub struct TrxVizApp {
    pub(crate) ui_mode: UiMode,
    pub(crate) scene: SceneState,
    pub(crate) viewport: ViewportState,
    pub(crate) workflow: WorkflowState,
    pub(crate) error_msg: Option<String>,
    pub(crate) status_msg: Option<String>,
    pub(crate) worker_tx: WorkerSender,
    pub(crate) worker_rx: WorkerReceiver,
    pub(crate) next_job_id: u64,
    pub(crate) pending_file_loads: Vec<PendingFileLoad>,
    pub(crate) import_dialog: ImportDialogState,
    pub(crate) merge_streamlines_dialog: MergeStreamlinesDialogState,
}

impl TrxVizApp {
    fn sync_viewport_camera_into_workflow_document(&mut self) {
        self.workflow.document.camera_3d = Some(self.viewport.workflow_camera_3d());
        self.workflow.document.slice_view_3d =
            Some(self.viewport.workflow_slice_view_3d(&self.scene.nifti_files));
        self.workflow.document.slice_visible_3d = Some(self.viewport.slice_visible);
    }

    fn copy_camera_3d_json(&mut self, ctx: &egui::Context) {
        let snippet = serde_json::json!({
            "camera_3d": self.viewport.workflow_camera_3d(),
            "slice_view_3d": self.viewport.workflow_slice_view_3d(&self.scene.nifti_files),
            "slice_visible_3d": self.viewport.slice_visible,
        });
        match serde_json::to_string_pretty(&snippet) {
            Ok(json) => {
                ctx.copy_text(json);
                self.status_msg =
                    Some("Copied 3D camera JSON to the clipboard. Paste it under document.".into());
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to serialize 3D camera: {err}"));
            }
        }
    }

    fn poll_worker_messages(&mut self, frame: &mut eframe::Frame) {
        while let Ok(message) = self.worker_rx.try_recv() {
            match message {
                WorkerMessage::TrxLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_trx(path, data, rs);
                            }
                        }
                        Err(err) => self.error_msg = Some(format!("Failed to load TRX: {err}")),
                    }
                }
                WorkerMessage::NiftiLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(vol) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_nifti(path, vol, rs);
                            }
                        }
                        Err(err) => self.error_msg = Some(format!("Failed to load NIfTI: {err}")),
                    }
                }
                WorkerMessage::GiftiLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(surface) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_gifti_surface(path, surface, rs);
                            }
                        }
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to load GIFTI surface: {err}"))
                        }
                    }
                }
                WorkerMessage::ParcellationLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(source) => self.apply_loaded_parcellation(path, source),
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to load parcellation: {err}"))
                        }
                    }
                }
                WorkerMessage::ImportedStreamlinesLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_trx(path, data, rs);
                            }
                        }
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to import streamlines: {err}"))
                        }
                    }
                }
                WorkerMessage::MergedStreamlinesCreated {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_trx(path.clone(), data, rs);
                                self.status_msg = Some(format!(
                                    "Created merged streamlines at {}",
                                    path.display()
                                ));
                                self.error_msg = None;
                            }
                        }
                        Err(err) => {
                            self.error_msg =
                                Some(format!("Failed to create merged streamlines: {err}"))
                        }
                    }
                }
            }
        }
    }

    fn active_task_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .pending_file_loads
            .iter()
            .map(|job| job.label.clone())
            .collect();

        for (node_uuid, (kind, _)) in &self.workflow.jobs_in_flight {
            let label = self
                .workflow
                .document
                .graph
                .get(*node_uuid)
                .map(|node| node.label.clone())
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| workflow_job_kind_title(*kind).to_string());
            labels.push(format!(
                "Building {} for {}",
                workflow_job_kind_title(*kind),
                label
            ));
        }

        labels
    }

    fn draw_activity_overlay(&self, ctx: &egui::Context) {
        let tasks = self.active_task_labels();
        if tasks.is_empty() {
            return;
        }

        egui::Area::new("activity_overlay".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(280.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label("Working");
                    });
                    ui.separator();
                    for task in tasks {
                        ui.small(task);
                    }
                });
            });
    }

    fn open_import_dialog(&mut self, path: Option<PathBuf>) {
        let detected = path
            .as_ref()
            .and_then(|selected| trx_rs::detect_format(selected).ok());
        self.import_dialog.open_with_path(path, detected);
    }

    pub(crate) fn open_files_dialog(&mut self) {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "TRXViz files",
                &[
                    "trx", "trk", "tck", "vtk", "tt", "gz", "nii", "gii", "gifti",
                ],
            )
            .pick_files()
        else {
            return;
        };

        let auto_import_streamlines = self.ui_mode == UiMode::Simple;
        for path in paths {
            self.open_path(path, auto_import_streamlines);
        }
    }

    fn open_path(&mut self, path: PathBuf, auto_import_streamlines: bool) {
        match helpers::classify_dropped_path(&path) {
            helpers::DroppedPathKind::OpenTrx => self.begin_load_trx(path),
            helpers::DroppedPathKind::ImportTractogram(_) if auto_import_streamlines => {
                self.begin_import_streamlines_path(path);
            }
            helpers::DroppedPathKind::ImportTractogram(_) => self.open_import_dialog(Some(path)),
            helpers::DroppedPathKind::OpenNifti => self.begin_load_nifti(path),
            helpers::DroppedPathKind::OpenParcellation => self.begin_load_parcellation(path),
            helpers::DroppedPathKind::OpenGifti => self.begin_load_gifti_surface(path),
            helpers::DroppedPathKind::Unsupported => {
                self.error_msg = Some(format!(
                    "Unknown or unsupported file type: {}",
                    path.display()
                ));
            }
        }
    }

    pub fn new(
        cc: &eframe::CreationContext<'_>,
        trx_path: Option<String>,
        nifti_path: Option<String>,
    ) -> Self {
        let (worker_tx, worker_rx) = std::sync::mpsc::channel();
        let (workflow_job_tx, workflow_job_rx) = std::sync::mpsc::channel();
        let mut app = Self {
            ui_mode: UiMode::Simple,
            scene: SceneState::default(),
            viewport: ViewportState::default(),
            workflow: WorkflowState::new(workflow_job_tx, workflow_job_rx),
            error_msg: None,
            status_msg: None,
            worker_tx,
            worker_rx,
            next_job_id: 1,
            pending_file_loads: Vec::new(),
            import_dialog: ImportDialogState::default(),
            merge_streamlines_dialog: MergeStreamlinesDialogState::default(),
        };

        if cc.wgpu_render_state.is_some() {
            if let Some(path) = trx_path {
                app.begin_load_trx(PathBuf::from(path));
            }
            if let Some(path) = nifti_path {
                app.begin_load_nifti(PathBuf::from(path));
            }
        }

        app
    }
}

impl eframe::App for TrxVizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.poll_worker_messages(frame);
        self.poll_workflow_job_messages();

        // Update slice positions if dirty
        if self.viewport.slices_dirty {
            if let Some(rs) = frame.wgpu_render_state() {
                let renderer = rs.renderer.read();
                if let Some(all) = renderer.callback_resources.get::<AllSliceResources>() {
                    for (file_id, sr) in &all.entries {
                        if let Some(nf) = self.scene.nifti_files.iter().find(|n| n.id == *file_id) {
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Axial,
                                self.viewport.slice_indices[0],
                                &nf.volume,
                            );
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Coronal,
                                self.viewport.slice_indices[1],
                                &nf.volume,
                            );
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Sagittal,
                                self.viewport.slice_indices[2],
                                &nf.volume,
                            );
                        }
                    }
                }
            }
            self.viewport.slices_dirty = false;
        }

        // ── Handle dropped files ──
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in &dropped {
            if let Some(path) = &file.path {
                self.open_path(path.clone(), self.ui_mode == UiMode::Simple);
            }
        }

        // ── Menu bar ──
        let menu_action = ui::menu_bar::show_menu_bar(ctx, self.ui_mode);
        if let Some(mode) = menu_action.switch_mode {
            self.ui_mode = mode;
        }
        if menu_action.open_files {
            self.open_files_dialog();
        }
        if menu_action.open_trx {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("TRX files", &["trx"])
                .pick_file()
            {
                self.begin_load_trx(path);
            }
        }
        if menu_action.new_workflow_project {
            self.new_workflow_project(frame);
        }
        if menu_action.open_workflow_project {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Workflow Project", &["json"])
                .pick_file()
            {
                self.open_workflow_project(path, frame);
            }
        }
        if menu_action.save_workflow_project {
            self.save_workflow_project(false);
        }
        if menu_action.save_workflow_project_as {
            self.save_workflow_project(true);
        }
        if menu_action.import_streamlines {
            self.open_import_dialog(None);
        }
        if menu_action.create_streamline_merge {
            self.merge_streamlines_dialog.open();
        }
        if menu_action.open_nifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                .pick_file()
            {
                self.begin_load_nifti(path);
            }
        }
        if menu_action.open_gifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GIFTI files", &["gii", "gifti"])
                .pick_file()
            {
                self.begin_load_gifti_surface(path);
            }
        }
        if menu_action.open_parcellation {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                .pick_file()
            {
                self.begin_load_parcellation(path);
            }
        }
        if menu_action.open_3d_window {
            self.viewport.window_3d_open = true;
        }
        if menu_action.open_2d_window {
            self.viewport.view_2d.window_open = true;
        }
        if menu_action.export_3d_view {
            self.viewport.export_dialog.open = true;
            self.viewport.export_dialog.target = state::ExportTarget::View3D;
        }
        if menu_action.export_2d_view {
            self.viewport.export_dialog.open = true;
            self.viewport.export_dialog.target = state::ExportTarget::View2D;
        }
        if menu_action.copy_camera_3d_json {
            self.copy_camera_3d_json(ctx);
        }
        if self.ui_mode == UiMode::Advanced || self.import_dialog.open {
            let import_action = ui::import_dialog::show_import_dialog(ctx, &mut self.import_dialog);
            if import_action.import_requested {
                if self.import_dialog.detected_format.is_some_and(|format| {
                    matches!(
                        format,
                        trx_rs::Format::Trk
                            | trx_rs::Format::Tck
                            | trx_rs::Format::Vtk
                            | trx_rs::Format::TinyTrack
                    )
                }) && self.import_dialog.source_path.is_some()
                {
                    let import_state = self.import_dialog.clone();
                    self.begin_import_streamlines(&import_state);
                    self.import_dialog.close();
                } else {
                    self.import_dialog.error_msg =
                        Some("Choose a supported foreign streamline file to import.".to_string());
                }
            }
        }
        if self.ui_mode == UiMode::Advanced || self.merge_streamlines_dialog.open {
            let merge_action = ui::merge_streamlines_dialog::show_merge_streamlines_dialog(
                ctx,
                &mut self.merge_streamlines_dialog,
            );
            if merge_action.merge_requested {
                if self.merge_streamlines_dialog.output_path.is_none() {
                    self.merge_streamlines_dialog.error_msg =
                        Some("Choose an output TRX path.".to_string());
                } else if self
                    .merge_streamlines_dialog
                    .rows
                    .iter()
                    .filter(|row| row.source_path.is_some() && row.detected_format.is_some())
                    .count()
                    < 2
                {
                    self.merge_streamlines_dialog.error_msg =
                        Some("Choose at least two supported streamline inputs.".to_string());
                } else {
                    let merge_state = self.merge_streamlines_dialog.clone();
                    self.begin_merge_streamlines(&merge_state);
                    self.merge_streamlines_dialog.close();
                }
            }
        }

        self.refresh_workflow_runtime_if_needed(ctx);
        if self.workflow.last_settled_revision == self.workflow.document_revision {
            self.queue_workflow_jobs();
        }
        self.sync_workflow_resources(frame);
        self.show_viewports(ctx);
        let open_files_after_ui = match self.ui_mode {
            UiMode::Simple => self.show_simple_shell(ctx),
            UiMode::Advanced => {
                self.show_advanced_shell(ctx, frame);
                false
            }
        };
        if open_files_after_ui {
            self.open_files_dialog();
        }

        self.draw_activity_overlay(ctx);
        if !self.active_task_labels().is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

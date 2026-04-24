mod callbacks;
mod file_apply;
mod file_loading;
mod helpers;
mod state;
mod ui;
mod workflow;

use std::path::PathBuf;

use state::{
    ImportDialogState, MergeStreamlinesDialogState, PendingFileLoad, ReferenceAffineDialogState,
    SceneState, UiMode, ViewportState, WorkerMessage, WorkerReceiver, WorkerSender, WorkflowState,
};
use trxviz_core::renderer::slice_renderer::{AllSliceResources, SliceAxis};
use trxviz_core::workflow::WorkflowNodeUuid;
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
    pub(crate) reference_affine_dialog: ReferenceAffineDialogState,
    pub(crate) merge_streamlines_dialog: MergeStreamlinesDialogState,
    /// Slice-local ODX glyph amplitude normalization used by the shader LUT path.
    pub(crate) odx_amp_norm: f32,
    pub(crate) max_storage_buffer_binding_size: Option<usize>,
    /// Cloned wgpu device for background GPU compute (tractography, etc.).
    pub(crate) gpu_device: Option<wgpu::Device>,
    /// Cloned wgpu queue for background GPU compute.
    pub(crate) gpu_queue: Option<wgpu::Queue>,
}

impl TrxVizApp {
    fn capture_document_camera_3d(&self) -> workflow::WorkflowCamera3D {
        workflow::WorkflowCamera3D {
            target: self.viewport.camera_3d().center.to_array(),
            azimuth_deg: self.viewport.camera_3d().yaw.to_degrees(),
            elevation_deg: self.viewport.camera_3d().pitch.to_degrees(),
            distance: self.viewport.camera_3d().distance,
        }
    }

    fn capture_document_render_3d(&self) -> trxviz_core::lighting::WorkflowRender3D {
        self.viewport.render_3d().clone().sanitized()
    }

    fn capture_document_slice_view_3d(&self) -> workflow::WorkflowSliceView3D {
        workflow::WorkflowSliceView3D {
            visible: self.viewport.slice_visible(),
            positions_ras: [
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 0),
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 1),
                self.viewport
                    .slice_world_position(&self.scene.nifti_files, 2),
            ],
        }
    }

    fn capture_document_slice_view_ui(&self) -> workflow::WorkflowSliceViewUi {
        crate::app::workflow::capture_gui_slice_view_state(&self.viewport)
    }

    fn apply_document_camera_3d_to_viewport(&mut self) {
        let Some(camera) = self.workflow.document.camera_3d else {
            return;
        };
        self.viewport.camera_3d_mut().center = glam::Vec3::from_array(camera.target);
        self.viewport.camera_3d_mut().yaw = camera.azimuth_deg.to_radians();
        self.viewport.camera_3d_mut().pitch = camera.elevation_deg.to_radians();
        self.viewport.camera_3d_mut().distance = camera.distance.max(0.1);
    }

    fn apply_document_render_3d_to_viewport(&mut self) {
        *self.viewport.render_3d_mut() = self
            .workflow
            .document
            .render_3d
            .clone()
            .unwrap_or_default()
            .sanitized();
    }

    fn apply_document_slice_view_3d_to_viewport(&mut self) {
        let Some(slice_view) = self.workflow.document.slice_view_3d else {
            return;
        };
        self.viewport.set_slice_visible_all(slice_view.visible);
        self.viewport
            .set_slice_world_offsets(slice_view.positions_ras);
        if let Some(nf) = self.scene.nifti_files.first() {
            self.viewport.set_slice_indices([
                nf.volume
                    .nearest_slice_index(0, slice_view.positions_ras[0]),
                nf.volume
                    .nearest_slice_index(1, slice_view.positions_ras[1]),
                nf.volume
                    .nearest_slice_index(2, slice_view.positions_ras[2]),
            ]);
            self.viewport.mark_slices_dirty();
        }
    }

    fn apply_document_slice_view_ui_to_viewport(&mut self) {
        if let Some(slice_view_ui) = self.workflow.document.slice_view_ui.clone() {
            crate::app::workflow::apply_gui_slice_view_state(&mut self.viewport, slice_view_ui);
        }
    }

    fn copy_camera_3d_json(&mut self, ctx: &egui::Context) {
        let snippet = serde_json::json!({
            "camera_3d": self.capture_document_camera_3d(),
            "slice_view_3d": self.capture_document_slice_view_3d(),
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
                WorkerMessage::AssetLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if matches!(data, trxviz_core::asset_loader::LoadedAsset::Odx(_)) {
                                self.reference_affine_dialog.close();
                            }
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_asset(path, data, rs);
                            }
                        }
                        Err(err) => {
                            if file_loading::needs_reference_affine_recovery(&path, &err) {
                                self.reference_affine_dialog.open_for_source(path);
                                self.error_msg = None;
                            } else {
                                self.error_msg = Some(format!("Failed to load asset: {err}"));
                            }
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

    /// Non-interactive tasks (file loads) that show in the activity
    /// overlay but can't be cancelled. Kept separate from workflow
    /// jobs because those get a Cancel button.
    fn pending_file_load_labels(&self) -> Vec<String> {
        self.pending_file_loads
            .iter()
            .map(|job| job.label.clone())
            .collect()
    }

    /// In-flight workflow jobs with their node UUIDs so the overlay
    /// can render a per-job Cancel button.
    fn in_flight_workflow_jobs(&self) -> Vec<(WorkflowNodeUuid, String)> {
        self.workflow
            .jobs_in_flight
            .iter()
            .map(|(node_uuid, (kind, _))| {
                let label = self
                    .workflow
                    .document
                    .graph
                    .get(*node_uuid)
                    .map(|node| node.label.clone())
                    .filter(|label| !label.is_empty())
                    .unwrap_or_else(|| workflow_job_kind_title(*kind).to_string());
                (
                    *node_uuid,
                    format!("Building {} for {}", workflow_job_kind_title(*kind), label),
                )
            })
            .collect()
    }

    fn draw_activity_overlay(&mut self, ctx: &egui::Context) {
        let file_loads = self.pending_file_load_labels();
        let jobs = self.in_flight_workflow_jobs();
        if file_loads.is_empty() && jobs.is_empty() {
            return;
        }

        // Needs to be interactable so the Cancel buttons can receive
        // clicks. The overlay itself is positioned in the top-right
        // corner and doesn't get in the way of the viewport.
        let mut cancel_requests: Vec<WorkflowNodeUuid> = Vec::new();
        egui::Area::new("activity_overlay".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(280.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label("Working");
                    });
                    ui.separator();
                    for label in &file_loads {
                        ui.small(label);
                    }
                    for (node_uuid, label) in &jobs {
                        ui.horizontal(|ui| {
                            ui.small(label);
                            ui.add_space(8.0);
                            if ui.small_button("Cancel").clicked() {
                                cancel_requests.push(*node_uuid);
                            }
                        });
                        // Render a progress bar whenever the worker
                        // has reported at least one `(done, total)`
                        // pair for this node. Jobs that finish before
                        // their first progress tick (reactive
                        // streamlines, small surface queries) get just
                        // the spinner.
                        if let Some(&(done, total)) = self.workflow.job_progress.get(node_uuid)
                            && total > 0
                        {
                            let frac = (done as f32 / total as f32).clamp(0.0, 1.0);
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(240.0)
                                    .text(format!("{done} / {total}")),
                            );
                        }
                    }
                });
            });
        for node_uuid in cancel_requests {
            self.request_cancel_workflow_job(node_uuid);
        }
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
            helpers::DroppedPathKind::OpenCifti => self.begin_load_cifti(path),
            helpers::DroppedPathKind::OpenParcellation => self.begin_load_parcellation(path),
            helpers::DroppedPathKind::OpenGifti => self.begin_load_gifti_surface(path),
            helpers::DroppedPathKind::OpenOdx => self.begin_load_odx(path, None),
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
            reference_affine_dialog: ReferenceAffineDialogState::default(),
            merge_streamlines_dialog: MergeStreamlinesDialogState::default(),
            odx_amp_norm: 1.0,
            max_storage_buffer_binding_size: None,
            gpu_device: None,
            gpu_queue: None,
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
        self.max_storage_buffer_binding_size = frame
            .wgpu_render_state()
            .map(|rs| rs.device.limits().max_storage_buffer_binding_size as usize);

        // Update slice positions if dirty
        if self.viewport.slices_dirty() {
            if let Some(rs) = frame.wgpu_render_state() {
                {
                    let renderer = rs.renderer.read();
                    if let Some(all) = renderer.callback_resources.get::<AllSliceResources>() {
                        for (file_id, sr) in &all.entries {
                            let vol_ref = self
                                .scene
                                .nifti_files
                                .iter()
                                .find(|n| n.id == *file_id)
                                .map(|nf| &nf.volume as &trxviz_core::data::nifti_data::NiftiVolume)
                                .or_else(|| {
                                    self.workflow
                                        .execution_cache
                                        .odx_dpv_materializations
                                        .values()
                                        .find(|m| m.source_id == *file_id)
                                        .map(|m| m.volume.as_ref())
                                });
                            if let Some(vol) = vol_ref {
                                sr.update_slice(
                                    &rs.queue,
                                    SliceAxis::Axial,
                                    self.viewport.slice_index(0),
                                    vol,
                                );
                                sr.update_slice(
                                    &rs.queue,
                                    SliceAxis::Coronal,
                                    self.viewport.slice_index(1),
                                    vol,
                                );
                                sr.update_slice(
                                    &rs.queue,
                                    SliceAxis::Sagittal,
                                    self.viewport.slice_index(2),
                                    vol,
                                );
                            }
                        }
                    }
                }
                // Re-upload ODX glyphs for the new axial slice (3D view).
                // Fixels are full-volume and don't need re-uploading on slice change —
                // the 2D slice views use shader slab clipping to show only the current slice.
                let mut renderer = rs.renderer.write();
                self.update_active_odx_slice_state(
                    &mut renderer.callback_resources,
                    &rs.device,
                    &rs.queue,
                );
            }
            self.viewport.clear_slices_dirty();
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
        if menu_action.export_to_blender {
            self.export_to_blender(trxviz_core::headless::HeadlessView::View3D);
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
            self.viewport.set_camera_3d_window_open(true);
        }
        if menu_action.open_2d_window {
            self.viewport.set_window_2d_open(true);
        }
        if menu_action.export_3d_view {
            let export_dialog = self.viewport.export_dialog_mut();
            export_dialog.open = true;
            export_dialog.target = state::ExportTarget::View3D;
        }
        if menu_action.export_2d_view {
            let export_dialog = self.viewport.export_dialog_mut();
            export_dialog.open = true;
            export_dialog.target = state::ExportTarget::View2D;
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
        if self.ui_mode == UiMode::Advanced || self.reference_affine_dialog.open {
            let dialog_action = ui::reference_affine_dialog::show_reference_affine_dialog(
                ctx,
                &mut self.reference_affine_dialog,
            );
            if dialog_action.retry_requested {
                if let Some(source_path) = self.reference_affine_dialog.source_path.clone() {
                    if let Some(reference_path) =
                        self.reference_affine_dialog.reference_path.clone()
                    {
                        self.begin_load_odx(source_path, Some(reference_path));
                        self.reference_affine_dialog.error_msg = None;
                    } else {
                        self.reference_affine_dialog.error_msg =
                            Some("Choose a NIfTI reference image before retrying.".to_string());
                    }
                } else {
                    self.reference_affine_dialog.error_msg =
                        Some("Choose a DSI Studio file to retry.".to_string());
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
                self.show_workspace(ctx, frame);
                false
            }
        };
        if self.workflow.document_revision != self.workflow.last_interactive_revision {
            ctx.request_repaint();
        }
        if open_files_after_ui {
            self.open_files_dialog();
        }

        self.draw_activity_overlay(ctx);
        if !self.pending_file_loads.is_empty() || !self.workflow.jobs_in_flight.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use trx_rs::{AnyTrxFile, ConversionOptions};
use trxviz_core::data::gifti_data::GiftiSurfaceData;
use trxviz_core::data::loaded_files::{FileId, StreamlineBacking};
use trxviz_core::data::nifti_data::NiftiVolume;
use trxviz_core::data::parcellation_data::ParcellationVolume;
use trxviz_core::data::trx_data::{RenderStyle, TrxGpuData};
use trxviz_core::renderer::glyph_renderer::GlyphResources;
use trxviz_core::renderer::mesh_renderer::MeshResources;
use trxviz_core::renderer::slice_renderer::AllSliceResources;
use trxviz_core::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};
use trxviz_core::scene::direct_streamline_import_warnings;

use super::*;

pub(crate) fn workflow_job_kind_title(kind: WorkflowJobKind) -> &'static str {
    match kind {
        WorkflowJobKind::ReactiveStreamline => "derived streamlines",
        WorkflowJobKind::SurfaceQuery => "surface depth query",
        WorkflowJobKind::SurfaceMap => "surface map",
        WorkflowJobKind::TubeGeometry => "tube geometry",
        WorkflowJobKind::BundleSurface => "bundle surface",
        WorkflowJobKind::BoundaryField => "boundary field",
    }
}

impl crate::app::TrxVizApp {
    pub(in crate::app) fn rebuild_workflow_editor_from_document(&mut self) {
        self.workflow.editor_snarl = snarl_from_graph(&self.workflow.document.graph);
    }

    pub(in crate::app) fn mark_workflow_semantic_edit(&mut self, at_time: f64) {
        self.workflow.document_revision += 1;
        self.workflow.last_semantic_edit_at = at_time;
        self.workflow.editor_interaction_active = true;
    }

    pub(in crate::app) fn mark_workflow_nonsemantic_edit(&mut self) {
        self.workflow.editor_interaction_active = true;
    }

    pub(in crate::app) fn poll_workflow_job_messages(&mut self) {
        let mut changed = false;
        while let Ok(message) = self.workflow.job_rx.try_recv() {
            match message {
                WorkflowJobMessage::Started {
                    node_uuid,
                    fingerprint,
                    ..
                } => {
                    if let Some(record) =
                        self.workflow.execution_cache.node_runs.get_mut(&node_uuid)
                        && record.current_fingerprint == Some(fingerprint)
                    {
                        record.status = WorkflowExecutionStatus::Running;
                        changed = true;
                    }
                }
                WorkflowJobMessage::Finished {
                    node_uuid,
                    fingerprint,
                    result,
                } => {
                    self.workflow.jobs_in_flight.remove(&node_uuid);
                    let Some(record) = self.workflow.execution_cache.node_runs.get_mut(&node_uuid)
                    else {
                        continue;
                    };
                    if record.current_fingerprint != Some(fingerprint) {
                        continue;
                    }
                    match result {
                        Ok(output) => match output {
                            WorkflowJobOutput::ReactiveStreamline(flow) => {
                                let summary =
                                    format!("{} streamlines", flow.selected_streamlines.len());
                                self.workflow
                                    .execution_cache
                                    .derived_streamline_cache
                                    .insert(node_uuid, CachedDerivedStreamline { flow });
                                mark_expensive_success(record, fingerprint, summary);
                                changed = true;
                            }
                            WorkflowJobOutput::SurfaceQuery(flow) => {
                                let summary =
                                    format!("{} streamlines", flow.selected_streamlines.len());
                                self.workflow
                                    .execution_cache
                                    .surface_query_cache
                                    .insert(node_uuid, CachedSurfaceQuery { flow });
                                mark_expensive_success(record, fingerprint, summary);
                                changed = true;
                            }
                            WorkflowJobOutput::SurfaceMap(map) => {
                                let summary = format!(
                                    "Surface streamline map for surface {}",
                                    map.surface_id
                                );
                                self.workflow
                                    .execution_cache
                                    .surface_streamline_map_cache
                                    .insert(node_uuid, CachedSurfaceStreamlineMap { map });
                                mark_expensive_success(record, fingerprint, summary);
                                changed = true;
                            }
                            WorkflowJobOutput::TubeGeometry { vertices, indices } => {
                                self.workflow.execution_cache.tube_geometry_cache.insert(
                                    node_uuid,
                                    CachedTubeGeometry {
                                        fingerprint,
                                        vertices,
                                        indices,
                                    },
                                );
                                mark_expensive_success(
                                    record,
                                    fingerprint,
                                    "Tube geometry ready".to_string(),
                                );
                                changed = true;
                            }
                            WorkflowJobOutput::BundleSurface { meshes } => {
                                let build_summary = if meshes.is_empty() {
                                    "Bundle surface is empty".to_string()
                                } else {
                                    format!("{} bundle surface mesh(es)", meshes.len())
                                };
                                self.workflow
                                    .execution_cache
                                    .bundle_surface_mesh_cache
                                    .insert(
                                        node_uuid,
                                        CachedBundleSurfaceMeshes {
                                            fingerprint,
                                            meshes,
                                        },
                                    );
                                mark_expensive_success(
                                    record,
                                    fingerprint,
                                    build_summary.clone(),
                                );
                                if let Some(draw) = self
                                    .workflow
                                    .runtime
                                    .scene_plan
                                    .bundle_draws
                                    .iter()
                                    .find(|draw| draw.node_uuid == node_uuid)
                                {
                                    let build_fingerprint = workflow_bundle_plan_fingerprint(
                                        &BundleSurfacePlan {
                                            build_node_uuid: draw.build_node_uuid,
                                            label: draw.label.clone(),
                                            flow: draw.flow.clone(),
                                            per_group: draw.per_group,
                                            voxel_size_mm: draw.voxel_size_mm,
                                            threshold: draw.threshold,
                                            smooth_sigma: draw.smooth_sigma,
                                            min_component_volume_mm3: draw
                                                .min_component_volume_mm3,
                                            opacity: draw.opacity,
                                        },
                                    );
                                    let build_record = self
                                        .workflow
                                        .execution_cache
                                        .node_runs
                                        .entry(draw.build_node_uuid)
                                        .or_default();
                                    mark_expensive_success(
                                        build_record,
                                        build_fingerprint,
                                        build_summary,
                                    );
                                }
                                changed = true;
                            }
                            WorkflowJobOutput::BoundaryField { field } => {
                                if let Some(field) = field {
                                    self.workflow.execution_cache.boundary_field_cache.insert(
                                        node_uuid,
                                        CachedBoundaryField { fingerprint, field },
                                    );
                                    mark_expensive_success(
                                        record,
                                        fingerprint,
                                        "Boundary field ready".to_string(),
                                    );
                                    changed = true;
                                } else {
                                    self.workflow
                                        .execution_cache
                                        .boundary_field_cache
                                        .remove(&node_uuid);
                                    mark_expensive_success(
                                        record,
                                        fingerprint,
                                        "Boundary field is empty".to_string(),
                                    );
                                    changed = true;
                                }
                            }
                        },
                        Err(error) => {
                            mark_expensive_failure(record, fingerprint, &error);
                            changed = true;
                        }
                    }
                }
            }
        }
        if changed {
            self.workflow.last_runtime_revision += 1;
        }
    }

    fn queue_workflow_job(
        &mut self,
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        kind: WorkflowJobKind,
        payload: WorkflowJobPayload,
    ) {
        if self.workflow.jobs_in_flight.contains_key(&node_uuid) {
            return;
        }
        let Some(record) = self.workflow.execution_cache.node_runs.get_mut(&node_uuid) else {
            return;
        };
        record.current_fingerprint = Some(fingerprint);
        record.status = WorkflowExecutionStatus::Queued;
        self.workflow
            .jobs_in_flight
            .insert(node_uuid, (kind, fingerprint));
        let tx = self.workflow.job_tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(WorkflowJobMessage::Started {
                node_uuid,
                fingerprint,
            });
            let result = run_workflow_job(payload);
            let _ = tx.send(WorkflowJobMessage::Finished {
                node_uuid,
                fingerprint,
                result,
            });
        });
    }

    pub(in crate::app) fn queue_workflow_jobs(&mut self) -> bool {
        for plan in self
            .workflow
            .runtime
            .scene_plan
            .reactive_streamline_plans
            .clone()
        {
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::ReactiveStreamline,
                    WorkflowJobPayload::ReactiveStreamline(plan),
                );
            }
        }

        if !self.workflow.run_expensive_requested && !self.workflow.run_session_active {
            return false;
        }

        let mut queued_any = false;
        self.workflow.run_session_active = true;

        for plan in self.workflow.runtime.scene_plan.surface_query_plans.clone() {
            let fingerprint =
                workflow_surface_query_fingerprint(&plan.flow, plan.surface_id, plan.depth_mm);
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::SurfaceQuery,
                    WorkflowJobPayload::SurfaceQuery(plan),
                );
                queued_any = true;
            }
        }

        for plan in self.workflow.runtime.scene_plan.surface_map_plans.clone() {
            let fingerprint = workflow_surface_projection_fingerprint(
                &plan.flow,
                plan.surface_id,
                plan.depth_mm,
                plan.dps_field.as_deref(),
            );
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::SurfaceMap,
                    WorkflowJobPayload::SurfaceMap(plan),
                );
                queued_any = true;
            }
        }

        for draw in self.workflow.runtime.scene_plan.streamline_draws.clone() {
            if draw.render_style != RenderStyle::Tubes {
                continue;
            }
            let fingerprint = workflow_streamline_fingerprint(&draw);
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&draw.node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                draw.node_uuid,
            ) {
                self.queue_workflow_job(
                    draw.node_uuid,
                    fingerprint,
                    WorkflowJobKind::TubeGeometry,
                    WorkflowJobPayload::TubeGeometry(draw),
                );
                queued_any = true;
            }
        }

        for plan in self
            .workflow
            .runtime
            .scene_plan
            .boundary_field_plans
            .clone()
        {
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            if should_queue_expensive_job(
                self.workflow
                    .execution_cache
                    .node_runs
                    .get(&plan.build_node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                plan.build_node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.build_node_uuid,
                    fingerprint,
                    WorkflowJobKind::BoundaryField,
                    WorkflowJobPayload::BoundaryField { plan },
                );
                queued_any = true;
            }
        }

        for draw in self.workflow.runtime.scene_plan.bundle_draws.clone() {
            let boundary_field = draw.boundary_field_node_uuid.and_then(|uuid| {
                self.workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.field.clone())
            });
            if draw.boundary_field_node_uuid.is_some() && boundary_field.is_none() {
                continue;
            }
            let fingerprint = workflow_bundle_display_fingerprint(
                &draw,
                draw.boundary_field_node_uuid.and_then(|uuid| {
                    self.workflow
                        .execution_cache
                        .boundary_field_cache
                        .get(&uuid)
                        .map(|cache| cache.fingerprint)
                }),
            );
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&draw.node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                draw.node_uuid,
            ) {
                let plan = BundleSurfacePlan {
                    build_node_uuid: draw.build_node_uuid,
                    label: draw.label.clone(),
                    flow: draw.flow.clone(),
                    per_group: draw.per_group,
                    voxel_size_mm: draw.voxel_size_mm,
                    threshold: draw.threshold,
                    smooth_sigma: draw.smooth_sigma,
                    min_component_volume_mm3: draw.min_component_volume_mm3,
                    opacity: draw.opacity,
                };
                self.queue_workflow_job(
                    draw.node_uuid,
                    fingerprint,
                    WorkflowJobKind::BundleSurface,
                    WorkflowJobPayload::BundleSurface {
                        plan,
                        color_mode: draw.color_mode,
                        boundary_field,
                    },
                );
                queued_any = true;
            }
        }

        self.workflow.run_expensive_requested = false;
        if !queued_any && self.workflow.jobs_in_flight.is_empty() {
            self.workflow.run_session_active = false;
        }
        queued_any
    }

    pub(in crate::app) fn refresh_workflow_runtime(&mut self, mode: WorkflowEvalMode) {
        ensure_node_uuids(&mut self.workflow.document);
        self.workflow.runtime = evaluate_scene_plan_with_mode(
            &self.workflow.document,
            &self.scene.trx_files,
            &self.scene.nifti_files,
            &self.scene.gifti_surfaces,
            &self.scene.parcellations,
            &mut self.workflow.display_runtimes,
            &mut self.workflow.next_draw_id,
            &mut self.workflow.execution_cache,
            mode,
        );
        self.workflow.last_runtime_revision += 1;
    }

    pub(in crate::app) fn refresh_workflow_runtime_if_needed(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|input| input.time);
        let needs_interactive = self.workflow.document_revision != self.workflow.last_interactive_revision;
        if needs_interactive {
            self.refresh_workflow_runtime(WorkflowEvalMode::Interactive);
            self.workflow.last_interactive_revision = self.workflow.document_revision;
        }

        let should_run_settled = self.workflow.run_expensive_requested
            || (self.workflow.document_revision != self.workflow.last_settled_revision
                && !self.workflow.editor_interaction_active
                && (now - self.workflow.last_semantic_edit_at) >= 0.150);
        if should_run_settled {
            self.refresh_workflow_runtime(WorkflowEvalMode::Settled);
            self.workflow.last_interactive_revision = self.workflow.document_revision;
            self.workflow.last_settled_revision = self.workflow.document_revision;
            self.workflow.editor_interaction_active = false;
        }
    }

    pub(in crate::app) fn sync_workflow_resources(&mut self, frame: &mut eframe::Frame) {
        if self.workflow.last_resource_sync_revision == self.workflow.last_runtime_revision {
            return;
        }
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        let mut renderer = rs.renderer.write();

        if renderer
            .callback_resources
            .get::<AllStreamlineResources>()
            .is_none()
        {
            renderer.callback_resources.insert(AllStreamlineResources {
                entries: Vec::new(),
            });
        }
        if renderer.callback_resources.get::<MeshResources>().is_none() {
            renderer
                .callback_resources
                .insert(MeshResources::new(&rs.device, rs.target_format));
        }
        if renderer
            .callback_resources
            .get::<GlyphResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(GlyphResources::new(&rs.device, rs.target_format));
        }

        let active_streamline_ids: HashSet<FileId> = self
            .workflow
            .runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| draw.draw_id)
            .collect();
        let active_bundle_ids: HashSet<FileId> = self
            .workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| draw.draw_id)
            .collect();
        let workflow_ids: HashSet<FileId> = self
            .workflow
            .display_runtimes
            .values()
            .map(|runtime| runtime.draw_id)
            .collect();

        if let Some(all) = renderer
            .callback_resources
            .get_mut::<AllStreamlineResources>()
        {
            for draw in &self.workflow.runtime.scene_plan.streamline_draws {
                let fingerprint = workflow_streamline_fingerprint(draw);
                let Some(runtime) = self.workflow.display_runtimes.get_mut(&draw.node_uuid) else {
                    log::error!(
                        "missing streamline display runtime for workflow node {:?}",
                        draw.node_uuid
                    );
                    continue;
                };
                let resource_exists = all.entries.iter().any(|(id, _)| *id == draw.draw_id);
                if draw.render_style == RenderStyle::Tubes
                    && !self
                        .workflow
                        .execution_cache
                        .node_runs
                        .get(&draw.node_uuid)
                        .is_some_and(|record| record.last_success_fingerprint == Some(fingerprint))
                {
                    continue;
                }
                if runtime.fingerprint == fingerprint && resource_exists {
                    continue;
                }

                let subset = materialize_flow_gpu(draw.flow.clone());
                let mut resource = StreamlineResources::new(&rs.device, rs.target_format, &subset);
                if draw.render_style == RenderStyle::Tubes {
                    let Some(cache) = self
                        .workflow
                        .execution_cache
                        .tube_geometry_cache
                        .get(&draw.node_uuid)
                        .filter(|cache| cache.fingerprint == fingerprint)
                    else {
                        continue;
                    };
                    resource.update_tube_geometry(&rs.device, &cache.vertices, &cache.indices);
                }

                if let Some(entry) = all.entries.iter_mut().find(|(id, _)| *id == draw.draw_id) {
                    *entry = (draw.draw_id, resource);
                } else {
                    all.entries.push((draw.draw_id, resource));
                }

                runtime.fingerprint = fingerprint;
            }

            all.entries
                .retain(|(id, _)| !workflow_ids.contains(id) || active_streamline_ids.contains(id));
        }

        if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
            for draw in &self.workflow.runtime.scene_plan.surface_draws {
                if let Some(scalars) = &draw.projection_scalars {
                    mesh_resources.update_surface_scalars(&rs.queue, draw.source_id, scalars);
                }
            }
        }

        let active_boundary_field_ids: HashSet<WorkflowNodeUuid> = self
            .workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .filter_map(|draw| draw.boundary_field_node_uuid)
            .chain(
                self.workflow
                    .runtime
                    .scene_plan
                    .boundary_glyph_draws
                    .iter()
                    .map(|draw| draw.build_node_uuid),
            )
            .collect();

        self.workflow
            .execution_cache
            .boundary_field_cache
            .retain(|uuid, _| active_boundary_field_ids.contains(uuid));

        if let Some(glyph_resources) = renderer.callback_resources.get_mut::<GlyphResources>() {
            if let Some(draw) = self
                .workflow
                .runtime
                .scene_plan
                .boundary_glyph_draws
                .iter()
                .find(|draw| draw.visible)
                .or_else(|| {
                    self.workflow
                        .runtime
                        .scene_plan
                        .boundary_glyph_draws
                        .first()
                })
            {
                if let Some(cache) = self
                    .workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&draw.build_node_uuid)
                {
                    let boundary_field_changed =
                        self.viewport.boundary_field_revision != cache.fingerprint;
                    let field = cache.field.clone();
                    glyph_resources.set_field(
                        &rs.device,
                        field.clone(),
                        draw.scale,
                        draw.min_contacts,
                    );
                    self.viewport.boundary_field = Some(field.clone());
                    self.viewport.boundary_field_revision = cache.fingerprint;
                    if boundary_field_changed && self.scene.nifti_files.is_empty() {
                        self.reset_slice_view_to_boundary_field(field.as_ref());
                    }
                } else {
                    glyph_resources.clear();
                    self.viewport.boundary_field = None;
                    self.viewport.boundary_field_revision = 0;
                }
            } else {
                glyph_resources.clear();
                self.viewport.boundary_field = None;
                self.viewport.boundary_field_revision = 0;
            }
        }

        if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
            for draw in &self.workflow.runtime.scene_plan.bundle_draws {
                let display_fingerprint = workflow_bundle_display_fingerprint(
                    draw,
                    draw.boundary_field_node_uuid.and_then(|uuid| {
                        self.workflow
                            .execution_cache
                            .boundary_field_cache
                            .get(&uuid)
                            .map(|cache| cache.fingerprint)
                    }),
                );
                let Some(runtime) = self.workflow.display_runtimes.get_mut(&draw.node_uuid) else {
                    log::error!(
                        "missing bundle display runtime for workflow node {:?}",
                        draw.node_uuid
                    );
                    continue;
                };
                let Some(cache) = self
                    .workflow
                    .execution_cache
                    .bundle_surface_mesh_cache
                    .get(&draw.node_uuid)
                    .filter(|cache| cache.fingerprint == display_fingerprint)
                else {
                    continue;
                };
                if !self
                    .workflow
                    .execution_cache
                    .node_runs
                    .get(&draw.node_uuid)
                    .is_some_and(|record| {
                        record.last_success_fingerprint == Some(display_fingerprint)
                    })
                {
                    continue;
                }
                if runtime.bundle_fingerprint == Some(display_fingerprint) {
                    continue;
                }
                runtime.bundle_meshes_cpu =
                    cache.meshes.iter().map(|(mesh, _)| mesh.clone()).collect();
                runtime.bundle_fingerprint = Some(display_fingerprint);

                if cache.meshes.is_empty() {
                    mesh_resources.clear_bundle_mesh(draw.draw_id);
                } else {
                    mesh_resources.set_bundle_meshes(draw.draw_id, &rs.device, &cache.meshes);
                }
            }

            for draw_id in workflow_ids
                .iter()
                .copied()
                .filter(|id| !active_bundle_ids.contains(id))
            {
                mesh_resources.clear_bundle_mesh(draw_id);
                if let Some(runtime) = self
                    .workflow
                    .display_runtimes
                    .values_mut()
                    .find(|runtime| runtime.draw_id == draw_id)
                {
                    runtime.bundle_fingerprint = None;
                    runtime.bundle_meshes_cpu.clear();
                }
            }
        }
        self.workflow.last_resource_sync_revision = self.workflow.last_runtime_revision;
    }

    pub(in crate::app) fn clear_loaded_scene(&mut self, frame: &mut eframe::Frame) {
        if let Some(rs) = frame.wgpu_render_state() {
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer
                .callback_resources
                .get_mut::<AllStreamlineResources>()
            {
                all.entries.clear();
            }
            if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                all.entries.clear();
            }
            if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
                for runtime in self.workflow.display_runtimes.values() {
                    mesh_resources.clear_bundle_mesh(runtime.draw_id);
                }
            }
            if let Some(glyph_resources) = renderer.callback_resources.get_mut::<GlyphResources>() {
                glyph_resources.clear();
            }
        }

        self.scene.trx_files.clear();
        self.scene.nifti_files.clear();
        self.scene.gifti_surfaces.clear();
        self.scene.parcellations.clear();
        self.pending_file_loads.clear();
        self.viewport.boundary_field = None;
        self.viewport.boundary_field_revision = 0;
        self.workflow.runtime = WorkflowRuntime::default();
        self.workflow.execution_cache = WorkflowExecutionCache::default();
        self.workflow.display_runtimes.clear();
        self.workflow.selection = None;
        self.workflow.node_feedback.clear();
        self.workflow.document = default_document();
        self.rebuild_workflow_editor_from_document();
        self.scene.next_file_id = 0;
        self.workflow.next_draw_id = 1_000_000;
        self.workflow.run_expensive_requested = false;
        self.workflow.run_session_active = false;
        self.workflow.jobs_in_flight.clear();
        self.workflow.document_revision += 1;
        self.workflow.last_interactive_revision = 0;
        self.workflow.last_settled_revision = 0;
        self.workflow.last_runtime_revision += 1;
        self.workflow.last_resource_sync_revision = 0;
        self.workflow.editor_interaction_active = false;
        self.workflow.last_semantic_edit_at = 0.0;
    }

    pub(in crate::app) fn new_workflow_project(&mut self, frame: &mut eframe::Frame) {
        self.clear_loaded_scene(frame);
        self.workflow.project_path = None;
        self.status_msg = Some("Started a new workflow project.".to_string());
        self.error_msg = None;
    }

    pub(in crate::app) fn save_workflow_project(&mut self, save_as: bool) {
        let target_path = if !save_as {
            self.workflow.project_path.clone()
        } else {
            None
        }
        .or_else(|| {
            rfd::FileDialog::new()
                .add_filter("Workflow Project", &["json"])
                .set_file_name("workflow.json")
                .save_file()
        });

        let Some(target_path) = target_path else {
            return;
        };

        match gui_save_project(
            &self.workflow.document,
            &self.workflow.workspace,
            &target_path,
        ) {
            Ok(()) => {
                self.workflow.project_path = Some(target_path.clone());
                self.status_msg = Some(format!(
                    "Saved workflow project to {}",
                    target_path.display()
                ));
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to save workflow project: {err}"));
            }
        }
    }

    pub(in crate::app) fn open_workflow_project(
        &mut self,
        path: PathBuf,
        frame: &mut eframe::Frame,
    ) {
        if frame.wgpu_render_state().is_none() {
            self.error_msg =
                Some("Cannot open a workflow project before the renderer is ready.".to_string());
            return;
        }

        let (project, workspace) = match gui_load_project(&path) {
            Ok(result) => result,
            Err(err) => {
                self.error_msg = Some(format!("Failed to read workflow project: {err}"));
                return;
            }
        };
        let mut workspace = workspace;
        if !workspace
            .tiles
            .tiles()
            .any(|tile| matches!(tile, egui_tiles::Tile::Pane(WorkspacePane::Preview)))
        {
            workspace = default_workspace_tree();
        }

        self.clear_loaded_scene(frame);
        let Some(rs) = frame.wgpu_render_state() else {
            self.error_msg =
                Some("Renderer state disappeared while opening the workflow project.".to_string());
            return;
        };

        for asset in project.document.assets.clone() {
            let load_result: Result<(), String> = match asset {
                WorkflowAssetDocument::Streamlines {
                    id,
                    path: asset_path,
                    imported,
                } => {
                    if imported {
                        trx_rs::read_tractogram(&asset_path, &ConversionOptions::default())
                            .map_err(|err| err.to_string())
                            .and_then(|tractogram| {
                                TrxGpuData::from_tractogram(&tractogram)
                                    .map_err(|err| err.to_string())
                                    .map(|data| crate::app::state::LoadedStreamlineSource {
                                        data,
                                        backing: StreamlineBacking::Imported(Arc::new(tractogram)),
                                        warnings: direct_streamline_import_warnings(
                                            &asset_path,
                                            &ConversionOptions {
                                                vtk_coordinate_mode:
                                                    trx_rs::VtkCoordinateMode::HeaderOrWarn,
                                                ..Default::default()
                                            },
                                        ),
                                    })
                            })
                            .map(|source| {
                                self.apply_loaded_trx_with_options(
                                    asset_path,
                                    source,
                                    rs,
                                    Some(id),
                                    false,
                                );
                            })
                    } else {
                        AnyTrxFile::load(&asset_path)
                            .map_err(|err| err.to_string())
                            .and_then(|any| {
                                TrxGpuData::from_any_trx(&any)
                                    .map_err(|err| err.to_string())
                                    .map(|data| crate::app::state::LoadedStreamlineSource {
                                        data,
                                        backing: StreamlineBacking::Native(Arc::new(any)),
                                        warnings: Vec::new(),
                                    })
                            })
                            .map(|source| {
                                self.apply_loaded_trx_with_options(
                                    asset_path,
                                    source,
                                    rs,
                                    Some(id),
                                    false,
                                );
                            })
                    }
                }
                WorkflowAssetDocument::Volume {
                    id,
                    path: asset_path,
                } => NiftiVolume::load(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|volume| {
                        self.apply_loaded_nifti_with_options(
                            asset_path,
                            volume,
                            rs,
                            Some(id),
                            false,
                        );
                    }),
                WorkflowAssetDocument::Surface {
                    id,
                    path: asset_path,
                } => GiftiSurfaceData::load(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|surface| {
                        self.apply_loaded_gifti_surface_with_options(
                            asset_path,
                            surface,
                            rs,
                            Some(id),
                            false,
                        );
                    }),
                WorkflowAssetDocument::Parcellation {
                    id,
                    path: asset_path,
                    label_table_path,
                } => ParcellationVolume::load(&asset_path, label_table_path.as_deref())
                    .map_err(|err| err.to_string())
                    .map(|data| {
                        self.apply_loaded_parcellation_with_options(
                            asset_path,
                            crate::app::state::LoadedParcellationSource {
                                data,
                                label_table_path,
                            },
                            Some(id),
                            false,
                        );
                    }),
            };

            if let Err(err) = load_result {
                self.error_msg = Some(format!("Failed to load workflow project asset: {err}"));
                return;
            }
        }

        self.workflow.document = project.document;
        self.workflow.workspace = workspace;
        ensure_node_uuids(&mut self.workflow.document);
        self.rebuild_workflow_editor_from_document();
        self.workflow.project_path = Some(path.clone());
        self.status_msg = Some(format!("Opened workflow project {}", path.display()));
        self.error_msg = None;
        self.workflow.document_revision += 1;
        self.workflow.last_interactive_revision = 0;
        self.workflow.last_settled_revision = 0;
        self.workflow.editor_interaction_active = false;
    }

    pub(in crate::app) fn save_streamline_node(&mut self, node_uuid: WorkflowNodeUuid) {
        let Some(plan) = self
            .workflow
            .runtime
            .save_streamline_targets
            .get(&node_uuid)
            .cloned()
        else {
            self.error_msg =
                Some("This save node does not have a connected streamline input.".to_string());
            return;
        };

        match save_streamline_plan(&plan) {
            Ok(()) => {
                self.workflow
                    .node_feedback
                    .insert(node_uuid, format!("Saved {}", plan.output_path.display()));
                self.status_msg = Some(format!(
                    "Saved streamlines to {}",
                    plan.output_path.display()
                ));
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to save streamlines: {err}"));
            }
        }
    }
}

fn should_queue_expensive_job(
    record: Option<&ExpensiveNodeRunRecord>,
    fingerprint: u64,
    in_flight: &HashMap<WorkflowNodeUuid, (WorkflowJobKind, u64)>,
    node_uuid: WorkflowNodeUuid,
) -> bool {
    if in_flight
        .get(&node_uuid)
        .is_some_and(|(_, queued_fingerprint)| *queued_fingerprint == fingerprint)
    {
        return false;
    }
    let Some(record) = record else {
        return true;
    };
    record.last_success_fingerprint != Some(fingerprint)
}

fn mark_expensive_failure(record: &mut ExpensiveNodeRunRecord, fingerprint: u64, error: &str) {
    record.current_fingerprint = Some(fingerprint);
    record.status = WorkflowExecutionStatus::Failed(error.to_string());
}

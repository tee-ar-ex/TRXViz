use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use trx_rs::{AnyTrxFile, ConversionOptions};
use trxviz_core::data::gifti_data::GiftiSurfaceData;
use trxviz_core::data::loaded_files::{FileId, StreamlineBacking};
use trxviz_core::data::nifti_data::NiftiVolume;
use trxviz_core::data::parcellation_data::ParcellationVolume;
use trxviz_core::data::trx_data::{RenderStyle, TrxGpuData};
use trxviz_core::headless::{
    HeadlessSceneExportFormat, HeadlessSceneExportOptions, HeadlessView, export_state_glb,
};
use trxviz_core::renderer::background_renderer::BackgroundResources;
use trxviz_core::renderer::glyph_renderer::{GlyphResources, OdxGlyphResourceKey, OdxGpuGlyphMode};
use trxviz_core::renderer::mesh_renderer::MeshResources;
use trxviz_core::renderer::slice_renderer::AllSliceResources;
use trxviz_core::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};
use trxviz_core::scene::HeadlessWorkflowState;
use trxviz_core::scene::direct_streamline_import_warnings;

use crate::app::callbacks::OdxFixelResources;

use super::*;

pub(crate) fn workflow_job_kind_title(kind: WorkflowJobKind) -> &'static str {
    match kind {
        WorkflowJobKind::ReactiveStreamline => "derived streamlines",
        WorkflowJobKind::SurfaceQuery => "surface depth query",
        WorkflowJobKind::SurfaceMap => "surface map",
        WorkflowJobKind::TubeGeometry => "tube geometry",
        WorkflowJobKind::BundleSurface => "bundle surface",
        WorkflowJobKind::BoundaryField => "boundary field",
        WorkflowJobKind::DipyTractography => "dipy tractography",
        WorkflowJobKind::YehTractography => "yeh tractography",
    }
}

fn hash_f32(hasher: &mut DefaultHasher, value: f32) {
    value.to_bits().hash(hasher);
}

fn hash_volume_scalars(
    hasher: &mut DefaultHasher,
    scalars: Option<&trxviz_core::data::cifti::VolumeScalars>,
) {
    match scalars {
        Some(volume) => {
            true.hash(hasher);
            volume.dims.hash(hasher);
            volume.values.len().hash(hasher);
            volume.metadata.map_name.hash(hasher);
            volume.metadata.series_index.hash(hasher);
            volume
                .metadata
                .suggested_range
                .map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
                .hash(hasher);
            volume.metadata.series_value.map(f32::to_bits).hash(hasher);
            for entry in volume.voxel_to_ras.to_cols_array() {
                hash_f32(hasher, entry);
            }
            for idx in [
                0,
                volume.values.len() / 2,
                volume.values.len().saturating_sub(1),
            ] {
                if let Some(value) = volume.values.get(idx) {
                    hash_f32(hasher, *value);
                }
            }
        }
        None => false.hash(hasher),
    }
}

fn active_fixel_draw_3d(
    app: &crate::app::TrxVizApp,
) -> Option<&trxviz_core::workflow::FixelDrawPlan> {
    app.workflow
        .runtime
        .scene_plan
        .fixel_3d_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| app.workflow.runtime.scene_plan.fixel_3d_draws.first())
}

fn active_fixel_draw_2d(
    app: &crate::app::TrxVizApp,
) -> Option<&trxviz_core::workflow::FixelDrawPlan> {
    app.workflow
        .runtime
        .scene_plan
        .fixel_2d_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| app.workflow.runtime.scene_plan.fixel_2d_draws.first())
}

fn fixel_upload_fingerprint(plan: &trxviz_core::workflow::FixelDrawPlan) -> u64 {
    use trxviz_core::data::odx_data::FixelScalarValues;

    let mut hasher = DefaultHasher::new();
    plan.field.source_id.hash(&mut hasher);
    plan.field.colormap_code.hash(&mut hasher);
    plan.field.scalars.name.hash(&mut hasher);
    match &plan.field.scalars.values {
        FixelScalarValues::Rgb(_) => 0u8.hash(&mut hasher),
        FixelScalarValues::Scalar(_) => 1u8.hash(&mut hasher),
    }
    hasher.finish()
}

fn active_odx_glyph_plan(
    app: &crate::app::TrxVizApp,
) -> Option<&trxviz_core::workflow::OdfGlyphDrawPlan> {
    app.workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| app.workflow.runtime.scene_plan.odf_glyph_draws.first())
}

fn active_odx_glyph_scene(
    app: &crate::app::TrxVizApp,
) -> Option<&Arc<trxviz_core::data::odx_data::OdxScene>> {
    active_odx_glyph_plan(app)
        .map(|plan| &plan.field.scene)
        .or(app.scene.odx_scene.as_ref())
}

fn active_odx_slice_state(app: &crate::app::TrxVizApp) -> Option<(usize, u32)> {
    if let Some(plan) = active_odx_glyph_plan(app) {
        let viewport_index = plan.slice_axis.viewport_index();
        return Some((
            plan.slice_axis.odx_axis(),
            app.viewport.slice_index(viewport_index) as u32,
        ));
    }
    app.scene
        .odx_scene
        .as_ref()
        .map(|_| (2usize, app.viewport.slice_index(0) as u32))
}

fn clamped_active_odx_sh_detail(
    app: &crate::app::TrxVizApp,
    scene: &trxviz_core::data::odx_data::OdxScene,
    slice_state: Option<(usize, u32)>,
) -> u32 {
    let requested = active_odx_glyph_plan(app)
        .map(|draw| draw.detail)
        .unwrap_or(trxviz_core::workflow::default_odf_glyph_detail());
    let Some((axis, slice_idx)) = slice_state else {
        return requested;
    };
    let Some(limit) = app.max_storage_buffer_binding_size else {
        return requested;
    };
    scene.clamp_sh_detail_for_slice(axis, slice_idx, requested, limit)
}

fn active_odx_glyph_resource_key(
    app: &crate::app::TrxVizApp,
    _device: &wgpu::Device,
) -> Option<OdxGlyphResourceKey> {
    let scene = active_odx_glyph_scene(app)?;
    let plan = active_odx_glyph_plan(app);
    make_odx_glyph_resource_key(
        scene,
        plan,
        active_odx_slice_state(app),
        app.max_storage_buffer_binding_size,
    )
}

fn make_odx_glyph_resource_key(
    scene: &Arc<trxviz_core::data::odx_data::OdxScene>,
    plan: Option<&trxviz_core::workflow::OdfGlyphDrawPlan>,
    slice_state: Option<(usize, u32)>,
    max_storage_buffer_binding_size: Option<usize>,
) -> Option<OdxGlyphResourceKey> {
    let source_kind = scene.glyph_source_kind()?;
    let mode = match source_kind {
        trxviz_core::data::odx_data::OdxGlyphSourceKind::Odf => OdxGpuGlyphMode::OdfSliceGather,
        trxviz_core::data::odx_data::OdxGlyphSourceKind::Sh => OdxGpuGlyphMode::ShCompute,
    };
    let (sphere_vertex_count, sphere_index_count, sh_detail) = match source_kind {
        trxviz_core::data::odx_data::OdxGlyphSourceKind::Odf => {
            let (vertices, indices) = scene.odf_render_geometry()?;
            (vertices.len(), indices.len(), None)
        }
        trxviz_core::data::odx_data::OdxGlyphSourceKind::Sh => {
            let requested = plan
                .map(|draw| draw.detail)
                .unwrap_or(trxviz_core::workflow::default_odf_glyph_detail());
            let detail = match (slice_state, max_storage_buffer_binding_size) {
                (Some((axis, slice_idx)), Some(limit)) => {
                    scene.clamp_sh_detail_for_slice(axis, slice_idx, requested, limit)
                }
                _ => requested,
            };
            let mesh = scene.sh_render_mesh(detail)?;
            (mesh.vertices().len(), mesh.indices().len(), Some(detail))
        }
    };
    let mut opacity_hasher = DefaultHasher::new();
    hash_volume_scalars(
        &mut opacity_hasher,
        plan.and_then(|draw| draw.opacity_scalars.as_ref()),
    );
    let mut size_hasher = DefaultHasher::new();
    hash_volume_scalars(
        &mut size_hasher,
        plan.and_then(|draw| draw.size_scalars.as_ref()),
    );
    Some(OdxGlyphResourceKey {
        scene_ptr: Arc::as_ptr(scene) as usize,
        source_kind,
        mode,
        sphere_vertex_count,
        sphere_index_count,
        sh_order: scene.sh_order(),
        sh_detail,
        slice_axis: slice_state.map(|(axis, _)| axis as u8),
        slice_index: slice_state.map(|(_, slice_idx)| slice_idx),
        subtract_iso: plan.map(|draw| draw.subtract_iso).unwrap_or(true),
        norm_within_voxel: plan.map(|draw| draw.norm_within_voxel).unwrap_or(false),
        opacity_gate_fingerprint: opacity_hasher.finish(),
        size_gate_fingerprint: size_hasher.finish(),
    })
}

fn odf_rows_per_chunk(
    scene: &trxviz_core::data::odx_data::OdxScene,
    device: &wgpu::Device,
) -> usize {
    scene
        .odf_rows_per_chunk(device.limits().max_storage_buffer_binding_size as usize)
        .unwrap_or(1)
}

fn sample_odx_gate_buffers(
    app: &crate::app::TrxVizApp,
    instances: &[trxviz_core::renderer::glyph_renderer::GlyphInstance],
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    use trxviz_core::data::odx_data::sample_volume_scalars_for_glyphs;

    let Some(plan) = active_odx_glyph_plan(app) else {
        return (None, None);
    };
    let opacity = plan
        .opacity_scalars
        .as_ref()
        .map(|volume| sample_volume_scalars_for_glyphs(instances, Some(volume)));
    let size = plan
        .size_scalars
        .as_ref()
        .map(|volume| sample_volume_scalars_for_glyphs(instances, Some(volume)));
    (opacity, size)
}

impl crate::app::TrxVizApp {
    fn effective_workflow_node_sizes(&self) -> HashMap<WorkflowNodeUuid, NodeSize> {
        let mut sizes = estimated_workflow_node_sizes(&self.workflow.document.graph);
        for (uuid, size) in &self.workflow.measured_node_sizes {
            if self.workflow.document.graph.contains(*uuid) {
                sizes.insert(*uuid, *size);
            }
        }
        sizes
    }

    fn retain_workflow_node_measurements(&mut self) {
        self.workflow
            .measured_node_sizes
            .retain(|uuid, _| self.workflow.document.graph.contains(*uuid));
        self.workflow
            .layout_reflow_nodes
            .retain(|uuid| self.workflow.document.graph.contains(*uuid));
        self.workflow.layout_reflow_pending = !self.workflow.layout_reflow_nodes.is_empty();
    }

    pub(in crate::app) fn arrange_workflow_graph(&mut self) -> Option<GraphRect> {
        self.retain_workflow_node_measurements();
        if self.workflow.document.graph.is_empty() {
            return None;
        }
        let sizes = self.effective_workflow_node_sizes();
        let layout = layout_workflow_graph(
            &self.workflow.document.graph,
            &sizes,
            &WorkflowLayoutOptions::default(),
        );
        apply_workflow_layout(&mut self.workflow.document.graph, &layout);
        self.workflow.layout_reflow_nodes.clear();
        self.workflow.layout_reflow_pending = false;
        self.rebuild_workflow_editor_from_document();
        Some(layout.bounds)
    }

    pub(in crate::app) fn apply_pending_workflow_layout_reflow(&mut self) -> bool {
        self.retain_workflow_node_measurements();
        if !self.workflow.layout_reflow_pending || self.workflow.layout_reflow_nodes.is_empty() {
            return false;
        }

        let seeds: Vec<_> = self.workflow.layout_reflow_nodes.iter().copied().collect();
        let component_nodes = weakly_connected_closure(&self.workflow.document.graph, &seeds);
        self.workflow.layout_reflow_nodes.clear();
        self.workflow.layout_reflow_pending = false;
        if component_nodes.is_empty() {
            return false;
        }

        let sizes = self.effective_workflow_node_sizes();
        let layout = layout_workflow_graph_subset(
            &self.workflow.document.graph,
            &sizes,
            &component_nodes,
            None,
            &WorkflowLayoutOptions::default(),
        );
        apply_workflow_layout(&mut self.workflow.document.graph, &layout);
        self.rebuild_workflow_editor_from_document();
        true
    }

    pub(in crate::app) fn ensure_active_odx_glyph_resources(
        &mut self,
        callback_resources: &mut egui_wgpu::CallbackResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let Some(glyph_resources) = callback_resources.get_mut::<GlyphResources>() else {
            return;
        };
        let Some(resource_key) = active_odx_glyph_resource_key(self, device) else {
            self.odx_amp_norm = 1.0;
            self.workflow.uploaded_odx_glyph_resource_key = None;
            return;
        };
        if self.workflow.uploaded_odx_glyph_resource_key == Some(resource_key)
            && glyph_resources.has_geometry()
        {
            return;
        }

        let scene = active_odx_glyph_scene(self)
            .expect("ODX scene for active resource key")
            .clone();
        let (axis, slice_idx) =
            active_odx_slice_state(self).expect("slice state for ODX glyph mode");

        match resource_key.mode {
            OdxGpuGlyphMode::PreSampledOdf => unreachable!("ODX glyphs now use slice-local upload"),
            OdxGpuGlyphMode::OdfSliceGather => {
                let (sphere_vertices, sphere_indices) = scene
                    .odf_render_geometry()
                    .expect("ODF geometry should exist for ODF glyph mode");
                let rows_per_chunk = odf_rows_per_chunk(&scene, device);
                let metadata = scene
                    .odf_slice_metadata(axis, slice_idx, rows_per_chunk)
                    .expect("ODF slice metadata should exist for ODF glyph mode");
                if metadata.instances.is_empty() {
                    glyph_resources.clear();
                    self.odx_amp_norm = 1.0;
                    self.workflow.uploaded_odx_glyph_resource_key = None;
                    return;
                }
                let odf_view = scene
                    .odf_view_f32()
                    .expect("ODF amplitudes should exist for ODF glyph mode");
                let (opacity_samples, size_samples) =
                    sample_odx_gate_buffers(self, &metadata.instances);
                self.odx_amp_norm = 1.0;
                glyph_resources.set_odx_slice_gather(
                    device,
                    queue,
                    resource_key.scene_ptr,
                    sphere_vertices,
                    sphere_indices,
                    &metadata.instances,
                    odf_view.as_flat_slice(),
                    scene.compact_voxel_count(),
                    scene
                        .odf_source_row_width()
                        .expect("ODF row width should exist for ODF glyph mode"),
                    scene
                        .odf_render_row_width()
                        .expect("ODF render row width should exist for ODF glyph mode"),
                    rows_per_chunk,
                    &metadata.chunk_worklists,
                    metadata.amp_norm,
                    scene.default_normalized_peak_length_mm(),
                    resource_key.subtract_iso,
                    resource_key.norm_within_voxel,
                    opacity_samples.as_deref(),
                    size_samples.as_deref(),
                );
            }
            OdxGpuGlyphMode::ShCompute => {
                let detail = clamped_active_odx_sh_detail(self, &scene, Some((axis, slice_idx)));
                let requested = active_odx_glyph_plan(self)
                    .map(|draw| draw.detail)
                    .unwrap_or(trxviz_core::workflow::default_odf_glyph_detail());
                if detail < requested {
                    self.status_msg = Some(format!(
                        "SH detail clamped to {detail} by GPU storage limit for the current slice."
                    ));
                }
                let mesh = scene
                    .sh_render_mesh(detail)
                    .expect("SH render mesh should exist for SH glyph mode");
                let instances = scene.glyph_instances_for_slice(axis, slice_idx, mesh.row_width());
                if instances.is_empty() {
                    glyph_resources.clear();
                    self.odx_amp_norm = 1.0;
                    self.workflow.uploaded_odx_glyph_resource_key = None;
                    return;
                }
                let coefficients = scene
                    .sh_coefficients_for_slice(axis, slice_idx)
                    .expect("SH coefficients should exist for SH glyph mode");
                let ncoeffs = scene
                    .sh_view_f32()
                    .expect("SH coefficients should exist for SH glyph mode")
                    .ncols();
                let (opacity_samples, size_samples) = sample_odx_gate_buffers(self, &instances);
                self.odx_amp_norm = 1.0;
                glyph_resources.set_odx_sh_volume(
                    device,
                    mesh.vertices(),
                    mesh.indices(),
                    &instances,
                    &coefficients,
                    ncoeffs,
                    mesh.transform_flat(),
                    mesh.source_dir_count(),
                    mesh.row_width(),
                    scene.default_normalized_peak_length_mm(),
                    resource_key.subtract_iso,
                    resource_key.norm_within_voxel,
                    opacity_samples.as_deref(),
                    size_samples.as_deref(),
                );
                let slice_indices: Vec<u32> = (0..instances.len() as u32).collect();
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("interactive_odx_sh_slice_encoder"),
                });
                glyph_resources.dispatch_odx_sh_slice(device, queue, &mut encoder, &slice_indices);
                queue.submit(std::iter::once(encoder.finish()));
            }
        }

        self.workflow.uploaded_odx_glyph_resource_key = Some(resource_key);
    }

    pub(in crate::app) fn update_active_odx_slice_state(
        &mut self,
        callback_resources: &mut egui_wgpu::CallbackResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.ensure_active_odx_glyph_resources(callback_resources, device, queue);
    }

    pub(in crate::app) fn rebuild_workflow_editor_from_document(&mut self) {
        self.retain_workflow_node_measurements();
        self.workflow.editor_snarl = snarl_from_graph(&self.workflow.document.graph);
    }

    pub(in crate::app) fn mark_workflow_semantic_edit(&mut self, at_time: f64) {
        self.workflow.document_revision += 1;
        self.workflow.last_semantic_edit_at = at_time;
        self.workflow.editor_interaction_active = true;
    }

    /// Mark a change to a render-only parameter (color, opacity, visibility toggle,
    /// outline thickness, etc.).  The graph is re-evaluated immediately on the next
    /// frame so the scene plan updates, but document_revision is NOT incremented, so
    /// no fingerprints become stale and the "Run Expensive Nodes" flow is never
    /// triggered by pure display changes.
    pub(in crate::app) fn mark_render_only_edit(&mut self) {
        self.workflow.render_only_changed = true;
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
                WorkflowJobMessage::Progress {
                    node_uuid,
                    fingerprint,
                    done,
                    total,
                } => {
                    // Ignore progress for an obsolete fingerprint — if
                    // the user edited params mid-run, the in-flight
                    // worker is now running an obsolete job that will
                    // be silently dropped on Finished. No point
                    // animating a bar for it.
                    let is_current = self
                        .workflow
                        .execution_cache
                        .node_runs
                        .get(&node_uuid)
                        .and_then(|r| r.current_fingerprint)
                        == Some(fingerprint);
                    if is_current {
                        self.workflow.job_progress.insert(node_uuid, (done, total));
                        changed = true;
                    }
                }
                WorkflowJobMessage::Finished {
                    node_uuid,
                    fingerprint,
                    result,
                } => {
                    self.workflow.jobs_in_flight.remove(&node_uuid);
                    self.workflow.cancel_flags.remove(&node_uuid);
                    self.workflow.job_progress.remove(&node_uuid);

                    // Node removed from the document while the job ran —
                    // no record to attribute the result to.
                    if !self
                        .workflow
                        .execution_cache
                        .node_runs
                        .contains_key(&node_uuid)
                    {
                        continue;
                    }

                    // Did the node's fingerprint move on while the job ran?
                    // If so, we still cache the output below (no compute
                    // wasted) but we must NOT advance the record's
                    // fingerprints — staleness detection would break and
                    // mark the node as "Ready" for parameters the user
                    // has since edited past.
                    let fingerprint_current = self
                        .workflow
                        .execution_cache
                        .node_runs
                        .get(&node_uuid)
                        .and_then(|r| r.current_fingerprint)
                        == Some(fingerprint);

                    let node_label = self
                        .workflow
                        .document
                        .graph
                        .get(node_uuid)
                        .map(|n| n.label.clone())
                        .unwrap_or_else(|| "node".to_string());

                    match result {
                        Ok(output) => {
                            // Cache the output regardless of currency.
                            // Downstream evaluators check the cache's own
                            // `fingerprint` field against the expected
                            // value, so obsolete entries are naturally
                            // treated as stale and re-queued.
                            let summary = match output {
                                WorkflowJobOutput::ReactiveStreamline(flow) => {
                                    let s =
                                        format!("{} streamlines", flow.selected_streamlines.len());
                                    self.workflow
                                        .execution_cache
                                        .derived_streamline_cache
                                        .insert(node_uuid, CachedDerivedStreamline { flow });
                                    s
                                }
                                WorkflowJobOutput::SurfaceQuery(flow) => {
                                    let s =
                                        format!("{} streamlines", flow.selected_streamlines.len());
                                    self.workflow
                                        .execution_cache
                                        .surface_query_cache
                                        .insert(node_uuid, CachedSurfaceQuery { flow });
                                    s
                                }
                                WorkflowJobOutput::SurfaceMap(map) => {
                                    let s =
                                        format!("Surface scalars ({} values)", map.values.len());
                                    self.workflow
                                        .execution_cache
                                        .surface_streamline_map_cache
                                        .insert(node_uuid, CachedSurfaceStreamlineMap { map });
                                    s
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
                                    "Tube geometry ready".to_string()
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
                                    // Sibling "build node" bookkeeping is
                                    // only meaningful for current results;
                                    // advancing it for obsolete ones would
                                    // falsely mark the build as up-to-date.
                                    if fingerprint_current
                                        && let Some(draw) = self
                                            .workflow
                                            .runtime
                                            .scene_plan
                                            .bundle_draws
                                            .iter()
                                            .find(|draw| draw.node_uuid == node_uuid)
                                    {
                                        let build_fingerprint =
                                            workflow_bundle_plan_fingerprint(&BundleSurfacePlan {
                                                build_node_uuid: draw.build_node_uuid,
                                                label: draw.label.clone(),
                                                flow: draw.flow.clone(),
                                                per_group: draw.per_group,
                                                build_mode: draw.build_mode,
                                                voxel_size_mm: draw.voxel_size_mm,
                                                threshold: draw.threshold,
                                                smooth_sigma: draw.smooth_sigma,
                                                min_component_volume_mm3: draw
                                                    .min_component_volume_mm3,
                                                tube_radius_mm: draw.tube_radius_mm,
                                                tube_sides: draw.tube_sides,
                                                opacity: draw.opacity,
                                            });
                                        let build_record = self
                                            .workflow
                                            .execution_cache
                                            .node_runs
                                            .entry(draw.build_node_uuid)
                                            .or_default();
                                        mark_expensive_success(
                                            build_record,
                                            build_fingerprint,
                                            build_summary.clone(),
                                        );
                                    }
                                    build_summary
                                }
                                WorkflowJobOutput::BoundaryField { field } => {
                                    if let Some(field) = field {
                                        self.workflow.execution_cache.boundary_field_cache.insert(
                                            node_uuid,
                                            CachedBoundaryField { fingerprint, field },
                                        );
                                        "Boundary field ready".to_string()
                                    } else {
                                        self.workflow
                                            .execution_cache
                                            .boundary_field_cache
                                            .remove(&node_uuid);
                                        "Boundary field is empty".to_string()
                                    }
                                }
                                WorkflowJobOutput::DipyTractography { flow } => {
                                    let s =
                                        format!("{} streamlines", flow.selected_streamlines.len());
                                    self.workflow
                                        .execution_cache
                                        .dipy_tractography_results
                                        .insert(
                                            node_uuid,
                                            CachedTractographyResult { fingerprint, flow },
                                        );
                                    s
                                }
                                WorkflowJobOutput::YehTractography { flow } => {
                                    let s =
                                        format!("{} streamlines", flow.selected_streamlines.len());
                                    self.workflow
                                        .execution_cache
                                        .yeh_tractography_results
                                        .insert(
                                            node_uuid,
                                            CachedTractographyResult { fingerprint, flow },
                                        );
                                    s
                                }
                            };

                            if fingerprint_current {
                                if let Some(record) =
                                    self.workflow.execution_cache.node_runs.get_mut(&node_uuid)
                                {
                                    mark_expensive_success(record, fingerprint, summary);
                                }
                            } else {
                                self.status_msg = Some(format!(
                                    "'{node_label}' finished but parameters changed — \
                                     result cached; re-run with current settings to refresh"
                                ));
                            }
                            changed = true;
                        }
                        Err(error) => {
                            let err_text = error.to_string();
                            let is_cancel =
                                matches!(error, trxviz_core::workflow::WorkflowError::Cancelled);
                            if fingerprint_current {
                                if let Some(record) =
                                    self.workflow.execution_cache.node_runs.get_mut(&node_uuid)
                                {
                                    // Cancel gets a dedicated mark so the
                                    // scheduler doesn't immediately
                                    // re-dispatch a fresh run at the same
                                    // fingerprint. Non-cancel errors go
                                    // through Failed (which keeps the door
                                    // open for retry — transient errors may
                                    // resolve on their own).
                                    if is_cancel {
                                        trxviz_core::workflow::mark_expensive_cancelled(
                                            record,
                                            fingerprint,
                                        );
                                    } else {
                                        mark_expensive_failure(record, fingerprint, &err_text);
                                    }
                                }
                                // Cancellation is a user action, not a
                                // failure — neutral status toast, not
                                // a red error.
                                if is_cancel {
                                    self.status_msg = Some(format!("'{node_label}' cancelled."));
                                } else {
                                    self.error_msg =
                                        Some(format!("'{node_label}' failed: {err_text}"));
                                }
                            } else {
                                let verb = if is_cancel { "cancelled" } else { "failed" };
                                self.status_msg = Some(format!(
                                    "'{node_label}' {verb} for obsolete parameters; \
                                     current run may still succeed"
                                ));
                            }
                            changed = true;
                        }
                    }
                }
            }
        }
        if changed {
            self.workflow.last_runtime_revision += 1;
            self.workflow.pending_job_completion = true;
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
        // Fresh CancelFlag per job. The worker gets a clone; the GUI
        // keeps the original in `cancel_flags` so a Cancel click on
        // this node can flip it. Entry is removed when the job
        // finishes (whether via success, failure, or cancellation).
        //
        // The progress callback forwards `WorkflowJobMessage::Progress`
        // through the same channel that carries Started / Finished.
        // `tx_for_progress` is a separate clone so the spawn closure's
        // own `tx` isn't moved twice; `mpsc::Sender::clone` is cheap.
        let tx = self.workflow.job_tx.clone();
        let tx_for_progress = tx.clone();
        let cancel =
            trxviz_core::workflow::CancelFlag::with_progress_callback(move |done, total| {
                let _ = tx_for_progress.send(WorkflowJobMessage::Progress {
                    node_uuid,
                    fingerprint,
                    done,
                    total,
                });
            });
        self.workflow.cancel_flags.insert(node_uuid, cancel.clone());
        std::thread::spawn(move || {
            let _ = tx.send(WorkflowJobMessage::Started {
                node_uuid,
                fingerprint,
            });
            let result = run_workflow_job(payload, cancel);
            let _ = tx.send(WorkflowJobMessage::Finished {
                node_uuid,
                fingerprint,
                result,
            });
        });
    }

    /// Flip the cancel flag for any in-flight job on this node. The
    /// worker observes the flip on its next poll (every ~1024 seeds
    /// for CPU, every batch for GPU) and returns
    /// `WorkflowError::Cancelled`. The GUI's Finished handler then
    /// emits a neutral "Cancelled" toast instead of a red error.
    /// No-op when no job is in flight for `node_uuid`.
    pub(in crate::app) fn request_cancel_workflow_job(&self, node_uuid: WorkflowNodeUuid) {
        if let Some(flag) = self.workflow.cancel_flags.get(&node_uuid) {
            flag.request_cancel();
        }
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

        // An explicit Run click clears any prior cancellations so the
        // user can retry at the same fingerprint just by clicking Run
        // again. Without this, a cancelled node would stay cancelled
        // until the user edited a param.
        if self.workflow.run_expensive_requested {
            for record in self.workflow.execution_cache.node_runs.values_mut() {
                record.last_cancelled_fingerprint = None;
            }
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
                plan.dps_field.as_ref().map(|field| field.as_str()),
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

        for plan in self
            .workflow
            .runtime
            .scene_plan
            .dipy_tractography_plans
            .clone()
        {
            let node_uuid = plan.node_uuid;
            // The plan is authoritative: its `fingerprint` field was
            // computed by the op during evaluate() from the op's config
            // + upstream identity, and travels with the plan through
            // dispatch / completion. Replaces the previous
            // `node_runs.current_fingerprint.unwrap_or(0)` fallback that
            // produced PR 1's silent-discard race on first-run.
            let fingerprint = plan.fingerprint.0;
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                node_uuid,
            ) {
                self.queue_workflow_job(
                    node_uuid,
                    fingerprint,
                    WorkflowJobKind::DipyTractography,
                    WorkflowJobPayload::DipyTractography {
                        plan,
                        device: self.gpu_device.clone(),
                        queue: self.gpu_queue.clone(),
                    },
                );
                queued_any = true;
            }
        }

        for plan in self
            .workflow
            .runtime
            .scene_plan
            .yeh_tractography_plans
            .clone()
        {
            let node_uuid = plan.node_uuid;
            // Plan-authoritative fingerprint — see the Dipy queue loop
            // above for rationale.
            let fingerprint = plan.fingerprint.0;
            if should_queue_expensive_job(
                self.workflow.execution_cache.node_runs.get(&node_uuid),
                fingerprint,
                &self.workflow.jobs_in_flight,
                node_uuid,
            ) {
                self.queue_workflow_job(
                    node_uuid,
                    fingerprint,
                    WorkflowJobKind::YehTractography,
                    WorkflowJobPayload::YehTractography { plan },
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
                    build_mode: draw.build_mode,
                    voxel_size_mm: draw.voxel_size_mm,
                    threshold: draw.threshold,
                    smooth_sigma: draw.smooth_sigma,
                    min_component_volume_mm3: draw.min_component_volume_mm3,
                    tube_radius_mm: draw.tube_radius_mm,
                    tube_sides: draw.tube_sides,
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
            &self.scene.cifti_files,
            &self.scene.gifti_surfaces,
            &self.scene.parcellations,
            &self.scene.odx_files,
            &mut self.workflow.display_runtimes,
            &mut self.workflow.next_draw_id,
            &mut self.workflow.execution_cache,
            mode,
        );
        self.workflow.last_runtime_revision += 1;
    }

    pub(in crate::app) fn refresh_workflow_runtime_if_needed(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|input| input.time);
        // Snapshot the job-completion latch up front: we need it in
        // both the Interactive and Settled gates, and the old code
        // cleared it inside the Interactive branch before the Settled
        // check had a chance to see it. Inline ops (TIP, Purifibre)
        // that depend on an expensive upstream (e.g. a BoundaryField
        // that just built in the background) need a Settled pass to
        // actually rebuild — TIP in particular gates its work on
        // `ctx.eval_mode == WorkflowEvalMode::Settled`, so an
        // Interactive-only refresh leaves it "Stale" indefinitely.
        let job_completion = self.workflow.pending_job_completion;
        let needs_interactive = self.workflow.document_revision
            != self.workflow.last_interactive_revision
            || self.workflow.render_only_changed
            || job_completion;
        if needs_interactive {
            self.refresh_workflow_runtime(WorkflowEvalMode::Interactive);
            self.workflow.last_interactive_revision = self.workflow.document_revision;
            self.workflow.render_only_changed = false;
            if self.workflow.pending_stage_camera_fit {
                self.reset_inflated_stage_camera();
                self.workflow.pending_stage_camera_fit = false;
            }
        }

        let should_run_settled = self.workflow.run_expensive_requested
            || job_completion
            || (self.workflow.document_revision != self.workflow.last_settled_revision
                && !self.workflow.editor_interaction_active
                && (now - self.workflow.last_semantic_edit_at) >= 0.150);
        if should_run_settled {
            self.refresh_workflow_runtime(WorkflowEvalMode::Settled);
            self.workflow.last_interactive_revision = self.workflow.document_revision;
            self.workflow.last_settled_revision = self.workflow.document_revision;
            self.workflow.editor_interaction_active = false;
        }

        // Single clear at the end so both branches above see the same
        // snapshot. Double-evaluating in one frame (Interactive +
        // Settled) when a job completes is the same pattern an
        // explicit Run click already produces, so no new risk.
        self.workflow.pending_job_completion = false;
    }

    pub(in crate::app) fn sync_workflow_resources(&mut self, frame: &mut eframe::Frame) {
        if self.workflow.last_resource_sync_revision == self.workflow.last_runtime_revision {
            return;
        }
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        if self.gpu_device.is_none() {
            self.gpu_device = Some(rs.device.clone());
            self.gpu_queue = Some(rs.queue.clone());
        }

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
        if renderer
            .callback_resources
            .get::<BackgroundResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(BackgroundResources::new(&rs.device, rs.target_format));
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
            self.workflow.uploaded_odx_glyph_resource_key = None;
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
            .chain(
                self.workflow
                    .runtime
                    .scene_plan
                    .voxel_mask_mesh_draws
                    .iter()
                    .map(|draw| draw.draw_id),
            )
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
            for draw in self
                .workflow
                .runtime
                .scene_plan
                .surface_draws
                .iter()
                .chain(self.workflow.runtime.scene_plan.stage_surface_draws.iter())
            {
                if !draw.vertex_rgba.is_empty() {
                    mesh_resources.update_surface_colors(
                        &rs.queue,
                        draw.source_id,
                        &draw.vertex_rgba,
                    );
                }
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
            // `boundary_fields_in_use` is populated by non-rendering
            // consumers (Purifibre today; future ops that hold a
            // BoundaryField input). Without this union the retain()
            // below evicts consumers' upstream fields and leaves the
            // consumer forever stale — the exact bug surfaced by
            // Purifibre + StreamlineDirectionField wired without a
            // BundleSurface or BoundaryGlyph renderer.
            .chain(
                self.workflow
                    .runtime
                    .scene_plan
                    .boundary_fields_in_use
                    .iter()
                    .copied(),
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
                        self.viewport.boundary_field_revision() != cache.fingerprint;
                    let field = cache.field.clone();
                    self.workflow.uploaded_odx_glyph_resource_key = None;
                    glyph_resources.set_field(
                        &rs.device,
                        field.clone(),
                        draw.scale,
                        draw.min_contacts,
                    );
                    self.viewport
                        .set_boundary_field(field.clone(), cache.fingerprint);
                    if boundary_field_changed && self.scene.nifti_files.is_empty() {
                        self.reset_slice_view_to_boundary_field(field.as_ref());
                    }
                } else if self.scene.odx_scene.is_none() {
                    glyph_resources.clear();
                    self.workflow.uploaded_odx_glyph_resource_key = None;
                    self.viewport.clear_boundary_field();
                } else {
                    self.viewport.clear_boundary_field();
                }
            } else if self.scene.odx_scene.is_none() {
                glyph_resources.clear();
                self.workflow.uploaded_odx_glyph_resource_key = None;
                self.viewport.clear_boundary_field();
            } else {
                self.viewport.clear_boundary_field();
            }
        }
        self.ensure_active_odx_glyph_resources(
            &mut renderer.callback_resources,
            &rs.device,
            &rs.queue,
        );
        self.update_active_odx_slice_state(&mut renderer.callback_resources, &rs.device, &rs.queue);
        if renderer
            .callback_resources
            .get::<OdxFixelResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(OdxFixelResources::new(&rs.device, rs.target_format));
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

            // Voxel-mask iso-surface meshes reuse the same bundle-mesh pipeline.
            for draw in &self.workflow.runtime.scene_plan.voxel_mask_mesh_draws {
                if let Some(cache) = self
                    .workflow
                    .execution_cache
                    .voxel_mask_mesh_cache
                    .get(&draw.node_uuid)
                    .filter(|cache| cache.fingerprint == draw.fingerprint)
                {
                    let one = [(cache.mesh.clone(), draw.label.clone())];
                    mesh_resources.set_bundle_meshes(draw.draw_id, &rs.device, &one);
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
        // Keep 3D and 2D fixel resources separate so each view can map a
        // different scalar field without fighting over a shared instance buffer.
        {
            use trxviz_core::data::odx_data::FixelScalarValues;

            if let Some(fr) = renderer.callback_resources.get_mut::<OdxFixelResources>() {
                if let Some(plan) = active_fixel_draw_3d(self) {
                    let fp = fixel_upload_fingerprint(plan);
                    if fp != self.workflow.uploaded_fixel_3d_fingerprint {
                        let scalars_vec: Option<Vec<f32>> = match &plan.field.scalars.values {
                            FixelScalarValues::Scalar(v) if plan.field.colormap_code != 0 => {
                                Some((**v).clone())
                            }
                            _ => None,
                        };
                        let instances = plan
                            .field
                            .scene
                            .all_fixels_with_scalars(scalars_vec.as_deref());
                        fr.resources_3d.set_fixels(&rs.device, &instances);
                        self.workflow.uploaded_fixel_3d_fingerprint = fp;
                    }
                } else if let Some(odx) = self.scene.odx_scene.as_ref() {
                    let mut hasher = DefaultHasher::new();
                    (Arc::as_ptr(odx) as usize).hash(&mut hasher);
                    0x0df1_0003_u64.hash(&mut hasher);
                    let fp = hasher.finish();
                    if fp != self.workflow.uploaded_fixel_3d_fingerprint {
                        fr.resources_3d.set_fixels(&rs.device, &odx.all_fixels());
                        self.workflow.uploaded_fixel_3d_fingerprint = fp;
                    }
                } else if self.workflow.uploaded_fixel_3d_fingerprint != 0 {
                    fr.resources_3d.clear();
                    self.workflow.uploaded_fixel_3d_fingerprint = 0;
                }

                if let Some(plan) = active_fixel_draw_2d(self) {
                    let fp = fixel_upload_fingerprint(plan);
                    if fp != self.workflow.uploaded_fixel_2d_fingerprint {
                        let scalars_vec: Option<Vec<f32>> = match &plan.field.scalars.values {
                            FixelScalarValues::Scalar(v) if plan.field.colormap_code != 0 => {
                                Some((**v).clone())
                            }
                            _ => None,
                        };
                        let instances = plan
                            .field
                            .scene
                            .all_fixels_with_scalars(scalars_vec.as_deref());
                        fr.resources_2d.set_fixels(&rs.device, &instances);
                        self.workflow.uploaded_fixel_2d_fingerprint = fp;
                    }
                } else if let Some(odx) = self.scene.odx_scene.as_ref() {
                    let mut hasher = DefaultHasher::new();
                    (Arc::as_ptr(odx) as usize).hash(&mut hasher);
                    0x0df1_0002_u64.hash(&mut hasher);
                    let fp = hasher.finish();
                    if fp != self.workflow.uploaded_fixel_2d_fingerprint {
                        fr.resources_2d.set_fixels(&rs.device, &odx.all_fixels());
                        self.workflow.uploaded_fixel_2d_fingerprint = fp;
                    }
                } else if self.workflow.uploaded_fixel_2d_fingerprint != 0 {
                    fr.resources_2d.clear();
                    self.workflow.uploaded_fixel_2d_fingerprint = 0;
                }
            }
        }

        // Upload materialized ODX DPV volumes into AllSliceResources so that
        // OdxVolumeSelect → VolumeDisplay actually renders.
        {
            use trxviz_core::renderer::slice_renderer::{SliceAxis, SliceResources};
            let plan_source_ids: HashSet<FileId> = self
                .workflow
                .runtime
                .scene_plan
                .volume_draws
                .iter()
                .map(|d| d.source_id)
                .collect();
            // Find a materialization per source_id referenced by the plan.
            let mut pending: HashMap<FileId, (WorkflowNodeUuid, String, Arc<NiftiVolume>)> =
                HashMap::new();
            for (node_uuid, m) in &self.workflow.execution_cache.odx_dpv_materializations {
                if plan_source_ids.contains(&m.source_id) {
                    pending.insert(
                        m.source_id,
                        (*node_uuid, m.dpv_name.clone(), m.volume.clone()),
                    );
                }
            }
            for (source_id, (node_uuid, dpv_name, volume)) in pending {
                let already = self
                    .workflow
                    .uploaded_dpv_by_source
                    .get(&source_id)
                    .map(|(n, d)| *n == node_uuid && d == &dpv_name)
                    .unwrap_or(false);
                let exists = renderer
                    .callback_resources
                    .get::<AllSliceResources>()
                    .map(|all| all.entries.iter().any(|(id, _)| *id == source_id))
                    .unwrap_or(false);
                if already && exists {
                    continue;
                }
                let vol_ref: &NiftiVolume = &volume;
                let slice_resources =
                    SliceResources::new(&rs.device, &rs.queue, rs.target_format, vol_ref);
                slice_resources.update_slice(
                    &rs.queue,
                    SliceAxis::Axial,
                    self.viewport.slice_index(0),
                    vol_ref,
                );
                slice_resources.update_slice(
                    &rs.queue,
                    SliceAxis::Coronal,
                    self.viewport.slice_index(1),
                    vol_ref,
                );
                slice_resources.update_slice(
                    &rs.queue,
                    SliceAxis::Sagittal,
                    self.viewport.slice_index(2),
                    vol_ref,
                );
                if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                    if let Some(entry) = all.entries.iter_mut().find(|(id, _)| *id == source_id) {
                        *entry = (source_id, slice_resources);
                    } else {
                        all.entries.push((source_id, slice_resources));
                    }
                } else {
                    renderer.callback_resources.insert(AllSliceResources {
                        entries: vec![(source_id, slice_resources)],
                    });
                }
                self.workflow
                    .uploaded_dpv_by_source
                    .insert(source_id, (node_uuid, dpv_name));
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
            if let Some(fixel_resources) =
                renderer.callback_resources.get_mut::<OdxFixelResources>()
            {
                fixel_resources.clear();
            }
        }

        self.scene.trx_files.clear();
        self.scene.nifti_files.clear();
        self.scene.gifti_surfaces.clear();
        self.scene.parcellations.clear();
        self.pending_file_loads.clear();
        self.viewport.clear_boundary_field();
        *self.viewport.render_3d_mut() = Default::default();
        self.workflow.runtime = WorkflowRuntime::default();
        self.workflow.execution_cache = WorkflowExecutionCache::default();
        self.workflow.display_runtimes.clear();
        self.workflow.selection = None;
        self.workflow.document.selection = None;
        self.workflow.node_feedback.clear();
        self.workflow.document = default_document();
        self.rebuild_workflow_editor_from_document();
        self.scene.next_file_id = 0;
        self.workflow.next_draw_id = 1_000_000;
        self.workflow.run_expensive_requested = false;
        self.workflow.run_session_active = false;
        self.workflow.pending_stage_camera_fit = false;
        self.workflow.render_only_changed = false;
        self.workflow.pending_job_completion = false;
        self.workflow.jobs_in_flight.clear();
        self.workflow.document_revision += 1;
        self.workflow.last_interactive_revision = 0;
        self.workflow.last_settled_revision = 0;
        self.workflow.last_runtime_revision += 1;
        self.workflow.last_resource_sync_revision = 0;
        self.workflow.uploaded_dpv_by_source.clear();
        self.workflow.uploaded_odx_glyph_resource_key = None;
        self.workflow.uploaded_fixel_3d_fingerprint = 0;
        self.workflow.uploaded_fixel_2d_fingerprint = 0;
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
        self.workflow.document.camera_3d = Some(self.capture_document_camera_3d());
        self.workflow.document.render_3d = Some(self.capture_document_render_3d());
        self.workflow.document.slice_view_3d = Some(self.capture_document_slice_view_3d());
        self.workflow.document.slice_view_ui = Some(self.capture_document_slice_view_ui());
        self.workflow.document.selection = self.workflow.selection;
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

    pub(in crate::app) fn export_to_blender(&mut self, view: HeadlessView) {
        match view {
            HeadlessView::View3D => {
                self.workflow.document.camera_3d = Some(self.capture_document_camera_3d());
                self.workflow.document.render_3d = Some(self.capture_document_render_3d());
                self.workflow.document.slice_view_3d = Some(self.capture_document_slice_view_3d());
                self.workflow.document.slice_view_ui = Some(self.capture_document_slice_view_ui());
            }
            HeadlessView::InflatedStage => {}
            HeadlessView::View2D => {
                self.error_msg = Some("2D views cannot be exported to Blender.".to_string());
                return;
            }
        }
        let default_name = self
            .workflow
            .project_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .map(|stem| format!("{stem}.glb"))
            .unwrap_or_else(|| "trxviz_scene.glb".to_string());
        let Some(output_path) = rfd::FileDialog::new()
            .add_filter("GLB files", &["glb"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };

        let (width, height, include_slices, target, azimuth_deg, elevation_deg, distance) =
            match view {
                HeadlessView::View3D => (
                    self.viewport.window_3d_size()[0].max(1.0).round() as u32,
                    self.viewport.window_3d_size()[1].max(1.0).round() as u32,
                    true,
                    None,
                    None,
                    None,
                    None,
                ),
                HeadlessView::InflatedStage => (
                    self.viewport.inflated_stage_size()[0].max(1.0).round() as u32,
                    self.viewport.inflated_stage_size()[1].max(1.0).round() as u32,
                    false,
                    Some(self.viewport.inflated_stage_camera().center),
                    Some(self.viewport.inflated_stage_camera().yaw.to_degrees()),
                    Some(self.viewport.inflated_stage_camera().pitch.to_degrees()),
                    Some(self.viewport.inflated_stage_camera().distance),
                ),
                HeadlessView::View2D => unreachable!(),
            };
        let options = HeadlessSceneExportOptions {
            format: HeadlessSceneExportFormat::Glb,
            include_camera: true,
            include_lights: true,
            include_slices,
            width,
            height,
            view,
            target,
            azimuth_deg,
            elevation_deg,
            distance,
        };

        let workflow = HeadlessWorkflowState {
            document: self.workflow.document.clone(),
            runtime: self.workflow.runtime.clone(),
            display_runtimes: self.workflow.display_runtimes.clone(),
            next_draw_id: self.workflow.next_draw_id,
            execution_cache: self.workflow.execution_cache.clone(),
            project_path: self.workflow.project_path.clone(),
        };

        match export_state_glb(&self.scene, workflow, &output_path, &options) {
            Ok(()) => {
                self.status_msg = Some(format!(
                    "Exported Blender scene to {}",
                    output_path.display()
                ));
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to export Blender scene: {err}"));
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
                WorkflowAssetDocument::Cifti {
                    id,
                    path: asset_path,
                    ..
                } => trxviz_core::data::cifti::LoadedCifti::load(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|data| {
                        self.apply_loaded_cifti_with_options(
                            asset_path.clone(),
                            trxviz_core::data::loaded_files::LoadedCifti {
                                id,
                                name: asset_path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| "cifti.nii".to_string()),
                                path: asset_path,
                                data: Arc::new(data),
                                visible: true,
                            },
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
                WorkflowAssetDocument::Odx {
                    id,
                    path: asset_path,
                } => trxviz_core::data::odx_data::OdxScene::open(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|odx_scene| {
                        let warnings = odx_scene.glyph_warnings().to_vec();
                        let name = asset_path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "odx".to_string());
                        self.scene
                            .odx_files
                            .push(trxviz_core::data::loaded_files::LoadedOdx {
                                id,
                                name,
                                path: asset_path,
                                scene: Arc::new(odx_scene),
                                warnings,
                                visible: true,
                            });
                    }),
            };

            if let Err(err) = load_result {
                self.error_msg = Some(format!("Failed to load workflow project asset: {err}"));
                return;
            }
        }

        self.workflow.document = project.document;
        self.apply_document_camera_3d_to_viewport();
        self.apply_document_render_3d_to_viewport();
        self.apply_document_slice_view_3d_to_viewport();
        self.apply_document_slice_view_ui_to_viewport();
        self.workflow.workspace = workspace;
        self.workflow.selection = self.workflow.document.selection;
        ensure_node_uuids(&mut self.workflow.document);
        self.rebuild_workflow_editor_from_document();
        self.workflow.pending_stage_camera_fit = true;
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
    // Already succeeded at this fingerprint — don't re-run.
    if record.last_success_fingerprint == Some(fingerprint) {
        return false;
    }
    // User cancelled at this fingerprint — don't auto-restart. Cleared
    // on any explicit Run click (see `queue_workflow_jobs` entry) or
    // when the fingerprint changes.
    if record.last_cancelled_fingerprint == Some(fingerprint) {
        return false;
    }
    true
}

fn mark_expensive_failure(record: &mut ExpensiveNodeRunRecord, fingerprint: u64, error: &str) {
    record.current_fingerprint = Some(fingerprint);
    record.status = WorkflowExecutionStatus::Failed(error.to_string());
}

#[cfg(test)]
mod tests {
    use trxviz_core::data::odx_data::OdxGlyphSourceKind;
    use trxviz_core::renderer::glyph_renderer::{OdxGlyphResourceKey, OdxGpuGlyphMode};

    #[test]
    fn odx_glyph_resource_key_changes_when_conditioning_flags_change() {
        let key_a = OdxGlyphResourceKey {
            scene_ptr: 1,
            source_kind: OdxGlyphSourceKind::Odf,
            mode: OdxGpuGlyphMode::OdfSliceGather,
            sphere_vertex_count: 642,
            sphere_index_count: 3840,
            sh_order: None,
            sh_detail: None,
            slice_axis: Some(2),
            slice_index: Some(0),
            subtract_iso: true,
            norm_within_voxel: false,
            opacity_gate_fingerprint: 0,
            size_gate_fingerprint: 0,
        };
        let key_b = OdxGlyphResourceKey {
            subtract_iso: false,
            ..key_a
        };
        let key_c = OdxGlyphResourceKey {
            norm_within_voxel: true,
            ..key_a
        };

        assert_ne!(key_a, key_b);
        assert_ne!(key_a, key_c);
    }
}

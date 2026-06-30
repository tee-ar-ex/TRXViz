use anyhow::{anyhow, bail};

use crate::data::trx_data::RenderStyle;
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::workflow::{
    BundleSurfacePlan, CachedBoundaryField, CachedBundleSurfaceMeshes, CachedDerivedStreamline,
    CachedSurfaceQuery, CachedSurfaceStreamlineMap, CachedTubeGeometry, CancelFlag,
    WorkflowJobOutput, WorkflowJobPayload, WorkflowNodeUuid, ensure_node_uuids,
    evaluate_scene_plan, mark_expensive_success, run_workflow_job, save_streamline_plan,
    workflow_boundary_plan_fingerprint, workflow_bundle_display_fingerprint,
    workflow_bundle_plan_fingerprint, workflow_reactive_streamline_fingerprint,
    workflow_streamline_fingerprint, workflow_surface_projection_fingerprint,
    workflow_surface_query_fingerprint,
};

pub(super) fn execute_workflow_to_completion(
    scene: &HeadlessScene,
    workflow: &mut HeadlessWorkflowState,
) -> anyhow::Result<()> {
    loop {
        refresh_workflow_runtime(scene, workflow);
        if let Some(error) = &workflow.runtime.graph_error {
            bail!("{error}");
        }

        let mut ran_job = false;

        for plan in workflow
            .runtime
            .scene_plan
            .reactive_streamline_plans
            .clone()
        {
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(
                    WorkflowJobPayload::ReactiveStreamline(plan),
                    CancelFlag::new(),
                )
                .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.surface_query_plans.clone() {
            let fingerprint =
                workflow_surface_query_fingerprint(&plan.flow, plan.surface_id, plan.depth_mm);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::SurfaceQuery(plan), CancelFlag::new())
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.surface_map_plans.clone() {
            let fingerprint = workflow_surface_projection_fingerprint(
                &plan.flow,
                plan.surface_id,
                plan.depth_mm,
                plan.dps_field.as_ref().map(|field| field.as_str()),
            );
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::SurfaceMap(plan), CancelFlag::new())
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for draw in workflow.runtime.scene_plan.streamline_draws.clone() {
            if draw.render_style != RenderStyle::Tubes {
                continue;
            }
            let fingerprint = workflow_streamline_fingerprint(&draw);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(draw.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                draw.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::TubeGeometry(draw), CancelFlag::new())
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.boundary_field_plans.clone() {
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.build_node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.build_node_uuid,
                fingerprint,
                run_workflow_job(
                    WorkflowJobPayload::BoundaryField { plan },
                    CancelFlag::new(),
                )
                .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.bundle_surface_plans.clone() {
            let fingerprint = workflow_bundle_plan_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.build_node_uuid)
                .or_default();
            if record.last_success_fingerprint != Some(fingerprint) {
                mark_expensive_success(
                    record,
                    fingerprint,
                    format!(
                        "Bundle surface build for {} streamline(s)",
                        plan.flow.selected_streamlines.len()
                    ),
                );
            }
        }

        for draw in workflow
            .runtime
            .scene_plan
            .draws
            .of_type::<crate::workflow::BundleDrawPlan>()
            .cloned()
            .collect::<Vec<_>>()
        {
            let boundary_field = draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
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
                    workflow
                        .execution_cache
                        .boundary_field_cache
                        .get(&uuid)
                        .map(|cache| cache.fingerprint)
                }),
            );
            let record = workflow
                .execution_cache
                .node_runs
                .entry(draw.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
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
            apply_job_result(
                &mut workflow.execution_cache,
                draw.node_uuid,
                fingerprint,
                run_workflow_job(
                    WorkflowJobPayload::BundleSurface {
                        plan,
                        color_mode: draw.color_mode,
                        boundary_field,
                    },
                    CancelFlag::new(),
                )
                .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        if !ran_job {
            break;
        }
    }

    refresh_workflow_runtime(scene, workflow);
    if let Some(error) = &workflow.runtime.graph_error {
        bail!("{error}");
    }
    for plan in workflow.runtime.save_streamline_targets.values() {
        save_streamline_plan(plan).map_err(|err| anyhow!(err))?;
    }
    Ok(())
}

pub(super) fn ensure_export_tube_geometry(
    workflow: &mut HeadlessWorkflowState,
) -> anyhow::Result<()> {
    for draw in workflow.runtime.scene_plan.streamline_draws.clone() {
        if !draw.visible {
            continue;
        }
        let fingerprint = workflow_streamline_fingerprint(&draw);
        let record = workflow
            .execution_cache
            .node_runs
            .entry(draw.node_uuid)
            .or_default();
        if record.last_success_fingerprint == Some(fingerprint)
            && workflow
                .execution_cache
                .tube_geometry_cache
                .get(&draw.node_uuid)
                .is_some_and(|cache| cache.fingerprint == fingerprint)
        {
            continue;
        }
        apply_job_result(
            &mut workflow.execution_cache,
            draw.node_uuid,
            fingerprint,
            run_workflow_job(WorkflowJobPayload::TubeGeometry(draw), CancelFlag::new())
                .map_err(|err| anyhow!(err))?,
        );
    }
    Ok(())
}

fn refresh_workflow_runtime(scene: &HeadlessScene, workflow: &mut HeadlessWorkflowState) {
    ensure_node_uuids(&mut workflow.document);
    workflow.runtime = evaluate_scene_plan(
        &workflow.document,
        &scene.trx_files,
        &scene.nifti_files,
        &scene.cifti_files,
        &scene.gifti_surfaces,
        &scene.parcellations,
        &scene.odx_files,
        &mut workflow.display_runtimes,
        &mut workflow.next_draw_id,
        &mut workflow.execution_cache,
    );
}

fn apply_job_result(
    cache: &mut crate::workflow::WorkflowExecutionCache,
    node_uuid: WorkflowNodeUuid,
    fingerprint: u64,
    output: WorkflowJobOutput,
) {
    let record = cache.node_runs.entry(node_uuid).or_default();
    match output {
        WorkflowJobOutput::ReactiveStreamline(flow) => {
            cache
                .derived_streamline_cache
                .insert(node_uuid, CachedDerivedStreamline { flow });
            mark_expensive_success(record, fingerprint, "reactive streamlines".to_string());
        }
        WorkflowJobOutput::SurfaceQuery(flow) => {
            cache
                .surface_query_cache
                .insert(node_uuid, CachedSurfaceQuery { flow });
            mark_expensive_success(record, fingerprint, "surface query".to_string());
        }
        WorkflowJobOutput::SurfaceMap(map) => {
            cache
                .surface_streamline_map_cache
                .insert(node_uuid, CachedSurfaceStreamlineMap { map });
            mark_expensive_success(record, fingerprint, "surface map".to_string());
        }
        WorkflowJobOutput::TubeGeometry { vertices, indices } => {
            cache.tube_geometry_cache.insert(
                node_uuid,
                CachedTubeGeometry {
                    fingerprint,
                    vertices,
                    indices,
                },
            );
            mark_expensive_success(record, fingerprint, "tube geometry".to_string());
        }
        WorkflowJobOutput::BundleSurface { meshes } => {
            let summary = if meshes.is_empty() {
                "Bundle surface is empty".to_string()
            } else {
                format!("{} bundle surface mesh(es)", meshes.len())
            };
            cache.bundle_surface_mesh_cache.insert(
                node_uuid,
                CachedBundleSurfaceMeshes {
                    fingerprint,
                    meshes,
                },
            );
            mark_expensive_success(record, fingerprint, summary);
        }
        WorkflowJobOutput::BoundaryField { field } => {
            let summary = field
                .as_ref()
                .map(|_| "Boundary field".to_string())
                .unwrap_or_else(|| "No boundary field".to_string());
            if let Some(field) = field {
                cache
                    .boundary_field_cache
                    .insert(node_uuid, CachedBoundaryField { fingerprint, field });
            } else {
                cache.boundary_field_cache.remove(&node_uuid);
            }
            mark_expensive_success(record, fingerprint, summary);
        }
        WorkflowJobOutput::DipyTractography { flow } => {
            cache.dipy_tractography_results.insert(
                node_uuid,
                crate::workflow::CachedTractographyResult { fingerprint, flow },
            );
            mark_expensive_success(record, fingerprint, "dipy tractography".to_string());
        }
        WorkflowJobOutput::YehTractography { flow } => {
            cache.yeh_tractography_results.insert(
                node_uuid,
                crate::workflow::CachedTractographyResult { fingerprint, flow },
            );
            mark_expensive_success(record, fingerprint, "yeh tractography".to_string());
        }
        WorkflowJobOutput::PrepareHausdorffPlan {
            plan,
            seed_mask,
            limiting_mask,
            no_end_mask,
            summary,
        } => {
            cache.hausdorff_plan_cache.insert(
                node_uuid,
                crate::workflow::CachedHausdorffPlan {
                    fingerprint,
                    plan,
                    seed_mask,
                    limiting_mask,
                    no_end_mask,
                    summary: summary.clone(),
                },
            );
            mark_expensive_success(record, fingerprint, summary);
        }
        WorkflowJobOutput::PreparePyafqPlan {
            plan,
            include_mask,
            exclude_mask,
            start_mask,
            end_mask,
            prob_map,
            summary,
        } => {
            cache.pyafq_plan_cache.insert(
                node_uuid,
                crate::workflow::CachedPyafqPlan {
                    fingerprint,
                    plan,
                    include_mask,
                    exclude_mask,
                    start_mask,
                    end_mask,
                    prob_map,
                    summary: summary.clone(),
                },
            );
            mark_expensive_success(record, fingerprint, summary);
        }
    }
}

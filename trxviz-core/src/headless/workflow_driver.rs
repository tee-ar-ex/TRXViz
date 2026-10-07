use anyhow::{anyhow, bail};

use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::workflow::{
    CancelFlag, SchedulePolicy, WorkflowEvalMode, apply_job_output, deferred_jobs,
    ensure_node_uuids, evaluate_scene_plan_with_mode, plan_jobs, run_workflow_job,
    save_streamline_plan, workflow_job_kind_title,
};

/// Evaluate the workflow and run every background job it asks for, inline,
/// until a settled evaluation asks for nothing more.
///
/// Each pass runs every job the scheduler considers dispatchable, then
/// re-evaluates so downstream requests see the results. Jobs whose inputs
/// are still stale are deferred to a later pass (plan-prep before
/// trackers, trackers before tube geometry). The pass cap turns a
/// fingerprint that churns on every evaluation into an error instead of
/// an infinite loop.
pub(super) fn execute_workflow_to_completion(
    scene: &HeadlessScene,
    workflow: &mut HeadlessWorkflowState,
    gpu: Option<&crate::gpu::GpuHandle>,
    run_expensive: bool,
) -> anyhow::Result<()> {
    let policy = SchedulePolicy {
        run_expensive,
        ..SchedulePolicy::run_all()
    };
    let max_passes = 8 + 4 * workflow.document.graph.len();
    let mut converged = false;
    for _ in 0..max_passes {
        refresh_workflow_runtime(scene, workflow);
        if let Some(error) = &workflow.runtime.graph_error {
            bail!("{error}");
        }
        let jobs = plan_jobs(&workflow.runtime, &workflow.execution_cache, &policy);
        if jobs.is_empty() {
            let stuck = deferred_jobs(&workflow.runtime, &workflow.execution_cache, &policy);
            if !stuck.is_empty() {
                let names: Vec<String> = stuck
                    .iter()
                    .map(|receipt| {
                        format!(
                            "{} (node {})",
                            workflow_job_kind_title(receipt.kind),
                            receipt.node_uuid.0
                        )
                    })
                    .collect();
                bail!(
                    "workflow cannot make progress: {} waiting on upstream inputs that never \
                     settled",
                    names.join(", ")
                );
            }
            converged = true;
            break;
        }
        for job in jobs {
            let output = run_workflow_job(job.payload, CancelFlag::new(), gpu).map_err(|err| {
                anyhow!(
                    "{} job for node {} failed: {err}",
                    workflow_job_kind_title(job.receipt.kind),
                    job.receipt.node_uuid.0
                )
            })?;
            apply_job_output(&mut workflow.execution_cache, &job.receipt, output);
        }
    }
    if !converged {
        bail!(
            "workflow did not converge after {max_passes} evaluation passes; \
             a node's fingerprint may be changing on every evaluation"
        );
    }

    for plan in workflow.runtime.save_streamline_targets.values() {
        save_streamline_plan(plan).map_err(|err| anyhow!(err))?;
    }
    Ok(())
}

fn refresh_workflow_runtime(scene: &HeadlessScene, workflow: &mut HeadlessWorkflowState) {
    ensure_node_uuids(&mut workflow.document);
    workflow.runtime = evaluate_scene_plan_with_mode(
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
        WorkflowEvalMode::Settled,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{WorkflowExecutionStatus, evaluate_tests::job_pipeline_fixture};

    #[test]
    fn headless_driver_runs_every_job_and_converges() {
        let (document, trx, surface, nodes) = job_pipeline_fixture();
        let scene = HeadlessScene {
            trx_files: vec![trx],
            gifti_surfaces: vec![surface],
            ..HeadlessScene::default()
        };
        let mut workflow = HeadlessWorkflowState {
            document,
            ..HeadlessWorkflowState::default()
        };

        execute_workflow_to_completion(&scene, &mut workflow, None, true).expect("converges");

        let cache = &workflow.execution_cache;
        assert!(cache.derived_streamline_cache.contains_key(&nodes.merge));
        assert!(cache.tube_geometry_cache.contains_key(&nodes.tubes));
        assert!(cache.surface_query_cache.contains_key(&nodes.query));
        assert!(
            cache
                .surface_streamline_map_cache
                .contains_key(&nodes.projection)
        );
        assert!(
            cache
                .bundle_surface_mesh_cache
                .contains_key(&nodes.bundle_display)
        );
        for uuid in [
            nodes.merge,
            nodes.tubes,
            nodes.query,
            nodes.projection,
            nodes.direction_field,
            nodes.bundle_build,
            nodes.bundle_display,
        ] {
            assert_eq!(
                cache.node_runs[&uuid].status,
                WorkflowExecutionStatus::Ready,
                "node {uuid:?}"
            );
        }
        // The final runtime reflects the settled state: nothing left to run.
        assert!(plan_jobs(&workflow.runtime, cache, &SchedulePolicy::run_all()).is_empty());
    }

    #[test]
    fn headless_driver_skips_expensive_when_asked() {
        let (document, trx, surface, nodes) = job_pipeline_fixture();
        let scene = HeadlessScene {
            trx_files: vec![trx],
            gifti_surfaces: vec![surface],
            ..HeadlessScene::default()
        };
        let mut workflow = HeadlessWorkflowState {
            document,
            ..HeadlessWorkflowState::default()
        };
        execute_workflow_to_completion(&scene, &mut workflow, None, false).expect("converges");
        let cache = &workflow.execution_cache;
        assert!(cache.derived_streamline_cache.contains_key(&nodes.merge));
        assert!(!cache.tube_geometry_cache.contains_key(&nodes.tubes));
        assert!(!cache.surface_query_cache.contains_key(&nodes.query));
    }
}

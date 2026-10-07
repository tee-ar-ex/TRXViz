//! Shared background-job scheduler.
//!
//! Both drivers (the egui app and the headless CLI) used to walk every
//! `SceneFramePlan` job field with their own copy of the "does this node
//! need a job, and is the cached run still current?" logic, their own
//! fingerprint recomputation, their own result-application `match`, and
//! their own status strings. They drifted: the headless loop dispatched
//! six of the ten job kinds, evaluated in `Interactive` mode, and marked
//! bundle builds Ready without running anything.
//!
//! This module is the single answer to three questions:
//!
//! - [`plan_jobs`]: given an evaluated runtime and the execution cache,
//!   which jobs should be dispatched right now?
//! - [`apply_job_output`] / [`apply_job_error`]: given a finished job,
//!   how do its results land in the cache and the node's run record?
//! - [`should_dispatch`]: the one gating rule (success / cancel / failure
//!   / in-flight / expensive-tier) both drivers obey.
//!
//! What stays driver-specific is *how* jobs run: the GUI spawns threads
//! and pumps an mpsc channel; the headless driver runs each job inline
//! and loops until [`plan_jobs`] is empty.

use std::collections::HashMap;

use crate::error::WorkflowError;

use super::{
    CachedBoundaryField, CachedBundleSurfaceMeshes, CachedDerivedStreamline, CachedHausdorffPlan,
    CachedPyafqPlan, CachedSurfaceQuery, CachedSurfaceStreamlineMap, CachedTractographyResult,
    CachedTubeGeometry, ExpensiveNodeRunRecord, WorkflowEvalMode, WorkflowExecutionCache,
    WorkflowJobKind, WorkflowJobOutput, WorkflowJobPayload, WorkflowNodeUuid, WorkflowRuntime,
    mark_expensive_cancelled, mark_expensive_failure, mark_expensive_success,
};

/// Identity of one dispatched job. Created by [`plan_jobs`], carried by
/// the driver alongside the running job, and handed back to
/// [`apply_job_output`] so the result lands on the right record at the
/// right fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobReceipt {
    /// The run-record key. For most kinds this is the producing node.
    /// For `BoundaryField` it is the *build* node; for `BundleSurface`
    /// it is the *display* node (see `build_node`).
    pub node_uuid: WorkflowNodeUuid,
    pub kind: WorkflowJobKind,
    pub fingerprint: u64,
    /// `BundleSurface` only: the sibling `BundleSurfaceBuild` node and
    /// its plan fingerprint. Marked Ready together with the display
    /// node when the job finishes at a still-current fingerprint.
    pub build_node: Option<(WorkflowNodeUuid, u64)>,
}

/// Whether a job runs automatically on every settled pass or only when
/// the driver has opted into expensive work (the GUI's Run button).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobTier {
    Auto,
    Expensive,
}

impl JobTier {
    pub fn of(kind: WorkflowJobKind) -> Self {
        match kind {
            WorkflowJobKind::ReactiveStreamline => JobTier::Auto,
            _ => JobTier::Expensive,
        }
    }
}

/// A job the scheduler has decided should run now.
pub struct ScheduledJob {
    pub receipt: JobReceipt,
    pub tier: JobTier,
    pub payload: WorkflowJobPayload,
}

/// A background job an op asks for during evaluation.
///
/// Ops push one of these into `SceneFramePlan::jobs` at the point they
/// prime their run record, carrying the same fingerprint, so the
/// scheduler never recomputes fingerprints and every driver agrees with
/// the op about what "current" means. `upstream_stale` lets the
/// scheduler hold a job back until the jobs feeding it have landed.
#[derive(Clone)]
pub struct JobRequest {
    pub receipt: JobReceipt,
    /// `true` when any input to the requesting op was itself stale
    /// (its producer's job has not run at the current fingerprint).
    pub upstream_stale: bool,
    pub payload: WorkflowJobPayload,
}

impl JobRequest {
    pub fn new(
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        upstream_stale: bool,
        payload: WorkflowJobPayload,
    ) -> Self {
        Self {
            receipt: JobReceipt {
                node_uuid,
                kind: payload.kind(),
                fingerprint,
                build_node: None,
            },
            upstream_stale,
            payload,
        }
    }

    /// `BundleSurface` only: record the sibling build node to mark
    /// Ready alongside the display node.
    pub fn with_build_node(mut self, build_node: WorkflowNodeUuid, build_fingerprint: u64) -> Self {
        self.receipt.build_node = Some((build_node, build_fingerprint));
        self
    }
}

/// Jobs currently dispatched but not yet finished, keyed by run-record
/// node. Any entry blocks further dispatch for that node regardless of
/// fingerprint: one job per node at a time.
pub type InFlightJobs = HashMap<WorkflowNodeUuid, (WorkflowJobKind, u64)>;

/// Driver-supplied inputs to the gating rule.
pub struct SchedulePolicy<'a> {
    /// Dispatch `Expensive`-tier jobs. Headless: always. GUI: only when
    /// the user clicked Run or a run session is still active.
    pub run_expensive: bool,
    pub in_flight: &'a InFlightJobs,
    /// Ignore a prior cancellation or failure at the current
    /// fingerprint. The GUI sets this on an explicit Run click so the
    /// user can retry without editing a parameter.
    pub retry_blocked: bool,
    /// Skip jobs whose inputs are known to be stale, so upstream jobs
    /// land before downstream ones are attempted (instead of, say,
    /// tracking a whole brain against a placeholder seed mask and then
    /// discarding the result when the real plan arrives).
    pub defer_when_upstream_stale: bool,
}

impl SchedulePolicy<'static> {
    /// "Run everything, nothing in flight": the headless policy.
    pub fn run_all() -> Self {
        static EMPTY: std::sync::OnceLock<InFlightJobs> = std::sync::OnceLock::new();
        Self {
            run_expensive: true,
            in_flight: EMPTY.get_or_init(InFlightJobs::new),
            retry_blocked: false,
            defer_when_upstream_stale: true,
        }
    }
}

/// The single gating rule. `record` is the node's run record if any.
pub fn should_dispatch(
    record: Option<&ExpensiveNodeRunRecord>,
    fingerprint: u64,
    tier: JobTier,
    in_flight: bool,
    policy: &SchedulePolicy<'_>,
) -> bool {
    if in_flight {
        return false;
    }
    if tier == JobTier::Expensive && !policy.run_expensive {
        return false;
    }
    let Some(record) = record else {
        return true;
    };
    if record.last_success_fingerprint == Some(fingerprint) {
        return false;
    }
    if !policy.retry_blocked {
        if record.last_cancelled_fingerprint == Some(fingerprint) {
            return false;
        }
        if record.last_failed_fingerprint == Some(fingerprint) {
            return false;
        }
    }
    true
}

/// Decide which jobs to dispatch from an evaluated runtime.
///
/// Returns nothing for an `Interactive` runtime: only a `Settled`
/// evaluation reflects a document the user has stopped editing.
///
/// Requests come back in evaluation (topological) order, so producers
/// precede consumers. With `defer_when_upstream_stale` a consumer whose
/// inputs are stale is held back for a later pass.
pub fn plan_jobs(
    runtime: &WorkflowRuntime,
    cache: &WorkflowExecutionCache,
    policy: &SchedulePolicy<'_>,
) -> Vec<ScheduledJob> {
    select_jobs(runtime, cache, policy, false)
}

/// The requests [`plan_jobs`] is holding back *only* because their
/// inputs are stale. A driver that has run out of dispatchable jobs can
/// use this to tell "done" from "stuck behind an upstream that will
/// never settle".
pub fn deferred_jobs(
    runtime: &WorkflowRuntime,
    cache: &WorkflowExecutionCache,
    policy: &SchedulePolicy<'_>,
) -> Vec<JobReceipt> {
    select_jobs(runtime, cache, policy, true)
        .into_iter()
        .map(|job| job.receipt)
        .collect()
}

fn select_jobs(
    runtime: &WorkflowRuntime,
    cache: &WorkflowExecutionCache,
    policy: &SchedulePolicy<'_>,
    only_deferred: bool,
) -> Vec<ScheduledJob> {
    if runtime.eval_mode != WorkflowEvalMode::Settled {
        return Vec::new();
    }
    runtime
        .scene_plan
        .jobs
        .iter()
        .filter(|request| {
            let deferred = policy.defer_when_upstream_stale && request.upstream_stale;
            deferred == only_deferred
        })
        .filter_map(|request| {
            let tier = JobTier::of(request.receipt.kind);
            let node_uuid = request.receipt.node_uuid;
            should_dispatch(
                cache.node_runs.get(&node_uuid),
                request.receipt.fingerprint,
                tier,
                policy.in_flight.contains_key(&node_uuid),
                policy,
            )
            .then(|| ScheduledJob {
                receipt: request.receipt.clone(),
                tier,
                payload: request.payload.clone(),
            })
        })
        .collect()
}

/// What [`apply_job_output`] did with a finished job.
pub struct AppliedJob {
    /// Human-readable one-liner stored on the run record.
    pub summary: String,
    /// `false` when the node's fingerprint moved on while the job ran.
    /// The output is still cached (no compute wasted; downstream
    /// consumers compare the cache entry's own fingerprint), but the
    /// run record is not advanced, so the node stays Stale and the
    /// scheduler will re-dispatch at the current fingerprint.
    pub fingerprint_current: bool,
}

/// Land a finished job's output in the execution cache and, when the
/// receipt is still current, advance the node's run record.
///
/// Returns `None` when the node has no run record any more (removed
/// from the document while the job ran); nothing is cached then.
pub fn apply_job_output(
    cache: &mut WorkflowExecutionCache,
    receipt: &JobReceipt,
    output: WorkflowJobOutput,
) -> Option<AppliedJob> {
    let node_uuid = receipt.node_uuid;
    let fingerprint = receipt.fingerprint;
    let fingerprint_current = cache
        .node_runs
        .get(&node_uuid)?
        .current_fingerprint
        .is_some_and(|current| current == fingerprint);

    let summary = match output {
        WorkflowJobOutput::ReactiveStreamline(flow) => {
            let s = format!("{} streamlines", flow.selected_streamlines.len());
            cache
                .derived_streamline_cache
                .insert(node_uuid, CachedDerivedStreamline { flow });
            s
        }
        WorkflowJobOutput::SurfaceQuery(flow) => {
            let s = format!("{} streamlines", flow.selected_streamlines.len());
            cache
                .surface_query_cache
                .insert(node_uuid, CachedSurfaceQuery { flow });
            s
        }
        WorkflowJobOutput::SurfaceMap(map) => {
            let s = format!("Surface scalars ({} values)", map.values.len());
            cache
                .surface_streamline_map_cache
                .insert(node_uuid, CachedSurfaceStreamlineMap { map });
            s
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
            "Tube geometry ready".to_string()
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
            // The sibling build node has no job of its own; it is
            // "done" when its display finished at a current fingerprint.
            if fingerprint_current && let Some((build_uuid, build_fingerprint)) = receipt.build_node
            {
                let build_record = cache.node_runs.entry(build_uuid).or_default();
                mark_expensive_success(build_record, build_fingerprint, summary.clone());
            }
            summary
        }
        WorkflowJobOutput::BoundaryField { field } => {
            if let Some(field) = field {
                cache
                    .boundary_field_cache
                    .insert(node_uuid, CachedBoundaryField { fingerprint, field });
                "Boundary field ready".to_string()
            } else {
                cache.boundary_field_cache.remove(&node_uuid);
                "Boundary field is empty".to_string()
            }
        }
        WorkflowJobOutput::DipyTractography { flow } => {
            let s = format!("{} streamlines", flow.selected_streamlines.len());
            cache
                .dipy_tractography_results
                .insert(node_uuid, CachedTractographyResult { fingerprint, flow });
            s
        }
        WorkflowJobOutput::YehTractography { flow } => {
            let s = format!("{} streamlines", flow.selected_streamlines.len());
            cache
                .yeh_tractography_results
                .insert(node_uuid, CachedTractographyResult { fingerprint, flow });
            s
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
                CachedHausdorffPlan {
                    fingerprint,
                    plan,
                    seed_mask,
                    limiting_mask,
                    no_end_mask,
                    summary: summary.clone(),
                },
            );
            summary
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
                CachedPyafqPlan {
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
            summary
        }
    };

    if fingerprint_current && let Some(record) = cache.node_runs.get_mut(&node_uuid) {
        mark_expensive_success(record, fingerprint, summary.clone());
    }

    Some(AppliedJob {
        summary,
        fingerprint_current,
    })
}

/// What [`apply_job_error`] recorded.
pub struct AppliedError {
    /// The job ended because its cancel flag was raised, not because
    /// it failed. Drivers should report this neutrally.
    pub cancelled: bool,
    /// As for [`AppliedJob::fingerprint_current`]. When `false` the
    /// record is left alone: the failure was for obsolete parameters.
    pub fingerprint_current: bool,
}

/// Record a failed or cancelled job on the node's run record.
///
/// Returns `None` when the node has no run record any more.
pub fn apply_job_error(
    cache: &mut WorkflowExecutionCache,
    receipt: &JobReceipt,
    error: &WorkflowError,
) -> Option<AppliedError> {
    let record = cache.node_runs.get_mut(&receipt.node_uuid)?;
    let fingerprint_current = record
        .current_fingerprint
        .is_some_and(|current| current == receipt.fingerprint);
    let cancelled = matches!(error, WorkflowError::Cancelled);
    if fingerprint_current {
        if cancelled {
            mark_expensive_cancelled(record, receipt.fingerprint);
        } else {
            mark_expensive_failure(record, receipt.fingerprint, &error.to_string());
        }
    }
    Some(AppliedError {
        cancelled,
        fingerprint_current,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::WorkflowExecutionStatus;

    fn record(status: WorkflowExecutionStatus) -> ExpensiveNodeRunRecord {
        ExpensiveNodeRunRecord {
            status,
            ..ExpensiveNodeRunRecord::default()
        }
    }

    #[test]
    fn dispatch_rules() {
        let none = InFlightJobs::new();
        let auto_policy = SchedulePolicy {
            run_expensive: false,
            in_flight: &none,
            retry_blocked: false,
            defer_when_upstream_stale: false,
        };
        let run_policy = SchedulePolicy {
            run_expensive: true,
            ..auto_policy
        };
        let retry_policy = SchedulePolicy {
            retry_blocked: true,
            ..run_policy
        };

        // No record: dispatch unless expensive-gated.
        assert!(should_dispatch(None, 1, JobTier::Auto, false, &auto_policy));
        assert!(!should_dispatch(
            None,
            1,
            JobTier::Expensive,
            false,
            &auto_policy
        ));
        assert!(should_dispatch(
            None,
            1,
            JobTier::Expensive,
            false,
            &run_policy
        ));

        // In flight always blocks.
        assert!(!should_dispatch(None, 1, JobTier::Auto, true, &run_policy));

        // Already succeeded at this fingerprint.
        let mut r = record(WorkflowExecutionStatus::Ready);
        r.last_success_fingerprint = Some(1);
        assert!(!should_dispatch(
            Some(&r),
            1,
            JobTier::Auto,
            false,
            &run_policy
        ));
        assert!(should_dispatch(
            Some(&r),
            2,
            JobTier::Auto,
            false,
            &run_policy
        ));

        // Cancelled / failed at this fingerprint block unless retrying.
        let mut r = record(WorkflowExecutionStatus::Failed("x".into()));
        r.last_cancelled_fingerprint = Some(1);
        assert!(!should_dispatch(
            Some(&r),
            1,
            JobTier::Auto,
            false,
            &run_policy
        ));
        assert!(should_dispatch(
            Some(&r),
            1,
            JobTier::Auto,
            false,
            &retry_policy
        ));
        let mut r = record(WorkflowExecutionStatus::Failed("x".into()));
        r.last_failed_fingerprint = Some(1);
        assert!(!should_dispatch(
            Some(&r),
            1,
            JobTier::Auto,
            false,
            &run_policy
        ));
        assert!(should_dispatch(
            Some(&r),
            1,
            JobTier::Auto,
            false,
            &retry_policy
        ));
    }

    #[test]
    fn interactive_runtime_yields_nothing() {
        let runtime = WorkflowRuntime::default();
        assert_eq!(runtime.eval_mode, WorkflowEvalMode::Interactive);
        let cache = WorkflowExecutionCache::default();
        assert!(plan_jobs(&runtime, &cache, &SchedulePolicy::run_all()).is_empty());
    }

    #[test]
    fn obsolete_output_is_cached_but_record_not_advanced() {
        let mut cache = WorkflowExecutionCache::default();
        let node = WorkflowNodeUuid(7);
        let mut r = record(WorkflowExecutionStatus::NeverRun);
        r.current_fingerprint = Some(99); // moved on since dispatch at 42
        cache.node_runs.insert(node, r);
        let receipt = JobReceipt {
            node_uuid: node,
            kind: WorkflowJobKind::TubeGeometry,
            fingerprint: 42,
            build_node: None,
        };
        let applied = apply_job_output(
            &mut cache,
            &receipt,
            WorkflowJobOutput::TubeGeometry {
                vertices: Vec::new(),
                indices: Vec::new(),
            },
        )
        .expect("record exists");
        assert!(!applied.fingerprint_current);
        assert!(cache.tube_geometry_cache.contains_key(&node));
        assert_eq!(cache.node_runs[&node].last_success_fingerprint, None);

        // Current fingerprint: record advances.
        cache.node_runs.get_mut(&node).unwrap().current_fingerprint = Some(42);
        let applied = apply_job_output(
            &mut cache,
            &receipt,
            WorkflowJobOutput::TubeGeometry {
                vertices: Vec::new(),
                indices: Vec::new(),
            },
        )
        .unwrap();
        assert!(applied.fingerprint_current);
        assert_eq!(cache.node_runs[&node].last_success_fingerprint, Some(42));
        assert_eq!(
            cache.node_runs[&node].status,
            WorkflowExecutionStatus::Ready
        );
    }

    #[test]
    fn unknown_node_is_ignored() {
        let mut cache = WorkflowExecutionCache::default();
        let receipt = JobReceipt {
            node_uuid: WorkflowNodeUuid(1),
            kind: WorkflowJobKind::TubeGeometry,
            fingerprint: 1,
            build_node: None,
        };
        assert!(
            apply_job_output(
                &mut cache,
                &receipt,
                WorkflowJobOutput::TubeGeometry {
                    vertices: Vec::new(),
                    indices: Vec::new(),
                },
            )
            .is_none()
        );
        assert!(apply_job_error(&mut cache, &receipt, &WorkflowError::Cancelled).is_none());
    }

    #[test]
    fn failure_persists_through_reprime_and_blocks_redispatch() {
        let mut cache = WorkflowExecutionCache::default();
        let node = WorkflowNodeUuid(3);
        let mut r = record(WorkflowExecutionStatus::Queued);
        r.current_fingerprint = Some(5);
        cache.node_runs.insert(node, r);
        let receipt = JobReceipt {
            node_uuid: node,
            kind: WorkflowJobKind::PreparePyafqPlan,
            fingerprint: 5,
            build_node: None,
        };
        let applied = apply_job_error(
            &mut cache,
            &receipt,
            &WorkflowError::Evaluation("bad working dir".into()),
        )
        .unwrap();
        assert!(!applied.cancelled);
        assert!(applied.fingerprint_current);

        // The next evaluation re-primes at the same fingerprint: the
        // Failed status must survive and the scheduler must not re-run.
        let record = cache.node_runs.get_mut(&node).unwrap();
        crate::workflow::prime_expensive_record(record, 5);
        assert!(matches!(record.status, WorkflowExecutionStatus::Failed(_)));
        let none = InFlightJobs::new();
        let policy = SchedulePolicy {
            run_expensive: true,
            in_flight: &none,
            retry_blocked: false,
            defer_when_upstream_stale: false,
        };
        assert!(!should_dispatch(
            Some(record),
            5,
            JobTier::Expensive,
            false,
            &policy
        ));

        // A parameter edit (new fingerprint) clears the block.
        crate::workflow::prime_expensive_record(record, 6);
        assert_eq!(record.status, WorkflowExecutionStatus::NeverRun);
        assert!(should_dispatch(
            Some(record),
            6,
            JobTier::Expensive,
            false,
            &policy
        ));
    }

    /// Evaluate the pipeline fixture in Settled mode against a shared
    /// cache, so job results feed the next evaluation.
    fn settle(
        document: &super::super::WorkflowDocument,
        trx: &crate::data::loaded_files::LoadedTrx,
        surface: &crate::scene::LoadedGiftiSurface,
        cache: &mut WorkflowExecutionCache,
    ) -> WorkflowRuntime {
        super::super::evaluate_scene_plan_with_mode(
            document,
            std::slice::from_ref(trx),
            &[],
            &[],
            std::slice::from_ref(surface),
            &[],
            &[],
            &mut HashMap::new(),
            &mut 1_000_000usize,
            cache,
            WorkflowEvalMode::Settled,
        )
    }

    fn kinds(jobs: &[ScheduledJob]) -> Vec<WorkflowJobKind> {
        let mut kinds: Vec<_> = jobs.iter().map(|job| job.receipt.kind).collect();
        kinds.sort_by_key(|kind| *kind as u8);
        kinds
    }

    #[test]
    fn pipeline_dispatches_producers_first_then_consumers_then_settles() {
        use super::super::evaluate_tests::job_pipeline_fixture;
        use super::super::run_workflow_job;
        use crate::workflow::{CancelFlag, WorkflowExecutionStatus};

        let (document, trx, surface, nodes) = job_pipeline_fixture();
        let mut cache = WorkflowExecutionCache::default();
        let policy = SchedulePolicy::run_all();

        // Pass 1: everything fed directly by the source is dispatchable.
        // Merge has no cached result yet, so it emits no output at all
        // and the tube display behind it cannot even evaluate (no
        // request, an evaluation error on the node). The bundle display
        // has no boundary field wired and its build input is current,
        // so it runs alongside the producers.
        let runtime = settle(&document, &trx, &surface, &mut cache);
        assert!(runtime.graph_error.is_none(), "{:?}", runtime.graph_error);
        let jobs = plan_jobs(&runtime, &cache, &policy);
        assert_eq!(
            kinds(&jobs),
            vec![
                WorkflowJobKind::ReactiveStreamline,
                WorkflowJobKind::SurfaceQuery,
                WorkflowJobKind::SurfaceMap,
                WorkflowJobKind::BundleSurface,
                WorkflowJobKind::BoundaryField,
            ]
        );
        assert!(deferred_jobs(&runtime, &cache, &policy).is_empty());
        assert!(
            runtime.node_state[&nodes.tubes].error.is_some(),
            "tube display has no input until Merge has run"
        );

        // Receipt keys: BoundaryField is keyed by the build node, and
        // BundleSurface by the display node with the build node as
        // sibling.
        let bundle = jobs
            .iter()
            .find(|job| job.receipt.kind == WorkflowJobKind::BundleSurface)
            .unwrap();
        assert_eq!(bundle.receipt.node_uuid, nodes.bundle_display);
        assert_eq!(
            bundle.receipt.build_node.map(|(uuid, _)| uuid),
            Some(nodes.bundle_build)
        );
        let field = jobs
            .iter()
            .find(|job| job.receipt.kind == WorkflowJobKind::BoundaryField)
            .unwrap();
        assert_eq!(field.receipt.node_uuid, nodes.direction_field);

        // Run pass 1 inline and apply.
        for job in jobs {
            let output = run_workflow_job(job.payload, CancelFlag::new(), None).expect("job");
            let applied = apply_job_output(&mut cache, &job.receipt, output).expect("known node");
            assert!(applied.fingerprint_current, "{:?}", job.receipt.kind);
        }
        assert!(cache.derived_streamline_cache.contains_key(&nodes.merge));
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
        assert_eq!(
            cache.node_runs[&nodes.bundle_build].status,
            WorkflowExecutionStatus::Ready,
            "sibling build node is marked Ready by the display job"
        );

        // Pass 2: Merge has landed, so only the tube job remains.
        let runtime = settle(&document, &trx, &surface, &mut cache);
        let jobs = plan_jobs(&runtime, &cache, &policy);
        assert_eq!(kinds(&jobs), vec![WorkflowJobKind::TubeGeometry]);
        assert!(deferred_jobs(&runtime, &cache, &policy).is_empty());
        for job in jobs {
            let output = run_workflow_job(job.payload, CancelFlag::new(), None).expect("job");
            apply_job_output(&mut cache, &job.receipt, output).expect("known node");
        }
        assert!(cache.tube_geometry_cache.contains_key(&nodes.tubes));

        // Pass 3: settled.
        let runtime = settle(&document, &trx, &surface, &mut cache);
        assert!(plan_jobs(&runtime, &cache, &policy).is_empty());
        assert!(deferred_jobs(&runtime, &cache, &policy).is_empty());
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

        // Without the Run gate only the Auto tier is offered, and an
        // in-flight entry blocks its node.
        let mut fresh = WorkflowExecutionCache::default();
        let runtime = settle(&document, &trx, &surface, &mut fresh);
        let none = InFlightJobs::new();
        let auto_only = SchedulePolicy {
            run_expensive: false,
            in_flight: &none,
            retry_blocked: false,
            defer_when_upstream_stale: true,
        };
        assert_eq!(
            kinds(&plan_jobs(&runtime, &fresh, &auto_only)),
            vec![WorkflowJobKind::ReactiveStreamline]
        );
        let mut in_flight = InFlightJobs::new();
        in_flight.insert(nodes.merge, (WorkflowJobKind::ReactiveStreamline, 0));
        let blocked = SchedulePolicy {
            in_flight: &in_flight,
            ..auto_only
        };
        assert!(plan_jobs(&runtime, &fresh, &blocked).is_empty());
    }

    #[test]
    fn hand_built_tracker_outputs_land_in_their_caches() {
        let mut cache = WorkflowExecutionCache::default();
        let flow = super::super::evaluate_tests::test_streamline_flow();
        for (node, kind, output) in [
            (
                WorkflowNodeUuid(1),
                WorkflowJobKind::DipyTractography,
                WorkflowJobOutput::DipyTractography { flow: flow.clone() },
            ),
            (
                WorkflowNodeUuid(2),
                WorkflowJobKind::YehTractography,
                WorkflowJobOutput::YehTractography { flow: flow.clone() },
            ),
        ] {
            cache.node_runs.insert(
                node,
                ExpensiveNodeRunRecord {
                    current_fingerprint: Some(9),
                    ..ExpensiveNodeRunRecord::default()
                },
            );
            let receipt = JobReceipt {
                node_uuid: node,
                kind,
                fingerprint: 9,
                build_node: None,
            };
            let applied = apply_job_output(&mut cache, &receipt, output).unwrap();
            assert!(applied.fingerprint_current);
            assert_eq!(applied.summary, "4 streamlines");
        }
        assert_eq!(
            cache.dipy_tractography_results[&WorkflowNodeUuid(1)].fingerprint,
            9
        );
        assert_eq!(
            cache.yeh_tractography_results[&WorkflowNodeUuid(2)].fingerprint,
            9
        );
    }
}

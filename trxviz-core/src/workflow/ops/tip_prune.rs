//! Topology-Informed Pruning op (DSI-Studio TIP).
//!
//! Voxelizes the currently-selected streamlines into a density grid,
//! drops streamlines that traverse enough low-support voxels, and
//! iterates. See `crate::workflow::tip` for the core algorithm.

use std::hash::{Hash, Hasher};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input, mark_expensive_success, prime_expensive_record,
    sync_node_state_from_run_record,
};
use crate::workflow::tip::{prune_by_topology, TipParams};
use crate::workflow::types::CachedTipPrune;
use crate::workflow::WorkflowEvalMode;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TipPruneOp {
    #[serde(default = "default_voxel_size_mm")]
    pub voxel_size_mm: f32,
    #[serde(default = "default_iterations")]
    pub iterations: u32,
    #[serde(default = "default_min_support")]
    pub min_support: u32,
    #[serde(default)]
    pub max_unsupported_fraction: f32,
}

fn default_voxel_size_mm() -> f32 { 1.0 }
fn default_iterations() -> u32 { 16 }
fn default_min_support() -> u32 { 1 }

impl Default for TipPruneOp {
    fn default() -> Self {
        Self {
            voxel_size_mm: default_voxel_size_mm(),
            iterations: default_iterations(),
            min_support: default_min_support(),
            max_unsupported_fraction: 0.0,
        }
    }
}

impl WorkflowOp for TipPruneOp {
    fn tag(&self) -> &'static str {
        "tip_prune"
    }

    fn title(&self) -> &'static str {
        "TIP Prune"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<EvaluatedValue>> {
        let mut flow = expect_streamline_input(ctx.inputs, self.title())?;
        let before = flow.selected_streamlines.len();

        // Fingerprint over params + the upstream dataset's semantic shape
        // + a sampled projection of the selected-streamline set. Matches
        // the semantic-hash approach Yeh uses (see yeh_tractography.rs).
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.voxel_size_mm.to_bits().hash(&mut h);
            self.iterations.hash(&mut h);
            self.min_support.hash(&mut h);
            self.max_unsupported_fraction.to_bits().hash(&mut h);
            flow.dataset.gpu_data.nb_streamlines.hash(&mut h);
            flow.dataset.gpu_data.nb_vertices.hash(&mut h);
            before.hash(&mut h);
            let stride = (before / 64).max(1);
            for i in (0..before).step_by(stride) {
                flow.selected_streamlines[i].hash(&mut h);
            }
            h.finish()
        };
        let upstream_stale = ctx.upstream_stale();

        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        prime_expensive_record(record, fingerprint);
        sync_node_state_from_run_record(ctx.node_state, record);

        let cached_matches = ctx
            .execution_cache
            .tip_prune_cache
            .get(&ctx.node.uuid)
            .map(|c| c.fingerprint == fingerprint)
            .unwrap_or(false);

        let should_rebuild = !cached_matches || upstream_stale;
        let rebuild_now = should_rebuild && ctx.eval_mode == WorkflowEvalMode::Settled;

        if rebuild_now {
            let params = TipParams {
                voxel_size_mm: self.voxel_size_mm,
                iterations: self.iterations,
                min_support: self.min_support,
                max_unsupported_fraction: self.max_unsupported_fraction,
            };
            let mut selected = flow.selected_streamlines.clone();
            let report =
                prune_by_topology(&flow.dataset.gpu_data, &mut selected, &params);
            let summary = format!(
                "{}→{} (−{}) in {} iter{}",
                before,
                report.kept,
                report.removed,
                report.iterations_run,
                if report.iterations_run == 1 { "" } else { "s" }
            );
            ctx.execution_cache.tip_prune_cache.insert(
                ctx.node.uuid,
                CachedTipPrune {
                    fingerprint,
                    selected,
                    summary: summary.clone(),
                },
            );
            let record = ctx
                .execution_cache
                .node_runs
                .entry(ctx.node.uuid)
                .or_default();
            mark_expensive_success(record, fingerprint, summary.clone());
            sync_node_state_from_run_record(ctx.node_state, record);
            ctx.node_state.summary = summary;
        }

        let cache = ctx.execution_cache.tip_prune_cache.get(&ctx.node.uuid);
        let stale = !cached_matches || upstream_stale;

        // Fresh cache hit: emit the cached pruned selection.
        if !stale {
            if let Some(c) = cache {
                flow.selected_streamlines = c.selected.clone();
                ctx.node_state.summary = c.summary.clone();
                return Ok(vec![EvaluatedValue {
                    value: WorkflowValue::Streamline(flow),
                    stale: false,
                }]);
            }
        }

        // Stale cache: serve last-known result but mark stale so downstream
        // skips expensive work until the user triggers a rerun.
        if let Some(c) = cache {
            flow.selected_streamlines = c.selected.clone();
            ctx.node_state.summary = format!("{} (stale)", c.summary);
            return Ok(vec![EvaluatedValue {
                value: WorkflowValue::Streamline(flow),
                stale: true,
            }]);
        }

        // No cache yet — pass input through unchanged, marked stale, so the
        // user sees something reasonable until they click Run.
        ctx.node_state.summary = "TIP not run yet — click Run downstream.".into();
        Ok(vec![EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            stale: true,
        }])
    }
}

impl From<TipPruneOp> for WorkflowNodeKind {
    fn from(op: TipPruneOp) -> Self {
        Self::TipPrune {
            voxel_size_mm: op.voxel_size_mm,
            iterations: op.iterations,
            min_support: op.min_support,
            max_unsupported_fraction: op.max_unsupported_fraction,
        }
    }
}

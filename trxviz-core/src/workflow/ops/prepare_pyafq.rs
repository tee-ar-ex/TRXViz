//! `PreparePyafqPlanOp` — builds a `TrackingPlan` from a pyAFQ derivatives
//! directory + a single canonical bundle name. Mirrors `PrepareHausdorffPlanOp`
//! but with no upstream ODF/streamline inputs — pyAFQ has already warped the
//! waypoint / exclusion / endpoint ROIs into subject space, so this op only
//! needs to discover those NIfTIs on disk and load them.
//!
//! See `gpu/plan_prep/pyafq.rs` for the file-discovery + mask-construction
//! logic, and `pyafq_bundles.rs` for the static catalog of canonical
//! bundles.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::WorkflowResult;
use crate::gpu::plan_prep::pyafq::PyafqPlanParams;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{PyafqPlanJob, WorkflowValue};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, VolumeBacking, WorkflowNodeKind, WorkflowOp,
    prime_expensive_record, sync_node_state_from_run_record,
};
use super::pyafq_bundles::{DEFAULT_DIST_TO_ENDPOINT_MM, DEFAULT_DIST_TO_WAYPOINT_MM, lookup};

fn default_to_space() -> String {
    String::new()
}
fn default_dist_to_waypoint_mm() -> f32 {
    DEFAULT_DIST_TO_WAYPOINT_MM
}
fn default_dist_to_exclusion_mm() -> f32 {
    0.0
}
fn default_dist_to_endpoint_mm() -> f32 {
    DEFAULT_DIST_TO_ENDPOINT_MM
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PreparePyafqPlanOp {
    /// pyAFQ derivatives root for one subject/session (or a parent that
    /// contains exactly one `ROIs/` subdir for the chosen bundle).
    #[serde(default)]
    pub working_dir: String,
    /// Display name from `PYAFQ_BUNDLES` (e.g. "Left Corticospinal").
    #[serde(default)]
    pub bundle_name: String,
    /// pyAFQ's `to_space` token. Default is empty — the inspector
    /// auto-detects the most common `_space-{X}_desc-` token in the
    /// working directory when this is blank.
    #[serde(default = "default_to_space")]
    pub to_space: String,
    /// Dilation radius applied to each waypoint mask. Mirrors pyAFQ's
    /// `dist_to_waypoint`.
    #[serde(default = "default_dist_to_waypoint_mm")]
    pub dist_to_waypoint_mm: f32,
    /// Dilation radius applied to each exclusion mask. pyAFQ uses 0 mm
    /// (point-in-mask); exposed for tweaking.
    #[serde(default = "default_dist_to_exclusion_mm")]
    pub dist_to_exclusion_mm: f32,
    /// Dilation radius applied to start / end masks. Mirrors pyAFQ's
    /// `dist_to_atlas`.
    #[serde(default = "default_dist_to_endpoint_mm")]
    pub dist_to_endpoint_mm: f32,
    /// `Some` overrides the bundle's catalog `min_len_mm`.
    #[serde(default)]
    pub override_min_len_mm: Option<f32>,
    #[serde(default)]
    pub override_max_len_mm: Option<f32>,
}

impl Default for PreparePyafqPlanOp {
    fn default() -> Self {
        Self {
            working_dir: String::new(),
            bundle_name: String::new(),
            to_space: default_to_space(),
            dist_to_waypoint_mm: default_dist_to_waypoint_mm(),
            dist_to_exclusion_mm: default_dist_to_exclusion_mm(),
            dist_to_endpoint_mm: default_dist_to_endpoint_mm(),
            override_min_len_mm: None,
            override_max_len_mm: None,
        }
    }
}

impl WorkflowOp for PreparePyafqPlanOp {
    fn tag(&self) -> &'static str {
        "prepare_pyafq_plan"
    }

    fn title(&self) -> &'static str {
        "Prepare pyAFQ Plan"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::TrackingPlan,
            PortKind::VoxelMask, // include union (visualization)
            PortKind::VoxelMask, // exclude union
            PortKind::VoxelMask, // start
            PortKind::VoxelMask, // end
            PortKind::Volume,    // continuous probability map (omitted when none)
        ]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }

    fn citation_keys(&self) -> &'static [&'static str] {
        &["yeatman2012tract"]
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        let bundle = if self.bundle_name.is_empty() {
            "the selected bundle"
        } else {
            self.bundle_name.as_str()
        };
        Some(Cow::Owned(format!(
            "A tractography plan for {bundle} was prepared from pyAFQ \
             derivatives [@yeatman2012tract] using a {wp:.1}-mm waypoint \
             tolerance, {ex:.1}-mm exclusion tolerance, and {ep:.1}-mm \
             endpoint tolerance.",
            wp = self.dist_to_waypoint_mm,
            ex = self.dist_to_exclusion_mm,
            ep = self.dist_to_endpoint_mm,
        )))
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        if self.working_dir.is_empty() {
            ctx.node_state.summary = "Pick a pyAFQ derivatives directory.".into();
            return Ok(empty_outputs());
        }
        if self.bundle_name.is_empty() {
            ctx.node_state.summary = "Pick a bundle.".into();
            return Ok(empty_outputs());
        }
        let Some(bundle_spec) = lookup(&self.bundle_name) else {
            ctx.node_state.summary = format!("Unknown bundle '{}'.", self.bundle_name);
            return Ok(empty_outputs());
        };

        let working_dir = PathBuf::from(&self.working_dir);

        // Fingerprint: any change here means the cache is stale.
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.working_dir.hash(&mut h);
            self.bundle_name.hash(&mut h);
            self.to_space.hash(&mut h);
            self.dist_to_waypoint_mm.to_bits().hash(&mut h);
            self.dist_to_exclusion_mm.to_bits().hash(&mut h);
            self.dist_to_endpoint_mm.to_bits().hash(&mut h);
            self.override_min_len_mm.map(f32::to_bits).hash(&mut h);
            self.override_max_len_mm.map(f32::to_bits).hash(&mut h);
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
            .pyafq_plan_cache
            .get(&ctx.node.uuid)
            .map(|c| c.fingerprint == fingerprint)
            .unwrap_or(false);

        // Plan build runs off-thread: when stale, push a job onto the
        // scene plan and let the GUI worker pool dispatch it. The Finished
        // arm fills `pyafq_plan_cache` and calls `mark_expensive_success`.
        let stale = !cached_matches || upstream_stale;

        if stale {
            let params = PyafqPlanParams {
                to_space: self.to_space.clone(),
                dist_to_waypoint_mm: self.dist_to_waypoint_mm,
                dist_to_exclusion_mm: self.dist_to_exclusion_mm,
                dist_to_endpoint_mm: self.dist_to_endpoint_mm,
                override_min_len_mm: self.override_min_len_mm,
                override_max_len_mm: self.override_max_len_mm,
            };

            ctx.scene_plan.pyafq_plan_jobs.push(PyafqPlanJob {
                node_uuid: ctx.node.uuid,
                fingerprint,
                label: ctx.node.label.clone(),
                working_dir,
                bundle_spec,
                params,
            });
        }

        let cache = ctx.execution_cache.pyafq_plan_cache.get(&ctx.node.uuid);

        let Some(cached) = cache else {
            ctx.node_state.summary = "Plan not built yet — click Run on a downstream node.".into();
            return Ok(empty_outputs_stale());
        };

        ctx.node_state.summary = if stale {
            format!("{} (stale)", cached.summary)
        } else {
            cached.summary.clone()
        };

        let mut outputs = vec![
            EvaluatedValue {
                value: WorkflowValue::TrackingPlan(cached.plan.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.include_mask.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.exclude_mask.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.start_mask.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.end_mask.clone()),
                stale,
            },
        ];
        if let Some(arc) = &cached.prob_map {
            outputs.push(EvaluatedValue {
                value: WorkflowValue::Volume(VolumeBacking::from_scalars((**arc).clone())),
                stale,
            });
        }
        Ok(outputs)
    }
}

/// Five empty placeholders for the case where the user hasn't picked a
/// directory or bundle yet. Marked `stale = false` because they are the
/// correct steady-state output for this configuration (no work pending).
/// The 6th (probability map) port is left unwired by omitting it.
fn empty_outputs() -> Vec<EvaluatedValue> {
    let empty_mask = Arc::new(crate::workflow::types::VoxelMask {
        dims: [0, 0, 0],
        voxel_to_ras: glam::Mat4::IDENTITY,
        data: Vec::new(),
        ..Default::default()
    });
    let empty_plan = Arc::new(crate::workflow::types::TrackingPlan {
        label: String::new(),
        grid_dims: [0, 0, 0],
        voxel_to_ras: glam::Mat4::IDENTITY,
        seed_mask: None,
        limiting_mask: None,
        roa_mask: None,
        term_mask: None,
        roi_masks: Vec::new(),
        end_masks: Vec::new(),
        no_end_mask: None,
        post_filter: None,
        min_len_mm: None,
        max_len_mm: None,
        max_angle_deg: None,
        step_size_mm: None,
        fixel_threshold: None,
        smooth_fraction: None,
        tolerance_mm: None,
        fixel_otsu: None,
    });
    vec![
        EvaluatedValue {
            value: WorkflowValue::TrackingPlan(empty_plan),
            stale: false,
        },
        EvaluatedValue {
            value: WorkflowValue::VoxelMask(empty_mask.clone()),
            stale: false,
        },
        EvaluatedValue {
            value: WorkflowValue::VoxelMask(empty_mask.clone()),
            stale: false,
        },
        EvaluatedValue {
            value: WorkflowValue::VoxelMask(empty_mask.clone()),
            stale: false,
        },
        EvaluatedValue {
            value: WorkflowValue::VoxelMask(empty_mask),
            stale: false,
        },
    ]
}

/// Empty placeholders, but marked stale so downstream nodes know to wait
/// for the cache to fill on the next Settled pass.
fn empty_outputs_stale() -> Vec<EvaluatedValue> {
    let mut out = empty_outputs();
    for ev in &mut out {
        ev.stale = true;
    }
    out
}

impl From<PreparePyafqPlanOp> for WorkflowNodeKind {
    fn from(op: PreparePyafqPlanOp) -> Self {
        Self::PreparePyafqPlan {
            working_dir: op.working_dir,
            bundle_name: op.bundle_name,
            to_space: op.to_space,
            dist_to_waypoint_mm: op.dist_to_waypoint_mm,
            dist_to_exclusion_mm: op.dist_to_exclusion_mm,
            dist_to_endpoint_mm: op.dist_to_endpoint_mm,
            override_min_len_mm: op.override_min_len_mm,
            override_max_len_mm: op.override_max_len_mm,
        }
    }
}

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::error::WorkflowResult;
use crate::gpu::plan_prep::hausdorff::{HausdorffPlanParams, build_hausdorff_plan};
use crate::workflow::WorkflowEvalMode;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{CachedHausdorffPlan, WorkflowValue};
use odx_rs::qc::OtsuScope;

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp, mark_expensive_success,
    prime_expensive_record, sync_node_state_from_run_record,
};

fn default_tolerance_mm() -> f32 {
    12.0
}
fn default_seed_tolerance_mm() -> f32 {
    2.0
}
fn default_seed_fixel_otsu_factor() -> f32 {
    0.5
}
fn default_not_end_fixel_otsu_factor() -> f32 {
    0.9
}
fn default_max_reference_points() -> u32 {
    20_000
}
fn default_otsu_scope() -> OtsuScope {
    OtsuScope::AllFixels
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PrepareHausdorffPlanOp {
    #[serde(default = "default_tolerance_mm")]
    pub tolerance_mm: f32,
    /// Dilation radius for the seed mask (mm). Kept ≤ `tolerance_mm` so
    /// seeds are drawn from near the reference bundle, not the full
    /// limiting envelope. DSI-Studio autotrack parity: small value.
    #[serde(default = "default_seed_tolerance_mm")]
    pub seed_tolerance_mm: f32,
    /// Which per-fixel scalar drives the Otsu threshold. `None` =
    /// auto-resolve in odx-rs priority order (amplitude, afd, qa).
    #[serde(default)]
    pub tracking_metric: Option<String>,
    /// `AllFixels` (default, catches smaller secondary peaks) or
    /// `PrimaryPeak` (DSI-Studio parity: Otsu over primary-peak-per-voxel).
    #[serde(default = "default_otsu_scope")]
    pub otsu_scope: OtsuScope,
    /// `seed_mask` = limiting ∩ (primary-peak metric ≥ factor × fixel_otsu).
    /// DSI-Studio-equivalent default 0.5 (= `default_otsu − 0.1` in their code).
    #[serde(default = "default_seed_fixel_otsu_factor")]
    pub seed_fixel_otsu_factor: f32,
    /// `no_end_mask` = limiting ∩ (primary-peak metric > factor × fixel_otsu).
    /// DSI-Studio-equivalent default 0.9 (= `2·(default_otsu+0.1) − (default_otsu−0.1)`).
    #[serde(default = "default_not_end_fixel_otsu_factor")]
    pub not_end_fixel_otsu_factor: f32,
    #[serde(default = "default_max_reference_points")]
    pub max_reference_points: u32,
}

impl Default for PrepareHausdorffPlanOp {
    fn default() -> Self {
        Self {
            tolerance_mm: default_tolerance_mm(),
            seed_tolerance_mm: default_seed_tolerance_mm(),
            tracking_metric: None,
            otsu_scope: default_otsu_scope(),
            seed_fixel_otsu_factor: default_seed_fixel_otsu_factor(),
            not_end_fixel_otsu_factor: default_not_end_fixel_otsu_factor(),
            max_reference_points: default_max_reference_points(),
        }
    }
}

impl WorkflowOp for PrepareHausdorffPlanOp {
    fn tag(&self) -> &'static str {
        "prepare_hausdorff_plan"
    }

    fn title(&self) -> &'static str {
        "Prepare Hausdorff Plan"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::OdfField, PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::TrackingPlan,
            PortKind::VoxelMask, // seed
            PortKind::VoxelMask, // limiting
            PortKind::VoxelMask, // no_end
        ]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }

    fn citation_keys(&self) -> &'static [&'static str] {
        // The Hausdorff-based plan prep mirrors the DSI Studio
        // autotrack/shape-analysis approach: reference-bundle-derived
        // seed/limiting/no-end masks plus Otsu-thresholded fixel metrics.
        &["yeh2020shape"]
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "A tractography plan was prepared from a reference bundle following \
             the DSI Studio shape-analysis approach [@yeh2020shape], using a \
             {tol:.1}-mm Hausdorff tolerance to define the limiting region, a \
             {seed_tol:.1}-mm seed dilation, and primary-peak Otsu factors of \
             {seed_factor:.2} (seed mask) and {no_end_factor:.2} (no-end mask); \
             up to {max_ref} reference points were retained.",
            tol = self.tolerance_mm,
            seed_tol = self.seed_tolerance_mm,
            seed_factor = self.seed_fixel_otsu_factor,
            no_end_factor = self.not_end_fixel_otsu_factor,
            max_ref = self.max_reference_points,
        )))
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let odf_field = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::OdfField(f) => f.clone(),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Prepare Hausdorff Plan requires an ODF field input".into(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Prepare Hausdorff Plan requires an ODF field input".into(),
                ));
            }
        };

        let reference_flow = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::Streamline(flow) => flow.clone(),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Prepare Hausdorff Plan requires a reference streamline input".into(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Prepare Hausdorff Plan requires a reference streamline input".into(),
                ));
            }
        };

        let loaded_odx = ctx.odx_assets.get(&odf_field.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(
                "Prepare Hausdorff Plan: missing ODX asset".into(),
            )
        })?;

        // Otsu is memoized on the OdxScene, so calling it every frame is
        // cheap after the first; safe to do in Interactive mode.
        let fixel_otsu = loaded_odx
            .scene
            .fixel_otsu(self.tracking_metric.as_deref(), self.otsu_scope)
            .map_err(|e| {
                crate::error::WorkflowError::Evaluation(format!(
                    "Prepare Hausdorff Plan: could not compute fixel Otsu — {e}"
                ))
            })?;

        let selected: Vec<u32> = reference_flow
            .selected_streamlines
            .iter()
            .map(|s| s.0)
            .collect();

        // Fingerprint: any change here means the cache is stale.
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            odf_field.source_id.hash(&mut h);
            reference_flow.dataset.gpu_data.nb_streamlines.hash(&mut h);
            reference_flow.dataset.gpu_data.nb_vertices.hash(&mut h);
            selected.len().hash(&mut h);
            // Sample a few selected ids so a selection change invalidates.
            let stride = (selected.len() / 64).max(1);
            for i in (0..selected.len()).step_by(stride) {
                selected[i].hash(&mut h);
            }
            self.tolerance_mm.to_bits().hash(&mut h);
            self.seed_tolerance_mm.to_bits().hash(&mut h);
            fixel_otsu.metric_name.hash(&mut h);
            fixel_otsu.threshold.to_bits().hash(&mut h);
            (self.otsu_scope as u8).hash(&mut h);
            self.seed_fixel_otsu_factor.to_bits().hash(&mut h);
            self.not_end_fixel_otsu_factor.to_bits().hash(&mut h);
            self.max_reference_points.hash(&mut h);
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
            .hausdorff_plan_cache
            .get(&ctx.node.uuid)
            .map(|c| c.fingerprint == fingerprint)
            .unwrap_or(false);

        // Decision tree:
        //   cache hit + upstream fresh → serve cached (stale=false)
        //   cache miss / stale + Interactive → serve cached if any with stale=true,
        //                                       else empty placeholders with stale=true
        //   cache miss / stale + Settled    → rebuild, cache, return (stale=false)
        let should_rebuild = !cached_matches || upstream_stale;
        let rebuild_now = should_rebuild && ctx.eval_mode == WorkflowEvalMode::Settled;

        if rebuild_now {
            let params = HausdorffPlanParams {
                tolerance_mm: self.tolerance_mm,
                seed_tolerance_mm: self.seed_tolerance_mm,
                tracking_metric: fixel_otsu.metric_name.clone(),
                fixel_otsu: fixel_otsu.threshold,
                seed_fixel_otsu_factor: self.seed_fixel_otsu_factor,
                not_end_fixel_otsu_factor: self.not_end_fixel_otsu_factor,
                max_reference_points: self.max_reference_points as usize,
            };

            let outputs = build_hausdorff_plan(
                &loaded_odx.scene,
                &reference_flow.dataset.gpu_data,
                &selected,
                ctx.node.label.clone(),
                &params,
            );

            let summary = format!(
                "{}: Otsu = {:.4} ({} samples, {})",
                fixel_otsu.metric_name,
                fixel_otsu.threshold,
                fixel_otsu.n_values,
                match fixel_otsu.scope {
                    OtsuScope::AllFixels => "all fixels",
                    OtsuScope::PrimaryPeak => "primary peak",
                }
            );

            ctx.execution_cache.hausdorff_plan_cache.insert(
                ctx.node.uuid,
                CachedHausdorffPlan {
                    fingerprint,
                    plan: Arc::new(outputs.plan),
                    seed_mask: outputs.seed_mask,
                    limiting_mask: outputs.limiting_mask,
                    no_end_mask: outputs.no_end_mask,
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

        let cache = ctx.execution_cache.hausdorff_plan_cache.get(&ctx.node.uuid);
        let stale = !cached_matches || upstream_stale;

        // No cache yet: emit empty placeholders with stale=true so
        // downstream nodes mark themselves stale too. Once a downstream
        // Run fires, Settled mode fills the cache on this same pass.
        let Some(cached) = cache else {
            ctx.node_state.summary = "Plan not built yet — click Run on a downstream node.".into();
            let dims = loaded_odx.scene.dimensions();
            let dims_u32 = [dims[0] as u32, dims[1] as u32, dims[2] as u32];
            let voxel_to_ras = loaded_odx.scene.voxel_to_ras();
            let empty_mask = Arc::new(crate::workflow::types::VoxelMask {
                dims: dims_u32,
                voxel_to_ras,
                data: Vec::new(),
            });
            let empty_plan = Arc::new(crate::workflow::types::TrackingPlan {
                label: ctx.node.label.clone(),
                grid_dims: dims_u32,
                voxel_to_ras,
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
                tolerance_mm: Some(self.tolerance_mm),
                fixel_otsu: Some(fixel_otsu.threshold),
            });
            return Ok(vec![
                EvaluatedValue {
                    value: WorkflowValue::TrackingPlan(empty_plan),
                    stale: true,
                },
                EvaluatedValue {
                    value: WorkflowValue::VoxelMask(empty_mask.clone()),
                    stale: true,
                },
                EvaluatedValue {
                    value: WorkflowValue::VoxelMask(empty_mask.clone()),
                    stale: true,
                },
                EvaluatedValue {
                    value: WorkflowValue::VoxelMask(empty_mask),
                    stale: true,
                },
            ]);
        };

        ctx.node_state.summary = if stale {
            format!("{} (stale)", cached.summary)
        } else {
            cached.summary.clone()
        };

        Ok(vec![
            EvaluatedValue {
                value: WorkflowValue::TrackingPlan(cached.plan.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.seed_mask.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.limiting_mask.clone()),
                stale,
            },
            EvaluatedValue {
                value: WorkflowValue::VoxelMask(cached.no_end_mask.clone()),
                stale,
            },
        ])
    }
}

impl From<PrepareHausdorffPlanOp> for WorkflowNodeKind {
    fn from(op: PrepareHausdorffPlanOp) -> Self {
        Self::PrepareHausdorffPlan {
            tolerance_mm: op.tolerance_mm,
            seed_tolerance_mm: op.seed_tolerance_mm,
            tracking_metric: op.tracking_metric,
            otsu_scope: op.otsu_scope,
            seed_fixel_otsu_factor: op.seed_fixel_otsu_factor,
            not_end_fixel_otsu_factor: op.not_end_fixel_otsu_factor,
            max_reference_points: op.max_reference_points,
        }
    }
}

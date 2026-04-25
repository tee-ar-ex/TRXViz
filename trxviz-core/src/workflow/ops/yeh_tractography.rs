use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::trx_data::TrxGpuData;
use crate::error::WorkflowResult;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{
    StreamlineDataset, StreamlineFlow, TrackingPlan, VoxelMask, WorkflowValue, YehTractographyPlan,
};

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, prime_expensive_record,
    sync_node_state_from_run_record,
};

fn default_step_size() -> f32 {
    1.0
}
fn default_max_angle() -> f32 {
    60.0
}
fn default_min_len() -> f32 {
    10.0
}
fn default_max_len() -> f32 {
    300.0
}
fn default_fixel_threshold() -> f32 {
    0.05
}
fn default_smooth() -> f32 {
    0.25
}
fn default_max_points() -> u32 {
    501
}
fn default_target_streamlines() -> u32 {
    30_000
}
fn default_max_seed_attempts() -> u32 {
    10_000_000
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YehTractographyOp {
    #[serde(default = "default_step_size")]
    pub step_size_mm: f32,
    #[serde(default = "default_max_angle")]
    pub max_angle_deg: f32,
    #[serde(default = "default_min_len")]
    pub min_len_mm: f32,
    #[serde(default = "default_max_len")]
    pub max_len_mm: f32,
    #[serde(default = "default_fixel_threshold")]
    pub fixel_threshold: f32,
    #[serde(default = "default_smooth")]
    pub smooth_fraction: f32,
    #[serde(default = "default_max_points")]
    pub max_points: u32,
    #[serde(default = "default_target_streamlines")]
    pub target_streamlines: u32,
    #[serde(default = "default_max_seed_attempts")]
    pub max_seed_attempts: u32,
    #[serde(default)]
    pub rng_seed: u64,
}

impl Default for YehTractographyOp {
    fn default() -> Self {
        Self {
            step_size_mm: default_step_size(),
            max_angle_deg: default_max_angle(),
            min_len_mm: default_min_len(),
            max_len_mm: default_max_len(),
            fixel_threshold: default_fixel_threshold(),
            smooth_fraction: default_smooth(),
            max_points: default_max_points(),
            target_streamlines: default_target_streamlines(),
            max_seed_attempts: default_max_seed_attempts(),
            rng_seed: 42,
        }
    }
}

impl YehTractographyOp {
    fn fingerprint(
        &self,
        odx_source_id: crate::data::loaded_files::FileId,
        mask: Option<&VoxelMask>,
    ) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        odx_source_id.hash(&mut h);
        match mask {
            Some(m) => {
                1u8.hash(&mut h);
                m.dims.hash(&mut h);
                for c in m.voxel_to_ras.to_cols_array() {
                    c.to_bits().hash(&mut h);
                }
                m.data.len().hash(&mut h);
                let stride = (m.data.len() / 256).max(1);
                for i in (0..m.data.len()).step_by(stride) {
                    m.data[i].hash(&mut h);
                }
            }
            None => {
                0u8.hash(&mut h);
            }
        }
        self.step_size_mm.to_bits().hash(&mut h);
        self.max_angle_deg.to_bits().hash(&mut h);
        self.min_len_mm.to_bits().hash(&mut h);
        self.max_len_mm.to_bits().hash(&mut h);
        self.fixel_threshold.to_bits().hash(&mut h);
        self.smooth_fraction.to_bits().hash(&mut h);
        self.max_points.hash(&mut h);
        self.target_streamlines.hash(&mut h);
        self.max_seed_attempts.hash(&mut h);
        self.rng_seed.hash(&mut h);
        h.finish()
    }

    fn empty_flow(label: &str) -> StreamlineFlow {
        let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(Vec::new(), vec![0]));
        let dataset = Arc::new(StreamlineDataset {
            name: label.to_string(),
            gpu_data,
            backing: crate::data::loaded_files::StreamlineBacking::Derived(Arc::new(
                trx_rs::Tractogram::new(),
            )),
        });
        StreamlineFlow {
            dataset,
            selected_streamlines: Vec::new(),
            color_mode: crate::data::trx_data::ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
        }
    }
}

impl WorkflowOp for YehTractographyOp {
    fn tag(&self) -> &'static str {
        "yeh_tractography"
    }

    fn title(&self) -> &'static str {
        "Yeh Tracking (Fixel)"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::Fixels,
            PortKind::VoxelMask,
            PortKind::TrackingPlan,
        ]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }

    fn citation_keys(&self) -> &'static [&'static str] {
        // Matches DSI Studio's own tractography boilerplate, which cites
        // Yeh 2013 (deterministic GQI tracking) alongside Yeh 2020 for
        // the augmented tracking strategies used to improve
        // reproducibility. The 2025 software paper credits DSI Studio
        // itself.
        &["yeh2025dsistudio", "yeh2013gqi", "yeh2020shape"]
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "Fixel-based deterministic tractography was performed following the \
             DSI Studio approach [@yeh2025dsistudio;@yeh2013gqi] with augmented \
             tracking strategies [@yeh2020shape] to improve reproducibility, \
             using a {step:.2}-mm step size, a maximum turning angle of {angle:.0}°, \
             length bounds of {min_len:.0}–{max_len:.0} mm, a fixel threshold of \
             {fx:.2}, and a smoothing fraction of {smooth:.2}; seeding targeted \
             {target} streamlines (up to {attempts} seed attempts).",
            step = self.step_size_mm,
            angle = self.max_angle_deg,
            min_len = self.min_len_mm,
            max_len = self.max_len_mm,
            fx = self.fixel_threshold,
            smooth = self.smooth_fraction,
            target = self.target_streamlines,
            attempts = self.max_seed_attempts,
        )))
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let fixels = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::Fixels(f) => f.clone(),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Yeh Tracking requires a Fixels input".into(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Yeh Tracking requires a Fixels input".into(),
                ));
            }
        };

        // Optional seed mask — when absent we seed over every voxel with ≥1 fixel.
        let direct_mask: Option<Arc<VoxelMask>> = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Yeh Tracking: second input must be a VoxelMask (or unconnected)".into(),
                    ));
                }
            },
            None => None,
        };

        // Optional TrackingPlan — when present, its seed_mask and length
        // bounds override the op's direct inputs / slider values. Anything
        // the plan leaves as `None` falls back to the local value.
        let plan_input: Option<Arc<TrackingPlan>> = match ctx.inputs.get(2).and_then(|v| v.as_ref())
        {
            Some(ev) => match &ev.value {
                WorkflowValue::TrackingPlan(p) => Some(p.clone()),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Yeh Tracking: third input must be a TrackingPlan (or unconnected)".into(),
                    ));
                }
            },
            None => None,
        };

        // Plan.seed_mask takes precedence if provided; otherwise the direct
        // VoxelMask input; otherwise None (whole-brain).
        let seed_mask: Option<Arc<VoxelMask>> = plan_input
            .as_ref()
            .and_then(|p| p.seed_mask.clone())
            .or(direct_mask);

        let effective = super::tracking_params::EffectiveTrackingParams::merge(
            super::tracking_params::OpTrackingDefaults {
                min_len_mm: self.min_len_mm,
                max_len_mm: self.max_len_mm,
                max_angle_deg: self.max_angle_deg,
                step_size_mm: self.step_size_mm,
                fixel_threshold: self.fixel_threshold,
                smooth_fraction: Some(self.smooth_fraction),
            },
            plan_input.as_deref(),
        );

        if let Some(p) = plan_input.as_ref() {
            super::tracking_params::record_plan_overrides(
                ctx.node_state,
                p,
                super::tracking_params::TrackingFieldSet::YEH,
            );
        }

        let loaded_odx = ctx.odx_assets.get(&fixels.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation("Missing ODX asset for Yeh tracking".into())
        })?;

        let odx_source_id = fixels.source_id;
        // Fingerprint includes the effective overrides so a plan change
        // triggers re-run.
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.fingerprint(odx_source_id, seed_mask.as_deref())
                .hash(&mut h);
            effective.hash_into(&mut h);
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

        let cached = ctx
            .execution_cache
            .yeh_tractography_results
            .get(&ctx.node.uuid)
            .cloned();

        let (flow, stale) = if let Some(ref c) = cached {
            if c.fingerprint == fingerprint && !upstream_stale {
                (c.flow.clone(), false)
            } else {
                (Self::empty_flow(&ctx.node.label), true)
            }
        } else {
            (Self::empty_flow(&ctx.node.label), true)
        };

        // Pass through any region/filter the plan carries. These fields
        // take effect at tracking time (per-step for limiting/roa/term,
        // post-hoc for roi/end/no_end/post_filter).
        let (limiting_mask, roa_mask, term_mask, roi_masks, end_masks, no_end_mask, post_filter) =
            if let Some(p) = plan_input.as_ref() {
                (
                    p.limiting_mask.clone(),
                    p.roa_mask.clone(),
                    p.term_mask.clone(),
                    p.roi_masks.clone(),
                    p.end_masks.clone(),
                    p.no_end_mask.clone(),
                    p.post_filter.clone(),
                )
            } else {
                (None, None, None, Vec::new(), Vec::new(), None, None)
            };

        if stale {
            ctx.scene_plan
                .yeh_tractography_plans
                .push(YehTractographyPlan {
                    node_uuid: ctx.node.uuid,
                    label: ctx.node.label.clone(),
                    fingerprint: super::super::op::ContentHash(fingerprint),
                    odx_source_id,
                    odx_scene: loaded_odx.scene.clone(),
                    seed_mask,
                    limiting_mask,
                    roa_mask,
                    term_mask,
                    roi_masks,
                    end_masks,
                    no_end_mask,
                    post_filter,
                    step_size_mm: effective.step_size_mm,
                    max_angle_deg: effective.max_angle_deg,
                    min_len_mm: effective.min_len_mm,
                    max_len_mm: effective.max_len_mm,
                    fixel_threshold: effective.fixel_threshold,
                    smooth_fraction: effective.smooth_fraction.unwrap_or(self.smooth_fraction),
                    max_points: self.max_points,
                    target_streamlines: self.target_streamlines,
                    max_seed_attempts: self.max_seed_attempts,
                    rng_seed: self.rng_seed,
                    fixel_otsu: plan_input.as_ref().and_then(|p| p.fixel_otsu),
                });
        }

        Ok(vec![super::super::EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            stale,
        }])
    }
}

impl From<YehTractographyOp> for WorkflowNodeKind {
    fn from(op: YehTractographyOp) -> Self {
        Self::YehTractography {
            step_size_mm: op.step_size_mm,
            max_angle_deg: op.max_angle_deg,
            min_len_mm: op.min_len_mm,
            max_len_mm: op.max_len_mm,
            fixel_threshold: op.fixel_threshold,
            smooth_fraction: op.smooth_fraction,
            max_points: op.max_points,
            target_streamlines: op.target_streamlines,
            max_seed_attempts: op.max_seed_attempts,
            rng_seed: op.rng_seed,
        }
    }
}

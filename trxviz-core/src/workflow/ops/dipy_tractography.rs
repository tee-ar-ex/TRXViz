use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::trx_data::TrxGpuData;
use crate::error::WorkflowResult;
use crate::units::StreamlineIndex;

use super::super::{
    CachedTractographyResult, DipyDirectionGetter, DipyTractographyPlan, EvalCtx, PortKind,
    StreamlineDataset, StreamlineFlow, VoxelMask, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    mark_expensive_success, prime_expensive_record, sync_node_state_from_run_record,
};
use crate::data::loaded_files::StreamlineBacking;
use crate::data::trx_data::ColorMode;

fn default_step_size() -> f32 {
    0.5
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
    0.1
}
fn default_relative_peak_threshold() -> f32 {
    0.25
}
fn default_seeds_per_voxel() -> u32 {
    1
}
fn default_max_points() -> u32 {
    501
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DipyTractographyOp {
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
    #[serde(default = "default_relative_peak_threshold")]
    pub relative_peak_threshold: f32,
    #[serde(default = "default_seeds_per_voxel")]
    pub seeds_per_voxel: u32,
    #[serde(default = "default_max_points")]
    pub max_points: u32,
    #[serde(default)]
    pub rng_seed: u64,
    pub direction_getter: DipyDirectionGetter,
}

impl Default for DipyTractographyOp {
    fn default() -> Self {
        Self {
            step_size_mm: default_step_size(),
            max_angle_deg: default_max_angle(),
            min_len_mm: default_min_len(),
            max_len_mm: default_max_len(),
            fixel_threshold: default_fixel_threshold(),
            relative_peak_threshold: default_relative_peak_threshold(),
            seeds_per_voxel: default_seeds_per_voxel(),
            max_points: default_max_points(),
            rng_seed: 42,
            direction_getter: DipyDirectionGetter::default(),
        }
    }
}

impl DipyTractographyOp {
    fn fingerprint(
        &self,
        odx_source_id: crate::data::loaded_files::FileId,
        mask: &VoxelMask,
    ) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        odx_source_id.hash(&mut h);
        mask.dims.hash(&mut h);
        for c in mask.voxel_to_ras.to_cols_array() {
            c.to_bits().hash(&mut h);
        }
        mask.data.len().hash(&mut h);
        // Hash a stride-sampled subset of bytes for cheap fingerprinting.
        let stride = (mask.data.len() / 256).max(1);
        for i in (0..mask.data.len()).step_by(stride) {
            mask.data[i].hash(&mut h);
        }
        self.step_size_mm.to_bits().hash(&mut h);
        self.max_angle_deg.to_bits().hash(&mut h);
        self.min_len_mm.to_bits().hash(&mut h);
        self.max_len_mm.to_bits().hash(&mut h);
        self.fixel_threshold.to_bits().hash(&mut h);
        self.relative_peak_threshold.to_bits().hash(&mut h);
        self.seeds_per_voxel.hash(&mut h);
        self.max_points.hash(&mut h);
        self.rng_seed.hash(&mut h);
        // Include the DG variant tag (and any inline params) in the
        // fingerprint so switching DGs invalidates cached results.
        match self.direction_getter {
            DipyDirectionGetter::Probabilistic => 0u8.hash(&mut h),
            DipyDirectionGetter::Ptt {
                probe_length_mm,
                probe_quality,
                probe_radius_mm,
                probe_count,
                max_curvature_per_mm,
                data_support_exponent,
                min_data_support,
                rejection_sampling_max_try,
            } => {
                1u8.hash(&mut h);
                probe_length_mm.to_bits().hash(&mut h);
                probe_quality.hash(&mut h);
                probe_radius_mm.to_bits().hash(&mut h);
                probe_count.hash(&mut h);
                max_curvature_per_mm.to_bits().hash(&mut h);
                data_support_exponent.to_bits().hash(&mut h);
                min_data_support.to_bits().hash(&mut h);
                rejection_sampling_max_try.hash(&mut h);
            }
        }
        h.finish()
    }

    /// Build an empty (zero-streamline) flow for the stale/pending state.
    fn empty_flow(label: &str) -> StreamlineFlow {
        let gpu_data = Arc::new(TrxGpuData::from_positions_and_offsets(Vec::new(), vec![0]));
        let dataset = Arc::new(StreamlineDataset {
            name: label.to_string(),
            gpu_data,
            backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
        });
        StreamlineFlow {
            dataset,
            selected_streamlines: Vec::new(),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
        }
    }
}

impl WorkflowOp for DipyTractographyOp {
    fn tag(&self) -> &'static str {
        "tractography"
    }

    fn title(&self) -> &'static str {
        "Dipy/GPUStreamlines Tractography (ODF)"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::OdfField,
            PortKind::VoxelMask,
            PortKind::TrackingPlan,
        ]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        // Unpack OdfField input
        let odf_field = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::OdfField(f) => f.clone(),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Dipy/GPUStreamlines tractography requires an ODF field input".into(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Dipy/GPUStreamlines tractography requires an ODF field input".into(),
                ));
            }
        };

        // Unpack seed VoxelMask input (optional when a plan supplies one).
        let direct_mask: Option<Arc<VoxelMask>> = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Dipy/GPUStreamlines tractography: second input must be a VoxelMask \
                         (or unconnected)"
                            .into(),
                    ));
                }
            },
            None => None,
        };

        // Optional TrackingPlan — overrides seed_mask + per-param sliders
        // and carries constraint masks / post filters.
        let plan_input: Option<Arc<crate::workflow::types::TrackingPlan>> =
            match ctx.inputs.get(2).and_then(|v| v.as_ref()) {
                Some(ev) => match &ev.value {
                    WorkflowValue::TrackingPlan(p) => Some(p.clone()),
                    _ => {
                        return Err(crate::error::WorkflowError::Evaluation(
                            "Dipy/GPUStreamlines tractography: third input must be a \
                             TrackingPlan (or unconnected)"
                                .into(),
                        ));
                    }
                },
                None => None,
            };

        let seed_mask: Arc<VoxelMask> = plan_input
            .as_ref()
            .and_then(|p| p.seed_mask.clone())
            .or(direct_mask)
            .ok_or_else(|| {
                crate::error::WorkflowError::Evaluation(
                    "Dipy/GPUStreamlines tractography needs either a VoxelMask input \
                     or a TrackingPlan with a seed_mask"
                        .into(),
                )
            })?;

        let loaded_odx = ctx.odx_assets.get(&odf_field.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation("Missing ODX asset for tractography".into())
        })?;

        // Param overrides from the plan.
        let effective_min_len = plan_input
            .as_ref()
            .and_then(|p| p.min_len_mm)
            .unwrap_or(self.min_len_mm);
        let effective_max_len = plan_input
            .as_ref()
            .and_then(|p| p.max_len_mm)
            .unwrap_or(self.max_len_mm);
        let effective_max_angle = plan_input
            .as_ref()
            .and_then(|p| p.max_angle_deg)
            .unwrap_or(self.max_angle_deg);
        let effective_step_size = plan_input
            .as_ref()
            .and_then(|p| p.step_size_mm)
            .unwrap_or(self.step_size_mm);
        let effective_fixel_threshold = plan_input
            .as_ref()
            .and_then(|p| p.fixel_threshold)
            .unwrap_or(self.fixel_threshold);

        // Record which fields the plan is overriding so the UI can grey out
        // the corresponding sliders and show their effective values.
        if let Some(p) = plan_input.as_ref() {
            let mut ov: Vec<String> = Vec::new();
            let mut vals: std::collections::BTreeMap<String, f32> =
                std::collections::BTreeMap::new();
            if p.seed_mask.is_some() {
                ov.push("seed_mask".into());
            }
            if let Some(v) = p.min_len_mm {
                ov.push("min_len_mm".into());
                vals.insert("min_len_mm".into(), v);
            }
            if let Some(v) = p.max_len_mm {
                ov.push("max_len_mm".into());
                vals.insert("max_len_mm".into(), v);
            }
            if let Some(v) = p.max_angle_deg {
                ov.push("max_angle_deg".into());
                vals.insert("max_angle_deg".into(), v);
            }
            if let Some(v) = p.step_size_mm {
                ov.push("step_size_mm".into());
                vals.insert("step_size_mm".into(), v);
            }
            if let Some(v) = p.fixel_threshold {
                ov.push("fixel_threshold".into());
                vals.insert("fixel_threshold".into(), v);
            }
            // Informational only — sentinel-driven random thresholds
            // center on 0.6·fixel_otsu.
            if let Some(v) = p.fixel_otsu {
                vals.insert("fixel_otsu".into(), v);
            }
            ctx.node_state.overridden_fields = ov;
            ctx.node_state.overridden_values = vals;
        }

        let odx_source_id = odf_field.source_id;
        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.fingerprint(odx_source_id, &seed_mask).hash(&mut h);
            effective_min_len.to_bits().hash(&mut h);
            effective_max_len.to_bits().hash(&mut h);
            effective_max_angle.to_bits().hash(&mut h);
            effective_step_size.to_bits().hash(&mut h);
            effective_fixel_threshold.to_bits().hash(&mut h);
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

        // Check the tractography-specific result cache
        let cached = ctx
            .execution_cache
            .dipy_tractography_results
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

        // If stale, push a DipyTractographyPlan so the app can queue the job.
        if stale {
            let (
                limiting_mask,
                roa_mask,
                term_mask,
                roi_masks,
                end_masks,
                no_end_mask,
                post_filter,
            ) = if let Some(p) = plan_input.as_ref() {
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
            ctx.scene_plan
                .dipy_tractography_plans
                .push(DipyTractographyPlan {
                    node_uuid: ctx.node.uuid,
                    label: ctx.node.label.clone(),
                    odx_source_id,
                    odx_scene: loaded_odx.scene.clone(),
                    seed_mask,
                    step_size_mm: effective_step_size,
                    max_angle_deg: effective_max_angle,
                    min_len_mm: effective_min_len,
                    max_len_mm: effective_max_len,
                    fixel_threshold: effective_fixel_threshold,
                    relative_peak_threshold: self.relative_peak_threshold,
                    seeds_per_voxel: self.seeds_per_voxel,
                    max_points: self.max_points,
                    rng_seed: self.rng_seed,
                    limiting_mask,
                    roa_mask,
                    term_mask,
                    roi_masks,
                    end_masks,
                    no_end_mask,
                    post_filter,
                    fixel_otsu: plan_input.as_ref().and_then(|p| p.fixel_otsu),
                    direction_getter: self.direction_getter,
                });
        }

        Ok(vec![super::super::EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            stale,
        }])
    }
}

impl From<DipyTractographyOp> for WorkflowNodeKind {
    fn from(op: DipyTractographyOp) -> Self {
        Self::DipyTractography {
            step_size_mm: op.step_size_mm,
            max_angle_deg: op.max_angle_deg,
            min_len_mm: op.min_len_mm,
            max_len_mm: op.max_len_mm,
            fixel_threshold: op.fixel_threshold,
            relative_peak_threshold: op.relative_peak_threshold,
            seeds_per_voxel: op.seeds_per_voxel,
            max_points: op.max_points,
            rng_seed: op.rng_seed,
            direction_getter: op.direction_getter,
        }
    }
}

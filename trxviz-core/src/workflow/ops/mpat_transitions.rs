//! `MpatTransitionsOp` — computes MITTENS v2 transition probabilities and builds/caches
//! the 26-class MPAT absorbing Markov chain (`mpat.Chain`), seed launches, fixel lobes,
//! and whole-brain reference traffic.
//!
//! Emits:
//! - Port 0 (`PortKind::MpatTransitions`): Handle to the cached transition state for downstream `MpatTractographyOp` nodes.
//! - Ports 1..=8 (`PortKind::Volume`): All 8 transition probability summary volumes:
//!   1. `CoDI (Double-ODF)` (`codi_double`)
//!   2. `CoDI (Single-ODF)` (`codi_single`)
//!   3. `CoAsy (Asymmetry)` (`coasy`)
//!   4. `P(Terminate Double)` (`p_terminate_double`)
//!   5. `P(Terminate Single)` (`p_terminate_single`)
//!   6. `Total ODF Amplitude` (`total_amplitude`)
//!   7. `KL Divergence (Single)` (`kl_single`)
//!   8. `Chain Dead-State Prob` (`chain_dead`)

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use crate::data::cifti::{ScalarKind, ScalarMetadata, VolumeScalars};
use crate::error::WorkflowResult;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{
    MpatPreset, MpatTransitionsHandle, MpatTransitionsJob, VoxelMask, WorkflowValue,
};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, VolumeBacking, WorkflowNodeKind, WorkflowOp,
    prime_expensive_record, sync_node_state_from_run_record,
};

pub fn default_mpat_angle_deg() -> f32 {
    35.0
}
pub fn default_mpat_step_vox() -> f32 {
    1.0
}
pub fn default_mpat_double_odf() -> bool {
    true
}
pub fn default_mpat_fod_thresh() -> f32 {
    0.05
}
pub fn default_mpat_fod_power() -> f32 {
    1.0
}
pub fn default_mpat_stop_mass_tau() -> f32 {
    0.0
}
pub fn default_mpat_curv_sigma_deg() -> f32 {
    0.0
}
pub fn default_mpat_probe_radius_vox() -> f32 {
    0.0
}
pub fn default_mpat_max_length_mm() -> f32 {
    250.0
}
pub fn default_mpat_wm_threshold() -> f32 {
    0.4
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MpatTransitionsOp {
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub python_path: String,
    #[serde(default)]
    pub preset: MpatPreset,
    #[serde(default = "default_mpat_angle_deg")]
    pub angle_deg: f32,
    #[serde(default = "default_mpat_step_vox")]
    pub step_vox: f32,
    #[serde(default = "default_mpat_double_odf")]
    pub double_odf: bool,
    #[serde(default = "default_mpat_fod_thresh")]
    pub fod_thresh: f32,
    #[serde(default = "default_mpat_fod_power")]
    pub fod_power: f32,
    #[serde(default = "default_mpat_stop_mass_tau")]
    pub stop_mass_tau: f32,
    #[serde(default = "default_mpat_curv_sigma_deg")]
    pub curv_sigma_deg: f32,
    #[serde(default = "default_mpat_probe_radius_vox")]
    pub probe_radius_vox: f32,
    #[serde(default = "default_mpat_max_length_mm")]
    pub max_length_mm: f32,
    #[serde(default = "default_mpat_wm_threshold")]
    pub wm_threshold: f32,
}

impl Default for MpatTransitionsOp {
    fn default() -> Self {
        Self {
            working_dir: String::new(),
            python_path: String::new(),
            preset: MpatPreset::Standard,
            angle_deg: default_mpat_angle_deg(),
            step_vox: default_mpat_step_vox(),
            double_odf: default_mpat_double_odf(),
            fod_thresh: default_mpat_fod_thresh(),
            fod_power: default_mpat_fod_power(),
            stop_mass_tau: default_mpat_stop_mass_tau(),
            curv_sigma_deg: default_mpat_curv_sigma_deg(),
            probe_radius_vox: default_mpat_probe_radius_vox(),
            max_length_mm: default_mpat_max_length_mm(),
            wm_threshold: default_mpat_wm_threshold(),
        }
    }
}

fn hash_optional_mask(mask: Option<&Arc<VoxelMask>>, h: &mut impl Hasher) {
    match mask {
        Some(m) => {
            1u8.hash(h);
            m.dims.hash(h);
            for c in m.voxel_to_ras.to_cols_array() {
                c.to_bits().hash(h);
            }
            m.data.len().hash(h);
            let stride = (m.data.len() / 256).max(1);
            for i in (0..m.data.len()).step_by(stride) {
                m.data[i].hash(h);
            }
        }
        None => 0u8.hash(h),
    }
}

impl WorkflowOp for MpatTransitionsOp {
    fn tag(&self) -> &'static str {
        "mpat_transitions"
    }

    fn title(&self) -> &'static str {
        "MPAT Transition Probabilities"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::OdfField, PortKind::VoxelMask, PortKind::VoxelMask]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::MpatTransitions,
            PortKind::Volume, // 1: CoDI (Double-ODF)
            PortKind::Volume, // 2: CoDI (Single-ODF)
            PortKind::Volume, // 3: CoAsy (Asymmetry)
            PortKind::Volume, // 4: P(Terminate Double-ODF)
            PortKind::Volume, // 5: P(Terminate Single-ODF)
            PortKind::Volume, // 6: Total ODF Amplitude
            PortKind::Volume, // 7: KL Divergence (Single-ODF)
            PortKind::Volume, // 8: Chain Dead-State Prob
        ]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }

    fn describe(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Builds MPAT's absorbing Markov chain (voxel-to-26-neighbour transition \
             probabilities) by running `mpat.trxviz_bridge` in a Python environment with \
             MPAT installed. Reads either a transitions ODX written by \
             `mpat --save-transitions odx`, which loads in seconds, or a plain fODF ODX, which \
             also needs tissue maps (a pyAFQ output folder, MPAT's `desc-mpat_dseg`, or a \
             wired Transient WM Mask). The chain is cached as `mpat_transitions_<hash>.npz` \
             and reused while the inputs and settings stay the same. See the \
             [MPAT guide](../../mpat.md).",
        )
    }

    fn citation_keys(&self) -> &'static [&'static str] {
        &["mpat"]
    }

    fn runs_original_software(&self) -> bool {
        true
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "Analytic voxel-to-voxel transition probabilities and the 26-direction-class \
             absorbing Markov chain were constructed with MITTENS and MPAT [@mpat] using a \
             {angle:.0}° turning cone, {step:.2}-voxel step size, {mode} look-ahead, \
             FOD threshold {thresh:.2}, FOD exponent {power:.2}, stopping mass τ={tau:.4}, \
             and curvature prior σ={curv:.1}°.",
            angle = self.angle_deg,
            step = self.step_vox,
            mode = if self.double_odf {
                "double-ODF"
            } else {
                "single-ODF"
            },
            thresh = self.fod_thresh,
            power = self.fod_power,
            tau = self.stop_mass_tau,
            curv = self.curv_sigma_deg,
        )))
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let odf_path = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::OdfField(f) => ctx
                    .odx_assets
                    .get(&f.source_id)
                    .map(|loaded| loaded.path.clone()),
                _ => None,
            },
            None => None,
        };

        if self.working_dir.trim().is_empty() && odf_path.is_none() {
            ctx.node_state.summary =
                "Pick a pyAFQ/MPAT directory or wire an ODF Field input.".into();
            return Ok(empty_outputs(false));
        }

        let transient_mask = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => None,
            },
            None => None,
        };

        let absorb_mask = match ctx.inputs.get(2).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => None,
            },
            None => None,
        };

        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.working_dir.trim().hash(&mut h);
            self.python_path.trim().hash(&mut h);
            odf_path.hash(&mut h);
            hash_optional_mask(transient_mask.as_ref(), &mut h);
            hash_optional_mask(absorb_mask.as_ref(), &mut h);
            self.preset.hash(&mut h);
            self.angle_deg.to_bits().hash(&mut h);
            self.step_vox.to_bits().hash(&mut h);
            self.double_odf.hash(&mut h);
            self.fod_thresh.to_bits().hash(&mut h);
            self.fod_power.to_bits().hash(&mut h);
            self.stop_mass_tau.to_bits().hash(&mut h);
            self.curv_sigma_deg.to_bits().hash(&mut h);
            self.probe_radius_vox.to_bits().hash(&mut h);
            self.max_length_mm.to_bits().hash(&mut h);
            self.wm_threshold.to_bits().hash(&mut h);
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
            .mpat_transitions_cache
            .get(&ctx.node.uuid)
            .map(|c| c.fingerprint == fingerprint)
            .unwrap_or(false);

        let stale = !cached_matches || upstream_stale;

        if stale {
            ctx.scene_plan
                .mpat_transitions_jobs
                .push(MpatTransitionsJob {
                    node_uuid: ctx.node.uuid,
                    fingerprint,
                    label: ctx.node.label.clone(),
                    working_dir: self.working_dir.clone(),
                    python_path: self.python_path.clone(),
                    odx_path: odf_path,
                    transient_mask,
                    absorb_mask,
                    preset: self.preset,
                    angle_deg: self.angle_deg,
                    step_vox: self.step_vox,
                    double_odf: self.double_odf,
                    fod_thresh: self.fod_thresh,
                    fod_power: self.fod_power,
                    stop_mass_tau: self.stop_mass_tau,
                    curv_sigma_deg: self.curv_sigma_deg,
                    probe_radius_vox: self.probe_radius_vox,
                    max_length_mm: self.max_length_mm,
                    wm_threshold: self.wm_threshold,
                });
        }

        let Some(cached) = ctx
            .execution_cache
            .mpat_transitions_cache
            .get(&ctx.node.uuid)
        else {
            ctx.node_state.summary =
                "Transitions not built yet — click Build MPAT Transitions.".into();
            return Ok(empty_outputs(true));
        };

        ctx.node_state.summary = if stale {
            format!("{} (stale)", cached.output.summary)
        } else {
            cached.output.summary.clone()
        };

        let out = &cached.output;
        let vol_eval = |arc: &Arc<VolumeScalars>| EvaluatedValue {
            value: WorkflowValue::Volume(VolumeBacking::from_scalars((**arc).clone())),
            stale,
        };

        Ok(vec![
            EvaluatedValue {
                value: WorkflowValue::MpatTransitions(out.handle.clone()),
                stale,
            },
            vol_eval(&out.codi_double),
            vol_eval(&out.codi_single),
            vol_eval(&out.coasy),
            vol_eval(&out.p_terminate_double),
            vol_eval(&out.p_terminate_single),
            vol_eval(&out.total_amplitude),
            vol_eval(&out.kl_single),
            vol_eval(&out.chain_dead),
        ])
    }
}

fn empty_outputs(stale: bool) -> Vec<EvaluatedValue> {
    let empty_handle = Arc::new(MpatTransitionsHandle {
        fingerprint: 0,
        state_path: PathBuf::new(),
        working_dir: None,
        dims: [0, 0, 0],
        voxel_to_ras: glam::Mat4::IDENTITY,
        summary: String::new(),
    });
    let empty_vol = VolumeScalars {
        dims: [1, 1, 1],
        voxel_to_ras: glam::Mat4::IDENTITY,
        values: vec![0.0],
        kind: ScalarKind::Continuous,
        metadata: ScalarMetadata {
            map_name: String::new(),
            suggested_range: Some((0.0, 1.0)),
            series_index: None,
            series_value: None,
            label_table: Vec::new(),
        },
    };
    let mut outputs = Vec::with_capacity(9);
    outputs.push(EvaluatedValue {
        value: WorkflowValue::MpatTransitions(empty_handle),
        stale,
    });
    for _ in 0..8 {
        outputs.push(EvaluatedValue {
            value: WorkflowValue::Volume(VolumeBacking::from_scalars(empty_vol.clone())),
            stale,
        });
    }
    outputs
}

impl From<MpatTransitionsOp> for WorkflowNodeKind {
    fn from(op: MpatTransitionsOp) -> Self {
        Self::MpatTransitions {
            working_dir: op.working_dir,
            python_path: op.python_path,
            preset: op.preset,
            angle_deg: op.angle_deg,
            step_vox: op.step_vox,
            double_odf: op.double_odf,
            fod_thresh: op.fod_thresh,
            fod_power: op.fod_power,
            stop_mass_tau: op.stop_mass_tau,
            curv_sigma_deg: op.curv_sigma_deg,
            probe_radius_vox: op.probe_radius_vox,
            max_length_mm: op.max_length_mm,
            wm_threshold: op.wm_threshold,
        }
    }
}

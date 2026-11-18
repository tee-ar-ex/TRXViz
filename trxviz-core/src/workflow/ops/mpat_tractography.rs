//! `MpatTractographyOp` — solves the MPAT absorbing Markov chain for a tractography plan
//! (carrying pyAFQ ROIs plus arbitrary user-wired Waypoint, Avoid, Endpoint, and Absorption
//! regions), quantifies analytical SIFT2 AFD and physical TDI, and extracts reactive-flux
//! streamlines and sub-voxel branch arbor ridges.
//!
//! Inputs:
//! - Port 0 (`PortKind::MpatTransitions`): Transition state from `MpatTransitionsOp`.
//! - Port 1 (`PortKind::VoxelMask`): Optional `Seed Mask` (overrides plan's seed mask).
//! - Port 2 (`PortKind::TrackingPlan`): Optional `Tracking Plan` from `PreparePyafqPlanOp`,
//!   `PrepareSimplePlanOp`, or chained `AddRoiOp` / `AddRoaOp` / `AddTermOp` / `AddEndRegionOp` nodes.
//!
//! Outputs (all 8 visible):
//! - Port 0 (`PortKind::Streamline`): `Flux Streamlines & Branch Arbor` (`.trx` with DPV `afd`,
//!   `sift2_afd`, `q_harmonic`, `coherence`, `k_eff`, `flux_magnitude`, and groups
//!   `00_Branch_Arbor_Ridges`, `01_Dense_Trunk_Core`, `02_Primary_Branches`, `03_Fine_Offshoots`, `Branch_*`).
//! - Port 1 (`PortKind::Volume`): `AFD Map (Analytical SIFT2)` (`afd`)
//! - Port 2 (`PortKind::Volume`): `Visitation Density` (`density`)
//! - Port 3 (`PortKind::Volume`): `Physical TDI (mm)` (`tdimm`)
//! - Port 4 (`PortKind::Volume`): `Harmonic Coordinate (q)` (`qharmonic`)
//! - Port 5 (`PortKind::Volume`): `Flux Coherence` (`coherence`)
//! - Port 6 (`PortKind::Volume`): `Branching K_eff` (`keff`)
//! - Port 7 (`PortKind::Volume`): `Endpoints Map` (`endpoints`)

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::cifti::{ScalarKind, ScalarMetadata, VolumeScalars};
use crate::data::loaded_files::StreamlineBacking;
use crate::data::trx_data::{ColorMode, TrxGpuData};
use crate::error::WorkflowResult;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{
    MpatSolverAccel, MpatTractographyJob, StreamlineDataset, StreamlineFlow, TrackingPlan,
    VoxelMask, WorkflowValue,
};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, VolumeBacking, WorkflowNodeKind, WorkflowOp,
    prime_expensive_record, sync_node_state_from_run_record,
};

/// One-sided bundles stay in their hemisphere, as MPAT's cross_midline=False bundles do:
/// MPAT's exported exclude masks leave the other hemisphere out and expect it added at solve time.
pub fn default_mpat_midline_exclusion() -> bool {
    true
}
pub fn default_mpat_seal_26() -> bool {
    true
}
pub fn default_mpat_solver_tol() -> f32 {
    1e-6
}
pub fn default_mpat_extract_streamlines() -> bool {
    true
}
pub fn default_mpat_extract_arbor() -> bool {
    true
}
pub fn default_mpat_n_streamlines() -> u32 {
    400
}
/// Cortical endpoint branches per bundle end (MPAT's `max_end_branches`).
pub fn default_mpat_n_arbor_branches() -> u32 {
    6
}
pub fn default_mpat_rk4_step_vox() -> f32 {
    0.35
}
pub fn default_mpat_min_len_mm() -> f32 {
    15.0
}
pub fn default_mpat_max_len_mm() -> f32 {
    280.0
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MpatTractographyOp {
    #[serde(default)]
    pub python_path: String,
    #[serde(default = "default_mpat_seal_26")]
    pub seal_26: bool,
    #[serde(default = "default_mpat_midline_exclusion")]
    pub midline_exclusion: bool,
    /// mm past the midline (world x = 0) the bundle may go before the other hemisphere is
    /// excluded (MPAT's `midline_tol`; its corticospinal rule is 10 mm).
    #[serde(default)]
    pub midline_tol_mm: f32,
    #[serde(default = "default_mpat_solver_tol")]
    pub solver_tol: f32,
    #[serde(default)]
    pub solver_accel: MpatSolverAccel,
    #[serde(default = "default_mpat_extract_streamlines")]
    pub extract_streamlines: bool,
    #[serde(default = "default_mpat_extract_arbor")]
    pub extract_arbor: bool,
    #[serde(default = "default_mpat_n_streamlines")]
    pub n_streamlines: u32,
    #[serde(default = "default_mpat_n_arbor_branches")]
    pub n_arbor_branches: u32,
    #[serde(default = "default_mpat_rk4_step_vox")]
    pub step_vox: f32,
    #[serde(default = "default_mpat_min_len_mm")]
    pub min_length_mm: f32,
    #[serde(default = "default_mpat_max_len_mm")]
    pub max_length_mm: f32,
}

impl Default for MpatTractographyOp {
    fn default() -> Self {
        Self {
            python_path: String::new(),
            seal_26: default_mpat_seal_26(),
            midline_exclusion: default_mpat_midline_exclusion(),
            midline_tol_mm: 0.0,
            solver_tol: default_mpat_solver_tol(),
            solver_accel: MpatSolverAccel::MinimalResidual,
            extract_streamlines: default_mpat_extract_streamlines(),
            extract_arbor: default_mpat_extract_arbor(),
            n_streamlines: default_mpat_n_streamlines(),
            n_arbor_branches: default_mpat_n_arbor_branches(),
            step_vox: default_mpat_rk4_step_vox(),
            min_length_mm: default_mpat_min_len_mm(),
            max_length_mm: default_mpat_max_len_mm(),
        }
    }
}

fn hash_mask_opt(mask: Option<&Arc<VoxelMask>>, h: &mut impl Hasher) {
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

fn hash_tracking_plan(plan: Option<&Arc<TrackingPlan>>, h: &mut impl Hasher) {
    match plan {
        Some(p) => {
            1u8.hash(h);
            p.label.hash(h);
            hash_mask_opt(p.seed_mask.as_ref(), h);
            hash_mask_opt(p.limiting_mask.as_ref(), h);
            hash_mask_opt(p.roa_mask.as_ref(), h);
            hash_mask_opt(p.term_mask.as_ref(), h);
            hash_mask_opt(p.no_end_mask.as_ref(), h);
            p.roi_masks.len().hash(h);
            for roi in &p.roi_masks {
                hash_mask_opt(Some(roi), h);
            }
            p.end_masks.len().hash(h);
            for end in &p.end_masks {
                hash_mask_opt(Some(end), h);
            }
            p.min_len_mm.map(f32::to_bits).hash(h);
            p.max_len_mm.map(f32::to_bits).hash(h);
        }
        None => 0u8.hash(h),
    }
}

impl WorkflowOp for MpatTractographyOp {
    fn tag(&self) -> &'static str {
        "mpat_tractography"
    }

    fn title(&self) -> &'static str {
        "MPAT Tractography"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::MpatTransitions,
            PortKind::VoxelMask,
            PortKind::TrackingPlan,
        ]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::Streamline, // 0: Flux Streamlines & Branch Arbor
            PortKind::Volume,     // 1: AFD Map (Analytical SIFT2)
            PortKind::Volume,     // 2: Visitation Density
            PortKind::Volume,     // 3: Physical TDI (mm)
            PortKind::Volume,     // 4: Harmonic Coordinate (q)
            PortKind::Volume,     // 5: Flux Coherence
            PortKind::Volume,     // 6: Branching K_eff
            PortKind::Volume,     // 7: Endpoints Map
        ]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Tractography
    }

    fn describe(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Solves one bundle on the chain from MPAT Transition Probabilities, using the \
             start, end, include and exclude regions of a tracking plan. While the solver \
             runs, the Visitation Density output shows the latest density frame, so the \
             bundle can be watched building up. Outputs the AFD, density, TDI, harmonic \
             coordinate, coherence, branching and endpoint maps, plus flux streamlines and a \
             branch arbor. Midline exclusion (on by default) keeps a one-sided bundle in its \
             hemisphere and is needed for the exclude masks MPAT writes. See the \
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
        let mut text = format!(
            "Bundle-conditioned Markov chain absorbing tractography (MPAT) [@mpat] was solved to \
             tolerance {tol:.1e} ({accel} acceleration).",
            tol = self.solver_tol,
            accel = self.solver_accel.label(),
        );
        if self.extract_streamlines {
            text.push_str(&format!(
                " Up to {n_sl} reactive-flux streamlines were integrated through the bundle's \
                 flux field with RK4 (step {step:.2} voxels, {min:.0}–{max:.0} mm).",
                n_sl = self.n_streamlines,
                step = self.step_vox,
                min = self.min_length_mm,
                max = self.max_length_mm,
            ));
        }
        if self.extract_arbor {
            text.push_str(&format!(
                " A branch arbor with up to {n_arbor} cortical branches per bundle end was \
                 traced along the bundle's cross-section ridges.",
                n_arbor = self.n_arbor_branches,
            ));
        }
        Some(Cow::Owned(text))
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let transitions = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::MpatTransitions(h) => Some(h.clone()),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "MPAT Tractography: port 0 must be MPAT Transitions".into(),
                    ));
                }
            },
            None => None,
        };

        let Some(transitions) = transitions else {
            ctx.node_state.summary = "Wire an MPAT Transition Probabilities node to Port 0.".into();
            return Ok(empty_outputs(false));
        };

        if transitions.state_path.as_os_str().is_empty() {
            ctx.node_state.summary =
                "Waiting for upstream MPAT Transition Probabilities to finish.".into();
            return Ok(empty_outputs(true));
        }

        let seed_mask = match ctx.inputs.get(1).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => Some(m.clone()),
                _ => None,
            },
            None => None,
        };

        let tracking_plan = match ctx.inputs.get(2).and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::TrackingPlan(p) => Some(p.clone()),
                _ => None,
            },
            None => None,
        };

        let effective_min_len = tracking_plan
            .as_ref()
            .and_then(|p| p.min_len_mm)
            .unwrap_or(self.min_length_mm);
        let effective_max_len = tracking_plan
            .as_ref()
            .and_then(|p| p.max_len_mm)
            .unwrap_or(self.max_length_mm);

        let fingerprint = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            transitions.fingerprint.hash(&mut h);
            transitions.state_path.hash(&mut h);
            self.python_path.trim().hash(&mut h);
            hash_mask_opt(seed_mask.as_ref(), &mut h);
            hash_tracking_plan(tracking_plan.as_ref(), &mut h);
            self.seal_26.hash(&mut h);
            self.midline_exclusion.hash(&mut h);
            self.midline_tol_mm.to_bits().hash(&mut h);
            self.solver_tol.to_bits().hash(&mut h);
            self.solver_accel.hash(&mut h);
            self.extract_streamlines.hash(&mut h);
            self.extract_arbor.hash(&mut h);
            self.n_streamlines.hash(&mut h);
            self.n_arbor_branches.hash(&mut h);
            self.step_vox.to_bits().hash(&mut h);
            effective_min_len.to_bits().hash(&mut h);
            effective_max_len.to_bits().hash(&mut h);
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
            .mpat_tractography_cache
            .get(&ctx.node.uuid)
            .map(|c| c.fingerprint == fingerprint)
            .unwrap_or(false);

        let stale = !cached_matches || upstream_stale;

        if stale {
            ctx.scene_plan
                .mpat_tractography_jobs
                .push(MpatTractographyJob {
                    node_uuid: ctx.node.uuid,
                    fingerprint,
                    label: ctx.node.label.clone(),
                    python_path: self.python_path.clone(),
                    transitions,
                    seed_mask,
                    tracking_plan,
                    seal_26: self.seal_26,
                    midline_exclusion: self.midline_exclusion,
                    midline_tol_mm: self.midline_tol_mm,
                    solver_tol: self.solver_tol,
                    solver_accel: self.solver_accel,
                    extract_streamlines: self.extract_streamlines,
                    extract_arbor: self.extract_arbor,
                    n_streamlines: self.n_streamlines,
                    n_arbor_branches: self.n_arbor_branches,
                    step_vox: self.step_vox,
                    min_length_mm: effective_min_len,
                    max_length_mm: effective_max_len,
                });
        }

        let mut outputs = match ctx
            .execution_cache
            .mpat_tractography_cache
            .get(&ctx.node.uuid)
        {
            None => {
                ctx.node_state.summary =
                    "Bundle not solved yet — click Run MPAT Tractography.".into();
                empty_outputs(true)
            }
            Some(cached) => {
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
                vec![
                    EvaluatedValue {
                        value: WorkflowValue::Streamline(out.flow.clone()),
                        stale,
                    },
                    vol_eval(&out.afd),
                    vol_eval(&out.density),
                    vol_eval(&out.tdimm),
                    vol_eval(&out.qharmonic),
                    vol_eval(&out.coherence),
                    vol_eval(&out.keff),
                    vol_eval(&out.endpoints),
                ]
            }
        };

        // While a run with these inputs streams, its latest frame stands in on the density
        // port so the bundle can be watched forming; the solved result above is untouched.
        if let Some((live_fp, frame)) = ctx.execution_cache.mpat_live_volumes.get(&ctx.node.uuid)
            && *live_fp == fingerprint
        {
            outputs[2] = EvaluatedValue {
                value: WorkflowValue::Volume(VolumeBacking::from_scalars((**frame).clone())),
                stale: true,
            };
        }
        Ok(outputs)
    }
}

/// One shared empty dataset for every unsolved MPAT node. Downstream fingerprints hash
/// the dataset's `Arc` pointer, so a fresh empty dataset per evaluation would look like
/// new data each time: the GUI re-queued a tube-geometry job on every evaluation, and the
/// headless driver never settled.
fn empty_dataset() -> Arc<StreamlineDataset> {
    static EMPTY: std::sync::OnceLock<Arc<StreamlineDataset>> = std::sync::OnceLock::new();
    EMPTY
        .get_or_init(|| {
            Arc::new(StreamlineDataset {
                name: "MPAT bundle (not solved)".to_string(),
                gpu_data: Arc::new(TrxGpuData::from_positions_and_offsets(Vec::new(), vec![0])),
                backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
            })
        })
        .clone()
}

fn empty_outputs(stale: bool) -> Vec<EvaluatedValue> {
    let empty_flow = StreamlineFlow {
        dataset: empty_dataset(),
        selected_streamlines: Vec::new(),
        color_mode: ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
    };
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
    let mut outputs = Vec::with_capacity(8);
    outputs.push(EvaluatedValue {
        value: WorkflowValue::Streamline(empty_flow),
        stale,
    });
    for _ in 0..7 {
        outputs.push(EvaluatedValue {
            value: WorkflowValue::Volume(VolumeBacking::from_scalars(empty_vol.clone())),
            stale,
        });
    }
    outputs
}

impl From<MpatTractographyOp> for WorkflowNodeKind {
    fn from(op: MpatTractographyOp) -> Self {
        Self::MpatTractography {
            python_path: op.python_path,
            seal_26: op.seal_26,
            midline_exclusion: op.midline_exclusion,
            midline_tol_mm: op.midline_tol_mm,
            solver_tol: op.solver_tol,
            solver_accel: op.solver_accel,
            extract_streamlines: op.extract_streamlines,
            extract_arbor: op.extract_arbor,
            n_streamlines: op.n_streamlines,
            n_arbor_branches: op.n_arbor_branches,
            step_vox: op.step_vox,
            min_length_mm: op.min_length_mm,
            max_length_mm: op.max_length_mm,
        }
    }
}

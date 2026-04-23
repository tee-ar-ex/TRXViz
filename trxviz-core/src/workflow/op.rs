use std::collections::HashMap;

use crate::data::loaded_files::{FileId, LoadedCifti, LoadedNifti, LoadedOdx, LoadedTrx};
use crate::scene::LoadedGiftiSurface;

use super::{
    EvaluatedValue, LoadedParcellation, NodeEvalState, SaveStreamlinePlan, SceneFramePlan,
    StreamlineDisplayRuntime, WorkflowEvalMode, WorkflowExecutionCache, WorkflowNode,
    WorkflowNodeUuid, WorkflowResult,
};

#[allow(dead_code)]
pub trait WorkflowOp: std::fmt::Debug {
    fn tag(&self) -> &'static str;
    fn title(&self) -> &'static str;
    fn input_ports(&self) -> &'static [super::PortKind];
    fn output_ports(&self) -> &'static [super::PortKind];

    fn default_label(&self) -> String {
        self.title().to_string()
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>>;
}

#[allow(dead_code)]
pub struct EvalCtx<'a, 'assets> {
    pub node: &'a WorkflowNode,
    pub inputs: &'a [Option<EvaluatedValue>],
    pub streamline_assets: &'a HashMap<FileId, &'assets LoadedTrx>,
    pub volume_assets: &'a HashMap<FileId, &'assets LoadedNifti>,
    pub cifti_assets: &'a HashMap<FileId, &'assets LoadedCifti>,
    pub surface_assets: &'a HashMap<FileId, &'assets LoadedGiftiSurface>,
    pub parcellation_assets: &'a HashMap<FileId, &'assets LoadedParcellation>,
    pub odx_assets: &'a HashMap<FileId, &'assets LoadedOdx>,
    pub display_ids: &'a mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    pub next_draw_id: &'a mut FileId,
    pub scene_plan: &'a mut SceneFramePlan,
    pub projection_by_surface: &'a mut HashMap<FileId, crate::data::cifti::SurfaceScalars>,
    pub save_targets: &'a mut HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    pub execution_cache: &'a mut WorkflowExecutionCache,
    pub node_state: &'a mut NodeEvalState,
    /// Interactive = per-frame redraw (do not spend on heavy recompute);
    /// Settled = user requested a run (OK to do heavy work).
    pub eval_mode: WorkflowEvalMode,
}

impl EvalCtx<'_, '_> {
    pub fn upstream_stale(&self) -> bool {
        self.inputs.iter().flatten().any(|value| value.stale)
    }
}

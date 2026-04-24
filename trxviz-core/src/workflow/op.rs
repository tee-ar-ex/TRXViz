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

    /// Pre-evaluation validation. Return any diagnostics the op has
    /// about its own configuration given the current environment —
    /// e.g. "PTT requires a GPU; switch to Probabilistic."
    ///
    /// Default impl returns no diagnostics. The GUI calls the
    /// registry-level `ops::validate(kind, env)` when rendering an
    /// inspector and displays diagnostics inline. Dispatch gates may
    /// also consult this to refuse to queue a node with errors; for
    /// now, the value is purely advisory.
    fn validate(&self, _env: &ValidateCtx) -> Vec<Diagnostic> {
        Vec::new()
    }

    /// Content-addressable hash of this op's output, given its
    /// configuration and upstream inputs. The output of `fingerprint`
    /// is what drives cache invalidation and "is this node stale?"
    /// decisions across the workflow.
    ///
    /// The default impl is correct for any op whose behavior depends
    /// only on its own parameters and its direct inputs: it hashes
    /// `tag()` + the `Debug` repr of `self` + the upstream fingerprints
    /// in port order. Ops that need finer control — e.g. hashing a
    /// mask's voxel bytes rather than its `Arc` pointer — override to
    /// handle that themselves.
    fn fingerprint(&self, ctx: &FingerprintCtx<'_>) -> ContentHash {
        default_fingerprint(self.tag(), self, ctx)
    }
}

/// Shared implementation of the default fingerprint: hash
/// `op.tag()` + `format!("{op:?}")` + each upstream fingerprint in
/// port order. Correct for any op whose Debug repr fully characterizes
/// its output. Factored out so overriding impls can still defer to it
/// for the "fold in upstream + base identity" boilerplate when they
/// only need to replace the body hashing.
pub fn default_fingerprint<O: std::fmt::Debug + ?Sized>(
    tag: &str,
    op: &O,
    ctx: &FingerprintCtx<'_>,
) -> ContentHash {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(tag.as_bytes());
    h.write(format!("{op:?}").as_bytes());
    for up in ctx.upstream {
        match up {
            Some(c) => {
                h.write_u8(1);
                h.write_u64(c.0);
            }
            None => h.write_u8(0),
        }
    }
    ContentHash(h.finish())
}

/// Content-addressable hash of an op's output. Returned by
/// `WorkflowOp::fingerprint`; lives on the plan (PR 2b/3) and keys
/// cache entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// The "no upstream" / "no previous value" sentinel. Ops that
    /// thread `ContentHash` through internal state use this as the
    /// default when no real value is available yet.
    pub const EMPTY: ContentHash = ContentHash(0);

    pub fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

impl From<u64> for ContentHash {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Context passed to `WorkflowOp::fingerprint`. Carries upstream input
/// fingerprints (by port order, matching `input_ports()`) so changes
/// propagate transitively, and the upstream values for ops that need
/// to hash content beyond what the upstream fingerprint already covers.
pub struct FingerprintCtx<'a> {
    /// Fingerprint of each upstream node in port order, or `None`
    /// when the port is unconnected. A change anywhere upstream flips
    /// the corresponding entry here, which flips our own fingerprint.
    pub upstream: &'a [Option<ContentHash>],
    /// Upstream evaluated values in port order. Same slice shape as
    /// `EvalCtx::inputs`. Most ops ignore this — their upstream
    /// fingerprints already fully characterize the inputs — but ops
    /// reading raw mask bytes or other non-hashable-by-upstream data
    /// may dip into it.
    #[allow(dead_code)]
    pub(crate) inputs: &'a [Option<EvaluatedValue>],
}

/// Environment information consumed by `WorkflowOp::validate`. Kept
/// deliberately small; extend as new ops surface new pre-dispatch
/// constraints.
#[derive(Clone, Copy, Debug)]
pub struct ValidateCtx {
    /// `true` when the caller has a wgpu device on hand. GPU-only
    /// direction getters (PTT today) use this to emit an error when
    /// the user has selected them on a headless or fallback build.
    pub gpu_available: bool,
}

/// Severity of a validation diagnostic. Matches typical editor
/// tooling — errors block, warnings inform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    /// Name of the op field the diagnostic is most strongly associated
    /// with (e.g. `"direction_getter"`). The GUI can use this to
    /// highlight the offending widget; `None` means "no specific field."
    pub field: Option<&'static str>,
    pub message: String,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            field: None,
            message: message.into(),
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            field: None,
            message: message.into(),
        }
    }

    pub fn on_field(mut self, field: &'static str) -> Self {
        self.field = Some(field);
        self
    }
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

use crate::units::StreamlineIndex;
use std::hash::{Hash, Hasher};

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct LimitStreamlinesOp {
    pub limit: usize,
    pub randomize: bool,
    pub seed: u64,
}

impl Default for LimitStreamlinesOp {
    fn default() -> Self {
        Self {
            limit: 30_000,
            randomize: false,
            seed: 1,
        }
    }
}

impl WorkflowOp for LimitStreamlinesOp {
    fn tag(&self) -> &'static str {
        "limit_streamlines"
    }

    fn title(&self) -> &'static str {
        "Limit Streamlines"
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
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let mut flow = expect_streamline_input(ctx.inputs, self.title())?;
        if self.randomize {
            flow.selected_streamlines
                .sort_by_key(|index: &StreamlineIndex| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    self.seed.hash(&mut hasher);
                    index.hash(&mut hasher);
                    hasher.finish()
                });
        }
        flow.selected_streamlines.truncate(self.limit);
        Ok(vec![WorkflowValue::Streamline(flow).into()])
    }
}

impl From<LimitStreamlinesOp> for WorkflowNodeKind {
    fn from(op: LimitStreamlinesOp) -> Self {
        Self::LimitStreamlines {
            limit: op.limit,
            randomize: op.randomize,
            seed: op.seed,
        }
    }
}

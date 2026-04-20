use std::hash::{Hash, Hasher};
use crate::units::StreamlineIndex;

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_streamline_input,
};

#[derive(Debug, Clone, Copy)]
pub struct RandomSubsetOp {
    pub limit: usize,
    pub seed: u64,
}

impl Default for RandomSubsetOp {
    fn default() -> Self {
        Self {
            limit: 10_000,
            seed: 1,
        }
    }
}

impl WorkflowOp for RandomSubsetOp {
    fn tag(&self) -> &'static str {
        "random_subset"
    }

    fn title(&self) -> &'static str {
        "Random Subset"
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
        flow.selected_streamlines.sort_by_key(|index: &StreamlineIndex| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.seed.hash(&mut hasher);
            index.hash(&mut hasher);
            hasher.finish()
        });
        flow.selected_streamlines.truncate(self.limit);
        Ok(vec![WorkflowValue::Streamline(flow).into()])
    }
}

impl From<RandomSubsetOp> for WorkflowNodeKind {
    fn from(op: RandomSubsetOp) -> Self {
        Self::RandomSubset {
            limit: op.limit,
            seed: op.seed,
        }
    }
}

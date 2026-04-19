use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::units::StreamlineIndex;

use super::super::{EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_streamline_input};

#[derive(Debug, Clone, Copy)]
pub struct RandomSubsetOp {
    pub limit: usize,
    pub seed: u64,
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
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let mut selected = flow.selected_streamlines.as_ref().clone();
        selected.sort_by_key(|index: &StreamlineIndex| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.seed.hash(&mut hasher);
            index.hash(&mut hasher);
            hasher.finish()
        });
        selected.truncate(self.limit);
        Ok(vec![
            WorkflowValue::Streamline(super::super::StreamlineFlow {
                selected_streamlines: Arc::new(selected),
                ..flow
            })
            .into(),
        ])
    }
}

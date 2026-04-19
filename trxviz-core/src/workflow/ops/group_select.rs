use std::collections::HashSet;
use std::sync::Arc;

use crate::units::StreamlineIndex;

use super::super::{
    EvalCtx, GroupFilter, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input,
};

#[derive(Debug, Clone)]
pub struct GroupSelectOp {
    pub groups: GroupFilter,
}

impl Default for GroupSelectOp {
    fn default() -> Self {
        Self {
            groups: GroupFilter::All,
        }
    }
}

impl WorkflowOp for GroupSelectOp {
    fn tag(&self) -> &'static str {
        "group_select"
    }

    fn title(&self) -> &'static str {
        "Group Select"
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
        match &self.groups {
            GroupFilter::All => Ok(vec![WorkflowValue::Streamline(flow).into()]),
            GroupFilter::None => Ok(vec![
                WorkflowValue::Streamline(super::super::StreamlineFlow {
                    selected_streamlines: Arc::new(Vec::new()),
                    ..flow
                })
                .into(),
            ]),
            GroupFilter::Selected(labels) => {
                if flow.dataset.gpu_data.groups.is_empty() {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Group Select needs streamline input with group memberships, but the input has no groups."
                            .to_string(),
                    ));
                }
                let keep: HashSet<StreamlineIndex> = flow
                    .dataset
                    .gpu_data
                    .groups
                    .iter()
                    .filter(|(name, _)| labels.contains(name))
                    .flat_map(|(_name, members): &(String, Vec<StreamlineIndex>)| {
                        members.iter().copied()
                    })
                    .collect();
                let selected = flow
                    .selected_streamlines
                    .iter()
                    .copied()
                    .filter(|index| keep.contains(index))
                    .collect();
                Ok(vec![
                    WorkflowValue::Streamline(super::super::StreamlineFlow {
                        selected_streamlines: Arc::new(selected),
                        ..flow
                    })
                    .into(),
                ])
            }
        }
    }
}

impl From<GroupSelectOp> for WorkflowNodeKind {
    fn from(op: GroupSelectOp) -> Self {
        Self::GroupSelect { groups: op.groups }
    }
}

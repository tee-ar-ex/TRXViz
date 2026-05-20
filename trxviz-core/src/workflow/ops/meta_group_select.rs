use super::super::{
    EvalCtx, GroupFilter, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
};
use crate::workflow::methods::OpCategory;

/// Broadcasts a single `GroupFilter` over the `GroupSelection` port so
/// multiple downstream `GroupSelectOp`s can share one selection. The
/// streamline input is used only so the inspector can populate group-name
/// autocomplete from a representative TRX; the streamlines themselves are
/// not forwarded.
#[derive(Debug, Clone)]
pub struct MetaGroupSelectOp {
    pub groups: GroupFilter,
}

impl Default for MetaGroupSelectOp {
    fn default() -> Self {
        Self {
            groups: GroupFilter::All,
        }
    }
}

impl WorkflowOp for MetaGroupSelectOp {
    fn tag(&self) -> &'static str {
        "meta_group_select"
    }

    fn title(&self) -> &'static str {
        "Meta Group Select"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::GroupSelection]
    }

    fn category(&self) -> OpCategory {
        OpCategory::StreamlineFilter
    }

    fn evaluate(
        &self,
        _ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        Ok(vec![
            WorkflowValue::GroupSelection(self.groups.clone()).into(),
        ])
    }
}

impl From<MetaGroupSelectOp> for WorkflowNodeKind {
    fn from(op: MetaGroupSelectOp) -> Self {
        Self::MetaGroupSelect { groups: op.groups }
    }
}

use crate::data::cifti::CiftiStructure;
use crate::workflow::methods::OpCategory;

use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_cifti_input,
};

#[derive(Debug, Clone, Copy)]
pub struct CiftiStructureOp {
    pub structure: CiftiStructure,
    pub map_index: usize,
}

impl WorkflowOp for CiftiStructureOp {
    fn tag(&self) -> &'static str {
        "cifti_structure"
    }

    fn title(&self) -> &'static str {
        match self.structure {
            CiftiStructure::CortexLeft => "CIFTI Left Cortex",
            CiftiStructure::CortexRight => "CIFTI Right Cortex",
            CiftiStructure::Subcortical => "CIFTI Subcortex",
        }
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Cifti]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        match self.structure {
            CiftiStructure::Subcortical => &[PortKind::VolumeScalars],
            _ => &[PortKind::SurfaceScalars],
        }
    }

    fn category(&self) -> OpCategory {
        OpCategory::StreamlineFilter
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let cifti_id = expect_cifti_input(ctx.inputs, self.title())?;
        let cifti = ctx.cifti_assets.get(&cifti_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing CIFTI asset {cifti_id}"))
        })?;
        match self.structure {
            CiftiStructure::CortexLeft => cifti
                .data
                .left_scalars
                .get(self.map_index)
                .cloned()
                .flatten()
                .map(|value| WorkflowValue::SurfaceScalars(value).into())
                .ok_or_else(|| {
                    crate::error::WorkflowError::Evaluation(format!(
                        "CIFTI left cortex map {} is unavailable",
                        self.map_index + 1
                    ))
                })
                .map(|v: super::super::EvaluatedValue| vec![v]),
            CiftiStructure::CortexRight => cifti
                .data
                .right_scalars
                .get(self.map_index)
                .cloned()
                .flatten()
                .map(|value| WorkflowValue::SurfaceScalars(value).into())
                .ok_or_else(|| {
                    crate::error::WorkflowError::Evaluation(format!(
                        "CIFTI right cortex map {} is unavailable",
                        self.map_index + 1
                    ))
                })
                .map(|v: super::super::EvaluatedValue| vec![v]),
            CiftiStructure::Subcortical => cifti
                .data
                .subcortical_scalars
                .get(self.map_index)
                .cloned()
                .flatten()
                .map(|value| WorkflowValue::VolumeScalars(value).into())
                .ok_or_else(|| {
                    crate::error::WorkflowError::Evaluation(format!(
                        "CIFTI subcortical map {} is unavailable",
                        self.map_index + 1
                    ))
                })
                .map(|v: super::super::EvaluatedValue| vec![v]),
        }
    }
}

impl From<CiftiStructureOp> for WorkflowNodeKind {
    fn from(op: CiftiStructureOp) -> Self {
        Self::CiftiStructure {
            structure: op.structure,
            map_index: op.map_index,
        }
    }
}

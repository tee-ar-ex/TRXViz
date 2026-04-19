use std::sync::Arc;

use glam::Vec3;

use super::super::{EvalCtx, PortKind, WorkflowOp, WorkflowValue, expect_streamline_input};
use crate::units::Millimeters;

#[derive(Debug, Clone, Copy)]
pub struct SphereQueryOp {
    pub center: [f32; 3],
    pub radius_mm: Millimeters,
}

impl WorkflowOp for SphereQueryOp {
    fn tag(&self) -> &'static str {
        "sphere_query"
    }

    fn title(&self) -> &'static str {
        "Sphere Query"
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
        let hits = flow.dataset.gpu_data.query_sphere(
            Vec3::new(self.center[0], self.center[1], self.center[2]),
            self.radius_mm,
        );
        let selected = flow
            .selected_streamlines
            .iter()
            .copied()
            .filter(|index| hits.contains(index))
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

use crate::data::trx_data::RenderStyle;
use crate::units::Millimeters;

use super::super::{
    EvalCtx, PortKind, StreamlineDisplayRuntime, StreamlineDrawPlan, WorkflowExecutionStatus,
    WorkflowOp, expect_streamline_input, prime_expensive_record, sync_node_state_from_run_record,
    workflow_streamline_fingerprint,
};

#[derive(Debug, Clone, Copy)]
pub struct StreamlineDisplayOp {
    pub enabled: bool,
    pub render_style: RenderStyle,
    pub tube_radius_mm: Millimeters,
    pub tube_sides: u32,
    pub slab_half_width_mm: Millimeters,
}

impl WorkflowOp for StreamlineDisplayOp {
    fn tag(&self) -> &'static str {
        "streamline_display"
    }

    fn title(&self) -> &'static str {
        "Streamline Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let runtime = ctx.display_ids.entry(ctx.node.uuid).or_insert_with(|| {
            let draw_id = *ctx.next_draw_id;
            *ctx.next_draw_id += 1;
            StreamlineDisplayRuntime {
                draw_id,
                ..Default::default()
            }
        });
        let plan = StreamlineDrawPlan {
            node_uuid: ctx.node.uuid,
            draw_id: runtime.draw_id,
            label: ctx.node.label.clone(),
            visible: self.enabled,
            flow,
            render_style: self.render_style,
            tube_radius_mm: self.tube_radius_mm,
            tube_sides: self.tube_sides,
            slab_half_width_mm: self.slab_half_width_mm,
        };
        ctx.node_state.summary = if self.enabled {
            "Visible".to_string()
        } else {
            "Hidden".to_string()
        };
        if self.render_style == RenderStyle::Tubes {
            let upstream_stale = ctx.upstream_stale();
            let fingerprint = workflow_streamline_fingerprint(&plan);
            let record = ctx
                .execution_cache
                .node_runs
                .entry(ctx.node.uuid)
                .or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(ctx.node_state, record);
            if upstream_stale && matches!(record.status, WorkflowExecutionStatus::Ready) {
                ctx.node_state.execution = Some(WorkflowExecutionStatus::Stale);
            }
        } else {
            ctx.node_state.execution = None;
        }
        ctx.scene_plan.streamline_draws.push(plan);
        Ok(Vec::new())
    }
}

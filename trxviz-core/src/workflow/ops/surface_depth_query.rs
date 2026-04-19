use super::super::{
    CachedSurfaceQuery, EvalCtx, PortKind, SurfaceQueryPlan, WorkflowOp, WorkflowValue,
    expect_streamline_input, expect_surface_input, prime_expensive_record,
    sync_node_state_from_run_record, workflow_surface_query_fingerprint,
};

#[derive(Debug, Clone, Copy)]
pub struct SurfaceDepthQueryOp {
    pub depth_mm: crate::units::Millimeters,
}

impl WorkflowOp for SurfaceDepthQueryOp {
    fn tag(&self) -> &'static str {
        "surface_depth_query"
    }

    fn title(&self) -> &'static str {
        "Surface Depth Query"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Surface]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let surface_id = expect_surface_input(ctx.inputs, self.title())?;
        let fingerprint = workflow_surface_query_fingerprint(&flow, surface_id, self.depth_mm);
        let upstream_stale = ctx.upstream_stale();
        let surface = ctx.surface_assets.get(&surface_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing surface {surface_id}"))
        })?;
        let record = ctx.execution_cache.node_runs.entry(ctx.node.uuid).or_default();
        prime_expensive_record(record, fingerprint);
        ctx.scene_plan.surface_query_plans.push(SurfaceQueryPlan {
            node_uuid: ctx.node.uuid,
            flow,
            surface_id,
            surface: surface.data.clone(),
            depth_mm: self.depth_mm,
        });

        sync_node_state_from_run_record(ctx.node_state, record);
        if let Some(CachedSurfaceQuery { flow }) = ctx.execution_cache.surface_query_cache.get(&ctx.node.uuid) {
            ctx.node_state.summary = format!("{} streamlines", flow.selected_streamlines.len());
            return Ok(vec![super::super::EvaluatedValue {
                value: WorkflowValue::Streamline(flow.clone()),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }]);
        }

        ctx.node_state.summary = ctx
            .node_state
            .execution
            .as_ref()
            .map(|status| status.label())
            .unwrap_or("Run required")
            .to_string();
        Ok(Vec::new())
    }
}

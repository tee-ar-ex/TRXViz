use super::super::{
    CachedSurfaceStreamlineMap, DpsFieldName, EvalCtx, PortKind, SurfaceMapPlan, WorkflowOp,
    WorkflowValue, expect_streamline_input, expect_surface_input, prime_expensive_record,
    summarize_value, sync_node_state_from_run_record, workflow_surface_projection_fingerprint,
};
use crate::units::Millimeters;

#[derive(Debug, Clone)]
pub struct SurfaceProjectionDensityOp {
    pub depth_mm: Millimeters,
}

#[derive(Debug, Clone)]
pub struct SurfaceProjectionMeanDpsOp {
    pub depth_mm: Millimeters,
    pub field: DpsFieldName,
}

fn evaluate_projection(
    ctx: &mut EvalCtx<'_, '_>,
    title: &str,
    depth_mm: Millimeters,
    dps_field: Option<DpsFieldName>,
) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
    let flow = expect_streamline_input(ctx.inputs, title)?;
    let surface_id = expect_surface_input(ctx.inputs, title)?;
    let fingerprint = workflow_surface_projection_fingerprint(
        &flow,
        surface_id,
        depth_mm,
        dps_field.as_ref().map(|field| field.as_str()),
    );
    let upstream_stale = ctx.upstream_stale();
    let surface = ctx.surface_assets.get(&surface_id).ok_or_else(|| {
        crate::error::WorkflowError::Evaluation(format!("Missing surface {surface_id}"))
    })?;
    let record = ctx.execution_cache.node_runs.entry(ctx.node.uuid).or_default();
    prime_expensive_record(record, fingerprint);
    ctx.scene_plan.surface_map_plans.push(SurfaceMapPlan {
        node_uuid: ctx.node.uuid,
        flow,
        surface_id,
        surface: surface.data.clone(),
        depth_mm,
        dps_field: dps_field.clone(),
    });

    sync_node_state_from_run_record(ctx.node_state, record);
    if let Some(CachedSurfaceStreamlineMap { map }) =
        ctx.execution_cache.surface_streamline_map_cache.get(&ctx.node.uuid)
    {
        if let Some(source_surface_id) = map.source_surface_id {
            ctx.projection_by_surface
                .insert(source_surface_id, map.clone());
        }
        ctx.node_state.summary = summarize_value(&WorkflowValue::SurfaceScalars(map.clone()));
        return Ok(vec![super::super::EvaluatedValue {
            value: WorkflowValue::SurfaceScalars(map.clone()),
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

impl WorkflowOp for SurfaceProjectionDensityOp {
    fn tag(&self) -> &'static str {
        "surface_projection_density"
    }

    fn title(&self) -> &'static str {
        "Map Streamlines to Surface"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Surface]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::SurfaceScalars]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        evaluate_projection(ctx, self.title(), self.depth_mm, None)
    }
}

impl WorkflowOp for SurfaceProjectionMeanDpsOp {
    fn tag(&self) -> &'static str {
        "surface_projection_mean_dps"
    }

    fn title(&self) -> &'static str {
        "Map Streamlines to Surface (Mean DPS)"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Surface]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::SurfaceScalars]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        evaluate_projection(ctx, self.title(), self.depth_mm, Some(self.field.clone()))
    }
}

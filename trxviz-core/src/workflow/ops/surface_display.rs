use super::super::{
    EvalCtx, PortKind, SurfaceDisplaySpace, SurfaceDrawPlan, SurfaceOverlayLayerConfig, WorkflowOp,
    WorkflowValue, compose_surface_appearance, expect_surface_appearance_input,
    expect_surface_input, mark_expensive_success, surface_display_model_matrix,
    sync_node_state_from_run_record, workflow_surface_overlay_fingerprint,
};
use crate::data::cifti::SurfaceScalars;
use crate::renderer::mesh_renderer::SurfaceColormap;

#[derive(Debug, Clone)]
pub struct SurfaceOverlayStackOp {
    pub layers: Vec<SurfaceOverlayLayerConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceDisplayOp {
    pub color: [f32; 3],
    pub opacity: f32,
    pub outline_color: [f32; 3],
    pub outline_thickness: f32,
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub gloss: f32,
    pub projection_colormap: SurfaceColormap,
    pub range_min: f32,
    pub range_max: f32,
    pub space: SurfaceDisplaySpace,
}

impl WorkflowOp for SurfaceOverlayStackOp {
    fn tag(&self) -> &'static str {
        "surface_overlay_stack"
    }

    fn title(&self) -> &'static str {
        "Surface Overlay Stack"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::SurfaceAppearance]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let surface_id = expect_surface_input(ctx.inputs, self.title())?;
        let surface = ctx.surface_assets.get(&surface_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing surface {surface_id}"))
        })?;
        let upstream_stale = ctx.upstream_stale();
        let fingerprint =
            workflow_surface_overlay_fingerprint(surface_id, &self.layers, upstream_stale);
        let appearance =
            compose_surface_appearance(surface_id, surface, &self.layers, &ctx.inputs[1..])?;
        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        let active_layers = self.layers.iter().filter(|layer| layer.enabled).count();
        mark_expensive_success(
            record,
            fingerprint,
            format!("{active_layers} active layer(s)"),
        );
        sync_node_state_from_run_record(ctx.node_state, record);
        Ok(vec![super::super::EvaluatedValue {
            value: WorkflowValue::SurfaceAppearance(appearance),
            stale: upstream_stale,
        }])
    }
}

impl WorkflowOp for SurfaceDisplayOp {
    fn tag(&self) -> &'static str {
        "surface_display"
    }

    fn title(&self) -> &'static str {
        "Surface Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::SurfaceAppearance]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let appearance = expect_surface_appearance_input(ctx.inputs, self.title())?;
        let source_id = appearance.source_id;
        let surface = ctx.surface_assets.get(&source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing surface {source_id}"))
        })?;
        let projection = None::<SurfaceScalars>;
        let projection_enabled = self.show_projection_map || projection.is_some();
        let final_range = projection
            .as_ref()
            .and_then(|p| p.metadata.suggested_range)
            .unwrap_or((self.range_min, self.range_max));
        let projection_scalars = projection.as_ref().map(|value| value.values.clone());
        ctx.projection_by_surface
            .extend(
                projection
                    .as_ref()
                    .cloned()
                    .into_iter()
                    .filter_map(|projection| {
                        projection
                            .source_surface_id
                            .map(|surface_id| (surface_id, projection))
                    }),
            );
        let draw = SurfaceDrawPlan {
            node_uuid: ctx.node.uuid,
            source_id,
            structure: appearance.structure,
            color: self.color,
            opacity: self.opacity,
            outline_color: self.outline_color,
            outline_thickness: self.outline_thickness,
            show_projection_map: projection_enabled,
            map_opacity: self.map_opacity,
            map_threshold: self.map_threshold,
            gloss: self.gloss,
            projection_colormap: self.projection_colormap,
            range_min: final_range.0,
            range_max: final_range.1,
            projection_scalars,
            vertex_rgba: appearance.vertex_rgba,
            space: self.space,
            model_matrix: surface_display_model_matrix(surface, appearance.structure, self.space)
                .to_cols_array_2d(),
        };
        match self.space {
            SurfaceDisplaySpace::Anatomical => ctx.scene_plan.surface_draws.push(draw),
            SurfaceDisplaySpace::Stage => ctx.scene_plan.stage_surface_draws.push(draw),
        }
        Ok(Vec::new())
    }
}

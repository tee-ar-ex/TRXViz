use super::super::{
    EvalCtx, OdfGlyphDrawPlan, PortKind, WorkflowNodeKind, WorkflowOp, default_false,
    default_full_opacity, default_odf_glyph_detail, default_odf_glyph_scale, default_true,
    expect_odf_field_input, optional_volume_scalars_input,
};
use super::super::{GlyphColormap, OpacityGate, SizeGate, WorkflowSliceViewKind};

#[derive(Debug, Clone, Copy)]
pub struct OdfGlyphRendererOp {
    pub scale: f32,
    pub subtract_iso: bool,
    pub norm_within_voxel: bool,
    pub opacity: f32,
    pub offset_from_slice: f32,
    pub gloss: f32,
    pub vertex_colormap: GlyphColormap,
    pub slice_axis: WorkflowSliceViewKind,
    pub opacity_gate: OpacityGate,
    pub size_gate: SizeGate,
    pub detail: u32,
    pub visible: bool,
}

impl Default for OdfGlyphRendererOp {
    fn default() -> Self {
        Self {
            scale: default_odf_glyph_scale(),
            subtract_iso: default_true(),
            norm_within_voxel: default_false(),
            opacity: default_full_opacity(),
            offset_from_slice: 0.0,
            gloss: 0.0,
            vertex_colormap: GlyphColormap::default(),
            slice_axis: WorkflowSliceViewKind::Axial,
            opacity_gate: OpacityGate::default(),
            size_gate: SizeGate::default(),
            detail: default_odf_glyph_detail(),
            visible: true,
        }
    }
}

impl WorkflowOp for OdfGlyphRendererOp {
    fn tag(&self) -> &'static str {
        "odf_glyph_renderer"
    }

    fn title(&self) -> &'static str {
        "ODF Glyph Renderer"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::OdfField,
            PortKind::VolumeScalars,
            PortKind::VolumeScalars,
        ]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let field = expect_odf_field_input(ctx.inputs, self.title())?;
        let opacity_scalars = optional_volume_scalars_input(ctx.inputs, 1);
        let size_scalars = optional_volume_scalars_input(ctx.inputs, 2);
        ctx.scene_plan.odf_glyph_draws.push(OdfGlyphDrawPlan {
            node_uuid: ctx.node.uuid,
            field,
            scale: self.scale,
            subtract_iso: self.subtract_iso,
            norm_within_voxel: self.norm_within_voxel,
            opacity: self.opacity,
            offset_from_slice: self.offset_from_slice,
            gloss: self.gloss,
            vertex_colormap: self.vertex_colormap,
            slice_axis: self.slice_axis,
            opacity_gate: self.opacity_gate,
            size_gate: self.size_gate,
            detail: self.detail,
            opacity_scalars,
            size_scalars,
            visible: self.visible,
        });
        Ok(Vec::new())
    }
}

impl From<OdfGlyphRendererOp> for WorkflowNodeKind {
    fn from(op: OdfGlyphRendererOp) -> Self {
        Self::OdfGlyphRenderer {
            scale: op.scale,
            subtract_iso: op.subtract_iso,
            norm_within_voxel: op.norm_within_voxel,
            opacity: op.opacity,
            offset_from_slice: op.offset_from_slice,
            gloss: op.gloss,
            vertex_colormap: op.vertex_colormap,
            slice_axis: op.slice_axis,
            opacity_gate: op.opacity_gate,
            size_gate: op.size_gate,
            detail: op.detail,
            visible: op.visible,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use odx_rs::{DType, OdxBuilder};

    use super::*;
    use crate::data::loaded_files::{LoadedCifti, LoadedNifti, LoadedOdx, LoadedTrx};
    use crate::data::odx_data::{OdfField, OdxScene};
    use crate::scene::LoadedGiftiSurface;
    use crate::workflow::{
        EvalCtx, EvaluatedValue, LoadedParcellation, NodeEvalState, SceneFramePlan,
        StreamlineDisplayRuntime, WorkflowExecutionCache, WorkflowNode, WorkflowNodeUuid,
        WorkflowValue,
    };

    fn build_test_odf_scene() -> Arc<OdxScene> {
        let full = odx_rs::formats::dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = odx_rs::formats::dsistudio_odf8::faces().to_vec();
        let dims = [1, 1, 1];
        let mask = vec![1u8];
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dims,
            mask,
        );
        builder.set_sphere(full.clone(), faces);
        builder.push_voxel_peaks(&[]);
        let values = vec![1.0f32; full.len()];
        builder.set_odf_data(
            "amplitudes",
            bytemuck::cast_slice(&values).to_vec(),
            full.len(),
            DType::Float32,
        );
        Arc::new(OdxScene::from_dataset(builder.finalize().unwrap()).unwrap())
    }

    #[test]
    fn evaluate_propagates_conditioning_flags() {
        let scene = build_test_odf_scene();
        let node = WorkflowNode {
            uuid: WorkflowNodeUuid(7),
            op: OdfGlyphRendererOp {
                subtract_iso: false,
                norm_within_voxel: true,
                ..OdfGlyphRendererOp::default()
            }
            .into(),
            label: "glyphs".into(),
        };
        let field = OdfField {
            source_id: 3,
            scene,
        };
        let input = EvaluatedValue::from(WorkflowValue::OdfField(field));
        let inputs = [Some(input)];
        let mut display_ids: HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime> = HashMap::new();
        let mut next_draw_id = 0usize;
        let mut scene_plan = SceneFramePlan::default();
        let mut projection_by_surface = HashMap::new();
        let mut save_targets = HashMap::new();
        let mut execution_cache = WorkflowExecutionCache::default();
        let mut node_state = NodeEvalState::default();
        let streamline_assets: HashMap<usize, &LoadedTrx> = HashMap::new();
        let volume_assets: HashMap<usize, &LoadedNifti> = HashMap::new();
        let cifti_assets: HashMap<usize, &LoadedCifti> = HashMap::new();
        let surface_assets: HashMap<usize, &LoadedGiftiSurface> = HashMap::new();
        let parcellation_assets: HashMap<usize, &LoadedParcellation> = HashMap::new();
        let odx_assets: HashMap<usize, &LoadedOdx> = HashMap::new();

        let op = OdfGlyphRendererOp {
            subtract_iso: false,
            norm_within_voxel: true,
            ..OdfGlyphRendererOp::default()
        };
        op.evaluate(&mut EvalCtx {
            node: &node,
            inputs: &inputs,
            streamline_assets: &streamline_assets,
            volume_assets: &volume_assets,
            cifti_assets: &cifti_assets,
            surface_assets: &surface_assets,
            parcellation_assets: &parcellation_assets,
            odx_assets: &odx_assets,
            display_ids: &mut display_ids,
            next_draw_id: &mut next_draw_id,
            scene_plan: &mut scene_plan,
            projection_by_surface: &mut projection_by_surface,
            save_targets: &mut save_targets,
            execution_cache: &mut execution_cache,
            node_state: &mut node_state,
            eval_mode: crate::workflow::WorkflowEvalMode::Settled,
        })
        .unwrap();

        assert_eq!(scene_plan.odf_glyph_draws.len(), 1);
        let draw = &scene_plan.odf_glyph_draws[0];
        assert!(!draw.subtract_iso);
        assert!(draw.norm_within_voxel);
    }
}

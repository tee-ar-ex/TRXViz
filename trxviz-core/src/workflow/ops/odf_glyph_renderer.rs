use super::super::{
    EvalCtx, OdfGlyphDrawPlan, PortKind, WorkflowNodeKind, WorkflowOp, default_full_opacity,
    default_odf_glyph_detail, default_odf_glyph_scale, expect_odf_field_input,
    optional_volume_scalars_input,
};
use super::super::{GlyphColormap, OpacityGate, SizeGate, WorkflowSliceViewKind};

#[derive(Debug, Clone, Copy)]
pub struct OdfGlyphRendererOp {
    pub scale: f32,
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

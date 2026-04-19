use crate::units::Millimeters;

use super::super::{
    EvalCtx, FixelDrawPlan, PortKind, WorkflowNodeKind, WorkflowOp, default_fixel_length_scale,
    default_fixel_line_width, default_fixel_slab_thickness_mm, default_full_opacity,
    expect_fixels_input,
};

#[derive(Debug, Clone, Copy)]
pub struct Fixel3DDisplayOp {
    pub line_width: f32,
    pub length_scale: f32,
    pub opacity: f32,
    pub offset_from_slice: f32,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Fixel2DDisplayOp {
    pub line_width: f32,
    pub opacity: f32,
    pub slab_thickness_mm: Millimeters,
    pub length_scale: f32,
    pub visible: bool,
}

impl Default for Fixel3DDisplayOp {
    fn default() -> Self {
        Self {
            line_width: default_fixel_line_width(),
            length_scale: default_fixel_length_scale(),
            opacity: default_full_opacity(),
            offset_from_slice: 0.0,
            visible: true,
        }
    }
}

impl Default for Fixel2DDisplayOp {
    fn default() -> Self {
        Self {
            line_width: default_fixel_line_width(),
            opacity: default_full_opacity(),
            slab_thickness_mm: default_fixel_slab_thickness_mm(),
            length_scale: default_fixel_length_scale(),
            visible: true,
        }
    }
}

impl WorkflowOp for Fixel3DDisplayOp {
    fn tag(&self) -> &'static str {
        "fixel_3d_display"
    }

    fn title(&self) -> &'static str {
        "Fixel 3D Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Fixels]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let field = expect_fixels_input(ctx.inputs, self.title())?;
        let colormap_code = field.colormap_code;
        let scalar_range = field.scalar_range;
        ctx.scene_plan.fixel_3d_draws.push(FixelDrawPlan {
            node_uuid: ctx.node.uuid,
            field,
            line_width: self.line_width,
            length_scale: self.length_scale,
            opacity: self.opacity,
            offset_from_slice: self.offset_from_slice,
            slab_thickness_mm: Millimeters(0.0),
            visible: self.visible,
            colormap_code,
            scalar_range,
        });
        Ok(Vec::new())
    }
}

impl WorkflowOp for Fixel2DDisplayOp {
    fn tag(&self) -> &'static str {
        "fixel_2d_display"
    }

    fn title(&self) -> &'static str {
        "Fixel 2D Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Fixels]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let field = expect_fixels_input(ctx.inputs, self.title())?;
        let colormap_code = field.colormap_code;
        let scalar_range = field.scalar_range;
        ctx.scene_plan.fixel_2d_draws.push(FixelDrawPlan {
            node_uuid: ctx.node.uuid,
            field,
            line_width: self.line_width,
            length_scale: self.length_scale,
            opacity: self.opacity,
            offset_from_slice: 0.0,
            slab_thickness_mm: self.slab_thickness_mm,
            visible: self.visible,
            colormap_code,
            scalar_range,
        });
        Ok(Vec::new())
    }
}

impl From<Fixel3DDisplayOp> for WorkflowNodeKind {
    fn from(op: Fixel3DDisplayOp) -> Self {
        Self::Fixel3DDisplay {
            line_width: op.line_width,
            length_scale: op.length_scale,
            opacity: op.opacity,
            offset_from_slice: op.offset_from_slice,
            visible: op.visible,
        }
    }
}

impl From<Fixel2DDisplayOp> for WorkflowNodeKind {
    fn from(op: Fixel2DDisplayOp) -> Self {
        Self::Fixel2DDisplay {
            line_width: op.line_width,
            opacity: op.opacity,
            slab_thickness_mm: op.slab_thickness_mm,
            length_scale: op.length_scale,
            visible: op.visible,
        }
    }
}

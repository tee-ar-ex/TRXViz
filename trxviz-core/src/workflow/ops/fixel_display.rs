use crate::units::Millimeters;
use crate::workflow::methods::OpCategory;

use super::super::{
    DrawPrimitive, EvalCtx, FixelDrawPlan, FixelView, OpacityGate, PortKind, WorkflowNodeKind,
    WorkflowOp, default_fixel_length_scale, default_fixel_line_width,
    default_fixel_slab_thickness_mm, default_full_opacity, expect_fixels_input,
};

#[derive(Debug, Clone, Copy)]
pub struct Fixel3DDisplayOp {
    pub line_width: f32,
    pub length_scale: f32,
    pub opacity: f32,
    pub offset_from_slice: f32,
    pub visible: bool,
    /// When `true`, the per-fixel opacity gate auto-derives from the
    /// scene's `default_fixel_otsu()` (fixels below the tracking Otsu
    /// band are ghosted at 10 % alpha). Uncheck to use `opacity_gate`
    /// verbatim.
    pub auto_gate_from_otsu: bool,
    pub opacity_gate: OpacityGate,
}

#[derive(Debug, Clone, Copy)]
pub struct Fixel2DDisplayOp {
    pub line_width: f32,
    pub opacity: f32,
    pub slab_thickness_mm: Millimeters,
    pub length_scale: f32,
    pub visible: bool,
    pub auto_gate_from_otsu: bool,
    pub opacity_gate: OpacityGate,
}

impl Default for Fixel3DDisplayOp {
    fn default() -> Self {
        Self {
            line_width: default_fixel_line_width(),
            length_scale: default_fixel_length_scale(),
            opacity: default_full_opacity(),
            offset_from_slice: 0.0,
            visible: true,
            auto_gate_from_otsu: true,
            opacity_gate: OpacityGate::default(),
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
            auto_gate_from_otsu: true,
            opacity_gate: OpacityGate::default(),
        }
    }
}

/// Compute the effective opacity gate for a fixel-family display op.
/// When `auto` is true and the scene exposes a default Otsu, the gate
/// ghosts sub-threshold fixels at `below = 0.1` alpha; above the
/// 0.7·otsu mark fixels are full alpha. When `auto` is false the user's
/// explicit `opacity_gate` is used verbatim.
fn resolve_opacity_gate(
    auto: bool,
    user: OpacityGate,
    scene: &crate::data::odx_data::OdxScene,
) -> OpacityGate {
    if !auto {
        return user;
    }
    let Some(otsu) = scene.default_fixel_otsu() else {
        // No tracking metric available → fall back to pass-through.
        return OpacityGate::default();
    };
    OpacityGate {
        range: (0.5 * otsu.threshold, 0.7 * otsu.threshold),
        below: 0.1,
        above: 1.0,
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

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let field = expect_fixels_input(ctx.inputs, self.title())?;
        let colormap_code = field.colormap_code;
        let scalar_range = field.scalar_range;
        let opacity_gate =
            resolve_opacity_gate(self.auto_gate_from_otsu, self.opacity_gate, &field.scene);
        ctx.scene_plan.draws.push(FixelDrawPlan {
            node_uuid: ctx.node.uuid,
            view: FixelView::ThreeD,
            field,
            line_width: self.line_width,
            length_scale: self.length_scale,
            opacity: self.opacity,
            offset_from_slice: self.offset_from_slice,
            slab_thickness_mm: Millimeters(0.0),
            visible: self.visible,
            colormap_code,
            scalar_range,
            opacity_gate,
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

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let field = expect_fixels_input(ctx.inputs, self.title())?;
        let colormap_code = field.colormap_code;
        let scalar_range = field.scalar_range;
        let opacity_gate =
            resolve_opacity_gate(self.auto_gate_from_otsu, self.opacity_gate, &field.scene);
        ctx.scene_plan.draws.push(FixelDrawPlan {
            node_uuid: ctx.node.uuid,
            view: FixelView::TwoD,
            field,
            line_width: self.line_width,
            length_scale: self.length_scale,
            opacity: self.opacity,
            offset_from_slice: 0.0,
            slab_thickness_mm: self.slab_thickness_mm,
            visible: self.visible,
            colormap_code,
            scalar_range,
            opacity_gate,
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
            auto_gate_from_otsu: op.auto_gate_from_otsu,
            opacity_gate: op.opacity_gate,
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
            auto_gate_from_otsu: op.auto_gate_from_otsu,
            opacity_gate: op.opacity_gate,
        }
    }
}

impl FixelDrawPlan {
    /// Content hash of what this fixel draw uploads as GPU instances: the
    /// field identity, its colormap code, and the scalar name/variant.
    /// Uniforms (line width, opacity, gate) are excluded — they don't
    /// rebuild the instance buffer; `view` is excluded too, since 3D and
    /// 2D draws are diffed against separate upload slots, so an identical
    /// field maps to identical buffers per view. Lives here as an inherent
    /// method (not on `DrawPrimitive`) because fingerprinting is per-op
    /// backend policy, not a universal trait contract — see `draw.rs`.
    pub fn upload_fingerprint(&self) -> u64 {
        use crate::data::odx_data::FixelScalarValues;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.field.source_id.hash(&mut hasher);
        self.field.colormap_code.hash(&mut hasher);
        self.field.scalars.name.hash(&mut hasher);
        match &self.field.scalars.values {
            FixelScalarValues::Rgb(_) => 0u8.hash(&mut hasher),
            FixelScalarValues::Scalar(_) => 1u8.hash(&mut hasher),
        }
        hasher.finish()
    }
}

impl DrawPrimitive for FixelDrawPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn DrawPrimitive> {
        Box::new(self.clone())
    }
}

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::bundle_mesh::{build_voxel_mask_boundary_mesh, build_voxel_mask_mesh};
use crate::error::WorkflowResult;
use crate::units::Millimeters;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{
    CachedVoxelMaskMesh, VoxelMask, VoxelMaskMeshDrawPlan, VoxelMaskRenderStyle,
    VoxelMaskSliceMode, WorkflowValue,
};

use super::super::{EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp};

fn default_color() -> [f32; 4] {
    [0.85, 0.55, 0.25, 1.0]
}
fn default_opacity() -> f32 {
    1.0
}
fn default_smooth_sigma() -> f32 {
    1.0
}
fn default_min_component_volume_mm3() -> Millimeters {
    Millimeters(0.0)
}
fn default_render_style() -> VoxelMaskRenderStyle {
    VoxelMaskRenderStyle::VoxelAccurate
}
fn default_slice_mode() -> VoxelMaskSliceMode {
    VoxelMaskSliceMode::Outline
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VoxelMaskDisplayOp {
    #[serde(default = "default_color")]
    pub color: [f32; 4],
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    #[serde(default = "default_smooth_sigma")]
    pub smooth_sigma: f32,
    #[serde(default = "default_min_component_volume_mm3")]
    pub min_component_volume_mm3: Millimeters,
    #[serde(default = "default_render_style")]
    pub style: VoxelMaskRenderStyle,
    #[serde(default = "default_slice_mode")]
    pub slice_mode: VoxelMaskSliceMode,
}

impl Default for VoxelMaskDisplayOp {
    fn default() -> Self {
        Self {
            color: default_color(),
            opacity: default_opacity(),
            smooth_sigma: default_smooth_sigma(),
            min_component_volume_mm3: default_min_component_volume_mm3(),
            style: default_render_style(),
            slice_mode: default_slice_mode(),
        }
    }
}

impl VoxelMaskDisplayOp {
    /// O(1) fingerprint: identity of the upstream `Arc<VoxelMask>` plus
    /// the display params that affect the cached mesh. The stride-sample
    /// content hash this replaces was costing ~35K bytes per ROI per
    /// frame — pure waste, since `Arc<VoxelMask>` only swaps when the
    /// upstream op rebuilds, and the fingerprint is just there to skip
    /// re-running marching cubes when nothing has changed.
    fn fingerprint(&self, mask: &Arc<VoxelMask>) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (Arc::as_ptr(mask) as usize).hash(&mut h);
        for c in self.color {
            c.to_bits().hash(&mut h);
        }
        self.style.hash(&mut h);
        if matches!(self.style, VoxelMaskRenderStyle::SmoothMesh) {
            self.smooth_sigma.to_bits().hash(&mut h);
            self.min_component_volume_mm3.0.to_bits().hash(&mut h);
        }
        h.finish()
    }
}

impl WorkflowOp for VoxelMaskDisplayOp {
    fn tag(&self) -> &'static str {
        "voxel_mask_display"
    }

    fn title(&self) -> &'static str {
        "Voxel Mask Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::VoxelMask]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn category(&self) -> OpCategory {
        OpCategory::Display
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        // Input 0: VoxelMask
        let mask: Arc<VoxelMask> = match ctx.inputs.first().and_then(|v| v.as_ref()) {
            Some(ev) => match &ev.value {
                WorkflowValue::VoxelMask(m) => m.clone(),
                _ => {
                    return Err(crate::error::WorkflowError::Evaluation(
                        "Voxel Mask Display requires a VoxelMask input".into(),
                    ));
                }
            },
            None => {
                return Err(crate::error::WorkflowError::Evaluation(
                    "Voxel Mask Display requires a VoxelMask input".into(),
                ));
            }
        };

        let fingerprint = self.fingerprint(&mask);
        let cached = ctx
            .execution_cache
            .voxel_mask_mesh_cache
            .get(&ctx.node.uuid)
            .cloned();

        let (mesh_opt, draw_id) = match cached {
            Some(c) if c.fingerprint == fingerprint => (Some(c.mesh), c.draw_id),
            _ => {
                // Mint a draw_id on first sight; stable across re-runs via cache.
                let draw_id = if let Some(c) = &cached {
                    c.draw_id
                } else {
                    let d = *ctx.next_draw_id;
                    *ctx.next_draw_id += 1;
                    d
                };
                let new_mesh = match self.style {
                    VoxelMaskRenderStyle::VoxelAccurate => build_voxel_mask_boundary_mesh(
                        mask.dims,
                        mask.voxel_to_ras,
                        &mask.data,
                        self.color,
                    ),
                    VoxelMaskRenderStyle::SmoothMesh => build_voxel_mask_mesh(
                        mask.dims,
                        mask.voxel_to_ras,
                        &mask.data,
                        self.color,
                        self.smooth_sigma,
                        self.min_component_volume_mm3,
                    ),
                };
                if let Some(mesh) = &new_mesh {
                    log::debug!(
                        "voxel_mask_display '{}': built mesh verts={} tris={}",
                        ctx.node.label,
                        mesh.vertices.len(),
                        mesh.indices.len() / 3
                    );
                    ctx.execution_cache.voxel_mask_mesh_cache.insert(
                        ctx.node.uuid,
                        CachedVoxelMaskMesh {
                            fingerprint,
                            mesh: mesh.clone(),
                            draw_id,
                        },
                    );
                } else {
                    log::debug!(
                        "voxel_mask_display '{}': marching cubes returned no mesh (mask nonzero={})",
                        ctx.node.label,
                        mask.count()
                    );
                    ctx.execution_cache
                        .voxel_mask_mesh_cache
                        .remove(&ctx.node.uuid);
                }
                (new_mesh, draw_id)
            }
        };

        ctx.node_state.summary = match &mesh_opt {
            Some(mesh) => format!(
                "{} vertices, {} triangles",
                mesh.vertices.len(),
                mesh.indices.len() / 3
            ),
            None => "empty mask".to_string(),
        };

        if mesh_opt.is_some() {
            ctx.scene_plan
                .voxel_mask_mesh_draws
                .push(VoxelMaskMeshDrawPlan {
                    node_uuid: ctx.node.uuid,
                    draw_id,
                    label: ctx.node.label.clone(),
                    fingerprint,
                    color: self.color,
                    opacity: self.opacity,
                    style: self.style,
                    slice_mode: self.slice_mode,
                    voxel_mask: Arc::clone(&mask),
                });
        }

        Ok(Vec::new())
    }
}

impl From<VoxelMaskDisplayOp> for WorkflowNodeKind {
    fn from(op: VoxelMaskDisplayOp) -> Self {
        Self::VoxelMaskDisplay {
            color: op.color,
            opacity: op.opacity,
            smooth_sigma: op.smooth_sigma,
            min_component_volume_mm3: op.min_component_volume_mm3,
            style: op.style,
            slice_mode: op.slice_mode,
        }
    }
}

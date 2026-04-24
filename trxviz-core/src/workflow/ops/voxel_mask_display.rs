use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::bundle_mesh::build_voxel_mask_mesh;
use crate::error::WorkflowResult;
use crate::units::Millimeters;
use crate::workflow::types::{
    CachedVoxelMaskMesh, VoxelMask, VoxelMaskMeshDrawPlan, WorkflowValue,
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
}

impl Default for VoxelMaskDisplayOp {
    fn default() -> Self {
        Self {
            color: default_color(),
            opacity: default_opacity(),
            smooth_sigma: default_smooth_sigma(),
            min_component_volume_mm3: default_min_component_volume_mm3(),
        }
    }
}

impl VoxelMaskDisplayOp {
    fn fingerprint(&self, mask: &VoxelMask) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        mask.dims.hash(&mut h);
        for c in mask.voxel_to_ras.to_cols_array() {
            c.to_bits().hash(&mut h);
        }
        mask.data.len().hash(&mut h);
        let stride = (mask.data.len() / 256).max(1);
        for i in (0..mask.data.len()).step_by(stride) {
            mask.data[i].hash(&mut h);
        }
        for c in self.color {
            c.to_bits().hash(&mut h);
        }
        self.smooth_sigma.to_bits().hash(&mut h);
        self.min_component_volume_mm3.0.to_bits().hash(&mut h);
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
                let new_mesh = build_voxel_mask_mesh(
                    mask.dims,
                    mask.voxel_to_ras,
                    &mask.data,
                    self.color,
                    self.smooth_sigma,
                    self.min_component_volume_mm3,
                );
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
        }
    }
}

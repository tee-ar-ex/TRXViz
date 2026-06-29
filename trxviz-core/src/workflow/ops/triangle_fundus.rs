use std::sync::Arc;

use super::super::{
    EvalCtx, PortKind, StreamlineDataset, StreamlineDisplayRuntime, StreamlineDrawPlan,
    StreamlineFlow, WorkflowExecutionStatus, WorkflowNodeKind, WorkflowOp, expect_streamline_input,
    prime_expensive_record, sync_node_state_from_run_record, workflow_streamline_fingerprint,
    workflow_triangle_fundus_fingerprint,
};
use crate::data::fundus_triangle::fit_streamline_triangle;
use crate::data::loaded_files::StreamlineBacking;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::error::WorkflowResult;
use crate::units::{Millimeters, StreamlineIndex};
use crate::workflow::jobs::materialize_flow_gpu;
use crate::workflow::methods::OpCategory;

/// Self-displaying op (like *Streamline Display*): takes a tractogram
/// and renders each input streamline's u-fiber **triangle** (E1→apex,
/// apex→E2, E1→E2) plus a short apex **normal** vector centred on the
/// apex (extends `normal_len_mm/2` each way, so PTT could later seed
/// from the apex in either direction). Triangle edges keep the source
/// streamline's group (palette matches their sheet); every normal
/// goes in one `apex_normals` group, which the palette gives its own
/// distinct colour.
///
/// The derived triangle/normal `StreamlineDataset` is **cached per
/// node**, keyed by a content fingerprint of the input flow + the
/// geometry params. Reusing the same `Arc` across frames keeps the
/// `Arc::as_ptr`-based draw fingerprint stable, so the (expensive)
/// tube-geometry background job runs once instead of every frame —
/// this is what fixes the cylinder-mode GUI lockup while keeping the
/// op a single self-displaying node.
#[derive(Debug, Clone, Copy)]
pub struct TriangleFundusOp {
    pub show_triangles: bool,
    pub show_normals: bool,
    pub normal_len_mm: f32,
    pub stride: usize,
    /// Render the geometry as cylinders (tubes) instead of flat
    /// lines, so the apex normals stand out against the surrounding
    /// streamlines.
    pub render_as_tubes: bool,
    pub tube_radius_mm: f32,
}

impl Default for TriangleFundusOp {
    fn default() -> Self {
        Self {
            show_triangles: true,
            show_normals: true,
            normal_len_mm: 3.0,
            stride: 1,
            render_as_tubes: false,
            tube_radius_mm: 0.4,
        }
    }
}

impl WorkflowOp for TriangleFundusOp {
    fn tag(&self) -> &'static str {
        "triangle_fundus"
    }

    fn title(&self) -> &'static str {
        "Triangle Fundus"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
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
    ) -> WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;

        // Content fingerprint of everything that changes the *built*
        // geometry. Render-only params (tubes/radius) are excluded —
        // they belong to the draw plan, not the dataset.
        let fingerprint = workflow_triangle_fundus_fingerprint(
            &flow,
            self.show_triangles,
            self.show_normals,
            self.normal_len_mm,
            self.stride,
        );

        // Reuse the cached derived dataset when nothing relevant
        // changed. A fresh `Arc::new` per frame would churn the
        // `Arc::as_ptr`-keyed draw fingerprint (`hash_flow`) and
        // re-run the tube-geometry job every frame — the lockup.
        let dataset = match ctx
            .execution_cache
            .triangle_fundus_datasets
            .get(&ctx.node.uuid)
        {
            Some((cached_fp, ds)) if *cached_fp == fingerprint => ds.clone(),
            _ => {
                let src = materialize_flow_gpu(flow.clone());

                // Source streamline index → its (first) group name.
                let mut sl_group: Vec<Option<String>> = vec![None; src.nb_streamlines];
                for (name, members) in &src.groups {
                    for m in members {
                        if let Some(slot) = sl_group.get_mut(m.0 as usize) {
                            if slot.is_none() {
                                *slot = Some(name.clone());
                            }
                        }
                    }
                }

                let stride = self.stride.max(1);
                let half = self.normal_len_mm * 0.5;
                let mut positions: Vec<[f32; 3]> = Vec::new();
                let mut offsets: Vec<u32> = vec![0];
                let mut groups: std::collections::HashMap<String, Vec<StreamlineIndex>> =
                    std::collections::HashMap::new();

                let mut push_seg = |positions: &mut Vec<[f32; 3]>,
                                    offsets: &mut Vec<u32>,
                                    a: glam::Vec3,
                                    b: glam::Vec3,
                                    group: &str,
                                    groups: &mut std::collections::HashMap<
                    String,
                    Vec<StreamlineIndex>,
                >| {
                    let idx = (offsets.len() - 1) as u32;
                    positions.push([a.x, a.y, a.z]);
                    positions.push([b.x, b.y, b.z]);
                    offsets.push(positions.len() as u32);
                    groups
                        .entry(group.to_string())
                        .or_default()
                        .push(StreamlineIndex(idx));
                };

                for s in (0..src.nb_streamlines).step_by(stride) {
                    let start = src.offsets[s] as usize;
                    let end = src.offsets[s + 1] as usize;
                    if end <= start {
                        continue;
                    }
                    let pts: Vec<glam::Vec3> = src.positions[start..end]
                        .iter()
                        .map(|p| glam::Vec3::from(*p))
                        .collect();
                    let Some(tri) = fit_streamline_triangle(&pts) else {
                        continue;
                    };
                    if self.show_triangles {
                        let tri_group = sl_group[s].as_deref().unwrap_or("triangles");
                        push_seg(
                            &mut positions,
                            &mut offsets,
                            tri.e1,
                            tri.apex,
                            tri_group,
                            &mut groups,
                        );
                        push_seg(
                            &mut positions,
                            &mut offsets,
                            tri.apex,
                            tri.e2,
                            tri_group,
                            &mut groups,
                        );
                        push_seg(
                            &mut positions,
                            &mut offsets,
                            tri.e1,
                            tri.e2,
                            tri_group,
                            &mut groups,
                        );
                    }
                    if self.show_normals {
                        // Normal centred on the apex: extends `half` each way.
                        let n = tri.plane_normal * half;
                        push_seg(
                            &mut positions,
                            &mut offsets,
                            tri.apex - n,
                            tri.apex + n,
                            "apex_normals",
                            &mut groups,
                        );
                    }
                }

                let mut gpu = TrxGpuData::from_positions_and_offsets(positions, offsets);
                let mut group_vec: Vec<(String, Vec<StreamlineIndex>)> =
                    groups.into_iter().collect();
                group_vec.sort_by(|a, b| a.0.cmp(&b.0));
                gpu.group_colors = vec![None; group_vec.len()];
                gpu.groups = group_vec;

                let ds = Arc::new(StreamlineDataset {
                    name: "triangle_fundus".to_string(),
                    gpu_data: Arc::new(gpu),
                    backing: StreamlineBacking::Derived(Arc::new(trx_rs::Tractogram::new())),
                });
                ctx.execution_cache
                    .triangle_fundus_datasets
                    .insert(ctx.node.uuid, (fingerprint, ds.clone()));
                ds
            }
        };

        let n_out = dataset.gpu_data.nb_streamlines;
        let derived = StreamlineFlow {
            dataset,
            selected_streamlines: (0..n_out as u32).map(StreamlineIndex).collect(),
            color_mode: ColorMode::Group,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
        };

        // Self-display: push a flat-line (or tube) draw plan, like
        // Streamline Display.
        let runtime = ctx.display_ids.entry(ctx.node.uuid).or_insert_with(|| {
            let draw_id = *ctx.next_draw_id;
            *ctx.next_draw_id += 1;
            StreamlineDisplayRuntime {
                draw_id,
                ..Default::default()
            }
        });
        let any_visible = self.show_triangles || self.show_normals;
        let render_style = if self.render_as_tubes {
            RenderStyle::Tubes
        } else {
            RenderStyle::Flat
        };
        let plan = StreamlineDrawPlan {
            node_uuid: ctx.node.uuid,
            draw_id: runtime.draw_id,
            label: ctx.node.label.clone(),
            visible: any_visible,
            flow: derived,
            render_style,
            tube_radius_mm: Millimeters(self.tube_radius_mm.max(0.01)),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
            opacity: 1.0,
        };
        ctx.node_state.summary = match (self.show_triangles, self.show_normals) {
            (true, true) => "Triangles + normals".to_string(),
            (true, false) => "Triangles".to_string(),
            (false, true) => "Normals".to_string(),
            (false, false) => "Hidden".to_string(),
        };
        // Tube geometry is built by a background job (like Streamline
        // Display); flat lines need no expensive job. The draw
        // fingerprint is stable because `derived.dataset` is a cached
        // `Arc`, so this primes once and does not churn.
        if render_style == RenderStyle::Tubes {
            let upstream_stale = ctx.upstream_stale();
            let fp = workflow_streamline_fingerprint(&plan);
            let record = ctx
                .execution_cache
                .node_runs
                .entry(ctx.node.uuid)
                .or_default();
            prime_expensive_record(record, fp);
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

impl From<TriangleFundusOp> for WorkflowNodeKind {
    fn from(op: TriangleFundusOp) -> Self {
        Self::TriangleFundus {
            show_triangles: op.show_triangles,
            show_normals: op.show_normals,
            normal_len_mm: op.normal_len_mm,
            stride: op.stride,
            render_as_tubes: op.render_as_tubes,
            tube_radius_mm: op.tube_radius_mm,
        }
    }
}

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::bundle_mesh::{
    BundleMesh, BundleMeshColorStrategy, build_bundle_mesh, build_streamtube_bundle_mesh,
};
use crate::data::cifti::{ScalarKind, ScalarMetadata, SurfaceScalars};
use crate::data::orientation_field::{BoundaryContactField, StreamlineSet};
use crate::data::trx_data::{TrxGpuData, build_tube_vertices_from_data, group_name_color};
use crate::units::StreamlineIndex;

use super::evaluate::{materialize_reactive_streamline_flow, robust_range};
use super::*;

pub fn workflow_job_kind_title(kind: WorkflowJobKind) -> &'static str {
    match kind {
        WorkflowJobKind::ReactiveStreamline => "derived streamlines",
        WorkflowJobKind::SurfaceQuery => "surface depth query",
        WorkflowJobKind::SurfaceMap => "surface map",
        WorkflowJobKind::TubeGeometry => "tube geometry",
        WorkflowJobKind::BundleSurface => "bundle surface",
        WorkflowJobKind::BoundaryField => "boundary field",
        WorkflowJobKind::DipyTractography => "dipy tractography",
        WorkflowJobKind::YehTractography => "yeh tractography",
    }
}

pub fn run_workflow_job(payload: WorkflowJobPayload) -> WorkflowResult<WorkflowJobOutput> {
    match payload {
        WorkflowJobPayload::ReactiveStreamline(plan) => Ok(WorkflowJobOutput::ReactiveStreamline(
            materialize_reactive_streamline_flow(&plan)?,
        )),
        WorkflowJobPayload::SurfaceQuery(plan) => {
            let hits = plan
                .flow
                .dataset
                .gpu_data
                .query_near_surface(&plan.surface, plan.depth_mm);
            let selected = plan
                .flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| hits.contains(index))
                .collect();
            Ok(WorkflowJobOutput::SurfaceQuery(StreamlineFlow {
                selected_streamlines: selected,
                ..plan.flow
            }))
        }
        WorkflowJobPayload::SurfaceMap(plan) => {
            let subset = plan
                .flow
                .dataset
                .gpu_data
                .subset_streamlines(&plan.flow.selected_streamlines);
            let dps_storage;
            let dps_values = if let Some(field) = &plan.dps_field {
                dps_storage = subset
                    .dps_data
                    .iter()
                    .find(|(name, _)| name == field.as_str())
                    .map(|(_name, values): &(String, Vec<f32>)| values.clone())
                    .ok_or_else(|| {
                        WorkflowError::Evaluation(format!(
                            "DPS field `{}` is not available",
                            field.as_str()
                        ))
                    })?;
                Some(dps_storage.as_slice())
            } else {
                None
            };
            let (density, projected) = subset.project_selected_to_surface(
                &plan.surface,
                &(0..subset.nb_streamlines as u32)
                    .map(StreamlineIndex)
                    .collect::<Vec<_>>(),
                plan.depth_mm,
                dps_values,
            );
            let scalars: Vec<f32> = if plan.dps_field.is_some() {
                projected
            } else {
                density
            };
            let (range_min, range_max) = robust_range(&scalars);
            Ok(WorkflowJobOutput::SurfaceMap(SurfaceScalars {
                structure: None,
                source_surface_id: Some(plan.surface_id),
                vertex_count: scalars.len(),
                values: scalars,
                kind: ScalarKind::Continuous,
                metadata: ScalarMetadata {
                    map_name: plan
                        .dps_field
                        .as_ref()
                        .map(|field| field.as_str().to_string())
                        .unwrap_or_else(|| "Streamline density".to_string()),
                    suggested_range: Some((range_min, range_max)),
                    series_index: None,
                    series_value: None,
                    label_table: Vec::new(),
                },
            }))
        }
        WorkflowJobPayload::TubeGeometry(draw) => {
            let subset = materialize_flow_gpu(draw.flow);
            let selected = (0..subset.nb_streamlines as u32)
                .map(StreamlineIndex)
                .collect::<Vec<_>>();
            let (positions, colors, offsets) = subset.selected_tube_data(&selected);
            let (vertices, indices) = build_tube_vertices_from_data(
                &positions,
                &colors,
                &offsets,
                draw.tube_radius_mm.0,
                draw.tube_sides,
            );
            Ok(WorkflowJobOutput::TubeGeometry { vertices, indices })
        }
        WorkflowJobPayload::BundleSurface {
            plan,
            color_mode,
            boundary_field,
        } => Ok(WorkflowJobOutput::BundleSurface {
            meshes: build_bundle_surface_meshes_with_color_mode(
                &plan,
                color_mode,
                boundary_field.as_deref(),
            ),
        }),
        WorkflowJobPayload::DipyTractography {
            plan,
            device,
            queue,
        } => {
            let flow = if let (Some(device), Some(queue)) = (device, queue) {
                crate::gpu::dipy::run_gpu_dipy(&plan, &device, &queue)?
            } else {
                super::cpu_dipy::run_cpu_dipy(&plan)?
            };
            Ok(WorkflowJobOutput::DipyTractography { flow })
        }
        WorkflowJobPayload::YehTractography { plan } => {
            let flow = super::cpu_yeh::run_cpu_yeh(&plan)?;
            Ok(WorkflowJobOutput::YehTractography { flow })
        }
        WorkflowJobPayload::BoundaryField { plan } => {
            let subset = materialize_flow_gpu(plan.flow);
            let selected = (0..subset.nb_streamlines as u32)
                .map(StreamlineIndex)
                .collect::<Vec<_>>();
            let (positions, _colors, offsets) = subset.selected_tube_data(&selected);
            if offsets.len() <= 1 {
                return Ok(WorkflowJobOutput::BoundaryField { field: None });
            }
            let params = crate::data::orientation_field::BoundaryGlyphParams {
                voxel_size_mm: plan.voxel_size_mm,
                sphere_lod: plan.sphere_lod,
                normalization: plan.normalization,
                binning_mode: plan.binning_mode,
                ..crate::data::orientation_field::BoundaryGlyphParams::default()
            };
            Ok(WorkflowJobOutput::BoundaryField {
                field: BoundaryContactField::build_from_streamlines(
                    &[StreamlineSet { positions, offsets }],
                    &params,
                )
                .map(Arc::new),
            })
        }
    }
}

pub fn sync_node_state_from_run_record(
    node_state: &mut NodeEvalState,
    record: &ExpensiveNodeRunRecord,
) {
    node_state.execution = Some(record.status.clone());
    node_state.fingerprint = record.current_fingerprint;
    node_state.last_result_summary = record.last_result_summary.clone();
}

pub fn prime_expensive_record(record: &mut ExpensiveNodeRunRecord, fingerprint: u64) {
    record.current_fingerprint = Some(fingerprint);
    if record.last_success_fingerprint == Some(fingerprint) {
        record.status = WorkflowExecutionStatus::Ready;
    } else if record.last_success_fingerprint.is_some() {
        record.status = WorkflowExecutionStatus::Stale;
    } else {
        record.status = WorkflowExecutionStatus::NeverRun;
    }
}

pub fn mark_expensive_success(
    record: &mut ExpensiveNodeRunRecord,
    fingerprint: u64,
    result_summary: String,
) {
    record.current_fingerprint = Some(fingerprint);
    record.last_success_fingerprint = Some(fingerprint);
    record.status = WorkflowExecutionStatus::Ready;
    record.last_result_summary = Some(result_summary);
}

pub fn materialize_flow_gpu(flow: StreamlineFlow) -> TrxGpuData {
    let mut subset = flow
        .dataset
        .gpu_data
        .subset_streamlines(&flow.selected_streamlines);
    let scalar_range = if flow.scalar_auto_range {
        None
    } else {
        Some((flow.scalar_range_min, flow.scalar_range_max))
    };
    subset.recolor(&flow.color_mode, scalar_range, flow.scalar_colormap);
    subset
}

pub fn bundle_surface_component_flows(plan: &BundleSurfacePlan) -> Vec<(String, StreamlineFlow)> {
    if !plan.per_group {
        return vec![(plan.label.clone(), plan.flow.clone())];
    }

    let selected: HashSet<StreamlineIndex> =
        plan.flow.selected_streamlines.iter().copied().collect();
    let mut components = Vec::new();
    for (group_name, members) in plan
        .flow
        .dataset
        .gpu_data
        .groups
        .iter()
        .map(|entry: &(String, Vec<StreamlineIndex>)| entry)
    {
        let group_selected: Vec<StreamlineIndex> = members
            .iter()
            .copied()
            .filter(|member| selected.contains(member))
            .collect();
        if group_selected.is_empty() {
            continue;
        }
        components.push((
            group_name.clone(),
            StreamlineFlow {
                dataset: plan.flow.dataset.clone(),
                selected_streamlines: group_selected,
                color_mode: plan.flow.color_mode.clone(),
                scalar_auto_range: plan.flow.scalar_auto_range,
                scalar_range_min: plan.flow.scalar_range_min,
                scalar_range_max: plan.flow.scalar_range_max,
                scalar_colormap: plan.flow.scalar_colormap,
            },
        ));
    }

    if components.is_empty() {
        vec![(plan.label.clone(), plan.flow.clone())]
    } else {
        components
    }
}

fn build_bundle_surface_meshes_with_color_mode(
    plan: &BundleSurfacePlan,
    color_mode: BundleSurfaceColorMode,
    boundary_field: Option<&BoundaryContactField>,
) -> Vec<(BundleMesh, String)> {
    bundle_surface_component_flows(plan)
        .into_iter()
        .filter_map(|(label, flow)| {
            let subset = materialize_flow_gpu(flow);
            if subset.nb_streamlines == 0 {
                return None;
            }
            let selected = (0..subset.nb_streamlines as u32)
                .map(StreamlineIndex)
                .collect::<Vec<_>>();
            match plan.build_mode {
                BundleSurfaceBuildMode::MarchingCubes => {
                    let (positions, colors) = subset.selected_vertex_data(&selected);
                    let solid_color =
                        bundle_surface_solid_color(&plan.flow, &label, plan.per_group);
                    let (strategy, boundary_field) = match color_mode {
                        BundleSurfaceColorMode::Solid => {
                            (BundleMeshColorStrategy::Constant(solid_color), None)
                        }
                        BundleSurfaceColorMode::BoundaryField => (
                            if boundary_field.is_some() {
                                BundleMeshColorStrategy::BoundaryField
                            } else {
                                BundleMeshColorStrategy::Constant(solid_color)
                            },
                            boundary_field,
                        ),
                        BundleSurfaceColorMode::SourceColors => {
                            (BundleMeshColorStrategy::SampledRgb, None)
                        }
                    };
                    build_bundle_mesh(
                        &positions,
                        &colors,
                        plan.voxel_size_mm.0,
                        plan.threshold,
                        plan.smooth_sigma,
                        plan.min_component_volume_mm3,
                        strategy,
                        boundary_field,
                    )
                    .map(|mesh| (mesh, label))
                }
                BundleSurfaceBuildMode::Streamtubes => {
                    let (positions, colors, offsets) = subset.selected_tube_data(&selected);
                    build_streamtube_bundle_mesh(
                        &positions,
                        &colors,
                        &offsets,
                        plan.tube_radius_mm,
                        plan.tube_sides,
                    )
                    .map(|mesh| (mesh, label))
                }
            }
        })
        .collect()
}

pub fn bundle_surface_solid_color(flow: &StreamlineFlow, label: &str, per_group: bool) -> [f32; 4] {
    if per_group
        && let Some(group_idx) = flow
            .dataset
            .gpu_data
            .groups
            .iter()
            .position(|(name, _)| name == label)
    {
        if let Some(Some(color)) = flow.dataset.gpu_data.group_colors.get(group_idx) {
            return *color;
        }
        if let Some(color) = group_name_color(label) {
            return color;
        }
    }
    pleasant_bundle_color(label)
}

fn pleasant_bundle_color(label: &str) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 8] = [
        [0.165, 0.455, 0.702, 1.0],
        [0.922, 0.467, 0.208, 1.0],
        [0.239, 0.698, 0.412, 1.0],
        [0.753, 0.353, 0.431, 1.0],
        [0.639, 0.471, 0.878, 1.0],
        [0.816, 0.686, 0.267, 1.0],
        [0.247, 0.651, 0.710, 1.0],
        [0.855, 0.400, 0.310, 1.0],
    ];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    PALETTE[(hasher.finish() as usize) % PALETTE.len()]
}

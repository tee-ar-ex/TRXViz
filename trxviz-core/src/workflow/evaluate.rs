use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use petgraph::Directed;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use trx_rs::{
    ConversionOptions, DType, DataArray, Tractogram, remove_duplicates_tractogram, write_tractogram,
};

use crate::data::cifti::{CiftiStructure, ScalarKind, ScalarMetadata, SurfaceScalars, VolumeScalars};
use crate::data::loaded_files::{
    FileId, LoadedCifti, LoadedNifti, LoadedOdx, LoadedTrx, StreamlineBacking,
};
use crate::data::odx_data::{FixelField, FixelScalars, OdfField, OdxCatalog};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::scene::LoadedGiftiSurface;

use super::jobs::{
    mark_expensive_success, prime_expensive_record, sync_node_state_from_run_record,
};
use super::*;

pub fn evaluate_scene_plan(
    document: &WorkflowDocument,
    streamline_assets: &[LoadedTrx],
    volume_assets: &[LoadedNifti],
    cifti_assets: &[LoadedCifti],
    surface_assets: &[LoadedGiftiSurface],
    parcellation_assets: &[LoadedParcellation],
    odx_assets: &[LoadedOdx],
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    execution_cache: &mut WorkflowExecutionCache,
) -> WorkflowRuntime {
    evaluate_scene_plan_with_mode(
        document,
        streamline_assets,
        volume_assets,
        cifti_assets,
        surface_assets,
        parcellation_assets,
        odx_assets,
        display_ids,
        next_draw_id,
        execution_cache,
        WorkflowEvalMode::Interactive,
    )
}

pub fn evaluate_scene_plan_with_mode(
    document: &WorkflowDocument,
    streamline_assets: &[LoadedTrx],
    volume_assets: &[LoadedNifti],
    cifti_assets: &[LoadedCifti],
    surface_assets: &[LoadedGiftiSurface],
    parcellation_assets: &[LoadedParcellation],
    odx_assets: &[LoadedOdx],
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    execution_cache: &mut WorkflowExecutionCache,
    mode: WorkflowEvalMode,
) -> WorkflowRuntime {
    let mut runtime = WorkflowRuntime::default();
    let compiled = compile_graph(document);
    let Ok((order, connections)) = compiled else {
        runtime.graph_error = compiled.err();
        return runtime;
    };

    let streamline_map: HashMap<FileId, &LoadedTrx> = streamline_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let volume_map: HashMap<FileId, &LoadedNifti> = volume_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let cifti_map: HashMap<FileId, &LoadedCifti> =
        cifti_assets.iter().map(|asset| (asset.id, asset)).collect();
    let surface_map: HashMap<FileId, &LoadedGiftiSurface> = surface_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let parcellation_map: HashMap<FileId, &LoadedParcellation> = parcellation_assets
        .iter()
        .map(|asset| (asset.asset.id, asset))
        .collect();
    let odx_map: HashMap<FileId, &LoadedOdx> =
        odx_assets.iter().map(|asset| (asset.id, asset)).collect();

    let mut values = HashMap::<WorkflowNodeUuid, Vec<EvaluatedValue>>::new();
    let mut projection_by_surface = HashMap::<FileId, SurfaceScalars>::new();

    for node_uuid in order {
        let Some(node) = document.graph.get(node_uuid) else {
            continue;
        };
        let input_values: Vec<Option<EvaluatedValue>> = node
            .kind
            .inputs()
            .iter()
            .enumerate()
            .map(|(input_idx, _)| {
                connections.get(&(node.uuid, input_idx)).and_then(|remote| {
                    values
                        .get(&remote.node)
                        .and_then(|vs| vs.get(remote.output).cloned())
                })
            })
            .collect();

        let mut node_state = NodeEvalState {
            summary: node.kind.title().to_string(),
            error: None,
            execution: None,
            fingerprint: None,
            last_result_summary: None,
            available_streamline_groups: Vec::new(),
        };
        let result = evaluate_node(
            node,
            &input_values,
            &streamline_map,
            &volume_map,
            &cifti_map,
            &surface_map,
            &parcellation_map,
            &odx_map,
            display_ids,
            next_draw_id,
            &mut runtime.scene_plan,
            &mut projection_by_surface,
            &mut runtime.save_streamline_targets,
            execution_cache,
            mode,
            &mut node_state,
        );

        match result {
            Ok(outputs) if !outputs.is_empty() => {
                let first = &outputs[0];
                if let WorkflowValue::Streamline(flow) = &first.value {
                    node_state.available_streamline_groups = flow
                        .dataset
                        .gpu_data
                        .groups
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect();
                }
                if node_state.summary == node.kind.title() {
                    node_state.summary = summarize_value(&first.value);
                }
                values.insert(node.uuid, outputs);
            }
            Ok(_) => {
                if node_state.summary == node.kind.title() {
                    node_state.summary = runtime
                        .save_streamline_targets
                        .get(&node.uuid)
                        .map(|target| format!("Ready to save to {}", target.output_path.display()))
                        .unwrap_or_else(|| node.kind.title().to_string());
                }
            }
            Err(error) => {
                node_state.summary = node.kind.title().to_string();
                node_state.error = Some(error);
            }
        }

        runtime.node_state.insert(node.uuid, node_state);
    }

    runtime
        .scene_plan
        .surface_draws
        .iter_mut()
        .for_each(|draw| {
            if draw.show_projection_map {
                if let Some(projection) = projection_by_surface.get(&draw.source_id) {
                    let range = projection.metadata.suggested_range.unwrap_or((0.0, 1.0));
                    draw.range_min = range.0;
                    draw.range_max = range.1;
                    draw.projection_scalars = Some(projection.values.clone());
                }
            }
        });

    runtime
}

fn compile_graph(
    document: &WorkflowDocument,
) -> Result<
    (
        Vec<WorkflowNodeUuid>,
        HashMap<(WorkflowNodeUuid, usize), OutPort>,
    ),
    String,
> {
    let mut graph = StableGraph::<WorkflowNodeUuid, (), Directed>::default();
    let mut uuid_to_graph = HashMap::new();

    for (uuid, _) in document.graph.nodes() {
        let graph_idx = graph.add_node(uuid);
        uuid_to_graph.insert(uuid, graph_idx);
    }

    let mut connections = HashMap::new();
    for wire in document.graph.wires() {
        let Some(from_idx) = uuid_to_graph.get(&wire.from.node).copied() else {
            continue;
        };
        let Some(to_idx) = uuid_to_graph.get(&wire.to.node).copied() else {
            continue;
        };
        graph.add_edge(from_idx, to_idx, ());
        connections.insert((wire.to.node, wire.to.input), wire.from);
    }

    let ordered =
        toposort(&graph, None).map_err(|_| "Workflow graph contains a cycle".to_string())?;
    let order = ordered
        .into_iter()
        .filter_map(|idx| graph.node_weight(idx).copied())
        .collect();

    Ok((order, connections))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_node(
    node: &WorkflowNode,
    inputs: &[Option<EvaluatedValue>],
    streamline_assets: &HashMap<FileId, &LoadedTrx>,
    volume_assets: &HashMap<FileId, &LoadedNifti>,
    cifti_assets: &HashMap<FileId, &LoadedCifti>,
    surface_assets: &HashMap<FileId, &LoadedGiftiSurface>,
    parcellation_assets: &HashMap<FileId, &LoadedParcellation>,
    odx_assets: &HashMap<FileId, &LoadedOdx>,
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    scene_plan: &mut SceneFramePlan,
    projection_by_surface: &mut HashMap<FileId, SurfaceScalars>,
    save_targets: &mut HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    execution_cache: &mut WorkflowExecutionCache,
    _mode: WorkflowEvalMode,
    node_state: &mut NodeEvalState,
) -> Result<Vec<EvaluatedValue>, String> {
    match &node.kind {
        WorkflowNodeKind::StreamlineSource { source_id } => {
            let source = streamline_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing streamline source {source_id}"))?;
            let dataset = Arc::new(StreamlineDataset {
                name: source.name.clone(),
                gpu_data: source.data.clone(),
                backing: source.backing.clone().ok_or_else(|| {
                    format!(
                        "Streamline source {} is missing export backing",
                        source.name
                    )
                })?,
            });
            let selected = (0..source.data.nb_streamlines as u32).collect();
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    dataset,
                    selected_streamlines: Arc::new(selected),
                    color_mode: ColorMode::DirectionRgb,
                    scalar_auto_range: true,
                    scalar_range_min: 0.0,
                    scalar_range_max: 1.0,
                })
                .into(),
            ])
        }
        WorkflowNodeKind::VolumeSource { source_id } => {
            volume_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing volume source {source_id}"))?;
            Ok(vec![WorkflowValue::Volume(*source_id).into()])
        }
        WorkflowNodeKind::CiftiSource { source_id } => {
            cifti_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing CIFTI source {source_id}"))?;
            Ok(vec![WorkflowValue::Cifti(*source_id).into()])
        }
        WorkflowNodeKind::SurfaceSource { source_id } => {
            surface_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing surface source {source_id}"))?;
            Ok(vec![WorkflowValue::Surface(*source_id).into()])
        }
        WorkflowNodeKind::CiftiStructure {
            structure,
            map_index,
        } => {
            let cifti_id = expect_cifti_input(inputs, "CIFTI Structure")?;
            let cifti = cifti_assets
                .get(&cifti_id)
                .ok_or_else(|| format!("Missing CIFTI asset {cifti_id}"))?;
            match structure {
                CiftiStructure::CortexLeft => cifti
                    .data
                    .left_scalars
                    .get(*map_index)
                    .cloned()
                    .flatten()
                    .map(|value| WorkflowValue::SurfaceScalars(value).into())
                    .ok_or_else(|| {
                        format!("CIFTI left cortex map {} is unavailable", map_index + 1)
                    })
                    .map(|v: EvaluatedValue| vec![v]),
                CiftiStructure::CortexRight => cifti
                    .data
                    .right_scalars
                    .get(*map_index)
                    .cloned()
                    .flatten()
                    .map(|value| WorkflowValue::SurfaceScalars(value).into())
                    .ok_or_else(|| {
                        format!("CIFTI right cortex map {} is unavailable", map_index + 1)
                    })
                    .map(|v: EvaluatedValue| vec![v]),
                CiftiStructure::Subcortical => cifti
                    .data
                    .subcortical_scalars
                    .get(*map_index)
                    .cloned()
                    .flatten()
                    .map(|value| WorkflowValue::VolumeScalars(value).into())
                    .ok_or_else(|| {
                        format!("CIFTI subcortical map {} is unavailable", map_index + 1)
                    })
                    .map(|v: EvaluatedValue| vec![v]),
            }
        }
        WorkflowNodeKind::ParcellationSource { source_id } => {
            parcellation_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing parcellation source {source_id}"))?;
            Ok(vec![WorkflowValue::Parcellation(*source_id).into()])
        }
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => {
            let flow = expect_streamline_input(inputs, "Limit Streamlines")?;
            let mut selected = flow.selected_streamlines.as_ref().clone();
            if *randomize {
                selected.sort_by_key(|index| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    index.hash(&mut hasher);
                    hasher.finish()
                });
            }
            selected.truncate(*limit);
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::GroupSelect { groups_csv } => {
            let flow = expect_streamline_input(inputs, "Group Select")?;
            match parse_group_filter(groups_csv) {
                GroupFilter::All => Ok(vec![WorkflowValue::Streamline(flow).into()]),
                GroupFilter::None => Ok(vec![
                    WorkflowValue::Streamline(StreamlineFlow {
                        selected_streamlines: Arc::new(Vec::new()),
                        ..flow
                    })
                    .into(),
                ]),
                GroupFilter::Selected(labels) => {
                    if flow.dataset.gpu_data.groups.is_empty() {
                        return Err(
                            "Group Select needs streamline input with group memberships, but the input has no groups."
                                .to_string(),
                        );
                    }
                    let keep: HashSet<u32> = flow
                        .dataset
                        .gpu_data
                        .groups
                        .iter()
                        .filter(|(name, _)| labels.contains(name))
                        .flat_map(|(_, members)| members.iter().copied())
                        .collect();
                    let selected = flow
                        .selected_streamlines
                        .iter()
                        .copied()
                        .filter(|index| keep.contains(index))
                        .collect();
                    Ok(vec![
                        WorkflowValue::Streamline(StreamlineFlow {
                            selected_streamlines: Arc::new(selected),
                            ..flow
                        })
                        .into(),
                    ])
                }
            }
        }
        WorkflowNodeKind::RandomSubset { limit, seed } => {
            let flow = expect_streamline_input(inputs, "Random Subset")?;
            let mut selected = flow.selected_streamlines.as_ref().clone();
            selected.sort_by_key(|index| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                seed.hash(&mut hasher);
                index.hash(&mut hasher);
                hasher.finish()
            });
            selected.truncate(*limit);
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::SphereQuery { center, radius_mm } => {
            let flow = expect_streamline_input(inputs, "Sphere Query")?;
            let hits = flow
                .dataset
                .gpu_data
                .query_sphere(Vec3::new(center[0], center[1], center[2]), *radius_mm);
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| hits.contains(index))
                .collect();
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => {
            let flow = expect_streamline_input(inputs, "Surface Depth Query")?;
            let surface_id = expect_surface_input(inputs, "Surface Depth Query")?;
            let fingerprint = workflow_surface_query_fingerprint(&flow, surface_id, *depth_mm);
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_query_plans.push(SurfaceQueryPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_query_cache.get(&node.uuid) {
                node_state.summary =
                    format!("{} streamlines", cache.flow.selected_streamlines.len());
                return Ok(vec![EvaluatedValue {
                    value: WorkflowValue::Streamline(cache.flow.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }]);
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(Vec::new())
        }
        WorkflowNodeKind::RemoveDuplicates { params } => {
            let flow = expect_streamline_input(inputs, "Remove Duplicates")?;
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::RemoveDuplicates {
                    params: params.clone(),
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::Merge => {
            let left = expect_streamline_input(inputs, node.kind.title())?;
            let right = match inputs.get(1).cloned().flatten() {
                Some(value) => match value.value {
                    WorkflowValue::Streamline(flow) => flow,
                    _ => {
                        return Err(format!(
                            "{} needs a right streamline input",
                            node.kind.title()
                        ));
                    }
                },
                None => {
                    return Err(format!(
                        "{} needs a right streamline input",
                        node.kind.title()
                    ));
                }
            };
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::Merge,
                left,
                right,
            };
            scene_plan.reactive_streamline_plans.push(plan);
            let plan = scene_plan
                .reactive_streamline_plans
                .last()
                .cloned()
                .expect("just pushed plan");
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::ParcelSelect { labels_csv } => {
            let source_id = expect_parcellation_input(inputs, "Parcel Select")?;
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let labels = resolve_selected_labels(labels_csv, &parcellation.asset.data);
            Ok(vec![
                WorkflowValue::ParcelSelection(ParcelSelection { source_id, labels }).into(),
            ])
        }
        WorkflowNodeKind::ParcelROI => {
            let flow = expect_streamline_input(inputs, "Parcel ROI")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROI")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROI is missing its parcellation".to_string())?;
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::ParcelROI {
                    parcellation: parcellation.asset.data.clone(),
                    labels: parcel_selection.labels,
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::ParcelROA => {
            let flow = expect_streamline_input(inputs, "Parcel ROA")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROA")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROA is missing its parcellation".to_string())?;
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::ParcelROA {
                    parcellation: parcellation.asset.data.clone(),
                    labels: parcel_selection.labels,
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::ParcelEnd { endpoint_count } => {
            let flow = expect_streamline_input(inputs, "Parcel End")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel End")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel End is missing its parcellation".to_string())?;
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::ParcelEnd {
                    parcellation: parcellation.asset.data.clone(),
                    labels: parcel_selection.labels,
                    endpoint_count: *endpoint_count,
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::ParcelLimiting | WorkflowNodeKind::ParcelTerminative => {
            let flow = expect_streamline_input(inputs, node.kind.title())?;
            let parcel_selection = expect_parcel_selection_input(inputs, node.kind.title())?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| format!("{} is missing its parcellation", node.kind.title()))?;
            let keep_inside = matches!(node.kind, WorkflowNodeKind::ParcelLimiting);
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::ParcelCrop {
                    parcellation: parcellation.asset.data.clone(),
                    labels: parcel_selection.labels,
                    keep_inside,
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::AddGroupsFromParcellation => {
            let flow = expect_streamline_input(inputs, "Add Groups From Parcellation")?;
            let source_id = match inputs.get(1).cloned().flatten() {
                Some(value) => match value.value {
                    WorkflowValue::Parcellation(source_id) => source_id,
                    _ => {
                        return Err(
                            "Add Groups From Parcellation needs a parcellation input".to_string()
                        );
                    }
                },
                _ => {
                    return Err(
                        "Add Groups From Parcellation needs a parcellation input".to_string()
                    );
                }
            };
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::AddGroupsFromParcellation {
                    parcellation: parcellation.asset.data.clone(),
                    parcellation_name: parcellation.asset.name.clone(),
                },
                left: flow.clone(),
                right: flow,
            };
            scene_plan.reactive_streamline_plans.push(plan.clone());
            evaluate_derived_streamline_plan(node, plan, inputs, execution_cache, node_state)
        }
        WorkflowNodeKind::ColorByDirection => {
            let flow = expect_streamline_input(inputs, "Color By Direction")?;
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::DirectionRgb,
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::ColorByGroup => {
            let flow = expect_streamline_input(inputs, "Color By Group")?;
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Group,
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::ColorByDPV { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPV")?;
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dpv(field.clone()),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPS")?;
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dps(field.clone()),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::UniformColor { color } => {
            let flow = expect_streamline_input(inputs, "Uniform Color")?;
            Ok(vec![
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Uniform(*color),
                    ..flow
                })
                .into(),
            ])
        }
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
            let flow = expect_streamline_input(inputs, "Map Streamlines to Surface")?;
            let surface_id = expect_surface_input(inputs, "Map Streamlines to Surface")?;
            let fingerprint =
                workflow_surface_projection_fingerprint(&flow, surface_id, *depth_mm, None);
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_map_plans.push(SurfaceMapPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
                dps_field: None,
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_streamline_map_cache.get(&node.uuid) {
                if let Some(surface_id) = cache.map.source_surface_id {
                    projection_by_surface.insert(surface_id, cache.map.clone());
                }
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceScalars(cache.map.clone()));
                return Ok(vec![EvaluatedValue {
                    value: WorkflowValue::SurfaceScalars(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }]);
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(Vec::new())
        }
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
            let flow = expect_streamline_input(inputs, "Map Streamlines to Surface (Mean DPS)")?;
            let surface_id = expect_surface_input(inputs, "Map Streamlines to Surface (Mean DPS)")?;
            let fingerprint =
                workflow_surface_projection_fingerprint(&flow, surface_id, *depth_mm, Some(field));
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_map_plans.push(SurfaceMapPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
                dps_field: Some(field.clone()),
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_streamline_map_cache.get(&node.uuid) {
                if let Some(surface_id) = cache.map.source_surface_id {
                    projection_by_surface.insert(surface_id, cache.map.clone());
                }
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceScalars(cache.map.clone()));
                return Ok(vec![EvaluatedValue {
                    value: WorkflowValue::SurfaceScalars(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }]);
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(Vec::new())
        }
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => {
            let flow = expect_streamline_input(inputs, "Streamline Display")?;
            let runtime = display_ids.entry(node.uuid).or_insert_with(|| {
                let draw_id = *next_draw_id;
                *next_draw_id += 1;
                StreamlineDisplayRuntime {
                    draw_id,
                    ..Default::default()
                }
            });
            let plan = StreamlineDrawPlan {
                node_uuid: node.uuid,
                draw_id: runtime.draw_id,
                label: node.label.clone(),
                visible: *enabled,
                flow,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            };
            node_state.summary = if *enabled {
                "Visible".to_string()
            } else {
                "Hidden".to_string()
            };
            if *render_style == RenderStyle::Tubes {
                let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
                let fingerprint = workflow_streamline_fingerprint(&plan);
                let record = execution_cache.node_runs.entry(node.uuid).or_default();
                prime_expensive_record(record, fingerprint);
                sync_node_state_from_run_record(node_state, record);
                if upstream_stale && matches!(record.status, WorkflowExecutionStatus::Ready) {
                    node_state.execution = Some(WorkflowExecutionStatus::Stale);
                }
            } else {
                node_state.execution = None;
            }
            scene_plan.streamline_draws.push(plan);
            Ok(Vec::new())
        }
        WorkflowNodeKind::BundleSurfaceBuild {
            per_group,
            build_mode,
            voxel_size_mm,
            threshold,
            smooth_sigma,
            min_component_volume_mm3,
            tube_radius_mm,
            tube_sides,
            opacity,
        } => {
            let flow = expect_streamline_input(inputs, "Bundle Surface Build")?;
            let bundle = BundleSurfacePlan {
                build_node_uuid: node.uuid,
                label: node.label.clone(),
                flow,
                per_group: *per_group,
                build_mode: *build_mode,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                opacity: *opacity,
            };
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let fingerprint = workflow_bundle_plan_fingerprint(&bundle);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.bundle_surface_plans.push(bundle.clone());
            Ok(vec![EvaluatedValue {
                value: WorkflowValue::BundleSurface(bundle),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }])
        }
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => {
            let source_id = expect_volume_input(inputs, "Volume Display")?;
            if volume_assets.get(&source_id).is_none() && odx_assets.get(&source_id).is_none() {
                return Err(format!("Missing volume {source_id}"));
            }
            scene_plan.volume_draws.push(VolumeDrawPlan {
                source_id,
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => {
            let scalars = expect_volume_scalars_input(inputs, "Volume Scalars Display")?;
            scene_plan.volume_scalar_draws.push(VolumeScalarDrawPlan {
                dims: scalars.dims,
                voxel_to_ras: scalars.voxel_to_ras.to_cols_array_2d(),
                colormap: *colormap,
                opacity: *opacity,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::SurfaceOverlayStack { layers } => {
            let surface_id = expect_surface_input(inputs, "Surface Overlay Stack")?;
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let upstream_stale = inputs.iter().flatten().any(|v| v.stale);
            let fingerprint =
                workflow_surface_overlay_fingerprint(surface_id, layers, upstream_stale);
            let appearance = compose_surface_appearance(surface_id, surface, layers, &inputs[1..])?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            let active_layers = layers.iter().filter(|l| l.enabled).count();
            mark_expensive_success(
                record,
                fingerprint,
                format!("{active_layers} active layer(s)"),
            );
            sync_node_state_from_run_record(node_state, record);
            Ok(vec![EvaluatedValue {
                value: WorkflowValue::SurfaceAppearance(appearance),
                stale: upstream_stale,
            }])
        }
        WorkflowNodeKind::SurfaceDisplay {
            color,
            opacity,
            outline_color,
            outline_thickness,
            show_projection_map,
            map_opacity,
            map_threshold,
            gloss,
            projection_colormap,
            range_min,
            range_max,
            space,
        } => {
            let appearance = expect_surface_appearance_input(inputs, "Surface Display")?;
            let source_id = appearance.source_id;
            let _surface = surface_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing surface {source_id}"))?;
            let projection = None::<SurfaceScalars>;
            let projection_enabled = *show_projection_map || projection.is_some();
            let final_range = projection
                .as_ref()
                .and_then(|p| p.metadata.suggested_range)
                .unwrap_or((*range_min, *range_max));
            let projection_scalars = projection.as_ref().map(|value| value.values.clone());
            projection_by_surface.extend(projection.as_ref().cloned().into_iter().filter_map(
                |projection| {
                    projection
                        .source_surface_id
                        .map(|surface_id| (surface_id, projection))
                },
            ));
            let draw = SurfaceDrawPlan {
                node_uuid: node.uuid,
                source_id,
                structure: appearance.structure,
                color: *color,
                opacity: *opacity,
                outline_color: *outline_color,
                outline_thickness: *outline_thickness,
                show_projection_map: projection_enabled,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: final_range.0,
                range_max: final_range.1,
                projection_scalars,
                vertex_rgba: appearance.vertex_rgba,
                space: *space,
                model_matrix: surface_display_model_matrix(_surface, appearance.structure, *space)
                    .to_cols_array_2d(),
            };
            match space {
                SurfaceDisplaySpace::Anatomical => scene_plan.surface_draws.push(draw),
                SurfaceDisplaySpace::Stage => scene_plan.stage_surface_draws.push(draw),
            }
            Ok(Vec::new())
        }
        WorkflowNodeKind::ParcellationDisplay {
            labels_csv,
            opacity,
        } => {
            let source_id = expect_parcellation_input(inputs, "Parcellation Display")?;
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let labels = resolve_selected_labels(labels_csv, &parcellation.asset.data);
            scene_plan.parcellation_draws.push(ParcellationDrawPlan {
                source_id,
                labels,
                opacity: *opacity,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => {
            let flow = expect_streamline_input(inputs, "Boundary Field Build")?;
            let plan = BoundaryFieldPlan {
                build_node_uuid: node.uuid,
                label: node.label.clone(),
                flow,
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            };
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.boundary_field_plans.push(plan.clone());
            Ok(vec![EvaluatedValue {
                value: WorkflowValue::BoundaryField(plan),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }])
        }
        WorkflowNodeKind::SaveStreamlines { output_path } => {
            let flow = expect_streamline_input(inputs, "Save Streamlines")?;
            if output_path.trim().is_empty() {
                return Err("Save Streamlines needs an output path".to_string());
            }
            save_targets.insert(
                node.uuid,
                SaveStreamlinePlan {
                    node_uuid: node.uuid,
                    output_path: PathBuf::from(output_path),
                    flow,
                },
            );
            Ok(Vec::new())
        }
        WorkflowNodeKind::BundleSurfaceDisplay {
            color_mode,
            outline_thickness,
        } => {
            let (bundle, stale) = expect_bundle_surface_input(inputs, "Bundle Surface Display")?;
            let boundary_field = inputs
                .get(1)
                .and_then(|value| value.as_ref())
                .map(|value| expect_boundary_field_input(Some(value), "Bundle Surface Display"))
                .transpose()?;
            let runtime = display_ids.entry(node.uuid).or_insert_with(|| {
                let draw_id = *next_draw_id;
                *next_draw_id += 1;
                StreamlineDisplayRuntime {
                    draw_id,
                    ..Default::default()
                }
            });
            let resolved_color_mode =
                if matches!(bundle.build_mode, BundleSurfaceBuildMode::Streamtubes) {
                    BundleSurfaceColorMode::SourceColors
                } else {
                    *color_mode
                };
            let draw = BundleDrawPlan {
                node_uuid: node.uuid,
                build_node_uuid: bundle.build_node_uuid,
                boundary_field_node_uuid: boundary_field
                    .as_ref()
                    .map(|(plan, _)| plan.build_node_uuid),
                draw_id: runtime.draw_id,
                label: bundle.label,
                flow: bundle.flow,
                per_group: bundle.per_group,
                color_mode: resolved_color_mode,
                build_mode: bundle.build_mode,
                voxel_size_mm: bundle.voxel_size_mm,
                threshold: bundle.threshold,
                smooth_sigma: bundle.smooth_sigma,
                min_component_volume_mm3: bundle.min_component_volume_mm3,
                tube_radius_mm: bundle.tube_radius_mm,
                tube_sides: bundle.tube_sides,
                opacity: bundle.opacity,
                outline_thickness: *outline_thickness,
            };
            let boundary_revision = draw.boundary_field_node_uuid.and_then(|uuid| {
                execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            });
            let display_fingerprint = workflow_bundle_display_fingerprint(&draw, boundary_revision);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, display_fingerprint);
            sync_node_state_from_run_record(node_state, record);
            let boundary_stale = boundary_field.as_ref().is_some_and(|(_, stale)| *stale);
            node_state.summary = if stale || boundary_stale {
                format!(
                    "Displaying stale bundle surface ({})",
                    resolved_color_mode.label()
                )
            } else {
                format!(
                    "Displaying bundle surface ({})",
                    resolved_color_mode.label()
                )
            };
            scene_plan.bundle_draws.push(draw);
            Ok(Vec::new())
        }
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => {
            let (plan, stale) = expect_boundary_field_input(
                inputs.first().and_then(|value| value.as_ref()),
                "Boundary Glyph Display",
            )?;
            let draw = BoundaryGlyphDrawPlan {
                node_uuid: node.uuid,
                build_node_uuid: plan.build_node_uuid,
                label: node.label.clone(),
                visible: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            };
            node_state.execution = None;
            node_state.summary = if !enabled {
                "Boundary field hidden".to_string()
            } else if stale {
                "Displaying stale boundary field".to_string()
            } else {
                "Displaying boundary field".to_string()
            };
            scene_plan.boundary_glyph_draws.push(draw);
            Ok(Vec::new())
        }
        WorkflowNodeKind::ParcelSurfaceBuild => {
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel Surface Build")?;
            scene_plan.parcellation_draws.push(ParcellationDrawPlan {
                source_id: parcel_selection.source_id,
                labels: parcel_selection.labels,
                opacity: 0.9,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::OdxSource { source_id } => {
            let asset = odx_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing ODX asset {source_id}"))?;
            let scene = asset.scene.clone();
            let dirs = scene.directions().to_vec();
            let default_scalars = FixelScalars::from_directions(*source_id, &dirs);
            let field = FixelField {
                source_id: *source_id,
                scene: scene.clone(),
                scalars: default_scalars.clone(),
                colormap_code: 0,
                scalar_range: (0.0, 1.0),
            };
            let odf = OdfField {
                source_id: *source_id,
                scene: scene.clone(),
            };
            let catalog = OdxCatalog::from_scene(*source_id, scene);
            Ok(vec![
                WorkflowValue::Fixels(field).into(),
                WorkflowValue::OdfField(odf).into(),
                WorkflowValue::OdxCatalog(catalog).into(),
                WorkflowValue::FixelScalars(default_scalars).into(),
            ])
        }
        WorkflowNodeKind::OdxVolumeSelect { dpv_name } => {
            let catalog = expect_odx_catalog_input(inputs, "ODX Volume Select")?;
            if dpv_name.is_empty() {
                return Err("ODX Volume Select needs a DPV name".to_string());
            }
            let volume = catalog
                .materialize_dpv(dpv_name)
                .map_err(|e| format!("Failed to materialize DPV '{dpv_name}': {e}"))?;
            let volume_scalars =
                volume_scalars_from_nifti_volume(&volume, dpv_name.clone(), catalog.source_id);
            // Stash the materialized volume in the execution cache so headless can pick it up.
            execution_cache.odx_dpv_materializations.insert(
                node.uuid,
                crate::workflow::types::OdxDpvMaterialization {
                    source_id: catalog.source_id,
                    dpv_name: dpv_name.clone(),
                    volume: Arc::new(volume),
                },
            );
            Ok(vec![
                WorkflowValue::Volume(catalog.source_id).into(),
                WorkflowValue::VolumeScalars(volume_scalars).into(),
            ])
        }
        WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => {
            let catalog = expect_odx_catalog_input(inputs, "ODX Fixel Scalar Select")?;
            if dpf_name.is_empty() {
                return Err("ODX Fixel Scalar Select needs a DPF name".to_string());
            }
            let values = catalog
                .scene
                .scalar_dpf_f32(dpf_name)
                .map_err(|e| format!("Failed to load DPF '{dpf_name}': {e}"))?;
            let scalars = FixelScalars::from_scalar(catalog.source_id, dpf_name.clone(), values);
            Ok(vec![WorkflowValue::FixelScalars(scalars).into()])
        }
        WorkflowNodeKind::ColorByFixelScalars {
            colormap,
            range,
            length_scale_by_scalar: _,
        } => {
            let mut field = expect_fixels_input(inputs, "Color By Fixel Scalars")?;
            let scalars = expect_fixel_scalars_input(inputs, "Color By Fixel Scalars")?;
            if scalars.fixel_count != field.scalars.fixel_count {
                return Err(format!(
                    "Fixel count mismatch: scalars have {} fixels, field has {}",
                    scalars.fixel_count, field.scalars.fixel_count
                ));
            }
            field.colormap_code = match colormap {
                SurfaceColormap::BlueWhiteRed => 5,
                SurfaceColormap::Viridis => 3,
                SurfaceColormap::Inferno => 4,
            };
            field.scalar_range = range.unwrap_or(scalars.range);
            field.scalars = scalars.clone();
            Ok(vec![
                WorkflowValue::Fixels(field).into(),
                WorkflowValue::FixelScalars(scalars).into(),
            ])
        }
        WorkflowNodeKind::Fixel3DDisplay {
            line_width,
            length_scale,
            opacity,
            offset_from_slice,
            visible,
        } => {
            let field = expect_fixels_input(inputs, "Fixel 3D Display")?;
            let colormap_code = field.colormap_code;
            let scalar_range = field.scalar_range;
            scene_plan.fixel_3d_draws.push(FixelDrawPlan {
                node_uuid: node.uuid,
                field,
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                slab_thickness_mm: 0.0,
                visible: *visible,
                colormap_code,
                scalar_range,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
        } => {
            let field = expect_fixels_input(inputs, "Fixel 2D Display")?;
            let colormap_code = field.colormap_code;
            let scalar_range = field.scalar_range;
            scene_plan.fixel_2d_draws.push(FixelDrawPlan {
                node_uuid: node.uuid,
                field,
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: 0.0,
                slab_thickness_mm: *slab_thickness_mm,
                visible: *visible,
                colormap_code,
                scalar_range,
            });
            Ok(Vec::new())
        }
        WorkflowNodeKind::OdfGlyphRenderer {
            scale,
            opacity,
            offset_from_slice,
            gloss,
            vertex_colormap,
            slice_axis,
            opacity_gate,
            size_gate,
            detail,
            visible,
        } => {
            let field = expect_odf_field_input(inputs, "ODF Glyph Renderer")?;
            let opacity_scalars = optional_volume_scalars_input(inputs, 1);
            let size_scalars = optional_volume_scalars_input(inputs, 2);
            scene_plan.odf_glyph_draws.push(OdfGlyphDrawPlan {
                node_uuid: node.uuid,
                field,
                scale: *scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                gloss: *gloss,
                vertex_colormap: *vertex_colormap,
                slice_axis: *slice_axis,
                opacity_gate: *opacity_gate,
                size_gate: *size_gate,
                detail: *detail,
                opacity_scalars,
                size_scalars,
                visible: *visible,
            });
            Ok(Vec::new())
        }
    }
}

fn expect_fixels_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<FixelField, String> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::Fixels(field) = &input.value {
            return Ok(field.clone());
        }
    }
    Err(format!("{label} needs a Fixels input"))
}

fn expect_fixel_scalars_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<FixelScalars, String> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::FixelScalars(s) = &input.value {
            return Ok(s.clone());
        }
    }
    Err(format!("{label} needs a FixelScalars input"))
}

fn expect_odf_field_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<OdfField, String> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::OdfField(f) = &input.value {
            return Ok(f.clone());
        }
    }
    Err(format!("{label} needs an OdfField input"))
}

fn expect_odx_catalog_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<OdxCatalog, String> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::OdxCatalog(c) = &input.value {
            return Ok(c.clone());
        }
    }
    Err(format!("{label} needs an OdxCatalog input"))
}

fn optional_volume_scalars_input(
    inputs: &[Option<EvaluatedValue>],
    index: usize,
) -> Option<VolumeScalars> {
    match inputs.get(index).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::VolumeScalars(v),
            ..
        }) => Some(v),
        _ => None,
    }
}

fn volume_scalars_from_nifti_volume(
    volume: &crate::data::nifti_data::NiftiVolume,
    map_name: String,
    _source_id: FileId,
) -> VolumeScalars {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &value in &volume.data {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    let suggested_range = if lo.is_finite() && hi.is_finite() {
        Some((lo, hi))
    } else {
        None
    };
    VolumeScalars {
        dims: volume.dims,
        voxel_to_ras: volume.voxel_to_ras,
        values: volume.data.clone(),
        kind: ScalarKind::Continuous,
        metadata: ScalarMetadata {
            map_name,
            suggested_range,
            series_index: None,
            series_value: None,
            label_table: Vec::new(),
        },
    }
}

fn expect_streamline_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<StreamlineFlow, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            ..
        }) => Ok(flow),
        _ => Err(format!("{label} needs a streamline input")),
    }
}

fn expect_surface_input(inputs: &[Option<EvaluatedValue>], label: &str) -> Result<FileId, String> {
    inputs
        .iter()
        .flatten()
        .find_map(|value| {
            if let WorkflowValue::Surface(surface_id) = &value.value {
                Some(*surface_id)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("{label} needs a surface input"))
}

fn expect_cifti_input(inputs: &[Option<EvaluatedValue>], label: &str) -> Result<FileId, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Cifti(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(format!("{label} needs a CIFTI input")),
    }
}

fn expect_bundle_surface_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<(BundleSurfacePlan, bool), String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::BundleSurface(bundle),
            stale,
        }) => Ok((bundle, stale)),
        Some(_) => Err(format!("{label} needs a bundle surface input")),
        None => Err(format!("{label} is missing an input")),
    }
}

fn expect_volume_scalars_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<VolumeScalars, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::VolumeScalars(value),
            ..
        }) => Ok(value),
        _ => Err(format!("{label} needs volume scalars")),
    }
}

fn expect_surface_appearance_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<SurfaceAppearance, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::SurfaceAppearance(value),
            ..
        }) => Ok(value),
        _ => Err(format!("{label} needs a surface appearance input")),
    }
}

fn expect_boundary_field_input(
    input: Option<&EvaluatedValue>,
    label: &str,
) -> Result<(BoundaryFieldPlan, bool), String> {
    match input {
        Some(EvaluatedValue {
            value: WorkflowValue::BoundaryField(plan),
            stale,
        }) => Ok((plan.clone(), *stale)),
        Some(_) => Err(format!("{label} needs a boundary field input")),
        None => Err(format!("{label} is missing an input")),
    }
}

fn expect_volume_input(inputs: &[Option<EvaluatedValue>], label: &str) -> Result<FileId, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Volume(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(format!("{label} needs a volume input")),
    }
}

fn expect_parcellation_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<FileId, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Parcellation(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(format!("{label} needs a parcellation input")),
    }
}

fn expect_parcel_selection_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<ParcelSelection, String> {
    match inputs.get(1).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::ParcelSelection(selection),
            ..
        }) => Ok(selection),
        _ => Err(format!("{label} needs a parcel selection input")),
    }
}

fn parse_csv_set(csv: &str) -> BTreeSet<String> {
    csv.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

enum GroupFilter {
    All,
    None,
    Selected(BTreeSet<String>),
}

fn parse_group_filter(csv: &str) -> GroupFilter {
    if csv.trim() == "__none__" {
        GroupFilter::None
    } else {
        let labels = parse_csv_set(csv);
        if labels.is_empty() {
            GroupFilter::All
        } else {
            GroupFilter::Selected(labels)
        }
    }
}

fn parse_label_ids(csv: &str) -> BTreeSet<u32> {
    csv.split(',')
        .map(str::trim)
        .filter_map(|value| value.parse::<u32>().ok())
        .collect()
}

fn resolve_selected_labels(csv: &str, parcellation: &ParcellationVolume) -> BTreeSet<u32> {
    let labels = parse_label_ids(csv);
    if !labels.is_empty() {
        return labels;
    }

    let mut resolved = BTreeSet::new();
    for &label in &parcellation.labels {
        if label != 0 {
            resolved.insert(label);
        }
    }
    resolved
}

fn evaluate_derived_streamline_plan(
    node: &WorkflowNode,
    plan: ReactiveStreamlinePlan,
    inputs: &[Option<EvaluatedValue>],
    execution_cache: &mut WorkflowExecutionCache,
    node_state: &mut NodeEvalState,
) -> Result<Vec<EvaluatedValue>, String> {
    let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
    let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
    let record = execution_cache.node_runs.entry(node.uuid).or_default();
    prime_expensive_record(record, fingerprint);
    sync_node_state_from_run_record(node_state, record);
    if let Some(cache) = execution_cache.derived_streamline_cache.get(&node.uuid) {
        node_state.summary = format!("{} streamlines", cache.flow.selected_streamlines.len());
        return Ok(vec![EvaluatedValue {
            value: WorkflowValue::Streamline(cache.flow.clone()),
            stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
        }]);
    }
    node_state.summary = node_state
        .execution
        .as_ref()
        .map(|status| status.label())
        .unwrap_or("Waiting")
        .to_string();
    Ok(Vec::new())
}

fn compose_surface_appearance(
    surface_id: FileId,
    surface: &LoadedGiftiSurface,
    layers: &[SurfaceOverlayLayerConfig],
    scalar_inputs: &[Option<EvaluatedValue>],
) -> Result<SurfaceAppearance, String> {
    let mut vertex_rgba = vec![DEFAULT_SURFACE_BASE_RGBA; surface.data.vertices.len()];
    let mut appearance_structure = None;
    if let Some(base) = layers.first() {
        for color in &mut vertex_rgba {
            *color = base.solid_color;
            color[3] = base.opacity.clamp(0.0, 1.0);
        }
    }
    let mut legend_labels = Vec::new();
    for (layer_index, layer) in layers.iter().enumerate() {
        if !layer.enabled {
            continue;
        }
        let Some(Some(EvaluatedValue {
            value: WorkflowValue::SurfaceScalars(scalars),
            ..
        })) = scalar_inputs.get(layer_index)
        else {
            if !layer.legend_label.trim().is_empty() {
                legend_labels.push(layer.legend_label.clone());
            }
            continue;
        };
        validate_surface_scalars(surface_id, surface, scalars)?;
        if appearance_structure.is_none() {
            appearance_structure = scalars.structure;
        }
        overlay_surface_scalars(&mut vertex_rgba, scalars, layer);
        if !layer.legend_label.trim().is_empty() {
            legend_labels.push(layer.legend_label.clone());
        } else if !scalars.metadata.map_name.trim().is_empty() {
            legend_labels.push(scalars.metadata.map_name.clone());
        }
    }
    Ok(SurfaceAppearance {
        source_id: surface_id,
        structure: appearance_structure,
        vertex_rgba,
        legend_labels,
    })
}

fn surface_display_model_matrix(
    surface: &LoadedGiftiSurface,
    structure: Option<CiftiStructure>,
    space: SurfaceDisplaySpace,
) -> glam::Mat4 {
    if space == SurfaceDisplaySpace::Anatomical {
        return glam::Mat4::IDENTITY;
    }
    let center = (surface.data.bbox_min + surface.data.bbox_max) * 0.5;
    let extents = surface.data.bbox_max - surface.data.bbox_min;
    let span = extents
        .x
        .abs()
        .max(extents.y.abs())
        .max(extents.z.abs())
        .max(1.0);
    let separation = span * 0.8;
    let (x_shift, turn_deg): (f32, f32) = match structure {
        Some(CiftiStructure::CortexLeft) => (separation, -90.0),
        Some(CiftiStructure::CortexRight) => (-separation, 90.0),
        _ => (0.0, 0.0),
    };
    glam::Mat4::from_translation(glam::Vec3::new(x_shift, 0.0, 0.0))
        * glam::Mat4::from_rotation_z(turn_deg.to_radians())
        * glam::Mat4::from_translation(-center)
}

fn validate_surface_scalars(
    surface_id: FileId,
    surface: &LoadedGiftiSurface,
    scalars: &SurfaceScalars,
) -> Result<(), String> {
    if scalars.vertex_count != surface.data.vertices.len() {
        return Err(format!(
            "Surface scalars have {} vertices but surface {} has {}",
            scalars.vertex_count,
            surface_id,
            surface.data.vertices.len()
        ));
    }
    if let Some(bound_surface_id) = scalars.source_surface_id
        && bound_surface_id != surface_id
    {
        return Err(format!(
            "Surface scalars are bound to surface {} and cannot be applied to surface {}",
            bound_surface_id, surface_id
        ));
    }
    Ok(())
}

fn overlay_surface_scalars(
    vertex_rgba: &mut [[f32; 4]],
    scalars: &SurfaceScalars,
    layer: &SurfaceOverlayLayerConfig,
) {
    let (range_min, range_max) = scalars
        .metadata
        .suggested_range
        .unwrap_or((layer.range_min, layer.range_max));
    let denom = (range_max - range_min).max(1e-6);
    for (dst, scalar) in vertex_rgba.iter_mut().zip(scalars.values.iter()) {
        if !scalar.is_finite() {
            continue;
        }
        let src = match scalars.kind {
            ScalarKind::Label if layer.use_label_colors => {
                label_rgba(*scalar as i32, &scalars.metadata)
            }
            _ => {
                if *scalar < layer.threshold_min || *scalar > layer.threshold_max {
                    continue;
                }
                let t = ((*scalar - range_min) / denom).clamp(0.0, 1.0);
                let rgb = surface_colormap_rgb(t, layer.colormap);
                [rgb[0], rgb[1], rgb[2], layer.opacity.clamp(0.0, 1.0)]
            }
        };
        alpha_blend(dst, src);
    }
}

fn label_rgba(label: i32, metadata: &crate::data::cifti::ScalarMetadata) -> [f32; 4] {
    metadata
        .label_table
        .iter()
        .find(|entry| entry.key == label)
        .map(|entry| entry.rgba)
        .unwrap_or([0.0, 0.0, 0.0, 0.0])
}

fn alpha_blend(dst: &mut [f32; 4], src: [f32; 4]) {
    let src_a = src[3].clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return;
    }
    let inv = 1.0 - src_a;
    dst[0] = dst[0] * inv + src[0] * src_a;
    dst[1] = dst[1] * inv + src[1] * src_a;
    dst[2] = dst[2] * inv + src[2] * src_a;
    dst[3] = (dst[3] + src_a).clamp(0.0, 1.0);
}

fn surface_colormap_rgb(t: f32, colormap: SurfaceColormap) -> [f32; 3] {
    match colormap {
        SurfaceColormap::BlueWhiteRed => {
            if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0]
            } else {
                let s = (1.0 - t) * 2.0;
                [1.0, s, s]
            }
        }
        SurfaceColormap::Viridis => {
            let anchors = [
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
            ];
            lerp_colormap(&anchors, t)
        }
        SurfaceColormap::Inferno => {
            let anchors = [
                [0.001, 0.000, 0.014],
                [0.125, 0.047, 0.290],
                [0.302, 0.073, 0.488],
                [0.511, 0.121, 0.561],
                [0.709, 0.212, 0.486],
                [0.865, 0.316, 0.347],
                [0.962, 0.471, 0.212],
                [0.988, 0.683, 0.139],
                [0.978, 0.893, 0.306],
            ];
            lerp_colormap(&anchors, t)
        }
    }
}

fn lerp_colormap(anchors: &[[f32; 3]], t: f32) -> [f32; 3] {
    if anchors.len() == 1 {
        return anchors[0];
    }
    let x = t.clamp(0.0, 1.0) * (anchors.len() as f32 - 1.0);
    let i = x.floor() as usize;
    let j = (i + 1).min(anchors.len() - 1);
    let f = x - i as f32;
    [
        anchors[i][0] * (1.0 - f) + anchors[j][0] * f,
        anchors[i][1] * (1.0 - f) + anchors[j][1] * f,
        anchors[i][2] * (1.0 - f) + anchors[j][2] * f,
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use glam::Vec3;
    use trx_rs::Tractogram;

    use super::*;
    use crate::data::gifti_data::GiftiSurfaceData;
    use crate::data::loaded_files::{LoadedTrx, StreamlineBacking};

    #[test]
    fn group_filter_empty_means_all() {
        assert!(matches!(parse_group_filter(""), GroupFilter::All));
    }

    #[test]
    fn group_filter_none_sentinel_means_none() {
        assert!(matches!(parse_group_filter("__none__"), GroupFilter::None));
    }

    #[test]
    fn group_filter_csv_keeps_selected_labels() {
        match parse_group_filter("CST_left, CST_right") {
            GroupFilter::Selected(labels) => {
                assert!(labels.contains("CST_left"));
                assert!(labels.contains("CST_right"));
            }
            GroupFilter::All | GroupFilter::None => panic!("expected explicit labels"),
        }
    }

    #[test]
    fn interactive_remove_duplicates_defers_work_and_uses_plan() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            .expect("first streamline");
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            .expect("duplicate streamline");
        let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).expect("gpu data"));
        let streamline_assets = vec![LoadedTrx {
            id: 0,
            name: "test".to_string(),
            path: PathBuf::from("test.trx"),
            data: gpu_data,
            backing: Some(StreamlineBacking::Imported(Arc::new(tractogram))),
            import_warnings: Vec::new(),
        }];

        let mut document = default_document();
        let source = make_node(
            &mut document,
            WorkflowNodeKind::StreamlineSource { source_id: 0 },
            GraphPos::new(0.0, 0.0),
        );
        let dedupe = make_node(
            &mut document,
            WorkflowNodeKind::RemoveDuplicates {
                params: trx_rs::DuplicateRemovalParams::default(),
            },
            GraphPos::new(200.0, 0.0),
        );
        document.graph.connect(
            OutPort {
                node: source,
                output: 0,
            },
            InPort {
                node: dedupe,
                input: 0,
            },
        );

        let runtime = evaluate_scene_plan_with_mode(
            &document,
            &streamline_assets,
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut HashMap::new(),
            &mut 1_000_000usize,
            &mut WorkflowExecutionCache::default(),
            WorkflowEvalMode::Interactive,
        );

        assert_eq!(runtime.scene_plan.reactive_streamline_plans.len(), 1);
        let state = runtime
            .node_state
            .get(&dedupe)
            .expect("remove duplicates state");
        assert!(state.error.is_none());
        assert!(matches!(
            state.execution,
            Some(WorkflowExecutionStatus::NeverRun)
        ));
    }

    #[test]
    fn interactive_unconnected_expensive_node_reports_only_input_error() {
        let mut document = default_document();
        let query = make_node(
            &mut document,
            WorkflowNodeKind::SurfaceDepthQuery { depth_mm: 2.0 },
            GraphPos::new(0.0, 0.0),
        );

        let runtime = evaluate_scene_plan_with_mode(
            &document,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut HashMap::new(),
            &mut 1_000_000usize,
            &mut WorkflowExecutionCache::default(),
            WorkflowEvalMode::Interactive,
        );

        assert!(runtime.scene_plan.surface_query_plans.is_empty());
        let state = runtime.node_state.get(&query).expect("query state");
        assert_eq!(
            state.error.as_deref(),
            Some("Surface Depth Query needs a streamline input")
        );
    }

    #[test]
    fn first_surface_overlay_input_uses_base_layer_config() {
        let surface = LoadedGiftiSurface {
            id: 7,
            name: "surface".to_string(),
            path: PathBuf::from("surface.gii"),
            data: Arc::new(GiftiSurfaceData {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                normals: vec![[0.0, 0.0, 1.0]; 2],
                indices: vec![0, 1, 1],
                bbox_min: Vec3::ZERO,
                bbox_max: Vec3::new(1.0, 0.0, 0.0),
            }),
            visible: true,
            opacity: 1.0,
            color: [0.72, 0.72, 0.72],
            outline_color: [0.0, 0.0, 0.0],
            outline_thickness: 1.0,
            show_projection_map: false,
            map_opacity: 1.0,
            map_threshold: 0.0,
            surface_gloss: 0.25,
            projection_colormap: SurfaceColormap::Inferno,
            auto_range: false,
            range_min: 0.0,
            range_max: 1.0,
        };
        let scalars = SurfaceScalars {
            structure: Some(CiftiStructure::CortexLeft),
            source_surface_id: None,
            vertex_count: 2,
            values: vec![1.0, 1.0],
            kind: ScalarKind::Continuous,
            metadata: crate::data::cifti::ScalarMetadata {
                map_name: "stat".to_string(),
                suggested_range: Some((0.0, 1.0)),
                series_index: None,
                series_value: None,
                label_table: Vec::new(),
            },
        };
        let layers = default_surface_overlay_layers();

        let appearance = compose_surface_appearance(
            surface.id,
            &surface,
            &layers,
            &[Some(EvaluatedValue {
                value: WorkflowValue::SurfaceScalars(scalars),
                stale: false,
            })],
        )
        .expect("appearance");

        assert_eq!(appearance.structure, Some(CiftiStructure::CortexLeft));
        assert!(
            appearance
                .vertex_rgba
                .iter()
                .any(|rgba| *rgba != DEFAULT_SURFACE_BASE_RGBA)
        );
        assert_eq!(appearance.legend_labels, vec!["Base".to_string()]);
    }
}

fn streamline_points(data: &TrxGpuData, streamline_index: usize) -> &[[f32; 3]] {
    let start = data.offsets[streamline_index] as usize;
    let end = data.offsets[streamline_index + 1] as usize;
    &data.positions[start..end]
}

fn summarize_value(value: &WorkflowValue) -> String {
    match value {
        WorkflowValue::Streamline(flow) => {
            format!("{} streamlines", flow.selected_streamlines.len())
        }
        WorkflowValue::Volume(_) => "Volume ready".to_string(),
        WorkflowValue::Surface(_) => "Surface ready".to_string(),
        WorkflowValue::Parcellation(_) => "Parcellation ready".to_string(),
        WorkflowValue::ParcelSelection(selection) => {
            format!("{} parcel labels", selection.labels.len())
        }
        WorkflowValue::SurfaceScalars(projection) => {
            format!("Surface scalars ({} values)", projection.values.len())
        }
        WorkflowValue::VolumeScalars(volume) => {
            format!(
                "Volume scalars {}x{}x{}",
                volume.dims[0], volume.dims[1], volume.dims[2]
            )
        }
        WorkflowValue::SurfaceAppearance(appearance) => {
            format!("Surface appearance for surface {}", appearance.source_id)
        }
        WorkflowValue::Cifti(_) => "CIFTI ready".to_string(),
        WorkflowValue::BundleSurface(bundle) => {
            if bundle.per_group {
                "Bundle surfaces split by group".to_string()
            } else {
                format!(
                    "Bundle surface from {} streamlines",
                    bundle.flow.selected_streamlines.len()
                )
            }
        }
        WorkflowValue::BoundaryField(plan) => {
            format!(
                "Boundary field from {} streamlines",
                plan.flow.selected_streamlines.len()
            )
        }
        WorkflowValue::Fixels(field) => format!("Fixels ({} peaks)", field.scalars.fixel_count),
        WorkflowValue::FixelScalars(s) => {
            format!("Fixel scalars '{}' ({} values)", s.name, s.fixel_count)
        }
        WorkflowValue::OdfField(_) => "ODF field ready".to_string(),
        WorkflowValue::OdxCatalog(c) => format!(
            "ODX catalog ({} DPV, {} DPF)",
            c.dpv_names.len(),
            c.dpf_names.len()
        ),
    }
}

pub(crate) fn robust_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo = finite[((n as f32) * 0.02).floor() as usize].min(finite[n - 1]);
    let hi = finite[((n as f32) * 0.98).floor() as usize].max(lo + 1e-6);
    (lo, hi)
}

pub(crate) fn add_groups_from_parcellation_from_label(
    label: &str,
    flow: &StreamlineFlow,
    parcellation: &ParcellationVolume,
    parcellation_name: &str,
) -> Result<StreamlineFlow, String> {
    let mut grouped = subset_tractogram_from_flow(flow)?;
    let prefix = parcellation_name
        .split('.')
        .next()
        .unwrap_or(parcellation_name)
        .trim()
        .to_string();
    let mut label_groups = BTreeMap::<u32, Vec<u32>>::new();

    for (new_index, &streamline_index) in flow.selected_streamlines.iter().enumerate() {
        let mut labels_hit = BTreeSet::new();
        for point in streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index as usize) {
            if let Some(label) = parcellation.sample_label_world(Vec3::from(*point)) {
                if label != 0 {
                    labels_hit.insert(label);
                }
            }
        }
        for label in labels_hit {
            label_groups
                .entry(label)
                .or_default()
                .push(new_index as u32);
        }
    }

    for (label, members) in label_groups {
        if members.is_empty() {
            continue;
        }
        let group_name = format!("{}_{}", prefix, parcellation.label_name(label));
        grouped.insert_group(group_name.clone(), members);
        let color = parcellation.label_color(label);
        let rgb = [[
            (color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        ]];
        grouped.insert_dpg(
            group_name,
            "color",
            DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
        );
    }

    let gpu_data = Arc::new(TrxGpuData::from_tractogram(&grouped).map_err(|err| err.to_string())?);
    let selected = (0..gpu_data.nb_streamlines as u32).collect();
    Ok(StreamlineFlow {
        dataset: Arc::new(StreamlineDataset {
            name: label.to_string(),
            gpu_data,
            backing: StreamlineBacking::Derived(Arc::new(grouped)),
        }),
        selected_streamlines: Arc::new(selected),
        color_mode: flow.color_mode.clone(),
        scalar_auto_range: flow.scalar_auto_range,
        scalar_range_min: flow.scalar_range_min,
        scalar_range_max: flow.scalar_range_max,
    })
}

fn crop_flow_to_parcels(
    flow: &StreamlineFlow,
    parcellation: &ParcellationVolume,
    labels: &BTreeSet<u32>,
    keep_inside: bool,
) -> Result<Tractogram, String> {
    let mut tractogram = Tractogram::new();
    for &streamline_index in flow.selected_streamlines.iter() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index as usize);
        let segments = if keep_inside {
            parcellation.crop_streamline_inside(points, labels)
        } else {
            parcellation.crop_streamline_outside(points, labels)
        };
        for segment in segments {
            tractogram
                .push_streamline(&segment)
                .map_err(|err| err.to_string())?;
        }
    }
    Ok(tractogram)
}

pub(crate) fn materialize_merged_streamlines(
    left: &StreamlineFlow,
    right: &StreamlineFlow,
) -> Result<Tractogram, String> {
    let left = subset_tractogram_from_flow(left)?;
    let right = subset_tractogram_from_flow(right)?;
    let mut out = Tractogram::with_header(left.header().clone());

    for streamline in left.streamlines() {
        out.push_streamline(streamline)
            .map_err(|err| err.to_string())?;
    }
    for streamline in right.streamlines() {
        out.push_streamline(streamline)
            .map_err(|err| err.to_string())?;
    }

    Ok(out)
}

pub(crate) fn materialize_reactive_streamline_flow(
    plan: &ReactiveStreamlinePlan,
) -> Result<StreamlineFlow, String> {
    match &plan.op {
        ReactiveStreamlineOp::Merge => {
            let tractogram = materialize_merged_streamlines(&plan.left, &plan.right)?;
            streamline_flow_from_tractogram(plan.label.clone(), &plan.left, tractogram)
        }
        ReactiveStreamlineOp::RemoveDuplicates { params } => {
            let tractogram = subset_tractogram_from_flow(&plan.left)?;
            let deduped =
                remove_duplicates_tractogram(&tractogram, params).map_err(|err| err.to_string())?;
            streamline_flow_from_tractogram(plan.label.clone(), &plan.left, deduped)
        }
        ReactiveStreamlineOp::ParcelROI {
            parcellation,
            labels,
        } => {
            let selected = plan
                .left
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points =
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation.streamline_hits_labels(points, labels)
                })
                .collect();
            Ok(StreamlineFlow {
                selected_streamlines: Arc::new(selected),
                ..plan.left.clone()
            })
        }
        ReactiveStreamlineOp::ParcelROA {
            parcellation,
            labels,
        } => {
            let selected = plan
                .left
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points =
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation.streamline_avoids_labels(points, labels)
                })
                .collect();
            Ok(StreamlineFlow {
                selected_streamlines: Arc::new(selected),
                ..plan.left.clone()
            })
        }
        ReactiveStreamlineOp::ParcelEnd {
            parcellation,
            labels,
            endpoint_count,
        } => {
            let selected = plan
                .left
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points =
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation.streamline_end_hits_labels(points, labels, *endpoint_count)
                })
                .collect();
            Ok(StreamlineFlow {
                selected_streamlines: Arc::new(selected),
                ..plan.left.clone()
            })
        }
        ReactiveStreamlineOp::ParcelCrop {
            parcellation,
            labels,
            keep_inside,
        } => {
            let tractogram = crop_flow_to_parcels(&plan.left, parcellation, labels, *keep_inside)?;
            streamline_flow_from_tractogram(plan.label.clone(), &plan.left, tractogram)
        }
        ReactiveStreamlineOp::AddGroupsFromParcellation {
            parcellation,
            parcellation_name,
        } => add_groups_from_parcellation_from_label(
            &plan.label,
            &plan.left,
            parcellation,
            parcellation_name,
        ),
    }
}

fn streamline_flow_from_tractogram(
    label: String,
    source_flow: &StreamlineFlow,
    tractogram: Tractogram,
) -> Result<StreamlineFlow, String> {
    let gpu_data =
        Arc::new(TrxGpuData::from_tractogram(&tractogram).map_err(|err| err.to_string())?);
    let selected = (0..gpu_data.nb_streamlines as u32).collect();
    Ok(StreamlineFlow {
        dataset: Arc::new(StreamlineDataset {
            name: label,
            gpu_data,
            backing: StreamlineBacking::Derived(Arc::new(tractogram)),
        }),
        selected_streamlines: Arc::new(selected),
        color_mode: source_flow.color_mode.clone(),
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
    })
}

fn subset_tractogram_from_flow(flow: &StreamlineFlow) -> Result<Tractogram, String> {
    let header = match &flow.dataset.backing {
        StreamlineBacking::Native(any) => any.header().clone(),
        StreamlineBacking::Imported(tractogram) | StreamlineBacking::Derived(tractogram) => {
            tractogram.header().clone()
        }
    };
    let mut tractogram = Tractogram::with_header(header);
    let mut remap = HashMap::with_capacity(flow.selected_streamlines.len());
    for (new_index, &index) in flow.selected_streamlines.iter().enumerate() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), index as usize);
        tractogram
            .push_streamline(points)
            .map_err(|err| err.to_string())?;
        remap.insert(index, new_index as u32);
    }
    for (group_idx, (name, members)) in flow.dataset.gpu_data.groups.iter().enumerate() {
        let remapped: Vec<u32> = members
            .iter()
            .filter_map(|member| remap.get(member).copied())
            .collect();
        if remapped.is_empty() {
            continue;
        }
        tractogram.insert_group(name.clone(), remapped);
        if let Some(Some(color)) = flow.dataset.gpu_data.group_colors.get(group_idx) {
            let rgb = [[
                (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (color[2].clamp(0.0, 1.0) * 255.0) as u8,
            ]];
            tractogram.insert_dpg(
                name.clone(),
                "color",
                DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
            );
        }
    }
    Ok(tractogram)
}

fn flow_selects_entire_dataset(flow: &StreamlineFlow) -> bool {
    flow.selected_streamlines.len() == flow.dataset.gpu_data.nb_streamlines
        && flow
            .selected_streamlines
            .iter()
            .enumerate()
            .all(|(expected, &actual)| expected == actual as usize)
}

pub fn save_streamline_plan(plan: &SaveStreamlinePlan) -> Result<(), String> {
    if plan.output_path.as_os_str().is_empty() {
        return Err("Save path is empty".to_string());
    }

    if flow_selects_entire_dataset(&plan.flow)
        && matches!(&plan.flow.dataset.backing, StreamlineBacking::Native(_))
        && plan
            .output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("trx"))
    {
        if let StreamlineBacking::Native(any) = &plan.flow.dataset.backing {
            return any.save(&plan.output_path).map_err(|err| err.to_string());
        }
    }

    let tractogram = subset_tractogram_from_flow(&plan.flow)?;
    let header = match &plan.flow.dataset.backing {
        StreamlineBacking::Native(any) => Some(any.header().clone()),
        StreamlineBacking::Imported(tractogram) | StreamlineBacking::Derived(tractogram) => {
            Some(tractogram.header().clone())
        }
    };
    let trx_positions_dtype = match &plan.flow.dataset.backing {
        StreamlineBacking::Native(any) => any.dtype(),
        StreamlineBacking::Imported(_) | StreamlineBacking::Derived(_) => DType::Float32,
    };
    write_tractogram(
        &plan.output_path,
        &tractogram,
        &ConversionOptions {
            header,
            trx_positions_dtype,
            ..Default::default()
        },
    )
    .map_err(|err| err.to_string())
}

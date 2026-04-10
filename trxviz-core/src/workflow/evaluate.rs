use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use petgraph::Directed;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use trx_rs::{ConversionOptions, DType, DataArray, Tractogram, write_tractogram};

use crate::data::loaded_files::{FileId, LoadedNifti, LoadedTrx, StreamlineBacking};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::scene::LoadedGiftiSurface;

use super::jobs::{prime_expensive_record, sync_node_state_from_run_record};
use super::*;

pub fn evaluate_scene_plan(
    document: &WorkflowDocument,
    streamline_assets: &[LoadedTrx],
    volume_assets: &[LoadedNifti],
    surface_assets: &[LoadedGiftiSurface],
    parcellation_assets: &[LoadedParcellation],
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    execution_cache: &mut WorkflowExecutionCache,
    _run_expensive: bool,
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
    let surface_map: HashMap<FileId, &LoadedGiftiSurface> = surface_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let parcellation_map: HashMap<FileId, &LoadedParcellation> = parcellation_assets
        .iter()
        .map(|asset| (asset.asset.id, asset))
        .collect();

    let mut values = HashMap::<WorkflowNodeUuid, EvaluatedValue>::new();
    let mut projection_by_surface = HashMap::<FileId, SurfaceStreamlineMap>::new();

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
                connections
                    .get(&(node.uuid, input_idx))
                    .and_then(|remote| values.get(remote).cloned())
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
            &surface_map,
            &parcellation_map,
            display_ids,
            next_draw_id,
            &mut runtime.scene_plan,
            &mut projection_by_surface,
            &mut runtime.save_streamline_targets,
            execution_cache,
            _run_expensive,
            &mut node_state,
        );

        match result {
            Ok(Some(value)) => {
                if let WorkflowValue::Streamline(flow) = &value.value {
                    node_state.available_streamline_groups = flow
                        .dataset
                        .gpu_data
                        .groups
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect();
                }
                if node_state.summary == node.kind.title() {
                    node_state.summary = summarize_value(&value.value);
                }
                values.insert(node.uuid, value);
            }
            Ok(None) => {
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
            if let Some(projection) = projection_by_surface.get(&draw.source_id) {
                draw.show_projection_map = true;
                draw.range_min = projection.range_min;
                draw.range_max = projection.range_max;
                draw.projection_scalars = Some(projection.scalars.clone());
            }
        });

    runtime
}

fn compile_graph(
    document: &WorkflowDocument,
) -> Result<
    (
        Vec<WorkflowNodeUuid>,
        HashMap<(WorkflowNodeUuid, usize), WorkflowNodeUuid>,
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
        connections.insert((wire.to.node, wire.to.input), wire.from.node);
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
    surface_assets: &HashMap<FileId, &LoadedGiftiSurface>,
    parcellation_assets: &HashMap<FileId, &LoadedParcellation>,
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    scene_plan: &mut SceneFramePlan,
    projection_by_surface: &mut HashMap<FileId, SurfaceStreamlineMap>,
    save_targets: &mut HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    execution_cache: &mut WorkflowExecutionCache,
    _run_expensive: bool,
    node_state: &mut NodeEvalState,
) -> Result<Option<EvaluatedValue>, String> {
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
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    dataset,
                    selected_streamlines: Arc::new(selected),
                    color_mode: ColorMode::DirectionRgb,
                    scalar_auto_range: true,
                    scalar_range_min: 0.0,
                    scalar_range_max: 1.0,
                })
                .into(),
            ))
        }
        WorkflowNodeKind::VolumeSource { source_id } => {
            volume_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing volume source {source_id}"))?;
            Ok(Some(WorkflowValue::Volume(*source_id).into()))
        }
        WorkflowNodeKind::SurfaceSource { source_id } => {
            surface_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing surface source {source_id}"))?;
            Ok(Some(WorkflowValue::Surface(*source_id).into()))
        }
        WorkflowNodeKind::ParcellationSource { source_id } => {
            parcellation_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing parcellation source {source_id}"))?;
            Ok(Some(WorkflowValue::Parcellation(*source_id).into()))
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
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::GroupSelect { groups_csv } => {
            let flow = expect_streamline_input(inputs, "Group Select")?;
            match parse_group_filter(groups_csv) {
                GroupFilter::All => Ok(Some(WorkflowValue::Streamline(flow).into())),
                GroupFilter::None => Ok(Some(
                    WorkflowValue::Streamline(StreamlineFlow {
                        selected_streamlines: Arc::new(Vec::new()),
                        ..flow
                    })
                    .into(),
                )),
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
                    Ok(Some(
                        WorkflowValue::Streamline(StreamlineFlow {
                            selected_streamlines: Arc::new(selected),
                            ..flow
                        })
                        .into(),
                    ))
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
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
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
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
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
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::Streamline(cache.flow.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::RemoveDuplicates => {
            let flow = expect_streamline_input(inputs, "Remove Duplicates")?;
            let mut seen = HashSet::new();
            let mut keep = Vec::new();
            for &streamline_index in flow.selected_streamlines.iter() {
                let key = streamline_key(flow.dataset.gpu_data.as_ref(), streamline_index as usize);
                if seen.insert(key) {
                    keep.push(streamline_index);
                }
            }
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(keep),
                    ..flow
                })
                .into(),
            ))
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
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.reactive_streamline_plans.push(plan);
            if let Some(cache) = execution_cache.derived_streamline_cache.get(&node.uuid) {
                node_state.summary =
                    format!("{} streamlines", cache.flow.selected_streamlines.len());
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::Streamline(cache.flow.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint),
                }));
            }
            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Waiting")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::ParcelSelect { labels_csv } => {
            let source_id = expect_parcellation_input(inputs, "Parcel Select")?;
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let labels = resolve_selected_labels(labels_csv, &parcellation.asset.data);
            Ok(Some(
                WorkflowValue::ParcelSelection(ParcelSelection { source_id, labels }).into(),
            ))
        }
        WorkflowNodeKind::ParcelROI => {
            let flow = expect_streamline_input(inputs, "Parcel ROI")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROI")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROI is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation
                        .asset
                        .data
                        .streamline_hits_labels(points, &parcel_selection.labels)
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelROA => {
            let flow = expect_streamline_input(inputs, "Parcel ROA")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROA")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROA is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation
                        .asset
                        .data
                        .streamline_avoids_labels(points, &parcel_selection.labels)
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelEnd { endpoint_count } => {
            let flow = expect_streamline_input(inputs, "Parcel End")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel End")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel End is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation.asset.data.streamline_end_hits_labels(
                        points,
                        &parcel_selection.labels,
                        *endpoint_count,
                    )
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelLimiting | WorkflowNodeKind::ParcelTerminative => {
            let flow = expect_streamline_input(inputs, node.kind.title())?;
            let parcel_selection = expect_parcel_selection_input(inputs, node.kind.title())?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| format!("{} is missing its parcellation", node.kind.title()))?;
            let tractogram = match node.kind {
                WorkflowNodeKind::ParcelLimiting => crop_flow_to_parcels(
                    &flow,
                    &parcellation.asset.data,
                    &parcel_selection.labels,
                    true,
                )?,
                _ => crop_flow_to_parcels(
                    &flow,
                    &parcellation.asset.data,
                    &parcel_selection.labels,
                    false,
                )?,
            };
            let gpu_data =
                Arc::new(TrxGpuData::from_tractogram(&tractogram).map_err(|err| err.to_string())?);
            let selected = (0..gpu_data.nb_streamlines as u32).collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    dataset: Arc::new(StreamlineDataset {
                        name: node.label.clone(),
                        gpu_data,
                        backing: StreamlineBacking::Derived(Arc::new(tractogram)),
                    }),
                    selected_streamlines: Arc::new(selected),
                    color_mode: flow.color_mode.clone(),
                    scalar_auto_range: true,
                    scalar_range_min: 0.0,
                    scalar_range_max: 1.0,
                })
                .into(),
            ))
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
            let grouped = add_groups_from_parcellation(
                node,
                &flow,
                &parcellation.asset.data,
                &parcellation.asset.name,
            )?;
            Ok(Some(WorkflowValue::Streamline(grouped).into()))
        }
        WorkflowNodeKind::ColorByDirection => {
            let flow = expect_streamline_input(inputs, "Color By Direction")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::DirectionRgb,
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByGroup => {
            let flow = expect_streamline_input(inputs, "Color By Group")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Group,
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByDPV { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPV")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dpv(field.clone()),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPS")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dps(field.clone()),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::UniformColor { color } => {
            let flow = expect_streamline_input(inputs, "Uniform Color")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Uniform(*color),
                    ..flow
                })
                .into(),
            ))
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
                projection_by_surface.insert(cache.map.surface_id, cache.map.clone());
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceStreamlineMap(cache.map.clone()));
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::SurfaceStreamlineMap(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
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
                projection_by_surface.insert(cache.map.surface_id, cache.map.clone());
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceStreamlineMap(cache.map.clone()));
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::SurfaceStreamlineMap(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
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
            Ok(None)
        }
        WorkflowNodeKind::BundleSurfaceBuild {
            per_group,
            voxel_size_mm,
            threshold,
            smooth_sigma,
            min_component_volume_mm3,
            opacity,
        } => {
            let flow = expect_streamline_input(inputs, "Bundle Surface Build")?;
            let bundle = BundleSurfacePlan {
                build_node_uuid: node.uuid,
                label: node.label.clone(),
                flow,
                per_group: *per_group,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                opacity: *opacity,
            };
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let fingerprint = workflow_bundle_plan_fingerprint(&bundle);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.bundle_surface_plans.push(bundle.clone());
            Ok(Some(EvaluatedValue {
                value: WorkflowValue::BundleSurface(bundle),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }))
        }
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => {
            let source_id = expect_volume_input(inputs, "Volume Display")?;
            let _ = volume_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing volume {source_id}"))?;
            scene_plan.volume_draws.push(VolumeDrawPlan {
                source_id,
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            });
            Ok(None)
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
        } => {
            let source_id = expect_surface_input(inputs, "Surface Display")?;
            let _surface = surface_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing surface {source_id}"))?;
            let projection = inputs
                .get(1)
                .and_then(|value| value.as_ref())
                .and_then(|value| {
                    if let WorkflowValue::SurfaceStreamlineMap(value) = &value.value {
                        Some(value.clone())
                    } else {
                        None
                    }
                });
            let projection_enabled = *show_projection_map || projection.is_some();
            let final_range = projection
                .as_ref()
                .map(|p| (p.range_min, p.range_max))
                .unwrap_or((*range_min, *range_max));
            let projection_scalars = projection.as_ref().map(|value| value.scalars.clone());
            projection_by_surface.extend(
                projection
                    .as_ref()
                    .cloned()
                    .into_iter()
                    .map(|projection| (projection.surface_id, projection)),
            );
            scene_plan.surface_draws.push(SurfaceDrawPlan {
                node_uuid: node.uuid,
                source_id,
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
            });
            Ok(None)
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
            Ok(None)
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
            Ok(Some(EvaluatedValue {
                value: WorkflowValue::BoundaryField(plan),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }))
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
            Ok(None)
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
                color_mode: *color_mode,
                voxel_size_mm: bundle.voxel_size_mm,
                threshold: bundle.threshold,
                smooth_sigma: bundle.smooth_sigma,
                min_component_volume_mm3: bundle.min_component_volume_mm3,
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
                format!("Displaying stale bundle surface ({})", color_mode.label())
            } else {
                format!("Displaying bundle surface ({})", color_mode.label())
            };
            scene_plan.bundle_draws.push(draw);
            Ok(None)
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
            Ok(None)
        }
        WorkflowNodeKind::ParcelSurfaceBuild => {
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel Surface Build")?;
            scene_plan.parcellation_draws.push(ParcellationDrawPlan {
                source_id: parcel_selection.source_id,
                labels: parcel_selection.labels,
                opacity: 0.9,
            });
            Ok(None)
        }
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

#[cfg(test)]
mod tests {
    use super::{GroupFilter, parse_group_filter};

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
}

fn streamline_points(data: &TrxGpuData, streamline_index: usize) -> &[[f32; 3]] {
    let start = data.offsets[streamline_index] as usize;
    let end = data.offsets[streamline_index + 1] as usize;
    &data.positions[start..end]
}

fn streamline_key(data: &TrxGpuData, streamline_index: usize) -> Vec<u8> {
    let points = streamline_points(data, streamline_index);
    let forward = bytemuck::cast_slice(points).to_vec();
    let mut reversed_points = points.to_vec();
    reversed_points.reverse();
    let reverse = bytemuck::cast_slice(reversed_points.as_slice()).to_vec();
    if reverse < forward { reverse } else { forward }
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
        WorkflowValue::SurfaceStreamlineMap(projection) => {
            format!(
                "Surface streamline map for surface {}",
                projection.surface_id
            )
        }
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

pub(crate) fn add_groups_from_parcellation(
    node: &WorkflowNode,
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
            name: node.label.clone(),
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

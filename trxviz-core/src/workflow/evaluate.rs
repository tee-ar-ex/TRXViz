use std::collections::HashMap;

use petgraph::Directed;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;

use crate::data::cifti::SurfaceScalars;
use crate::data::loaded_files::{FileId, LoadedCifti, LoadedNifti, LoadedOdx, LoadedTrx};
use crate::scene::LoadedGiftiSurface;

use super::*;

pub(crate) use super::eval_inputs::{
    expect_boundary_field_input, expect_bundle_surface_input, expect_cifti_input,
    expect_fixel_scalars_input, expect_fixels_input, expect_odf_field_input,
    expect_odx_catalog_input, expect_parcel_selection_input, expect_parcellation_input,
    expect_streamline_input, expect_surface_appearance_input, expect_surface_input,
    optional_group_selection_input, optional_volume_input, resolve_selected_labels,
    volume_scalars_from_nifti_volume,
};
pub use super::eval_streamlines::save_streamline_plan;
pub(crate) use super::eval_streamlines::{
    evaluate_derived_streamline_plan, materialize_reactive_streamline_flow, robust_range,
    summarize_value,
};
pub(crate) use super::eval_surface::{compose_surface_appearance, surface_display_model_matrix};

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
    debug_assert!(super::ops::validate_registry().is_ok());
    let compiled = compile_graph(document);
    let Ok((order, connections)) = compiled else {
        runtime.graph_error = compiled.err().map(|e| e.to_string());
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
            .op
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
            summary: node.op.title().to_string(),
            error: None,
            execution: None,
            fingerprint: None,
            last_result_summary: None,
            available_streamline_groups: Vec::new(),
            available_dps_fields: Vec::new(),
            available_dpv_fields: Vec::new(),
            overridden_fields: Vec::new(),
            overridden_values: std::collections::BTreeMap::new(),
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

        // Fall back to the node's streamline *input* for the autocomplete
        // group list when the output isn't itself a streamline flow. This
        // lets ops like MetaGroupSelect (which emits a GroupSelection)
        // still populate the inspector's group-name suggestions from the
        // upstream TRX wired into their streamline input.
        let input_streamline_groups: Vec<String> = input_values
            .iter()
            .flatten()
            .find_map(|v| match &v.value {
                WorkflowValue::Streamline(flow) => Some(
                    flow.dataset
                        .gpu_data
                        .groups
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        match result {
            Ok(outputs) if !outputs.is_empty() => {
                let first = &outputs[0];
                if let WorkflowValue::Streamline(flow) = &first.value {
                    node_state.available_streamline_groups = flow
                        .dataset
                        .gpu_data
                        .groups
                        .iter()
                        .map(|(name, _members)| name.clone())
                        .collect();
                    // Surface the DPS/DPV field names of this node's
                    // output so the inspector for downstream
                    // ColorByDps/ColorByDpv nodes can show a combobox
                    // populated with what's actually available. (We
                    // populate from the *upstream* node's output —
                    // i.e. ColorByDps reads its own input flow's
                    // names — because every Color* op passes the
                    // dataset through unchanged in shape.)
                    node_state.available_dps_fields = flow.dataset.gpu_data.dps_names.clone();
                    node_state.available_dpv_fields = flow.dataset.gpu_data.dpv_names.clone();
                }
                if node_state.summary == node.op.title() {
                    node_state.summary = summarize_value(&first.value);
                }
                values.insert(node.uuid, outputs);
            }
            Ok(_) => {
                if node_state.summary == node.op.title() {
                    node_state.summary = runtime
                        .save_streamline_targets
                        .get(&node.uuid)
                        .map(|target| format!("Ready to save to {}", target.output_path.display()))
                        .unwrap_or_else(|| node.op.title().to_string());
                }
            }
            Err(error) => {
                node_state.summary = node.op.title().to_string();
                node_state.error = Some(error.to_string());
            }
        }

        if node_state.available_streamline_groups.is_empty() {
            node_state.available_streamline_groups = input_streamline_groups;
        }

        runtime.node_state.insert(node.uuid, node_state);
    }

    // Patch projection-map scalars into surface draws (both spaces) now
    // that projection_by_surface is fully populated. of_type_mut covers
    // every SurfaceDrawPlan regardless of space — the same set the two
    // typed fields used to hold.
    runtime
        .scene_plan
        .draws
        .of_type_mut::<SurfaceDrawPlan>()
        .for_each(|draw| {
            if draw.show_projection_map
                && let Some(projection) = projection_by_surface.get(&draw.source_id)
            {
                let range = projection.metadata.suggested_range.unwrap_or((0.0, 1.0));
                draw.range_min = range.0;
                draw.range_max = range.1;
                draw.projection_scalars = Some(projection.values.clone());
            }
        });

    // Collapse multiple independent VolumeDisplay draws into one
    // CPU-composited slice stack (per-layer alpha, one quad per axis) so
    // co-registered volumes overlay instead of rendering as N opaque
    // coplanar quads that z-fight (the multi-volume flicker) and whose
    // background paints over fixels.
    fold_volume_draws_into_composite(&mut runtime.scene_plan, &volume_map, execution_cache);

    runtime
}

fn compile_graph(
    document: &WorkflowDocument,
) -> WorkflowResult<(
    Vec<WorkflowNodeUuid>,
    HashMap<(WorkflowNodeUuid, usize), OutPort>,
)> {
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

    let ordered = toposort(&graph, None)
        .map_err(|_| WorkflowError::Evaluation("Workflow graph contains a cycle".to_string()))?;
    let order = ordered
        .into_iter()
        .filter_map(|idx| graph.node_weight(idx).copied())
        .collect();

    Ok((order, connections))
}

/// Resolve a `VolumeBacking` to its `VolumeScalars`, mirroring
/// `EvalCtx::scalars_for` but callable from the post-evaluation pass
/// (which has `volume_map` + `execution_cache` in scope but no `EvalCtx`).
/// Returns `None` for an already-`Composite` backing or a missing asset.
fn resolve_volume_scalars(
    backing: &VolumeBacking,
    volume_map: &HashMap<FileId, &LoadedNifti>,
    execution_cache: &mut WorkflowExecutionCache,
) -> Option<std::sync::Arc<crate::data::cifti::VolumeScalars>> {
    match backing {
        VolumeBacking::InMemory { scalars, .. } => Some(scalars.clone()),
        VolumeBacking::File(id) => {
            if let Some(cached) = execution_cache.volume_scalars_cache.get(id) {
                return Some(cached.clone());
            }
            let loaded = volume_map.get(id)?;
            let scalars = std::sync::Arc::new(volume_scalars_from_nifti_volume(
                &loaded.volume,
                String::new(),
                *id,
            ));
            execution_cache
                .volume_scalars_cache
                .insert(*id, scalars.clone());
            Some(scalars)
        }
        VolumeBacking::Composite { .. } => None,
    }
}

/// Fold 2+ independent `VolumeDisplay` draws into a single
/// `VolumeBacking::Composite` so they render as one CPU-composited slice
/// quad per axis (with correct per-layer alpha) instead of N opaque
/// coplanar quads. Layer 0 = the first-evaluated draw (bottom); later
/// draws overlay on top, matching `VolumeOverlayStackOp`'s base-first
/// convention. Scenes with <2 volume draws, or any that already contain
/// an explicit `Composite` (a Volume Overlay Stack output), are left
/// untouched, so single-volume behavior is byte-identical.
fn fold_volume_draws_into_composite(
    scene_plan: &mut SceneFramePlan,
    volume_map: &HashMap<FileId, &LoadedNifti>,
    execution_cache: &mut WorkflowExecutionCache,
) {
    if scene_plan.draws.of_type::<VolumeDrawPlan>().count() < 2 {
        return;
    }
    if scene_plan
        .draws
        .of_type::<VolumeDrawPlan>()
        .any(|d| matches!(d.source, VolumeBacking::Composite { .. }))
    {
        return;
    }

    let mut layers: Vec<(
        std::sync::Arc<crate::data::cifti::VolumeScalars>,
        VolumeOverlayLayerConfig,
    )> = Vec::new();
    for draw in scene_plan.draws.of_type::<VolumeDrawPlan>() {
        // If any layer's scalars can't be resolved (missing asset), bail
        // and leave the draws untouched rather than silently dropping data.
        let Some(scalars) = resolve_volume_scalars(&draw.source, volume_map, execution_cache)
        else {
            return;
        };
        let interpolation = if matches!(scalars.kind, crate::data::cifti::ScalarKind::Label) {
            Interp::Nearest
        } else {
            Interp::Trilinear
        };
        layers.push((
            scalars,
            VolumeOverlayLayerConfig {
                enabled: true,
                opacity: draw.opacity,
                colormap: draw.colormap,
                window_center: draw.window_center,
                window_width: draw.window_width,
                // VolumeDrawPlan carries no threshold; a permissive gate
                // lets the overlay-layer alpha-by-value behavior alone hide
                // empty voxels.
                threshold_min: f32::NEG_INFINITY,
                threshold_max: f32::INFINITY,
                interpolation,
                legend_label: String::new(),
            },
        ));
    }

    let dims = layers[0].0.dims;
    let voxel_to_ras = layers[0].0.voxel_to_ras;
    let stack = CompositeVolumeStack {
        dims,
        voxel_to_ras,
        layers,
    };
    let handle = stack.handle();
    let composite = VolumeDrawPlan {
        source: VolumeBacking::Composite {
            handle,
            stack: std::sync::Arc::new(stack),
        },
        // Per-layer colormap/window/opacity live in the stack; the
        // draw-level fields are unused for a Composite source.
        colormap: crate::data::loaded_files::VolumeColormap::Grayscale,
        opacity: 1.0,
        window_center: 0.5,
        window_width: 1.0,
    };
    // Drop the per-volume draws (keeping every other draw in push order)
    // and append the merged composite.
    scene_plan
        .draws
        .retain(|d| d.as_any().downcast_ref::<VolumeDrawPlan>().is_none());
    scene_plan.draws.push(composite);
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
    mode: WorkflowEvalMode,
    node_state: &mut NodeEvalState,
) -> WorkflowResult<Vec<EvaluatedValue>> {
    let mut op_ctx = EvalCtx {
        node,
        inputs,
        streamline_assets,
        volume_assets,
        cifti_assets,
        surface_assets,
        parcellation_assets,
        odx_assets,
        display_ids,
        next_draw_id,
        scene_plan,
        projection_by_surface,
        save_targets,
        execution_cache,
        node_state,
        eval_mode: mode,
    };
    super::ops::evaluate(&node.op, &mut op_ctx)
}

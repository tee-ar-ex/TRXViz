use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use glam::Vec3;
use petgraph::Directed;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use trx_rs::{
    ConversionOptions, DType, DataArray, Tractogram, remove_duplicates_tractogram, write_tractogram,
};

use crate::data::cifti::{
    CiftiStructure, ScalarKind, ScalarMetadata, SurfaceScalars, VolumeScalars,
};
use crate::data::loaded_files::{
    FileId, LoadedCifti, LoadedNifti, LoadedOdx, LoadedTrx, StreamlineBacking,
};
use crate::data::odx_data::{FixelField, FixelScalars, OdfField, OdxCatalog};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::TrxGpuData;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::scene::LoadedGiftiSurface;
use crate::units::{ParcelId, StreamlineIndex};

use super::jobs::{prime_expensive_record, sync_node_state_from_run_record};
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
                        .map(|(name, _members): &(String, Vec<StreamlineIndex>)| name.clone())
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
                node_state.error = Some(error.to_string());
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
    };
    super::ops::evaluate(&node.kind, &mut op_ctx)
}

pub(crate) fn expect_fixels_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FixelField> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::Fixels(field) = &input.value {
            return Ok(field.clone());
        }
    }
    Err(WorkflowError::Evaluation(format!(
        "{label} needs a Fixels input"
    )))
}

pub(crate) fn expect_fixel_scalars_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FixelScalars> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::FixelScalars(s) = &input.value {
            return Ok(s.clone());
        }
    }
    Err(WorkflowError::Evaluation(format!(
        "{label} needs a FixelScalars input"
    )))
}

pub(crate) fn expect_odf_field_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<OdfField> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::OdfField(f) = &input.value {
            return Ok(f.clone());
        }
    }
    Err(WorkflowError::Evaluation(format!(
        "{label} needs an OdfField input"
    )))
}

pub(crate) fn expect_odx_catalog_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<OdxCatalog> {
    for input in inputs.iter().flatten() {
        if let WorkflowValue::OdxCatalog(c) = &input.value {
            return Ok(c.clone());
        }
    }
    Err(WorkflowError::Evaluation(format!(
        "{label} needs an OdxCatalog input"
    )))
}

pub(crate) fn optional_volume_scalars_input(
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

pub(crate) fn volume_scalars_from_nifti_volume(
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

pub(crate) fn expect_streamline_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<StreamlineFlow> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            ..
        }) => Ok(flow),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a streamline input"
        ))),
    }
}

pub(crate) fn expect_surface_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FileId> {
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
        .ok_or_else(|| WorkflowError::Evaluation(format!("{label} needs a surface input")))
}

pub(crate) fn expect_cifti_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FileId> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Cifti(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a CIFTI input"
        ))),
    }
}

pub(crate) fn expect_bundle_surface_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<(BundleSurfacePlan, bool)> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::BundleSurface(bundle),
            stale,
        }) => Ok((bundle, stale)),
        Some(_) => Err(WorkflowError::Evaluation(format!(
            "{label} needs a bundle surface input"
        ))),
        None => Err(WorkflowError::Evaluation(format!(
            "{label} is missing an input"
        ))),
    }
}

pub(crate) fn expect_volume_scalars_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<VolumeScalars> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::VolumeScalars(value),
            ..
        }) => Ok(value),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs volume scalars"
        ))),
    }
}

pub(crate) fn expect_surface_appearance_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<SurfaceAppearance> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::SurfaceAppearance(value),
            ..
        }) => Ok(value),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a surface appearance input"
        ))),
    }
}

pub(crate) fn expect_boundary_field_input(
    input: Option<&EvaluatedValue>,
    label: &str,
) -> WorkflowResult<(BoundaryFieldPlan, bool)> {
    match input {
        Some(EvaluatedValue {
            value: WorkflowValue::BoundaryField(plan),
            stale,
        }) => Ok((plan.clone(), *stale)),
        Some(_) => Err(WorkflowError::Evaluation(format!(
            "{label} needs a boundary field input"
        ))),
        None => Err(WorkflowError::Evaluation(format!(
            "{label} is missing an input"
        ))),
    }
}

pub(crate) fn expect_volume_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FileId> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Volume(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a volume input"
        ))),
    }
}

pub(crate) fn expect_parcellation_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<FileId> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Parcellation(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a parcellation input"
        ))),
    }
}

pub(crate) fn expect_parcel_selection_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> WorkflowResult<ParcelSelection> {
    match inputs.get(1).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::ParcelSelection(selection),
            ..
        }) => Ok(selection),
        _ => Err(WorkflowError::Evaluation(format!(
            "{label} needs a parcel selection input"
        ))),
    }
}

pub(crate) fn resolve_selected_labels(
    labels: &ParcelIdSet,
    parcellation: &ParcellationVolume,
) -> BTreeSet<ParcelId> {
    if !labels.is_empty() {
        return labels.0.clone();
    }

    let mut resolved = BTreeSet::new();
    for &label in &parcellation.labels {
        if label.0 != 0 {
            resolved.insert(label);
        }
    }
    resolved
}

pub(crate) fn evaluate_derived_streamline_plan(
    node: &WorkflowNode,
    plan: ReactiveStreamlinePlan,
    inputs: &[Option<EvaluatedValue>],
    execution_cache: &mut WorkflowExecutionCache,
    node_state: &mut NodeEvalState,
) -> WorkflowResult<Vec<EvaluatedValue>> {
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

pub(crate) fn compose_surface_appearance(
    surface_id: FileId,
    surface: &LoadedGiftiSurface,
    layers: &[SurfaceOverlayLayerConfig],
    scalar_inputs: &[Option<EvaluatedValue>],
) -> WorkflowResult<SurfaceAppearance> {
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

pub(crate) fn surface_display_model_matrix(
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
) -> WorkflowResult<()> {
    if scalars.vertex_count != surface.data.vertices.len() {
        return Err(WorkflowError::Evaluation(format!(
            "Surface scalars have {} vertices but surface {} has {}",
            scalars.vertex_count,
            surface_id,
            surface.data.vertices.len()
        )));
    }
    if let Some(bound_surface_id) = scalars.source_surface_id
        && bound_surface_id != surface_id
    {
        return Err(WorkflowError::Evaluation(format!(
            "Surface scalars are bound to surface {} and cannot be applied to surface {}",
            bound_surface_id, surface_id
        )));
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
    use crate::units::Millimeters;

    #[test]
    fn group_filter_empty_means_all() {
        assert_eq!(GroupFilter::from_csv(""), GroupFilter::All);
    }

    #[test]
    fn group_filter_none_sentinel_means_none() {
        assert_eq!(GroupFilter::from_csv("__none__"), GroupFilter::None);
    }

    #[test]
    fn group_filter_csv_keeps_selected_labels() {
        match GroupFilter::from_csv("CST_left, CST_right") {
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
            WorkflowNodeKind::SurfaceDepthQuery {
                depth_mm: Millimeters(2.0),
            },
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

pub(crate) fn summarize_value(value: &WorkflowValue) -> String {
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
) -> WorkflowResult<StreamlineFlow> {
    let mut grouped = subset_tractogram_from_flow(flow)?;
    let prefix = parcellation_name
        .split('.')
        .next()
        .unwrap_or(parcellation_name)
        .trim()
        .to_string();
    let mut label_groups = BTreeMap::<ParcelId, Vec<u32>>::new();

    for (new_index, &streamline_index) in flow.selected_streamlines.iter().enumerate() {
        let mut labels_hit = BTreeSet::new();
        for point in streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index.0 as usize)
        {
            if let Some(label) = parcellation.sample_label_world(Vec3::from(*point)) {
                if label.0 != 0 {
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

    let gpu_data = Arc::new(TrxGpuData::from_tractogram(&grouped)?);
    let selected = (0..gpu_data.nb_streamlines as u32)
        .map(StreamlineIndex)
        .collect();
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
    labels: &BTreeSet<ParcelId>,
    keep_inside: bool,
) -> WorkflowResult<Tractogram> {
    let mut tractogram = Tractogram::new();
    for &streamline_index in flow.selected_streamlines.iter() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index.0 as usize);
        let segments = if keep_inside {
            parcellation.crop_streamline_inside(points, labels)
        } else {
            parcellation.crop_streamline_outside(points, labels)
        };
        for segment in segments {
            tractogram
                .push_streamline(&segment)
                .map_err(|e| WorkflowError::Other(e.into()))?;
        }
    }
    Ok(tractogram)
}

pub(crate) fn materialize_merged_streamlines(
    left: &StreamlineFlow,
    right: &StreamlineFlow,
) -> WorkflowResult<Tractogram> {
    let left = subset_tractogram_from_flow(left)?;
    let right = subset_tractogram_from_flow(right)?;
    let mut out = Tractogram::with_header(left.header().clone());

    for streamline in left.streamlines() {
        out.push_streamline(streamline)
            .map_err(|e| WorkflowError::Other(e.into()))?;
    }
    for streamline in right.streamlines() {
        out.push_streamline(streamline)
            .map_err(|e| WorkflowError::Other(e.into()))?;
    }

    Ok(out)
}

pub(crate) fn materialize_reactive_streamline_flow(
    plan: &ReactiveStreamlinePlan,
) -> WorkflowResult<StreamlineFlow> {
    match &plan.op {
        ReactiveStreamlineOp::Merge => {
            let tractogram = materialize_merged_streamlines(&plan.left, &plan.right)?;
            streamline_flow_from_tractogram(plan.label.clone(), &plan.left, tractogram)
        }
        ReactiveStreamlineOp::RemoveDuplicates { params } => {
            let tractogram = subset_tractogram_from_flow(&plan.left)?;
            let deduped = remove_duplicates_tractogram(&tractogram, params)
                .map_err(|e| WorkflowError::Other(e.into()))?;
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
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), index.0 as usize);
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
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), index.0 as usize);
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
                        streamline_points(plan.left.dataset.gpu_data.as_ref(), index.0 as usize);
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
) -> WorkflowResult<StreamlineFlow> {
    let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram)?);
    let selected = (0..gpu_data.nb_streamlines as u32)
        .map(StreamlineIndex)
        .collect();
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

fn subset_tractogram_from_flow(flow: &StreamlineFlow) -> WorkflowResult<Tractogram> {
    let header = match &flow.dataset.backing {
        StreamlineBacking::Native(any) => any.header().clone(),
        StreamlineBacking::Imported(tractogram) | StreamlineBacking::Derived(tractogram) => {
            tractogram.header().clone()
        }
    };
    let mut tractogram = Tractogram::with_header(header);
    let mut remap = HashMap::with_capacity(flow.selected_streamlines.len());
    for (new_index, &index) in flow.selected_streamlines.iter().enumerate() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), index.0 as usize);
        tractogram
            .push_streamline(points)
            .map_err(|e| WorkflowError::Other(e.into()))?;
        remap.insert(index, new_index as u32);
    }
    for (group_idx, (name, members)) in flow
        .dataset
        .gpu_data
        .groups
        .iter()
        .enumerate()
        .map(|(idx, entry): (usize, &(String, Vec<StreamlineIndex>))| (idx, entry))
    {
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
            .all(|(expected, &actual)| expected == actual.0 as usize)
}

pub fn save_streamline_plan(plan: &SaveStreamlinePlan) -> WorkflowResult<()> {
    if plan.output_path.as_os_str().is_empty() {
        return Err(WorkflowError::Evaluation("Save path is empty".to_string()));
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
            return any
                .save(&plan.output_path)
                .map_err(|e| WorkflowError::Other(e.into()));
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
    .map_err(|e| WorkflowError::Other(e.into()))?;
    Ok(())
}

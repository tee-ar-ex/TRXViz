use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use glam::Vec3;
use trx_rs::{
    ConversionOptions, DType, DataArray, Tractogram, remove_duplicates_tractogram, write_tractogram,
};

use crate::data::loaded_files::StreamlineBacking;
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::TrxGpuData;
use crate::units::{ParcelId, StreamlineIndex};

use super::jobs::{prime_expensive_record, sync_node_state_from_run_record};
use super::*;

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
        WorkflowValue::Volume(b) => match b {
            crate::workflow::VolumeBacking::File(_) => "Volume ready".to_string(),
            crate::workflow::VolumeBacking::InMemory { scalars, .. } => {
                format!(
                    "Volume {}x{}x{}",
                    scalars.dims[0], scalars.dims[1], scalars.dims[2]
                )
            }
            crate::workflow::VolumeBacking::Composite { stack, .. } => {
                let active = stack.layers.iter().filter(|(_, c)| c.enabled).count();
                format!(
                    "Composite {}x{}x{} ({active} active layer(s))",
                    stack.dims[0], stack.dims[1], stack.dims[2]
                )
            }
        },
        WorkflowValue::Surface(_) => "Surface ready".to_string(),
        WorkflowValue::Parcellation(_) => "Parcellation ready".to_string(),
        WorkflowValue::ParcelSelection(selection) => {
            format!("{} parcel labels", selection.labels.len())
        }
        WorkflowValue::GroupSelection(filter) => match filter {
            crate::workflow::GroupFilter::All => "All groups".to_string(),
            crate::workflow::GroupFilter::None => "No groups".to_string(),
            crate::workflow::GroupFilter::Selected(labels) => {
                format!("{} group selection", labels.len())
            }
        },
        WorkflowValue::SurfaceScalars(projection) => {
            format!("Surface scalars ({} values)", projection.values.len())
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
        WorkflowValue::VoxelMask(m) => format!("{} voxels", m.count()),
        WorkflowValue::TrackingPlan(p) => {
            let parts = [
                p.seed_mask.as_ref().map(|_| "seed"),
                p.limiting_mask.as_ref().map(|_| "limiting"),
                p.roa_mask.as_ref().map(|_| "roa"),
                p.term_mask.as_ref().map(|_| "term"),
                p.no_end_mask.as_ref().map(|_| "no_end"),
                p.post_filter.as_ref().map(|_| "post_filter"),
            ];
            let active: Vec<&str> = parts.iter().filter_map(|p| *p).collect();
            if active.is_empty() {
                "tracking plan (empty)".to_string()
            } else {
                format!("tracking plan ({})", active.join(", "))
            }
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
        selected_streamlines: selected,
        color_mode: flow.color_mode.clone(),
        scalar_auto_range: flow.scalar_auto_range,
        scalar_range_min: flow.scalar_range_min,
        scalar_range_max: flow.scalar_range_max,
        scalar_colormap: flow.scalar_colormap,
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
            let left = &plan.left;
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
                dataset: left.dataset.clone(),
                selected_streamlines: selected,
                color_mode: left.color_mode.clone(),
                scalar_auto_range: left.scalar_auto_range,
                scalar_range_min: left.scalar_range_min,
                scalar_range_max: left.scalar_range_max,
                scalar_colormap: left.scalar_colormap,
            })
        }
        ReactiveStreamlineOp::ParcelROA {
            parcellation,
            labels,
        } => {
            let left = &plan.left;
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
                dataset: left.dataset.clone(),
                selected_streamlines: selected,
                color_mode: left.color_mode.clone(),
                scalar_auto_range: left.scalar_auto_range,
                scalar_range_min: left.scalar_range_min,
                scalar_range_max: left.scalar_range_max,
                scalar_colormap: left.scalar_colormap,
            })
        }
        ReactiveStreamlineOp::ParcelEnd {
            parcellation,
            labels,
            endpoint_count,
        } => {
            let left = &plan.left;
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
                dataset: left.dataset.clone(),
                selected_streamlines: selected,
                color_mode: left.color_mode.clone(),
                scalar_auto_range: left.scalar_auto_range,
                scalar_range_min: left.scalar_range_min,
                scalar_range_max: left.scalar_range_max,
                scalar_colormap: left.scalar_colormap,
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
        selected_streamlines: selected,
        color_mode: source_flow.color_mode.clone(),
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
        scalar_colormap: source_flow.scalar_colormap,
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

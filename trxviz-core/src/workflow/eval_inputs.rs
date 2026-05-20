use std::collections::BTreeSet;

use crate::data::cifti::{ScalarKind, ScalarMetadata, VolumeScalars};
use crate::data::loaded_files::FileId;
use crate::data::odx_data::{FixelField, FixelScalars, OdfField, OdxCatalog};
use crate::data::parcellation_data::ParcellationVolume;
use crate::units::ParcelId;

use super::*;

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

pub(crate) fn optional_volume_input(
    inputs: &[Option<EvaluatedValue>],
    index: usize,
) -> Option<crate::workflow::VolumeBacking> {
    match inputs.get(index).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Volume(b),
            ..
        }) => Some(b),
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
) -> WorkflowResult<crate::workflow::VolumeBacking> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Volume(backing),
            ..
        }) => Ok(backing),
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

pub(crate) fn optional_group_selection_input(
    inputs: &[Option<EvaluatedValue>],
    index: usize,
) -> Option<crate::workflow::GroupFilter> {
    match inputs.get(index).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::GroupSelection(filter),
            ..
        }) => Some(filter),
        _ => None,
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

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::data::loaded_files::FileId;
use crate::data::trx_data::ColorMode;

use super::*;

/// Fingerprint for SurfaceOverlayStack. Captures all layer configuration and
/// whether upstream scalar inputs are currently stale, so that downstream nodes
/// can detect when the composed vertex-colour array has genuinely changed.
pub fn workflow_surface_overlay_fingerprint(
    surface_id: FileId,
    layers: &[SurfaceOverlayLayerConfig],
    upstream_stale: bool,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface_id.hash(&mut hasher);
    upstream_stale.hash(&mut hasher);
    layers.len().hash(&mut hasher);
    for layer in layers {
        layer.enabled.hash(&mut hasher);
        for c in layer.solid_color {
            c.to_bits().hash(&mut hasher);
        }
        layer.opacity.to_bits().hash(&mut hasher);
        (layer.colormap as u32).hash(&mut hasher);
        layer.range_min.to_bits().hash(&mut hasher);
        layer.range_max.to_bits().hash(&mut hasher);
        layer.threshold_min.to_bits().hash(&mut hasher);
        layer.threshold_max.to_bits().hash(&mut hasher);
        layer.use_label_colors.hash(&mut hasher);
    }
    hasher.finish()
}

pub fn workflow_streamline_fingerprint(draw: &StreamlineDrawPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    draw.label.hash(&mut hasher);
    (draw.render_style as u32).hash(&mut hasher);
    draw.tube_radius_mm.0.to_bits().hash(&mut hasher);
    draw.tube_sides.hash(&mut hasher);
    draw.slab_half_width_mm.0.to_bits().hash(&mut hasher);
    draw.opacity.to_bits().hash(&mut hasher);
    hash_flow(&draw.flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_reactive_streamline_fingerprint(plan: &ReactiveStreamlinePlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    match &plan.op {
        ReactiveStreamlineOp::Merge => 0u8.hash(&mut hasher),
        ReactiveStreamlineOp::RemoveDuplicates { params } => {
            1u8.hash(&mut hasher);
            params.mode.hash(&mut hasher);
            params.tolerance_mm.to_bits().hash(&mut hasher);
            params.endpoint_tolerance_mm.to_bits().hash(&mut hasher);
            params.min_shared_voxel_fraction.to_bits().hash(&mut hasher);
        }
        ReactiveStreamlineOp::ParcelROI {
            parcellation,
            labels,
        } => {
            2u8.hash(&mut hasher);
            labels.hash(&mut hasher);
            parcellation.dims.hash(&mut hasher);
        }
        ReactiveStreamlineOp::ParcelROA {
            parcellation,
            labels,
        } => {
            3u8.hash(&mut hasher);
            labels.hash(&mut hasher);
            parcellation.dims.hash(&mut hasher);
        }
        ReactiveStreamlineOp::ParcelEnd {
            parcellation,
            labels,
            endpoint_count,
        } => {
            4u8.hash(&mut hasher);
            labels.hash(&mut hasher);
            endpoint_count.hash(&mut hasher);
            parcellation.dims.hash(&mut hasher);
        }
        ReactiveStreamlineOp::ParcelCrop {
            parcellation,
            labels,
            keep_inside,
        } => {
            5u8.hash(&mut hasher);
            labels.hash(&mut hasher);
            keep_inside.hash(&mut hasher);
            parcellation.dims.hash(&mut hasher);
        }
        ReactiveStreamlineOp::AddGroupsFromParcellation {
            parcellation,
            parcellation_name,
        } => {
            6u8.hash(&mut hasher);
            parcellation_name.hash(&mut hasher);
            parcellation.dims.hash(&mut hasher);
        }
    }
    hash_flow(&plan.left, &mut hasher);
    hash_flow(&plan.right, &mut hasher);
    hasher.finish()
}

/// Content fingerprint for the self-displaying `TriangleFundusOp`'s
/// derived geometry. Captures everything that changes the *built*
/// triangle/normal dataset: the input flow plus the geometry params.
/// Render-only params (`render_as_tubes`, `tube_radius_mm`) are
/// deliberately excluded — they don't change the dataset, only the
/// draw plan, and are already covered by
/// `workflow_streamline_fingerprint` which gates the tube job.
/// Keeping the cached dataset stable across frames is what stops the
/// `Arc::as_ptr` draw fingerprint from churning (the cylinder lockup).
pub fn workflow_triangle_fundus_fingerprint(
    flow: &StreamlineFlow,
    show_triangles: bool,
    show_normals: bool,
    normal_len_mm: f32,
    stride: usize,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    show_triangles.hash(&mut hasher);
    show_normals.hash(&mut hasher);
    normal_len_mm.to_bits().hash(&mut hasher);
    stride.hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_surface_query_fingerprint(
    flow: &StreamlineFlow,
    surface_id: FileId,
    depth_mm: crate::units::Millimeters,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface_id.hash(&mut hasher);
    depth_mm.0.to_bits().hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_surface_projection_fingerprint(
    flow: &StreamlineFlow,
    surface_id: FileId,
    depth_mm: crate::units::Millimeters,
    field: Option<&str>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface_id.hash(&mut hasher);
    depth_mm.0.to_bits().hash(&mut hasher);
    field.hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_bundle_build_fingerprint(draw: &BundleDrawPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    draw.label.hash(&mut hasher);
    draw.per_group.hash(&mut hasher);
    draw.build_mode.hash(&mut hasher);
    draw.voxel_size_mm.0.to_bits().hash(&mut hasher);
    draw.threshold.to_bits().hash(&mut hasher);
    draw.smooth_sigma.to_bits().hash(&mut hasher);
    draw.min_component_volume_mm3.0.to_bits().hash(&mut hasher);
    draw.tube_radius_mm.0.to_bits().hash(&mut hasher);
    draw.tube_sides.hash(&mut hasher);
    // opacity is excluded: it is a render-only parameter (GPU uniform) and does not
    // affect mesh geometry, so opacity changes must not trigger a mesh rebuild.
    hash_flow(&draw.flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_bundle_display_fingerprint(
    draw: &BundleDrawPlan,
    boundary_field_revision: Option<u64>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    workflow_bundle_build_fingerprint(draw).hash(&mut hasher);
    draw.color_mode.hash(&mut hasher);
    boundary_field_revision.hash(&mut hasher);
    hasher.finish()
}

pub fn workflow_bundle_plan_fingerprint(plan: &BundleSurfacePlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    plan.per_group.hash(&mut hasher);
    plan.build_mode.hash(&mut hasher);
    plan.voxel_size_mm.0.to_bits().hash(&mut hasher);
    plan.threshold.to_bits().hash(&mut hasher);
    plan.smooth_sigma.to_bits().hash(&mut hasher);
    plan.min_component_volume_mm3.0.to_bits().hash(&mut hasher);
    plan.tube_radius_mm.0.to_bits().hash(&mut hasher);
    plan.tube_sides.hash(&mut hasher);
    // opacity is excluded: it is a render-only parameter (GPU uniform) and does not
    // affect mesh geometry, so opacity changes must not trigger a mesh rebuild.
    hash_flow(&plan.flow, &mut hasher);
    hasher.finish()
}

/// Content fingerprint for `SampleVolumeAlongStreamline`'s derived
/// streamline dataset. Captures everything that changes the per-streamline
/// mean values: the input flow, the DPS field name written to the output,
/// and the volume backing identity. Used to keep the output `Arc` stable
/// across Interactive evaluations so downstream `hash_flow` pointers don't
/// churn and re-trigger tube/bundle rebuilds.
pub fn workflow_sample_volume_along_streamline_fingerprint(
    flow: &StreamlineFlow,
    dps_name: &str,
    backing: &VolumeBacking,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    dps_name.hash(&mut hasher);
    backing.slice_key().hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

pub fn workflow_boundary_plan_fingerprint(plan: &BoundaryFieldPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    plan.voxel_size_mm.0.to_bits().hash(&mut hasher);
    plan.sphere_lod.hash(&mut hasher);
    plan.normalization.hash(&mut hasher);
    plan.binning_mode.hash(&mut hasher);
    hash_flow(&plan.flow, &mut hasher);
    hasher.finish()
}

fn hash_flow(flow: &StreamlineFlow, state: &mut impl Hasher) {
    flow.dataset.name.hash(state);
    // Hash the dataset's `Arc` pointer identity. This makes the
    // fingerprint change whenever an upstream op produces a *new*
    // dataset (e.g. Purifibre attaching a `"fico"` DPS field, or
    // re-scoring with different params). Sharing an `Arc::clone` of
    // an existing dataset preserves identity and the fingerprint —
    // exactly what we want for ops like `ColorByDps` that just set
    // a different `color_mode` on the same data.
    //
    // Without this, downstream renders would silently reuse a cached
    // GPU upload of the previous dataset content (e.g. old FICO
    // values) and the user would see stale colors.
    //
    // Pointer addresses are session-local; that's fine because
    // fingerprints are in-memory cache keys, not on-disk identifiers.
    (Arc::as_ptr(&flow.dataset) as usize).hash(state);
    // DPS / DPV field-name lists are still useful in the hash so a
    // downstream consumer's fingerprint changes when a NEW field
    // appears in the same Arc (rare but possible if someone uses
    // `Arc::make_mut`). Cheap; keep them.
    for name in &flow.dataset.gpu_data.dps_names {
        name.hash(state);
    }
    for name in &flow.dataset.gpu_data.dpv_names {
        name.hash(state);
    }
    flow.selected_streamlines.len().hash(state);
    for index in flow.selected_streamlines.iter().take(128).copied() {
        index.hash(state);
    }
    // Colormap matters for the actual rendered colors when
    // color_mode is a scalar mode. Hash it so changing the colormap
    // invalidates downstream upload caches.
    flow.scalar_colormap.hash(state);
    match &flow.color_mode {
        ColorMode::DirectionRgb => 0u8.hash(state),
        ColorMode::Dpv(name) => {
            1u8.hash(state);
            name.hash(state);
        }
        ColorMode::Dps(name) => {
            2u8.hash(state);
            name.hash(state);
        }
        ColorMode::Group => 3u8.hash(state),
        ColorMode::Uniform(color) => {
            4u8.hash(state);
            for channel in color {
                channel.to_bits().hash(state);
            }
        }
    }
}

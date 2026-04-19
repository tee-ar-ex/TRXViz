mod add_groups_from_parcellation;
mod color_by_direction;
mod color_by_dps;
mod color_by_dpv;
mod color_by_group;
mod group_select;
mod limit_streamlines;
mod merge;
mod parcel_reactive;
mod parcel_select;
mod parcellation_display;
mod parcellation_source;
mod random_subset;
mod remove_duplicates;
mod save_streamlines;
mod surface_depth_query;
mod surface_projection;
mod sphere_query;
mod streamline_display;
mod streamline_source;
mod uniform_color;

use super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowResult};
use crate::error::WorkflowError;

pub(super) fn try_evaluate(
    kind: &WorkflowNodeKind,
    ctx: &mut EvalCtx<'_, '_>,
) -> Option<WorkflowResult<Vec<super::EvaluatedValue>>> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp {
                source_id: *source_id,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => Some(
            limit_streamlines::LimitStreamlinesOp {
                limit: *limit,
                randomize: *randomize,
                seed: *seed,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::GroupSelect { groups } => Some(
            group_select::GroupSelectOp {
                groups: groups.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::RandomSubset { limit, seed } => Some(
            random_subset::RandomSubsetOp {
                limit: *limit,
                seed: *seed,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SphereQuery { center, radius_mm } => Some(
            sphere_query::SphereQueryOp {
                center: *center,
                radius_mm: *radius_mm,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => Some(
            surface_depth_query::SurfaceDepthQueryOp {
                depth_mm: *depth_mm,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::RemoveDuplicates { params } => Some(
            remove_duplicates::RemoveDuplicatesOp {
                params: params.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::Merge => Some(merge::MergeOp.evaluate(ctx)),
        WorkflowNodeKind::AddGroupsFromParcellation => {
            Some(add_groups_from_parcellation::AddGroupsFromParcellationOp.evaluate(ctx))
        }
        WorkflowNodeKind::ParcelSelect { labels } => Some(
            parcel_select::ParcelSelectOp {
                labels: labels.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ParcelROI => Some(parcel_reactive::ParcelRoiOp.evaluate(ctx)),
        WorkflowNodeKind::ParcelROA => Some(parcel_reactive::ParcelRoaOp.evaluate(ctx)),
        WorkflowNodeKind::ParcelEnd { endpoint_count } => Some(
            parcel_reactive::ParcelEndOp {
                endpoint_count: *endpoint_count,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ParcelLimiting => Some(
            parcel_reactive::ParcelCropOp { keep_inside: true }.evaluate(ctx),
        ),
        WorkflowNodeKind::ParcelTerminative => Some(
            parcel_reactive::ParcelCropOp { keep_inside: false }.evaluate(ctx),
        ),
        WorkflowNodeKind::ColorByDirection => {
            Some(color_by_direction::ColorByDirectionOp.evaluate(ctx))
        }
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.evaluate(ctx)),
        WorkflowNodeKind::ColorByDPV { field } => Some(
            color_by_dpv::ColorByDpvOp {
                field: field.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ColorByDPS { field } => Some(
            color_by_dps::ColorByDpsOp {
                field: field.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::UniformColor { color } => Some(
            uniform_color::UniformColorOp { color: *color }.evaluate(ctx),
        ),
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => Some(
            surface_projection::SurfaceProjectionDensityOp {
                depth_mm: *depth_mm,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => Some(
            surface_projection::SurfaceProjectionMeanDpsOp {
                depth_mm: *depth_mm,
                field: field.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => Some(
            streamline_display::StreamlineDisplayOp {
                enabled: *enabled,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SaveStreamlines { output_path } => Some(
            save_streamlines::SaveStreamlinesOp {
                output_path: output_path.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ParcellationDisplay { labels, opacity } => Some(
            parcellation_display::ParcellationDisplayOp {
                labels: labels.clone(),
                opacity: *opacity,
            }
            .evaluate(ctx),
        ),
        _ => None,
    }
}

pub(super) fn title(kind: &WorkflowNodeKind) -> Option<&'static str> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => {
            Some(streamline_source::StreamlineSourceOp { source_id: *source_id }.title())
        }
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => Some(
            limit_streamlines::LimitStreamlinesOp {
                limit: *limit,
                randomize: *randomize,
                seed: *seed,
            }
            .title(),
        ),
        WorkflowNodeKind::GroupSelect { groups } => {
            Some(group_select::GroupSelectOp { groups: groups.clone() }.title())
        }
        WorkflowNodeKind::RandomSubset { limit, seed } => Some(
            random_subset::RandomSubsetOp {
                limit: *limit,
                seed: *seed,
            }
            .title(),
        ),
        WorkflowNodeKind::SphereQuery { center, radius_mm } => Some(
            sphere_query::SphereQueryOp {
                center: *center,
                radius_mm: *radius_mm,
            }
            .title(),
        ),
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => Some(
            surface_depth_query::SurfaceDepthQueryOp {
                depth_mm: *depth_mm,
            }
            .title(),
        ),
        WorkflowNodeKind::RemoveDuplicates { params } => Some(
            remove_duplicates::RemoveDuplicatesOp {
                params: params.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::Merge => Some(merge::MergeOp.title()),
        WorkflowNodeKind::AddGroupsFromParcellation => {
            Some(add_groups_from_parcellation::AddGroupsFromParcellationOp.title())
        }
        WorkflowNodeKind::ParcelSelect { labels } => Some(
            parcel_select::ParcelSelectOp {
                labels: labels.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::ParcelROI => Some(parcel_reactive::ParcelRoiOp.title()),
        WorkflowNodeKind::ParcelROA => Some(parcel_reactive::ParcelRoaOp.title()),
        WorkflowNodeKind::ParcelEnd { endpoint_count } => Some(
            parcel_reactive::ParcelEndOp {
                endpoint_count: *endpoint_count,
            }
            .title(),
        ),
        WorkflowNodeKind::ParcelLimiting => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: true }.title())
        }
        WorkflowNodeKind::ParcelTerminative => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: false }.title())
        }
        WorkflowNodeKind::ColorByDirection => Some(color_by_direction::ColorByDirectionOp.title()),
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.title()),
        WorkflowNodeKind::ColorByDPV { field } => {
            Some(color_by_dpv::ColorByDpvOp { field: field.clone() }.title())
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            Some(color_by_dps::ColorByDpsOp { field: field.clone() }.title())
        }
        WorkflowNodeKind::UniformColor { color } => {
            Some(uniform_color::UniformColorOp { color: *color }.title())
        }
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => Some(
            surface_projection::SurfaceProjectionDensityOp {
                depth_mm: *depth_mm,
            }
            .title(),
        ),
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => Some(
            surface_projection::SurfaceProjectionMeanDpsOp {
                depth_mm: *depth_mm,
                field: field.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => Some(
            streamline_display::StreamlineDisplayOp {
                enabled: *enabled,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            }
            .title(),
        ),
        WorkflowNodeKind::SaveStreamlines { output_path } => Some(
            save_streamlines::SaveStreamlinesOp {
                output_path: output_path.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::ParcellationDisplay { labels, opacity } => Some(
            parcellation_display::ParcellationDisplayOp {
                labels: labels.clone(),
                opacity: *opacity,
            }
            .title(),
        ),
        _ => None,
    }
}

pub(super) fn input_ports(kind: &WorkflowNodeKind) -> Option<&'static [PortKind]> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp { source_id: *source_id }.input_ports(),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => Some(
            limit_streamlines::LimitStreamlinesOp {
                limit: *limit,
                randomize: *randomize,
                seed: *seed,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::GroupSelect { groups } => {
            Some(group_select::GroupSelectOp { groups: groups.clone() }.input_ports())
        }
        WorkflowNodeKind::RandomSubset { limit, seed } => Some(
            random_subset::RandomSubsetOp {
                limit: *limit,
                seed: *seed,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SphereQuery { center, radius_mm } => Some(
            sphere_query::SphereQueryOp {
                center: *center,
                radius_mm: *radius_mm,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => Some(
            surface_depth_query::SurfaceDepthQueryOp {
                depth_mm: *depth_mm,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::RemoveDuplicates { params } => Some(
            remove_duplicates::RemoveDuplicatesOp {
                params: params.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::Merge => Some(merge::MergeOp.input_ports()),
        WorkflowNodeKind::AddGroupsFromParcellation => Some(
            add_groups_from_parcellation::AddGroupsFromParcellationOp.input_ports(),
        ),
        WorkflowNodeKind::ParcelSelect { labels } => Some(
            parcel_select::ParcelSelectOp {
                labels: labels.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ParcelROI => Some(parcel_reactive::ParcelRoiOp.input_ports()),
        WorkflowNodeKind::ParcelROA => Some(parcel_reactive::ParcelRoaOp.input_ports()),
        WorkflowNodeKind::ParcelEnd { endpoint_count } => Some(
            parcel_reactive::ParcelEndOp {
                endpoint_count: *endpoint_count,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ParcelLimiting => Some(
            parcel_reactive::ParcelCropOp { keep_inside: true }.input_ports(),
        ),
        WorkflowNodeKind::ParcelTerminative => Some(
            parcel_reactive::ParcelCropOp { keep_inside: false }.input_ports(),
        ),
        WorkflowNodeKind::ColorByDirection => {
            Some(color_by_direction::ColorByDirectionOp.input_ports())
        }
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.input_ports()),
        WorkflowNodeKind::ColorByDPV { field } => {
            Some(color_by_dpv::ColorByDpvOp { field: field.clone() }.input_ports())
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            Some(color_by_dps::ColorByDpsOp { field: field.clone() }.input_ports())
        }
        WorkflowNodeKind::UniformColor { color } => {
            Some(uniform_color::UniformColorOp { color: *color }.input_ports())
        }
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => Some(
            surface_projection::SurfaceProjectionDensityOp {
                depth_mm: *depth_mm,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => Some(
            surface_projection::SurfaceProjectionMeanDpsOp {
                depth_mm: *depth_mm,
                field: field.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => Some(
            streamline_display::StreamlineDisplayOp {
                enabled: *enabled,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SaveStreamlines { output_path } => Some(
            save_streamlines::SaveStreamlinesOp {
                output_path: output_path.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ParcellationDisplay { labels, opacity } => Some(
            parcellation_display::ParcellationDisplayOp {
                labels: labels.clone(),
                opacity: *opacity,
            }
            .input_ports(),
        ),
        _ => None,
    }
}

pub(super) fn output_ports(kind: &WorkflowNodeKind) -> Option<&'static [PortKind]> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp { source_id: *source_id }.output_ports(),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => Some(
            limit_streamlines::LimitStreamlinesOp {
                limit: *limit,
                randomize: *randomize,
                seed: *seed,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::GroupSelect { groups } => {
            Some(group_select::GroupSelectOp { groups: groups.clone() }.output_ports())
        }
        WorkflowNodeKind::RandomSubset { limit, seed } => Some(
            random_subset::RandomSubsetOp {
                limit: *limit,
                seed: *seed,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SphereQuery { center, radius_mm } => Some(
            sphere_query::SphereQueryOp {
                center: *center,
                radius_mm: *radius_mm,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => Some(
            surface_depth_query::SurfaceDepthQueryOp {
                depth_mm: *depth_mm,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::RemoveDuplicates { params } => Some(
            remove_duplicates::RemoveDuplicatesOp {
                params: params.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::Merge => Some(merge::MergeOp.output_ports()),
        WorkflowNodeKind::AddGroupsFromParcellation => Some(
            add_groups_from_parcellation::AddGroupsFromParcellationOp.output_ports(),
        ),
        WorkflowNodeKind::ParcelSelect { labels } => Some(
            parcel_select::ParcelSelectOp {
                labels: labels.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ParcelROI => Some(parcel_reactive::ParcelRoiOp.output_ports()),
        WorkflowNodeKind::ParcelROA => Some(parcel_reactive::ParcelRoaOp.output_ports()),
        WorkflowNodeKind::ParcelEnd { endpoint_count } => Some(
            parcel_reactive::ParcelEndOp {
                endpoint_count: *endpoint_count,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ParcelLimiting => Some(
            parcel_reactive::ParcelCropOp { keep_inside: true }.output_ports(),
        ),
        WorkflowNodeKind::ParcelTerminative => Some(
            parcel_reactive::ParcelCropOp { keep_inside: false }.output_ports(),
        ),
        WorkflowNodeKind::ColorByDirection => {
            Some(color_by_direction::ColorByDirectionOp.output_ports())
        }
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.output_ports()),
        WorkflowNodeKind::ColorByDPV { field } => {
            Some(color_by_dpv::ColorByDpvOp { field: field.clone() }.output_ports())
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            Some(color_by_dps::ColorByDpsOp { field: field.clone() }.output_ports())
        }
        WorkflowNodeKind::UniformColor { color } => {
            Some(uniform_color::UniformColorOp { color: *color }.output_ports())
        }
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => Some(
            surface_projection::SurfaceProjectionDensityOp {
                depth_mm: *depth_mm,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => Some(
            surface_projection::SurfaceProjectionMeanDpsOp {
                depth_mm: *depth_mm,
                field: field.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => Some(
            streamline_display::StreamlineDisplayOp {
                enabled: *enabled,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SaveStreamlines { output_path } => Some(
            save_streamlines::SaveStreamlinesOp {
                output_path: output_path.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ParcellationDisplay { labels, opacity } => Some(
            parcellation_display::ParcellationDisplayOp {
                labels: labels.clone(),
                opacity: *opacity,
            }
            .output_ports(),
        ),
        _ => None,
    }
}

pub(super) fn validate_registry() -> WorkflowResult<()> {
    for tag in [
        streamline_source::StreamlineSourceOp { source_id: 0 }.tag(),
        parcellation_source::ParcellationSourceOp { source_id: 0 }.tag(),
        limit_streamlines::LimitStreamlinesOp {
            limit: 0,
            randomize: false,
            seed: 0,
        }
        .tag(),
        group_select::GroupSelectOp {
            groups: super::GroupFilter::All,
        }
        .tag(),
        random_subset::RandomSubsetOp { limit: 0, seed: 0 }.tag(),
        sphere_query::SphereQueryOp {
            center: [0.0; 3],
            radius_mm: crate::units::Millimeters(0.0),
        }
        .tag(),
        remove_duplicates::RemoveDuplicatesOp {
            params: trx_rs::DuplicateRemovalParams::default(),
        }
        .tag(),
        merge::MergeOp.tag(),
        add_groups_from_parcellation::AddGroupsFromParcellationOp.tag(),
        parcel_select::ParcelSelectOp {
            labels: super::ParcelIdSet::default(),
        }
        .tag(),
        parcel_reactive::ParcelRoiOp.tag(),
        parcel_reactive::ParcelRoaOp.tag(),
        parcel_reactive::ParcelEndOp { endpoint_count: 1 }.tag(),
        parcel_reactive::ParcelCropOp { keep_inside: true }.tag(),
        parcel_reactive::ParcelCropOp { keep_inside: false }.tag(),
        color_by_direction::ColorByDirectionOp.tag(),
        color_by_group::ColorByGroupOp.tag(),
        color_by_dpv::ColorByDpvOp {
            field: super::DpvFieldName::default(),
        }
        .tag(),
        color_by_dps::ColorByDpsOp {
            field: super::DpsFieldName::default(),
        }
        .tag(),
        uniform_color::UniformColorOp { color: [0.0; 4] }.tag(),
        surface_depth_query::SurfaceDepthQueryOp {
            depth_mm: crate::units::Millimeters(0.0),
        }
        .tag(),
        surface_projection::SurfaceProjectionDensityOp {
            depth_mm: crate::units::Millimeters(0.0),
        }
        .tag(),
        surface_projection::SurfaceProjectionMeanDpsOp {
            depth_mm: crate::units::Millimeters(0.0),
            field: super::DpsFieldName::default(),
        }
        .tag(),
        streamline_display::StreamlineDisplayOp {
            enabled: true,
            render_style: crate::data::trx_data::RenderStyle::Flat,
            tube_radius_mm: crate::units::Millimeters(0.0),
            tube_sides: 0,
            slab_half_width_mm: crate::units::Millimeters(0.0),
        }
        .tag(),
        save_streamlines::SaveStreamlinesOp {
            output_path: String::new(),
        }
        .tag(),
        parcellation_display::ParcellationDisplayOp {
            labels: super::ParcelIdSet::default(),
            opacity: 1.0,
        }
        .tag(),
    ] {
        if tag.is_empty() {
            return Err(WorkflowError::Evaluation(
                "Workflow op registry contains an empty tag".to_string(),
            ));
        }
    }
    Ok(())
}

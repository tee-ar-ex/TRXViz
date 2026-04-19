mod add_groups_from_parcellation;
mod bundle_boundary;
mod cifti_source;
mod cifti_structure;
mod color_by_direction;
mod color_by_dps;
mod color_by_dpv;
mod color_by_fixel_scalars;
mod color_by_group;
mod fixel_display;
mod group_select;
mod limit_streamlines;
mod merge;
mod odf_glyph_renderer;
mod odx_select;
mod odx_source;
mod parcel_reactive;
mod parcel_select;
mod parcellation_display;
mod parcellation_source;
mod random_subset;
mod remove_duplicates;
mod save_streamlines;
mod sphere_query;
mod streamline_display;
mod streamline_source;
mod surface_depth_query;
mod surface_display;
mod surface_projection;
mod surface_source;
mod uniform_color;
mod volume_display;
mod volume_source;

pub use add_groups_from_parcellation::AddGroupsFromParcellationOp;
pub use bundle_boundary::{
    BoundaryFieldBuildOp, BoundaryGlyphDisplayOp, BundleSurfaceBuildOp, BundleSurfaceDisplayOp,
    ParcelSurfaceBuildOp,
};
pub use cifti_source::CiftiSourceOp;
pub use cifti_structure::CiftiStructureOp;
pub use color_by_direction::ColorByDirectionOp;
pub use color_by_dps::ColorByDpsOp;
pub use color_by_dpv::ColorByDpvOp;
pub use color_by_fixel_scalars::ColorByFixelScalarsOp;
pub use color_by_group::ColorByGroupOp;
pub use fixel_display::{Fixel2DDisplayOp, Fixel3DDisplayOp};
pub use group_select::GroupSelectOp;
pub use limit_streamlines::LimitStreamlinesOp;
pub use merge::MergeOp;
pub use odf_glyph_renderer::OdfGlyphRendererOp;
pub use odx_select::{OdxFixelScalarSelectOp, OdxVolumeSelectOp};
pub use odx_source::OdxSourceOp;
pub use parcel_reactive::{ParcelCropOp, ParcelEndOp, ParcelRoaOp, ParcelRoiOp};
pub use parcel_select::ParcelSelectOp;
pub use parcellation_display::ParcellationDisplayOp;
pub use parcellation_source::ParcellationSourceOp;
pub use random_subset::RandomSubsetOp;
pub use remove_duplicates::RemoveDuplicatesOp;
pub use save_streamlines::SaveStreamlinesOp;
pub use sphere_query::SphereQueryOp;
pub use streamline_display::StreamlineDisplayOp;
pub use streamline_source::StreamlineSourceOp;
pub use surface_depth_query::SurfaceDepthQueryOp;
pub use surface_display::{SurfaceDisplayOp, SurfaceOverlayStackOp};
pub use surface_projection::{SurfaceProjectionDensityOp, SurfaceProjectionMeanDpsOp};
pub use surface_source::SurfaceSourceOp;
pub use uniform_color::UniformColorOp;
pub use volume_display::{VolumeDisplayOp, VolumeScalarsDisplayOp};
pub use volume_source::VolumeSourceOp;

use super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowResult};
use crate::error::WorkflowError;

pub(super) fn evaluate(
    kind: &WorkflowNodeKind,
    ctx: &mut EvalCtx<'_, '_>,
) -> WorkflowResult<Vec<super::EvaluatedValue>> {
    try_evaluate(kind, ctx).expect("workflow op registry should cover all node kinds")
}

fn try_evaluate(
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
        WorkflowNodeKind::VolumeSource { source_id } => Some(
            volume_source::VolumeSourceOp {
                source_id: *source_id,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::CiftiSource { source_id } => Some(
            cifti_source::CiftiSourceOp {
                source_id: *source_id,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SurfaceSource { source_id } => Some(
            surface_source::SurfaceSourceOp {
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
        WorkflowNodeKind::CiftiStructure {
            structure,
            map_index,
        } => Some(
            cifti_structure::CiftiStructureOp {
                structure: *structure,
                map_index: *map_index,
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
        WorkflowNodeKind::ParcelLimiting => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: true }.evaluate(ctx))
        }
        WorkflowNodeKind::ParcelTerminative => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: false }.evaluate(ctx))
        }
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
        WorkflowNodeKind::UniformColor { color } => {
            Some(uniform_color::UniformColorOp { color: *color }.evaluate(ctx))
        }
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
        WorkflowNodeKind::OdxSource { source_id } => Some(
            odx_source::OdxSourceOp {
                source_id: *source_id,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => Some(
            odx_select::OdxFixelScalarSelectOp {
                dpf_name: dpf_name.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(
            odx_select::OdxVolumeSelectOp {
                dpv_name: dpv_name.clone(),
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ColorByFixelScalars {
            colormap,
            range,
            length_scale_by_scalar,
        } => Some(
            color_by_fixel_scalars::ColorByFixelScalarsOp {
                colormap: *colormap,
                range: *range,
                length_scale_by_scalar: *length_scale_by_scalar,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::Fixel3DDisplay {
            line_width,
            length_scale,
            opacity,
            offset_from_slice,
            visible,
        } => Some(
            fixel_display::Fixel3DDisplayOp {
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                visible: *visible,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
        } => Some(
            fixel_display::Fixel2DDisplayOp {
                line_width: *line_width,
                opacity: *opacity,
                slab_thickness_mm: *slab_thickness_mm,
                length_scale: *length_scale,
                visible: *visible,
            }
            .evaluate(ctx),
        ),
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
        } => Some(
            odf_glyph_renderer::OdfGlyphRendererOp {
                scale: *scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                gloss: *gloss,
                vertex_colormap: *vertex_colormap,
                slice_axis: *slice_axis,
                opacity_gate: *opacity_gate,
                size_gate: *size_gate,
                detail: *detail,
                visible: *visible,
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
        } => Some(
            bundle_boundary::BundleSurfaceBuildOp {
                per_group: *per_group,
                build_mode: *build_mode,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                opacity: *opacity,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => Some(
            volume_display::VolumeDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => Some(
            volume_display::VolumeScalarsDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::SurfaceOverlayStack { layers } => Some(
            surface_display::SurfaceOverlayStackOp {
                layers: layers.clone(),
            }
            .evaluate(ctx),
        ),
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
        } => Some(
            surface_display::SurfaceDisplayOp {
                color: *color,
                opacity: *opacity,
                outline_color: *outline_color,
                outline_thickness: *outline_thickness,
                show_projection_map: *show_projection_map,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: *range_min,
                range_max: *range_max,
                space: *space,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => Some(
            bundle_boundary::BoundaryFieldBuildOp {
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::BundleSurfaceDisplay {
            color_mode,
            outline_thickness,
        } => Some(
            bundle_boundary::BundleSurfaceDisplayOp {
                color_mode: *color_mode,
                outline_thickness: *outline_thickness,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => Some(
            bundle_boundary::BoundaryGlyphDisplayOp {
                enabled: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            }
            .evaluate(ctx),
        ),
        WorkflowNodeKind::ParcelSurfaceBuild => {
            Some(bundle_boundary::ParcelSurfaceBuildOp.evaluate(ctx))
        }
    }
}

pub(super) fn title(kind: &WorkflowNodeKind) -> &'static str {
    title_opt(kind).expect("workflow op registry should cover all node kinds")
}

fn title_opt(kind: &WorkflowNodeKind) -> Option<&'static str> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::VolumeSource { source_id } => Some(
            volume_source::VolumeSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::CiftiSource { source_id } => Some(
            cifti_source::CiftiSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::SurfaceSource { source_id } => Some(
            surface_source::SurfaceSourceOp {
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
        WorkflowNodeKind::GroupSelect { groups } => Some(
            group_select::GroupSelectOp {
                groups: groups.clone(),
            }
            .title(),
        ),
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
        WorkflowNodeKind::CiftiStructure {
            structure,
            map_index,
        } => Some(
            cifti_structure::CiftiStructureOp {
                structure: *structure,
                map_index: *map_index,
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
        WorkflowNodeKind::ColorByDPV { field } => Some(
            color_by_dpv::ColorByDpvOp {
                field: field.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::ColorByDPS { field } => Some(
            color_by_dps::ColorByDpsOp {
                field: field.clone(),
            }
            .title(),
        ),
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
        WorkflowNodeKind::OdxSource { source_id } => Some(
            odx_source::OdxSourceOp {
                source_id: *source_id,
            }
            .title(),
        ),
        WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => Some(
            odx_select::OdxFixelScalarSelectOp {
                dpf_name: dpf_name.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(
            odx_select::OdxVolumeSelectOp {
                dpv_name: dpv_name.clone(),
            }
            .title(),
        ),
        WorkflowNodeKind::ColorByFixelScalars {
            colormap,
            range,
            length_scale_by_scalar,
        } => Some(
            color_by_fixel_scalars::ColorByFixelScalarsOp {
                colormap: *colormap,
                range: *range,
                length_scale_by_scalar: *length_scale_by_scalar,
            }
            .title(),
        ),
        WorkflowNodeKind::Fixel3DDisplay {
            line_width,
            length_scale,
            opacity,
            offset_from_slice,
            visible,
        } => Some(
            fixel_display::Fixel3DDisplayOp {
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                visible: *visible,
            }
            .title(),
        ),
        WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
        } => Some(
            fixel_display::Fixel2DDisplayOp {
                line_width: *line_width,
                opacity: *opacity,
                slab_thickness_mm: *slab_thickness_mm,
                length_scale: *length_scale,
                visible: *visible,
            }
            .title(),
        ),
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
        } => Some(
            odf_glyph_renderer::OdfGlyphRendererOp {
                scale: *scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                gloss: *gloss,
                vertex_colormap: *vertex_colormap,
                slice_axis: *slice_axis,
                opacity_gate: *opacity_gate,
                size_gate: *size_gate,
                detail: *detail,
                visible: *visible,
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
        } => Some(
            bundle_boundary::BundleSurfaceBuildOp {
                per_group: *per_group,
                build_mode: *build_mode,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                opacity: *opacity,
            }
            .title(),
        ),
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => Some(
            volume_display::VolumeDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            }
            .title(),
        ),
        WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => Some(
            volume_display::VolumeScalarsDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
            }
            .title(),
        ),
        WorkflowNodeKind::SurfaceOverlayStack { layers } => Some(
            surface_display::SurfaceOverlayStackOp {
                layers: layers.clone(),
            }
            .title(),
        ),
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
        } => Some(
            surface_display::SurfaceDisplayOp {
                color: *color,
                opacity: *opacity,
                outline_color: *outline_color,
                outline_thickness: *outline_thickness,
                show_projection_map: *show_projection_map,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: *range_min,
                range_max: *range_max,
                space: *space,
            }
            .title(),
        ),
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => Some(
            bundle_boundary::BoundaryFieldBuildOp {
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            }
            .title(),
        ),
        WorkflowNodeKind::BundleSurfaceDisplay {
            color_mode,
            outline_thickness,
        } => Some(
            bundle_boundary::BundleSurfaceDisplayOp {
                color_mode: *color_mode,
                outline_thickness: *outline_thickness,
            }
            .title(),
        ),
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => Some(
            bundle_boundary::BoundaryGlyphDisplayOp {
                enabled: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            }
            .title(),
        ),
        WorkflowNodeKind::ParcelSurfaceBuild => Some(bundle_boundary::ParcelSurfaceBuildOp.title()),
    }
}

pub(super) fn input_ports(kind: &WorkflowNodeKind) -> Option<&'static [PortKind]> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::VolumeSource { source_id } => Some(
            volume_source::VolumeSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::CiftiSource { source_id } => Some(
            cifti_source::CiftiSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SurfaceSource { source_id } => Some(
            surface_source::SurfaceSourceOp {
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
        WorkflowNodeKind::GroupSelect { groups } => Some(
            group_select::GroupSelectOp {
                groups: groups.clone(),
            }
            .input_ports(),
        ),
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
        WorkflowNodeKind::CiftiStructure {
            structure,
            map_index,
        } => Some(
            cifti_structure::CiftiStructureOp {
                structure: *structure,
                map_index: *map_index,
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
        WorkflowNodeKind::AddGroupsFromParcellation => {
            Some(add_groups_from_parcellation::AddGroupsFromParcellationOp.input_ports())
        }
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
        WorkflowNodeKind::ParcelLimiting => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: true }.input_ports())
        }
        WorkflowNodeKind::ParcelTerminative => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: false }.input_ports())
        }
        WorkflowNodeKind::ColorByDirection => {
            Some(color_by_direction::ColorByDirectionOp.input_ports())
        }
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.input_ports()),
        WorkflowNodeKind::ColorByDPV { field } => Some(
            color_by_dpv::ColorByDpvOp {
                field: field.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ColorByDPS { field } => Some(
            color_by_dps::ColorByDpsOp {
                field: field.clone(),
            }
            .input_ports(),
        ),
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
        WorkflowNodeKind::OdxSource { source_id } => Some(
            odx_source::OdxSourceOp {
                source_id: *source_id,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => Some(
            odx_select::OdxFixelScalarSelectOp {
                dpf_name: dpf_name.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(
            odx_select::OdxVolumeSelectOp {
                dpv_name: dpv_name.clone(),
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ColorByFixelScalars {
            colormap,
            range,
            length_scale_by_scalar,
        } => Some(
            color_by_fixel_scalars::ColorByFixelScalarsOp {
                colormap: *colormap,
                range: *range,
                length_scale_by_scalar: *length_scale_by_scalar,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::Fixel3DDisplay {
            line_width,
            length_scale,
            opacity,
            offset_from_slice,
            visible,
        } => Some(
            fixel_display::Fixel3DDisplayOp {
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                visible: *visible,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
        } => Some(
            fixel_display::Fixel2DDisplayOp {
                line_width: *line_width,
                opacity: *opacity,
                slab_thickness_mm: *slab_thickness_mm,
                length_scale: *length_scale,
                visible: *visible,
            }
            .input_ports(),
        ),
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
        } => Some(
            odf_glyph_renderer::OdfGlyphRendererOp {
                scale: *scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                gloss: *gloss,
                vertex_colormap: *vertex_colormap,
                slice_axis: *slice_axis,
                opacity_gate: *opacity_gate,
                size_gate: *size_gate,
                detail: *detail,
                visible: *visible,
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
        } => Some(
            bundle_boundary::BundleSurfaceBuildOp {
                per_group: *per_group,
                build_mode: *build_mode,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                opacity: *opacity,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => Some(
            volume_display::VolumeDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => Some(
            volume_display::VolumeScalarsDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::SurfaceOverlayStack { .. } => None,
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
        } => Some(
            surface_display::SurfaceDisplayOp {
                color: *color,
                opacity: *opacity,
                outline_color: *outline_color,
                outline_thickness: *outline_thickness,
                show_projection_map: *show_projection_map,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: *range_min,
                range_max: *range_max,
                space: *space,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => Some(
            bundle_boundary::BoundaryFieldBuildOp {
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::BundleSurfaceDisplay {
            color_mode,
            outline_thickness,
        } => Some(
            bundle_boundary::BundleSurfaceDisplayOp {
                color_mode: *color_mode,
                outline_thickness: *outline_thickness,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => Some(
            bundle_boundary::BoundaryGlyphDisplayOp {
                enabled: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            }
            .input_ports(),
        ),
        WorkflowNodeKind::ParcelSurfaceBuild => {
            Some(bundle_boundary::ParcelSurfaceBuildOp.input_ports())
        }
    }
}

pub(super) fn output_ports(kind: &WorkflowNodeKind) -> &'static [PortKind] {
    output_ports_opt(kind).expect("workflow op registry should cover all node kinds")
}

fn output_ports_opt(kind: &WorkflowNodeKind) -> Option<&'static [PortKind]> {
    match kind {
        WorkflowNodeKind::StreamlineSource { source_id } => Some(
            streamline_source::StreamlineSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ParcellationSource { source_id } => Some(
            parcellation_source::ParcellationSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::VolumeSource { source_id } => Some(
            volume_source::VolumeSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::CiftiSource { source_id } => Some(
            cifti_source::CiftiSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SurfaceSource { source_id } => Some(
            surface_source::SurfaceSourceOp {
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
        WorkflowNodeKind::GroupSelect { groups } => Some(
            group_select::GroupSelectOp {
                groups: groups.clone(),
            }
            .output_ports(),
        ),
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
        WorkflowNodeKind::CiftiStructure {
            structure,
            map_index,
        } => Some(
            cifti_structure::CiftiStructureOp {
                structure: *structure,
                map_index: *map_index,
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
        WorkflowNodeKind::AddGroupsFromParcellation => {
            Some(add_groups_from_parcellation::AddGroupsFromParcellationOp.output_ports())
        }
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
        WorkflowNodeKind::ParcelLimiting => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: true }.output_ports())
        }
        WorkflowNodeKind::ParcelTerminative => {
            Some(parcel_reactive::ParcelCropOp { keep_inside: false }.output_ports())
        }
        WorkflowNodeKind::ColorByDirection => {
            Some(color_by_direction::ColorByDirectionOp.output_ports())
        }
        WorkflowNodeKind::ColorByGroup => Some(color_by_group::ColorByGroupOp.output_ports()),
        WorkflowNodeKind::ColorByDPV { field } => Some(
            color_by_dpv::ColorByDpvOp {
                field: field.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ColorByDPS { field } => Some(
            color_by_dps::ColorByDpsOp {
                field: field.clone(),
            }
            .output_ports(),
        ),
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
        WorkflowNodeKind::OdxSource { source_id } => Some(
            odx_source::OdxSourceOp {
                source_id: *source_id,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => Some(
            odx_select::OdxFixelScalarSelectOp {
                dpf_name: dpf_name.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::OdxVolumeSelect { dpv_name } => Some(
            odx_select::OdxVolumeSelectOp {
                dpv_name: dpv_name.clone(),
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ColorByFixelScalars {
            colormap,
            range,
            length_scale_by_scalar,
        } => Some(
            color_by_fixel_scalars::ColorByFixelScalarsOp {
                colormap: *colormap,
                range: *range,
                length_scale_by_scalar: *length_scale_by_scalar,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::Fixel3DDisplay {
            line_width,
            length_scale,
            opacity,
            offset_from_slice,
            visible,
        } => Some(
            fixel_display::Fixel3DDisplayOp {
                line_width: *line_width,
                length_scale: *length_scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                visible: *visible,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::Fixel2DDisplay {
            line_width,
            opacity,
            slab_thickness_mm,
            length_scale,
            visible,
        } => Some(
            fixel_display::Fixel2DDisplayOp {
                line_width: *line_width,
                opacity: *opacity,
                slab_thickness_mm: *slab_thickness_mm,
                length_scale: *length_scale,
                visible: *visible,
            }
            .output_ports(),
        ),
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
        } => Some(
            odf_glyph_renderer::OdfGlyphRendererOp {
                scale: *scale,
                opacity: *opacity,
                offset_from_slice: *offset_from_slice,
                gloss: *gloss,
                vertex_colormap: *vertex_colormap,
                slice_axis: *slice_axis,
                opacity_gate: *opacity_gate,
                size_gate: *size_gate,
                detail: *detail,
                visible: *visible,
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
        } => Some(
            bundle_boundary::BundleSurfaceBuildOp {
                per_group: *per_group,
                build_mode: *build_mode,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                min_component_volume_mm3: *min_component_volume_mm3,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                opacity: *opacity,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => Some(
            volume_display::VolumeDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => Some(
            volume_display::VolumeScalarsDisplayOp {
                colormap: *colormap,
                opacity: *opacity,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::SurfaceOverlayStack { layers } => Some(
            surface_display::SurfaceOverlayStackOp {
                layers: layers.clone(),
            }
            .output_ports(),
        ),
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
        } => Some(
            surface_display::SurfaceDisplayOp {
                color: *color,
                opacity: *opacity,
                outline_color: *outline_color,
                outline_thickness: *outline_thickness,
                show_projection_map: *show_projection_map,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: *range_min,
                range_max: *range_max,
                space: *space,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => Some(
            bundle_boundary::BoundaryFieldBuildOp {
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::BundleSurfaceDisplay {
            color_mode,
            outline_thickness,
        } => Some(
            bundle_boundary::BundleSurfaceDisplayOp {
                color_mode: *color_mode,
                outline_thickness: *outline_thickness,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => Some(
            bundle_boundary::BoundaryGlyphDisplayOp {
                enabled: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            }
            .output_ports(),
        ),
        WorkflowNodeKind::ParcelSurfaceBuild => {
            Some(bundle_boundary::ParcelSurfaceBuildOp.output_ports())
        }
    }
}

pub(super) fn validate_registry() -> WorkflowResult<()> {
    for tag in [
        streamline_source::StreamlineSourceOp { source_id: 0 }.tag(),
        parcellation_source::ParcellationSourceOp { source_id: 0 }.tag(),
        volume_source::VolumeSourceOp { source_id: 0 }.tag(),
        cifti_source::CiftiSourceOp { source_id: 0 }.tag(),
        surface_source::SurfaceSourceOp { source_id: 0 }.tag(),
        odx_source::OdxSourceOp { source_id: 0 }.tag(),
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
        cifti_structure::CiftiStructureOp {
            structure: crate::data::cifti::CiftiStructure::CortexLeft,
            map_index: 0,
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
        odx_select::OdxFixelScalarSelectOp {
            dpf_name: String::new(),
        }
        .tag(),
        odx_select::OdxVolumeSelectOp {
            dpv_name: String::new(),
        }
        .tag(),
        color_by_fixel_scalars::ColorByFixelScalarsOp {
            colormap: crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
            range: None,
            length_scale_by_scalar: false,
        }
        .tag(),
        fixel_display::Fixel3DDisplayOp {
            line_width: 0.0,
            length_scale: 1.0,
            opacity: 1.0,
            offset_from_slice: 0.0,
            visible: true,
        }
        .tag(),
        fixel_display::Fixel2DDisplayOp {
            line_width: 0.0,
            opacity: 1.0,
            slab_thickness_mm: crate::units::Millimeters(0.0),
            length_scale: 1.0,
            visible: true,
        }
        .tag(),
        odf_glyph_renderer::OdfGlyphRendererOp {
            scale: 1.0,
            opacity: 1.0,
            offset_from_slice: 0.0,
            gloss: 0.0,
            vertex_colormap: super::GlyphColormap::Directional,
            slice_axis: super::WorkflowSliceViewKind::Axial,
            opacity_gate: super::OpacityGate::default(),
            size_gate: super::SizeGate::default(),
            detail: 1,
            visible: true,
        }
        .tag(),
        parcellation_display::ParcellationDisplayOp {
            labels: super::ParcelIdSet::default(),
            opacity: 1.0,
        }
        .tag(),
        bundle_boundary::BundleSurfaceBuildOp {
            per_group: false,
            build_mode: super::BundleSurfaceBuildMode::MarchingCubes,
            voxel_size_mm: crate::units::Millimeters(0.0),
            threshold: 0.0,
            smooth_sigma: 0.0,
            min_component_volume_mm3: crate::units::Millimeters(0.0),
            tube_radius_mm: crate::units::Millimeters(0.0),
            tube_sides: 0,
            opacity: 1.0,
        }
        .tag(),
        volume_display::VolumeDisplayOp {
            colormap: crate::data::loaded_files::VolumeColormap::Grayscale,
            opacity: 1.0,
            window_center: 0.0,
            window_width: 1.0,
        }
        .tag(),
        volume_display::VolumeScalarsDisplayOp {
            colormap: crate::data::loaded_files::VolumeColormap::Grayscale,
            opacity: 1.0,
        }
        .tag(),
        surface_display::SurfaceOverlayStackOp { layers: Vec::new() }.tag(),
        surface_display::SurfaceDisplayOp {
            color: [0.0; 3],
            opacity: 1.0,
            outline_color: [0.0; 3],
            outline_thickness: 0.0,
            show_projection_map: false,
            map_opacity: 1.0,
            map_threshold: 0.0,
            gloss: 0.0,
            projection_colormap: crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
            range_min: 0.0,
            range_max: 1.0,
            space: super::SurfaceDisplaySpace::Anatomical,
        }
        .tag(),
        bundle_boundary::BoundaryFieldBuildOp {
            voxel_size_mm: crate::units::Millimeters(0.0),
            sphere_lod: 0,
            normalization: crate::data::orientation_field::BoundaryGlyphNormalization::GlobalPeak,
        }
        .tag(),
        bundle_boundary::BundleSurfaceDisplayOp {
            color_mode: super::BundleSurfaceColorMode::Solid,
            outline_thickness: 0.0,
        }
        .tag(),
        bundle_boundary::BoundaryGlyphDisplayOp {
            enabled: true,
            scale: 1.0,
            density_3d_step: 1,
            slice_density_step: 1,
            color_mode: crate::data::orientation_field::BoundaryGlyphColorMode::DirectionRgb,
            min_contacts: 1,
        }
        .tag(),
        bundle_boundary::ParcelSurfaceBuildOp.tag(),
    ] {
        if tag.is_empty() {
            return Err(WorkflowError::Evaluation(
                "Workflow op registry contains an empty tag".to_string(),
            ));
        }
    }
    Ok(())
}

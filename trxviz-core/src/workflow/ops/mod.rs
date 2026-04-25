mod add_groups_from_parcellation;
mod bundle_boundary;
mod cifti_source;
mod cifti_structure;
mod color_by_direction;
mod color_by_dps;
mod color_by_dpv;
mod color_by_fixel_scalars;
mod color_by_group;
mod dipy_tractography;
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
mod plan_add;
mod prepare_hausdorff;
mod prepare_simple;
mod purifibre;
mod random_subset;
mod remove_duplicates;
mod roi_ops;
mod save_streamlines;
mod sphere_query;
mod streamline_display;
mod streamline_source;
mod surface_depth_query;
mod surface_display;
mod surface_projection;
mod surface_source;
mod tip_prune;
mod tracking_params;
mod uniform_color;
mod volume_display;
mod volume_source;
mod voxel_mask_display;
mod yeh_tractography;

pub use add_groups_from_parcellation::AddGroupsFromParcellationOp;
pub use bundle_boundary::{
    BoundaryGlyphDisplayOp, BundleSurfaceBuildOp, BundleSurfaceDisplayOp, ParcelSurfaceBuildOp,
    StreamlineDirectionFieldOp,
};
pub use cifti_source::CiftiSourceOp;
pub use cifti_structure::CiftiStructureOp;
pub use color_by_direction::ColorByDirectionOp;
pub use color_by_dps::ColorByDpsOp;
pub use color_by_dpv::ColorByDpvOp;
pub use color_by_fixel_scalars::ColorByFixelScalarsOp;
pub use color_by_group::ColorByGroupOp;
pub use dipy_tractography::DipyTractographyOp;
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
pub use plan_add::{AddEndRegionOp, AddLimitingOp, AddNoEndOp, AddRoaOp, AddRoiOp, AddTermOp};
pub use prepare_hausdorff::PrepareHausdorffPlanOp;
pub use prepare_simple::PrepareSimplePlanOp;
pub use purifibre::PurifibreOp;
pub use random_subset::RandomSubsetOp;
pub use remove_duplicates::RemoveDuplicatesOp;
pub use roi_ops::{RoiFromParcelOp, RoiFromShapeOp, RoiFromVolumeOp, RoiShape};
pub use save_streamlines::SaveStreamlinesOp;
pub use sphere_query::SphereQueryOp;
pub use streamline_display::StreamlineDisplayOp;
pub use streamline_source::StreamlineSourceOp;
pub use surface_depth_query::SurfaceDepthQueryOp;
pub use surface_display::{SurfaceDisplayOp, SurfaceOverlayStackOp};
pub use surface_projection::{SurfaceProjectionDensityOp, SurfaceProjectionMeanDpsOp};
pub use surface_source::SurfaceSourceOp;
pub use tip_prune::TipPruneOp;
pub use uniform_color::UniformColorOp;
pub use volume_display::{VolumeDisplayOp, VolumeScalarsDisplayOp};
pub use volume_source::VolumeSourceOp;
pub use voxel_mask_display::VoxelMaskDisplayOp;
pub use yeh_tractography::YehTractographyOp;

use super::{
    BundleSurfaceBuildMode, BundleSurfaceColorMode, DpsFieldName, DpvFieldName, EvalCtx,
    GlyphColormap, GroupFilter, OpacityGate, ParcelIdSet, PortKind, SizeGate, SurfaceDisplaySpace,
    SurfaceOverlayLayerConfig, WorkflowOp, WorkflowResult, WorkflowSliceViewKind,
    default_boundary_field_normalization, default_boundary_field_sphere_lod,
    default_boundary_field_voxel_size_mm, default_boundary_glyph_color_mode,
    default_boundary_glyph_density_3d_step, default_boundary_glyph_min_contacts,
    default_boundary_glyph_scale, default_boundary_glyph_slice_density_step,
    default_bundle_surface_min_component_volume_mm3, default_bundle_surface_outline_thickness,
    default_enabled, default_false, default_fixel_colormap, default_fixel_length_scale,
    default_fixel_line_width, default_fixel_slab_thickness_mm, default_full_opacity,
    default_odf_glyph_detail, default_odf_glyph_scale, default_surface_overlay_layers,
    default_true, default_workflow_slice_view_kind,
};
use crate::data::cifti::CiftiStructure;
use crate::data::loaded_files::{FileId, VolumeColormap};
use crate::data::orientation_field::{BoundaryGlyphColorMode, BoundaryGlyphNormalization};
use crate::data::trx_data::RenderStyle;
use crate::error::WorkflowError;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::units::Millimeters;
use trx_rs::DuplicateRemovalParams;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowNodeKind {
    StreamlineSource {
        source_id: FileId,
    },
    VolumeSource {
        source_id: FileId,
    },
    CiftiSource {
        source_id: FileId,
    },
    SurfaceSource {
        source_id: FileId,
    },
    CiftiStructure {
        structure: CiftiStructure,
        map_index: usize,
    },
    ParcellationSource {
        source_id: FileId,
    },
    LimitStreamlines {
        limit: usize,
        randomize: bool,
        seed: u64,
    },
    GroupSelect {
        groups: GroupFilter,
    },
    RandomSubset {
        limit: usize,
        seed: u64,
    },
    SphereQuery {
        center: [f32; 3],
        radius_mm: Millimeters,
    },
    SurfaceDepthQuery {
        depth_mm: Millimeters,
    },
    RemoveDuplicates {
        params: DuplicateRemovalParams,
    },
    TipPrune {
        voxel_size_mm: f32,
        iterations: u32,
        min_support: u32,
        max_unsupported_fraction: f32,
    },
    Purifibre {
        trim_fraction: f32,
        puri_fraction: f32,
        spherical_smoothing_deg: f32,
    },
    Merge,
    AddGroupsFromParcellation,
    ParcelSelect {
        labels: ParcelIdSet,
    },
    ParcelROI,
    ParcelROA,
    ParcelEnd {
        endpoint_count: usize,
    },
    ParcelLimiting,
    ParcelTerminative,
    ParcelSurfaceBuild,
    ColorByDirection,
    ColorByGroup,
    ColorByDPV {
        field: DpvFieldName,
        colormap: crate::renderer::mesh_renderer::SurfaceColormap,
    },
    ColorByDPS {
        field: DpsFieldName,
        colormap: crate::renderer::mesh_renderer::SurfaceColormap,
    },
    UniformColor {
        color: [f32; 4],
    },
    SurfaceProjectionDensity {
        depth_mm: Millimeters,
    },
    SurfaceProjectionMeanDps {
        depth_mm: Millimeters,
        field: DpsFieldName,
    },
    SurfaceOverlayStack {
        #[serde(default = "default_surface_overlay_layers")]
        layers: Vec<SurfaceOverlayLayerConfig>,
    },
    BundleSurfaceBuild {
        #[serde(default)]
        per_group: bool,
        build_mode: BundleSurfaceBuildMode,
        voxel_size_mm: Millimeters,
        threshold: f32,
        smooth_sigma: f32,
        #[serde(default = "default_bundle_surface_min_component_volume_mm3")]
        min_component_volume_mm3: Millimeters,
        tube_radius_mm: Millimeters,
        tube_sides: u32,
        opacity: f32,
    },
    StreamlineDirectionField {
        #[serde(default = "default_boundary_field_voxel_size_mm")]
        voxel_size_mm: Millimeters,
        #[serde(default = "default_boundary_field_sphere_lod")]
        sphere_lod: u32,
        #[serde(default = "default_boundary_field_normalization")]
        normalization: BoundaryGlyphNormalization,
        binning_mode: crate::data::orientation_field::DirectionFieldBinningMode,
    },
    StreamlineDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        render_style: RenderStyle,
        tube_radius_mm: Millimeters,
        tube_sides: u32,
        slab_half_width_mm: Millimeters,
        opacity: f32,
    },
    VolumeDisplay {
        colormap: VolumeColormap,
        opacity: f32,
        window_center: f32,
        window_width: f32,
    },
    SurfaceDisplay {
        color: [f32; 3],
        opacity: f32,
        outline_color: [f32; 3],
        outline_thickness: f32,
        show_projection_map: bool,
        map_opacity: f32,
        map_threshold: f32,
        gloss: f32,
        projection_colormap: SurfaceColormap,
        range_min: f32,
        range_max: f32,
        space: SurfaceDisplaySpace,
    },
    VolumeScalarsDisplay {
        colormap: VolumeColormap,
        opacity: f32,
    },
    BundleSurfaceDisplay {
        #[serde(default)]
        color_mode: BundleSurfaceColorMode,
        #[serde(default = "default_bundle_surface_outline_thickness")]
        outline_thickness: f32,
    },
    BoundaryGlyphDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        #[serde(default = "default_boundary_glyph_scale")]
        scale: f32,
        #[serde(default = "default_boundary_glyph_density_3d_step")]
        density_3d_step: usize,
        #[serde(default = "default_boundary_glyph_slice_density_step")]
        slice_density_step: usize,
        #[serde(default = "default_boundary_glyph_color_mode")]
        color_mode: BoundaryGlyphColorMode,
        #[serde(default = "default_boundary_glyph_min_contacts")]
        min_contacts: u32,
    },
    ParcellationDisplay {
        labels: ParcelIdSet,
        opacity: f32,
    },
    SaveStreamlines {
        output_path: String,
    },
    OdxSource {
        source_id: FileId,
    },
    OdxFixelScalarSelect {
        #[serde(default)]
        dpf_name: String,
    },
    OdxVolumeSelect {
        #[serde(default)]
        dpv_name: String,
    },
    ColorByFixelScalars {
        #[serde(default = "default_fixel_colormap")]
        colormap: SurfaceColormap,
        #[serde(default)]
        range: Option<(f32, f32)>,
        #[serde(default)]
        length_scale_by_scalar: bool,
    },
    Fixel3DDisplay {
        #[serde(default = "default_fixel_line_width")]
        line_width: f32,
        #[serde(default = "default_fixel_length_scale")]
        length_scale: f32,
        #[serde(default = "default_full_opacity")]
        opacity: f32,
        #[serde(default)]
        offset_from_slice: f32,
        #[serde(default = "default_enabled")]
        visible: bool,
        #[serde(default = "default_true")]
        auto_gate_from_otsu: bool,
        #[serde(default)]
        opacity_gate: OpacityGate,
    },
    Fixel2DDisplay {
        #[serde(default = "default_fixel_line_width")]
        line_width: f32,
        #[serde(default = "default_full_opacity")]
        opacity: f32,
        #[serde(default = "default_fixel_slab_thickness_mm")]
        slab_thickness_mm: Millimeters,
        #[serde(default = "default_fixel_length_scale")]
        length_scale: f32,
        #[serde(default = "default_enabled")]
        visible: bool,
        #[serde(default = "default_true")]
        auto_gate_from_otsu: bool,
        #[serde(default)]
        opacity_gate: OpacityGate,
    },
    OdfGlyphRenderer {
        #[serde(default = "default_odf_glyph_scale")]
        scale: f32,
        #[serde(default = "default_true")]
        subtract_iso: bool,
        #[serde(default = "default_false")]
        norm_within_voxel: bool,
        #[serde(default = "default_full_opacity")]
        opacity: f32,
        #[serde(default)]
        offset_from_slice: f32,
        #[serde(default)]
        gloss: f32,
        #[serde(default)]
        vertex_colormap: GlyphColormap,
        #[serde(default = "default_workflow_slice_view_kind")]
        slice_axis: WorkflowSliceViewKind,
        #[serde(default)]
        opacity_gate: OpacityGate,
        #[serde(default)]
        size_gate: SizeGate,
        #[serde(default = "default_odf_glyph_detail")]
        detail: u32,
        #[serde(default = "default_enabled")]
        visible: bool,
    },
    // ── Tractography ops ────────────────────────────────────────────
    RoiFromParcel {
        #[serde(default)]
        labels: ParcelIdSet,
    },
    RoiFromVolume {
        #[serde(default)]
        threshold: f32,
    },
    RoiFromShape {
        #[serde(default)]
        center_ras: [f32; 3],
        #[serde(default)]
        radius_or_half_extent_mm: Millimeters,
        #[serde(default)]
        shape: RoiShape,
    },
    VoxelMaskDisplay {
        #[serde(default)]
        color: [f32; 4],
        #[serde(default)]
        opacity: f32,
        #[serde(default)]
        smooth_sigma: f32,
        #[serde(default)]
        min_component_volume_mm3: Millimeters,
    },
    AddRoi,
    AddRoa,
    AddEndRegion,
    AddNoEnd,
    AddLimiting,
    AddTerm,
    PrepareSimplePlan {
        #[serde(default)]
        override_step: bool,
        #[serde(default)]
        step_size_mm: f32,
        #[serde(default)]
        override_angle: bool,
        #[serde(default)]
        max_angle_deg: f32,
        #[serde(default)]
        override_min_len: bool,
        #[serde(default)]
        min_len_mm: f32,
        #[serde(default)]
        override_max_len: bool,
        #[serde(default)]
        max_len_mm: f32,
        #[serde(default)]
        override_fixel_threshold: bool,
        #[serde(default)]
        fixel_threshold: f32,
        #[serde(default)]
        override_smooth: bool,
        #[serde(default)]
        smooth_fraction: f32,
        #[serde(default)]
        override_fixel_otsu: bool,
        #[serde(default)]
        fixel_otsu: f32,
    },
    PrepareHausdorffPlan {
        #[serde(default)]
        tolerance_mm: f32,
        #[serde(default)]
        seed_tolerance_mm: f32,
        #[serde(default)]
        tracking_metric: Option<String>,
        #[serde(default)]
        otsu_scope: odx_rs::qc::OtsuScope,
        #[serde(default)]
        seed_fixel_otsu_factor: f32,
        #[serde(default)]
        not_end_fixel_otsu_factor: f32,
        #[serde(default)]
        max_reference_points: u32,
    },
    DipyTractography {
        #[serde(default)]
        step_size_mm: f32,
        #[serde(default)]
        max_angle_deg: f32,
        #[serde(default)]
        min_len_mm: f32,
        #[serde(default)]
        max_len_mm: f32,
        #[serde(default)]
        fixel_threshold: f32,
        #[serde(default)]
        relative_peak_threshold: f32,
        #[serde(default)]
        seeds_per_voxel: u32,
        #[serde(default)]
        max_points: u32,
        #[serde(default)]
        rng_seed: u64,
        direction_getter: super::types::DipyDirectionGetter,
    },
    YehTractography {
        #[serde(default)]
        step_size_mm: f32,
        #[serde(default)]
        max_angle_deg: f32,
        #[serde(default)]
        min_len_mm: f32,
        #[serde(default)]
        max_len_mm: f32,
        #[serde(default)]
        fixel_threshold: f32,
        #[serde(default)]
        smooth_fraction: f32,
        #[serde(default)]
        max_points: u32,
        #[serde(default)]
        target_streamlines: u32,
        #[serde(default)]
        max_seed_attempts: u32,
        #[serde(default)]
        rng_seed: u64,
    },
}

impl WorkflowNodeKind {
    pub fn title(&self) -> &'static str {
        title(self)
    }

    pub fn inputs(&self) -> Vec<PortKind> {
        match self {
            Self::SurfaceOverlayStack { layers } => {
                let mut ports = Vec::with_capacity(layers.len() + 1);
                ports.push(PortKind::Surface);
                ports.extend(std::iter::repeat(PortKind::SurfaceScalars).take(layers.len()));
                ports
            }
            _ => input_ports(self)
                .expect("handled by workflow op registry")
                .to_vec(),
        }
    }

    pub fn outputs(&self) -> Vec<PortKind> {
        output_ports(self).to_vec()
    }
}

macro_rules! with_workflow_op {
    ($kind:expr, |$op:ident| $body:expr) => {
        match $kind {
            WorkflowNodeKind::StreamlineSource { source_id } => {
                let $op = streamline_source::StreamlineSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::ParcellationSource { source_id } => {
                let $op = parcellation_source::ParcellationSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::VolumeSource { source_id } => {
                let $op = volume_source::VolumeSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::CiftiSource { source_id } => {
                let $op = cifti_source::CiftiSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::SurfaceSource { source_id } => {
                let $op = surface_source::SurfaceSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::LimitStreamlines {
                limit,
                randomize,
                seed,
            } => {
                let $op = limit_streamlines::LimitStreamlinesOp {
                    limit: *limit,
                    randomize: *randomize,
                    seed: *seed,
                };
                $body
            }
            WorkflowNodeKind::GroupSelect { groups } => {
                let $op = group_select::GroupSelectOp {
                    groups: groups.clone(),
                };
                $body
            }
            WorkflowNodeKind::RandomSubset { limit, seed } => {
                let $op = random_subset::RandomSubsetOp {
                    limit: *limit,
                    seed: *seed,
                };
                $body
            }
            WorkflowNodeKind::SphereQuery { center, radius_mm } => {
                let $op = sphere_query::SphereQueryOp {
                    center: *center,
                    radius_mm: *radius_mm,
                };
                $body
            }
            WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => {
                let $op = surface_depth_query::SurfaceDepthQueryOp {
                    depth_mm: *depth_mm,
                };
                $body
            }
            WorkflowNodeKind::CiftiStructure {
                structure,
                map_index,
            } => {
                let $op = cifti_structure::CiftiStructureOp {
                    structure: *structure,
                    map_index: *map_index,
                };
                $body
            }
            WorkflowNodeKind::RemoveDuplicates { params } => {
                let $op = remove_duplicates::RemoveDuplicatesOp {
                    params: params.clone(),
                };
                $body
            }
            WorkflowNodeKind::TipPrune {
                voxel_size_mm,
                iterations,
                min_support,
                max_unsupported_fraction,
            } => {
                let $op = tip_prune::TipPruneOp {
                    voxel_size_mm: *voxel_size_mm,
                    iterations: *iterations,
                    min_support: *min_support,
                    max_unsupported_fraction: *max_unsupported_fraction,
                };
                $body
            }
            WorkflowNodeKind::Purifibre {
                trim_fraction,
                puri_fraction,
                spherical_smoothing_deg,
            } => {
                let $op = purifibre::PurifibreOp {
                    trim_fraction: *trim_fraction,
                    puri_fraction: *puri_fraction,
                    spherical_smoothing_deg: *spherical_smoothing_deg,
                };
                $body
            }
            WorkflowNodeKind::Merge => {
                let $op = merge::MergeOp;
                $body
            }
            WorkflowNodeKind::AddGroupsFromParcellation => {
                let $op = add_groups_from_parcellation::AddGroupsFromParcellationOp;
                $body
            }
            WorkflowNodeKind::ParcelSelect { labels } => {
                let $op = parcel_select::ParcelSelectOp {
                    labels: labels.clone(),
                };
                $body
            }
            WorkflowNodeKind::ParcelROI => {
                let $op = parcel_reactive::ParcelRoiOp;
                $body
            }
            WorkflowNodeKind::ParcelROA => {
                let $op = parcel_reactive::ParcelRoaOp;
                $body
            }
            WorkflowNodeKind::ParcelEnd { endpoint_count } => {
                let $op = parcel_reactive::ParcelEndOp {
                    endpoint_count: *endpoint_count,
                };
                $body
            }
            WorkflowNodeKind::ParcelLimiting => {
                let $op = parcel_reactive::ParcelCropOp { keep_inside: true };
                $body
            }
            WorkflowNodeKind::ParcelTerminative => {
                let $op = parcel_reactive::ParcelCropOp { keep_inside: false };
                $body
            }
            WorkflowNodeKind::ParcelSurfaceBuild => {
                let $op = bundle_boundary::ParcelSurfaceBuildOp;
                $body
            }
            WorkflowNodeKind::ColorByDirection => {
                let $op = color_by_direction::ColorByDirectionOp;
                $body
            }
            WorkflowNodeKind::ColorByGroup => {
                let $op = color_by_group::ColorByGroupOp;
                $body
            }
            WorkflowNodeKind::ColorByDPV { field, colormap } => {
                let $op = color_by_dpv::ColorByDpvOp {
                    field: field.clone(),
                    colormap: *colormap,
                };
                $body
            }
            WorkflowNodeKind::ColorByDPS { field, colormap } => {
                let $op = color_by_dps::ColorByDpsOp {
                    field: field.clone(),
                    colormap: *colormap,
                };
                $body
            }
            WorkflowNodeKind::UniformColor { color } => {
                let $op = uniform_color::UniformColorOp { color: *color };
                $body
            }
            WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
                let $op = surface_projection::SurfaceProjectionDensityOp {
                    depth_mm: *depth_mm,
                };
                $body
            }
            WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
                let $op = surface_projection::SurfaceProjectionMeanDpsOp {
                    depth_mm: *depth_mm,
                    field: field.clone(),
                };
                $body
            }
            WorkflowNodeKind::SurfaceOverlayStack { layers } => {
                let $op = surface_display::SurfaceOverlayStackOp {
                    layers: layers.clone(),
                };
                $body
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
                let $op = bundle_boundary::BundleSurfaceBuildOp {
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
                $body
            }
            WorkflowNodeKind::StreamlineDirectionField {
                voxel_size_mm,
                sphere_lod,
                normalization,
                binning_mode,
            } => {
                let $op = bundle_boundary::StreamlineDirectionFieldOp {
                    voxel_size_mm: *voxel_size_mm,
                    sphere_lod: *sphere_lod,
                    normalization: *normalization,
                    binning_mode: *binning_mode,
                };
                $body
            }
            WorkflowNodeKind::StreamlineDisplay {
                enabled,
                render_style,
                tube_radius_mm,
                tube_sides,
                slab_half_width_mm,
                opacity,
            } => {
                let $op = streamline_display::StreamlineDisplayOp {
                    enabled: *enabled,
                    render_style: *render_style,
                    tube_radius_mm: *tube_radius_mm,
                    tube_sides: *tube_sides,
                    slab_half_width_mm: *slab_half_width_mm,
                    opacity: *opacity,
                };
                $body
            }
            WorkflowNodeKind::VolumeDisplay {
                colormap,
                opacity,
                window_center,
                window_width,
            } => {
                let $op = volume_display::VolumeDisplayOp {
                    colormap: *colormap,
                    opacity: *opacity,
                    window_center: *window_center,
                    window_width: *window_width,
                };
                $body
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
                let $op = surface_display::SurfaceDisplayOp {
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
                };
                $body
            }
            WorkflowNodeKind::VolumeScalarsDisplay { colormap, opacity } => {
                let $op = volume_display::VolumeScalarsDisplayOp {
                    colormap: *colormap,
                    opacity: *opacity,
                };
                $body
            }
            WorkflowNodeKind::BundleSurfaceDisplay {
                color_mode,
                outline_thickness,
            } => {
                let $op = bundle_boundary::BundleSurfaceDisplayOp {
                    color_mode: *color_mode,
                    outline_thickness: *outline_thickness,
                };
                $body
            }
            WorkflowNodeKind::BoundaryGlyphDisplay {
                enabled,
                scale,
                density_3d_step,
                slice_density_step,
                color_mode,
                min_contacts,
            } => {
                let $op = bundle_boundary::BoundaryGlyphDisplayOp {
                    enabled: *enabled,
                    scale: *scale,
                    density_3d_step: *density_3d_step,
                    slice_density_step: *slice_density_step,
                    color_mode: *color_mode,
                    min_contacts: *min_contacts,
                };
                $body
            }
            WorkflowNodeKind::ParcellationDisplay { labels, opacity } => {
                let $op = parcellation_display::ParcellationDisplayOp {
                    labels: labels.clone(),
                    opacity: *opacity,
                };
                $body
            }
            WorkflowNodeKind::SaveStreamlines { output_path } => {
                let $op = save_streamlines::SaveStreamlinesOp {
                    output_path: output_path.clone(),
                };
                $body
            }
            WorkflowNodeKind::OdxSource { source_id } => {
                let $op = odx_source::OdxSourceOp {
                    source_id: *source_id,
                };
                $body
            }
            WorkflowNodeKind::OdxFixelScalarSelect { dpf_name } => {
                let $op = odx_select::OdxFixelScalarSelectOp {
                    dpf_name: dpf_name.clone(),
                };
                $body
            }
            WorkflowNodeKind::OdxVolumeSelect { dpv_name } => {
                let $op = odx_select::OdxVolumeSelectOp {
                    dpv_name: dpv_name.clone(),
                };
                $body
            }
            WorkflowNodeKind::ColorByFixelScalars {
                colormap,
                range,
                length_scale_by_scalar,
            } => {
                let $op = color_by_fixel_scalars::ColorByFixelScalarsOp {
                    colormap: *colormap,
                    range: *range,
                    length_scale_by_scalar: *length_scale_by_scalar,
                };
                $body
            }
            WorkflowNodeKind::Fixel3DDisplay {
                line_width,
                length_scale,
                opacity,
                offset_from_slice,
                visible,
                auto_gate_from_otsu,
                opacity_gate,
            } => {
                let $op = fixel_display::Fixel3DDisplayOp {
                    line_width: *line_width,
                    length_scale: *length_scale,
                    opacity: *opacity,
                    offset_from_slice: *offset_from_slice,
                    visible: *visible,
                    auto_gate_from_otsu: *auto_gate_from_otsu,
                    opacity_gate: *opacity_gate,
                };
                $body
            }
            WorkflowNodeKind::Fixel2DDisplay {
                line_width,
                opacity,
                slab_thickness_mm,
                length_scale,
                visible,
                auto_gate_from_otsu,
                opacity_gate,
            } => {
                let $op = fixel_display::Fixel2DDisplayOp {
                    line_width: *line_width,
                    opacity: *opacity,
                    slab_thickness_mm: *slab_thickness_mm,
                    length_scale: *length_scale,
                    visible: *visible,
                    auto_gate_from_otsu: *auto_gate_from_otsu,
                    opacity_gate: *opacity_gate,
                };
                $body
            }
            WorkflowNodeKind::OdfGlyphRenderer {
                scale,
                subtract_iso,
                norm_within_voxel,
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
                let $op = odf_glyph_renderer::OdfGlyphRendererOp {
                    scale: *scale,
                    subtract_iso: *subtract_iso,
                    norm_within_voxel: *norm_within_voxel,
                    opacity: *opacity,
                    offset_from_slice: *offset_from_slice,
                    gloss: *gloss,
                    vertex_colormap: *vertex_colormap,
                    slice_axis: *slice_axis,
                    opacity_gate: *opacity_gate,
                    size_gate: *size_gate,
                    detail: *detail,
                    visible: *visible,
                };
                $body
            }
            WorkflowNodeKind::RoiFromParcel { labels } => {
                let $op = roi_ops::RoiFromParcelOp {
                    labels: labels.clone(),
                };
                $body
            }
            WorkflowNodeKind::RoiFromVolume { threshold } => {
                let $op = roi_ops::RoiFromVolumeOp {
                    threshold: *threshold,
                };
                $body
            }
            WorkflowNodeKind::RoiFromShape {
                center_ras,
                radius_or_half_extent_mm,
                shape,
            } => {
                let $op = roi_ops::RoiFromShapeOp {
                    center_ras: *center_ras,
                    radius_or_half_extent_mm: *radius_or_half_extent_mm,
                    shape: *shape,
                };
                $body
            }
            WorkflowNodeKind::VoxelMaskDisplay {
                color,
                opacity,
                smooth_sigma,
                min_component_volume_mm3,
            } => {
                let $op = voxel_mask_display::VoxelMaskDisplayOp {
                    color: *color,
                    opacity: *opacity,
                    smooth_sigma: *smooth_sigma,
                    min_component_volume_mm3: *min_component_volume_mm3,
                };
                $body
            }
            WorkflowNodeKind::AddRoi => {
                let $op = plan_add::AddRoiOp;
                $body
            }
            WorkflowNodeKind::AddRoa => {
                let $op = plan_add::AddRoaOp;
                $body
            }
            WorkflowNodeKind::AddEndRegion => {
                let $op = plan_add::AddEndRegionOp;
                $body
            }
            WorkflowNodeKind::AddNoEnd => {
                let $op = plan_add::AddNoEndOp;
                $body
            }
            WorkflowNodeKind::AddLimiting => {
                let $op = plan_add::AddLimitingOp;
                $body
            }
            WorkflowNodeKind::AddTerm => {
                let $op = plan_add::AddTermOp;
                $body
            }
            WorkflowNodeKind::PrepareSimplePlan {
                override_step,
                step_size_mm,
                override_angle,
                max_angle_deg,
                override_min_len,
                min_len_mm,
                override_max_len,
                max_len_mm,
                override_fixel_threshold,
                fixel_threshold,
                override_smooth,
                smooth_fraction,
                override_fixel_otsu,
                fixel_otsu,
            } => {
                let $op = prepare_simple::PrepareSimplePlanOp {
                    override_step: *override_step,
                    step_size_mm: *step_size_mm,
                    override_angle: *override_angle,
                    max_angle_deg: *max_angle_deg,
                    override_min_len: *override_min_len,
                    min_len_mm: *min_len_mm,
                    override_max_len: *override_max_len,
                    max_len_mm: *max_len_mm,
                    override_fixel_threshold: *override_fixel_threshold,
                    fixel_threshold: *fixel_threshold,
                    override_smooth: *override_smooth,
                    smooth_fraction: *smooth_fraction,
                    override_fixel_otsu: *override_fixel_otsu,
                    fixel_otsu: *fixel_otsu,
                };
                $body
            }
            WorkflowNodeKind::PrepareHausdorffPlan {
                tolerance_mm,
                seed_tolerance_mm,
                tracking_metric,
                otsu_scope,
                seed_fixel_otsu_factor,
                not_end_fixel_otsu_factor,
                max_reference_points,
            } => {
                let $op = prepare_hausdorff::PrepareHausdorffPlanOp {
                    tolerance_mm: *tolerance_mm,
                    seed_tolerance_mm: *seed_tolerance_mm,
                    tracking_metric: tracking_metric.clone(),
                    otsu_scope: *otsu_scope,
                    seed_fixel_otsu_factor: *seed_fixel_otsu_factor,
                    not_end_fixel_otsu_factor: *not_end_fixel_otsu_factor,
                    max_reference_points: *max_reference_points,
                };
                $body
            }
            WorkflowNodeKind::DipyTractography {
                step_size_mm,
                max_angle_deg,
                min_len_mm,
                max_len_mm,
                fixel_threshold,
                relative_peak_threshold,
                seeds_per_voxel,
                max_points,
                rng_seed,
                direction_getter,
            } => {
                let $op = dipy_tractography::DipyTractographyOp {
                    step_size_mm: *step_size_mm,
                    max_angle_deg: *max_angle_deg,
                    min_len_mm: *min_len_mm,
                    max_len_mm: *max_len_mm,
                    fixel_threshold: *fixel_threshold,
                    relative_peak_threshold: *relative_peak_threshold,
                    seeds_per_voxel: *seeds_per_voxel,
                    max_points: *max_points,
                    rng_seed: *rng_seed,
                    direction_getter: *direction_getter,
                };
                $body
            }
            WorkflowNodeKind::YehTractography {
                step_size_mm,
                max_angle_deg,
                min_len_mm,
                max_len_mm,
                fixel_threshold,
                smooth_fraction,
                max_points,
                target_streamlines,
                max_seed_attempts,
                rng_seed,
            } => {
                let $op = yeh_tractography::YehTractographyOp {
                    step_size_mm: *step_size_mm,
                    max_angle_deg: *max_angle_deg,
                    min_len_mm: *min_len_mm,
                    max_len_mm: *max_len_mm,
                    fixel_threshold: *fixel_threshold,
                    smooth_fraction: *smooth_fraction,
                    max_points: *max_points,
                    target_streamlines: *target_streamlines,
                    max_seed_attempts: *max_seed_attempts,
                    rng_seed: *rng_seed,
                };
                $body
            }
        }
    };
}

pub(super) fn evaluate(
    kind: &WorkflowNodeKind,
    ctx: &mut EvalCtx<'_, '_>,
) -> WorkflowResult<Vec<super::EvaluatedValue>> {
    with_workflow_op!(kind, |op| op.evaluate(ctx))
}

pub fn validate(kind: &WorkflowNodeKind, env: &super::ValidateCtx) -> Vec<super::Diagnostic> {
    with_workflow_op!(kind, |op| op.validate(env))
}

pub fn fingerprint(kind: &WorkflowNodeKind, ctx: &super::FingerprintCtx) -> super::ContentHash {
    with_workflow_op!(kind, |op| op.fingerprint(ctx))
}

// The four accessors below route the new methods-boilerplate /
// documentation trait methods through the op registry. They have no
// in-tree callers yet — the consumers (methods-report assembly and
// the `trxviz-docgen` crate) land in follow-up chunks. Keep the
// `allow(dead_code)` in place until then.

#[allow(dead_code)]
pub fn citation_keys(kind: &WorkflowNodeKind) -> &'static [&'static str] {
    with_workflow_op!(kind, |op| op.citation_keys())
}

/// Render this op's methods-boilerplate sentence with its parameter
/// values interpolated. Returns `None` if the op contributes no
/// methods prose (sources, display nodes, pure routing). Materializes
/// the sentence to an owned `String` because the trait's `Cow` borrows
/// from the temporary op value constructed inside the dispatch macro.
#[allow(dead_code)]
pub fn boilerplate(kind: &WorkflowNodeKind) -> Option<String> {
    with_workflow_op!(kind, |op| op.boilerplate().map(|c| c.into_owned()))
}

#[allow(dead_code)]
pub fn describe(kind: &WorkflowNodeKind) -> std::borrow::Cow<'static, str> {
    with_workflow_op!(kind, |op| op.describe())
}

#[allow(dead_code)]
pub fn category(kind: &WorkflowNodeKind) -> super::methods::OpCategory {
    with_workflow_op!(kind, |op| op.category())
}

pub(super) fn title(kind: &WorkflowNodeKind) -> &'static str {
    with_workflow_op!(kind, |op| op.title())
}

pub(super) fn input_ports(kind: &WorkflowNodeKind) -> Option<&'static [PortKind]> {
    match kind {
        WorkflowNodeKind::SurfaceOverlayStack { .. } => None,
        _ => Some(with_workflow_op!(kind, |op| op.input_ports())),
    }
}

pub(super) fn output_ports(kind: &WorkflowNodeKind) -> &'static [PortKind] {
    with_workflow_op!(kind, |op| op.output_ports())
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
        tip_prune::TipPruneOp::default().tag(),
        purifibre::PurifibreOp::default().tag(),
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
            colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
        }
        .tag(),
        color_by_dps::ColorByDpsOp {
            field: super::DpsFieldName::default(),
            colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
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
            opacity: 1.0,
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
            auto_gate_from_otsu: true,
            opacity_gate: OpacityGate::default(),
        }
        .tag(),
        fixel_display::Fixel2DDisplayOp {
            line_width: 0.0,
            opacity: 1.0,
            slab_thickness_mm: crate::units::Millimeters(0.0),
            length_scale: 1.0,
            visible: true,
            auto_gate_from_otsu: true,
            opacity_gate: OpacityGate::default(),
        }
        .tag(),
        odf_glyph_renderer::OdfGlyphRendererOp {
            scale: 1.0,
            subtract_iso: true,
            norm_within_voxel: false,
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
        bundle_boundary::StreamlineDirectionFieldOp {
            voxel_size_mm: crate::units::Millimeters(0.0),
            sphere_lod: 0,
            normalization: crate::data::orientation_field::BoundaryGlyphNormalization::GlobalPeak,
            binning_mode: crate::data::orientation_field::DirectionFieldBinningMode::default(),
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
        roi_ops::RoiFromParcelOp::default().tag(),
        roi_ops::RoiFromVolumeOp::default().tag(),
        roi_ops::RoiFromShapeOp::default().tag(),
        voxel_mask_display::VoxelMaskDisplayOp::default().tag(),
        prepare_hausdorff::PrepareHausdorffPlanOp::default().tag(),
        prepare_simple::PrepareSimplePlanOp::default().tag(),
        plan_add::AddRoiOp.tag(),
        plan_add::AddRoaOp.tag(),
        plan_add::AddEndRegionOp.tag(),
        plan_add::AddNoEndOp.tag(),
        plan_add::AddLimitingOp.tag(),
        plan_add::AddTermOp.tag(),
        dipy_tractography::DipyTractographyOp::default().tag(),
        yeh_tractography::YehTractographyOp::default().tag(),
    ] {
        if tag.is_empty() {
            return Err(WorkflowError::Evaluation(
                "Workflow op registry contains an empty tag".to_string(),
            ));
        }
    }
    Ok(())
}

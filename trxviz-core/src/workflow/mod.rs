//! Workflow document types, evaluation, persistence, and execution helpers.

pub use crate::error::{WorkflowError, WorkflowResult};

pub(super) mod cpu_dipy;
pub(super) mod cpu_yeh;
pub mod draw;
mod eval_inputs;
mod eval_streamlines;
mod eval_surface;
mod evaluate;
#[cfg(test)]
mod evaluate_tests;
mod fingerprint;
mod graph;
mod jobs;
mod layout;
pub mod methods;
mod op;
pub(crate) mod ops;
pub mod port_labels;
mod project_io;
pub(super) mod purifibre;
mod seed;
mod seed_odx;
mod simple_workflow;
pub(super) mod tip;
pub(crate) mod tracking;
pub(crate) mod tracking_filters;
mod types;

pub use evaluate::{evaluate_scene_plan, evaluate_scene_plan_with_mode, save_streamline_plan};
pub use fingerprint::{
    workflow_boundary_plan_fingerprint, workflow_bundle_display_fingerprint,
    workflow_bundle_plan_fingerprint, workflow_reactive_streamline_fingerprint,
    workflow_sample_volume_along_streamline_fingerprint, workflow_streamline_fingerprint,
    workflow_surface_overlay_fingerprint, workflow_surface_projection_fingerprint,
    workflow_surface_query_fingerprint, workflow_triangle_fundus_fingerprint,
};
pub use graph::{GraphPos, GraphRect, InPort, OutPort, Wire, WorkflowGraph, WorkflowNodeEntry};
pub use jobs::{
    bundle_surface_component_flows, bundle_surface_solid_color, mark_expensive_cancelled,
    mark_expensive_success, materialize_flow_gpu, prime_expensive_record, run_workflow_job,
    sync_node_state_from_run_record, workflow_job_kind_title,
};
pub use layout::{
    NodeSize, WorkflowLayoutOptions, WorkflowLayoutResult, apply_workflow_layout,
    estimate_workflow_node_size, estimated_workflow_node_sizes, layout_workflow_graph,
    layout_workflow_graph_subset, weakly_connected_closure,
};
pub use project_io::{
    load_workflow_project_from_path, relativized_document, resolve_document_asset_paths,
    save_workflow_project_to_path,
};
pub use seed::*;
pub use seed_odx::*;
pub use simple_workflow::*;
pub use types::*;

pub use draw::{DrawList, DrawPrimitive, UploadCache, UploadSlot};
pub use op::{ContentHash, Diagnostic, DiagnosticSeverity, FingerprintCtx, ValidateCtx};
pub use ops::fingerprint as fingerprint_op;
pub use ops::pyafq_bundles;
pub use ops::validate as validate_op;
pub use tracking::CancelFlag;

pub(crate) use evaluate::{
    compose_surface_appearance, expect_boundary_field_input, expect_bundle_surface_input,
    expect_cifti_input, expect_surface_appearance_input, surface_display_model_matrix,
};
pub(crate) use evaluate::{evaluate_derived_streamline_plan, expect_streamline_input};
pub(crate) use evaluate::{
    expect_fixel_scalars_input, expect_fixels_input, expect_odf_field_input,
    expect_odx_catalog_input, expect_parcel_selection_input, expect_parcellation_input,
    expect_surface_input, optional_group_selection_input, optional_volume_input,
    resolve_selected_labels, summarize_value, volume_scalars_from_nifti_volume,
};
pub(crate) use op::EvalCtx;
pub use op::WorkflowOp;
pub use ops::{
    AddEndRegionOp, AddGroupsFromParcellationOp, AddLimitingOp, AddNoEndOp, AddRoaOp, AddRoiOp,
    AddTermOp, BoundaryGlyphDisplayOp, BundleSurfaceBuildOp, BundleSurfaceDisplayOp, CiftiSourceOp,
    CiftiStructureOp, ColorByDirectionOp, ColorByDpsOp, ColorByDpvOp, ColorByFixelScalarsOp,
    ColorByGroupOp, DipyTractographyOp, Fixel2DDisplayOp, Fixel3DDisplayOp, GroupSelectOp,
    LimitStreamlinesOp, MergeOp, MetaGroupSelectOp, OdfGlyphRendererOp, OdxFixelScalarSelectOp,
    OdxSourceOp, OdxVolumeSelectOp, ParcelCropOp, ParcelEndOp, ParcelRoaOp, ParcelRoiOp,
    ParcelSelectOp, ParcelSurfaceBuildOp, ParcellationDisplayOp, ParcellationSourceOp,
    PrepareHausdorffPlanOp, PreparePyafqPlanOp, PrepareSimplePlanOp, PurifibreOp, RandomSubsetOp,
    RemoveDuplicatesOp, RoiFromParcelOp, RoiFromShapeOp, RoiFromVolumeOp, RoiShape,
    SampleVolumeAlongStreamlineOp, SaveStreamlinesOp, SphereQueryOp, StreamlineDirectionFieldOp,
    StreamlineDisplayOp, StreamlineSourceOp, SurfaceDepthQueryOp, SurfaceDisplayOp,
    SurfaceOverlayStackOp, SurfaceProjectionDensityOp, SurfaceProjectionMeanDpsOp, SurfaceSourceOp,
    TipPruneOp, TriangleFundusOp, UniformColorOp, VolumeDisplayOp, VolumeOverlayStackOp,
    VolumeSourceOp, VoxelMaskDisplayOp, WorkflowNodeKind, YehTractographyOp,
};
pub(crate) use types::{EvaluatedValue, WorkflowValue};

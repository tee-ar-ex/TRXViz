use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use trx_rs::Tractogram;

use super::*;
use crate::data::cifti::{CiftiStructure, ScalarKind, SurfaceScalars};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{LoadedTrx, StreamlineBacking};
use crate::data::trx_data::{ColorMode, TrxGpuData};
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::scene::LoadedGiftiSurface;
use crate::units::{Millimeters, StreamlineIndex};

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

fn test_streamline_flow() -> StreamlineFlow {
    let mut tractogram = Tractogram::new();
    for start in [0.0, 2.0, 4.0, 6.0] {
        tractogram
            .push_streamline(&[[start, 0.0, 0.0], [start + 1.0, 0.5, 0.25]])
            .expect("streamline");
    }
    let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).expect("gpu data"));
    StreamlineFlow {
        dataset: Arc::new(StreamlineDataset {
            name: "test".to_string(),
            gpu_data,
            backing: StreamlineBacking::Imported(Arc::new(tractogram)),
        }),
        selected_streamlines: (0..4).map(StreamlineIndex).collect(),
        color_mode: ColorMode::DirectionRgb,
        scalar_auto_range: true,
        scalar_range_min: 0.0,
        scalar_range_max: 1.0,
    }
}

fn evaluate_streamline_op(op: &impl WorkflowOp, node_op: WorkflowNodeKind, flow: StreamlineFlow) -> StreamlineFlow {
    let node = WorkflowNode {
        uuid: WorkflowNodeUuid(1),
        label: op.title().to_string(),
        op: node_op,
    };
    let inputs = vec![Some(EvaluatedValue {
        value: WorkflowValue::Streamline(flow),
        stale: false,
    })];
    let streamline_assets = HashMap::new();
    let volume_assets = HashMap::new();
    let cifti_assets = HashMap::new();
    let surface_assets = HashMap::new();
    let parcellation_assets = HashMap::new();
    let odx_assets = HashMap::new();
    let mut display_ids = HashMap::new();
    let mut next_draw_id = 1;
    let mut scene_plan = SceneFramePlan::default();
    let mut projection_by_surface = HashMap::new();
    let mut save_targets = HashMap::new();
    let mut execution_cache = WorkflowExecutionCache::default();
    let mut node_state = NodeEvalState::default();
    let mut ctx = EvalCtx {
        node: &node,
        inputs: &inputs,
        streamline_assets: &streamline_assets,
        volume_assets: &volume_assets,
        cifti_assets: &cifti_assets,
        surface_assets: &surface_assets,
        parcellation_assets: &parcellation_assets,
        odx_assets: &odx_assets,
        display_ids: &mut display_ids,
        next_draw_id: &mut next_draw_id,
        scene_plan: &mut scene_plan,
        projection_by_surface: &mut projection_by_surface,
        save_targets: &mut save_targets,
        execution_cache: &mut execution_cache,
        node_state: &mut node_state,
    };
    let outputs = op.evaluate(&mut ctx).expect("streamline op output");
    match outputs.into_iter().next().expect("first output").value {
        WorkflowValue::Streamline(flow) => flow,
        _ => panic!("expected streamline output"),
    }
}

#[test]
fn limit_streamlines_preserves_order_with_owned_selection_vec() {
    let flow = evaluate_streamline_op(
        &LimitStreamlinesOp {
            limit: 2,
            randomize: false,
            seed: 99,
        },
        WorkflowNodeKind::LimitStreamlines {
            limit: 2,
            randomize: false,
            seed: 99,
        },
        test_streamline_flow(),
    );

    assert_eq!(
        flow.selected_streamlines,
        vec![StreamlineIndex(0), StreamlineIndex(1)]
    );
}

#[test]
fn random_subset_is_deterministic_and_reuses_dataset_boundary() {
    let upstream = evaluate_streamline_op(
        &LimitStreamlinesOp {
            limit: 3,
            randomize: false,
            seed: 1,
        },
        WorkflowNodeKind::LimitStreamlines {
            limit: 3,
            randomize: false,
            seed: 1,
        },
        test_streamline_flow(),
    );
    let first = evaluate_streamline_op(
        &RandomSubsetOp { limit: 2, seed: 17 },
        WorkflowNodeKind::RandomSubset { limit: 2, seed: 17 },
        upstream.clone(),
    );
    let second = evaluate_streamline_op(
        &RandomSubsetOp { limit: 2, seed: 17 },
        WorkflowNodeKind::RandomSubset { limit: 2, seed: 17 },
        upstream.clone(),
    );

    assert_eq!(first.selected_streamlines, second.selected_streamlines);
    assert_eq!(first.selected_streamlines.len(), 2);
    assert!(Arc::ptr_eq(&first.dataset, &upstream.dataset));
    assert!(Arc::ptr_eq(&second.dataset, &upstream.dataset));
}

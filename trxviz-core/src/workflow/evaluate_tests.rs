use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use trx_rs::Tractogram;

use super::*;
use crate::data::cifti::{CiftiStructure, ScalarKind, SurfaceScalars};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{LoadedTrx, StreamlineBacking};
use crate::data::trx_data::TrxGpuData;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::scene::LoadedGiftiSurface;
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

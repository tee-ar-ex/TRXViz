/// Integration tests for the workflow evaluator.
///
/// These tests exercise `evaluate_scene_plan_with_mode` end-to-end using
/// in-memory tractograms — no GPU, no file I/O required.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use trx_rs::Tractogram;
use trxviz_core::data::loaded_files::{LoadedTrx, StreamlineBacking};
use trxviz_core::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use trxviz_core::units::{Millimeters, StreamlineIndex};
use trxviz_core::workflow::{
    GraphPos, GroupFilter, InPort, OutPort, WorkflowEvalMode, WorkflowExecutionCache,
    WorkflowNodeKind, default_document, evaluate_scene_plan_with_mode, make_node,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn two_point_tractogram() -> Tractogram {
    let mut t = Tractogram::new();
    t.push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        .expect("push streamline");
    t
}

fn tractogram_with_n_streamlines(n: usize) -> Tractogram {
    let mut t = Tractogram::new();
    for i in 0..n {
        let x = i as f32;
        t.push_streamline(&[[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0]])
            .expect("push streamline");
    }
    t
}

fn loaded_trx(id: usize, tractogram: Tractogram) -> LoadedTrx {
    let gpu_data =
        Arc::new(TrxGpuData::from_tractogram(&tractogram).expect("gpu data from tractogram"));
    LoadedTrx {
        id,
        name: format!("asset_{id}"),
        path: PathBuf::from(format!("asset_{id}.trx")),
        data: gpu_data,
        backing: Some(StreamlineBacking::Imported(Arc::new(tractogram))),
        import_warnings: Vec::new(),
    }
}

fn eval(
    document: &trxviz_core::workflow::WorkflowDocument,
    streamline_assets: &[LoadedTrx],
) -> trxviz_core::workflow::WorkflowRuntime {
    evaluate_scene_plan_with_mode(
        document,
        streamline_assets,
        &[],
        &[],
        &[],
        &[],
        &[],
        &mut HashMap::new(),
        &mut 1_000_000usize,
        &mut WorkflowExecutionCache::default(),
        WorkflowEvalMode::Interactive,
    )
}

fn connect(
    doc: &mut trxviz_core::workflow::WorkflowDocument,
    from: trxviz_core::workflow::WorkflowNodeUuid,
    from_output: usize,
    to: trxviz_core::workflow::WorkflowNodeUuid,
    to_input: usize,
) {
    doc.graph.connect(
        OutPort {
            node: from,
            output: from_output,
        },
        InPort {
            node: to,
            input: to_input,
        },
    );
}

// ---------------------------------------------------------------------------
// Source nodes
// ---------------------------------------------------------------------------

#[test]
fn streamline_source_missing_asset_records_error() {
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 99 },
        GraphPos::new(0.0, 0.0),
    );

    let runtime = eval(&doc, &[]);

    let state = runtime.node_state.get(&src).expect("node state present");
    assert!(state.error.is_some(), "expected error for missing asset");
}

#[test]
fn streamline_source_present_asset_has_no_error() {
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );

    let runtime = eval(&doc, &[trx]);

    let state = runtime.node_state.get(&src).expect("node state present");
    assert!(state.error.is_none(), "unexpected error: {:?}", state.error);
}

// ---------------------------------------------------------------------------
// LimitStreamlines
// ---------------------------------------------------------------------------

#[test]
fn limit_streamlines_reduces_count() {
    let trx = loaded_trx(0, tractogram_with_n_streamlines(10));
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let limit = make_node(
        &mut doc,
        WorkflowNodeKind::LimitStreamlines {
            limit: 3,
            randomize: false,
            seed: 0,
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, limit, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, limit, 0, display, 0);

    let runtime = eval(&doc, &[trx]);

    assert_eq!(runtime.scene_plan.streamline_draws.len(), 1);
    let draw = &runtime.scene_plan.streamline_draws[0];
    assert_eq!(draw.flow.selected_streamlines.len(), 3);
}

#[test]
fn limit_streamlines_zero_produces_empty_selection() {
    let trx = loaded_trx(0, tractogram_with_n_streamlines(5));
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let limit = make_node(
        &mut doc,
        WorkflowNodeKind::LimitStreamlines {
            limit: 0,
            randomize: false,
            seed: 0,
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, limit, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, limit, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert_eq!(
        runtime.scene_plan.streamline_draws[0]
            .flow
            .selected_streamlines
            .len(),
        0
    );
}

// ---------------------------------------------------------------------------
// RandomSubset
// ---------------------------------------------------------------------------

#[test]
fn random_subset_is_deterministic_for_same_seed() {
    let n = 20;
    let trx_a = loaded_trx(0, tractogram_with_n_streamlines(n));
    let trx_b = loaded_trx(0, tractogram_with_n_streamlines(n));

    let build_doc = || {
        let mut doc = default_document();
        let src = make_node(
            &mut doc,
            WorkflowNodeKind::StreamlineSource { source_id: 0 },
            GraphPos::new(0.0, 0.0),
        );
        let subset = make_node(
            &mut doc,
            WorkflowNodeKind::RandomSubset { limit: 5, seed: 42 },
            GraphPos::new(200.0, 0.0),
        );
        connect(&mut doc, src, 0, subset, 0);
        let display = make_node(
            &mut doc,
            WorkflowNodeKind::StreamlineDisplay {
                enabled: true,
                render_style: RenderStyle::Flat,
                tube_radius_mm: Millimeters(0.5),
                tube_sides: 8,
                slab_half_width_mm: Millimeters(5.0),
            },
            GraphPos::new(400.0, 0.0),
        );
        connect(&mut doc, subset, 0, display, 0);
        doc
    };

    let doc_a = build_doc();
    let doc_b = build_doc();
    let runtime_a = eval(&doc_a, &[trx_a]);
    let runtime_b = eval(&doc_b, &[trx_b]);

    let sel_a = &runtime_a.scene_plan.streamline_draws[0]
        .flow
        .selected_streamlines;
    let sel_b = &runtime_b.scene_plan.streamline_draws[0]
        .flow
        .selected_streamlines;
    assert_eq!(sel_a.len(), 5);
    assert_eq!(sel_a, sel_b, "same seed must produce same subset");
}

// ---------------------------------------------------------------------------
// GroupSelect
// ---------------------------------------------------------------------------

#[test]
fn group_select_all_passes_through_unchanged() {
    let trx = loaded_trx(0, tractogram_with_n_streamlines(4));
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let sel = make_node(
        &mut doc,
        WorkflowNodeKind::GroupSelect {
            groups: GroupFilter::All,
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, sel, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, sel, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert_eq!(
        runtime.scene_plan.streamline_draws[0]
            .flow
            .selected_streamlines
            .len(),
        4
    );
}

#[test]
fn group_select_none_produces_empty_flow() {
    let trx = loaded_trx(0, tractogram_with_n_streamlines(4));
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let sel = make_node(
        &mut doc,
        WorkflowNodeKind::GroupSelect {
            groups: GroupFilter::None,
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, sel, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, sel, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert_eq!(
        runtime.scene_plan.streamline_draws[0]
            .flow
            .selected_streamlines
            .len(),
        0
    );
}

#[test]
fn group_select_by_label_filters_correctly() {
    let mut tractogram = tractogram_with_n_streamlines(4);
    // streamlines 0 and 1 belong to "GroupA"; 2 and 3 to "GroupB"
    tractogram.insert_group("GroupA", vec![0, 1]);
    tractogram.insert_group("GroupB", vec![2, 3]);
    let trx = loaded_trx(0, tractogram);

    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let sel = make_node(
        &mut doc,
        WorkflowNodeKind::GroupSelect {
            groups: GroupFilter::from_csv("GroupA"),
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, sel, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, sel, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    let selected = &runtime.scene_plan.streamline_draws[0]
        .flow
        .selected_streamlines;
    assert_eq!(selected.len(), 2);
    assert!(selected.contains(&StreamlineIndex(0)));
    assert!(selected.contains(&StreamlineIndex(1)));
}

// ---------------------------------------------------------------------------
// Color nodes
// ---------------------------------------------------------------------------

#[test]
fn color_by_direction_sets_direction_mode() {
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let color = make_node(
        &mut doc,
        WorkflowNodeKind::ColorByDirection,
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, color, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, color, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert!(matches!(
        runtime.scene_plan.streamline_draws[0].flow.color_mode,
        ColorMode::DirectionRgb
    ));
}

#[test]
fn color_by_group_sets_group_mode() {
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let color = make_node(
        &mut doc,
        WorkflowNodeKind::ColorByGroup,
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, color, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, color, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert!(matches!(
        runtime.scene_plan.streamline_draws[0].flow.color_mode,
        ColorMode::Group
    ));
}

#[test]
fn uniform_color_sets_uniform_mode() {
    let red = [1.0_f32, 0.0, 0.0, 1.0];
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let color = make_node(
        &mut doc,
        WorkflowNodeKind::UniformColor { color: red },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, color, 0);
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(400.0, 0.0),
    );
    connect(&mut doc, color, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert!(matches!(
        runtime.scene_plan.streamline_draws[0].flow.color_mode,
        ColorMode::Uniform(c) if c == red
    ));
}

// ---------------------------------------------------------------------------
// StreamlineDisplay
// ---------------------------------------------------------------------------

#[test]
fn streamline_display_adds_entry_to_scene_plan() {
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: true,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert_eq!(runtime.scene_plan.streamline_draws.len(), 1);
    assert!(runtime.scene_plan.streamline_draws[0].visible);
}

#[test]
fn hidden_streamline_display_still_adds_entry_but_visible_false() {
    let trx = loaded_trx(0, two_point_tractogram());
    let mut doc = default_document();
    let src = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let display = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineDisplay {
            enabled: false,
            render_style: RenderStyle::Flat,
            tube_radius_mm: Millimeters(0.5),
            tube_sides: 8,
            slab_half_width_mm: Millimeters(5.0),
        },
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, src, 0, display, 0);

    let runtime = eval(&doc, &[trx]);
    assert_eq!(runtime.scene_plan.streamline_draws.len(), 1);
    assert!(!runtime.scene_plan.streamline_draws[0].visible);
}

// ---------------------------------------------------------------------------
// Merge (reactive plan)
// ---------------------------------------------------------------------------

#[test]
fn merge_queues_one_reactive_plan() {
    let trx_a = loaded_trx(0, two_point_tractogram());
    let trx_b = loaded_trx(1, two_point_tractogram());
    let mut doc = default_document();
    let src_a = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 0 },
        GraphPos::new(0.0, 0.0),
    );
    let src_b = make_node(
        &mut doc,
        WorkflowNodeKind::StreamlineSource { source_id: 1 },
        GraphPos::new(0.0, 200.0),
    );
    let merge = make_node(
        &mut doc,
        WorkflowNodeKind::Merge,
        GraphPos::new(200.0, 100.0),
    );
    connect(&mut doc, src_a, 0, merge, 0);
    connect(&mut doc, src_b, 0, merge, 1);

    let runtime = eval(&doc, &[trx_a, trx_b]);
    assert_eq!(runtime.scene_plan.reactive_streamline_plans.len(), 1);
}

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

#[test]
fn disconnected_color_node_records_error() {
    let mut doc = default_document();
    let _color = make_node(
        &mut doc,
        WorkflowNodeKind::ColorByDirection,
        GraphPos::new(0.0, 0.0),
    );

    let runtime = eval(&doc, &[]);
    let any_error = runtime.node_state.values().any(|s| s.error.is_some());
    assert!(any_error, "disconnected color node should record an error");
}

#[test]
fn cycle_in_graph_sets_graph_error() {
    let mut doc = default_document();
    let a = make_node(
        &mut doc,
        WorkflowNodeKind::ColorByDirection,
        GraphPos::new(0.0, 0.0),
    );
    let b = make_node(
        &mut doc,
        WorkflowNodeKind::ColorByGroup,
        GraphPos::new(200.0, 0.0),
    );
    connect(&mut doc, a, 0, b, 0);
    connect(&mut doc, b, 0, a, 0);

    let runtime = eval(&doc, &[]);
    assert!(
        runtime.graph_error.is_some(),
        "cyclic graph must set graph_error"
    );
}

// ---------------------------------------------------------------------------
// Performance benchmark (ignored by default — run with `cargo test -- --ignored`)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow benchmark — run explicitly with `-- --ignored`"]
fn bench_recolor_100k_streamlines() {
    let build_doc = |color_kind: WorkflowNodeKind| {
        let mut doc = default_document();
        let src = make_node(
            &mut doc,
            WorkflowNodeKind::StreamlineSource { source_id: 0 },
            GraphPos::new(0.0, 0.0),
        );
        let color = make_node(&mut doc, color_kind, GraphPos::new(200.0, 0.0));
        connect(&mut doc, src, 0, color, 0);
        let display = make_node(
            &mut doc,
            WorkflowNodeKind::StreamlineDisplay {
                enabled: true,
                render_style: RenderStyle::Flat,
                tube_radius_mm: Millimeters(0.5),
                tube_sides: 8,
                slab_half_width_mm: Millimeters(5.0),
            },
            GraphPos::new(400.0, 0.0),
        );
        connect(&mut doc, color, 0, display, 0);
        doc
    };

    for color_kind in [
        WorkflowNodeKind::ColorByDirection,
        WorkflowNodeKind::ColorByGroup,
    ] {
        let doc = build_doc(color_kind);
        let assets = [loaded_trx(0, tractogram_with_n_streamlines(100_000))];
        let start = std::time::Instant::now();
        let runtime = eval(&doc, &assets);
        let elapsed = start.elapsed();
        assert_eq!(
            runtime.scene_plan.streamline_draws[0]
                .flow
                .selected_streamlines
                .len(),
            100_000
        );
        println!("recolor 100k: {elapsed:?}");
    }
}

//! Real-data benchmark/regression test for Yeh tractography.
//!
//! Loads a paired DSI-Studio GQI `.fz` subject + an `autotrack`
//! reference bundle (`.tck.gz`), runs `PrepareHausdorffPlan` +
//! `YehTractography` via the public workflow evaluator + job dispatch,
//! and checks that the output bundle roughly overlaps the reference.
//!
//! The fixtures are not committed to the repo. Resolution order:
//!   1. `TRXVIZ_REAL_TEST_DATA` env var → directory containing the files.
//!   2. Fallback: `/Users/mcieslak/projects/odx/test_data`.
//!
//! Gated with `#[ignore]` so `cargo test` stays fast; run explicitly:
//!
//!     cargo test -p trxviz-core --test yeh_bench_cst --release \
//!         -- --ignored --nocapture

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Vec3;

use trxviz_core::asset_loader::AssetLoader;
use trxviz_core::data::loaded_files::{LoadedOdx, LoadedTrx};
use trxviz_core::scene::LoadedStreamlineSource;
use trxviz_core::data::odx_data::OdxScene;
use trxviz_core::data::trx_data::TrxGpuData;
use trxviz_core::gpu::plan_prep::hausdorff::{
    HausdorffPlanParams, build_hausdorff_plan,
};
use trxviz_core::units::Millimeters;
use trxviz_core::workflow::{
    GraphPos, WorkflowEvalMode, WorkflowExecutionCache, WorkflowJobOutput,
    WorkflowJobPayload, WorkflowNodeKind, default_document,
    evaluate_scene_plan_with_mode, make_node, run_workflow_job,
};

const ODX_FILE: &str = "sub-20124_ses-1_space-ACPC_desc-preproc_dwi.gqi.fz";
const REF_TCK: &str =
    "sub-20124_ses-1_space-ACPC_model-gqi_bundle-ProjectionBrainstemCorticospinalTractL_streamlines.tck.gz";

/// Resolve the fixture directory, or `None` when neither candidate is
/// present. Handled gracefully so the test `return`s cleanly instead of
/// hard-failing on CI machines without the data.
fn fixture_dir() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = std::env::var_os("TRXVIZ_REAL_TEST_DATA")
        .map(|v| vec![PathBuf::from(v)])
        .unwrap_or_else(|| vec![PathBuf::from("/Users/mcieslak/projects/odx/test_data")]);
    for dir in candidates {
        if dir.join(ODX_FILE).is_file() && dir.join(REF_TCK).is_file() {
            return Some(dir);
        }
    }
    None
}

fn load_odx(path: &Path, id: usize) -> LoadedOdx {
    let scene = OdxScene::load(path).expect("load .fz");
    LoadedOdx {
        id,
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "odx".into()),
        path: path.to_path_buf(),
        scene: Arc::new(scene),
        warnings: Vec::new(),
        visible: true,
    }
}

fn load_trx(path: &Path, id: usize) -> LoadedTrx {
    let source = LoadedStreamlineSource::load(path).expect("load .tck.gz");
    LoadedTrx {
        id,
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "trx".into()),
        path: path.to_path_buf(),
        data: Arc::new(source.data),
        backing: Some(source.backing),
        import_warnings: source.warnings,
    }
}

/// Flatten a `TrxGpuData` into `(points, per_streamline_ranges)` so the
/// similarity helpers can iterate streamlines explicitly.
fn flatten(data: &TrxGpuData) -> (Vec<[f32; 3]>, Vec<std::ops::Range<usize>>) {
    let points = data.positions.clone();
    let ranges = data
        .offsets
        .windows(2)
        .map(|w| (w[0] as usize)..(w[1] as usize))
        .collect();
    (points, ranges)
}

/// Mean per-point minimum distance from `streamline` to the reference
/// point cloud. This is the per-streamline score the `PostFilter::Hausdorff`
/// applies; we compute the *value* here rather than a pass/fail bool.
fn mean_min_distance(streamline: &[[f32; 3]], reference_points: &[[f32; 3]]) -> f32 {
    if streamline.is_empty() || reference_points.is_empty() {
        return f32::INFINITY;
    }
    let mut acc = 0.0f32;
    for p in streamline {
        let p = Vec3::from_array(*p);
        let mut min_d2 = f32::INFINITY;
        for q in reference_points {
            let d2 = (Vec3::from_array(*q) - p).length_squared();
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        acc += min_d2.sqrt();
    }
    acc / streamline.len() as f32
}

/// Summed segment-length of each streamline, in mm.
fn streamline_length(streamline: &[[f32; 3]]) -> f32 {
    streamline
        .windows(2)
        .map(|w| (Vec3::from_array(w[1]) - Vec3::from_array(w[0])).length())
        .sum()
}

/// Stride-subsample a flat point list down to ≤ `max_points`.
fn subsample(points: &[[f32; 3]], max_points: usize) -> Vec<[f32; 3]> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max_points);
    points.iter().copied().step_by(stride).collect()
}

#[test]
#[ignore]
fn yeh_benchmark_cst_l() {
    // Enable `log::info!` output from `cpu_yeh` so `RUST_LOG=info` works.
    let _ = env_logger::builder().is_test(true).try_init();

    let Some(dir) = fixture_dir() else {
        eprintln!(
            "[yeh-bench] skipping: fixture not found. \
             Set TRXVIZ_REAL_TEST_DATA or place {ODX_FILE} + {REF_TCK} \
             under /Users/mcieslak/projects/odx/test_data."
        );
        return;
    };
    let odx_path = dir.join(ODX_FILE);
    let ref_path = dir.join(REF_TCK);

    // ── load fixtures ───────────────────────────────────────────────
    let odx_id: usize = 1;
    let ref_id: usize = 2;
    let odx_asset = load_odx(&odx_path, odx_id);
    let ref_asset = load_trx(&ref_path, ref_id);

    let ref_scene = odx_asset.scene.clone();
    let ref_gpu = ref_asset.data.clone();

    let ref_n = ref_gpu.offsets.len().saturating_sub(1);
    let (ref_points_flat, ref_ranges) = flatten(&ref_gpu);
    eprintln!(
        "[yeh-bench] loaded ref bundle: {} streamlines, {} total points",
        ref_n,
        ref_points_flat.len()
    );

    // ── build a Hausdorff plan directly (we only need its outputs as
    //    constraints on the Yeh tracker; the full workflow path is
    //    overkill for test setup) ─────────────────────────────────────
    // Resolve the tracking-metric Otsu up front so the plan builds its
    // masks from data-driven thresholds rather than hardcoded ones.
    let fixel_otsu = ref_scene
        .default_fixel_otsu()
        .expect("ODX has a resolvable tracking metric")
        .clone();
    let params = HausdorffPlanParams {
        tolerance_mm: 12.0,
        seed_tolerance_mm: 2.0,
        tracking_metric: fixel_otsu.metric_name.clone(),
        fixel_otsu: fixel_otsu.threshold,
        seed_fixel_otsu_factor: 0.5,
        not_end_fixel_otsu_factor: 0.9,
        max_reference_points: 20_000,
    };
    let haus = build_hausdorff_plan(
        &ref_scene,
        &ref_gpu,
        &[],
        "CST-L reference".into(),
        &params,
    );
    eprintln!(
        "[yeh-bench] Hausdorff plan: seed={} voxels, limiting={} voxels, \
         no_end={} voxels, min_len={:?}mm, max_len={:?}mm",
        haus.seed_mask.count(),
        haus.limiting_mask.count(),
        haus.no_end_mask.count(),
        haus.plan.min_len_mm,
        haus.plan.max_len_mm,
    );

    // ── drive Yeh through the workflow evaluator + job dispatch ─────
    let mut document = default_document();
    let odx_src = make_node(
        &mut document,
        WorkflowNodeKind::OdxSource { source_id: odx_id },
        GraphPos::new(0.0, 0.0),
    );
    let yeh = make_node(
        &mut document,
        WorkflowNodeKind::YehTractography {
            step_size_mm: 1.0,
            max_angle_deg: 60.0,
            min_len_mm: 10.0,
            max_len_mm: 300.0,
            fixel_threshold: 0.05,
            smooth_fraction: 0.25,
            max_points: 501,
            target_streamlines: 5_000,
            max_seed_attempts: 1_000_000,
            rng_seed: 42,
        },
        GraphPos::new(400.0, 0.0),
    );
    // Fixels port 0 of OdxSource → Yeh input 0.
    document.graph.connect(
        trxviz_core::workflow::OutPort { node: odx_src, output: 0 },
        trxviz_core::workflow::InPort { node: yeh, input: 0 },
    );
    // Yeh input 1 (VoxelMask seed) + 2 (TrackingPlan) are unconnected;
    // we instead *inject* the Hausdorff-derived plan into the emitted
    // YehTractographyPlan below. (Wiring PrepareHausdorffPlan through
    // the graph would require a StreamlineSource for the reference,
    // which this test already has in hand as a `LoadedTrx`.)

    let runtime = evaluate_scene_plan_with_mode(
        &document,
        &[ref_asset],
        &[],
        &[],
        &[],
        &[],
        &[odx_asset],
        &mut HashMap::new(),
        &mut 1_000_000usize,
        &mut WorkflowExecutionCache::default(),
        WorkflowEvalMode::Interactive,
    );

    let mut plans = runtime.scene_plan.yeh_tractography_plans;
    assert_eq!(plans.len(), 1, "exactly one yeh plan should be queued");
    let mut plan = plans.remove(0);

    // Graft the Hausdorff-derived seed + post-hoc filter onto the Yeh
    // plan. We deliberately leave `limiting_mask` / `no_end_mask`
    // unset: the bench wants to verify Yeh can *produce* plausible
    // streamlines and the post-hoc Hausdorff filter enforces proximity
    // to the reference. A tighter per-step limiting tube would require
    // a curvature-aware tracker we don't fully implement.
    plan.seed_mask = Some(haus.seed_mask.clone());
    plan.post_filter = haus.plan.post_filter.clone();
    if let Some(v) = haus.plan.min_len_mm {
        plan.min_len_mm = v;
    }
    if let Some(v) = haus.plan.max_len_mm {
        plan.max_len_mm = v;
    }

    let t0 = std::time::Instant::now();
    let output = run_workflow_job(WorkflowJobPayload::YehTractography { plan })
        .expect("yeh job ran");
    let elapsed = t0.elapsed();

    let flow = match output {
        WorkflowJobOutput::YehTractography { flow } => flow,
        other => panic!("expected YehTractography output, got {:?}", std::mem::discriminant(&other)),
    };

    let kept = flow.selected_streamlines.len();
    let gpu = &flow.dataset.gpu_data;
    let (cand_points, cand_ranges) = flatten(gpu);

    // ── similarity scoring ───────────────────────────────────────────
    let ref_points_sub = subsample(&ref_points_flat, 20_000);

    // Candidate streamlines: mean-min distance to reference.
    let mut forward_distances: Vec<f32> = Vec::with_capacity(cand_ranges.len());
    let mut lengths: Vec<f32> = Vec::with_capacity(cand_ranges.len());
    for range in &cand_ranges {
        let sl = &cand_points[range.clone()];
        forward_distances.push(mean_min_distance(sl, &ref_points_sub));
        lengths.push(streamline_length(sl));
    }
    let forward_mean = if forward_distances.is_empty() {
        f32::INFINITY
    } else {
        forward_distances.iter().sum::<f32>() / forward_distances.len() as f32
    };

    // Reference streamlines: for each, min distance to any candidate.
    // Subsample candidate points so this is O(ref_streamlines × cand_points_sub).
    let cand_points_sub = subsample(&cand_points, 20_000);
    let cov_tolerance = 12.0f32;
    let mut ref_lengths: Vec<f32> = Vec::with_capacity(ref_ranges.len());
    let mut ref_mean_min: Vec<f32> = Vec::with_capacity(ref_ranges.len());
    for range in &ref_ranges {
        let sl = &ref_points_flat[range.clone()];
        ref_lengths.push(streamline_length(sl));
        ref_mean_min.push(mean_min_distance(sl, &cand_points_sub));
    }
    let covered = ref_mean_min
        .iter()
        .filter(|&&d| d.is_finite() && d <= cov_tolerance)
        .count();
    let coverage_frac = if ref_mean_min.is_empty() {
        0.0
    } else {
        covered as f32 / ref_mean_min.len() as f32
    };

    let median_cand_len = median(&mut lengths.clone());
    let ref_min_len = ref_lengths
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let ref_max_len = ref_lengths.iter().copied().fold(0.0f32, f32::max);

    eprintln!(
        "[yeh-bench] CST-L: {}/{} streamlines kept in {:.1}s\n\
         \tforward_mean={:.1}mm  coverage@{}mm={:.1}%  median_len={:.1}mm\n\
         \tref_streamlines={}  ref_length_range=[{:.1}, {:.1}]mm",
        kept,
        5_000,
        elapsed.as_secs_f32(),
        forward_mean,
        cov_tolerance as u32,
        coverage_frac * 100.0,
        median_cand_len,
        ref_n,
        ref_min_len,
        ref_max_len,
    );

    // ── conservative assertions ──────────────────────────────────────
    assert!(
        kept >= 200,
        "yeh produced only {kept} streamlines; filter likely too aggressive"
    );
    assert!(
        forward_mean <= 16.0,
        "forward Hausdorff mean {:.2} exceeds filter cap 16mm — post_filter not applied?",
        forward_mean
    );
    assert!(
        coverage_frac >= 0.25,
        "reference coverage {:.1}% below 25% threshold at {}mm",
        coverage_frac * 100.0,
        cov_tolerance as u32
    );

    // `Millimeters` import is kept for parity with other tests even though
    // we don't strictly need it here.
    let _ = Millimeters(0.0);
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

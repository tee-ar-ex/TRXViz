//! Real-data benchmark/regression test for GPU PTT (Aydogan & Shi 2021)
//! tractography. Mirror of `dipy_bench_cst.rs` but with
//! `direction_getter = Ptt`.
//!
//! Loads an SH-bearing ODX subject + a reference bundle (`.tck.gz`),
//! builds a `PrepareHausdorffPlan` from the reference, runs
//! `DipyTractography` (PTT variant) on the GPU via the public workflow
//! evaluator + job dispatch, and checks that the output bundle roughly
//! overlaps the reference.
//!
//! Skip behaviors (in order):
//!   1. No fixture files found → skip with helpful message.
//!   2. ODX has no SH coefficients → skip.
//!   3. No GPU adapter available (headless CI, etc.) → skip.
//!
//! Fixture resolution mirrors `dipy_bench_cst.rs`:
//!   - Env vars `TRXVIZ_DIPY_ODX` + `TRXVIZ_DIPY_REF_TCK` (absolute paths)
//!   - Env var `TRXVIZ_REAL_TEST_DATA` directory
//!   - Fallback: `/Users/mcieslak/projects/odx/test_data`.
//!
//! Gated with `#[ignore]`. Run explicitly:
//!
//!     cargo test -p trxviz-core --test dipy_ptt_bench_cst --release \
//!         -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Vec3;

use trxviz_core::asset_loader::AssetLoader;
use trxviz_core::data::loaded_files::{LoadedOdx, LoadedTrx};
use trxviz_core::data::odx_data::OdxScene;
use trxviz_core::data::trx_data::TrxGpuData;
use trxviz_core::gpu::plan_prep::hausdorff::{HausdorffPlanParams, build_hausdorff_plan};
use trxviz_core::scene::LoadedStreamlineSource;
use trxviz_core::workflow::{
    DipyDirectionGetter, DipyTractographyPlan, WorkflowJobOutput, WorkflowJobPayload,
    WorkflowNodeUuid, run_workflow_job,
};

const DEFAULT_ODX_FILE: &str = "sub-20124_ses-1_space-ACPC_desc-preproc_dwi.gqi.fz";
const DEFAULT_REF_TCK: &str = "sub-20124_ses-1_space-ACPC_model-gqi_bundle-ProjectionBrainstemCorticospinalTractL_streamlines.tck.gz";

fn fixture_paths() -> Option<(PathBuf, PathBuf)> {
    if let (Some(odx), Some(tck)) = (
        std::env::var_os("TRXVIZ_DIPY_ODX"),
        std::env::var_os("TRXVIZ_DIPY_REF_TCK"),
    ) {
        let odx = PathBuf::from(odx);
        let tck = PathBuf::from(tck);
        if odx.is_file() && tck.is_file() {
            return Some((odx, tck));
        }
        return None;
    }
    let dir = std::env::var_os("TRXVIZ_REAL_TEST_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/mcieslak/projects/odx/test_data"));
    let odx = dir.join(DEFAULT_ODX_FILE);
    let tck = dir.join(DEFAULT_REF_TCK);
    if odx.is_file() && tck.is_file() {
        Some((odx, tck))
    } else {
        None
    }
}

fn load_odx(path: &Path, id: usize) -> LoadedOdx {
    let scene = OdxScene::load(path).expect("load ODX");
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

fn flatten(data: &TrxGpuData) -> (Vec<[f32; 3]>, Vec<std::ops::Range<usize>>) {
    let points = data.positions.clone();
    let ranges = data
        .offsets
        .windows(2)
        .map(|w| (w[0] as usize)..(w[1] as usize))
        .collect();
    (points, ranges)
}

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

fn streamline_length(streamline: &[[f32; 3]]) -> f32 {
    streamline
        .windows(2)
        .map(|w| (Vec3::from_array(w[1]) - Vec3::from_array(w[0])).length())
        .sum()
}

fn subsample(points: &[[f32; 3]], max_points: usize) -> Vec<[f32; 3]> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max_points);
    points.iter().copied().step_by(stride).collect()
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

/// Try to acquire a wgpu device. Returns `None` when no GPU adapter is
/// available — bench skips cleanly in that case.
fn try_acquire_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ptt_bench_device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((device, queue))
}

#[test]
#[ignore]
fn dipy_ptt_benchmark_cst_l() {
    let _ = env_logger::builder().is_test(true).try_init();

    let Some((odx_path, ref_path)) = fixture_paths() else {
        eprintln!(
            "[ptt-bench] skipping: fixture not found.\n\
             Set TRXVIZ_DIPY_ODX + TRXVIZ_DIPY_REF_TCK to point at an\n\
             SH-bearing ODX subject + a matching .tck.gz reference bundle."
        );
        return;
    };

    let odx_id: usize = 1;
    let ref_id: usize = 2;
    let odx_asset = load_odx(&odx_path, odx_id);
    let ref_asset = load_trx(&ref_path, ref_id);

    if odx_asset.scene.sh_view_f32().is_none() {
        eprintln!(
            "[ptt-bench] skipping: ODX '{}' has no SH coefficients.\n\
             PTT requires an SH (CSD/SS3T/MAPMRI-derived) ODX.\n\
             Override TRXVIZ_DIPY_ODX to point at an SH-bearing fixture.",
            odx_path.display(),
        );
        return;
    }

    let Some((device, queue)) = try_acquire_gpu() else {
        eprintln!("[ptt-bench] skipping: no GPU adapter available.");
        return;
    };
    eprintln!("[ptt-bench] GPU adapter acquired.");

    let ref_scene = odx_asset.scene.clone();
    let ref_gpu = ref_asset.data.clone();

    let ref_n = ref_gpu.offsets.len().saturating_sub(1);
    let (ref_points_flat, ref_ranges) = flatten(&ref_gpu);
    eprintln!(
        "[ptt-bench] loaded ref bundle: {} streamlines, {} total points",
        ref_n,
        ref_points_flat.len()
    );

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
    let haus = build_hausdorff_plan(&ref_scene, &ref_gpu, &[], "CST-L reference".into(), &params);
    eprintln!(
        "[ptt-bench] Hausdorff plan: seed={} voxels, limiting={} voxels, no_end={} voxels",
        haus.seed_mask.count(),
        haus.limiting_mask.count(),
        haus.no_end_mask.count(),
    );

    // ── construct DipyTractographyPlan directly ──────────────────────
    //
    // We bypass the workflow evaluator here for two reasons:
    //   1. The dipy op requires either a wired VoxelMask input or a
    //      TrackingPlan with a seed_mask, and refuses to queue a plan
    //      otherwise. Wiring those through a graph for a smoke test
    //      would mean fabricating intermediate nodes (StreamlineSource
    //      → PrepareHausdorffPlan → DipyTractography).
    //   2. We already have the Hausdorff-derived seed_mask + post-filter
    //      in hand. Constructing the plan directly is much shorter.
    //
    // (Side-note: the existing dipy_bench_cst.rs has the same latent
    // bug — it wires only port 0 too, but skips before evaluation
    // because its default fixture has no SH coefficients. Worth fixing
    // there too in a follow-up.)
    let _ = ref_asset; // kept loaded for the Hausdorff plan construction above.
    let plan = DipyTractographyPlan {
        node_uuid: WorkflowNodeUuid(0),
        label: "ptt-bench CST-L".into(),
        // Plan constructed by hand rather than through the evaluator,
        // so no op-computed fingerprint. Same story as `dipy_bench_cst.rs`.
        fingerprint: trxviz_core::workflow::ContentHash::EMPTY,
        odx_source_id: odx_id,
        odx_scene: odx_asset.scene.clone(),
        seed_mask: Some(haus.seed_mask.clone()),
        step_size_mm: 0.5,
        max_angle_deg: 30.0,
        min_len_mm: haus.plan.min_len_mm.unwrap_or(10.0),
        max_len_mm: haus.plan.max_len_mm.unwrap_or(300.0),
        fixel_threshold: 0.0,
        relative_peak_threshold: 0.5,
        seeds_per_voxel: 2,
        max_points: 501,
        rng_seed: 42,
        limiting_mask: None,
        roa_mask: None,
        term_mask: None,
        roi_masks: Vec::new(),
        end_masks: Vec::new(),
        no_end_mask: None,
        post_filter: haus.plan.post_filter.clone(),
        fixel_otsu: None,
        direction_getter: DipyDirectionGetter::ptt_default(),
    };
    let n_seeds_planned = plan.seed_mask.as_ref().map_or(0, |m| m.count());

    let t0 = std::time::Instant::now();
    let output = run_workflow_job(
        WorkflowJobPayload::DipyTractography {
            plan,
            device: Some(device),
            queue: Some(queue),
        },
        trxviz_core::workflow::CancelFlag::new(),
    )
    .expect("dipy PTT job ran");
    let elapsed = t0.elapsed();

    let flow = match output {
        WorkflowJobOutput::DipyTractography { flow } => flow,
        other => panic!(
            "expected DipyTractography output, got {:?}",
            std::mem::discriminant(&other)
        ),
    };

    let kept = flow.selected_streamlines.len();
    let gpu = &flow.dataset.gpu_data;
    let (cand_points, cand_ranges) = flatten(gpu);

    let ref_points_sub = subsample(&ref_points_flat, 20_000);
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
    let ref_min_len = ref_lengths.iter().copied().fold(f32::INFINITY, f32::min);
    let ref_max_len = ref_lengths.iter().copied().fold(0.0f32, f32::max);

    eprintln!(
        "[ptt-bench] CST-L: {} streamlines kept from {} seed voxels in {:.1}s\n\
         \tforward_mean={:.1}mm  coverage@{}mm={:.1}%  median_len={:.1}mm\n\
         \tref_streamlines={}  ref_length_range=[{:.1}, {:.1}]mm",
        kept,
        n_seeds_planned,
        elapsed.as_secs_f32(),
        forward_mean,
        cov_tolerance as u32,
        coverage_frac * 100.0,
        median_cand_len,
        ref_n,
        ref_min_len,
        ref_max_len,
    );

    // Looser asserts than dipy/yeh — PTT v0.1 may produce fewer
    // streamlines than the well-tuned probabilistic path. The point of
    // the bench is to verify the GPU pipeline runs end-to-end and the
    // post-hoc Hausdorff filter takes effect.
    assert!(
        kept >= 10,
        "PTT produced only {kept} streamlines; pipeline may be broken"
    );
    assert!(
        forward_mean <= 16.0,
        "forward Hausdorff mean {:.2} exceeds filter cap 16mm — post_filter not applied?",
        forward_mean
    );
}

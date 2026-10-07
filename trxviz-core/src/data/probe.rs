//! Quantitative point queries against a loaded ODX scene: given a world
//! position (from `renderer::picking`), report the voxel's per-voxel
//! scalars, fixels, SH coefficients, and ODF profile. Streamline probing
//! is future work.

use glam::Vec3;

use crate::data::odx_data::OdxScene;

/// One fixel at the probed voxel.
#[derive(Clone, Debug)]
pub struct ProbeFixel {
    /// RAS+ unit direction.
    pub direction: [f32; 3],
    /// `(dpf name, value)` for every scalar (single-column) DPF array.
    pub scalars: Vec<(String, f32)>,
}

/// SH coefficients at the probed voxel (detail query only).
#[derive(Clone, Debug)]
pub struct ProbeSh {
    /// Human-readable basis, e.g. "tournier07 (lmax 8)".
    pub basis: String,
    pub coeffs: Vec<f32>,
    /// Max amplitude over the render sphere, if an evaluator is available.
    pub peak_amplitude: Option<f32>,
    /// Whether evaluated amplitudes are clamped to `>= 0` (the render
    /// path's convention); raw negative lobes are hidden when true.
    pub clamped_nonnegative: bool,
}

/// Sampled ODF amplitudes at the probed voxel (detail query only).
#[derive(Clone, Debug)]
pub struct ProbeOdf {
    /// One amplitude per stored sphere vertex.
    pub amplitudes: Vec<f32>,
    /// Stored domain covers a hemisphere only (renderer mirrors it).
    pub hemisphere: bool,
}

/// Everything known about the voxel under a probed world position.
#[derive(Clone, Debug)]
pub struct OdxProbeReport {
    pub world: [f32; 3],
    /// Rounded voxel grid index in the ODX frame.
    pub ijk: [i64; 3],
    /// Compact (masked) row, `None` when the voxel is outside the mask.
    pub compact_row: Option<u32>,
    /// `(dpv name, value)` for every scalar (single-column) DPV array.
    pub dpv: Vec<(String, f32)>,
    pub fixels: Vec<ProbeFixel>,
    pub sh: Option<ProbeSh>,
    pub odf: Option<ProbeOdf>,
    /// The canonical primary fixel metric present in this file
    /// (`amplitude` → `afd` → `qa`, mirroring odx-rs QC).
    pub primary_fixel_metric: Option<String>,
}

/// Probe the ODX scene at a world position. Returns `None` when the
/// rounded voxel falls outside the volume bounds; out-of-mask voxels
/// inside the bounds yield a report with `compact_row: None` and empty
/// value lists. `want_detail` additionally fills `sh`/`odf` (pinned-probe
/// content — skip on hover, it evaluates the SH basis).
pub fn probe_odx(
    scene: &OdxScene,
    world: Vec3,
    detail: u32,
    want_detail: bool,
) -> Option<OdxProbeReport> {
    let lookup = scene.source_lookup();
    let v = lookup.world_to_voxel([world.x as f64, world.y as f64, world.z as f64]);
    let ijk = [
        v[0].round() as i64,
        v[1].round() as i64,
        v[2].round() as i64,
    ];
    let dims = scene.dimensions();
    if ijk
        .iter()
        .zip(dims.iter())
        .any(|(&c, &d)| c < 0 || c >= d as i64)
    {
        return None;
    }
    let compact_row = lookup.compact_row(ijk[0], ijk[1], ijk[2]);

    let mut report = OdxProbeReport {
        world: [world.x, world.y, world.z],
        ijk,
        compact_row,
        dpv: Vec::new(),
        fixels: Vec::new(),
        sh: None,
        odf: None,
        primary_fixel_metric: primary_fixel_metric(scene),
    };
    let Some(row) = compact_row else {
        return Some(report);
    };
    let row = row as usize;
    let dataset = scene.dataset();

    // Per-voxel scalars: every single-column DPV array.
    for name in scene.dpv_names() {
        if let Ok(view) = dataset.dpv::<f32>(name)
            && view.ncols() == 1
            && let Some(&value) = view.row(row).first()
        {
            report.dpv.push((name.to_string(), value));
        }
    }

    // Fixels: the voxel's CSR span plus every scalar DPF column.
    let offsets = scene.offsets();
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;
    let dpf_views: Vec<(String, odx_rs::TypedView2D<'_, f32>)> = scene
        .dpf_names()
        .into_iter()
        .filter_map(|name| {
            let view = dataset.dpf::<f32>(name).ok()?;
            (view.ncols() == 1).then(|| (name.to_string(), view))
        })
        .collect();
    for fixel_idx in start..end {
        let scalars = dpf_views
            .iter()
            .filter_map(|(name, view)| {
                view.row(fixel_idx)
                    .first()
                    .map(|&value| (name.clone(), value))
            })
            .collect();
        report.fixels.push(ProbeFixel {
            direction: scene.directions()[fixel_idx],
            scalars,
        });
    }

    if want_detail {
        report.sh = probe_sh(scene, row, detail);
        report.odf = probe_odf(scene, row);
    }

    Some(report)
}

fn probe_sh(scene: &OdxScene, row: usize, detail: u32) -> Option<ProbeSh> {
    let view = scene.sh_view_f32()?;
    let coeffs = view.row(row).to_vec();
    let mesh = scene.sh_render_mesh(detail);
    let (basis, peak_amplitude, clamped) = match &mesh {
        Some(mesh) => {
            let evaluator = mesh.evaluator();
            let basis = match evaluator.kind() {
                Ok(kind) => format!("{kind:?}"),
                Err(_) => "unknown".to_string(),
            };
            let amps = evaluator.apply_row(&coeffs);
            let peak = amps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (
                basis,
                peak.is_finite().then_some(peak),
                evaluator.is_nonnegative(),
            )
        }
        None => ("unknown".to_string(), None, true),
    };
    Some(ProbeSh {
        basis,
        coeffs,
        peak_amplitude,
        clamped_nonnegative: clamped,
    })
}

fn probe_odf(scene: &OdxScene, row: usize) -> Option<ProbeOdf> {
    let view = scene.odf_view_f32()?;
    Some(ProbeOdf {
        amplitudes: view.row(row).to_vec(),
        hemisphere: scene.glyph_source_is_hemisphere(),
    })
}

/// First canonical primary metric present among the file's DPF arrays,
/// mirroring odx-rs QC's `amplitude` → `afd` → `qa` priority.
pub fn primary_fixel_metric(scene: &OdxScene) -> Option<String> {
    let names = scene.dpf_names();
    ["amplitude", "afd", "qa"]
        .iter()
        .find(|c| names.iter().any(|n| n == *c))
        .map(|c| c.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use odx_rs::{DType, OdxBuilder};

    /// 2x2x2 grid, 3 masked voxels with distinct fixel counts, one DPV and
    /// one DPF array. Scaled affine so world != voxel coordinates.
    fn synthetic_scene() -> OdxScene {
        // Mask C-order (i slowest, k fastest): flat = i*4 + j*2 + k.
        // Masked: (0,0,0), (0,1,1), (1,1,0) → compact rows 0, 1, 2.
        let mut mask = vec![0u8; 8];
        mask[0] = 1;
        mask[3] = 1;
        mask[6] = 1;
        let mut builder = OdxBuilder::new(
            [
                [2.0, 0.0, 0.0, -10.0],
                [0.0, 2.0, 0.0, 5.0],
                [0.0, 0.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [2, 2, 2],
            mask,
        );
        // Fixels per compact voxel: 1, 2, 0.
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
        builder.push_voxel_peaks(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        builder.push_voxel_peaks(&[]);
        builder.set_dpv_data(
            "gfa",
            bytemuck::cast_slice(&[0.1f32, 0.2, 0.3]).to_vec(),
            1,
            DType::Float32,
        );
        builder.set_dpf_data(
            "amplitude",
            bytemuck::cast_slice(&[1.5f32, 2.5, 3.5]).to_vec(),
            1,
            DType::Float32,
        );
        OdxScene::from_dataset(builder.finalize().unwrap()).unwrap()
    }

    #[test]
    fn probe_reports_voxel_values() {
        let scene = synthetic_scene();
        // Compact row 1 is voxel (0,1,1) → world (-10, 7, 5).
        let report = probe_odx(&scene, Vec3::new(-10.0, 7.0, 5.0), 1, true).expect("in bounds");
        assert_eq!(report.ijk, [0, 1, 1]);
        assert_eq!(report.compact_row, Some(1));
        assert_eq!(report.dpv, vec![("gfa".to_string(), 0.2)]);
        assert_eq!(report.fixels.len(), 2);
        assert_eq!(report.fixels[0].direction, [0.0, 1.0, 0.0]);
        assert_eq!(
            report.fixels[0].scalars,
            vec![("amplitude".to_string(), 2.5)]
        );
        assert_eq!(
            report.fixels[1].scalars,
            vec![("amplitude".to_string(), 3.5)]
        );
        assert_eq!(report.primary_fixel_metric.as_deref(), Some("amplitude"));
        // No SH/ODF arrays in this file.
        assert!(report.sh.is_none());
        assert!(report.odf.is_none());
    }

    #[test]
    fn probe_out_of_mask_and_out_of_bounds() {
        let scene = synthetic_scene();
        // Voxel (1,0,0) is in bounds but unmasked → empty report.
        let report = probe_odx(&scene, Vec3::new(-8.0, 5.0, 3.0), 1, false).expect("in bounds");
        assert_eq!(report.compact_row, None);
        assert!(report.dpv.is_empty());
        assert!(report.fixels.is_empty());
        // Far outside the grid → None.
        assert!(probe_odx(&scene, Vec3::new(100.0, 100.0, 100.0), 1, false).is_none());
    }

    #[test]
    fn probe_nearest_voxel_rounding() {
        let scene = synthetic_scene();
        // 0.9 mm off the voxel center (voxel size 2 mm) still rounds home.
        let report = probe_odx(&scene, Vec3::new(-9.1, 6.2, 4.4), 1, false).expect("in bounds");
        assert_eq!(report.ijk, [0, 1, 1]);
        assert_eq!(report.compact_row, Some(1));
    }
}

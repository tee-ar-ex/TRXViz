//! `SampleVolumeAlongStreamlineOp` — for each input streamline, trilinearly
//! samples a `VolumeScalars` at every vertex, averages along the streamline,
//! and attaches the per-streamline mean as a DPS field on the output.
//!
//! Generic: works on any `(Streamline, VolumeScalars)` pair. The motivating
//! use case is pyAFQ probability-map scoring (mean prob along a streamline
//! → DPS), but it's equally useful for FA / MD / any other scalar volume.
//!
//! The output is the same `StreamlineFlow` with a new DPS slot pushed onto
//! its `gpu_data`. Wire `Color By DPS` downstream to visualize the score.

use std::borrow::Cow;

use glam::Vec3;

use crate::data::cifti::VolumeScalars;
use crate::data::trx_data::TrxGpuData;
use crate::error::WorkflowResult;
use crate::workflow::methods::OpCategory;
use crate::workflow::types::{StreamlineDataset, StreamlineFlow};

use super::super::{
    EvalCtx, EvaluatedValue, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue,
    expect_streamline_input, optional_volume_input,
    workflow_sample_volume_along_streamline_fingerprint,
};

fn default_dps_name() -> String {
    "vol_mean".to_string()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleVolumeAlongStreamlineOp {
    /// Name of the DPS field written to the output streamlines. Wire
    /// `Color By DPS(<this name>)` downstream to visualize the score.
    #[serde(default = "default_dps_name")]
    pub dps_name: String,
}

impl Default for SampleVolumeAlongStreamlineOp {
    fn default() -> Self {
        Self {
            dps_name: default_dps_name(),
        }
    }
}

impl WorkflowOp for SampleVolumeAlongStreamlineOp {
    fn tag(&self) -> &'static str {
        "sample_volume_along_streamline"
    }

    fn title(&self) -> &'static str {
        "Sample Volume Along Streamline"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline, PortKind::Volume]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn category(&self) -> OpCategory {
        OpCategory::StreamlineFilter
    }

    fn boilerplate(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "Each streamline was scored by trilinearly sampling the input \
             scalar volume at every vertex and averaging along the \
             streamline; the mean was attached as the `{}` DPS field.",
            self.dps_name,
        )))
    }

    fn evaluate(&self, ctx: &mut EvalCtx<'_, '_>) -> WorkflowResult<Vec<EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;

        let backing = optional_volume_input(ctx.inputs, 1).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!(
                "{} needs a Volume input on port 1",
                self.title()
            ))
        })?;

        // Reuse the cached derived dataset when nothing relevant changed.
        // Trilinearly sampling every vertex of every streamline and
        // deep-cloning `TrxGpuData` is expensive, and the new Arc would
        // otherwise change the downstream `Arc::as_ptr` hash on every
        // Interactive frame, busting tube/bundle caches downstream.
        let fingerprint =
            workflow_sample_volume_along_streamline_fingerprint(&flow, &self.dps_name, &backing);
        let dataset = match ctx
            .execution_cache
            .sample_volume_along_streamline_cache
            .get(&ctx.node.uuid)
        {
            Some((cached_fp, ds)) if *cached_fp == fingerprint => ds.clone(),
            _ => {
                let scalars = ctx.scalars_for(&backing)?;
                let scored = attach_volume_dps(&flow.dataset, scalars.as_ref(), &self.dps_name);
                let ds = std::sync::Arc::new(scored);
                ctx.execution_cache
                    .sample_volume_along_streamline_cache
                    .insert(ctx.node.uuid, (fingerprint, ds.clone()));
                ds
            }
        };

        let scored_flow = StreamlineFlow { dataset, ..flow };

        Ok(vec![EvaluatedValue {
            value: WorkflowValue::Streamline(scored_flow),
            stale: false,
        }])
    }
}

impl From<SampleVolumeAlongStreamlineOp> for WorkflowNodeKind {
    fn from(op: SampleVolumeAlongStreamlineOp) -> Self {
        Self::SampleVolumeAlongStreamline {
            dps_name: op.dps_name,
        }
    }
}

/// Compute mean trilinearly-sampled value along each streamline and write
/// it onto a clone of the dataset's `gpu_data` as a DPS field named
/// `dps_name`. Replaces any existing field with that name.
fn attach_volume_dps(
    dataset: &StreamlineDataset,
    scalars: &VolumeScalars,
    dps_name: &str,
) -> StreamlineDataset {
    let mut gpu = (*dataset.gpu_data).clone();
    let values = sample_means(&gpu, scalars);
    set_dps(&mut gpu, dps_name, values);
    StreamlineDataset {
        name: dataset.name.clone(),
        gpu_data: std::sync::Arc::new(gpu),
        backing: dataset.backing.clone(),
    }
}

fn set_dps(gpu: &mut TrxGpuData, name: &str, values: Vec<f32>) {
    if let Some(pos) = gpu.dps_data.iter().position(|(n, _)| n == name) {
        gpu.dps_data[pos].1 = values;
    } else {
        gpu.dps_data.push((name.to_string(), values));
        if !gpu.dps_names.iter().any(|n| n == name) {
            gpu.dps_names.push(name.to_string());
        }
    }
}

/// One mean per streamline. Vertices that fall outside the volume are
/// skipped; if a streamline has zero in-bounds samples, its mean is NaN
/// (Color By DPS already handles NaN as "no color").
fn sample_means(gpu: &TrxGpuData, scalars: &VolumeScalars) -> Vec<f32> {
    let n = gpu.nb_streamlines;
    if n == 0 || scalars.dims.contains(&0) {
        return vec![f32::NAN; n];
    }
    let ras_to_vox = scalars.voxel_to_ras.inverse();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = gpu.offsets[i] as usize;
        let end = gpu.offsets[i + 1] as usize;
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for v in start..end {
            let p = Vec3::from_array(gpu.positions[v]);
            let voxel = ras_to_vox.transform_point3(p);
            if let Some(s) = crate::data::sampling::trilinear(scalars, voxel) {
                acc += s;
                count += 1;
            }
        }
        out.push(if count > 0 {
            acc / count as f32
        } else {
            f32::NAN
        });
    }
    out
}

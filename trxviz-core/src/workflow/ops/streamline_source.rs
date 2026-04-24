use std::sync::Arc;

use crate::data::loaded_files::FileId;
use crate::data::trx_data::ColorMode;
use crate::units::StreamlineIndex;

use super::super::{
    EvalCtx, PortKind, StreamlineDataset, StreamlineFlow, WorkflowNodeKind, WorkflowOp,
    WorkflowValue,
};

#[derive(Debug, Clone, Copy)]
pub struct StreamlineSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for StreamlineSourceOp {
    fn tag(&self) -> &'static str {
        "streamline_source"
    }

    fn title(&self) -> &'static str {
        "Streamline Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let source = ctx.streamline_assets.get(&self.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!(
                "Missing streamline source {}",
                self.source_id
            ))
        })?;

        // Reuse the cached `Arc<StreamlineDataset>` when the source's
        // `Arc<TrxGpuData>` is the same instance as last eval. The
        // downstream fingerprint chain (`hash_flow`) keys off
        // `Arc::as_ptr(&dataset)`, so allocating a fresh Arc each
        // eval would invalidate every downstream cache (boundary
        // field, purifibre, …) on every interactive frame.
        let dataset = match ctx
            .execution_cache
            .streamline_source_datasets
            .get(&ctx.node.uuid)
        {
            Some(cached) if Arc::ptr_eq(&cached.gpu_data, &source.data) => cached.clone(),
            _ => {
                let new_ds = Arc::new(StreamlineDataset {
                    name: source.name.clone(),
                    gpu_data: source.data.clone(),
                    backing: source.backing.clone().ok_or_else(|| {
                        crate::error::WorkflowError::Evaluation(format!(
                            "Streamline source {} is missing export backing",
                            source.name
                        ))
                    })?,
                });
                ctx.execution_cache
                    .streamline_source_datasets
                    .insert(ctx.node.uuid, new_ds.clone());
                new_ds
            }
        };
        let selected = (0..source.data.nb_streamlines as u32)
            .map(StreamlineIndex)
            .collect();
        Ok(vec![
            WorkflowValue::Streamline(StreamlineFlow {
                dataset,
                selected_streamlines: selected,
                color_mode: ColorMode::DirectionRgb,
                scalar_auto_range: true,
                scalar_range_min: 0.0,
                scalar_range_max: 1.0,
                scalar_colormap: crate::renderer::mesh_renderer::SurfaceColormap::default(),
            })
            .into(),
        ])
    }
}

impl From<StreamlineSourceOp> for WorkflowNodeKind {
    fn from(op: StreamlineSourceOp) -> Self {
        Self::StreamlineSource {
            source_id: op.source_id,
        }
    }
}

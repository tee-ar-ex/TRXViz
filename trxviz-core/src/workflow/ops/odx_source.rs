use crate::data::loaded_files::FileId;
use crate::data::odx_data::{FixelField, FixelScalars, OdfField, OdxCatalog};

use super::super::{EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue};

#[derive(Debug, Clone, Copy)]
pub struct OdxSourceOp {
    pub source_id: FileId,
}

impl WorkflowOp for OdxSourceOp {
    fn tag(&self) -> &'static str {
        "odx_source"
    }

    fn title(&self) -> &'static str {
        "ODX Source"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[
            PortKind::Fixels,
            PortKind::OdfField,
            PortKind::OdxCatalog,
            PortKind::FixelScalars,
        ]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let asset = ctx.odx_assets.get(&self.source_id).ok_or_else(|| {
            crate::error::WorkflowError::Evaluation(format!("Missing ODX asset {}", self.source_id))
        })?;
        let scene = asset.scene.clone();
        let dirs = scene.directions().to_vec();
        let default_scalars = FixelScalars::from_directions(self.source_id, &dirs);
        let field = FixelField {
            source_id: self.source_id,
            scene: scene.clone(),
            scalars: default_scalars.clone(),
            colormap_code: 0,
            scalar_range: (0.0, 1.0),
        };
        let odf = OdfField {
            source_id: self.source_id,
            scene: scene.clone(),
        };
        let catalog = OdxCatalog::from_scene(self.source_id, scene);
        Ok(vec![
            WorkflowValue::Fixels(field).into(),
            WorkflowValue::OdfField(odf).into(),
            WorkflowValue::OdxCatalog(catalog).into(),
            WorkflowValue::FixelScalars(default_scalars).into(),
        ])
    }
}

impl From<OdxSourceOp> for WorkflowNodeKind {
    fn from(op: OdxSourceOp) -> Self {
        Self::OdxSource {
            source_id: op.source_id,
        }
    }
}

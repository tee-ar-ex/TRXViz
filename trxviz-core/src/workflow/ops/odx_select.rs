use super::super::{
    EvalCtx, PortKind, WorkflowNodeKind, WorkflowOp, WorkflowValue, expect_odx_catalog_input,
    volume_scalars_from_nifti_volume,
};

#[derive(Debug, Clone)]
pub struct OdxVolumeSelectOp {
    pub dpv_name: String,
}

#[derive(Debug, Clone)]
pub struct OdxFixelScalarSelectOp {
    pub dpf_name: String,
}

impl Default for OdxVolumeSelectOp {
    fn default() -> Self {
        Self {
            dpv_name: String::new(),
        }
    }
}

impl Default for OdxFixelScalarSelectOp {
    fn default() -> Self {
        Self {
            dpf_name: String::new(),
        }
    }
}

impl WorkflowOp for OdxVolumeSelectOp {
    fn tag(&self) -> &'static str {
        "odx_volume_select"
    }

    fn title(&self) -> &'static str {
        "ODX Volume Select"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::OdxCatalog]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::Volume, PortKind::VolumeScalars]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let catalog = expect_odx_catalog_input(ctx.inputs, self.title())?;
        if self.dpv_name.is_empty() {
            return Err(crate::error::WorkflowError::Evaluation(
                "ODX Volume Select needs a DPV name".to_string(),
            ));
        }
        let volume = catalog.materialize_dpv(&self.dpv_name).map_err(|e| {
            crate::error::WorkflowError::Evaluation(format!(
                "Failed to materialize DPV '{}': {e}",
                self.dpv_name
            ))
        })?;
        let volume_scalars =
            volume_scalars_from_nifti_volume(&volume, self.dpv_name.clone(), catalog.source_id);
        ctx.execution_cache.odx_dpv_materializations.insert(
            ctx.node.uuid,
            crate::workflow::types::OdxDpvMaterialization {
                source_id: catalog.source_id,
                dpv_name: self.dpv_name.clone(),
                volume: std::sync::Arc::new(volume),
            },
        );
        Ok(vec![
            WorkflowValue::Volume(catalog.source_id).into(),
            WorkflowValue::VolumeScalars(volume_scalars).into(),
        ])
    }
}

impl WorkflowOp for OdxFixelScalarSelectOp {
    fn tag(&self) -> &'static str {
        "odx_fixel_scalar_select"
    }

    fn title(&self) -> &'static str {
        "ODX Fixel Scalar Select"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::OdxCatalog]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::FixelScalars]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let catalog = expect_odx_catalog_input(ctx.inputs, self.title())?;
        if self.dpf_name.is_empty() {
            return Err(crate::error::WorkflowError::Evaluation(
                "ODX Fixel Scalar Select needs a DPF name".to_string(),
            ));
        }
        let values = catalog.scene.scalar_dpf_f32(&self.dpf_name).map_err(|e| {
            crate::error::WorkflowError::Evaluation(format!(
                "Failed to load DPF '{}': {e}",
                self.dpf_name
            ))
        })?;
        let scalars = crate::data::odx_data::FixelScalars::from_scalar(
            catalog.source_id,
            self.dpf_name.clone(),
            values,
        );
        Ok(vec![WorkflowValue::FixelScalars(scalars).into()])
    }
}

impl From<OdxVolumeSelectOp> for WorkflowNodeKind {
    fn from(op: OdxVolumeSelectOp) -> Self {
        Self::OdxVolumeSelect {
            dpv_name: op.dpv_name,
        }
    }
}

impl From<OdxFixelScalarSelectOp> for WorkflowNodeKind {
    fn from(op: OdxFixelScalarSelectOp) -> Self {
        Self::OdxFixelScalarSelect {
            dpf_name: op.dpf_name,
        }
    }
}

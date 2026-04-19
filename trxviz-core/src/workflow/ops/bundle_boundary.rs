use super::super::{
    BoundaryFieldPlan, BoundaryGlyphDrawPlan, BundleDrawPlan, BundleSurfaceBuildMode,
    BundleSurfaceColorMode, BundleSurfacePlan, EvalCtx, PortKind, StreamlineDisplayRuntime,
    WorkflowOp, expect_boundary_field_input, expect_bundle_surface_input,
    expect_parcel_selection_input, expect_streamline_input, prime_expensive_record,
    sync_node_state_from_run_record, workflow_boundary_plan_fingerprint,
    workflow_bundle_display_fingerprint, workflow_bundle_plan_fingerprint,
};

#[derive(Debug, Clone, Copy)]
pub struct BundleSurfaceBuildOp {
    pub per_group: bool,
    pub build_mode: BundleSurfaceBuildMode,
    pub voxel_size_mm: crate::units::Millimeters,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub min_component_volume_mm3: crate::units::Millimeters,
    pub tube_radius_mm: crate::units::Millimeters,
    pub tube_sides: u32,
    pub opacity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryFieldBuildOp {
    pub voxel_size_mm: crate::units::Millimeters,
    pub sphere_lod: u32,
    pub normalization: crate::data::orientation_field::BoundaryGlyphNormalization,
}

#[derive(Debug, Clone, Copy)]
pub struct BundleSurfaceDisplayOp {
    pub color_mode: BundleSurfaceColorMode,
    pub outline_thickness: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryGlyphDisplayOp {
    pub enabled: bool,
    pub scale: f32,
    pub density_3d_step: usize,
    pub slice_density_step: usize,
    pub color_mode: crate::data::orientation_field::BoundaryGlyphColorMode,
    pub min_contacts: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ParcelSurfaceBuildOp;

impl WorkflowOp for BundleSurfaceBuildOp {
    fn tag(&self) -> &'static str {
        "bundle_surface_build"
    }

    fn title(&self) -> &'static str {
        "Bundle Surface Build"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::BundleSurface]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let bundle = BundleSurfacePlan {
            build_node_uuid: ctx.node.uuid,
            label: ctx.node.label.clone(),
            flow,
            per_group: self.per_group,
            build_mode: self.build_mode,
            voxel_size_mm: self.voxel_size_mm,
            threshold: self.threshold,
            smooth_sigma: self.smooth_sigma,
            min_component_volume_mm3: self.min_component_volume_mm3,
            tube_radius_mm: self.tube_radius_mm,
            tube_sides: self.tube_sides,
            opacity: self.opacity,
        };
        let upstream_stale = ctx.upstream_stale();
        let fingerprint = workflow_bundle_plan_fingerprint(&bundle);
        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        prime_expensive_record(record, fingerprint);
        sync_node_state_from_run_record(ctx.node_state, record);
        ctx.scene_plan.bundle_surface_plans.push(bundle.clone());
        Ok(vec![super::super::EvaluatedValue {
            value: super::super::WorkflowValue::BundleSurface(bundle),
            stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
        }])
    }
}

impl WorkflowOp for BoundaryFieldBuildOp {
    fn tag(&self) -> &'static str {
        "boundary_field_build"
    }

    fn title(&self) -> &'static str {
        "Boundary Field Build"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::Streamline]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::BoundaryField]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let flow = expect_streamline_input(ctx.inputs, self.title())?;
        let plan = BoundaryFieldPlan {
            build_node_uuid: ctx.node.uuid,
            label: ctx.node.label.clone(),
            flow,
            voxel_size_mm: self.voxel_size_mm,
            sphere_lod: self.sphere_lod,
            normalization: self.normalization,
        };
        let upstream_stale = ctx.upstream_stale();
        let fingerprint = workflow_boundary_plan_fingerprint(&plan);
        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        prime_expensive_record(record, fingerprint);
        sync_node_state_from_run_record(ctx.node_state, record);
        ctx.scene_plan.boundary_field_plans.push(plan.clone());
        Ok(vec![super::super::EvaluatedValue {
            value: super::super::WorkflowValue::BoundaryField(plan),
            stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
        }])
    }
}

impl WorkflowOp for BundleSurfaceDisplayOp {
    fn tag(&self) -> &'static str {
        "bundle_surface_display"
    }

    fn title(&self) -> &'static str {
        "Bundle Surface Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::BundleSurface, PortKind::BoundaryField]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let (bundle, stale) = expect_bundle_surface_input(ctx.inputs, self.title())?;
        let boundary_field = ctx
            .inputs
            .get(1)
            .and_then(|value| value.as_ref())
            .map(|value| expect_boundary_field_input(Some(value), self.title()))
            .transpose()?;
        let runtime = ctx.display_ids.entry(ctx.node.uuid).or_insert_with(|| {
            let draw_id = *ctx.next_draw_id;
            *ctx.next_draw_id += 1;
            StreamlineDisplayRuntime {
                draw_id,
                ..Default::default()
            }
        });
        let resolved_color_mode =
            if matches!(bundle.build_mode, BundleSurfaceBuildMode::Streamtubes) {
                BundleSurfaceColorMode::SourceColors
            } else {
                self.color_mode
            };
        let draw = BundleDrawPlan {
            node_uuid: ctx.node.uuid,
            build_node_uuid: bundle.build_node_uuid,
            boundary_field_node_uuid: boundary_field
                .as_ref()
                .map(|(plan, _)| plan.build_node_uuid),
            draw_id: runtime.draw_id,
            label: bundle.label,
            flow: bundle.flow,
            per_group: bundle.per_group,
            color_mode: resolved_color_mode,
            build_mode: bundle.build_mode,
            voxel_size_mm: bundle.voxel_size_mm,
            threshold: bundle.threshold,
            smooth_sigma: bundle.smooth_sigma,
            min_component_volume_mm3: bundle.min_component_volume_mm3,
            tube_radius_mm: bundle.tube_radius_mm,
            tube_sides: bundle.tube_sides,
            opacity: bundle.opacity,
            outline_thickness: self.outline_thickness,
        };
        let boundary_revision = draw.boundary_field_node_uuid.and_then(|uuid| {
            ctx.execution_cache
                .boundary_field_cache
                .get(&uuid)
                .map(|cache| cache.fingerprint)
        });
        let display_fingerprint = workflow_bundle_display_fingerprint(&draw, boundary_revision);
        let record = ctx
            .execution_cache
            .node_runs
            .entry(ctx.node.uuid)
            .or_default();
        prime_expensive_record(record, display_fingerprint);
        sync_node_state_from_run_record(ctx.node_state, record);
        let boundary_stale = boundary_field.as_ref().is_some_and(|(_, stale)| *stale);
        ctx.node_state.summary = if stale || boundary_stale {
            format!(
                "Displaying stale bundle surface ({})",
                resolved_color_mode.label()
            )
        } else {
            format!(
                "Displaying bundle surface ({})",
                resolved_color_mode.label()
            )
        };
        ctx.scene_plan.bundle_draws.push(draw);
        Ok(Vec::new())
    }
}

impl WorkflowOp for BoundaryGlyphDisplayOp {
    fn tag(&self) -> &'static str {
        "boundary_glyph_display"
    }

    fn title(&self) -> &'static str {
        "Boundary Glyph Display"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::BoundaryField]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let (plan, stale) = expect_boundary_field_input(
            ctx.inputs.first().and_then(|value| value.as_ref()),
            self.title(),
        )?;
        let draw = BoundaryGlyphDrawPlan {
            node_uuid: ctx.node.uuid,
            build_node_uuid: plan.build_node_uuid,
            label: ctx.node.label.clone(),
            visible: self.enabled,
            scale: self.scale,
            density_3d_step: self.density_3d_step,
            slice_density_step: self.slice_density_step,
            color_mode: self.color_mode,
            min_contacts: self.min_contacts,
        };
        ctx.node_state.execution = None;
        ctx.node_state.summary = if !self.enabled {
            "Boundary field hidden".to_string()
        } else if stale {
            "Displaying stale boundary field".to_string()
        } else {
            "Displaying boundary field".to_string()
        };
        ctx.scene_plan.boundary_glyph_draws.push(draw);
        Ok(Vec::new())
    }
}

impl WorkflowOp for ParcelSurfaceBuildOp {
    fn tag(&self) -> &'static str {
        "parcel_surface_build"
    }

    fn title(&self) -> &'static str {
        "Parcel Surface Build"
    }

    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ParcelSelection]
    }

    fn output_ports(&self) -> &'static [PortKind] {
        &[]
    }

    fn evaluate(
        &self,
        ctx: &mut EvalCtx<'_, '_>,
    ) -> crate::error::WorkflowResult<Vec<super::super::EvaluatedValue>> {
        let parcel_selection = expect_parcel_selection_input(ctx.inputs, self.title())?;
        ctx.scene_plan
            .parcellation_draws
            .push(super::super::ParcellationDrawPlan {
                source_id: parcel_selection.source_id,
                labels: parcel_selection.labels,
                opacity: 0.9,
            });
        Ok(Vec::new())
    }
}

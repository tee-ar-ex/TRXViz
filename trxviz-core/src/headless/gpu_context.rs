use anyhow::anyhow;
use glam::Vec3;
use pollster::block_on;

use super::SceneBounds;
use crate::data::odx_data::OdxScene;
use crate::data::odx_data::{FixelScalarValues, OdfAmplitudeConditioning, OdxGlyphSourceKind};
use crate::data::trx_data::RenderStyle;
use crate::renderer::background_renderer::BackgroundResources;
use crate::renderer::fixel_renderer::FixelResources;
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::MeshResources;
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::workflow::{
    default_odf_glyph_detail, materialize_flow_gpu, workflow_bundle_display_fingerprint,
    workflow_streamline_fingerprint,
};

pub(super) const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub(super) struct GpuSceneResources {
    pub(super) background: BackgroundResources,
    pub(super) streamlines: AllStreamlineResources,
    pub(super) slices: AllSliceResources,
    pub(super) meshes: MeshResources,
    pub(super) glyphs: GlyphResources,
    pub(super) fixels_3d: FixelResources,
    pub(super) fixels_2d: FixelResources,
    pub(super) bounds: SceneBounds,
}

pub(super) struct GpuContext {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
}

pub(super) fn active_fixel_draw_3d(
    workflow: &HeadlessWorkflowState,
) -> Option<&crate::workflow::FixelDrawPlan> {
    workflow
        .runtime
        .scene_plan
        .fixel_3d_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| workflow.runtime.scene_plan.fixel_3d_draws.first())
}

pub(super) fn active_fixel_draw_2d(
    workflow: &HeadlessWorkflowState,
) -> Option<&crate::workflow::FixelDrawPlan> {
    workflow
        .runtime
        .scene_plan
        .fixel_2d_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first())
}

pub(super) fn create_gpu_context() -> anyhow::Result<GpuContext> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map_err(|_| anyhow!("no headless GPU backend available"))?;

    let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("trxviz-headless-device"),
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 8192,
            max_buffer_size: 1 << 30,
            ..base_limits
        },
        ..Default::default()
    };
    let (device, queue) = block_on(adapter.request_device(&descriptor))
        .map_err(|err| anyhow!("failed to create GPU device: {err}"))?;
    Ok(GpuContext { device, queue })
}

pub(super) fn build_gpu_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
) -> anyhow::Result<GpuSceneResources> {
    let mut bounds_min = Vec3::splat(f32::INFINITY);
    let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);

    let mut expand = |point: Vec3| {
        bounds_min = bounds_min.min(point);
        bounds_max = bounds_max.max(point);
    };

    let background = BackgroundResources::new(device, TARGET_FORMAT);
    let mut slices = AllSliceResources {
        entries: Vec::new(),
    };
    for nifti in &scene.nifti_files {
        let slice_resources = SliceResources::new(device, queue, TARGET_FORMAT, &nifti.volume);
        slice_resources.update_slice(
            queue,
            SliceAxis::Axial,
            scene.slice_indices[0],
            &nifti.volume,
        );
        slice_resources.update_slice(
            queue,
            SliceAxis::Coronal,
            scene.slice_indices[1],
            &nifti.volume,
        );
        slice_resources.update_slice(
            queue,
            SliceAxis::Sagittal,
            scene.slice_indices[2],
            &nifti.volume,
        );
        slices.entries.push((nifti.id, slice_resources));

        for x in [0.0, nifti.volume.dims[0] as f32] {
            for y in [0.0, nifti.volume.dims[1] as f32] {
                for z in [0.0, nifti.volume.dims[2] as f32] {
                    expand(nifti.volume.voxel_to_world(Vec3::new(x, y, z)));
                }
            }
        }
    }

    let mut meshes = MeshResources::new(device, TARGET_FORMAT);
    for surface in &scene.gifti_surfaces {
        meshes.add_surface(surface.id, device, surface.data.as_ref());
        expand(surface.data.bbox_min);
        expand(surface.data.bbox_max);
    }
    for draw in &workflow.runtime.scene_plan.surface_draws {
        if !draw.vertex_rgba.is_empty() {
            meshes.update_surface_colors(queue, draw.source_id, &draw.vertex_rgba);
        }
        if let Some(scalars) = &draw.projection_scalars {
            meshes.update_surface_scalars(queue, draw.source_id, scalars);
        }
    }

    let mut streamlines = AllStreamlineResources {
        entries: Vec::new(),
    };
    for draw in &workflow.runtime.scene_plan.streamline_draws {
        let fingerprint = workflow_streamline_fingerprint(draw);
        let subset = materialize_flow_gpu(draw.flow.clone());
        for position in &subset.positions {
            expand(Vec3::from(*position));
        }
        let mut resource = StreamlineResources::new(device, TARGET_FORMAT, &subset);
        if draw.render_style == RenderStyle::Tubes {
            let cache = workflow
                .execution_cache
                .tube_geometry_cache
                .get(&draw.node_uuid)
                .filter(|cache| cache.fingerprint == fingerprint)
                .ok_or_else(|| anyhow!("missing tube geometry for {}", draw.label))?;
            resource.update_tube_geometry(device, &cache.vertices, &cache.indices);
        }
        streamlines.entries.push((draw.draw_id, resource));
    }

    for draw in &workflow.runtime.scene_plan.voxel_mask_mesh_draws {
        if let Some(cache) = workflow
            .execution_cache
            .voxel_mask_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == draw.fingerprint)
        {
            let one = [(cache.mesh.clone(), draw.label.clone())];
            meshes.set_bundle_meshes(draw.draw_id, device, &one);
            for vertex in &cache.mesh.vertices {
                expand(Vec3::from(vertex.position));
            }
        }
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        if let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        {
            meshes.set_bundle_meshes(draw.draw_id, device, &cache.meshes);
            for (mesh, _) in &cache.meshes {
                for vertex in &mesh.vertices {
                    expand(Vec3::from(vertex.position));
                }
            }
        }
    }

    let mut glyphs = GlyphResources::new(device, TARGET_FORMAT);
    if let Some(draw) = workflow
        .runtime
        .scene_plan
        .boundary_glyph_draws
        .iter()
        .find(|draw| draw.visible)
        .or_else(|| workflow.runtime.scene_plan.boundary_glyph_draws.first())
    {
        if let Some(cache) = workflow
            .execution_cache
            .boundary_field_cache
            .get(&draw.build_node_uuid)
        {
            glyphs.set_field(device, cache.field.clone(), draw.scale, draw.min_contacts);
            let origin = cache.field.grid.origin_ras;
            let size = Vec3::new(
                cache.field.grid.dims[0] as f32,
                cache.field.grid.dims[1] as f32,
                cache.field.grid.dims[2] as f32,
            ) * cache.field.grid.voxel_size_mm.0;
            expand(origin);
            expand(origin + size);
        }
    }

    let mut fixels_3d = FixelResources::new(device, TARGET_FORMAT);
    let mut fixels_2d = FixelResources::new(device, TARGET_FORMAT);

    // ODX ODF glyphs and fixels — driven by the evaluated workflow plan.
    let axial_slice = scene.slice_indices[2] as u32;
    let odf_plan = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first());
    if let Some(plan) = odf_plan {
        let odx = &plan.field.scene;
        let conditioning = OdfAmplitudeConditioning::new(plan.subtract_iso, plan.norm_within_voxel);
        match odx.glyph_source_kind() {
            Some(OdxGlyphSourceKind::Odf) => {
                let (sphere_vertices, sphere_indices) = odx
                    .odf_render_geometry()
                    .expect("ODF geometry should exist for ODF glyph mode");
                let use_slice_local = odx_odf_exceeds_binding_limit(odx, device);
                let slice_index = scene.slice_indices[plan.slice_axis.viewport_index()] as u32;
                let instances = if use_slice_local {
                    odx.glyph_instances_for_slice(
                        plan.slice_axis.odx_axis(),
                        slice_index,
                        odx.odf_render_row_width()
                            .expect("ODF render row width should exist for ODF glyph mode"),
                    )
                } else {
                    odx.glyph_instances_full_volume(
                        odx.odf_render_row_width()
                            .expect("ODF render row width should exist for ODF glyph mode"),
                    )
                };
                if !instances.is_empty() {
                    if use_slice_local {
                        let amplitudes = odx
                            .conditioned_odf_amplitudes_for_slice(
                                plan.slice_axis.odx_axis(),
                                slice_index,
                                conditioning,
                            )
                            .expect("ODF amplitudes should exist for slice-local ODF mode");
                        glyphs.set_odx_slice_odf(
                            device,
                            sphere_vertices,
                            sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    } else {
                        let amplitudes = odx
                            .conditioned_odf_amplitudes_full_sphere(conditioning)
                            .expect("ODF amplitudes should exist for ODF glyph mode");
                        glyphs.set_odx_odf_volume(
                            device,
                            sphere_vertices,
                            sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    }
                }
            }
            Some(OdxGlyphSourceKind::Sh) => {
                let slice_index = scene.slice_indices[plan.slice_axis.viewport_index()] as u32;
                let detail = clamped_sh_detail_for_slice(
                    odx,
                    plan.detail,
                    plan.slice_axis.odx_axis(),
                    slice_index,
                    device,
                );
                let mesh = odx
                    .sh_render_mesh(detail)
                    .expect("SH render mesh should exist for SH glyph mode");
                let instances = odx.glyph_instances_for_slice(
                    plan.slice_axis.odx_axis(),
                    slice_index,
                    mesh.row_width(),
                );
                if !instances.is_empty() {
                    let coefficients = odx
                        .sh_coefficients_for_slice(plan.slice_axis.odx_axis(), slice_index)
                        .expect("SH coefficients should exist for SH glyph mode");
                    glyphs.set_odx_sh_volume(
                        device,
                        mesh.vertices(),
                        mesh.indices(),
                        &instances,
                        &coefficients,
                        odx.sh_view_f32()
                            .expect("SH coefficients should exist for SH glyph mode")
                            .ncols(),
                        mesh.transform_flat(),
                        mesh.source_dir_count(),
                        mesh.row_width(),
                        odx.default_normalized_peak_length_mm(),
                        plan.subtract_iso,
                        plan.norm_within_voxel,
                        None,
                        None,
                    );
                    let slice_indices: Vec<u32> = (0..instances.len() as u32).collect();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("headless_odx_sh_initial_slice_encoder"),
                        });
                    glyphs.dispatch_odx_sh_slice(device, queue, &mut encoder, &slice_indices);
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
            None => {}
        }
        for &center in odx.centers_ras() {
            expand(Vec3::from(center));
        }
    } else if let Some(odx) = &scene.odx_scene {
        let conditioning = OdfAmplitudeConditioning::new(true, false);
        match odx.glyph_source_kind() {
            Some(OdxGlyphSourceKind::Odf) => {
                let (sphere_vertices, sphere_indices) = odx
                    .odf_render_geometry()
                    .expect("ODF geometry should exist for ODF glyph mode");
                let use_slice_local = odx_odf_exceeds_binding_limit(odx, device);
                let instances = if use_slice_local {
                    odx.glyph_instances_for_slice(
                        2,
                        axial_slice,
                        odx.odf_render_row_width()
                            .expect("ODF render row width should exist for ODF glyph mode"),
                    )
                } else {
                    odx.glyph_instances_full_volume(
                        odx.odf_render_row_width()
                            .expect("ODF render row width should exist for ODF glyph mode"),
                    )
                };
                if !instances.is_empty() {
                    if use_slice_local {
                        let amplitudes = odx
                            .conditioned_odf_amplitudes_for_slice(2, axial_slice, conditioning)
                            .expect("ODF amplitudes should exist for slice-local ODF mode");
                        glyphs.set_odx_slice_odf(
                            device,
                            sphere_vertices,
                            sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    } else {
                        let amplitudes = odx
                            .conditioned_odf_amplitudes_full_sphere(conditioning)
                            .expect("ODF amplitudes should exist for ODF glyph mode");
                        glyphs.set_odx_odf_volume(
                            device,
                            sphere_vertices,
                            sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    }
                }
            }
            Some(OdxGlyphSourceKind::Sh) => {
                let detail = clamped_sh_detail_for_slice(
                    odx,
                    default_odf_glyph_detail(),
                    2,
                    axial_slice,
                    device,
                );
                let mesh = odx
                    .sh_render_mesh(detail)
                    .expect("SH render mesh should exist for SH glyph mode");
                let instances = odx.glyph_instances_for_slice(2, axial_slice, mesh.row_width());
                if !instances.is_empty() {
                    let coefficients = odx
                        .sh_coefficients_for_slice(2, axial_slice)
                        .expect("SH coefficients should exist for SH glyph mode");
                    glyphs.set_odx_sh_volume(
                        device,
                        mesh.vertices(),
                        mesh.indices(),
                        &instances,
                        &coefficients,
                        odx.sh_view_f32()
                            .expect("SH coefficients should exist for SH glyph mode")
                            .ncols(),
                        mesh.transform_flat(),
                        mesh.source_dir_count(),
                        mesh.row_width(),
                        odx.default_normalized_peak_length_mm(),
                        true,
                        false,
                        None,
                        None,
                    );
                    let slice_indices: Vec<u32> = (0..instances.len() as u32).collect();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("headless_odx_sh_fallback_slice_encoder"),
                        });
                    glyphs.dispatch_odx_sh_slice(device, queue, &mut encoder, &slice_indices);
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
            None => {}
        }
        for &center in odx.centers_ras() {
            expand(Vec3::from(center));
        }
    }

    if let Some(plan) = active_fixel_draw_3d(workflow) {
        let scalars_vec: Option<Vec<f32>> = match &plan.field.scalars.values {
            FixelScalarValues::Scalar(values) if plan.colormap_code != 0 => {
                Some((**values).clone())
            }
            _ => None,
        };
        let mut fixel_instances = plan
            .field
            .scene
            .all_fixels_with_scalars(scalars_vec.as_deref());
        for inst in &mut fixel_instances {
            inst.length *= plan.length_scale;
        }
        if !fixel_instances.is_empty() {
            fixels_3d.set_fixels(device, &fixel_instances);
        }
    } else if let Some(odx) = &scene.odx_scene {
        let fixel_instances = odx.all_fixels();
        if !fixel_instances.is_empty() {
            fixels_3d.set_fixels(device, &fixel_instances);
        }
    }

    if let Some(plan) = active_fixel_draw_2d(workflow) {
        let scalars_vec: Option<Vec<f32>> = match &plan.field.scalars.values {
            FixelScalarValues::Scalar(values) if plan.colormap_code != 0 => {
                Some((**values).clone())
            }
            _ => None,
        };
        let mut fixel_instances = plan
            .field
            .scene
            .all_fixels_with_scalars(scalars_vec.as_deref());
        for inst in &mut fixel_instances {
            inst.length *= plan.length_scale;
        }
        if !fixel_instances.is_empty() {
            fixels_2d.set_fixels(device, &fixel_instances);
        }
    } else if let Some(odx) = &scene.odx_scene {
        let fixel_instances = odx.all_fixels();
        if !fixel_instances.is_empty() {
            fixels_2d.set_fixels(device, &fixel_instances);
        }
    }

    if !bounds_min.is_finite() || !bounds_max.is_finite() {
        bounds_min = scene.volume_center - Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
        bounds_max = scene.volume_center + Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
    }

    Ok(GpuSceneResources {
        background,
        streamlines,
        slices,
        meshes,
        glyphs,
        fixels_3d,
        fixels_2d,
        bounds: SceneBounds {
            min: bounds_min,
            max: bounds_max,
        },
    })
}

fn odx_odf_exceeds_binding_limit(odx: &OdxScene, device: &wgpu::Device) -> bool {
    odx.compact_voxel_count()
        .saturating_mul(odx.odf_render_row_width().unwrap_or(0))
        .saturating_mul(std::mem::size_of::<f32>())
        > device.limits().max_storage_buffer_binding_size as usize
}

fn clamped_sh_detail_for_slice(
    odx: &OdxScene,
    requested_detail: u32,
    axis: usize,
    slice_idx: u32,
    device: &wgpu::Device,
) -> u32 {
    odx.clamp_sh_detail_for_slice(
        axis,
        slice_idx,
        requested_detail,
        device.limits().max_storage_buffer_binding_size as usize,
    )
}

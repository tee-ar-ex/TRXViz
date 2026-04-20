use std::path::Path;

use anyhow::anyhow;
use glam::Vec3;

use super::HeadlessRenderData;
use super::gpu_context::{GpuSceneResources, TARGET_FORMAT};
use super::readback::readback_texture_to_png;
use crate::lighting::{SceneLightingParams, WorkflowRender3D};
use crate::renderer::camera::OrthoSliceCamera;
use crate::renderer::slice_renderer::SliceAxis;
use crate::renderer::viewport::ViewportIndex;
use crate::scene::HeadlessScene;
use crate::workflow::{
    WorkflowOrthoSliceCamera, WorkflowSliceViewKind, WorkflowSliceViewUi, WorkflowView2DMode,
};

#[derive(Clone, Copy)]
struct ViewportRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct SlicePanel {
    rect: ViewportRect,
    axis_index: usize,
    slice_index: usize,
    slice_pos: f32,
}

#[cfg(feature = "png-export")]
pub(super) fn render_scene2d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    slice_view_ui: Option<WorkflowSliceViewUi>,
    scene: &HeadlessScene,
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let slice_view_ui = slice_view_ui.ok_or_else(|| {
        anyhow!(
            "2D project rendering requires saved slice_view_ui state; open the project in TRXViz and save it first"
        )
    })?;

    let panels = build_2d_panels(&slice_view_ui, scene, width, height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_2d_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_2d_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let lighting = SceneLightingParams::default();
    let neutral_render = WorkflowRender3D {
        vignette_strength: 0.0,
        exposure: 1.0,
        contrast: 1.0,
        ..Default::default()
    };

    for panel in &panels {
        let viewport = ViewportIndex::from_slice_axis_index(panel.axis_index);
        let aspect = panel.rect.width as f32 / panel.rect.height.max(1) as f32;
        let camera = &slice_view_ui.slice_cameras[panel.axis_index];
        let view_proj =
            build_slice_camera(panel.axis_index, camera).view_projection(aspect, panel.slice_pos);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources
                .slices
                .entries
                .iter()
                .find(|(id, _)| *id == volume.file_id)
            {
                slice.update_uniforms(
                    queue,
                    viewport.into(),
                    view_proj,
                    volume.window_center,
                    volume.window_width,
                    volume.colormap,
                    volume.opacity,
                );
            }
        }
        let (slab_normal, slab_center) = slice_plane_for_panel(scene, panel);
        for streamline in &render_data.streamline_draws {
            if !streamline.visible {
                continue;
            }
            if let Some((_, resource)) = resources
                .streamlines
                .entries
                .iter()
                .find(|(id, _)| *id == streamline.file_id)
            {
                let hw = streamline.tube_radius.max(0.5);
                resource.update_uniforms(
                    queue,
                    viewport.into(),
                    view_proj,
                    Vec3::ZERO,
                    0,
                    slab_normal,
                    slab_center,
                    hw,
                    0.5,
                    lighting,
                    &neutral_render,
                    0.0,
                    1.0,
                );
            }
        }
        if render_data.glyph_visible || render_data.odx_visible {
            resources.glyphs.update_uniforms(
                queue,
                viewport.into(),
                view_proj,
                Vec3::ZERO,
                slab_normal,
                slab_center,
                1.0,
                render_data.glyph_color_mode,
                render_data.glyph_slice_density_step,
                render_data.odf_glyph_opacity,
                render_data.odf_glyph_gloss,
                lighting,
                &neutral_render,
                0.0,
                1.0,
            );
        }
        if render_data.odx_visible && render_data.odx_fixel_2d_visible {
            resources.fixels_2d.update_uniforms(
                queue,
                viewport.into(),
                view_proj,
                Vec3::ZERO,
                slab_normal,
                slab_center,
                render_data.fixel_2d_slab_half_width_mm.0,
                1,
                render_data.fixel_2d_line_width,
                render_data.fixel_2d_opacity,
                lighting,
                &neutral_render,
                0.0,
                1.0,
            );
            resources.fixels_2d.update_colormap(
                queue,
                viewport.into(),
                render_data.fixel_2d_colormap_code,
                (
                    render_data.fixel_2d_scalar_range[0],
                    render_data.fixel_2d_scalar_range[1],
                ),
            );
        }
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trxviz_headless_2d_encoder"),
    });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_2d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let render_pass: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut render_pass) };

        for panel in &panels {
            let viewport = ViewportIndex::from_slice_axis_index(panel.axis_index);
            render_pass.set_viewport(
                panel.rect.x as f32,
                panel.rect.y as f32,
                panel.rect.width as f32,
                panel.rect.height as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(
                panel.rect.x,
                panel.rect.y,
                panel.rect.width,
                panel.rect.height,
            );

            for volume in &render_data.volume_draws {
                if let Some((_, slice)) = resources
                    .slices
                    .entries
                    .iter()
                    .find(|(id, _)| *id == volume.file_id)
                {
                    render_pass.set_pipeline(&slice.pipeline);
                    render_pass.set_bind_group(0, slice.bind_group(viewport.into()), &[]);
                    render_pass.set_index_buffer(
                        slice.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass
                        .set_vertex_buffer(0, slice.quad_buffers[panel.axis_index].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }

            if render_data.any_visible_streamlines {
                for streamline in &render_data.streamline_draws {
                    if !streamline.visible {
                        continue;
                    }
                    if let Some((_, resource)) = resources
                        .streamlines
                        .entries
                        .iter()
                        .find(|(id, _)| *id == streamline.file_id)
                    {
                        render_pass.set_pipeline(&resource.slice_pipeline);
                        render_pass.set_bind_group(0, resource.bind_group(viewport.into()), &[]);
                        render_pass.set_vertex_buffer(0, resource.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, resource.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, resource.tangent_buffer.slice(..));
                        render_pass.set_index_buffer(
                            resource.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..resource.num_indices, 0, 0..1);
                    }
                }
            }
            if render_data.glyph_visible || render_data.odx_visible {
                resources.glyphs.paint(render_pass, viewport.into(), true);
            }
            if render_data.odx_visible && render_data.odx_fixel_2d_visible {
                resources
                    .fixels_2d
                    .paint(render_pass, viewport.into(), true);
            }
        }
    }

    readback_texture_to_png(device, queue, encoder, &texture, width, height, output_path)
}

fn build_2d_panels(
    slice_view_ui: &WorkflowSliceViewUi,
    scene: &HeadlessScene,
    width: u32,
    height: u32,
) -> Vec<SlicePanel> {
    const SPACING: u32 = 8;
    match slice_view_ui.mode {
        WorkflowView2DMode::Slice => {
            let axis_index = axis_index_for_kind(slice_view_ui.single_view);
            let slice_index = scene.slice_indices[axis_index];
            vec![SlicePanel {
                rect: ViewportRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                axis_index,
                slice_index,
                slice_pos: slice_world_position(scene, axis_index),
            }]
        }
        WorkflowView2DMode::Ortho => {
            if slice_view_ui.ortho_show_row {
                let panel_width = ((width.saturating_sub(2 * SPACING)) / 3).max(1);
                (0..3)
                    .map(|axis_index| SlicePanel {
                        rect: ViewportRect {
                            x: axis_index as u32 * (panel_width + SPACING),
                            y: 0,
                            width: panel_width,
                            height: height.max(1),
                        },
                        axis_index,
                        slice_index: scene.slice_indices[axis_index],
                        slice_pos: slice_world_position(scene, axis_index),
                    })
                    .collect()
            } else {
                let panel_width = ((width.saturating_sub(SPACING)) / 2).max(1);
                let panel_height = ((height.saturating_sub(SPACING)) / 2).max(1);
                vec![
                    SlicePanel {
                        rect: ViewportRect {
                            x: 0,
                            y: 0,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 0,
                        slice_index: scene.slice_indices[0],
                        slice_pos: slice_world_position(scene, 0),
                    },
                    SlicePanel {
                        rect: ViewportRect {
                            x: panel_width + SPACING,
                            y: 0,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 1,
                        slice_index: scene.slice_indices[1],
                        slice_pos: slice_world_position(scene, 1),
                    },
                    SlicePanel {
                        rect: ViewportRect {
                            x: panel_width + SPACING,
                            y: panel_height + SPACING,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 2,
                        slice_index: scene.slice_indices[2],
                        slice_pos: slice_world_position(scene, 2),
                    },
                ]
            }
        }
        WorkflowView2DMode::Lightbox => {
            let axis_index = axis_index_for_kind(slice_view_ui.lightbox_axis);
            let rows = slice_view_ui.lightbox_rows.max(1);
            let cols = slice_view_ui.lightbox_cols.max(1);
            let tile_width = ((width.saturating_sub(SPACING * cols.saturating_sub(1) as u32))
                / cols as u32)
                .max(1);
            let tile_height = ((height.saturating_sub(SPACING * rows.saturating_sub(1) as u32))
                / rows as u32)
                .max(1);
            let total = rows * cols;
            let center_tile = total / 2;
            let center_index = scene.slice_indices[axis_index];
            let max_index = max_slice_index(scene, axis_index);
            let mut panels = Vec::with_capacity(total);
            for row in 0..rows {
                for col in 0..cols {
                    let tile = row * cols + col;
                    let delta = tile as isize - center_tile as isize;
                    let index = center_index.saturating_add_signed(delta).min(max_index);
                    panels.push(SlicePanel {
                        rect: ViewportRect {
                            x: col as u32 * (tile_width + SPACING),
                            y: row as u32 * (tile_height + SPACING),
                            width: tile_width,
                            height: tile_height,
                        },
                        axis_index,
                        slice_index: index,
                        slice_pos: slice_world_position_for_index(scene, axis_index, index),
                    });
                }
            }
            panels
        }
    }
}

fn axis_index_for_kind(kind: WorkflowSliceViewKind) -> usize {
    match kind {
        WorkflowSliceViewKind::Axial => 0,
        WorkflowSliceViewKind::Coronal => 1,
        WorkflowSliceViewKind::Sagittal => 2,
    }
}

fn slice_plane_for_panel(scene: &HeadlessScene, panel: &SlicePanel) -> (Vec3, Vec3) {
    if let Some(nf) = scene.nifti_files.first() {
        nf.volume.slice_plane(panel.axis_index, panel.slice_index)
    } else {
        let normal = match panel.axis_index {
            0 => Vec3::Z,
            1 => Vec3::Y,
            _ => Vec3::X,
        };
        let center = match panel.axis_index {
            0 => Vec3::new(0.0, 0.0, panel.slice_pos),
            1 => Vec3::new(0.0, panel.slice_pos, 0.0),
            _ => Vec3::new(panel.slice_pos, 0.0, 0.0),
        };
        (normal, center)
    }
}

fn build_slice_camera(axis_index: usize, camera: &WorkflowOrthoSliceCamera) -> OrthoSliceCamera {
    OrthoSliceCamera {
        axis: match axis_index {
            0 => SliceAxis::Axial,
            1 => SliceAxis::Coronal,
            _ => SliceAxis::Sagittal,
        },
        center: camera.center,
        half_extent: camera.half_extent,
        rotation: camera.rotation,
    }
}

fn slice_world_position(scene: &HeadlessScene, axis_index: usize) -> f32 {
    slice_world_position_for_index(scene, axis_index, scene.slice_indices[axis_index])
}

fn slice_world_position_for_index(scene: &HeadlessScene, axis_index: usize, index: usize) -> f32 {
    if let Some(nf) = scene.nifti_files.first() {
        let idx = index as f32;
        let world = match axis_index {
            0 => nf.volume.voxel_to_world(Vec3::new(0.0, 0.0, idx)),
            1 => nf.volume.voxel_to_world(Vec3::new(0.0, idx, 0.0)),
            _ => nf.volume.voxel_to_world(Vec3::new(idx, 0.0, 0.0)),
        };
        match axis_index {
            0 => world.z,
            1 => world.y,
            _ => world.x,
        }
    } else {
        scene.slice_world_offsets[axis_index]
    }
}

fn max_slice_index(scene: &HeadlessScene, axis_index: usize) -> usize {
    scene
        .nifti_files
        .first()
        .map(|nf| match axis_index {
            0 => nf.volume.dims[2].saturating_sub(1),
            1 => nf.volume.dims[1].saturating_sub(1),
            _ => nf.volume.dims[0].saturating_sub(1),
        })
        .unwrap_or(scene.slice_indices[axis_index].saturating_add(128))
}

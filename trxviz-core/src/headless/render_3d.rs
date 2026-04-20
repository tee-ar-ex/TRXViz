use std::path::Path;

use super::gpu_context::{
    GpuSceneResources, TARGET_FORMAT,
};
use super::readback::readback_texture_to_png;
use super::{
    HeadlessRenderData,
};
use crate::data::trx_data::RenderStyle;
use crate::lighting::WorkflowRender3D;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::viewport::ViewportIndex;

#[cfg(feature = "png-export")]
pub(super) fn render_scene3d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
    render_3d: &WorkflowRender3D,
    slice_visible: [bool; 3],
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let viewport_3d: usize = ViewportIndex::Perspective3D.into();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_color"),
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
        label: Some("trxviz_headless_depth"),
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

    let aspect = width as f32 / height.max(1) as f32;
    let view_proj = camera.view_projection(aspect);
    let camera_pos = camera.eye();
    let camera_dir = camera.view_direction();
    let lighting = render_3d.scene_lighting();
    let bounds_radius = ((resources.bounds.max - resources.bounds.min) * 0.5)
        .length()
        .max(1.0);
    let fog_span = (camera.distance + bounds_radius).max(1.0);
    let fog_near = fog_span * render_3d.fog_start_fraction;
    let fog_far = fog_span * render_3d.fog_end_fraction;
    resources.background.update(
        queue,
        &render_3d.background,
        render_3d.exposure,
        render_3d.contrast,
        render_3d.vignette_strength,
    );

    for volume in &render_data.volume_draws {
        if let Some((_, slice)) = resources
            .slices
            .entries
            .iter()
            .find(|(id, _)| *id == volume.file_id)
        {
            slice.update_uniforms(
                queue,
                viewport_3d,
                view_proj,
                volume.window_center,
                volume.window_width,
                volume.colormap,
                volume.opacity,
            );
        }
    }
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
            let aux = if streamline.render_style == RenderStyle::DepthCue {
                300.0
            } else {
                streamline.tube_radius
            };
            resource.update_uniforms(
                queue,
                viewport_3d,
                view_proj,
                camera_pos,
                streamline.render_style as u32,
                glam::Vec3::Z,
                glam::Vec3::ZERO,
                0.0,
                aux,
                lighting,
                render_3d,
                fog_near,
                fog_far,
            );
        }
    }
    for (surface_id, uniform_slot, style) in &render_data.surface_draws {
        resources.meshes.update_surface_uniforms(
            queue,
            *surface_id,
            *uniform_slot,
            view_proj,
            style,
            camera_pos,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    for bundle in &render_data.bundle_draws {
        resources.meshes.update_bundle_uniforms(
            bundle.file_id,
            queue,
            view_proj,
            camera_pos,
            bundle.opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.glyph_visible || render_data.odx_visible {
        resources.glyphs.update_uniforms(
            queue,
            viewport_3d,
            view_proj,
            camera_pos,
            glam::Vec3::Z,
            glam::Vec3::ZERO,
            0.0,
            render_data.glyph_color_mode,
            render_data.glyph_density_3d_step,
            render_data.odf_glyph_opacity,
            render_data.odf_glyph_gloss,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.odx_visible && render_data.odx_fixel_3d_visible {
        resources.fixels_3d.update_uniforms(
            queue,
            viewport_3d,
            view_proj,
            camera_pos,
            glam::Vec3::Z,
            glam::Vec3::ZERO,
            0.0,
            1,
            render_data.fixel_3d_line_width,
            render_data.fixel_3d_opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
        resources.fixels_3d.update_colormap(
            queue,
            viewport_3d,
            render_data.fixel_3d_colormap_code,
            (
                render_data.fixel_3d_scalar_range[0],
                render_data.fixel_3d_scalar_range[1],
            ),
        );
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trxviz_headless_encoder"),
    });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: render_3d.background.bottom_color()[0] as f64,
                        g: render_3d.background.bottom_color()[1] as f64,
                        b: render_3d.background.bottom_color()[2] as f64,
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

        render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        resources.background.paint(render_pass);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources
                .slices
                .entries
                .iter()
                .find(|(id, _)| *id == volume.file_id)
            {
                render_pass.set_pipeline(&slice.pipeline);
                render_pass.set_bind_group(0, slice.bind_group(viewport_3d), &[]);
                render_pass
                    .set_index_buffer(slice.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                for i in 0..3 {
                    if !slice_visible[i] {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, slice.quad_buffers[i].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
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
                    render_pass.set_bind_group(0, resource.bind_group(viewport_3d), &[]);
                    if streamline.render_style == RenderStyle::Tubes {
                        if let (Some(vertices), Some(indices)) =
                            (&resource.tube_vertex_buffer, &resource.tube_index_buffer)
                        {
                            render_pass.set_pipeline(&resource.tube_pipeline);
                            render_pass.set_vertex_buffer(0, vertices.slice(..));
                            render_pass
                                .set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..resource.num_tube_indices, 0, 0..1);
                        }
                    } else {
                        render_pass.set_pipeline(&resource.pipeline);
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
        }

        if !render_data.surface_draws.is_empty() {
            resources
                .meshes
                .paint_opaque(render_pass, &render_data.surface_draws);
        }
        if !render_data.bundle_draws.is_empty() {
            let bundle_draws = render_data
                .bundle_draws
                .iter()
                .map(|draw| (draw.file_id, draw.opacity))
                .collect::<Vec<_>>();
            resources
                .meshes
                .paint_bundle_opaque(render_pass, &bundle_draws);
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &bundle_draws,
                camera_pos,
                camera_dir,
            );
        } else if !render_data.surface_draws.is_empty() {
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &[],
                camera_pos,
                camera_dir,
            );
        }
        if render_data.glyph_visible || render_data.odx_visible {
            resources.glyphs.paint(render_pass, viewport_3d, false);
        }
        if render_data.odx_visible && render_data.odx_fixel_3d_visible {
            resources.fixels_3d.paint(render_pass, viewport_3d, false);
        }
    }

    readback_texture_to_png(device, queue, encoder, &texture, width, height, output_path)
}

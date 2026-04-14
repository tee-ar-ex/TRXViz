use trxviz_core::data::orientation_field::BoundaryGlyphColorMode;
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::lighting::{SceneLightingParams, WorkflowRender3D};
use trxviz_core::renderer::background_renderer::BackgroundResources;
use trxviz_core::renderer::glyph_renderer::GlyphResources;
use trxviz_core::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use trxviz_core::renderer::slice_renderer::AllSliceResources;
use trxviz_core::renderer::streamline_renderer::AllStreamlineResources;

// ── Paint Callbacks ──

#[derive(Clone)]
pub(super) struct VolumeDrawInfo {
    pub file_id: usize,
    pub window_center: f32,
    pub window_width: f32,
    pub colormap: u32,
    pub opacity: f32,
}

#[derive(Clone)]
pub(super) struct StreamlineDrawInfo {
    pub file_id: usize,
    pub visible: bool,
    pub render_style: RenderStyle,
    pub tube_radius: f32,
    pub slab_half_width: f32,
}

#[derive(Clone)]
pub(super) struct BundleDrawInfo {
    pub file_id: usize,
    pub opacity: f32,
}

pub(super) struct Scene3DCallback {
    pub(super) view_proj: glam::Mat4,
    pub(super) camera_pos: glam::Vec3,
    pub(super) camera_dir: glam::Vec3,
    pub(super) streamline_draws: Vec<StreamlineDrawInfo>,
    pub(super) show_streamlines: bool,
    pub(super) volume_draws: Vec<VolumeDrawInfo>,
    pub(super) slice_visible: [bool; 3],
    pub(super) surface_draws: Vec<(usize, usize, MeshDrawStyle)>,
    pub(super) bundle_draws: Vec<BundleDrawInfo>,
    pub(super) show_boundary_glyphs: bool,
    pub(super) boundary_glyph_color_mode: BoundaryGlyphColorMode,
    pub(super) boundary_glyph_draw_step: u32,
    pub(super) scene_lighting: SceneLightingParams,
    pub(super) render_3d: WorkflowRender3D,
    pub(super) fog_near: f32,
    pub(super) fog_far: f32,
}

impl egui_wgpu::CallbackTrait for Scene3DCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(bg) = callback_resources.get_mut::<BackgroundResources>() {
            bg.update(
                queue,
                &self.render_3d.background,
                self.render_3d.exposure,
                self.render_3d.contrast,
                self.render_3d.vignette_strength,
            );
        }
        if let Some(all) = callback_resources.get_mut::<AllStreamlineResources>() {
            for sd in &self.streamline_draws {
                if !sd.visible {
                    continue;
                }
                if let Some((_, res)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                    let aux = if sd.render_style == RenderStyle::DepthCue {
                        300.0
                    } else {
                        sd.tube_radius
                    };
                    res.update_uniforms(
                        queue,
                        0,
                        self.view_proj,
                        self.camera_pos,
                        sd.render_style as u32,
                        3,
                        0.0,
                        0.0,
                        aux,
                        self.scene_lighting,
                        &self.render_3d,
                        self.fog_near,
                        self.fog_far,
                    );
                }
            }
        }
        if let Some(all) = callback_resources.get_mut::<AllSliceResources>() {
            for vd in &self.volume_draws {
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.file_id) {
                    sr.update_uniforms(
                        queue,
                        0,
                        self.view_proj,
                        vd.window_center,
                        vd.window_width,
                        vd.colormap,
                        vd.opacity,
                    );
                }
            }
        }
        if let Some(res) = callback_resources.get_mut::<MeshResources>() {
            for (surface_index, uniform_slot, style) in &self.surface_draws {
                res.update_surface_uniforms(
                    queue,
                    *surface_index,
                    *uniform_slot,
                    self.view_proj,
                    style,
                    self.camera_pos,
                    self.scene_lighting,
                    &self.render_3d,
                    self.fog_near,
                    self.fog_far,
                );
            }
            for bd in &self.bundle_draws {
                res.update_bundle_uniforms(
                    bd.file_id,
                    queue,
                    self.view_proj,
                    self.camera_pos,
                    bd.opacity,
                    self.scene_lighting,
                    &self.render_3d,
                    self.fog_near,
                    self.fog_far,
                );
            }
        }
        if self.show_boundary_glyphs {
            if let Some(gr) = callback_resources.get_mut::<GlyphResources>() {
                gr.update_uniforms(
                    queue,
                    0,
                    self.view_proj,
                    self.camera_pos,
                    3,
                    0.0,
                    0.0,
                    self.boundary_glyph_color_mode,
                    self.boundary_glyph_draw_step,
                    self.scene_lighting,
                    &self.render_3d,
                    self.fog_near,
                    self.fog_far,
                );
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if let Some(bg) = callback_resources.get::<BackgroundResources>() {
            bg.paint(render_pass);
        }

        if let Some(all) = callback_resources.get::<AllSliceResources>() {
            for vd in &self.volume_draws {
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.file_id) {
                    render_pass.set_pipeline(&sr.pipeline);
                    render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                    render_pass.set_index_buffer(
                        sr.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    for i in 0..3 {
                        if !self.slice_visible[i] {
                            continue;
                        }
                        render_pass.set_vertex_buffer(0, sr.quad_buffers[i].slice(..));
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                }
            }
        }

        if self.show_streamlines && !self.streamline_draws.is_empty() {
            if let Some(all) = callback_resources.get::<AllStreamlineResources>() {
                for sd in &self.streamline_draws {
                    if !sd.visible {
                        continue;
                    }
                    if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                        render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                        if sd.render_style == RenderStyle::Tubes {
                            if sr.num_tube_indices == 0 {
                                continue;
                            }
                            if let (Some(tvb), Some(tib)) =
                                (&sr.tube_vertex_buffer, &sr.tube_index_buffer)
                            {
                                render_pass.set_pipeline(&sr.tube_pipeline);
                                render_pass.set_vertex_buffer(0, tvb.slice(..));
                                render_pass
                                    .set_index_buffer(tib.slice(..), wgpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(0..sr.num_tube_indices, 0, 0..1);
                            }
                        } else {
                            if sr.num_indices == 0 {
                                continue;
                            }
                            render_pass.set_pipeline(&sr.pipeline);
                            render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                            render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                            render_pass.set_vertex_buffer(2, sr.tangent_buffer.slice(..));
                            render_pass.set_index_buffer(
                                sr.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
                        }
                    }
                }
            }
        }

        if let Some(mr) = callback_resources.get::<MeshResources>() {
            if !self.surface_draws.is_empty() {
                mr.paint_opaque(render_pass, &self.surface_draws);
            }
            if !self.bundle_draws.is_empty() {
                let bundle_draws: Vec<(usize, f32)> = self
                    .bundle_draws
                    .iter()
                    .map(|draw| (draw.file_id, draw.opacity))
                    .collect();
                mr.paint_bundle_opaque(render_pass, &bundle_draws);
                mr.paint_transparent(
                    render_pass,
                    &self.surface_draws,
                    &bundle_draws,
                    self.camera_pos,
                    self.camera_dir,
                );
            } else if !self.surface_draws.is_empty() {
                mr.paint_transparent(
                    render_pass,
                    &self.surface_draws,
                    &[],
                    self.camera_pos,
                    self.camera_dir,
                );
            }
        }
        if self.show_boundary_glyphs {
            if let Some(gr) = callback_resources.get::<GlyphResources>() {
                gr.paint(render_pass, 0, false);
            }
        }
    }
}

pub(super) struct SliceViewCallback {
    pub(super) view_proj: glam::Mat4,
    pub(super) quad_index: usize,
    pub(super) bind_group_index: usize,
    pub(super) volume_draws: Vec<VolumeDrawInfo>,
    pub(super) streamline_draws: Vec<StreamlineDrawInfo>,
    pub(super) show_streamlines: bool,
    /// Slab clipping axis for streamlines: 0=X, 1=Y, 2=Z.
    pub(super) slab_axis: u32,
    pub(super) slab_min: f32,
    pub(super) slab_max: f32,
    pub(super) show_boundary_glyphs: bool,
    pub(super) boundary_glyph_color_mode: BoundaryGlyphColorMode,
    pub(super) boundary_glyph_draw_step: u32,
    pub(super) scene_lighting: SceneLightingParams,
}

impl egui_wgpu::CallbackTrait for SliceViewCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let neutral_render = WorkflowRender3D {
            vignette_strength: 0.0,
            exposure: 1.0,
            contrast: 1.0,
            ..Default::default()
        };
        if let Some(all) = callback_resources.get_mut::<AllSliceResources>() {
            for vd in &self.volume_draws {
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.file_id) {
                    sr.update_uniforms(
                        queue,
                        self.bind_group_index,
                        self.view_proj,
                        vd.window_center,
                        vd.window_width,
                        vd.colormap,
                        vd.opacity,
                    );
                }
            }
        }
        if let Some(all) = callback_resources.get_mut::<AllStreamlineResources>() {
            for sd in &self.streamline_draws {
                if !sd.visible {
                    continue;
                }
                if let Some((_, res)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                    let slab_min = self.slab_min - sd.slab_half_width;
                    let slab_max = self.slab_max + sd.slab_half_width;
                    // Slice views always render flat lines regardless of 3D render style.
                    res.update_uniforms(
                        queue,
                        self.bind_group_index,
                        self.view_proj,
                        glam::Vec3::ZERO,
                        0, // flat
                        self.slab_axis,
                        slab_min,
                        slab_max,
                        0.5,
                        self.scene_lighting,
                        &neutral_render,
                        0.0,
                        1.0,
                    );
                }
            }
        }
        if self.show_boundary_glyphs {
            if let Some(gr) = callback_resources.get_mut::<GlyphResources>() {
                gr.update_uniforms(
                    queue,
                    self.bind_group_index,
                    self.view_proj,
                    glam::Vec3::ZERO,
                    self.slab_axis,
                    self.slab_min,
                    self.slab_max,
                    self.boundary_glyph_color_mode,
                    self.boundary_glyph_draw_step,
                    self.scene_lighting,
                    &neutral_render,
                    0.0,
                    1.0,
                );
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if let Some(all) = callback_resources.get::<AllSliceResources>() {
            for vd in &self.volume_draws {
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.file_id) {
                    render_pass.set_pipeline(&sr.pipeline);
                    render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                    render_pass.set_index_buffer(
                        sr.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass.set_vertex_buffer(0, sr.quad_buffers[self.quad_index].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }

        if self.show_streamlines && !self.streamline_draws.is_empty() {
            if let Some(all) = callback_resources.get::<AllStreamlineResources>() {
                for sd in &self.streamline_draws {
                    if !sd.visible {
                        continue;
                    }
                    if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                        if sr.num_indices == 0 {
                            continue;
                        }
                        render_pass.set_pipeline(&sr.slice_pipeline);
                        render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                        render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, sr.tangent_buffer.slice(..));
                        render_pass
                            .set_index_buffer(sr.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
                    }
                }
            }
        }
        if self.show_boundary_glyphs {
            if let Some(gr) = callback_resources.get::<GlyphResources>() {
                gr.paint(render_pass, self.bind_group_index, true);
            }
        }
    }
}

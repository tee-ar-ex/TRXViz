use trxviz_core::data::orientation_field::BoundaryGlyphColorMode;
use trxviz_core::data::trx_data::RenderStyle;
use trxviz_core::lighting::{SceneLightingParams, WorkflowRender3D};
use trxviz_core::renderer::background_renderer::BackgroundResources;
use trxviz_core::renderer::fixel_renderer::FixelResources;
use trxviz_core::renderer::glyph_renderer::GlyphResources;
use trxviz_core::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use trxviz_core::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResourceKind};
use trxviz_core::renderer::streamline_renderer::AllStreamlineResources;
use trxviz_core::renderer::viewport::ViewportIndex;

fn glyph_colormap_code(cm: trxviz_core::workflow::GlyphColormap) -> u32 {
    use trxviz_core::workflow::GlyphColormap as G;
    match cm {
        G::Directional => 0,
        G::Plasma => 2,
        G::Viridis => 3,
        G::Inferno => 4,
        G::BlueWhiteRed => 5,
    }
}

pub(in crate::app) struct OdxFixelResources {
    pub(in crate::app) resources_3d: FixelResources,
    pub(in crate::app) resources_2d: FixelResources,
}

impl OdxFixelResources {
    pub(in crate::app) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self {
            resources_3d: FixelResources::new(device, target_format),
            resources_2d: FixelResources::new(device, target_format),
        }
    }

    pub(in crate::app) fn clear(&mut self) {
        self.resources_3d.clear();
        self.resources_2d.clear();
    }
}

// ── Paint Callbacks ──

#[derive(Clone)]
pub(super) struct VolumeDrawInfo {
    pub slice_key: usize,
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
    pub opacity: f32,
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
    pub(super) show_odx_glyphs: bool,
    pub(super) show_odx_fixels: bool,
    pub(super) odx_glyph_opacity: f32,
    pub(super) odx_glyph_gloss: f32,
    pub(super) odx_glyph_scale: f32,
    pub(super) odx_glyph_colormap: trxviz_core::workflow::GlyphColormap,
    pub(super) odx_opacity_gate: [f32; 4],
    pub(super) odx_size_gate: [f32; 4],
    pub(super) odx_amp_norm: f32,
    pub(super) odx_fixel_line_width: f32,
    pub(super) odx_fixel_opacity: f32,
    pub(super) odx_fixel_opacity_gate: [f32; 4],
    pub(super) odx_fixel_length_scale: f32,
    pub(super) odx_fixel_visible: bool,
    pub(super) odx_fixel_colormap_code: u32,
    pub(super) odx_fixel_scalar_range: [f32; 2],
    /// Slab parameters that clip the 3D fixel pass to the current ODF slice.
    pub(super) odx_fixel_3d_slab_normal: glam::Vec3,
    pub(super) odx_fixel_3d_slab_center: glam::Vec3,
    pub(super) odx_fixel_3d_slab_half_width: f32,
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
                        glam::Vec3::Z,    // slab_normal (irrelevant — slab disabled)
                        glam::Vec3::ZERO, // slab_center
                        0.0,              // slab_half_width = 0 → disabled
                        aux,
                        sd.opacity,
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
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.slice_key) {
                    match sr {
                        SliceResourceKind::Scalar(s) => s.update_uniforms(
                            queue,
                            0,
                            self.view_proj,
                            vd.window_center,
                            vd.window_width,
                            vd.colormap,
                            vd.opacity,
                        ),
                        SliceResourceKind::Composite(c) => {
                            c.update_uniforms(queue, 0, self.view_proj, vd.opacity);
                        }
                    }
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
        if self.show_boundary_glyphs || self.show_odx_glyphs {
            if let Some(gr) = callback_resources.get_mut::<GlyphResources>() {
                let viewport_3d: usize = ViewportIndex::Perspective3D.into();
                let color_mode = if self.show_odx_glyphs {
                    BoundaryGlyphColorMode::DirectionRgb
                } else {
                    self.boundary_glyph_color_mode
                };
                let draw_step = if self.show_odx_glyphs {
                    1
                } else {
                    self.boundary_glyph_draw_step
                };
                let (glyph_op, glyph_gl) = if self.show_odx_glyphs {
                    (self.odx_glyph_opacity, self.odx_glyph_gloss)
                } else {
                    (0.95, 0.0)
                };
                gr.update_uniforms(
                    queue,
                    viewport_3d,
                    self.view_proj,
                    self.camera_pos,
                    glam::Vec3::Z,    // slab_normal (irrelevant — slab disabled)
                    glam::Vec3::ZERO, // slab_center
                    0.0,              // slab_half_width = 0 → disabled
                    color_mode,
                    draw_step,
                    glyph_op,
                    glyph_gl,
                    self.scene_lighting,
                    &self.render_3d,
                    self.fog_near,
                    self.fog_far,
                );
                let scale_mul = if self.show_odx_glyphs {
                    self.odx_glyph_scale
                } else {
                    0.0
                };
                gr.update_scale_mul(queue, viewport_3d, scale_mul);
                if self.show_odx_glyphs {
                    gr.update_color_mode(
                        queue,
                        viewport_3d,
                        glyph_colormap_code(self.odx_glyph_colormap),
                    );
                    gr.update_amp_norm(queue, viewport_3d, self.odx_amp_norm);
                    gr.update_opacity_gate(queue, viewport_3d, self.odx_opacity_gate);
                    gr.update_size_gate(queue, viewport_3d, self.odx_size_gate);
                }
            }
        }
        if self.show_odx_fixels && self.odx_fixel_visible {
            if let Some(fr) = callback_resources.get_mut::<OdxFixelResources>() {
                let viewport_3d: usize = ViewportIndex::Perspective3D.into();
                fr.resources_3d.update_uniforms(
                    queue,
                    viewport_3d,
                    self.view_proj,
                    self.camera_pos,
                    self.odx_fixel_3d_slab_normal,
                    self.odx_fixel_3d_slab_center,
                    self.odx_fixel_3d_slab_half_width,
                    1, // draw_step
                    self.odx_fixel_line_width,
                    self.odx_fixel_opacity,
                    self.odx_fixel_opacity_gate,
                    self.scene_lighting,
                    &self.render_3d,
                    self.fog_near,
                    self.fog_far,
                );
                fr.resources_3d
                    .update_length_mul(queue, viewport_3d, self.odx_fixel_length_scale);
                fr.resources_3d.update_colormap(
                    queue,
                    viewport_3d,
                    self.odx_fixel_colormap_code,
                    (
                        self.odx_fixel_scalar_range[0],
                        self.odx_fixel_scalar_range[1],
                    ),
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
            let viewport_3d: usize = ViewportIndex::Perspective3D.into();
            for vd in &self.volume_draws {
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.slice_key) {
                    match sr {
                        SliceResourceKind::Scalar(s) => {
                            render_pass.set_pipeline(&s.pipeline);
                            render_pass.set_bind_group(0, s.bind_group(viewport_3d), &[]);
                            render_pass.set_index_buffer(
                                s.quad_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            for i in 0..3 {
                                if !self.slice_visible[i] {
                                    continue;
                                }
                                render_pass.set_vertex_buffer(0, s.quad_buffers[i].slice(..));
                                render_pass.draw_indexed(0..6, 0, 0..1);
                            }
                        }
                        SliceResourceKind::Composite(c) => {
                            render_pass.set_pipeline(&c.pipeline);
                            render_pass.set_index_buffer(
                                c.quad_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            for i in 0..3 {
                                if !self.slice_visible[i] {
                                    continue;
                                }
                                let axis = match i {
                                    0 => SliceAxis::Axial,
                                    1 => SliceAxis::Coronal,
                                    _ => SliceAxis::Sagittal,
                                };
                                render_pass.set_bind_group(0, c.bind_group(viewport_3d, axis), &[]);
                                render_pass.set_vertex_buffer(0, c.quad_buffers[i].slice(..));
                                render_pass.draw_indexed(0..6, 0, 0..1);
                            }
                        }
                    }
                }
            }
        }

        if self.show_streamlines && !self.streamline_draws.is_empty() {
            if let Some(all) = callback_resources.get::<AllStreamlineResources>() {
                let viewport_3d: usize = ViewportIndex::Perspective3D.into();
                for sd in &self.streamline_draws {
                    if !sd.visible {
                        continue;
                    }
                    if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                        render_pass.set_bind_group(0, sr.bind_group(viewport_3d), &[]);
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
        if self.show_boundary_glyphs || self.show_odx_glyphs {
            if let Some(gr) = callback_resources.get::<GlyphResources>() {
                gr.paint(render_pass, ViewportIndex::Perspective3D.into(), false);
            }
        }
        if self.show_odx_fixels && self.odx_fixel_visible {
            if let Some(fr) = callback_resources.get::<OdxFixelResources>() {
                fr.resources_3d
                    .paint(render_pass, ViewportIndex::Perspective3D.into(), false);
            }
        }
    }
}

pub(super) struct SliceViewCallback {
    pub(super) view_proj: glam::Mat4,
    pub(super) quad_index: usize,
    pub(super) viewport: ViewportIndex,
    pub(super) volume_draws: Vec<VolumeDrawInfo>,
    pub(super) streamline_draws: Vec<StreamlineDrawInfo>,
    pub(super) show_streamlines: bool,
    /// Slab clipping plane (works for both axis-aligned and oblique volumes).
    pub(super) slab_normal: glam::Vec3,
    pub(super) slab_center: glam::Vec3,
    pub(super) slab_half_width: f32,
    pub(super) show_boundary_glyphs: bool,
    pub(super) boundary_glyph_color_mode: BoundaryGlyphColorMode,
    pub(super) boundary_glyph_draw_step: u32,
    pub(super) show_odx_glyphs: bool,
    pub(super) show_odx_fixels: bool,
    pub(super) odx_glyph_opacity: f32,
    pub(super) odx_glyph_gloss: f32,
    pub(super) odx_fixel_line_width: f32,
    pub(super) odx_fixel_opacity: f32,
    pub(super) odx_fixel_opacity_gate: [f32; 4],
    pub(super) odx_fixel_slab_half_width_mm: f32,
    pub(super) odx_glyph_scale: f32,
    pub(super) odx_fixel_length_scale: f32,
    pub(super) odx_fixel_visible: bool,
    pub(super) odx_fixel_colormap_code: u32,
    pub(super) odx_fixel_scalar_range: [f32; 2],
    pub(super) odx_glyph_colormap: trxviz_core::workflow::GlyphColormap,
    pub(super) odx_opacity_gate: [f32; 4],
    pub(super) odx_size_gate: [f32; 4],
    pub(super) odx_amp_norm: f32,
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
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.slice_key) {
                    match sr {
                        SliceResourceKind::Scalar(s) => s.update_uniforms(
                            queue,
                            self.viewport.into(),
                            self.view_proj,
                            vd.window_center,
                            vd.window_width,
                            vd.colormap,
                            vd.opacity,
                        ),
                        SliceResourceKind::Composite(c) => c.update_uniforms(
                            queue,
                            self.viewport.into(),
                            self.view_proj,
                            vd.opacity,
                        ),
                    }
                }
            }
        }
        if let Some(all) = callback_resources.get_mut::<AllStreamlineResources>() {
            for sd in &self.streamline_draws {
                if !sd.visible {
                    continue;
                }
                if let Some((_, res)) = all.entries.iter().find(|(id, _)| *id == sd.file_id) {
                    // Expand the slab half-width by the per-streamline bundle width.
                    let hw = self.slab_half_width + sd.slab_half_width;
                    // Slice views always render flat lines regardless of 3D render style.
                    res.update_uniforms(
                        queue,
                        self.viewport.into(),
                        self.view_proj,
                        glam::Vec3::ZERO,
                        0, // flat
                        self.slab_normal,
                        self.slab_center,
                        hw,
                        0.5,
                        sd.opacity,
                        self.scene_lighting,
                        &neutral_render,
                        0.0,
                        1.0,
                    );
                }
            }
        }
        if self.show_boundary_glyphs || self.show_odx_glyphs {
            if let Some(gr) = callback_resources.get_mut::<GlyphResources>() {
                let color_mode = if self.show_odx_glyphs {
                    BoundaryGlyphColorMode::DirectionRgb
                } else {
                    self.boundary_glyph_color_mode
                };
                let draw_step = if self.show_odx_glyphs {
                    1
                } else {
                    self.boundary_glyph_draw_step
                };
                let (glyph_op, glyph_gl) = if self.show_odx_glyphs {
                    (self.odx_glyph_opacity, self.odx_glyph_gloss)
                } else {
                    (0.95, 0.0)
                };
                gr.update_uniforms(
                    queue,
                    self.viewport.into(),
                    self.view_proj,
                    glam::Vec3::ZERO,
                    self.slab_normal,
                    self.slab_center,
                    self.slab_half_width,
                    color_mode,
                    draw_step,
                    glyph_op,
                    glyph_gl,
                    self.scene_lighting,
                    &neutral_render,
                    0.0,
                    1.0,
                );
                let scale_mul = if self.show_odx_glyphs {
                    self.odx_glyph_scale
                } else {
                    0.0
                };
                gr.update_scale_mul(queue, self.viewport.into(), scale_mul);
                if self.show_odx_glyphs {
                    gr.update_color_mode(
                        queue,
                        self.viewport.into(),
                        glyph_colormap_code(self.odx_glyph_colormap),
                    );
                    gr.update_amp_norm(queue, self.viewport.into(), self.odx_amp_norm);
                    gr.update_opacity_gate(queue, self.viewport.into(), self.odx_opacity_gate);
                    gr.update_size_gate(queue, self.viewport.into(), self.odx_size_gate);
                }
            }
        }
        if self.show_odx_fixels && self.odx_fixel_visible {
            if let Some(fr) = callback_resources.get_mut::<OdxFixelResources>() {
                fr.resources_2d.update_uniforms(
                    queue,
                    self.viewport.into(),
                    self.view_proj,
                    glam::Vec3::ZERO,
                    self.slab_normal,
                    self.slab_center,
                    self.odx_fixel_slab_half_width_mm,
                    1, // draw_step
                    self.odx_fixel_line_width,
                    self.odx_fixel_opacity,
                    self.odx_fixel_opacity_gate,
                    self.scene_lighting,
                    &WorkflowRender3D {
                        vignette_strength: 0.0,
                        exposure: 1.0,
                        contrast: 1.0,
                        ..Default::default()
                    },
                    0.0,
                    1.0,
                );
                fr.resources_2d.update_length_mul(
                    queue,
                    self.viewport.into(),
                    self.odx_fixel_length_scale,
                );
                fr.resources_2d.update_colormap(
                    queue,
                    self.viewport.into(),
                    self.odx_fixel_colormap_code,
                    (
                        self.odx_fixel_scalar_range[0],
                        self.odx_fixel_scalar_range[1],
                    ),
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
                if let Some((_, sr)) = all.entries.iter().find(|(id, _)| *id == vd.slice_key) {
                    let axis = match self.quad_index {
                        0 => SliceAxis::Axial,
                        1 => SliceAxis::Coronal,
                        _ => SliceAxis::Sagittal,
                    };
                    match sr {
                        SliceResourceKind::Scalar(s) => {
                            render_pass.set_pipeline(&s.pipeline);
                            render_pass.set_bind_group(0, s.bind_group(self.viewport.into()), &[]);
                            render_pass.set_index_buffer(
                                s.quad_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            render_pass
                                .set_vertex_buffer(0, s.quad_buffers[self.quad_index].slice(..));
                            render_pass.draw_indexed(0..6, 0, 0..1);
                        }
                        SliceResourceKind::Composite(c) => {
                            render_pass.set_pipeline(&c.pipeline);
                            render_pass.set_bind_group(
                                0,
                                c.bind_group(self.viewport.into(), axis),
                                &[],
                            );
                            render_pass.set_index_buffer(
                                c.quad_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            render_pass
                                .set_vertex_buffer(0, c.quad_buffers[self.quad_index].slice(..));
                            render_pass.draw_indexed(0..6, 0, 0..1);
                        }
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
                        if sr.num_indices == 0 {
                            continue;
                        }
                        render_pass.set_pipeline(&sr.slice_pipeline);
                        render_pass.set_bind_group(0, sr.bind_group(self.viewport.into()), &[]);
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
        if self.show_boundary_glyphs || self.show_odx_glyphs {
            if let Some(gr) = callback_resources.get::<GlyphResources>() {
                gr.paint(render_pass, self.viewport.into(), true);
            }
        }
        if self.show_odx_fixels && self.odx_fixel_visible {
            if let Some(fr) = callback_resources.get::<OdxFixelResources>() {
                fr.resources_2d
                    .paint(render_pass, self.viewport.into(), true);
            }
        }
    }
}

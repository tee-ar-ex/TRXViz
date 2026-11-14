use std::path::Path;

use anyhow::{Context, anyhow};
use glam::Vec3;
use image::ColorType;

use crate::scene::{HeadlessScene, HeadlessWorkflowState};

#[cfg(feature = "png-export")]
pub(super) struct OverlaySegment2D {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub rgba: [u8; 4],
    pub thickness: f32,
    pub clip_rect: [u32; 4],
}

#[cfg(feature = "png-export")]
fn tri_axis_value(p: Vec3, axis_index: usize) -> f32 {
    match axis_index {
        0 => p.z,
        1 => p.y,
        _ => p.x,
    }
}

#[cfg(feature = "png-export")]
fn intersect_edge_with_slice_t(
    p0: Vec3,
    p1: Vec3,
    axis_index: usize,
    slice_pos: f32,
    eps: f32,
) -> Option<(Vec3, f32)> {
    let c0 = tri_axis_value(p0, axis_index);
    let c1 = tri_axis_value(p1, axis_index);
    let d0 = c0 - slice_pos;
    let d1 = c1 - slice_pos;
    if d0.abs() <= eps && d1.abs() <= eps {
        return None;
    }
    if d0.abs() <= eps {
        return Some((p0, 0.0));
    }
    if d1.abs() <= eps {
        return Some((p1, 1.0));
    }
    if d0 * d1 > 0.0 {
        return None;
    }
    let t = d0 / (d0 - d1);
    Some((p0 + (p1 - p0) * t, t))
}

#[cfg(feature = "png-export")]
fn lerp_rgba(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

#[cfg(feature = "png-export")]
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

#[cfg(feature = "png-export")]
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_slice_contour_segments(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
    axis_index: usize,
    slice_pos: f32,
    view_proj: glam::Mat4,
    clip_rect: [u32; 4],
    camera_pos_3d: Option<Vec3>,
    slice_visible_3d: [bool; 3],
    slice_positions_3d: [f32; 3],
) -> Vec<OverlaySegment2D> {
    let mut out = Vec::new();
    let eps = 1e-4f32;
    let first_nifti = scene.nifti_files.first().map(|nf| &nf.volume);

    let is_occluded_3d = |p: Vec3| -> bool {
        let Some(cam) = camera_pos_3d else {
            return false;
        };
        for other_axis in 0..3 {
            if other_axis == axis_index || !slice_visible_3d[other_axis] {
                continue;
            }
            let other_pos = slice_positions_3d[other_axis];
            let cam_c = tri_axis_value(cam, other_axis);
            let p_c = tri_axis_value(p, other_axis);
            let d_cam = cam_c - other_pos;
            let d_p = p_c - other_pos;
            if d_cam * d_p < 0.0 && (cam_c - p_c).abs() > 1e-5 {
                let t_hit = d_cam / (cam_c - p_c);
                if t_hit > 0.0 && t_hit < 1.0 {
                    let p_hit = cam + (p - cam) * t_hit;
                    if let Some(vol) = first_nifti {
                        if vol.sample_world_nearest(p_hit) > 1e-6 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    };

    let project = |world: Vec3| -> Option<[f32; 2]> {
        let clip = view_proj * world.extend(1.0);
        if clip.w <= 1e-5 {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        Some([
            clip_rect[0] as f32 + (ndc_x + 1.0) * 0.5 * clip_rect[2] as f32,
            clip_rect[1] as f32 + (1.0 - ndc_y) * 0.5 * clip_rect[3] as f32,
        ])
    };

    for draw in workflow
        .runtime
        .scene_plan
        .surface_draws(crate::workflow::SurfaceDisplaySpace::Anatomical)
    {
        if draw.opacity <= 0.01 || draw.outline_thickness <= 0.01 {
            continue;
        }
        let Some(surface) = scene
            .gifti_surfaces
            .iter()
            .find(|surface| surface.id == draw.source_id)
        else {
            continue;
        };

        let (smin, smax) = match axis_index {
            0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
            1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
            _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
        };
        if slice_pos < smin - eps || slice_pos > smax + eps {
            continue;
        }

        let fallback_rgba = [
            draw.outline_color[0],
            draw.outline_color[1],
            draw.outline_color[2],
            1.0,
        ];
        let vertex_colors = surface
            .data
            .vertex_colors
            .as_deref()
            .filter(|vc| vc.len() == surface.data.vertices.len());
        let thickness = draw.outline_thickness.clamp(0.25, 8.0);
        let contour_opacity = draw.opacity.max(0.92).clamp(0.0, 1.0);

        for tri in surface.data.indices.chunks_exact(3) {
            let ia = tri[0] as usize;
            let ib = tri[1] as usize;
            let ic = tri[2] as usize;
            let a = Vec3::from(surface.data.vertices[ia]);
            let b = Vec3::from(surface.data.vertices[ib]);
            let c = Vec3::from(surface.data.vertices[ic]);

            let tmin = tri_axis_value(a, axis_index)
                .min(tri_axis_value(b, axis_index))
                .min(tri_axis_value(c, axis_index));
            let tmax = tri_axis_value(a, axis_index)
                .max(tri_axis_value(b, axis_index))
                .max(tri_axis_value(c, axis_index));
            if slice_pos < tmin - eps || slice_pos > tmax + eps {
                continue;
            }

            let mut pts: Vec<(Vec3, [f32; 4])> = Vec::with_capacity(3);
            for ((p0, p1), (i0, i1)) in [((a, b), (ia, ib)), ((b, c), (ib, ic)), ((c, a), (ic, ia))]
            {
                let Some((p, t)) = intersect_edge_with_slice_t(p0, p1, axis_index, slice_pos, eps)
                else {
                    continue;
                };
                if pts
                    .iter()
                    .any(|(q, _)| (*q - p).length_squared() <= eps * eps)
                {
                    continue;
                }
                let col = match vertex_colors {
                    Some(vc) => lerp_rgba(vc[i0], vc[i1], t),
                    None => fallback_rgba,
                };
                pts.push((p, col));
            }
            if pts.len() < 2 {
                continue;
            }
            let (e0, e1) = if pts.len() == 2 {
                (pts[0], pts[1])
            } else {
                let mut best = (pts[0], pts[1]);
                let mut best_d2 = (pts[1].0 - pts[0].0).length_squared();
                for i in 0..pts.len() {
                    for j in (i + 1)..pts.len() {
                        let d2 = (pts[j].0 - pts[i].0).length_squared();
                        if d2 > best_d2 {
                            best = (pts[i], pts[j]);
                            best_d2 = d2;
                        }
                    }
                }
                best
            };

            if is_occluded_3d(0.5 * (e0.0 + e1.0)) {
                continue;
            }
            let (Some(sp0), Some(sp1)) = (project(e0.0), project(e1.0)) else {
                continue;
            };
            let seg = [
                0.5 * (e0.1[0] + e1.1[0]),
                0.5 * (e0.1[1] + e1.1[1]),
                0.5 * (e0.1[2] + e1.1[2]),
                0.5 * (e0.1[3] + e1.1[3]),
            ];
            let rgba = if vertex_colors.is_some() {
                [
                    linear_to_srgb_u8(seg[0]),
                    linear_to_srgb_u8(seg[1]),
                    linear_to_srgb_u8(seg[2]),
                    (seg[3].clamp(0.0, 1.0) * contour_opacity * 255.0).round() as u8,
                ]
            } else {
                [
                    (seg[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (seg[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (seg[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (seg[3].clamp(0.0, 1.0) * contour_opacity * 255.0).round() as u8,
                ]
            };
            out.push(OverlaySegment2D {
                p0: sp0,
                p1: sp1,
                rgba,
                thickness,
                clip_rect,
            });
        }
    }

    for draw in workflow
        .runtime
        .scene_plan
        .draws
        .of_type::<crate::workflow::BundleDrawPlan>()
    {
        if draw.opacity <= 0.01 || draw.outline_thickness <= 0.01 {
            continue;
        }
        let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
        else {
            continue;
        };
        let thickness = draw.outline_thickness.clamp(0.25, 8.0);
        let contour_opacity = draw.opacity.max(0.92).clamp(0.0, 1.0);

        for (mesh, _) in &cache.meshes {
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }
            let mut bbox_min = Vec3::splat(f32::INFINITY);
            let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);
            for vertex in &mesh.vertices {
                let pos = Vec3::from(vertex.position);
                bbox_min = bbox_min.min(pos);
                bbox_max = bbox_max.max(pos);
            }
            let (smin, smax) = match axis_index {
                0 => (bbox_min.z, bbox_max.z),
                1 => (bbox_min.y, bbox_max.y),
                _ => (bbox_min.x, bbox_max.x),
            };
            if slice_pos < smin - eps || slice_pos > smax + eps {
                continue;
            }

            for tri in mesh.indices.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                let a = Vec3::from(mesh.vertices[ia].position);
                let b = Vec3::from(mesh.vertices[ib].position);
                let c = Vec3::from(mesh.vertices[ic].position);

                let tmin = tri_axis_value(a, axis_index)
                    .min(tri_axis_value(b, axis_index))
                    .min(tri_axis_value(c, axis_index));
                let tmax = tri_axis_value(a, axis_index)
                    .max(tri_axis_value(b, axis_index))
                    .max(tri_axis_value(c, axis_index));
                if slice_pos < tmin - eps || slice_pos > tmax + eps {
                    continue;
                }

                let mut pts: Vec<Vec3> = Vec::with_capacity(3);
                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                    if let Some((p, _)) =
                        intersect_edge_with_slice_t(p0, p1, axis_index, slice_pos, eps)
                    {
                        if !pts.iter().any(|q| (*q - p).length_squared() <= eps * eps) {
                            pts.push(p);
                        }
                    }
                }
                if pts.len() < 2 {
                    continue;
                }
                let (p0, p1) = if pts.len() == 2 {
                    (pts[0], pts[1])
                } else {
                    let mut best = (pts[0], pts[1]);
                    let mut best_d2 = (pts[1] - pts[0]).length_squared();
                    for i in 0..pts.len() {
                        for j in (i + 1)..pts.len() {
                            let d2 = (pts[j] - pts[i]).length_squared();
                            if d2 > best_d2 {
                                best = (pts[i], pts[j]);
                                best_d2 = d2;
                            }
                        }
                    }
                    best
                };

                if is_occluded_3d(0.5 * (p0 + p1)) {
                    continue;
                }
                let (Some(sp0), Some(sp1)) = (project(p0), project(p1)) else {
                    continue;
                };
                let rgb = [
                    (mesh.vertices[ia].color[0]
                        + mesh.vertices[ib].color[0]
                        + mesh.vertices[ic].color[0])
                        / 3.0,
                    (mesh.vertices[ia].color[1]
                        + mesh.vertices[ib].color[1]
                        + mesh.vertices[ic].color[1])
                        / 3.0,
                    (mesh.vertices[ia].color[2]
                        + mesh.vertices[ib].color[2]
                        + mesh.vertices[ic].color[2])
                        / 3.0,
                ];
                let rgba = [
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (contour_opacity * 255.0).round() as u8,
                ];
                out.push(OverlaySegment2D {
                    p0: sp0,
                    p1: sp1,
                    rgba,
                    thickness,
                    clip_rect,
                });
            }
        }
    }

    out
}

#[cfg(feature = "png-export")]
fn rasterize_overlays(rgba: &mut [u8], width: u32, height: u32, overlays: &[OverlaySegment2D]) {
    let w_i = width as i32;
    let h_i = height as i32;
    for seg in overlays {
        let half_t = 0.5 * seg.thickness.max(0.5);
        let feather = 0.75f32;
        let r_max = half_t + feather;
        let min_x = (seg.p0[0].min(seg.p1[0]) - r_max).floor() as i32;
        let max_x = (seg.p0[0].max(seg.p1[0]) + r_max).ceil() as i32;
        let min_y = (seg.p0[1].min(seg.p1[1]) - r_max).floor() as i32;
        let max_y = (seg.p0[1].max(seg.p1[1]) + r_max).ceil() as i32;

        let cx0 = (seg.clip_rect[0] as i32).max(0);
        let cy0 = (seg.clip_rect[1] as i32).max(0);
        let cx1 = ((seg.clip_rect[0] + seg.clip_rect[2]) as i32).min(w_i);
        let cy1 = ((seg.clip_rect[1] + seg.clip_rect[3]) as i32).min(h_i);

        let x0 = min_x.max(cx0);
        let x1 = max_x.min(cx1 - 1);
        let y0 = min_y.max(cy0);
        let y1 = max_y.min(cy1 - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }

        let dx = seg.p1[0] - seg.p0[0];
        let dy = seg.p1[1] - seg.p0[1];
        let len2 = dx * dx + dy * dy;
        let base_a = seg.rgba[3] as f32 / 255.0;

        for py in y0..=y1 {
            let fy = py as f32 + 0.5;
            for px in x0..=x1 {
                let fx = px as f32 + 0.5;
                let t = if len2 > 1e-6 {
                    (((fx - seg.p0[0]) * dx + (fy - seg.p0[1]) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let qx = seg.p0[0] + t * dx;
                let qy = seg.p0[1] + t * dy;
                let dist = ((fx - qx) * (fx - qx) + (fy - qy) * (fy - qy)).sqrt();
                if dist >= r_max {
                    continue;
                }
                let cov = if dist <= half_t - feather * 0.5 {
                    1.0
                } else {
                    ((r_max - dist) / (1.5 * feather)).clamp(0.0, 1.0)
                };
                let a = (base_a * cov).clamp(0.0, 1.0);
                if a <= 1e-3 {
                    continue;
                }
                let idx = ((py as usize) * (width as usize) + (px as usize)) * 4;
                for c in 0..3 {
                    let dst = rgba[idx + c] as f32;
                    let src = seg.rgba[c] as f32;
                    rgba[idx + c] = (src * a + dst * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
                }
                rgba[idx + 3] = 255;
            }
        }
    }
}

#[cfg(feature = "png-export")]
#[allow(dead_code)]
pub(super) fn readback_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    readback_texture_to_png_with_overlays(
        device,
        queue,
        encoder,
        texture,
        width,
        height,
        &[],
        output_path,
    )
}

#[cfg(feature = "png-export")]
#[allow(clippy::too_many_arguments)]
pub(super) fn readback_texture_to_png_with_overlays(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    overlays: &[OverlaySegment2D],
    output_path: &Path,
) -> anyhow::Result<()> {
    let padded_bytes_per_row = ((width * 4 + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
        / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trxviz_headless_readback"),
        size: padded_bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| anyhow!("failed to receive GPU readback status"))?
        .map_err(|err| anyhow!("failed to map render output: {err}"))?;

    let mapped = buffer_slice.get_mapped_range();
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * width as usize * 4;
        rgba[dst_offset..dst_offset + width as usize * 4]
            .copy_from_slice(&mapped[src_offset..src_offset + width as usize * 4]);
    }
    drop(mapped);
    output_buffer.unmap();

    if !overlays.is_empty() {
        rasterize_overlays(&mut rgba, width, height, overlays);
    }

    image::save_buffer(output_path, &rgba, width, height, ColorType::Rgba8)
        .with_context(|| format!("saving PNG to {}", output_path.display()))?;
    Ok(())
}

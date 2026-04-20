use std::path::Path;

use anyhow::{Context, anyhow};
use image::ColorType;

#[cfg(feature = "png-export")]
pub(super) fn readback_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
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

    image::save_buffer(output_path, &rgba, width, height, ColorType::Rgba8)
        .with_context(|| format!("saving PNG to {}", output_path.display()))?;
    Ok(())
}

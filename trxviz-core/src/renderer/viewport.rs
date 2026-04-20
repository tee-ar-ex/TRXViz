use std::marker::PhantomData;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub const NUM_VIEWPORTS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportIndex {
    Perspective3D = 0,
    SliceAxial = 1,
    SliceCoronal = 2,
    SliceSagittal = 3,
}

impl From<ViewportIndex> for usize {
    fn from(value: ViewportIndex) -> Self {
        value as usize
    }
}

pub struct ViewportUniformSet<U> {
    buffers: [wgpu::Buffer; NUM_VIEWPORTS],
    bind_groups: [wgpu::BindGroup; NUM_VIEWPORTS],
    _marker: PhantomData<U>,
}

impl<U: Pod> ViewportUniformSet<U> {
    pub fn new<F>(
        device: &wgpu::Device,
        label_prefix: &str,
        initial_uniforms: &U,
        mut build_bind_group: F,
    ) -> Self
    where
        F: FnMut(usize, &wgpu::Buffer) -> wgpu::BindGroup,
    {
        let buffers = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label_prefix}_{}", viewport_label(i))),
                contents: bytemuck::bytes_of(initial_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let bind_groups = std::array::from_fn(|i| build_bind_group(i, &buffers[i]));
        Self {
            buffers,
            bind_groups,
            _marker: PhantomData,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, viewport: usize, uniforms: &U) {
        queue.write_buffer(&self.buffers[viewport], 0, bytemuck::bytes_of(uniforms));
    }

    pub fn write(&self, queue: &wgpu::Queue, viewport: usize, offset: u64, bytes: &[u8]) {
        queue.write_buffer(&self.buffers[viewport], offset, bytes);
    }

    pub fn bind_group(&self, viewport: usize) -> &wgpu::BindGroup {
        &self.bind_groups[viewport]
    }

    pub fn buffer(&self, viewport: usize) -> &wgpu::Buffer {
        &self.buffers[viewport]
    }

    pub fn rebuild_bind_groups<F>(&mut self, mut build_bind_group: F)
    where
        F: FnMut(usize, &wgpu::Buffer) -> wgpu::BindGroup,
    {
        self.bind_groups = std::array::from_fn(|i| build_bind_group(i, &self.buffers[i]));
    }
}

fn viewport_label(index: usize) -> &'static str {
    match index {
        0 => "3d",
        1 => "axial",
        2 => "coronal",
        3 => "sagittal",
        _ => "unknown",
    }
}

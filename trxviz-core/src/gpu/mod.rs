pub mod dipy;
pub mod plan_prep;

/// A wgpu device + queue pair handed to jobs that can use the GPU.
///
/// The job scheduler and `WorkflowJobPayload` are GPU-agnostic; whoever
/// runs a job (the GUI's worker threads, the headless driver) decides
/// whether a device is available and passes it to `run_workflow_job`.
/// `wgpu::Device` and `wgpu::Queue` are reference-counted handles, so
/// cloning this is cheap.
#[derive(Clone)]
pub struct GpuHandle {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

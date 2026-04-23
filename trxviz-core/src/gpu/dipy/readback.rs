//! GPU buffer-map readback helper with a bounded timeout.
//!
//! Replaces the older `map_async(..., |_| {})` + `poll(wait_indefinitely)`
//! pattern (see PR 1 bug #5), which silently ate `BufferAsyncError` and
//! hung forever if the driver never fired the map callback. The new
//! helper routes the callback through an mpsc channel, polls the device
//! between short recv-timeouts, and returns a `WorkflowError` on a global
//! deadline so a wedged driver can't keep the worker thread stuck.

use crate::error::{WorkflowError, WorkflowResult};

/// Timeout for a single batch's GPU readback. 30 s is ~100× the worst
/// real batch we've observed; crossing it almost certainly means the
/// driver wedged and we'd rather surface an error than hang.
pub(super) const GPU_READBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Map the given buffer slices for reading, blocking until all maps
/// complete, the map errors, or the global `timeout` fires.
pub(super) fn map_slices_blocking(
    device: &wgpu::Device,
    slices: &[wgpu::BufferSlice<'_>],
    timeout: std::time::Duration,
) -> WorkflowResult<()> {
    use std::sync::mpsc::{RecvTimeoutError, channel};

    let (tx, rx) = channel::<Result<(), wgpu::BufferAsyncError>>();
    for slice in slices {
        let tx = tx.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    }
    drop(tx);

    let n = slices.len();
    let start = std::time::Instant::now();
    let mut got = 0usize;
    while got < n {
        // Some backends only fire callbacks when the device is polled.
        let _ = device.poll(wgpu::PollType::Poll);
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(Ok(())) => got += 1,
            Ok(Err(e)) => {
                return Err(WorkflowError::Evaluation(format!(
                    "GPU buffer map failed: {e}"
                )));
            }
            Err(RecvTimeoutError::Timeout) => {
                if start.elapsed() > timeout {
                    return Err(WorkflowError::Evaluation(format!(
                        "GPU buffer map exceeded {:.0}s timeout ({}/{} buffers mapped); \
                         driver or shader may be wedged",
                        timeout.as_secs_f32(),
                        got,
                        n,
                    )));
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(WorkflowError::Evaluation(format!(
                    "GPU buffer map channel closed after only {}/{} buffers mapped",
                    got, n,
                )));
            }
        }
    }
    Ok(())
}

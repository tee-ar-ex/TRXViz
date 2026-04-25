//! Helpers that build `TrackingPlan`s from ODX data + a reference bundle.
//!
//! These run on the CPU (plan construction is cheap relative to GPU tracking)
//! and are invoked by the `Prepare*Plan` workflow ops.

pub mod hausdorff;
pub(crate) mod mask_dilate;
pub mod pyafq;

//! Shared workflow, scene, and rendering infrastructure for the TRXViz ecosystem.
//!
//! This crate is intended to be reused by both the desktop application and the headless CLI so
//! rendering behavior and workflow execution stay aligned across both surfaces.

pub mod data;
/// Structured error types for the workflow evaluator and persistence layer.
pub mod error;
/// Headless offscreen rendering entrypoints and PNG export helpers.
pub mod headless;
/// Scene lighting presets and parameters shared by render paths.
pub mod lighting;
/// Renderer-facing GPU resources, cameras, and draw-path infrastructure.
pub mod renderer;
/// Shared scene and asset state used by GUI and headless execution.
pub mod scene;
/// Workflow document types, evaluation, loading, and execution helpers.
pub mod workflow;

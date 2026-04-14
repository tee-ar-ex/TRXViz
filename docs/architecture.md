# Architecture

TRXViz is a Cargo workspace with three crates. This page explains what each
crate is responsible for, why the project is split this way, and how the
pieces fit together at runtime.

## The three crates

```
trxviz/                (workspace root)
├── trxviz-core/       library: data loading, workflow, rendering primitives
├── trxviz/            binary: eframe/egui desktop GUI
└── trxviz-cli/        binary: headless command-line renderer
```

### `trxviz-core` — the shared library

All of the non-UI logic lives here:

- **Data layer** (`src/data/`): readers for TRX, NIfTI, GIFTI, CIFTI, and
  parcellation files. Pure CPU-side representations with no GPU or windowing
  dependencies.
- **Renderer** (`src/renderer/`): wgpu pipelines for streamlines, meshes,
  slices, and glyphs. Camera math. Shader sources.
- **Workflow** (`src/workflow/`): the DAG data model (`WorkflowGraph`,
  `WorkflowNode`), the `WorkflowRuntime` that evaluates it with fingerprint
  caching, and the per-node evaluation logic.
- **Headless entry point** (`src/headless.rs`): project and loose-asset
  render/export helpers drive the full pipeline for anatomical 3D, saved 2D
  project views, and inflated-stage rendering/export without ever touching
  winit, eframe, or a display server.

### `trxviz` — the desktop GUI

A thin-ish shell on top of `trxviz-core`. It owns:

- The `eframe` application, egui event loop, and window management.
- The egui-tiles workspace layout (assets pane, preview, graph editor,
  inspector).
- The `egui-snarl` node-graph editor widget.
- Interactive concerns the CLI does not need: background job scheduling,
  cancellation, progress reporting, import/merge dialogs.

The GUI drives the same `WorkflowRuntime` that `trxviz-core` exposes — it does
not reimplement node evaluation.

### `trxviz-cli` — the headless renderer

A small `clap`-based binary (≈160 lines) that parses arguments, constructs a
`HeadlessRenderOptions`, and calls into `trxviz_core::headless`. Zero egui
dependencies. This crate exists so that TRXViz visualizations can be produced
as part of batch data-processing pipelines on machines with no display.

## Why split it this way

The project started as a monolithic eframe app. Splitting became necessary
when the goal shifted to also support headless rendering on compute nodes.

Three observations drove the layout:

1. **Headless rendering must not depend on a display server.** Anything the
   CLI needs has to compile and run without winit, eframe, X11, or Wayland.
   This is a hard constraint, not a preference.
2. **The GUI and the CLI must render the same project identically.** A
   project file rendered via the CLI on a compute node should produce the
   same PNG as the GUI on a workstation. That means they must share *one*
   implementation of the data model, the workflow runtime, and the rendering
   pipelines.
3. **GUI concerns should not contaminate the library.** Job scheduling,
   cancellation, progress bars, and import dialogs are interactive concerns
   that do not belong in the shared library.

Splitting into `core` (shared library) + `trxviz` (GUI shell) + `cli` (headless
binary) is the smallest arrangement that satisfies all three.

## How a render happens

### In the GUI

1. The user opens a project file or edits nodes in the graph pane.
2. `TrxVizApp::update` runs each frame, calling `refresh_workflow_runtime`
   and `queue_workflow_jobs` to drive `WorkflowRuntime` forward.
3. Expensive nodes (tube geometry, bundle surfaces, boundary fields) run on
   background threads and send results back over `mpsc` channels.
4. Completed node outputs populate `CallbackResources`; the next frame's
   `egui-wgpu` paint callbacks draw the scene into the preview pane.

### In the CLI

1. `clap` parses the command-line arguments into a `HeadlessRenderOptions`.
2. `trxviz_core::headless::render_project_png` loads the project file,
   constructs a `WorkflowRuntime`, and evaluates the graph to completion
   (synchronously — no job scheduler, no cancellation).
3. A headless wgpu instance is created via `pollster::block_on`, the scene is
   drawn into an offscreen texture, and the texture is written to a PNG.

The code path from `WorkflowRuntime::run` to pixels is identical in both
cases; only the driver around it differs.

## The workflow graph data model

`WorkflowDocument` holds the canonical workflow state:

- `graph: WorkflowGraph` — a pure-serde DAG of `WorkflowNode` keyed by
  `WorkflowNodeUuid`. This type lives in `trxviz_core::workflow::graph` and
  has zero egui dependencies, so it can be loaded, evaluated, and
  round-tripped on a headless host.
- `workspace: egui_tiles::Tree<WorkspacePane>` — the pane layout. Still
  egui-coupled (the CLI does not need it).

In the GUI, the `egui-snarl` editor widget is built fresh each frame from the
canonical `WorkflowGraph` (`snarl_from_graph`) and synced back after user
edits (`sync_graph_from_snarl`). This keeps the on-disk format decoupled from
`egui-snarl`'s internal representation — if that crate changes its serde
shape, your saved projects still load.

## External dependencies worth knowing about

- **`trx-rs`** — TRX reader/writer plus imported tractogram support. Pulled from [GitHub](https://github.com/tee-ar-ex/trx-rs)
  as a git dependency. Not a member of the TRXViz workspace so that it can evolve on its own cadence.
- **`wgpu`** — GPU abstraction. Used by both the GUI and the headless
  renderer; all shaders live in `trxviz-core/src/shaders/`.
- **`egui` / `eframe` / `egui-snarl` / `egui_tiles`** — only used by the
  `trxviz` GUI crate and a few GUI-facing types in `trxviz-core` that
  will eventually move behind a feature flag.

## Conventions

- **Coordinates**: all world-space positions use the **RAS+ neuroimaging
  convention** (X=Right, Y=Anterior, Z=Superior).
- **GPU structs**: anything uploaded to a uniform or vertex buffer derives
  `bytemuck::Pod` + `Zeroable`.
- **Split position/color buffers**: streamline vertex data lives in two
  buffers so recoloring 100k+ streamlines only rewrites the 16-byte/vertex
  color buffer, not the 12-byte/vertex positions.

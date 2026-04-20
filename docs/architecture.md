# Architecture

TRXViz is a three-crate Cargo workspace with a shared core library, a desktop GUI,
and a headless CLI. 

## Workspace layout

```text
trxviz/                workspace root
├── trxviz-core/       shared library
├── trxviz/            eframe/egui desktop GUI
└── trxviz-cli/        clap-based headless renderer
```

## `trxviz-core`

`trxviz-core` owns the reusable, non-UI parts of the system:

- `asset_loader`:
  shared asset-kind detection and loader registry for TRX, imported tractograms,
  NIfTI, GIFTI, CIFTI, parcellations, and ODX inputs.
- `data/`:
  CPU-side format loaders and domain data structures.
- `workflow/`:
  the persisted `WorkflowDocument`, graph model, typed workflow ops, evaluator,
  fingerprinting, seeding, and project I/O.
- `renderer/`:
  wgpu pipelines, cameras, viewport-uniform helpers, shaders, and shared color
  mapping.
- `headless/`:
  offscreen render/export orchestration. This is now decomposed into focused
  modules such as `gpu_context`, `scene_loader`, `workflow_driver`,
  `render_data`, `render_2d`, `render_3d`, `readback`, `bake`, and
  `export_glb/`.
- `scene`:
  shared scene/asset state used by both GUI and headless entry points.

`trxviz-core` is intentionally free of egui/eframe dependencies. GUI-specific
state and widgets stay in the `trxviz` crate.

## `trxviz`

The desktop app is an egui shell over `trxviz-core`. It owns:

- window/event-loop integration via `eframe`
- workspace layout and panes
- the node-graph editor widget
- background file-load and workflow-job scheduling
- import/merge dialogs and other interactive-only UX

The GUI now splits file ingestion into two layers:

- `file_loading.rs`:
  background job kickoff, import helpers, merge helpers, and ODX recovery if missing an affine 
- `file_apply.rs`:
  scene mutation and workflow-asset registration once a loaded asset is ready

Persisted view state lives in `WorkflowDocument`. `ViewportState` is now
runtime/editor state only.

## `trxviz-cli`

The CLI is a thin wrapper around `trxviz_core::headless`. It parses arguments,
loads either a workflow project or loose assets, and renders/export scenes
without any GUI dependencies.

The CLI and GUI share the same asset classification and loading logic
through `trxviz_core::asset_loader`.

## Workflow model

`WorkflowDocument` is the canonical persisted model. It owns:

- the workflow graph
- persisted view state (`camera_3d`, `render_3d`, `slice_view_3d`,
  `slice_view_ui`)
- persisted selection
- registered workflow assets

Workflow execution is op-registry based:

- each node stores a `WorkflowNodeKind` serde surface owned by the workflow-op
  registry
- per-op behavior lives under `trxviz-core/src/workflow/ops/`
- `evaluate.rs` is a thin scheduler/dispatch facade

This keeps “add a new node” work localized to one op module plus one registry
entry.

## Render flow

### GUI

1. Load or edit a workflow document.
2. Evaluate the workflow graph and queue any expensive jobs.
3. Apply completed job results back into shared scene/runtime state.
4. Render through egui-wgpu callbacks using the same renderer types as the
   headless path.

### CLI

1. Parse arguments into a headless render request.
2. Load assets/project state through the shared loader path.
3. Evaluate the workflow to completion synchronously.
4. Build GPU resources, render to offscreen textures, and export PNG or GLB.

The evaluator and renderer stack is shared; only the driving shell differs.

## Notable conventions

- World-space coordinates use RAS+ neuroimaging conventions.
- Semantic newtypes such as `Millimeters`, `ParcelId`, and `StreamlineIndex`
  are used in the workflow/runtime layer instead of raw scalar identifiers.
- Heavy immutable dataset boundaries may still use `Arc`, but the workflow
  evaluator itself is intentionally single-threaded.

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

### Background jobs

Ops that need expensive work (derived streamlines, surface queries, tube
geometry, boundary fields, bundle meshes, plan preparation, tractography) do
not run it inline. During evaluation they push a `JobRequest` onto
`SceneFramePlan::jobs`, carrying the fingerprint they primed their run record
with and whether their inputs are stale. `workflow/scheduler.rs` is the single
place that decides what to dispatch (`plan_jobs`) and how a finished job lands
in the execution cache (`apply_job_output` / `apply_job_error`):

- `ReactiveStreamline` jobs are `Auto` tier and run on every settled pass; every
  other kind is `Expensive` and runs only when the driver's `SchedulePolicy`
  says so (the GUI's Run button, always in the CLI).
- A request whose inputs are stale is deferred until the upstream jobs land, so
  a tracker never runs against a placeholder seed mask.
- A node that failed at its current fingerprint stays `Failed` and is not
  re-dispatched until the user edits it or clicks Run again.

The GUI and the headless driver differ only in *how* they run the jobs the
scheduler hands them: worker threads plus an mpsc channel in the app, an inline
loop in `headless/workflow_driver.rs`.

## Render flow

### GUI

1. Load or edit a workflow document.
2. Evaluate the workflow graph; once settled, ask the shared scheduler which
   jobs to spawn on worker threads.
3. Apply completed job results back into the execution cache through the
   shared scheduler.
4. Render through egui-wgpu callbacks using the same renderer types as the
   headless path.

### CLI

1. Parse arguments into a headless render request.
2. Load assets/project state through the shared loader path.
3. Loop: evaluate in `Settled` mode, run every job the shared scheduler
   returns inline, until it returns none (or `--skip-expensive` limits it to
   the automatic tier).
4. Build GPU resources, render to offscreen textures, and export PNG or GLB.

The evaluator and renderer stack is shared; only the driving shell differs.

## Notable conventions

- World-space coordinates use RAS+ neuroimaging conventions.
- Semantic newtypes such as `Millimeters`, `ParcelId`, and `StreamlineIndex`
  are used in the workflow/runtime layer instead of raw scalar identifiers.
- Heavy immutable dataset boundaries may still use `Arc`, but the workflow
  evaluator itself is intentionally single-threaded.

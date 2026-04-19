# trxviz-core

Shared workflow, scene, and rendering primitives for the TRXViz toolchain.

## Purpose

`trxviz-core` is a **non-UI** Rust library used by:

- `trxviz` — the desktop GUI (adds egui/eframe on top)
- `trxviz-cli` — the headless render entrypoint

The crate deliberately carries **no** egui/eframe/egui-snarl/egui_tiles
dependencies. GUI code belongs in the `trxviz` crate. Running
`cargo tree -p trxviz-core | grep -i egui` should return nothing.

## Cargo features

| Feature | Default | Description |
|---|---|---|
| `png-export` | ✓ | PNG readback via the `image` crate (`render_project_png`, `render_assets_png`) |
| `glb-export` | ✓ | GLB/glTF scene export (`export_project_glb`, `export_assets_glb`) |

To build a minimal core without any image output (e.g. for a library consumer
that only needs the workflow engine):

```toml
trxviz-core = { ..., default-features = false }
```

## Main Modules

- `headless` — offscreen scene rendering and export (PNG behind `png-export`,
  GLB behind `glb-export`). This module will be decomposed into sub-modules in a
  future refactor (Stage 6 of the refactor plan).
- `workflow` — project loading, graph evaluation, and execution helpers
- `scene` — shared scene and asset state
- `renderer` — GPU resource and draw-path infrastructure
- `data` — CPU-side file and asset loading for TRX, NIfTI, GIFTI, CIFTI, ODX,
  and parcellation files

## API Docs

Generate local API docs with:

```bash
cargo doc --no-deps -p trxviz-core
```

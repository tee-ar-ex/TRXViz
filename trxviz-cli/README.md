# trxviz-cli

Minimal command-line rendering for TRXViz workflows and scenes.

## Purpose

Use `trxviz-cli` when you need reproducible PNG output or Blender-oriented GLB export without
opening the desktop app.

## Render A Workflow Project

```bash
cargo run -- render --project workflow.json --out scene.png
```

Render the saved 2D viewer state from a project:

```bash
cargo run -- render --project workflow.json --view 2d --out scene.png
```

Render the saved inflated-stage layout from a project:

```bash
cargo run -- render --project workflow.json --view stage --out stage.png
```

## Render Loose Assets

```bash
cargo run -- render \
  --tractogram tractogram.trx \
  --nifti volume.nii.gz \
  --out scene.png
```

## Export A Blender Scene

```bash
cargo run -- export-scene --project workflow.json --out scene.glb
```

Export the inflated stage instead of the anatomical 3D scene:

```bash
cargo run -- export-scene --project workflow.json --view stage --out stage.glb
```

## Useful Options

- `--view 3d|2d|stage` for `render`
- `--view 3d|stage` for `export-scene`
- `--width` / `--height`
- `--target`
- `--azimuth`
- `--elevation`
- `--distance`

## Notes

- The CLI is intentionally small and command-focused.
- `--view 2d` is supported for `render`, not for `export-scene`.
- Stage rendering and stage GLB export are driven by saved workflow projects and reflect surfaces
  routed to `Stage`.
- For the full product documentation, use the main TRXViz docs portal in the `trxviz` repo.
- For the shared Rust APIs behind the CLI, see `trxviz-core`.

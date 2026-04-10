# TRXViz

A cross-platform desktop application for visualizing [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files with NIfTI-1 background volumes.

Built with Rust, [egui](https://github.com/emilk/egui), and [wgpu](https://wgpu.rs/) for GPU-accelerated rendering.

Full documentation lives in the MkDocs portal at the workspace root ([`docs/`](../docs/)) and
is published to the TRXViz docs site.

<p align="center">
  <img src="assets/logo-512x512.png" alt="TRXViz logo" width="256">
</p>

## Features

- **4-viewport layout** — 3D perspective view with axial, coronal, and sagittal slice views
- **GPU-accelerated rendering** via wgpu (Metal on macOS, Vulkan/DX12 on Linux/Windows)
- **NIfTI-1 volume slices** correctly aligned in RAS+ coordinates using the NIfTI affine
- **Depth-tested 3D rendering** for proper occlusion of streamlines and slices
- **Multiple coloring modes** — direction RGB, per-vertex (DPV) scalar, per-streamline (DPS) scalar, group color, or uniform
- **Group visibility controls** — toggle individual streamline groups on/off
- **Interactive cameras** — orbit/zoom in 3D, pan/zoom/scroll through slices in 2D
- **Crosshair overlays** on slice views showing the position of the other two slice planes
- **Intensity windowing** — adjustable center/width for volume display
- **Large dataset support** — separate position/color GPU buffers for efficient recoloring of 100k+ streamline datasets

## Usage

```bash
# Run the GUI from the workspace root
cargo run -p trxviz --release -- tractogram.trx background.nii.gz

# Or from the crate directory
cd trxviz
cargo run --release
```

### Headless CLI

```bash
# Render a saved workflow project offscreen
cargo run -p trxviz-cli -- render --project workflow.json --out scene.png

# Or render a scene from loose assets
cargo run -p trxviz-cli -- render --trx tractogram.trx --nifti background.nii.gz --out scene.png
```

## Building

Requires [Rust](https://rustup.rs/) 1.88+.

```bash
# Build the GUI app
cargo build -p trxviz --release

# The binary is at target/release/trxviz
```

For headless rendering:

```bash
# Build the CLI
cargo build -p trxviz-cli --release

# The binary is at target/release/trxviz-cli
```

### macOS app bundle

```bash
cargo build --release

# The .app bundle is at target/release/TRXViz.app
# To rebuild after code changes:
cp target/release/trxviz "target/release/TRXViz.app/Contents/MacOS/TRXViz"
touch "target/release/TRXViz.app"
```

## Controls

| Viewport | Action | Input |
|----------|--------|-------|
| 3D | Orbit | Left-click drag |
| 3D | Zoom | Scroll |
| Slice | Pan | Left-click drag |
| Slice | Zoom | Right-click drag |
| Slice | Change slice | Scroll |

## Dependencies

- [trx-rs](https://github.com/tee-ar-ex/trx-rs) — TRX file reading
- [eframe](https://github.com/emilk/egui/tree/master/crates/eframe) / [egui](https://github.com/emilk/egui) — GUI framework
- [wgpu](https://wgpu.rs/) — GPU rendering
- [nifti](https://crates.io/crates/nifti) — NIfTI-1 volume loading
- [rfd](https://crates.io/crates/rfd) — Native file dialogs

## Headless Notes

`trxviz-cli` does not require X11 or Wayland, but it still requires a usable `wgpu` backend and graphics driver on the host. If no headless adapter is available, the command exits with a clear error.

## License

BSD 2-Clause

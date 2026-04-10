# TRXViz

A cross-platform desktop application for visualizing [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files with NIfTI-1 background volumes. Includes a headless CLI for producing renders as part of data-processing pipelines on compute nodes with no display.

Built with Rust, [egui](https://github.com/emilk/egui), and [wgpu](https://wgpu.rs/) for GPU-accelerated rendering.

<p align="center">
  <img src="trxviz/assets/logo-512x512.png" alt="TRXViz logo" width="256">
</p>

## Crates

| Crate | Description |
|---|---|
| [`trxviz`](trxviz/) | Desktop GUI built on eframe/egui |
| [`trxviz-core`](trxviz-core/) | Shared data loading, workflow engine, and wgpu rendering primitives |
| [`trxviz-cli`](trxviz-cli/) | Headless command-line renderer |

## Features

- **4-viewport layout** — 3D perspective view with axial, coronal, and sagittal slice views
- **GPU-accelerated rendering** via wgpu (Metal on macOS, Vulkan/DX12 on Linux/Windows)
- **Workflow editor** — node graph for building and reusing visualization pipelines
- **Headless rendering** — render workflow projects to PNG on display-less compute nodes
- **NIfTI-1 volume slices** correctly aligned in RAS+ coordinates using the NIfTI affine
- **Multiple coloring modes** — direction RGB, per-vertex (DPV) scalar, per-streamline (DPS) scalar, group color, or uniform
- **Group visibility controls** — toggle individual streamline groups on/off
- **Interactive cameras** — orbit/zoom in 3D, pan/zoom/scroll through slices in 2D
- **Intensity windowing** — adjustable center/width for volume display
- **Large dataset support** — separate position/color GPU buffers for efficient recoloring of 100k+ streamline datasets

## Quick start

```bash
# GUI — open a tractogram with an optional background volume
cargo run -p trxviz --release -- tractogram.trx background.nii.gz

# CLI — render a saved workflow project to PNG
cargo run -p trxviz-cli -- render --project workflow.json --out scene.png

# CLI — render from loose assets
cargo run -p trxviz-cli -- render --trx tractogram.trx --nifti background.nii.gz --out scene.png
```

## Building

Requires [Rust](https://rustup.rs/) 1.88+.

```bash
# Build everything
cargo build --release

# Binaries: target/release/trxviz  target/release/trxviz-cli
```

### macOS app bundle

```bash
cargo build -p trxviz --release
cp target/release/trxviz "target/release/TRXViz.app/Contents/MacOS/TRXViz"
touch "target/release/TRXViz.app"
```

### Linux system dependencies

```bash
sudo apt-get install -y \
  libwayland-dev libxkbcommon-dev libxkbcommon-x11-dev \
  libx11-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  libxext-dev libglib2.0-dev libgtk-3-dev
```

## Controls

| Viewport | Action | Input |
|---|---|---|
| 3D | Orbit | Left-click drag |
| 3D | Zoom | Scroll |
| Slice | Pan | Left-click drag |
| Slice | Zoom | Right-click drag |
| Slice | Change slice | Scroll |

## Headless rendering

`trxviz-cli` does not require X11 or Wayland, but does require a usable wgpu backend and graphics driver. On headless Linux servers, Mesa's software rasterizer (`llvmpipe`) works as a fallback if no GPU is available.

## Documentation

Full documentation is at the [TRXViz docs site](https://YOUR_GITHUB_USERNAME.github.io/TRXViz/) (built from [`docs/`](docs/)).

## License

BSD 2-Clause

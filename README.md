# TRXViz

A cross-platform desktop application for visualizing [TRX](https://github.com/tee-ar-ex/trx-spec)
brain tractography files with NIfTI-1 background volumes, GIFTI surfaces, and CIFTI scalar data.
Includes a headless CLI for producing renders as part of data-processing pipelines on compute
nodes with no display.

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
- **CIFTI support** — load `.dscalar.nii`, `.dtseries.nii`, `.dlabel.nii`, and `.pscalar.nii`
- **GPU-accelerated rendering** via wgpu (Metal on macOS, Vulkan/DX12 on Linux/Windows)
- **Workflow editor** — node graph for building and reusing visualization pipelines
- **Headless rendering** — render workflow projects to PNG on display-less compute nodes
- **Inflated Stage** — pop-out viewer for non-anatomical cortical surface layouts
- **NIfTI-1 volume slices** correctly aligned in RAS+ coordinates using the NIfTI affine
- **Multiple coloring modes** — direction RGB, per-vertex (DPV) scalar, per-streamline (DPS) scalar, group color, or uniform
- **Group visibility controls** — toggle individual streamline groups on/off
- **Interactive cameras** — orbit/zoom in 3D, pan/zoom/scroll through slices in 2D
- **Intensity windowing** — adjustable center/width for volume display
- **Large dataset support** — separate position/color GPU buffers for efficient recoloring of 100k+ streamline datasets

## Quick start

For most users, start with a prebuilt release from the
[TRXViz GitHub releases page](https://github.com/tee-ar-ex/TRXViz/releases) and only use the cargo
commands below if you want to build from source.

```bash
# GUI — open a tractogram with an optional background volume
cargo run -p trxviz --release -- tractogram.trx background.nii.gz

# CLI — render a saved workflow project to PNG
cargo run -p trxviz-cli -- render --project workflow.json --out scene.png

# CLI — render the inflated stage from a saved workflow project
cargo run -p trxviz-cli -- render --project workflow.json --view stage --out stage.png

# CLI — render from loose assets
cargo run -p trxviz-cli -- render --tractogram tractogram.trx --nifti background.nii.gz --out scene.png

# CLI — export the inflated stage as a Blender-oriented GLB
cargo run -p trxviz-cli -- export-scene --project workflow.json --view stage --out stage.glb
```

## CIFTI Notes

TRXViz supports CIFTI scalar files with suffixes `.dscalar.nii`, `.dtseries.nii`, `.dlabel.nii`,
and `.pscalar.nii`.

- Cortical CIFTI data is visualized on separately loaded GIFTI surfaces.
- Non-anatomical cortical layouts use the `Inflated Stage` pop-out viewer rather than the main 3D
  anatomical scene.
- CIFTI workflow editing happens in Advanced mode.

## Downloading Executables

Prebuilt GUI and CLI executables are published on the
[TRXViz GitHub releases page](https://github.com/tee-ar-ex/TRXViz/releases).

- Download the asset for your platform from the latest release.
- On macOS, move `TRXViz.app` into `/Applications`.
- On Windows, unzip the release archive to a normal local folder.
- The macOS CLI binary is bundled inside the app at
  `/Applications/TRXViz.app/Contents/MacOS/trxviz-cli`.
- On Windows, `trxviz.exe` and `trxviz-cli.exe` are included in the extracted release folder.

### Opening The Unsigned macOS App

Because the macOS app is unsigned, Gatekeeper may block the first launch.

1. Try opening `TRXViz.app` once from Finder.
2. Open `System Settings > Privacy & Security`.
3. In the `Security` section, click `Open Anyway` for TRXViz.
4. Confirm `Open` and authenticate if prompted.

You can also Control-click the app in Finder and choose `Open` to create the same exception.

### Opening The Unsigned Windows App

Windows may show a Microsoft Defender SmartScreen warning for unsigned releases.

1. Open `trxviz.exe`.
2. Click `More info`.
3. Click `Run anyway` if you trust the downloaded release.

Some Windows 11 systems with Smart App Control enabled may block unsigned apps more aggressively.
If that happens, use a machine that permits the app or build from source locally.

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

## Saved 3D Camera

Workflow project JSON can now store the current 3D camera under `document.camera_3d` and the 3D
slice state under `document.slice_view_3d`.

- Save a workflow project after framing the view you want to preserve.
- Use `View > Copy 3D Camera JSON` in the app, or the `Copy 3D Camera` button in the preview
  toolbar, to copy a JSON snippet for manual editing.
- Saved GUI projects also restore the slice-plane positions and 2D slice-view pan/zoom state.
- `trxviz-cli render --project workflow.json --view 2d` renders the saved 2D viewer state to a
  PNG using the project’s stored slice layout and cameras.
- Saved GUI projects also restore the slice-plane positions and 2D slice-view pan/zoom state.
- `trxviz-cli render --project workflow.json` uses the saved camera by default unless you pass
  `--target`, `--azimuth`, `--elevation`, or `--distance`.

## Headless rendering

`trxviz-cli` does not require X11 or Wayland, but does require a usable wgpu backend and graphics driver. On headless Linux servers, Mesa's software rasterizer (`llvmpipe`) works as a fallback if no GPU is available.

## Documentation

Full documentation is at the [TRXViz docs site](https://YOUR_GITHUB_USERNAME.github.io/TRXViz/) (built from [`docs/`](docs/)).

## License

BSD 2-Clause

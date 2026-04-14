# Command Line

## Purpose

`trxviz-cli` is the reproducible rendering interface for TRXViz.

Use it when you need:

- PNG output without opening the GUI
- Blender-oriented scene export as `GLB`
- automation in scripts or batch jobs
- stable renders from saved workflow project files, including inflated-stage layouts

## Core Command

```bash
trxviz-cli render --out scene.png [input options]
```

## Workflow Project Rendering

```bash
trxviz-cli render --project workflow.json --out scene.png
```

This is the preferred path for reproducible figure generation.

If the workflow project contains `document.camera_3d` and `document.slice_view_3d`,
`trxviz-cli` uses those saved 3D view settings by default. Command-line camera flags override the
saved project camera when provided.

To render the saved 2D viewer state instead of the 3D scene:

```bash
trxviz-cli render --project workflow.json --view 2d --out scene.png
```

Project-driven 2D rendering reads:

- `document.slice_view_3d` for the shared slice-plane positions and visibility
- top-level `slice_view_ui` for the saved 2D mode, layout, and per-slice pan/zoom state

If `slice_view_ui` is missing, `--view 2d` exits with an error telling the user to open and save
the project from the GUI first.

To render the saved inflated-stage layout instead of the anatomical 3D scene:

```bash
trxviz-cli render --project workflow.json --view stage --out stage.png
```

Stage rendering is project-driven and reflects the surfaces currently routed to `Stage`.

## Loose Asset Rendering

```bash
trxviz-cli render \
  --tractogram tractogram.trx \
  --nifti volume.nii.gz \
  --surface mesh.gii \
  --out scene.png
```

The `--tractogram` input follows the same format-detection rules as the GUI. Native `.trx` files
load directly, while `.trk`, `.trk.gz`, `.tck`, `.tck.gz`, `.vtk`, `.tt`, and `.tt.gz` are
imported according to their extension before rendering.

## Camera and Output Controls

- `--view`: choose `3d`, `2d`, or `stage` project rendering; defaults to `3d`
- `--width` / `--height`: output image size
- `--target`: override camera target
- `--azimuth`, `--elevation`, `--distance`: camera placement controls

## Blender Scene Export

```bash
trxviz-cli export-scene --project workflow.json --out scene.glb
```

This exports the visible 3D scene as a Blender-oriented `GLB`, including visible surfaces, bundle
meshes, streamline tubes, visible slice planes, the saved 3D camera, and an approximate lighting
rig.

To export the saved inflated-stage layout instead:

```bash
trxviz-cli export-scene --project workflow.json --view stage --out stage.glb
```

Stage export is project-driven and reflects the surfaces currently routed to `Stage`. `--view 2d`
is render-only and is not supported by `export-scene`.

The export writes only the `GLB`. For Blender-side presentation styling, use one of the scripts in
the separate `trxviz-blender-styles` repo after import.

Useful toggles:

- `--include-camera=true|false`
- `--include-lights=true|false`
- `--include-slices=true|false`

The exporter preserves the visible workflow result (mostly). 

## Installation on macOS

The CLI binary is bundled inside the TRXViz app at:

```
/Applications/TRXViz.app/Contents/MacOS/trxviz-cli
```

To call it from anywhere, add a symlink:

```bash
ln -s /Applications/TRXViz.app/Contents/MacOS/trxviz-cli /usr/local/bin/trxviz-cli
```

On Linux and Windows the CLI is included alongside the GUI binary in the release archive.

## Headless Caveat

The CLI does not require X11 or Wayland, but it still requires a usable `wgpu` backend and a
working graphics driver on the host. "Headless" means no display server, not CPU-only rendering.

## Containers and Compute Nodes

On GPU-less hosts (e.g. Apptainer/Singularity containers on CPU-only nodes), wgpu falls back to
OpenGL via Mesa's `llvmpipe` software rasterizer. Install the following in your container image:

```
# Debian/Ubuntu
apt-get install -y libgl1-mesa-dri libgles2-mesa libegl1-mesa
```

On nodes with a GPU, pass `--nv` (NVIDIA) or `--rocm` (AMD) to your Apptainer invocation to
expose the host Vulkan driver into the container — wgpu will pick it up automatically and
`llvmpipe` won't be needed.

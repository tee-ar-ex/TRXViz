# Command Line

## Purpose

`trxviz-cli` is the reproducible rendering interface for TRXViz.

Use it when you need:

- PNG output without opening the GUI
- automation in scripts or batch jobs
- stable renders from saved workflow project files

## Core Command

```bash
trxviz-cli render --out scene.png [input options]
```

## Workflow Project Rendering

```bash
trxviz-cli render --project workflow.json --out scene.png
```

This is the preferred path for reproducible figure generation.

## Loose Asset Rendering

```bash
trxviz-cli render \
  --trx tractogram.trx \
  --nifti volume.nii.gz \
  --surface mesh.gii \
  --out scene.png
```

## Camera and Output Controls

- `--width` / `--height`: output image size
- `--target`: override camera target
- `--azimuth`, `--elevation`, `--distance`: camera placement controls

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

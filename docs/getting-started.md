# Getting Started

## Recommended: Download Prebuilt Executables

Prebuilt GUI and CLI executables are published on the
[TRXViz GitHub releases page](https://github.com/tee-ar-ex/TRXViz/releases).

- Download the archive or app bundle for your platform from the latest release assets.
- On macOS, move `TRXViz.app` into `/Applications` if you want it installed system-wide.
- On Windows, unzip the release archive to a folder such as `C:\Program Files\TRXViz` or
  `%LOCALAPPDATA%\TRXViz`.
- The macOS release also includes the CLI inside the app bundle at
  `/Applications/TRXViz.app/Contents/MacOS/trxviz-cli`.
- On Windows, the GUI and CLI executables are included side-by-side in the extracted release
  folder.

## First Launch On macOS

macOS may block the first launch because the app is currently unsigned and not notarized.

1. Open `TRXViz.app` once from Finder. macOS will block it and show a warning.
2. Open `System Settings > Privacy & Security`.
3. Scroll to the `Security` section and click `Open Anyway` for TRXViz.
4. Confirm by clicking `Open` on the follow-up dialog and enter your password if macOS asks.

You can also Control-click the app in Finder, choose `Open`, then confirm `Open`. After the first
approval, macOS saves the exception and later launches work normally.

## First Launch On Windows

Windows may show a Microsoft Defender SmartScreen warning because the executables are unsigned.

1. Open `trxviz.exe`.
2. If Windows shows `Windows protected your PC`, click `More info`.
3. Click `Run anyway` if you trust the downloaded release.

On some Windows 11 systems, Smart App Control may block unsigned apps more aggressively than the
standard SmartScreen prompt. If that happens, you may need to use a machine where Smart App
Control allows the app, or build TRXViz from source locally.

## Run the Desktop App

- macOS: open `TRXViz.app`.
- Windows: run `trxviz.exe`.
- Linux: run the extracted `trxviz` binary, or build from source if no release asset is available
  for your environment.

If you start without opening a file directly, import data from the GUI after launch.

## Use the Bundled CLI

- macOS: `/Applications/TRXViz.app/Contents/MacOS/trxviz-cli`
- Windows: `trxviz-cli.exe` from the extracted release folder

Example:

```bash
trxviz-cli render --project workflow.json --out scene.png
```

## Build From Source

Only use this path if you want to develop TRXViz or need a platform-specific local build.

- Rust 1.88+
- A working GPU backend for `wgpu`

```bash
cargo run -p trxviz --release -- tractogram.trx background.nii.gz
```

## First Headless Render

Use the CLI when you need a reproducible PNG instead of an interactive window.

```bash
cargo run -p trxviz-cli -- render --project workflow.json --out scene.png
```

Or render a scene from loose assets:

```bash
cargo run -p trxviz-cli -- render \
  --tractogram tractogram.trx \
  --nifti background.nii.gz \
  --out scene.png
```

To render a saved inflated-stage layout instead of the anatomical 3D scene:

```bash
cargo run -p trxviz-cli -- render --project workflow.json --view stage --out stage.png
```

## Input Types

- TRX tractograms
- Imported tractogram formats including TRK, TCK/TCK.GZ, VTK, and TT.GZ
- NIfTI volumes
- GIFTI surfaces
- CIFTI scalar files: `.dscalar.nii`, `.dtseries.nii`, `.dlabel.nii`, `.pscalar.nii`
- Parcellation volumes
- Workflow project JSON

## First CIFTI Workflow

For a cortical CIFTI visualization:

1. Open the CIFTI file.
2. Open compatible GIFTI surface meshes for the left and right cortex.
3. Switch to Advanced mode.
4. Connect the CIFTI cortex outputs into a `SurfaceOverlayStack` on the target surfaces.
5. Route the surface display to `Stage` if you want a non-anatomical inflated layout.

CIFTI files can be loaded in the app generally, but Simple mode does not expose editable CIFTI
workflow branches.

## Where To Go Next

- [GUI Guide](gui-guide.md) for viewer behavior and export
- [Workflows](workflows.md) for graph-based usage
- [Command Line](cli.md) for render automation, saved-camera behavior, and headless caveats

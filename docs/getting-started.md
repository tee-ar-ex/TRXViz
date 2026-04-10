# Getting Started

## Build Requirements

- Rust 1.88+
- A working GPU backend for `wgpu`

## Run the Desktop App

```bash
cargo run -p trxviz --release -- tractogram.trx background.nii.gz
```

If you start without arguments, open data from the GUI after launch.

## First Headless Render

Use the CLI when you need a reproducible PNG instead of an interactive window.

```bash
cargo run -p trxviz-cli -- render --project workflow.json --out scene.png
```

Or render a scene from loose assets:

```bash
cargo run -p trxviz-cli -- render \
  --trx tractogram.trx \
  --nifti background.nii.gz \
  --out scene.png
```

## Input Types

- TRX and imported tractogram formats
- NIfTI volumes
- GIFTI surfaces
- Parcellation volumes
- Workflow project JSON

## Where To Go Next

- [GUI Guide](gui-guide.md) for viewer behavior and export
- [Workflows](workflows.md) for graph-based usage
- [Command Line](cli.md) for render automation and headless caveats

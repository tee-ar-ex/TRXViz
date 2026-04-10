# trxviz-cli

Minimal command-line rendering for TRXViz workflows and scenes.

## Purpose

Use `trxviz-cli` when you need reproducible PNG output without opening the desktop app.

## Render A Workflow Project

```bash
cargo run -- render --project workflow.json --out scene.png
```

## Render Loose Assets

```bash
cargo run -- render \
  --trx tractogram.trx \
  --nifti volume.nii.gz \
  --out scene.png
```

## Useful Options

- `--width` / `--height`
- `--target`
- `--azimuth`
- `--elevation`
- `--distance`

## Notes

- The CLI is intentionally small and command-focused.
- For the full product documentation, use the main TRXViz docs portal in the `trxviz` repo.
- For the shared Rust APIs behind the CLI, see `trxviz-core`.

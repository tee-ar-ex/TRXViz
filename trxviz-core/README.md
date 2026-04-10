# trxviz-core

Shared workflow, scene, and rendering primitives for the TRXViz toolchain.

## Purpose

`trxviz-core` is the reusable Rust library behind:

- `trxviz`: the desktop GUI
- `trxviz-cli`: the headless render entrypoint

It owns the non-UI logic that should stay consistent across both surfaces.

## Main Modules

- `headless`: offscreen scene rendering to PNG
- `workflow`: project loading, graph evaluation, and execution helpers
- `scene`: shared scene and asset state
- `renderer`: GPU resource and draw-path infrastructure
- `data`: CPU-side file and asset loading

## API Docs

Generate local API docs with:

```bash
cargo doc --no-deps
```

In the long term, this crate is intended to be the primary developer-facing API surface of the
TRXViz ecosystem.

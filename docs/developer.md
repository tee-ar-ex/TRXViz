# Developer Notes

## Crate Boundaries

- `trxviz`: GUI shell and product documentation portal
- `trxviz-cli`: headless rendering entrypoint
- `trxviz-core`: shared workflow, scene, and rendering primitives

## API Surface

The `trxviz-core` crate is the long-term home for reusable rendering and workflow execution logic.
The app and CLI should stay thin over that shared core instead of growing parallel implementations.

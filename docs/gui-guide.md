# GUI Guide

## Interface Model

TRXViz is designed around two usage modes:

- **Simple**: adjust display, color, visibility, and a constrained set of workflow-affecting controls
- **Advanced**: edit the full workflow graph directly

The desktop app combines:

- a 3D perspective viewer
- axial, coronal, and sagittal slice views
- asset and inspector panes
- workflow graph editing when advanced mode is enabled

## What The GUI Is Best For

- exploratory viewing
- debugging workflow results interactively
- tuning display settings before generating figures
- inspecting slice alignment and bundle/surface overlays

## Export Behavior

TRXViz can export both 3D and 2D views to PNG.

- 3D exports capture the 3D viewer state
- 2D exports capture the selected 2D presentation mode
- scale multiplies the current viewport resolution

For batch figure generation, prefer the CLI so output can be scripted and reproduced.

If you save the project first, `trxviz-cli render --project workflow.json --view 2d` reproduces
the saved 2D viewer state headlessly.

## Saving And Reusing The 3D Camera

Workflow project saves now include the current 3D camera under `document.camera_3d` and the 3D
slice state under `document.slice_view_3d`.

- Save the workflow project after framing the view you want.
- Reopen the project later to restore the same 3D target, azimuth, elevation, distance, and
  axial/coronal/sagittal slice positions and visibility.
- GUI project saves also restore the 2D slice-view mode and per-slice pan/zoom state.
- Use `View > Copy 3D Camera JSON` or the `Copy 3D Camera` button in the preview toolbar to copy
  a JSON snippet you can paste under the `document` object in a workflow project manually.

Example snippet:

```json
{
  "camera_3d": {
    "target": [12.3, -18.0, 42.1],
    "azimuth_deg": 45.0,
    "elevation_deg": 25.0,
    "distance": 180.0
  },
  "slice_view_3d": {
    "visible": [true, false, true],
    "positions_ras": [10.0, 20.0, 30.0]
  }
}
```

# GUI Guide

## Modes

TRXViz is designed around two usage modes:

- **Simple**: adjust display, color, visibility, and a constrained set of workflow-affecting controls
- **Advanced**: edit the full workflow graph directly

The desktop app combines:

- a 3D perspective viewer
- an optional `Inflated Stage` pop-out viewer for non-anatomical cortical layouts (like inflated or Very Inflated surfaces)
- axial, coronal, and sagittal slice views
- asset and inspector panes
- workflow graph editing when advanced mode is enabled

## What The GUI Is Best For

- exploratory viewing
- debugging workflow results interactively
- tuning display settings before generating figures
- inspecting slice alignment, bundle/surface overlays, and CIFTI-driven surface appearance

## Export Behavior

TRXViz can export anatomical 3D, inflated-stage, and 2D views to PNG.

- 3D exports capture the 3D viewer state
- Inflated-stage exports capture the `Inflated Stage` pop-out state
- 2D exports capture the selected 2D presentation mode
- scale multiplies the current viewport resolution

For batch figure generation, prefer the CLI so output can be scripted and reproduced.

If you save the project first, `trxviz-cli render --project workflow.json --view 2d` reproduces
the saved 2D viewer state headlessly.

If you save the project first, `trxviz-cli render --project workflow.json --view stage` reproduces
the saved inflated-stage layout headlessly.

## Inflated Stage

`Inflated Stage` is a pop-out-only viewer for non-anatomical cortical surface presentations.

- Open it from the preview toolbar using `Open Inflated Stage`.
- It renders surfaces whose `SurfaceDisplay.space` is set to `Stage`.
- It does not use anatomical axis markers, because the displayed layouts are not in anatomical
  world space.
- Its `Export` menu supports `PNG` and `Blender (GLB)`.

This is the intended presentation surface for inflated or spherical cortical GIFTI meshes and for
CIFTI-driven cortical displays that should not appear in the main anatomical 3D scene.

## Surface Placement

`SurfaceDisplay.space` controls where a surface display appears:

- `Anatomical` renders in the main 3D viewer.
- `Stage` renders in `Inflated Stage`.

Surfaces whose filenames suggest non-anatomical layouts, such as bounded `inflated` or `sphere`,
may default to `Stage` when first added to a workflow branch.

## Surface Overlay Stack

`SurfaceOverlayStack` builds a styled surface appearance from one target surface plus ordered
scalar layers.

- Layer 0 is the base layer and also provides the fallback base color when no scalar input is connected.
- Each layer has its own colormap, numeric range, thresholding, opacity, visibility, and legend label.
- `dlabel` overlays can use attached label-table colors instead of numeric colormaps.
- CIFTI cortex outputs and streamline-derived surface scalars can both feed the same overlay stack.

### Layer controls

Each layer in the inspector exposes the following controls. Controls marked *(scalar input required)*
only have a visible effect when a scalar input is connected to that layer's input port.

**Enabled** — toggles the layer on or off. Disabled layers are skipped entirely; the layer beneath
shows through.

**Legend label** — text shown in the figure legend. Purely cosmetic.

**Fallback base color** *(base layer only)* — the RGBA color applied to every vertex before any
scalar overlays are blended. This is what you see when no scalar input is connected to layer 0, or
for any vertex not reached by an enabled overlay.

**Opacity** — *(scalar input required)* the alpha value blended over lower layers for each
colored vertex. 1.0 fully replaces the color below; 0.0 makes the layer invisible. Vertices that
are skipped by thresholding are unaffected and remain fully transparent to this layer.

**Use label table colors** *(scalar input required)* — for `dlabel` scalar inputs, uses the
integer label index to look up RGBA colors from the attached label table instead of the colormap.
When this is on, the Colormap, Min, Max, and threshold controls are all ignored.

**Colormap** *(scalar input required, disabled when "Use label table colors" is on)* — the color
ramp mapped across the scalar range. Values at Min receive the first ramp color; values at Max
receive the last.

**Min / Max** *(scalar input required)* — the scalar range that spans the full colormap. Values
outside this range are clamped to the nearest endpoint rather than hidden. When the scalar
metadata includes a suggested range, that range is used and these fields act as a fallback only.

**Thresh min / Thresh max** *(scalar input required)* — a hard visibility gate. Vertices whose
scalar value falls below Thresh min or above Thresh max are skipped (not colored), letting the
layer beneath show through. Defaults to −∞ / +∞ (no thresholding). Useful for masking
low-density regions or clipping outliers.

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

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

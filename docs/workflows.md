# Workflows

## Workflow Model

TRXViz workflows describe how assets are selected, transformed, colored, displayed, and exported.

The graph can combine:

- source assets
- streamline selection/filter nodes
- display/color nodes
- CIFTI structure extraction and surface overlay nodes
- surface projection and query nodes
- bundle surface and boundary field generation
- save/export nodes

## When To Use Workflows

- building figure-ready subsets
- comparing multiple display strategies
- running the same rendering logic repeatedly on shared project JSON
- moving from ad hoc GUI exploration to scripted CLI rendering

## Simple vs Advanced

- **Simple mode** is a constrained asset-first editor built on top of the workflow document
- **Advanced mode** is the full workspace and graph editor

These are intentionally different surfaces, not two symmetric wrappers over the same UI.

If a loaded project exceeds the Simple-mode constraint surface, switch to Advanced mode and treat
the graph as the source of truth.

CIFTI workflow editing is an Advanced-mode feature. More generally, any project with branching or
node wiring that no longer maps cleanly onto the simple bindings should be edited in Advanced mode.

## Project Files

Workflow projects are stored as JSON and can be rendered headlessly through `trxviz-cli`.

That makes project files the best handoff format between:

- interactive setup in the app
- automated rendering on another machine
- reproducible figure regeneration later

## CIFTI Workflow Shape

When you add a CIFTI asset, TRXViz seeds a structure-aware branch:

`CIFTI Source` -> `CIFTI Left Cortex`  
`CIFTI Source` -> `CIFTI Right Cortex`  
`CIFTI Source` -> `CIFTI Subcortex`

The cortical outputs are intended to connect into `Surface Overlay Stack` nodes on loaded GIFTI
surfaces. That keeps the scalar data separate from the target geometry and lets the same surface
receive CIFTI overlays, streamline-derived overlays, or both.

For non-anatomical cortical viewing, set the corresponding `SurfaceDisplay.space` to `Stage`.
Surface assets whose filenames contain bounded `inflated` or `sphere` may default to `Stage`
automatically when their initial workflow branch is created.

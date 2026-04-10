# Workflows

## Workflow Model

TRXViz workflows describe how assets are selected, transformed, colored, displayed, and exported.

The graph can combine:

- source assets
- streamline selection/filter nodes
- display/color nodes
- surface projection and query nodes
- bundle surface and boundary field generation
- save/export nodes

## When To Use Workflows

- building figure-ready subsets
- comparing multiple display strategies
- running the same rendering logic repeatedly on shared project JSON
- moving from ad hoc GUI exploration to scripted CLI rendering

## Simple vs Advanced

- **Simple mode** keeps common workflows editable through constrained controls
- **Advanced mode** exposes the full graph for arbitrary editing

If a loaded project exceeds the simple surface, switch to advanced mode and treat the graph as the
source of truth.

## Project Files

Workflow projects are stored as JSON and can be rendered headlessly through `trxviz-cli`.

That makes project files the best handoff format between:

- interactive setup in the app
- automated rendering on another machine
- reproducible figure regeneration later

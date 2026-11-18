# MPAT Tractography

[MPAT](https://github.com/rokemlab/MPAT) (Markov-chain absorbing tractography) computes a
bundle as an expectation over every streamline that satisfies the bundle's ROIs. It solves an
absorbing Markov chain instead of tracking streamlines one at a time. TRXViz can run MPAT from a
workflow and show the bundle's density building up while the solver converges.

Two workflow nodes do this:

- **MPAT Transition Probabilities** builds the chain, or loads it from an ODX file.
- **MPAT Tractography** solves one bundle on that chain, using the start, end, include and
  exclude regions of a tracking plan.

Both nodes run MPAT in a separate Python process. TRXViz doesn't contain MPAT's code; it calls the
MPAT installed in a Python environment you choose.

## What you need

- TRXViz ([Getting Started](getting-started.md)).
- A Python environment with MPAT installed, following
  [MPAT's installation page](https://github.com/rokemlab/MPAT/blob/main/docs/source/install.md).
- One participant's preprocessed DWI (with its `.bval` and `.bvec`) and a T1w image registered to
  it, for example from QSIPrep.

QSIRecon isn't needed: the `mpat` command writes everything the TRXViz nodes read.

## 1. Make the inputs with MPAT

Run `mpat` on the participant with `--save-transitions odx`, and list the bundles you want to look
at:

```bash
mpat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ \
    --n-proc 8 --max-mem 16GB --bundles "Left Arcuate" --save-transitions odx
```

This runs pyAFQ and MPAT and writes to `out/mpat/`. Two kinds of file in there are the inputs for
TRXViz:

| File | Holds |
|---|---|
| `<stem>_desc-transitions_dwimap.odx` | The Markov chain and the fODF it was built from |
| `<stem>_bundle-LeftArcuate_desc-start_mask.nii.gz` (and `end`, `include`, `exclude`) | The bundle's ROIs, warped to the participant and on the DWI grid |

`<stem>` is the participant's file name stem. It is the part of the transitions file's name before
`_desc-transitions_dwimap.odx`.

Some details:

- `--bundles` takes pyAFQ's bundle names, spaces included. Leave it out to do every bundle.
- ROI masks are written only for the bundles that are run.
- The run also solves those bundles, so `out/mpat/` holds MPAT's own result to compare against.
- The transitions ODX keeps the settings it was built with, and the TRXViz node's settings must
  match them:
  - **Standard preset:** matches a default `mpat` run.
  - **PTT preset:** matches a run with `--ptt`.
  - **Other flags:** if you changed `--angle`, `--step`, `--fod-thresh` or another setting, set
    the same values in the node. The sidecar `<stem>_desc-transitions_dwimap.json` lists the
    settings under `config`.

## 2. Fill in the template workflow

Download the template, [`mpat-live-bundle.json`](assets/workflows/mpat-live-bundle.json). Its
graph is drawn on the [MPAT live bundle](examples/mpat-live-bundle.md) example page. Its file
paths are placeholders:

| Placeholder | Replace with |
|---|---|
| `{stem}` | The participant's stem from step 1 |
| `{bundle}` | The bundle's name as it appears in file names, e.g. `LeftArcuate` |
| `{t1w}` | The full path of a T1w image to show behind the bundle |

Fill them in with `sed` and save the result in the MPAT output folder:

```bash
sed -e 's|{stem}|sub-01_ses-01_space-ACPC|g' \
    -e 's|{bundle}|LeftArcuate|g' \
    -e 's|{t1w}|/data/qsiprep/sub-01/anat/sub-01_space-ACPC_desc-preproc_T1w.nii.gz|g' \
    mpat-live-bundle.json > out/mpat/sub-01_LeftArcuate_workflow.json
```

The ODX and mask paths in the template are relative. TRXViz looks for them in the folder that
holds the workflow file, so the workflow file has to be saved next to MPAT's outputs.

## 3. Open the workflow with MPAT's Python

Start TRXViz with the workflow and the Python environment that has MPAT:

```bash
trxviz --python-env /path/to/envs/mpat/bin/python out/mpat/sub-01_LeftArcuate_workflow.json
```

On macOS the executable is `/Applications/TRXViz.app/Contents/MacOS/TRXViz`.

The MPAT nodes use the first of these Python interpreters that is set:

1. the node's own **Python environment** field;
2. `--python-env` (or its alias `--mpat-python`);
3. the `MPAT_PYTHON` environment variable;
4. `python3` on your `PATH`.

Each MPAT node shows the interpreter it will use under **Active Python**.

## 4. Build the transitions, then solve the bundle

1. Select **MPAT Transition Probabilities** and check that its preset matches the `mpat` run.
   Then click **Build MPAT Transitions**.
   - From a transitions ODX this takes seconds, because the chain is loaded, not rebuilt.
   - The node's outputs then hold eight maps of the transition probabilities, such as CoDI and
     the termination probability.
2. Select **MPAT Tractography** and click **Run MPAT Tractography**. While the solver runs:
   - The bundle's density glows in the 3-D view and grows as the solver converges. The
     **Live density glow** node draws the node's **Visitation Density** output with direct
     volume rendering.
   - The job's spinner shows the solver's status line and a progress bar. **Cancel** stops the
     run.
3. When the solve finishes, the node's outputs hold the solved maps and the live frame is
   dropped. A run that is cancelled or fails leaves no half-converged frame behind.

Both results are cached in the `trxviz_mpat_cache` folder of your system's temporary directory.
A node runs again only when its inputs or settings change.

## Adjusting the view

The template starts with only the glow and faint ROIs showing. These display nodes control the
rest:

| Node | What to change |
|---|---|
| **Live density glow** | Window center and width set the range of densities drawn: lower the center to show the bundle's fainter parts. The colormap is `Hot`. |
| **Start / End / Include mask display** | Drawn at 12% opacity. Raise the opacity to see the ROIs clearly, or set it to 0 to hide them. The exclude mask isn't drawn, because it covers most of the brain. |
| **Flux streamlines & arbor** | Hidden. Tick **Visible** to show the streamlines and branch arbor extracted after the solve. |
| **ODF glyphs** | Hidden. Tick **Visible** to draw the fODF stored in the ODX. |
| **T1w slices** | The background anatomy. |

## Other bundles

- **Another bundle:** run `sed` again with a different `{bundle}`. All the workflows can use the
  same transitions ODX, and the chain is cached, so later workflows load it straight away. To
  solve several bundles in one workflow, copy the plan and the MPAT Tractography node and wire
  each copy to the same **Transitions** output.
- **Midline exclusion:** on by default.
  - It keeps a one-sided bundle in the hemisphere of its ROIs, as MPAT does for bundles that
    don't cross the midline.
  - MPAT's exclude masks leave the other hemisphere out and expect this setting to add it, so
    keep it on for them.
  - Turn it off for callosal and other commissural bundles.
  - The tolerance sets how far past the midline a bundle may go. MPAT allows 10 mm for the
    corticospinal tracts, which meet near the midline in the brainstem.
- **26-connected ROI shell sealing:** off in the template, because MPAT's include and exclude
  masks are already sealed. Turn it on for ROIs you draw yourself, so diagonal steps can't slip
  through a thin ROI.

## Drawing your own ROIs

A tracking plan can take any voxel mask:

- **ROI from Shape:** a sphere or box at an RAS coordinate, on the ODX's grid.
- **ROI from Parcellation:** labels picked from a parcellation.
- **ROI from Volume Mask:** a thresholded NIfTI.

Add masks to the plan with these nodes:

- **Add ROI** adds an include region (a waypoint).
- **Add ROA** adds an exclude region.
- **Add End Region** adds an end region:
  - the first one added is the bundle's start;
  - the second is its end;
  - any further ones are merged into the end.

Masks on other grids are resampled onto the chain's grid (nearest neighbour).

## Starting from a plain fODF

Wiring an fODF ODX from another source (such as consh or QSIRecon) into **MPAT Transition
Probabilities** builds a new chain. The build takes minutes rather than seconds, and the
transition settings can then be changed freely. The chain also needs tissue maps, which come
from either of these:

- Set **Working dir** to the participant's pyAFQ or MPAT output folder. TRXViz searches it for
  pyAFQ's tissue PVE or MPAT's `desc-mpat_dseg`.
- Wire a white-matter mask into **Transient WM Mask**, and optionally a mask of voxels where
  streamlines stop into **Absorbing Mask**.

The chain is saved as `mpat_transitions_<hash>.npz` in the **Working dir**, or next to the ODX
when none is set. It is loaded from there the next time the same inputs and settings are used.

## Rendering without the GUI

`trxviz-cli render` runs the MPAT jobs to completion and then renders the scene:

```bash
MPAT_PYTHON=/path/to/envs/mpat/bin/python \
    trxviz-cli render --project out/mpat/sub-01_LeftArcuate_workflow.json --out arcuate.png
```

`trxviz-cli methods --project out/mpat/sub-01_LeftArcuate_workflow.json` writes a methods
paragraph describing the settings used.

!!! note "QSIRecon"
    QSIRecon's MPAT workflow is in review
    ([PennLINC/qsirecon#433](https://github.com/PennLINC/qsirecon/pull/433)). It writes the same
    ROI masks and, with `save_transitions: odx`, the transitions ODX, under BIDS derivative names.
    Until this page covers it, point the template's paths at those files by hand.

## Troubleshooting

| Message | Fix |
|---|---|
| `Pick a pyAFQ/MPAT directory or wire an ODF Field input.` | The transitions node has no ODX. Check the ODX path in the workflow file. |
| `<setting> is … in the file but … was requested` | The node's settings differ from those the transitions ODX was built with. Pick the matching preset, or the values from the ODX's JSON sidecar. |
| `a transient WM mask can't be applied to prebuilt transitions` | A transitions ODX keeps its own tissue maps. Disconnect the mask inputs, or start from a plain fODF. |
| `No tissue maps were found for this fODF` | A plain fODF needs tissue maps: set **Working dir** or wire a **Transient WM Mask**. |
| `Failed to launch Python interpreter …` | The interpreter doesn't exist. Set it in the node, with `--python-env`, or with `MPAT_PYTHON`. |
| `No module named 'mpat'` | The interpreter runs, but MPAT isn't installed in its environment. |
| The bundle spills into the other hemisphere | Midline exclusion is off, but the bundle uses MPAT's exclude mask. Turn it back on. |

MPAT's own documentation covers the method and every `mpat` option, in the
[`docs/source`](https://github.com/rokemlab/MPAT/tree/main/docs/source) folder of its
repository.

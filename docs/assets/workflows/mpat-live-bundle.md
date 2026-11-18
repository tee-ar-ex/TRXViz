## What this workflow does

It solves one bundle with MPAT from the files a `mpat --save-transitions odx` run writes, and
shows the bundle's density building up in 3-D while the solver converges. The MPAT guide linked
above explains how to make those files and how to fill in the template's `{stem}`, `{bundle}`
and `{t1w}` placeholders.

A few things to look at in the diagram:

- **The four ROI masks become one tracking plan.**
  - Each mask is read as a NIfTI and thresholded into a voxel mask.
  - **Add End Region** is chained twice: the first region added is the bundle's start, the
    second its end.
  - **Add ROI** adds the include mask and **Add ROA** the exclude mask.
- **The chain and the plan meet at MPAT Tractography.**
  - The transitions ODX feeds MPAT Transition Probabilities, which loads the chain MPAT built.
  - The finished plan comes in on the tractography node's **Tracking Plan** port.
- **The glow is the Visitation Density output.**
  - A Volume display draws it with direct volume rendering and the `Hot` colormap.
  - While the solver runs, that output shows the latest density frame.
- **Display nodes start quiet.** The flux streamlines and ODF glyphs are hidden, and the ROIs
  are drawn at 12% opacity, so the glow stands out. Turn them up in each node's inspector.

## What this workflow does

This is the kind of plan you'd run to reproduce a known bundle in a new
subject: prepare a Hausdorff-shape plan from a reference tractogram,
then seed deterministic tracking from the resulting masks.

A few things to look at in the diagram:

- **Two parallel branches feed the plan.** The reference bundle and the
  reconstruction's primary peaks are independent inputs — the plan node
  combines them to produce the seed mask, the limiting region, and the
  no-end mask in one pass.
- **The tracking node consumes all three masks at once.** The fixel
  tractography op takes the full plan as a single input port, not three
  separate ones, so the diagram stays readable as the plan grows.
# Troubleshooting

## No Headless GPU Backend Available

The CLI render path still needs a usable `wgpu` adapter. Headless means “no display server
required,” not “software-only rendering.” See [Containers and Compute Nodes](cli.md#containers-and-compute-nodes)
for how to set up Mesa `llvmpipe` as a fallback in Apptainer/Singularity images.

## Project Opens In Advanced Mode

That usually means the workflow exceeds the simple shell’s editable subset. Treat the graph as the
source of truth and work in advanced mode.

## Rendered Output Does Not Match The GUI

Check:

- whether the project JSON was saved after your GUI edits
- camera overrides used in the CLI
- any workflow nodes that produce derived geometry or boundary fields

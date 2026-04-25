//! Curated workflow examples built programmatically. Each example is a
//! `WorkflowDocument` constructed in code (no `.trxviz` files needed on
//! disk yet) plus a short intro paragraph. The docgen pipeline turns
//! each into a pair of files under `docs/examples/`:
//!
//! - `<slug>.svg` — the layered SVG diagram.
//! - `<slug>.md` — an intro + the embedded SVG + a preview of the
//!   methods-boilerplate that `generate_methods_report` would emit for
//!   this workflow.
//!
//! Examples are intentionally tiny — their job is to show the rendering
//! pipeline end-to-end. Real user workflows belong in saved `.trxviz`
//! files once we wire project loading.

use std::path::{Path, PathBuf};

use trxviz_core::workflow::{
    GraphPos, InPort, OutPort, WorkflowDocument, WorkflowNode, WorkflowNodeKind, WorkflowNodeUuid,
    default_document, load_workflow_project_from_path,
};

pub struct Example {
    pub slug: &'static str,
    pub title: &'static str,
    pub intro: &'static str,
    pub document: WorkflowDocument,
    /// Optional longer-form prose rendered between the SVG diagram and
    /// the methods-boilerplate section. For saved examples this is
    /// loaded from a sibling `<name>.md` next to the workflow JSON
    /// (e.g. `docs/assets/workflows/autotrack.md`); programmatic
    /// examples leave it `None` unless they have something to add.
    pub commentary: Option<String>,
}

pub fn all_examples(repo_root: &Path) -> anyhow::Result<Vec<Example>> {
    let mut examples = vec![save_streamlines_example(), purifibre_pipeline_example()];
    examples.extend(load_saved_examples(repo_root)?);
    Ok(examples)
}

/// Loaded from `docs/assets/workflows/*.json` — real saved projects
/// (or bare documents) are a better advertisement than hand-built
/// graphs, and they exercise the loader's tolerant parsing.
fn load_saved_examples(repo_root: &Path) -> anyhow::Result<Vec<Example>> {
    let entries: &[(&str, &str, &str, &str)] = &[(
        "autotrack",
        "docs/assets/workflows/autotrack.json",
        "Autotrack",
        "A saved autotrack workflow. Loaded from the checked-in JSON via \
         the tolerant project loader and laid out by the same engine that \
         renders the programmatic examples above.",
    )];
    let mut out = Vec::new();
    for (slug, rel_path, title, intro) in entries {
        let path: PathBuf = repo_root.join(rel_path);
        if !path.exists() {
            continue;
        }
        let project = load_workflow_project_from_path(&path)
            .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", path.display()))?;
        let commentary = sibling_commentary(&path)?;
        out.push(Example {
            slug,
            title,
            intro,
            document: project.document,
            commentary,
        });
    }
    Ok(out)
}

/// Read `<workflow>.md` next to the given workflow JSON, if it exists.
/// Lets contributors add explanatory prose alongside a saved example
/// without touching Rust code.
fn sibling_commentary(workflow_path: &Path) -> anyhow::Result<Option<String>> {
    let md_path = workflow_path.with_extension("md");
    if !md_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&md_path)
        .map_err(|e| anyhow::anyhow!("failed to read commentary {}: {e}", md_path.display()))?;
    let trimmed = text.trim();
    Ok(if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    })
}

fn save_streamlines_example() -> Example {
    // Streamline Source → Save Streamlines. Smallest end-to-end graph
    // the docs can show: a source producing a port, a terminal sink
    // consuming it.
    let mut doc = default_document();
    let src = WorkflowNodeUuid(1);
    let sink = WorkflowNodeUuid(2);
    doc.graph.insert_node(
        WorkflowNode {
            uuid: src,
            op: WorkflowNodeKind::StreamlineSource { source_id: 0 },
            label: "source".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: sink,
            op: WorkflowNodeKind::SaveStreamlines {
                output_path: String::new(),
            },
            label: "save".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.connect(
        OutPort {
            node: src,
            output: 0,
        },
        InPort {
            node: sink,
            input: 0,
        },
    );
    doc.next_node_uuid = 3;

    Example {
        slug: "save-streamlines",
        title: "Save streamlines",
        intro: "The smallest useful workflow: read streamlines from a source \
                and write them back to disk. Useful as a sanity check that the \
                source loader, the streamline port type, and the file writer all \
                line up.",
        document: doc,
        commentary: None,
    }
}

fn purifibre_pipeline_example() -> Example {
    // Streamline Source → Purifibre → Save. Purifibre needs a
    // BoundaryField on its second input, which has no source op yet
    // in TRXViz — so the second input stays unconnected. The diagram
    // still demonstrates the layering and the multi-output node layout.
    let mut doc = default_document();
    let src = WorkflowNodeUuid(1);
    let puri = WorkflowNodeUuid(2);
    let sink = WorkflowNodeUuid(3);
    doc.graph.insert_node(
        WorkflowNode {
            uuid: src,
            op: WorkflowNodeKind::StreamlineSource { source_id: 0 },
            label: "source".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: puri,
            op: WorkflowNodeKind::Purifibre {
                trim_fraction: 0.1,
                puri_fraction: 0.1,
                spherical_smoothing_deg: 15.0,
            },
            label: "purifibre".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: sink,
            op: WorkflowNodeKind::SaveStreamlines {
                output_path: String::new(),
            },
            label: "save".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.connect(
        OutPort {
            node: src,
            output: 0,
        },
        InPort {
            node: puri,
            input: 0,
        },
    );
    doc.graph.connect(
        OutPort {
            node: puri,
            output: 0,
        },
        InPort {
            node: sink,
            input: 0,
        },
    );
    doc.next_node_uuid = 4;

    Example {
        slug: "purifibre-pipeline",
        title: "Purifibre cleanup",
        intro: "Streamlines loaded from a source are cleaned with Purifibre \
                (trim + spherical smoothing) and then saved. The Purifibre op \
                has two outputs — the cleaned streamlines and the rejected \
                ones — so the diagram shows how multi-output nodes lay out.",
        document: doc,
        commentary: None,
    }
}

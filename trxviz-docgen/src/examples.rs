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
    let mut examples = vec![
        save_streamlines_example(),
        purifibre_pipeline_example(),
        volume_mip_and_bundle_cross_section_example(),
    ];
    examples.extend(load_saved_examples(repo_root)?);
    Ok(examples)
}

/// Loaded from `docs/assets/workflows/*.json` — real saved projects
/// (or bare documents) are a better advertisement than hand-built
/// graphs, and they exercise the loader's tolerant parsing.
fn load_saved_examples(repo_root: &Path) -> anyhow::Result<Vec<Example>> {
    let entries: &[(&str, &str, &str, &str)] = &[
        (
            "autotrack",
            "docs/assets/workflows/autotrack.json",
            "Autotrack",
            "A saved autotrack workflow. Loaded from the checked-in JSON via \
             the tolerant project loader and laid out by the same engine that \
             renders the programmatic examples above.",
        ),
        (
            "mpat-live-bundle",
            "docs/assets/workflows/mpat-live-bundle.json",
            "MPAT live bundle",
            "The template from the [MPAT guide](../mpat.md): solve one bundle \
             with MPAT and watch its density build up as the solver runs. \
             [Download the workflow](../assets/workflows/mpat-live-bundle.json).",
        ),
    ];
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

fn volume_mip_and_bundle_cross_section_example() -> Example {
    use trxviz_core::data::loaded_files::{VolumeColormap, VolumeRender3DMode};
    use trxviz_core::data::trx_data::RenderStyle;
    use trxviz_core::renderer::mesh_renderer::SurfaceColormap;
    use trxviz_core::units::Millimeters;
    use trxviz_core::workflow::{BundleSurfaceBuildMode, BundleSurfaceColorMode, DpvFieldName};

    let mut doc = default_document();
    let t1_src = WorkflowNodeUuid(1);
    let t1_disp = WorkflowNodeUuid(2);
    let vol_src = WorkflowNodeUuid(3);
    let vol_disp = WorkflowNodeUuid(4);
    let trx_src = WorkflowNodeUuid(5);
    let color_dpv = WorkflowNodeUuid(6);
    let trx_disp = WorkflowNodeUuid(7);
    let bsurf_build = WorkflowNodeUuid(8);
    let bsurf_disp = WorkflowNodeUuid(9);

    doc.graph.insert_node(
        WorkflowNode {
            uuid: t1_src,
            op: WorkflowNodeKind::VolumeSource { source_id: 1 },
            label: "T1w Volume".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: t1_disp,
            op: WorkflowNodeKind::VolumeDisplay {
                colormap: VolumeColormap::Grayscale,
                opacity: 0.9,
                window_center: 0.45,
                window_width: 0.85,
                render_3d_mode: VolumeRender3DMode::Off,
            },
            label: "T1w 2D Slices".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.connect(
        OutPort {
            node: t1_src,
            output: 0,
        },
        InPort {
            node: t1_disp,
            input: 0,
        },
    );

    doc.graph.insert_node(
        WorkflowNode {
            uuid: vol_src,
            op: WorkflowNodeKind::VolumeSource { source_id: 2 },
            label: "Bundle AFD Volume".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: vol_disp,
            op: WorkflowNodeKind::VolumeDisplay {
                colormap: VolumeColormap::Inferno,
                opacity: 0.85,
                window_center: 0.22,
                window_width: 0.44,
                render_3d_mode: VolumeRender3DMode::Mip,
            },
            label: "3D Volume MIP".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.connect(
        OutPort {
            node: vol_src,
            output: 0,
        },
        InPort {
            node: vol_disp,
            input: 0,
        },
    );

    doc.graph.insert_node(
        WorkflowNode {
            uuid: trx_src,
            op: WorkflowNodeKind::StreamlineSource { source_id: 3 },
            label: "Branch Arbor (.trx)".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: color_dpv,
            op: WorkflowNodeKind::ColorByDPV {
                field: DpvFieldName("q_harmonic".into()),
                colormap: SurfaceColormap::Viridis,
            },
            label: "Color by q_harmonic".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: trx_disp,
            op: WorkflowNodeKind::StreamlineDisplay {
                enabled: true,
                render_style: RenderStyle::Tubes,
                tube_radius_mm: Millimeters(0.55),
                tube_sides: 8,
                slab_half_width_mm: Millimeters(2.0),
                opacity: 0.96,
            },
            label: "Branch Streamtubes".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: bsurf_build,
            op: WorkflowNodeKind::BundleSurfaceBuild {
                per_group: false,
                build_mode: BundleSurfaceBuildMode::MarchingCubes,
                voxel_size_mm: Millimeters(1.5),
                threshold: 0.75,
                smooth_sigma: 1.1,
                min_component_volume_mm3: Millimeters(8.0),
                tube_radius_mm: Millimeters(1.4),
                tube_sides: 12,
                opacity: 0.22,
            },
            label: "Cross-Section Surface".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.insert_node(
        WorkflowNode {
            uuid: bsurf_disp,
            op: WorkflowNodeKind::BundleSurfaceDisplay {
                color_mode: BundleSurfaceColorMode::SourceColors,
                outline_thickness: 2.8,
            },
            label: "Shell + Slice Contours".into(),
        },
        GraphPos::new(0.0, 0.0),
    );
    doc.graph.connect(
        OutPort {
            node: trx_src,
            output: 0,
        },
        InPort {
            node: color_dpv,
            input: 0,
        },
    );
    doc.graph.connect(
        OutPort {
            node: color_dpv,
            output: 0,
        },
        InPort {
            node: trx_disp,
            input: 0,
        },
    );
    doc.graph.connect(
        OutPort {
            node: color_dpv,
            output: 0,
        },
        InPort {
            node: bsurf_build,
            input: 0,
        },
    );
    doc.graph.connect(
        OutPort {
            node: bsurf_build,
            output: 0,
        },
        InPort {
            node: bsurf_disp,
            input: 0,
        },
    );
    doc.next_node_uuid = 10;

    Example {
        slug: "volume-mip-and-bundle-cross-section",
        title: "3D Volume MIP + Branch Tubes + Cross-Section Contours",
        intro: "Combines 2D T1w orthogonal slices (`VolumeRender3DMode::Off`), \
                a 3D GPU ray-marched bundle density MIP (`VolumeRender3DMode::Mip`), \
                DPV-colored branch streamtubes, and a Marching Cubes cross-sectional \
                envelope (`BundleSurfaceBuild` -> `BundleSurfaceDisplay`) that projects \
                local streamline-colored contour rings onto the intersecting T1w slices.",
        document: doc,
        commentary: None,
    }
}

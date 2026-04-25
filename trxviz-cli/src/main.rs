use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use glam::Vec3;
use trxviz_core::headless::{
    AssetArgs, HeadlessRenderOptions, HeadlessSceneExportFormat, HeadlessSceneExportOptions,
    HeadlessView, export_assets_glb, export_project_glb, render_assets_png, render_project_png,
};
use trxviz_core::workflow::load_workflow_project_from_path;
use trxviz_core::workflow::methods::{NON_ENDORSEMENT_NOTICE, generate_methods_report};

#[derive(Parser)]
#[command(name = "trxviz-cli")]
#[command(about = "Headless rendering for TRXViz workflows and scenes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Render(RenderArgs),
    ExportScene(ExportSceneArgs),
    /// Generate a Methods-section markdown file and matching BibTeX for
    /// a saved workflow, so users can cite the upstream authors of the
    /// methods TRXViz re-implements. The two files feed directly into
    /// a Pandoc/CSL pipeline (`pandoc --citeproc`).
    Methods(MethodsArgs),
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ViewArg {
    #[value(name = "3d")]
    View3d,
    #[value(name = "2d")]
    View2d,
    #[value(name = "stage")]
    Stage,
}

#[derive(Args)]
struct RenderArgs {
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "tractogram")]
    tractogram_paths: Vec<PathBuf>,
    #[arg(long = "nifti")]
    nifti_paths: Vec<PathBuf>,
    #[arg(long = "surface")]
    surface_paths: Vec<PathBuf>,
    #[arg(long = "parcellation")]
    parcellation_paths: Vec<PathBuf>,
    #[arg(long = "odx")]
    odx_paths: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    #[arg(long, value_enum, default_value_t = ViewArg::View3d)]
    view: ViewArg,
    #[arg(long, value_parser = parse_vec3)]
    target: Option<Vec3>,
    #[arg(long)]
    azimuth: Option<f32>,
    #[arg(long)]
    elevation: Option<f32>,
    #[arg(long)]
    distance: Option<f32>,
}

#[derive(Args)]
struct ExportSceneArgs {
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "tractogram")]
    tractogram_paths: Vec<PathBuf>,
    #[arg(long = "nifti")]
    nifti_paths: Vec<PathBuf>,
    #[arg(long = "surface")]
    surface_paths: Vec<PathBuf>,
    #[arg(long = "parcellation")]
    parcellation_paths: Vec<PathBuf>,
    #[arg(long = "odx")]
    odx_paths: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    #[arg(long, value_enum, default_value_t = ViewArg::View3d)]
    view: ViewArg,
    #[arg(long, value_parser = parse_vec3)]
    target: Option<Vec3>,
    #[arg(long)]
    azimuth: Option<f32>,
    #[arg(long)]
    elevation: Option<f32>,
    #[arg(long)]
    distance: Option<f32>,
    #[arg(long, default_value_t = true)]
    include_camera: bool,
    #[arg(long, default_value_t = true)]
    include_lights: bool,
    #[arg(long, default_value_t = true)]
    include_slices: bool,
}

#[derive(Args)]
struct MethodsArgs {
    /// Path to the saved TRXViz workflow (.trxviz / .json).
    #[arg(long)]
    project: PathBuf,
    /// Where to write the generated Methods markdown.
    #[arg(long, short = 'o', default_value = "methods.md")]
    out: PathBuf,
    /// Where to write the filtered BibTeX. Contains only entries whose
    /// keys appear in the generated markdown.
    #[arg(long, default_value = "references.bib")]
    bib: PathBuf,
    /// Suppress the non-endorsement banner on stderr. The notice
    /// itself is always embedded in the markdown output regardless;
    /// this only hides the stderr copy.
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Render(args) => run_render(args),
        Command::ExportScene(args) => run_export_scene(args),
        Command::Methods(args) => run_methods(args),
    }
}

fn run_render(args: RenderArgs) -> anyhow::Result<()> {
    let options = HeadlessRenderOptions {
        width: args.width,
        height: args.height,
        view: match args.view {
            ViewArg::View3d => HeadlessView::View3D,
            ViewArg::View2d => HeadlessView::View2D,
            ViewArg::Stage => HeadlessView::InflatedStage,
        },
        target: args.target,
        azimuth_deg: args.azimuth,
        elevation_deg: args.elevation,
        distance: args.distance,
    };

    if let Some(project_path) = args.project {
        if !(args.tractogram_paths.is_empty()
            && args.nifti_paths.is_empty()
            && args.surface_paths.is_empty()
            && args.parcellation_paths.is_empty())
        {
            bail!("cannot combine --project with loose asset arguments");
        }
        render_project_png(&project_path, &args.out, &options)
            .with_context(|| format!("rendering project {}", project_path.display()))?;
        return Ok(());
    }

    if options.view == HeadlessView::View2D {
        bail!("--view 2d currently requires --project");
    }

    let assets = AssetArgs {
        tractogram_paths: args.tractogram_paths,
        nifti_paths: args.nifti_paths,
        surface_paths: args.surface_paths,
        parcellation_paths: args.parcellation_paths,
        odx_paths: args.odx_paths,
    };
    render_assets_png(&assets, &args.out, &options)
        .with_context(|| format!("rendering scene to {}", args.out.display()))?;
    Ok(())
}

fn run_export_scene(args: ExportSceneArgs) -> anyhow::Result<()> {
    let view = match args.view {
        ViewArg::View3d => HeadlessView::View3D,
        ViewArg::Stage => HeadlessView::InflatedStage,
        ViewArg::View2d => bail!("--view 2d is not supported for scene export"),
    };
    let options = HeadlessSceneExportOptions {
        format: HeadlessSceneExportFormat::Glb,
        include_camera: args.include_camera,
        include_lights: args.include_lights,
        include_slices: args.include_slices,
        width: args.width,
        height: args.height,
        view,
        target: args.target,
        azimuth_deg: args.azimuth,
        elevation_deg: args.elevation,
        distance: args.distance,
    };

    if let Some(project_path) = args.project {
        if !(args.tractogram_paths.is_empty()
            && args.nifti_paths.is_empty()
            && args.surface_paths.is_empty()
            && args.parcellation_paths.is_empty())
        {
            bail!("cannot combine --project with loose asset arguments");
        }
        export_project_glb(&project_path, &args.out, &options)
            .with_context(|| format!("exporting project {}", project_path.display()))?;
        return Ok(());
    }

    let assets = AssetArgs {
        tractogram_paths: args.tractogram_paths,
        nifti_paths: args.nifti_paths,
        surface_paths: args.surface_paths,
        parcellation_paths: args.parcellation_paths,
        odx_paths: args.odx_paths,
    };
    export_assets_glb(&assets, &args.out, &options)
        .with_context(|| format!("exporting scene to {}", args.out.display()))?;
    Ok(())
}

fn run_methods(args: MethodsArgs) -> anyhow::Result<()> {
    let project = load_workflow_project_from_path(&args.project)
        .with_context(|| format!("loading workflow {}", args.project.display()))?;

    let report = generate_methods_report(&project.document);

    std::fs::write(&args.out, &report.body_markdown)
        .with_context(|| format!("writing {}", args.out.display()))?;
    std::fs::write(&args.bib, &report.bibtex)
        .with_context(|| format!("writing {}", args.bib.display()))?;

    if !args.quiet {
        // Print the banner on stderr so stdout stays empty for
        // scripting (the markdown file itself is the real output).
        eprintln!("── TRXViz: not the authoritative implementation ──");
        eprintln!("{NON_ENDORSEMENT_NOTICE}");
        eprintln!();
        eprintln!(
            "Wrote methods prose to {} and filtered bibliography to {} ({} citation keys).",
            args.out.display(),
            args.bib.display(),
            report.citation_keys.len(),
        );
    }

    Ok(())
}

fn parse_vec3(value: &str) -> Result<Vec3, String> {
    let parts = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err("expected x,y,z".to_string());
    }
    let x = parts[0]
        .parse::<f32>()
        .map_err(|_| "invalid x component".to_string())?;
    let y = parts[1]
        .parse::<f32>()
        .map_err(|_| "invalid y component".to_string())?;
    let z = parts[2]
        .parse::<f32>()
        .map_err(|_| "invalid z component".to_string())?;
    Ok(Vec3::new(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_vec3_accepts_three_components() {
        assert_eq!(parse_vec3("1,2,3").unwrap(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(
            parse_vec3(" 1.5, -2, 0 ").unwrap(),
            Vec3::new(1.5, -2.0, 0.0)
        );
    }

    #[test]
    fn parse_vec3_rejects_invalid_values() {
        assert!(parse_vec3("1,2").is_err());
        assert!(parse_vec3("1,2,nope").is_err());
    }

    #[test]
    fn clap_parses_render_project_command() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "render",
            "--project",
            "workflow.json",
            "--out",
            "scene.png",
            "--width",
            "800",
            "--height",
            "600",
            "--target",
            "1,2,3",
        ]);

        match cli.command {
            Command::Render(args) => {
                assert_eq!(args.project, Some(PathBuf::from("workflow.json")));
                assert_eq!(args.out, PathBuf::from("scene.png"));
                assert_eq!(args.width, 800);
                assert_eq!(args.height, 600);
                assert!(matches!(args.view, ViewArg::View3d));
                assert_eq!(args.target, Some(Vec3::new(1.0, 2.0, 3.0)));
            }
            _ => panic!("expected render command"),
        }
    }

    #[test]
    fn clap_parses_tractogram_flag() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "render",
            "--tractogram",
            "bundle.tck.gz",
            "--out",
            "scene.png",
        ]);

        match cli.command {
            Command::Render(args) => {
                assert_eq!(args.tractogram_paths, vec![PathBuf::from("bundle.tck.gz")]);
            }
            _ => panic!("expected render command"),
        }
    }

    #[test]
    fn clap_parses_2d_view_flag() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "render",
            "--project",
            "workflow.json",
            "--view",
            "2d",
            "--out",
            "scene.png",
        ]);

        match cli.command {
            Command::Render(args) => {
                assert!(matches!(args.view, ViewArg::View2d));
            }
            _ => panic!("expected render command"),
        }
    }

    #[test]
    fn clap_parses_export_scene_command() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "export-scene",
            "--project",
            "workflow.json",
            "--out",
            "scene.glb",
            "--width",
            "1600",
        ]);

        match cli.command {
            Command::ExportScene(args) => {
                assert_eq!(args.project, Some(PathBuf::from("workflow.json")));
                assert_eq!(args.out, PathBuf::from("scene.glb"));
                assert_eq!(args.width, 1600);
                assert!(args.include_camera);
                assert!(args.include_lights);
                assert!(args.include_slices);
            }
            _ => panic!("expected export-scene command"),
        }
    }

    #[test]
    fn clap_parses_methods_command_with_defaults() {
        let cli = Cli::parse_from(["trxviz-cli", "methods", "--project", "workflow.trxviz"]);
        match cli.command {
            Command::Methods(args) => {
                assert_eq!(args.project, PathBuf::from("workflow.trxviz"));
                assert_eq!(args.out, PathBuf::from("methods.md"));
                assert_eq!(args.bib, PathBuf::from("references.bib"));
                assert!(!args.quiet);
            }
            _ => panic!("expected methods command"),
        }
    }

    #[test]
    fn clap_parses_methods_command_with_overrides() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "methods",
            "--project",
            "workflow.trxviz",
            "-o",
            "/tmp/m.md",
            "--bib",
            "/tmp/r.bib",
            "--quiet",
        ]);
        match cli.command {
            Command::Methods(args) => {
                assert_eq!(args.out, PathBuf::from("/tmp/m.md"));
                assert_eq!(args.bib, PathBuf::from("/tmp/r.bib"));
                assert!(args.quiet);
            }
            _ => panic!("expected methods command"),
        }
    }

    #[test]
    fn run_methods_writes_markdown_and_bib_from_real_project() {
        use std::fs;
        use trxviz_core::workflow::{
            DipyDirectionGetter, GraphPos, WorkflowGraph, WorkflowNode, WorkflowNodeKind,
            WorkflowNodeUuid, default_document, load_workflow_project_from_path,
            save_workflow_project_to_path,
        };

        // Build a minimal project with a Purifibre node so the
        // generated report has at least one method citation in
        // addition to `trxviz`.
        let mut doc = default_document();
        let mut graph = WorkflowGraph::new();
        graph.insert_node(
            WorkflowNode {
                uuid: WorkflowNodeUuid(1),
                op: WorkflowNodeKind::Purifibre {
                    trim_fraction: 0.1,
                    puri_fraction: 0.15,
                    spherical_smoothing_deg: 15.0,
                },
                label: "puri".into(),
            },
            GraphPos::ZERO,
        );
        // Throw in a PTT tractography node so we exercise the GPU-PTT
        // citation through the full CLI path.
        graph.insert_node(
            WorkflowNode {
                uuid: WorkflowNodeUuid(2),
                op: WorkflowNodeKind::DipyTractography {
                    step_size_mm: 0.5,
                    max_angle_deg: 60.0,
                    min_len_mm: 10.0,
                    max_len_mm: 300.0,
                    fixel_threshold: 0.1,
                    relative_peak_threshold: 0.25,
                    seeds_per_voxel: 1,
                    max_points: 501,
                    rng_seed: 42,
                    direction_getter: DipyDirectionGetter::Ptt {
                        probe_length_mm: 1.5,
                        probe_quality: 4,
                        probe_radius_mm: 0.0,
                        probe_count: 1,
                        max_curvature_per_mm: 1.0,
                        data_support_exponent: 1.0,
                        min_data_support: 0.05,
                        rejection_sampling_max_try: 100,
                    },
                },
                label: "track".into(),
            },
            GraphPos::ZERO,
        );
        doc.graph = graph;

        let dir = std::env::temp_dir().join(format!("trxviz-methods-cli-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let project_path = dir.join("wf.trxviz");
        let out_md = dir.join("methods.md");
        let out_bib = dir.join("references.bib");

        save_workflow_project_to_path(&doc, &project_path).unwrap();
        // Round-trip through load to match what the CLI does.
        load_workflow_project_from_path(&project_path).unwrap();

        run_methods(MethodsArgs {
            project: project_path.clone(),
            out: out_md.clone(),
            bib: out_bib.clone(),
            quiet: true,
        })
        .unwrap();

        let md = fs::read_to_string(&out_md).unwrap();
        let bib = fs::read_to_string(&out_bib).unwrap();

        assert!(md.contains("Not the authoritative implementation"));
        assert!(md.contains("[@trxviz]"));
        assert!(md.contains("Purifibre"));
        assert!(md.contains("Parallel Transport"));

        // Filtered bib must contain every key the markdown cites and
        // nothing more exotic. Spot-check a couple.
        assert!(bib.contains("@software{trxviz"));
        assert!(bib.contains("purifibre"));
        assert!(bib.contains("gpustreamlines"));
        assert!(bib.contains("ptt"));
        // Unused keys from citations.bib are stripped.
        assert!(!bib.contains("yeh2013gqi"));

        let _ = fs::remove_dir_all(&dir);
    }

    /// Pipe the CLI's emitted markdown + bib through `pandoc --citeproc`
    /// to catch broken `[@key]` citations early. Pandoc resolves each
    /// citation against the bibtex; an unknown key produces a warning
    /// to stderr and a literal `[?]` in the output. We assert neither.
    /// Skipped when pandoc isn't installed (e.g. minimal CI images).
    #[test]
    fn cli_methods_output_passes_pandoc_citeproc_clean() {
        use std::fs;
        use std::process::Command as Proc;
        use trxviz_core::workflow::{
            GraphPos, WorkflowGraph, WorkflowNode, WorkflowNodeKind, WorkflowNodeUuid,
            default_document, save_workflow_project_to_path,
        };

        if Proc::new("pandoc").arg("--version").output().is_err() {
            eprintln!("skipping: pandoc not on PATH");
            return;
        }

        let mut doc = default_document();
        let mut graph = WorkflowGraph::new();
        graph.insert_node(
            WorkflowNode {
                uuid: WorkflowNodeUuid(1),
                op: WorkflowNodeKind::Purifibre {
                    trim_fraction: 0.1,
                    puri_fraction: 0.15,
                    spherical_smoothing_deg: 15.0,
                },
                label: "puri".into(),
            },
            GraphPos::ZERO,
        );
        doc.graph = graph;

        let dir = std::env::temp_dir().join(format!("trxviz-pandoc-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let project_path = dir.join("wf.trxviz");
        let out_md = dir.join("methods.md");
        let out_bib = dir.join("references.bib");
        save_workflow_project_to_path(&doc, &project_path).unwrap();

        run_methods(MethodsArgs {
            project: project_path,
            out: out_md.clone(),
            bib: out_bib.clone(),
            quiet: true,
        })
        .unwrap();

        let output = Proc::new("pandoc")
            .arg("--citeproc")
            .arg("--bibliography")
            .arg(&out_bib)
            .arg("--to=plain")
            .arg(&out_md)
            .output()
            .expect("pandoc failed to launch");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "pandoc exited non-zero: stdout={stdout}\nstderr={stderr}",
        );
        // pandoc renders unknown keys as a literal `**[Smith?]**` (or
        // similar bracket-question-mark token); the lowercase phrase
        // it logs to stderr is "Citeproc: citation ... not found".
        assert!(
            !stderr.to_lowercase().contains("not found"),
            "pandoc reported missing citations: {stderr}",
        );
        assert!(
            !stdout.contains("[?]"),
            "pandoc rendered an unresolved citation token: {stdout}",
        );
        // The body should still mention Purifibre — sanity check that
        // pandoc consumed our actual file rather than failing silently.
        assert!(
            stdout.contains("Purifibre"),
            "pandoc output missing expected content: {stdout}",
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clap_parses_export_scene_stage_command() {
        let cli = Cli::parse_from([
            "trxviz-cli",
            "export-scene",
            "--project",
            "workflow.json",
            "--out",
            "scene.glb",
            "--view",
            "stage",
        ]);

        match cli.command {
            Command::ExportScene(args) => {
                assert_eq!(args.project, Some(PathBuf::from("workflow.json")));
                assert_eq!(args.out, PathBuf::from("scene.glb"));
                assert!(matches!(args.view, ViewArg::Stage));
            }
            _ => panic!("expected export-scene command"),
        }
    }
}

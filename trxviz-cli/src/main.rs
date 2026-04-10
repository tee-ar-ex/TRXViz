use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use glam::Vec3;
use trxviz_core::headless::{
    AssetArgs, HeadlessRenderOptions, render_assets_png, render_project_png,
};

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
}

#[derive(Args)]
struct RenderArgs {
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "trx")]
    trx_paths: Vec<PathBuf>,
    #[arg(long = "nifti")]
    nifti_paths: Vec<PathBuf>,
    #[arg(long = "surface")]
    surface_paths: Vec<PathBuf>,
    #[arg(long = "parcellation")]
    parcellation_paths: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    #[arg(long, value_parser = parse_vec3)]
    target: Option<Vec3>,
    #[arg(long)]
    azimuth: Option<f32>,
    #[arg(long)]
    elevation: Option<f32>,
    #[arg(long)]
    distance: Option<f32>,
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
    }
}

fn run_render(args: RenderArgs) -> anyhow::Result<()> {
    let options = HeadlessRenderOptions {
        width: args.width,
        height: args.height,
        target: args.target,
        azimuth_deg: args.azimuth,
        elevation_deg: args.elevation,
        distance: args.distance,
    };

    if let Some(project_path) = args.project {
        if !(args.trx_paths.is_empty()
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

    let assets = AssetArgs {
        trx_paths: args.trx_paths,
        nifti_paths: args.nifti_paths,
        surface_paths: args.surface_paths,
        parcellation_paths: args.parcellation_paths,
    };
    render_assets_png(&assets, &args.out, &options)
        .with_context(|| format!("rendering scene to {}", args.out.display()))?;
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
                assert_eq!(args.target, Some(Vec3::new(1.0, 2.0, 3.0)));
            }
        }
    }
}

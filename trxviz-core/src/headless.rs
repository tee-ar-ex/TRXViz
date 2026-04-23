//! Headless rendering entrypoints for project JSON and loose-asset scene capture.

mod bake;
mod export_glb;
mod gpu_context;
mod readback;
mod render_2d;
mod render_3d;
mod render_data;
mod scene_loader;
mod workflow_driver;

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use glam::Vec3;

use self::export_glb::{build_glb_scene, compute_scene_bounds};
use self::gpu_context::{build_gpu_resources, create_gpu_context};
use self::render_2d::render_scene2d_to_png;
use self::render_3d::render_scene3d_to_png;
use self::render_data::{build_camera, build_render_data, compute_render_bounds};
use self::scene_loader::{load_asset_args_state, load_project_state};
use self::workflow_driver::{ensure_export_tube_geometry, execute_workflow_to_completion};
use crate::data::orientation_field::BoundaryGlyphColorMode;
use crate::data::trx_data::RenderStyle;
use crate::renderer::mesh_renderer::MeshDrawStyle;
use crate::scene::{HeadlessScene, HeadlessWorkflowState};
use crate::units::Millimeters;

#[cfg(test)]
use crate::data::odx_data::OdxScene;
#[cfg(test)]
use crate::workflow::WorkflowNodeUuid;
#[cfg(test)]
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadlessView {
    View3D,
    View2D,
    InflatedStage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadlessSceneExportFormat {
    Glb,
}

pub struct HeadlessSceneExportOptions {
    pub format: HeadlessSceneExportFormat,
    pub include_camera: bool,
    pub include_lights: bool,
    pub include_slices: bool,
    pub width: u32,
    pub height: u32,
    pub view: HeadlessView,
    pub target: Option<Vec3>,
    pub azimuth_deg: Option<f32>,
    pub elevation_deg: Option<f32>,
    pub distance: Option<f32>,
}

impl Default for HeadlessSceneExportOptions {
    fn default() -> Self {
        Self {
            format: HeadlessSceneExportFormat::Glb,
            include_camera: true,
            include_lights: true,
            include_slices: true,
            width: 1920,
            height: 1080,
            view: HeadlessView::View3D,
            target: None,
            azimuth_deg: None,
            elevation_deg: None,
            distance: None,
        }
    }
}

pub struct HeadlessRenderOptions {
    pub width: u32,
    pub height: u32,
    pub view: HeadlessView,
    pub target: Option<Vec3>,
    pub azimuth_deg: Option<f32>,
    pub elevation_deg: Option<f32>,
    pub distance: Option<f32>,
}

impl Default for HeadlessRenderOptions {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            view: HeadlessView::View3D,
            target: None,
            azimuth_deg: None,
            elevation_deg: None,
            distance: None,
        }
    }
}

#[derive(Default)]
pub struct AssetArgs {
    pub tractogram_paths: Vec<PathBuf>,
    pub nifti_paths: Vec<PathBuf>,
    pub surface_paths: Vec<PathBuf>,
    pub parcellation_paths: Vec<PathBuf>,
    pub odx_paths: Vec<PathBuf>,
}

struct HeadlessRenderData {
    surface_draws: Vec<(usize, usize, MeshDrawStyle)>,
    volume_draws: Vec<VolumeDrawInfo>,
    streamline_draws: Vec<StreamlineDrawInfo>,
    bundle_draws: Vec<BundleDrawInfo>,
    any_visible_streamlines: bool,
    glyph_visible: bool,
    glyph_color_mode: BoundaryGlyphColorMode,
    glyph_density_3d_step: u32,
    glyph_slice_density_step: u32,
    odx_visible: bool,
    odx_fixel_3d_visible: bool,
    odx_fixel_2d_visible: bool,
    fixel_3d_line_width: f32,
    fixel_3d_opacity: f32,
    fixel_3d_colormap_code: u32,
    fixel_3d_scalar_range: [f32; 2],
    fixel_3d_opacity_gate: [f32; 4],
    fixel_2d_line_width: f32,
    fixel_2d_slab_half_width_mm: Millimeters,
    fixel_2d_opacity: f32,
    fixel_2d_colormap_code: u32,
    fixel_2d_scalar_range: [f32; 2],
    fixel_2d_opacity_gate: [f32; 4],
    odf_glyph_opacity: f32,
    odf_glyph_gloss: f32,
}

pub(super) struct VolumeDrawInfo {
    file_id: usize,
    window_center: f32,
    window_width: f32,
    colormap: u32,
    opacity: f32,
}

struct StreamlineDrawInfo {
    file_id: usize,
    visible: bool,
    render_style: RenderStyle,
    tube_radius: f32,
}

struct BundleDrawInfo {
    file_id: usize,
    opacity: f32,
}

#[derive(Clone, Copy)]
struct SceneBounds {
    min: Vec3,
    max: Vec3,
}

/// Load a workflow project and render the visible scene to a PNG.
#[cfg(feature = "png-export")]
pub fn render_project_png(
    project_path: &Path,
    output_path: &Path,
    options: &HeadlessRenderOptions,
) -> anyhow::Result<()> {
    let (scene, workflow) = load_project_state(project_path)?;
    render_loaded_scene(scene, workflow, output_path, options)
}

/// Build a default scene from loose assets and render it to a PNG.
#[cfg(feature = "png-export")]
pub fn render_assets_png(
    args: &AssetArgs,
    output_path: &Path,
    options: &HeadlessRenderOptions,
) -> anyhow::Result<()> {
    let (scene, workflow) = load_asset_args_state(args)?;
    render_loaded_scene(scene, workflow, output_path, options)
}

/// Load a workflow project and export the visible 3D scene to a GLB.
pub fn export_project_glb(
    project_path: &Path,
    output_path: &Path,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<()> {
    let (scene, workflow) = load_project_state(project_path)?;
    export_loaded_scene(&scene, workflow, output_path, options)
}

/// Build a default scene from loose assets and export the visible 3D scene to a GLB.
pub fn export_assets_glb(
    args: &AssetArgs,
    output_path: &Path,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<()> {
    let (scene, workflow) = load_asset_args_state(args)?;
    export_loaded_scene(&scene, workflow, output_path, options)
}

/// Export an already-loaded GUI/headless scene state to GLB without going through project JSON.
pub fn export_state_glb(
    scene: &HeadlessScene,
    workflow: HeadlessWorkflowState,
    output_path: &Path,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<()> {
    export_loaded_scene(scene, workflow, output_path, options)
}

#[cfg(feature = "png-export")]
fn render_loaded_scene(
    mut scene: HeadlessScene,
    mut workflow: HeadlessWorkflowState,
    output_path: &Path,
    options: &HeadlessRenderOptions,
) -> anyhow::Result<()> {
    execute_workflow_to_completion(&scene, &mut workflow)?;
    let gpu = create_gpu_context()?;
    let mut resources = build_gpu_resources(&gpu.device, &gpu.queue, &scene, &workflow)
        .context("building GPU resources")?;
    let render_3d = workflow.document.render_3d.clone().unwrap_or_default();
    let render_data = build_render_data(&scene, &workflow, options.view);
    if render_data.glyph_visible {
        scene.boundary_field = workflow
            .runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible)
            .and_then(|draw| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&draw.build_node_uuid)
            })
            .map(|cache| cache.field.clone());
    }
    if options.view == HeadlessView::View2D {
        return render_scene2d_to_png(
            &gpu.device,
            &gpu.queue,
            &mut resources,
            &render_data,
            workflow.document.slice_view_ui.clone(),
            &scene,
            options.width,
            options.height,
            output_path,
        );
    }
    let camera_bounds = if options.view == HeadlessView::InflatedStage {
        compute_render_bounds(&scene, &render_data)
    } else {
        resources.bounds
    };
    let camera = build_camera(
        &camera_bounds,
        workflow.document.camera_3d,
        options,
        options.width as f32 / options.height as f32,
    );
    render_scene3d_to_png(
        &gpu.device,
        &gpu.queue,
        &mut resources,
        &render_data,
        &camera,
        &render_3d,
        if options.view == HeadlessView::InflatedStage {
            [false; 3]
        } else {
            scene.slice_visible
        },
        options.width,
        options.height,
        output_path,
    )
}

fn export_loaded_scene(
    scene: &HeadlessScene,
    mut workflow: HeadlessWorkflowState,
    output_path: &Path,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<()> {
    if options.format != HeadlessSceneExportFormat::Glb {
        bail!("unsupported scene export format");
    }

    execute_workflow_to_completion(scene, &mut workflow)?;
    ensure_export_tube_geometry(&mut workflow)?;
    let render_data = build_render_data(scene, &workflow, options.view);
    let bounds = if options.view == HeadlessView::InflatedStage {
        compute_render_bounds(scene, &render_data)
    } else {
        compute_scene_bounds(scene, &workflow)
    };
    let camera = build_camera(
        &bounds,
        workflow.document.camera_3d,
        &HeadlessRenderOptions {
            width: options.width,
            height: options.height,
            view: options.view,
            target: options.target,
            azimuth_deg: options.azimuth_deg,
            elevation_deg: options.elevation_deg,
            distance: options.distance,
        },
        options.width as f32 / options.height.max(1) as f32,
    );
    let render_3d = workflow.document.render_3d.clone().unwrap_or_default();
    let bytes = build_glb_scene(scene, &workflow, &render_data, &camera, &render_3d, options)
        .context("building GLB scene")?;
    std::fs::write(output_path, bytes)
        .with_context(|| format!("writing GLB to {}", output_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::odx_data::{FixelField, FixelScalars};
    use odx_rs::OdxBuilder;

    fn make_test_fixel_scene() -> Arc<OdxScene> {
        let full = odx_rs::formats::dsistudio_odf8::full_vertices_ras().to_vec();
        let faces = odx_rs::formats::dsistudio_odf8::faces().to_vec();
        let mut builder = OdxBuilder::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [1, 1, 1],
            vec![1u8],
        );
        builder.set_sphere(full, faces);
        builder.push_voxel_peaks(&[[1.0, 0.0, 0.0]]);
        Arc::new(OdxScene::from_dataset(builder.finalize().unwrap()).unwrap())
    }

    fn make_test_fixel_draw(
        scene: &Arc<OdxScene>,
        node_uuid: WorkflowNodeUuid,
        line_width: f32,
        opacity: f32,
        slab_thickness_mm: Millimeters,
        visible: bool,
        colormap_code: u32,
        scalar_range: (f32, f32),
    ) -> crate::workflow::FixelDrawPlan {
        crate::workflow::FixelDrawPlan {
            node_uuid,
            field: FixelField {
                source_id: 17,
                scene: scene.clone(),
                scalars: FixelScalars::from_scalar(17, "qa".into(), vec![scalar_range.0]),
                colormap_code,
                scalar_range,
            },
            line_width,
            length_scale: 1.0,
            opacity,
            offset_from_slice: 0.0,
            slab_thickness_mm,
            visible,
            colormap_code,
            scalar_range,
            opacity_gate: crate::workflow::OpacityGate::default(),
        }
    }

    #[test]
    fn build_render_data_keeps_2d_and_3d_fixel_styles_independent() {
        let odx_scene = make_test_fixel_scene();
        let mut scene = HeadlessScene {
            odx_scene: Some(odx_scene.clone()),
            ..Default::default()
        };
        scene.slice_visible = [true, true, true];

        let mut workflow = HeadlessWorkflowState::default();
        workflow
            .runtime
            .scene_plan
            .fixel_3d_draws
            .push(make_test_fixel_draw(
                &odx_scene,
                WorkflowNodeUuid(101),
                0.125,
                0.4,
                Millimeters(8.0),
                true,
                3,
                (10.0, 20.0),
            ));
        workflow
            .runtime
            .scene_plan
            .fixel_2d_draws
            .push(make_test_fixel_draw(
                &odx_scene,
                WorkflowNodeUuid(202),
                0.5,
                0.9,
                Millimeters(14.0),
                true,
                4,
                (30.0, 40.0),
            ));

        let render_data = build_render_data(&scene, &workflow, HeadlessView::View3D);

        assert_eq!(render_data.fixel_3d_line_width, 0.125);
        assert_eq!(render_data.fixel_3d_opacity, 0.4);
        assert_eq!(render_data.fixel_3d_colormap_code, 3);
        assert_eq!(render_data.fixel_3d_scalar_range, [10.0, 20.0]);

        assert_eq!(render_data.fixel_2d_line_width, 0.5);
        assert_eq!(render_data.fixel_2d_opacity, 0.9);
        assert_eq!(render_data.fixel_2d_colormap_code, 4);
        assert_eq!(render_data.fixel_2d_scalar_range, [30.0, 40.0]);
        assert_eq!(render_data.fixel_2d_slab_half_width_mm, Millimeters(7.0));
    }
}

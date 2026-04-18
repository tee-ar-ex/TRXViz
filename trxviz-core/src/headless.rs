//! Headless rendering entrypoints for project JSON and loose-asset scene capture.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use glam::{Mat4, Vec3};
use image::{ColorType, ImageEncoder};
use pollster::block_on;
use serde_json::{Map, Value, json};
use trx_rs::{AnyTrxFile, ConversionOptions};

use crate::data::bundle_mesh::BundleMesh;
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{
    FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use crate::data::nifti_data::NiftiVolume;
use crate::data::odx_data::OdxScene;
use crate::data::orientation_field::BoundaryGlyphColorMode;
use crate::data::parcellation_data::{ParcellationVolume, guess_label_table_path};
use crate::data::trx_data::{RenderStyle, TrxGpuData};
use crate::lighting::{SceneLightingParams, WorkflowRender3D};
use crate::renderer::background_renderer::BackgroundResources;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::fixel_renderer::FixelResources;
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};
use crate::scene::{
    HeadlessScene, HeadlessWorkflowState, LoadedGiftiSurface, LoadedParcellationSource,
    LoadedStreamlineSource, direct_streamline_import_warnings,
};
use crate::workflow::{
    BundleSurfacePlan, CachedBoundaryField, CachedBundleSurfaceMeshes, CachedDerivedStreamline,
    CachedSurfaceQuery, CachedSurfaceStreamlineMap, CachedTubeGeometry, LoadedParcellation,
    ParcellationAsset, WorkflowAssetDocument, WorkflowCamera3D, WorkflowJobOutput,
    WorkflowJobPayload, WorkflowNodeUuid, WorkflowSliceViewKind, WorkflowSliceViewUi,
    WorkflowView2DMode, add_default_nodes_for_asset, ensure_node_uuids, evaluate_scene_plan,
    load_workflow_project_from_path, mark_expensive_success, resolve_document_asset_paths,
    run_workflow_job, save_streamline_plan, set_default_odx_fixel_3d_visibility,
    workflow_boundary_plan_fingerprint,
    workflow_bundle_display_fingerprint, workflow_bundle_plan_fingerprint,
    workflow_reactive_streamline_fingerprint, workflow_streamline_fingerprint,
    workflow_surface_projection_fingerprint, workflow_surface_query_fingerprint,
};

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const GLTF_AXIS_CONVERSION: glam::Mat3 = glam::Mat3::from_cols_array(&[
    1.0, 0.0, 0.0, //
    0.0, 0.0, -1.0, //
    0.0, 1.0, 0.0, //
]);

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
    fixel_line_width: f32,
    fixel_2d_slab_half_width_mm: f32,
    fixel_opacity: f32,
    odf_glyph_opacity: f32,
    odf_glyph_gloss: f32,
}

struct VolumeDrawInfo {
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

struct GpuSceneResources {
    background: BackgroundResources,
    streamlines: AllStreamlineResources,
    slices: AllSliceResources,
    meshes: MeshResources,
    glyphs: GlyphResources,
    fixels: FixelResources,
    bounds: SceneBounds,
}

#[derive(Clone, Copy)]
struct ViewportRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct SlicePanel {
    rect: ViewportRect,
    axis_index: usize,
    slice_index: usize,
    slice_pos: f32,
}

/// Load a workflow project and render the visible scene to a PNG.
pub fn render_project_png(
    project_path: &Path,
    output_path: &Path,
    options: &HeadlessRenderOptions,
) -> anyhow::Result<()> {
    let (scene, workflow) = load_project_state(project_path)?;
    render_loaded_scene(scene, workflow, output_path, options)
}

/// Build a default scene from loose assets and render it to a PNG.
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
            workflow.slice_view_ui,
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

fn load_project_state(
    project_path: &Path,
) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut project = load_workflow_project_from_path(project_path).map_err(|err| anyhow!(err))?;
    resolve_document_asset_paths(&mut project.document, project_path);

    let mut scene = HeadlessScene::default();
    for asset in project.document.assets.clone() {
        match asset {
            WorkflowAssetDocument::Streamlines { id, path, imported } => {
                let source = if imported {
                    let tractogram = trx_rs::read_tractogram(&path, &ConversionOptions::default())
                        .map_err(|err| anyhow!(err.to_string()))?;
                    LoadedStreamlineSource {
                        data: TrxGpuData::from_tractogram(&tractogram)
                            .map_err(|err| anyhow!(err.to_string()))?,
                        backing: StreamlineBacking::Imported(Arc::new(tractogram)),
                        warnings: direct_streamline_import_warnings(
                            &path,
                            &ConversionOptions {
                                vtk_coordinate_mode: trx_rs::VtkCoordinateMode::HeaderOrWarn,
                                ..Default::default()
                            },
                        ),
                    }
                } else {
                    let any = AnyTrxFile::load(&path).map_err(|err| anyhow!(err.to_string()))?;
                    load_streamline_source_from_any(any)?
                };
                apply_loaded_trx(&mut scene, path, source, Some(id));
            }
            WorkflowAssetDocument::Volume { id, path } => {
                let volume = NiftiVolume::load(&path).map_err(|err| anyhow!(err.to_string()))?;
                apply_loaded_nifti(&mut scene, path, volume, Some(id));
            }
            WorkflowAssetDocument::Cifti { .. } => {}
            WorkflowAssetDocument::Surface { id, path } => {
                let surface =
                    GiftiSurfaceData::load(&path).map_err(|err| anyhow!(err.to_string()))?;
                apply_loaded_surface(&mut scene, path, surface, Some(id));
            }
            WorkflowAssetDocument::Parcellation {
                id,
                path,
                label_table_path,
            } => {
                let source = ParcellationVolume::load(&path, label_table_path.as_deref())
                    .map(|data| LoadedParcellationSource {
                        data,
                        label_table_path,
                    })
                    .map_err(|err| anyhow!(err.to_string()))?;
                apply_loaded_parcellation(&mut scene, path, source, Some(id));
            }
            WorkflowAssetDocument::Odx { id, path } => {
                let odx_scene = OdxScene::open(&path)
                    .map_err(|err| anyhow!("opening ODX {}: {}", path.display(), err))?;
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                scene.odx_files.push(crate::data::loaded_files::LoadedOdx {
                    id,
                    name,
                    path,
                    scene: Arc::new(odx_scene),
                    visible: true,
                });
            }
        }
    }

    if let Some(slice_view) = project.document.slice_view_3d {
        scene.slice_visible = slice_view.visible;
        scene.slice_world_offsets = slice_view.positions_ras;
        if let Some(nifti) = scene.nifti_files.first() {
            scene.slice_indices = [
                nifti
                    .volume
                    .nearest_slice_index(0, slice_view.positions_ras[0]),
                nifti
                    .volume
                    .nearest_slice_index(1, slice_view.positions_ras[1]),
                nifti
                    .volume
                    .nearest_slice_index(2, slice_view.positions_ras[2]),
            ];
        }
    } else if let Some(slice_visible) = project.document.slice_visible_3d {
        scene.slice_visible = slice_visible;
    }

    let workflow = HeadlessWorkflowState {
        document: project.document,
        slice_view_ui: project.slice_view_ui,
        project_path: Some(project_path.to_path_buf()),
        ..Default::default()
    };
    Ok((scene, workflow))
}

fn load_asset_args_state(
    args: &AssetArgs,
) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut scene = HeadlessScene::default();
    let mut workflow = HeadlessWorkflowState::default();

    for path in &args.tractogram_paths {
        let (source, imported) = load_streamline_source(path)?;
        let id = apply_loaded_trx(&mut scene, path.clone(), source, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Streamlines {
                id,
                path: path.clone(),
                imported,
            },
            Some(
                scene
                    .trx_files
                    .iter()
                    .find(|asset| asset.id == id)
                    .map(|asset| asset.data.nb_streamlines.min(30_000))
                    .unwrap_or(30_000),
            ),
        );
    }

    for path in &args.nifti_paths {
        let volume = NiftiVolume::load(path).map_err(|err| anyhow!(err.to_string()))?;
        let id = apply_loaded_nifti(&mut scene, path.clone(), volume, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Volume {
                id,
                path: path.clone(),
            },
            None,
        );
    }

    for path in &args.surface_paths {
        let surface = GiftiSurfaceData::load(path).map_err(|err| anyhow!(err.to_string()))?;
        let id = apply_loaded_surface(&mut scene, path.clone(), surface, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Surface {
                id,
                path: path.clone(),
            },
            None,
        );
    }

    for path in &args.parcellation_paths {
        let label_table_path = guess_label_table_path(path);
        let source = ParcellationVolume::load(path, label_table_path.as_deref())
            .map(|data| LoadedParcellationSource {
                data,
                label_table_path: label_table_path.clone(),
            })
            .map_err(|err| anyhow!(err.to_string()))?;
        let id = apply_loaded_parcellation(&mut scene, path.clone(), source, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Parcellation {
                id,
                path: path.clone(),
                label_table_path,
            },
            None,
        );
    }

    for path in &args.odx_paths {
        let odx_scene =
            OdxScene::open(path).with_context(|| format!("loading ODX file {}", path.display()))?;
        let show_fixel_3d_by_default = odx_scene.glyph_source_kind().is_none();
        let dims = odx_scene.dimensions();
        // Use the ODX affine to set volume center and extent for camera framing.
        let header = odx_rs::OdxDataset::open(path)
            .ok()
            .map(|ds| ds.header().clone());
        if let Some(h) = &header {
            let affine = &h.voxel_to_rasmm;
            let mid = [
                dims[0] as f64 * 0.5,
                dims[1] as f64 * 0.5,
                dims[2] as f64 * 0.5,
            ];
            let cx = (affine[0][0] * mid[0]
                + affine[0][1] * mid[1]
                + affine[0][2] * mid[2]
                + affine[0][3]) as f32;
            let cy = (affine[1][0] * mid[0]
                + affine[1][1] * mid[1]
                + affine[1][2] * mid[2]
                + affine[1][3]) as f32;
            let cz = (affine[2][0] * mid[0]
                + affine[2][1] * mid[1]
                + affine[2][2] * mid[2]
                + affine[2][3]) as f32;
            scene.volume_center = Vec3::new(cx, cy, cz);
            let dx = (affine[0][0].powi(2) + affine[1][0].powi(2) + affine[2][0].powi(2)).sqrt();
            scene.volume_extent = (dims[0].max(dims[1]).max(dims[2]) as f64 * dx) as f32;
        }
        // Set default slice indices to volume midpoints.
        scene.slice_indices = [
            dims[0] as usize / 2,
            dims[1] as usize / 2,
            dims[2] as usize / 2,
        ];
        let odx_arc = Arc::new(odx_scene);
        let id = scene.next_file_id;
        scene.next_file_id += 1;
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        scene.odx_files.push(crate::data::loaded_files::LoadedOdx {
            id,
            name,
            path: path.clone(),
            scene: Arc::clone(&odx_arc),
            visible: true,
        });
        scene.odx_scene = Some(odx_arc);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Odx {
                id,
                path: path.clone(),
            },
            None,
        );
        let _ = set_default_odx_fixel_3d_visibility(
            &mut workflow.document,
            id,
            show_fixel_3d_by_default,
        );
    }

    if workflow.document.assets.is_empty() && scene.odx_scene.is_none() {
        bail!("no input assets were provided");
    }

    Ok((scene, workflow))
}

fn register_asset_default_nodes(
    document: &mut crate::workflow::WorkflowDocument,
    asset: WorkflowAssetDocument,
    streamline_limit: Option<usize>,
) {
    document.assets.push(asset.clone());
    let pos = crate::workflow::suggest_asset_branch_origin(document);
    let _ = add_default_nodes_for_asset(document, &asset, pos, streamline_limit);
}

fn load_streamline_source_from_any(any: AnyTrxFile) -> anyhow::Result<LoadedStreamlineSource> {
    TrxGpuData::from_any_trx(&any)
        .map(|data| LoadedStreamlineSource {
            data,
            backing: StreamlineBacking::Native(Arc::new(any)),
            warnings: Vec::new(),
        })
        .map_err(|err| anyhow!(err.to_string()))
}

fn load_streamline_source(path: &Path) -> anyhow::Result<(LoadedStreamlineSource, bool)> {
    let format = trx_rs::detect_format(path).map_err(|err| anyhow!(err.to_string()))?;
    if format == trx_rs::Format::Trx {
        let any = AnyTrxFile::load(path).map_err(|err| anyhow!(err.to_string()))?;
        let source = load_streamline_source_from_any(any)?;
        return Ok((source, false));
    }

    let options = ConversionOptions::default();
    let warnings = direct_streamline_import_warnings(path, &options);
    let tractogram =
        trx_rs::read_tractogram(path, &options).map_err(|err| anyhow!(err.to_string()))?;
    let data = TrxGpuData::from_tractogram(&tractogram).map_err(|err| anyhow!(err.to_string()))?;
    Ok((
        LoadedStreamlineSource {
            data,
            backing: StreamlineBacking::Imported(Arc::new(tractogram)),
            warnings,
        },
        true,
    ))
}

fn allocate_file_id(scene: &mut HeadlessScene, explicit_id: Option<FileId>) -> FileId {
    if let Some(id) = explicit_id {
        scene.next_file_id = scene.next_file_id.max(id + 1);
        id
    } else {
        let id = scene.next_file_id;
        scene.next_file_id += 1;
        id
    }
}

fn apply_loaded_trx(
    scene: &mut HeadlessScene,
    path: PathBuf,
    source: LoadedStreamlineSource,
    explicit_id: Option<FileId>,
) -> FileId {
    let LoadedStreamlineSource {
        data,
        backing,
        warnings,
    } = source;
    let is_first = scene.trx_files.is_empty()
        && scene.nifti_files.is_empty()
        && scene.gifti_surfaces.is_empty();
    scene.volume_center = data.center();
    scene.volume_extent = data.extent();
    if is_first {
        scene.slice_world_offsets = [
            scene.volume_center.z,
            scene.volume_center.y,
            scene.volume_center.x,
        ];
    }
    let id = allocate_file_id(scene, explicit_id);
    let data = Arc::new(data);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file.trx".to_string());
    scene.trx_files.push(LoadedTrx {
        id,
        name,
        path,
        data,
        backing: Some(backing),
        import_warnings: warnings,
    });
    id
}

fn apply_loaded_nifti(
    scene: &mut HeadlessScene,
    path: PathBuf,
    vol: NiftiVolume,
    explicit_id: Option<FileId>,
) -> FileId {
    let first_nifti = scene.nifti_files.is_empty();
    let slice_indices = [vol.dims[2] / 2, vol.dims[1] / 2, vol.dims[0] / 2];
    let is_first = scene.nifti_files.is_empty()
        && scene.trx_files.is_empty()
        && scene.gifti_surfaces.is_empty();
    if is_first {
        scene.volume_center = vol.voxel_to_world(Vec3::new(
            vol.dims[0] as f32 / 2.0,
            vol.dims[1] as f32 / 2.0,
            vol.dims[2] as f32 / 2.0,
        ));
        scene.volume_extent = (vol.voxel_to_world(Vec3::new(
            vol.dims[0] as f32,
            vol.dims[1] as f32,
            vol.dims[2] as f32,
        )) - vol.voxel_to_world(Vec3::ZERO))
        .length();
    }
    let id = allocate_file_id(scene, explicit_id);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "volume.nii".to_string());
    scene.nifti_files.push(LoadedNifti {
        id,
        name,
        volume: vol,
        colormap: VolumeColormap::Grayscale,
        opacity: 1.0,
        z_order: scene.nifti_files.len() as i32,
        window_center: 0.5,
        window_width: 1.0,
        visible: true,
    });
    if first_nifti {
        scene.slice_indices = slice_indices;
        scene.slice_world_offsets = [
            scene.volume_center.z,
            scene.volume_center.y,
            scene.volume_center.x,
        ];
    }
    id
}

fn apply_loaded_surface(
    scene: &mut HeadlessScene,
    path: PathBuf,
    surface: GiftiSurfaceData,
    explicit_id: Option<FileId>,
) -> FileId {
    let first_scene_asset = scene.trx_files.is_empty()
        && scene.nifti_files.is_empty()
        && scene.gifti_surfaces.is_empty()
        && scene.parcellations.is_empty();
    let id = allocate_file_id(scene, explicit_id);
    let initial_surface_view = first_scene_asset.then(|| {
        let center = (surface.bbox_min + surface.bbox_max) * 0.5;
        let extent = (surface.bbox_max - surface.bbox_min).length().max(1.0);
        (center, extent)
    });
    let surface = Arc::new(surface);
    let color = [0.72, 0.72, 0.72];
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "surface.gii".to_string());
    scene.gifti_surfaces.push(LoadedGiftiSurface {
        id,
        name,
        path,
        data: surface,
        visible: true,
        opacity: 1.0,
        color,
        outline_color: color,
        outline_thickness: 1.25,
        show_projection_map: false,
        map_opacity: 1.0,
        map_threshold: 0.0,
        surface_gloss: 0.45,
        projection_colormap: crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
        auto_range: true,
        range_min: 0.0,
        range_max: 1.0,
    });
    if let Some((center, extent)) = initial_surface_view {
        scene.volume_center = center;
        scene.volume_extent = extent;
        scene.slice_world_offsets = [center.z, center.y, center.x];
    }
    id
}

fn apply_loaded_parcellation(
    scene: &mut HeadlessScene,
    path: PathBuf,
    source: LoadedParcellationSource,
    explicit_id: Option<FileId>,
) -> FileId {
    let id = allocate_file_id(scene, explicit_id);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "parcellation.nii".to_string());
    scene.parcellations.push(LoadedParcellation {
        asset: ParcellationAsset {
            id,
            name,
            path,
            data: Arc::new(source.data),
            label_table_path: source.label_table_path,
            visible: true,
        },
    });
    id
}

fn refresh_workflow_runtime(scene: &HeadlessScene, workflow: &mut HeadlessWorkflowState) {
    ensure_node_uuids(&mut workflow.document);
    workflow.runtime = evaluate_scene_plan(
        &workflow.document,
        &scene.trx_files,
        &scene.nifti_files,
        &scene.cifti_files,
        &scene.gifti_surfaces,
        &scene.parcellations,
        &scene.odx_files,
        &mut workflow.display_runtimes,
        &mut workflow.next_draw_id,
        &mut workflow.execution_cache,
    );
}

fn odx_odf_exceeds_binding_limit(odx: &OdxScene, device: &wgpu::Device) -> bool {
    odx.compact_voxel_count()
        .saturating_mul(odx.glyph_row_width())
        .saturating_mul(std::mem::size_of::<f32>())
        > device.limits().max_storage_buffer_binding_size as usize
}

fn execute_workflow_to_completion(
    scene: &HeadlessScene,
    workflow: &mut HeadlessWorkflowState,
) -> anyhow::Result<()> {
    loop {
        refresh_workflow_runtime(scene, workflow);
        if let Some(error) = &workflow.runtime.graph_error {
            bail!("{error}");
        }

        let mut ran_job = false;

        for plan in workflow
            .runtime
            .scene_plan
            .reactive_streamline_plans
            .clone()
        {
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::ReactiveStreamline(plan))
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.surface_query_plans.clone() {
            let fingerprint =
                workflow_surface_query_fingerprint(&plan.flow, plan.surface_id, plan.depth_mm);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::SurfaceQuery(plan))
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.surface_map_plans.clone() {
            let fingerprint = workflow_surface_projection_fingerprint(
                &plan.flow,
                plan.surface_id,
                plan.depth_mm,
                plan.dps_field.as_deref(),
            );
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::SurfaceMap(plan))
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for draw in workflow.runtime.scene_plan.streamline_draws.clone() {
            if draw.render_style != RenderStyle::Tubes {
                continue;
            }
            let fingerprint = workflow_streamline_fingerprint(&draw);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(draw.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                draw.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::TubeGeometry(draw))
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.boundary_field_plans.clone() {
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.build_node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            apply_job_result(
                &mut workflow.execution_cache,
                plan.build_node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::BoundaryField { plan })
                    .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        for plan in workflow.runtime.scene_plan.bundle_surface_plans.clone() {
            let fingerprint = workflow_bundle_plan_fingerprint(&plan);
            let record = workflow
                .execution_cache
                .node_runs
                .entry(plan.build_node_uuid)
                .or_default();
            if record.last_success_fingerprint != Some(fingerprint) {
                mark_expensive_success(
                    record,
                    fingerprint,
                    format!(
                        "Bundle surface build for {} streamline(s)",
                        plan.flow.selected_streamlines.len()
                    ),
                );
            }
        }

        for draw in workflow.runtime.scene_plan.bundle_draws.clone() {
            let boundary_field = draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.field.clone())
            });
            if draw.boundary_field_node_uuid.is_some() && boundary_field.is_none() {
                continue;
            }
            let fingerprint = workflow_bundle_display_fingerprint(
                &draw,
                draw.boundary_field_node_uuid.and_then(|uuid| {
                    workflow
                        .execution_cache
                        .boundary_field_cache
                        .get(&uuid)
                        .map(|cache| cache.fingerprint)
                }),
            );
            let record = workflow
                .execution_cache
                .node_runs
                .entry(draw.node_uuid)
                .or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            let plan = BundleSurfacePlan {
                build_node_uuid: draw.build_node_uuid,
                label: draw.label.clone(),
                flow: draw.flow.clone(),
                per_group: draw.per_group,
                build_mode: draw.build_mode,
                voxel_size_mm: draw.voxel_size_mm,
                threshold: draw.threshold,
                smooth_sigma: draw.smooth_sigma,
                min_component_volume_mm3: draw.min_component_volume_mm3,
                tube_radius_mm: draw.tube_radius_mm,
                tube_sides: draw.tube_sides,
                opacity: draw.opacity,
            };
            apply_job_result(
                &mut workflow.execution_cache,
                draw.node_uuid,
                fingerprint,
                run_workflow_job(WorkflowJobPayload::BundleSurface {
                    plan,
                    color_mode: draw.color_mode,
                    boundary_field,
                })
                .map_err(|err| anyhow!(err))?,
            );
            ran_job = true;
        }

        if !ran_job {
            break;
        }
    }

    refresh_workflow_runtime(scene, workflow);
    if let Some(error) = &workflow.runtime.graph_error {
        bail!("{error}");
    }
    for plan in workflow.runtime.save_streamline_targets.values() {
        save_streamline_plan(plan).map_err(|err| anyhow!(err))?;
    }
    Ok(())
}

fn apply_job_result(
    cache: &mut crate::workflow::WorkflowExecutionCache,
    node_uuid: WorkflowNodeUuid,
    fingerprint: u64,
    output: WorkflowJobOutput,
) {
    let record = cache.node_runs.entry(node_uuid).or_default();
    match output {
        WorkflowJobOutput::ReactiveStreamline(flow) => {
            cache
                .derived_streamline_cache
                .insert(node_uuid, CachedDerivedStreamline { flow });
            mark_expensive_success(record, fingerprint, "reactive streamlines".to_string());
        }
        WorkflowJobOutput::SurfaceQuery(flow) => {
            cache
                .surface_query_cache
                .insert(node_uuid, CachedSurfaceQuery { flow });
            mark_expensive_success(record, fingerprint, "surface query".to_string());
        }
        WorkflowJobOutput::SurfaceMap(map) => {
            cache
                .surface_streamline_map_cache
                .insert(node_uuid, CachedSurfaceStreamlineMap { map });
            mark_expensive_success(record, fingerprint, "surface map".to_string());
        }
        WorkflowJobOutput::TubeGeometry { vertices, indices } => {
            cache.tube_geometry_cache.insert(
                node_uuid,
                CachedTubeGeometry {
                    fingerprint,
                    vertices,
                    indices,
                },
            );
            mark_expensive_success(record, fingerprint, "tube geometry".to_string());
        }
        WorkflowJobOutput::BundleSurface { meshes } => {
            let summary = if meshes.is_empty() {
                "Bundle surface is empty".to_string()
            } else {
                format!("{} bundle surface mesh(es)", meshes.len())
            };
            cache.bundle_surface_mesh_cache.insert(
                node_uuid,
                CachedBundleSurfaceMeshes {
                    fingerprint,
                    meshes,
                },
            );
            mark_expensive_success(record, fingerprint, summary);
        }
        WorkflowJobOutput::BoundaryField { field } => {
            let summary = field
                .as_ref()
                .map(|_| "Boundary field".to_string())
                .unwrap_or_else(|| "No boundary field".to_string());
            if let Some(field) = field {
                cache
                    .boundary_field_cache
                    .insert(node_uuid, CachedBoundaryField { fingerprint, field });
            } else {
                cache.boundary_field_cache.remove(&node_uuid);
            }
            mark_expensive_success(record, fingerprint, summary);
        }
    }
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn create_gpu_context() -> anyhow::Result<GpuContext> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map_err(|_| anyhow!("no headless GPU backend available"))?;

    let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("trxviz-headless-device"),
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 8192,
            max_buffer_size: 1 << 30,
            ..base_limits
        },
        ..Default::default()
    };
    let (device, queue) = block_on(adapter.request_device(&descriptor))
        .map_err(|err| anyhow!("failed to create GPU device: {err}"))?;
    Ok(GpuContext { device, queue })
}

fn build_gpu_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
) -> anyhow::Result<GpuSceneResources> {
    let mut bounds_min = Vec3::splat(f32::INFINITY);
    let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);

    let mut expand = |point: Vec3| {
        bounds_min = bounds_min.min(point);
        bounds_max = bounds_max.max(point);
    };

    let background = BackgroundResources::new(device, TARGET_FORMAT);
    let mut slices = AllSliceResources {
        entries: Vec::new(),
    };
    for nifti in &scene.nifti_files {
        let slice_resources = SliceResources::new(device, queue, TARGET_FORMAT, &nifti.volume);
        slice_resources.update_slice(
            queue,
            SliceAxis::Axial,
            scene.slice_indices[0],
            &nifti.volume,
        );
        slice_resources.update_slice(
            queue,
            SliceAxis::Coronal,
            scene.slice_indices[1],
            &nifti.volume,
        );
        slice_resources.update_slice(
            queue,
            SliceAxis::Sagittal,
            scene.slice_indices[2],
            &nifti.volume,
        );
        slices.entries.push((nifti.id, slice_resources));

        for x in [0.0, nifti.volume.dims[0] as f32] {
            for y in [0.0, nifti.volume.dims[1] as f32] {
                for z in [0.0, nifti.volume.dims[2] as f32] {
                    expand(nifti.volume.voxel_to_world(Vec3::new(x, y, z)));
                }
            }
        }
    }

    let mut meshes = MeshResources::new(device, TARGET_FORMAT);
    for surface in &scene.gifti_surfaces {
        meshes.add_surface(surface.id, device, surface.data.as_ref());
        expand(surface.data.bbox_min);
        expand(surface.data.bbox_max);
    }
    for draw in &workflow.runtime.scene_plan.surface_draws {
        if !draw.vertex_rgba.is_empty() {
            meshes.update_surface_colors(queue, draw.source_id, &draw.vertex_rgba);
        }
        if let Some(scalars) = &draw.projection_scalars {
            meshes.update_surface_scalars(queue, draw.source_id, scalars);
        }
    }

    let mut streamlines = AllStreamlineResources {
        entries: Vec::new(),
    };
    for draw in &workflow.runtime.scene_plan.streamline_draws {
        let fingerprint = workflow_streamline_fingerprint(draw);
        let subset = crate::workflow::materialize_flow_gpu(draw.flow.clone());
        for position in &subset.positions {
            expand(Vec3::from(*position));
        }
        let mut resource = StreamlineResources::new(device, TARGET_FORMAT, &subset);
        if draw.render_style == RenderStyle::Tubes {
            let cache = workflow
                .execution_cache
                .tube_geometry_cache
                .get(&draw.node_uuid)
                .filter(|cache| cache.fingerprint == fingerprint)
                .ok_or_else(|| anyhow!("missing tube geometry for {}", draw.label))?;
            resource.update_tube_geometry(device, &cache.vertices, &cache.indices);
        }
        streamlines.entries.push((draw.draw_id, resource));
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        if let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        {
            meshes.set_bundle_meshes(draw.draw_id, device, &cache.meshes);
            for (mesh, _) in &cache.meshes {
                for vertex in &mesh.vertices {
                    expand(Vec3::from(vertex.position));
                }
            }
        }
    }

    let mut glyphs = GlyphResources::new(device, TARGET_FORMAT);
    if let Some(draw) = workflow
        .runtime
        .scene_plan
        .boundary_glyph_draws
        .iter()
        .find(|draw| draw.visible)
        .or_else(|| workflow.runtime.scene_plan.boundary_glyph_draws.first())
    {
        if let Some(cache) = workflow
            .execution_cache
            .boundary_field_cache
            .get(&draw.build_node_uuid)
        {
            glyphs.set_field(device, cache.field.clone(), draw.scale, draw.min_contacts);
            let origin = cache.field.grid.origin_ras;
            let size = Vec3::new(
                cache.field.grid.dims[0] as f32,
                cache.field.grid.dims[1] as f32,
                cache.field.grid.dims[2] as f32,
            ) * cache.field.grid.voxel_size_mm;
            expand(origin);
            expand(origin + size);
        }
    }

    let mut fixels = FixelResources::new(device, TARGET_FORMAT);

    // ODX ODF glyphs and fixels — driven by the evaluated workflow plan.
    let axial_slice = scene.slice_indices[2] as u32;
    let odf_plan = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|plan| plan.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first());
    if let Some(plan) = odf_plan {
        let odx = &plan.field.scene;
        match odx.glyph_source_kind() {
            Some(crate::data::odx_data::OdxGlyphSourceKind::Odf) => {
                let use_slice_local = odx_odf_exceeds_binding_limit(odx, device);
                let slice_index = scene.slice_indices[plan.slice_axis.viewport_index()] as u32;
                let instances = if use_slice_local {
                    odx.glyph_instances_for_slice(plan.slice_axis.odx_axis(), slice_index)
                } else {
                    odx.glyph_instances_full_volume()
                };
                if !instances.is_empty() {
                    if use_slice_local {
                        let amplitudes = odx
                            .odf_amplitudes_for_slice(plan.slice_axis.odx_axis(), slice_index)
                            .expect("ODF amplitudes should exist for slice-local ODF mode");
                        glyphs.set_odx_slice_odf(
                            device,
                            &odx.sphere_vertices,
                            &odx.sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    } else {
                        let amplitudes = odx
                            .odf_amplitudes_full_sphere()
                            .expect("ODF amplitudes should exist for ODF glyph mode");
                        glyphs.set_odx_odf_volume(
                            device,
                            &odx.sphere_vertices,
                            &odx.sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    }
                }
            }
            Some(crate::data::odx_data::OdxGlyphSourceKind::Sh) => {
                let slice_index = scene.slice_indices[plan.slice_axis.viewport_index()] as u32;
                let instances =
                    odx.glyph_instances_for_slice(plan.slice_axis.odx_axis(), slice_index);
                if !instances.is_empty() {
                    let coefficients = odx
                        .sh_coefficients_for_slice(plan.slice_axis.odx_axis(), slice_index)
                        .expect("SH coefficients should exist for SH glyph mode");
                    glyphs.set_odx_sh_volume(
                        device,
                        &odx.sphere_vertices,
                        &odx.sphere_indices,
                        &instances,
                        &coefficients,
                        odx.sh_view_f32()
                            .expect("SH coefficients should exist for SH glyph mode")
                            .ncols(),
                        odx.sh_transform_flat()
                            .expect("SH transform should exist for SH glyph mode"),
                        odx.sh_source_dir_count()
                            .expect("SH direction count should exist for SH glyph mode"),
                        odx.glyph_row_width(),
                        None,
                        None,
                    );
                    let slice_indices: Vec<u32> = (0..instances.len() as u32).collect();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("headless_odx_sh_initial_slice_encoder"),
                        });
                    glyphs.dispatch_odx_sh_slice(device, queue, &mut encoder, &slice_indices);
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
            None => {}
        }
        for &center in odx.centers_ras() {
            expand(Vec3::from(center));
        }
    } else if let Some(odx) = &scene.odx_scene {
        match odx.glyph_source_kind() {
            Some(crate::data::odx_data::OdxGlyphSourceKind::Odf) => {
                let use_slice_local = odx_odf_exceeds_binding_limit(odx, device);
                let instances = if use_slice_local {
                    odx.glyph_instances_for_slice(2, axial_slice)
                } else {
                    odx.glyph_instances_full_volume()
                };
                if !instances.is_empty() {
                    if use_slice_local {
                        let amplitudes = odx
                            .odf_amplitudes_for_slice(2, axial_slice)
                            .expect("ODF amplitudes should exist for slice-local ODF mode");
                        glyphs.set_odx_slice_odf(
                            device,
                            &odx.sphere_vertices,
                            &odx.sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    } else {
                        let amplitudes = odx
                            .odf_amplitudes_full_sphere()
                            .expect("ODF amplitudes should exist for ODF glyph mode");
                        glyphs.set_odx_odf_volume(
                            device,
                            &odx.sphere_vertices,
                            &odx.sphere_indices,
                            &instances,
                            &amplitudes,
                            None,
                            None,
                        );
                    }
                }
            }
            Some(crate::data::odx_data::OdxGlyphSourceKind::Sh) => {
                let instances = odx.glyph_instances_for_slice(2, axial_slice);
                if !instances.is_empty() {
                    let coefficients = odx
                        .sh_coefficients_for_slice(2, axial_slice)
                        .expect("SH coefficients should exist for SH glyph mode");
                    glyphs.set_odx_sh_volume(
                        device,
                        &odx.sphere_vertices,
                        &odx.sphere_indices,
                        &instances,
                        &coefficients,
                        odx.sh_view_f32()
                            .expect("SH coefficients should exist for SH glyph mode")
                            .ncols(),
                        odx.sh_transform_flat()
                            .expect("SH transform should exist for SH glyph mode"),
                        odx.sh_source_dir_count()
                            .expect("SH direction count should exist for SH glyph mode"),
                        odx.glyph_row_width(),
                        None,
                        None,
                    );
                    let slice_indices: Vec<u32> = (0..instances.len() as u32).collect();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("headless_odx_sh_fallback_slice_encoder"),
                        });
                    glyphs.dispatch_odx_sh_slice(device, queue, &mut encoder, &slice_indices);
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
            None => {}
        }
        for &center in odx.centers_ras() {
            expand(Vec3::from(center));
        }
    }

    let fixel_plan = workflow
        .runtime
        .scene_plan
        .fixel_3d_draws
        .iter()
        .find(|p| p.visible)
        .or_else(|| workflow.runtime.scene_plan.fixel_3d_draws.first())
        .or_else(|| {
            workflow
                .runtime
                .scene_plan
                .fixel_2d_draws
                .iter()
                .find(|p| p.visible)
        })
        .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first());
    if let Some(plan) = fixel_plan {
        let mut fixel_instances = plan.field.scene.fixels_for_slice(2, axial_slice);
        for inst in &mut fixel_instances {
            inst.length *= plan.length_scale;
        }
        if !fixel_instances.is_empty() {
            fixels.set_fixels(device, &fixel_instances);
        }
    } else if let Some(odx) = &scene.odx_scene {
        let fixel_instances = odx.fixels_for_slice(2, axial_slice);
        if !fixel_instances.is_empty() {
            fixels.set_fixels(device, &fixel_instances);
        }
    }

    if !bounds_min.is_finite() || !bounds_max.is_finite() {
        bounds_min = scene.volume_center - Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
        bounds_max = scene.volume_center + Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
    }

    Ok(GpuSceneResources {
        background,
        streamlines,
        slices,
        meshes,
        glyphs,
        fixels,
        bounds: SceneBounds {
            min: bounds_min,
            max: bounds_max,
        },
    })
}

fn build_render_data(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
    view: HeadlessView,
) -> HeadlessRenderData {
    let surface_draws = match view {
        HeadlessView::InflatedStage => stage_surface_draw_instances(scene, workflow),
        _ => workflow
            .runtime
            .scene_plan
            .surface_draws
            .iter()
            .map(|draw| {
                (
                    draw.source_id,
                    0,
                    MeshDrawStyle {
                        color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                        scalar_min: draw.range_min,
                        scalar_max: draw.range_max,
                        scalar_enabled: draw.show_projection_map,
                        vertex_color_enabled: !draw.vertex_rgba.is_empty(),
                        colormap: draw.projection_colormap,
                        gloss: draw.gloss,
                        map_opacity: draw.map_opacity,
                        map_threshold: draw.map_threshold,
                        model_matrix: draw.model_matrix,
                    },
                )
            })
            .collect(),
    };
    let volume_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
            .runtime
            .scene_plan
            .volume_draws
            .iter()
            .map(|draw| VolumeDrawInfo {
                file_id: draw.source_id,
                window_center: draw.window_center,
                window_width: draw.window_width,
                colormap: draw.colormap.as_u32(),
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>()
    };
    let streamline_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
            .runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| StreamlineDrawInfo {
                file_id: draw.draw_id,
                visible: draw.visible,
                render_style: draw.render_style,
                tube_radius: draw.tube_radius_mm,
            })
            .collect::<Vec<_>>()
    };
    let bundle_draws = if view == HeadlessView::InflatedStage {
        Vec::new()
    } else {
        workflow
            .runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| BundleDrawInfo {
                file_id: draw.draw_id,
                opacity: draw.opacity,
            })
            .collect::<Vec<_>>()
    };
    let glyph_draw = if view == HeadlessView::InflatedStage {
        None
    } else {
        workflow
            .runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible)
    };
    let odx_has_glyph_field = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|p| p.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
        .map(|plan| plan.field.scene.has_glyph_field())
        .or_else(|| scene.odx_scene.as_ref().map(|odx| odx.has_glyph_field()))
        .unwrap_or(false);
    let odx_glyphs_active = workflow
        .runtime
        .scene_plan
        .odf_glyph_draws
        .iter()
        .find(|p| p.visible)
        .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
        .map(|p| p.visible)
        .unwrap_or(scene.odx_scene.is_some());

    HeadlessRenderData {
        any_visible_streamlines: streamline_draws.iter().any(|draw| draw.visible),
        surface_draws,
        volume_draws,
        streamline_draws,
        bundle_draws,
        glyph_visible: glyph_draw.is_some()
            && !workflow.execution_cache.boundary_field_cache.is_empty(),
        glyph_color_mode: glyph_draw
            .map(|draw| draw.color_mode)
            .unwrap_or(BoundaryGlyphColorMode::DirectionRgb),
        glyph_density_3d_step: glyph_draw
            .map(|draw| draw.density_3d_step as u32)
            .unwrap_or(1),
        glyph_slice_density_step: glyph_draw
            .map(|draw| draw.slice_density_step as u32)
            .unwrap_or(1),
        odx_visible: scene.odx_scene.is_some()
            || !workflow.runtime.scene_plan.odf_glyph_draws.is_empty()
            || !workflow.runtime.scene_plan.fixel_3d_draws.is_empty()
            || !workflow.runtime.scene_plan.fixel_2d_draws.is_empty(),
        odx_fixel_3d_visible: workflow
            .runtime
            .scene_plan
            .fixel_3d_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.fixel_3d_draws.first())
            .map(|p| p.visible)
            .unwrap_or(true)
            && !(odx_has_glyph_field && odx_glyphs_active),
        odx_fixel_2d_visible: workflow
            .runtime
            .scene_plan
            .fixel_2d_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first())
            .map(|p| p.visible)
            .unwrap_or(scene.odx_scene.is_some()),
        fixel_line_width: workflow
            .runtime
            .scene_plan
            .fixel_3d_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.fixel_3d_draws.first())
            .map(|p| p.line_width)
            .or_else(|| {
                workflow
                    .runtime
                    .scene_plan
                    .fixel_2d_draws
                    .iter()
                    .find(|p| p.visible)
                    .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first())
                    .map(|p| p.line_width)
            })
            .unwrap_or(0.006),
        fixel_2d_slab_half_width_mm: workflow
            .runtime
            .scene_plan
            .fixel_2d_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first())
            .map(|p| (p.slab_thickness_mm * 0.5).max(0.0))
            .unwrap_or(1.0),
        fixel_opacity: workflow
            .runtime
            .scene_plan
            .fixel_3d_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.fixel_3d_draws.first())
            .map(|p| p.opacity)
            .or_else(|| {
                workflow
                    .runtime
                    .scene_plan
                    .fixel_2d_draws
                    .iter()
                    .find(|p| p.visible)
                    .or_else(|| workflow.runtime.scene_plan.fixel_2d_draws.first())
                    .map(|p| p.opacity)
            })
            .unwrap_or(1.0),
        odf_glyph_opacity: workflow
            .runtime
            .scene_plan
            .odf_glyph_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
            .map(|p| p.opacity)
            .unwrap_or(1.0),
        odf_glyph_gloss: workflow
            .runtime
            .scene_plan
            .odf_glyph_draws
            .iter()
            .find(|p| p.visible)
            .or_else(|| workflow.runtime.scene_plan.odf_glyph_draws.first())
            .map(|p| p.gloss)
            .unwrap_or(0.0),
    }
}

fn stage_surface_draw_instances(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
) -> Vec<(usize, usize, MeshDrawStyle)> {
    let mut draws = Vec::new();
    for draw in &workflow.runtime.scene_plan.stage_surface_draws {
        let Some(surface) = scene
            .gifti_surfaces
            .iter()
            .find(|surface| surface.id == draw.source_id)
        else {
            continue;
        };
        for (uniform_slot, model_matrix) in stage_instance_model_matrices(
            draw.structure,
            surface.data.bbox_min,
            surface.data.bbox_max,
        )
        .into_iter()
        .enumerate()
        {
            draws.push((
                draw.source_id,
                uniform_slot,
                MeshDrawStyle {
                    color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                    scalar_min: draw.range_min,
                    scalar_max: draw.range_max,
                    scalar_enabled: draw.show_projection_map,
                    vertex_color_enabled: !draw.vertex_rgba.is_empty(),
                    colormap: draw.projection_colormap,
                    gloss: draw.gloss,
                    map_opacity: draw.map_opacity,
                    map_threshold: draw.map_threshold,
                    model_matrix: model_matrix.to_cols_array_2d(),
                },
            ));
        }
    }
    draws
}

fn compute_render_bounds(scene: &HeadlessScene, render_data: &HeadlessRenderData) -> SceneBounds {
    let mut bounds_min = Vec3::splat(f32::INFINITY);
    let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;

    for (surface_id, _, style) in &render_data.surface_draws {
        let Some(surface) = scene
            .gifti_surfaces
            .iter()
            .find(|surface| surface.id == *surface_id)
        else {
            continue;
        };
        let model = Mat4::from_cols_array_2d(&style.model_matrix);
        for corner in bbox_corners(surface.data.bbox_min, surface.data.bbox_max) {
            let point = model.transform_point3(corner);
            bounds_min = bounds_min.min(point);
            bounds_max = bounds_max.max(point);
            any = true;
        }
    }

    if !any {
        return SceneBounds {
            min: scene.volume_center - Vec3::splat(scene.volume_extent.max(1.0) * 0.5),
            max: scene.volume_center + Vec3::splat(scene.volume_extent.max(1.0) * 0.5),
        };
    }

    SceneBounds {
        min: bounds_min,
        max: bounds_max,
    }
}

fn build_camera(
    bounds: &SceneBounds,
    saved_camera: Option<WorkflowCamera3D>,
    options: &HeadlessRenderOptions,
    aspect: f32,
) -> OrbitCamera {
    let saved_target = saved_camera.map(|camera| Vec3::from_array(camera.target));
    let center = options
        .target
        .or(saved_target)
        .unwrap_or((bounds.min + bounds.max) * 0.5);
    let radius = ((bounds.max - bounds.min) * 0.5).length().max(1.0);
    let mut camera = OrbitCamera::new(center, fit_distance(radius, aspect));
    camera.yaw = options
        .azimuth_deg
        .or(saved_camera.map(|camera| camera.azimuth_deg))
        .unwrap_or(45.0)
        .to_radians();
    camera.pitch = options
        .elevation_deg
        .or(saved_camera.map(|camera| camera.elevation_deg))
        .unwrap_or(25.0)
        .to_radians();
    camera.distance = options
        .distance
        .or(saved_camera.map(|camera| camera.distance))
        .unwrap_or(camera.distance)
        .max(0.1);
    camera
}

fn fit_distance(radius: f32, aspect: f32) -> f32 {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let half_y = fov_y * 0.5;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let limiting_half_angle = half_y.min(half_x).max(0.1);
    (radius / limiting_half_angle.sin()) * 1.1
}

fn bbox_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}

fn stage_instance_model_matrices(
    structure: Option<crate::data::cifti::CiftiStructure>,
    bbox_min: Vec3,
    bbox_max: Vec3,
) -> Vec<Mat4> {
    let center = (bbox_min + bbox_max) * 0.5;
    let extents = bbox_max - bbox_min;
    let span = extents
        .x
        .abs()
        .max(extents.y.abs())
        .max(extents.z.abs())
        .max(1.0);
    let separation = span * 0.55;
    let lateral_row_z = span * 0.42;
    let medial_row_z = -span * 0.42;
    let center_transform = Mat4::from_translation(-center);

    match structure {
        Some(crate::data::cifti::CiftiStructure::CortexLeft) => vec![
            stage_panel_transform(center_transform, separation, lateral_row_z, 90.0),
            stage_panel_transform(center_transform, separation, medial_row_z, -90.0),
        ],
        Some(crate::data::cifti::CiftiStructure::CortexRight) => vec![
            stage_panel_transform(center_transform, -separation, lateral_row_z, -90.0),
            stage_panel_transform(center_transform, -separation, medial_row_z, 90.0),
        ],
        _ => vec![Mat4::IDENTITY],
    }
}

fn stage_panel_transform(
    center_transform: Mat4,
    x_shift: f32,
    z_shift: f32,
    turn_deg: f32,
) -> Mat4 {
    Mat4::from_translation(Vec3::new(x_shift, 0.0, z_shift))
        * Mat4::from_rotation_z(turn_deg.to_radians())
        * center_transform
}

fn render_scene3d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
    render_3d: &WorkflowRender3D,
    slice_visible: [bool; 3],
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let aspect = width as f32 / height.max(1) as f32;
    let view_proj = camera.view_projection(aspect);
    let camera_pos = camera.eye();
    let camera_dir = camera.view_direction();
    let lighting = render_3d.scene_lighting();
    let bounds_radius = ((resources.bounds.max - resources.bounds.min) * 0.5)
        .length()
        .max(1.0);
    let fog_span = (camera.distance + bounds_radius).max(1.0);
    let fog_near = fog_span * render_3d.fog_start_fraction;
    let fog_far = fog_span * render_3d.fog_end_fraction;
    resources.background.update(
        queue,
        &render_3d.background,
        render_3d.exposure,
        render_3d.contrast,
        render_3d.vignette_strength,
    );

    for volume in &render_data.volume_draws {
        if let Some((_, slice)) = resources
            .slices
            .entries
            .iter()
            .find(|(id, _)| *id == volume.file_id)
        {
            slice.update_uniforms(
                queue,
                0,
                view_proj,
                volume.window_center,
                volume.window_width,
                volume.colormap,
                volume.opacity,
            );
        }
    }
    for streamline in &render_data.streamline_draws {
        if !streamline.visible {
            continue;
        }
        if let Some((_, resource)) = resources
            .streamlines
            .entries
            .iter()
            .find(|(id, _)| *id == streamline.file_id)
        {
            let aux = if streamline.render_style == RenderStyle::DepthCue {
                300.0
            } else {
                streamline.tube_radius
            };
            resource.update_uniforms(
                queue,
                0,
                view_proj,
                camera_pos,
                streamline.render_style as u32,
                glam::Vec3::Z,    // slab_normal (irrelevant — slab disabled)
                glam::Vec3::ZERO, // slab_center
                0.0,              // slab_half_width = 0 → disabled
                aux,
                lighting,
                render_3d,
                fog_near,
                fog_far,
            );
        }
    }
    for (surface_id, uniform_slot, style) in &render_data.surface_draws {
        resources.meshes.update_surface_uniforms(
            queue,
            *surface_id,
            *uniform_slot,
            view_proj,
            style,
            camera_pos,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    for bundle in &render_data.bundle_draws {
        resources.meshes.update_bundle_uniforms(
            bundle.file_id,
            queue,
            view_proj,
            camera_pos,
            bundle.opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.glyph_visible || render_data.odx_visible {
        resources.glyphs.update_uniforms(
            queue,
            0,
            view_proj,
            camera_pos,
            glam::Vec3::Z,    // slab_normal (irrelevant — slab disabled)
            glam::Vec3::ZERO, // slab_center
            0.0,              // slab_half_width = 0 → disabled
            render_data.glyph_color_mode,
            render_data.glyph_density_3d_step,
            render_data.odf_glyph_opacity,
            render_data.odf_glyph_gloss,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }
    if render_data.odx_visible && render_data.odx_fixel_3d_visible {
        resources.fixels.update_uniforms(
            queue,
            0,
            view_proj,
            camera_pos,
            glam::Vec3::Z,    // slab_normal (irrelevant — slab disabled)
            glam::Vec3::ZERO, // slab_center
            0.0,              // slab_half_width = 0 → disabled
            1,
            render_data.fixel_line_width,
            render_data.fixel_opacity,
            lighting,
            render_3d,
            fog_near,
            fog_far,
        );
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trxviz_headless_encoder"),
    });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: render_3d.background.bottom_color()[0] as f64,
                        g: render_3d.background.bottom_color()[1] as f64,
                        b: render_3d.background.bottom_color()[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let render_pass: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut render_pass) };

        render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        resources.background.paint(render_pass);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources
                .slices
                .entries
                .iter()
                .find(|(id, _)| *id == volume.file_id)
            {
                render_pass.set_pipeline(&slice.pipeline);
                render_pass.set_bind_group(0, &slice.bind_groups[0], &[]);
                render_pass
                    .set_index_buffer(slice.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                for i in 0..3 {
                    if !slice_visible[i] {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, slice.quad_buffers[i].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }

        if render_data.any_visible_streamlines {
            for streamline in &render_data.streamline_draws {
                if !streamline.visible {
                    continue;
                }
                if let Some((_, resource)) = resources
                    .streamlines
                    .entries
                    .iter()
                    .find(|(id, _)| *id == streamline.file_id)
                {
                    render_pass.set_bind_group(0, &resource.bind_groups[0], &[]);
                    if streamline.render_style == RenderStyle::Tubes {
                        if let (Some(vertices), Some(indices)) =
                            (&resource.tube_vertex_buffer, &resource.tube_index_buffer)
                        {
                            render_pass.set_pipeline(&resource.tube_pipeline);
                            render_pass.set_vertex_buffer(0, vertices.slice(..));
                            render_pass
                                .set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..resource.num_tube_indices, 0, 0..1);
                        }
                    } else {
                        render_pass.set_pipeline(&resource.pipeline);
                        render_pass.set_vertex_buffer(0, resource.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, resource.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, resource.tangent_buffer.slice(..));
                        render_pass.set_index_buffer(
                            resource.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..resource.num_indices, 0, 0..1);
                    }
                }
            }
        }

        if !render_data.surface_draws.is_empty() {
            resources
                .meshes
                .paint_opaque(render_pass, &render_data.surface_draws);
        }
        if !render_data.bundle_draws.is_empty() {
            let bundle_draws = render_data
                .bundle_draws
                .iter()
                .map(|draw| (draw.file_id, draw.opacity))
                .collect::<Vec<_>>();
            resources
                .meshes
                .paint_bundle_opaque(render_pass, &bundle_draws);
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &bundle_draws,
                camera_pos,
                camera_dir,
            );
        } else if !render_data.surface_draws.is_empty() {
            resources.meshes.paint_transparent(
                render_pass,
                &render_data.surface_draws,
                &[],
                camera_pos,
                camera_dir,
            );
        }
        if render_data.glyph_visible || render_data.odx_visible {
            resources.glyphs.paint(render_pass, 0, false);
        }
        if render_data.odx_visible && render_data.odx_fixel_3d_visible {
            resources.fixels.paint(render_pass, 0, false);
        }
    }

    readback_texture_to_png(device, queue, encoder, &texture, width, height, output_path)
}

fn render_scene2d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    slice_view_ui: Option<WorkflowSliceViewUi>,
    scene: &HeadlessScene,
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let slice_view_ui = slice_view_ui.ok_or_else(|| {
        anyhow!(
            "2D project rendering requires saved slice_view_ui state; open the project in TRXViz and save it first"
        )
    })?;

    let panels = build_2d_panels(&slice_view_ui, scene, width, height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_2d_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trxviz_headless_2d_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let lighting = SceneLightingParams::default();
    let neutral_render = WorkflowRender3D {
        vignette_strength: 0.0,
        exposure: 1.0,
        contrast: 1.0,
        ..Default::default()
    };

    for panel in &panels {
        let bind_group_index = panel.axis_index + 1;
        let aspect = panel.rect.width as f32 / panel.rect.height.max(1) as f32;
        let camera = &slice_view_ui.slice_cameras[panel.axis_index];
        let view_proj =
            build_slice_camera(panel.axis_index, camera).view_projection(aspect, panel.slice_pos);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources
                .slices
                .entries
                .iter()
                .find(|(id, _)| *id == volume.file_id)
            {
                slice.update_uniforms(
                    queue,
                    bind_group_index,
                    view_proj,
                    volume.window_center,
                    volume.window_width,
                    volume.colormap,
                    volume.opacity,
                );
            }
        }
        let (slab_normal, slab_center) = slice_plane_for_panel(scene, panel);
        for streamline in &render_data.streamline_draws {
            if !streamline.visible {
                continue;
            }
            if let Some((_, resource)) = resources
                .streamlines
                .entries
                .iter()
                .find(|(id, _)| *id == streamline.file_id)
            {
                let hw = streamline.tube_radius.max(0.5);
                resource.update_uniforms(
                    queue,
                    bind_group_index,
                    view_proj,
                    glam::Vec3::ZERO,
                    0,
                    slab_normal,
                    slab_center,
                    hw,
                    0.5,
                    lighting,
                    &neutral_render,
                    0.0,
                    1.0,
                );
            }
        }
        if render_data.glyph_visible || render_data.odx_visible {
            resources.glyphs.update_uniforms(
                queue,
                bind_group_index,
                view_proj,
                glam::Vec3::ZERO,
                slab_normal,
                slab_center,
                1.0, // 1 mm slab half-width (placeholder; headless doesn't track voxel size)
                render_data.glyph_color_mode,
                render_data.glyph_slice_density_step,
                render_data.odf_glyph_opacity,
                render_data.odf_glyph_gloss,
                lighting,
                &neutral_render,
                0.0,
                1.0,
            );
        }
        if render_data.odx_visible && render_data.odx_fixel_2d_visible {
            resources.fixels.update_uniforms(
                queue,
                bind_group_index,
                view_proj,
                glam::Vec3::ZERO,
                slab_normal,
                slab_center,
                render_data.fixel_2d_slab_half_width_mm,
                1,
                render_data.fixel_line_width,
                render_data.fixel_opacity,
                lighting,
                &neutral_render,
                0.0,
                1.0,
            );
        }
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trxviz_headless_2d_encoder"),
    });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_2d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let render_pass: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut render_pass) };

        for panel in &panels {
            let bind_group_index = panel.axis_index + 1;
            render_pass.set_viewport(
                panel.rect.x as f32,
                panel.rect.y as f32,
                panel.rect.width as f32,
                panel.rect.height as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(
                panel.rect.x,
                panel.rect.y,
                panel.rect.width,
                panel.rect.height,
            );

            for volume in &render_data.volume_draws {
                if let Some((_, slice)) = resources
                    .slices
                    .entries
                    .iter()
                    .find(|(id, _)| *id == volume.file_id)
                {
                    render_pass.set_pipeline(&slice.pipeline);
                    render_pass.set_bind_group(0, &slice.bind_groups[bind_group_index], &[]);
                    render_pass.set_index_buffer(
                        slice.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass
                        .set_vertex_buffer(0, slice.quad_buffers[panel.axis_index].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }

            if render_data.any_visible_streamlines {
                for streamline in &render_data.streamline_draws {
                    if !streamline.visible {
                        continue;
                    }
                    if let Some((_, resource)) = resources
                        .streamlines
                        .entries
                        .iter()
                        .find(|(id, _)| *id == streamline.file_id)
                    {
                        render_pass.set_pipeline(&resource.slice_pipeline);
                        render_pass.set_bind_group(0, &resource.bind_groups[bind_group_index], &[]);
                        render_pass.set_vertex_buffer(0, resource.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, resource.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, resource.tangent_buffer.slice(..));
                        render_pass.set_index_buffer(
                            resource.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..resource.num_indices, 0, 0..1);
                    }
                }
            }
            if render_data.glyph_visible || render_data.odx_visible {
                resources.glyphs.paint(render_pass, bind_group_index, true);
            }
            if render_data.odx_visible && render_data.odx_fixel_2d_visible {
                resources.fixels.paint(render_pass, bind_group_index, true);
            }
        }
    }

    readback_texture_to_png(device, queue, encoder, &texture, width, height, output_path)
}

fn build_2d_panels(
    slice_view_ui: &WorkflowSliceViewUi,
    scene: &HeadlessScene,
    width: u32,
    height: u32,
) -> Vec<SlicePanel> {
    const SPACING: u32 = 8;
    match slice_view_ui.mode {
        WorkflowView2DMode::Slice => {
            let axis_index = axis_index_for_kind(slice_view_ui.single_view);
            let slice_index = scene.slice_indices[axis_index];
            vec![SlicePanel {
                rect: ViewportRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                axis_index,
                slice_index,
                slice_pos: slice_world_position(scene, axis_index),
            }]
        }
        WorkflowView2DMode::Ortho => {
            if slice_view_ui.ortho_show_row {
                let panel_width = ((width.saturating_sub(2 * SPACING)) / 3).max(1);
                (0..3)
                    .map(|axis_index| SlicePanel {
                        rect: ViewportRect {
                            x: axis_index as u32 * (panel_width + SPACING),
                            y: 0,
                            width: panel_width,
                            height: height.max(1),
                        },
                        axis_index,
                        slice_index: scene.slice_indices[axis_index],
                        slice_pos: slice_world_position(scene, axis_index),
                    })
                    .collect()
            } else {
                let panel_width = ((width.saturating_sub(SPACING)) / 2).max(1);
                let panel_height = ((height.saturating_sub(SPACING)) / 2).max(1);
                vec![
                    SlicePanel {
                        rect: ViewportRect {
                            x: 0,
                            y: 0,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 0,
                        slice_index: scene.slice_indices[0],
                        slice_pos: slice_world_position(scene, 0),
                    },
                    SlicePanel {
                        rect: ViewportRect {
                            x: panel_width + SPACING,
                            y: 0,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 1,
                        slice_index: scene.slice_indices[1],
                        slice_pos: slice_world_position(scene, 1),
                    },
                    SlicePanel {
                        rect: ViewportRect {
                            x: panel_width + SPACING,
                            y: panel_height + SPACING,
                            width: panel_width,
                            height: panel_height,
                        },
                        axis_index: 2,
                        slice_index: scene.slice_indices[2],
                        slice_pos: slice_world_position(scene, 2),
                    },
                ]
            }
        }
        WorkflowView2DMode::Lightbox => {
            let axis_index = axis_index_for_kind(slice_view_ui.lightbox_axis);
            let rows = slice_view_ui.lightbox_rows.max(1);
            let cols = slice_view_ui.lightbox_cols.max(1);
            let tile_width = ((width.saturating_sub(SPACING * cols.saturating_sub(1) as u32))
                / cols as u32)
                .max(1);
            let tile_height = ((height.saturating_sub(SPACING * rows.saturating_sub(1) as u32))
                / rows as u32)
                .max(1);
            let total = rows * cols;
            let center_tile = total / 2;
            let center_index = scene.slice_indices[axis_index];
            let max_index = max_slice_index(scene, axis_index);
            let mut panels = Vec::with_capacity(total);
            for row in 0..rows {
                for col in 0..cols {
                    let tile = row * cols + col;
                    let delta = tile as isize - center_tile as isize;
                    let index = center_index.saturating_add_signed(delta).min(max_index);
                    panels.push(SlicePanel {
                        rect: ViewportRect {
                            x: col as u32 * (tile_width + SPACING),
                            y: row as u32 * (tile_height + SPACING),
                            width: tile_width,
                            height: tile_height,
                        },
                        axis_index,
                        slice_index: index,
                        slice_pos: slice_world_position_for_index(scene, axis_index, index),
                    });
                }
            }
            panels
        }
    }
}

fn axis_index_for_kind(kind: WorkflowSliceViewKind) -> usize {
    match kind {
        WorkflowSliceViewKind::Axial => 0,
        WorkflowSliceViewKind::Coronal => 1,
        WorkflowSliceViewKind::Sagittal => 2,
    }
}

/// Return the slab plane (normal, center) for a slice panel.
/// Falls back to world-axis normals when no NIfTI is loaded.
fn slice_plane_for_panel(scene: &HeadlessScene, panel: &SlicePanel) -> (glam::Vec3, glam::Vec3) {
    if let Some(nf) = scene.nifti_files.first() {
        nf.volume.slice_plane(panel.axis_index, panel.slice_index)
    } else {
        let normal = match panel.axis_index {
            0 => glam::Vec3::Z,
            1 => glam::Vec3::Y,
            _ => glam::Vec3::X,
        };
        let center = match panel.axis_index {
            0 => glam::Vec3::new(0.0, 0.0, panel.slice_pos),
            1 => glam::Vec3::new(0.0, panel.slice_pos, 0.0),
            _ => glam::Vec3::new(panel.slice_pos, 0.0, 0.0),
        };
        (normal, center)
    }
}

fn build_slice_camera(
    axis_index: usize,
    camera: &crate::workflow::WorkflowOrthoSliceCamera,
) -> crate::renderer::camera::OrthoSliceCamera {
    crate::renderer::camera::OrthoSliceCamera {
        axis: match axis_index {
            0 => SliceAxis::Axial,
            1 => SliceAxis::Coronal,
            _ => SliceAxis::Sagittal,
        },
        center: camera.center,
        half_extent: camera.half_extent,
        rotation: camera.rotation,
    }
}

fn slice_world_position(scene: &HeadlessScene, axis_index: usize) -> f32 {
    slice_world_position_for_index(scene, axis_index, scene.slice_indices[axis_index])
}

fn slice_world_position_for_index(scene: &HeadlessScene, axis_index: usize, index: usize) -> f32 {
    if let Some(nf) = scene.nifti_files.first() {
        let idx = index as f32;
        let world = match axis_index {
            0 => nf.volume.voxel_to_world(Vec3::new(0.0, 0.0, idx)),
            1 => nf.volume.voxel_to_world(Vec3::new(0.0, idx, 0.0)),
            _ => nf.volume.voxel_to_world(Vec3::new(idx, 0.0, 0.0)),
        };
        match axis_index {
            0 => world.z,
            1 => world.y,
            _ => world.x,
        }
    } else {
        scene.slice_world_offsets[axis_index]
    }
}

fn max_slice_index(scene: &HeadlessScene, axis_index: usize) -> usize {
    scene
        .nifti_files
        .first()
        .map(|nf| match axis_index {
            0 => nf.volume.dims[2].saturating_sub(1),
            1 => nf.volume.dims[1].saturating_sub(1),
            _ => nf.volume.dims[0].saturating_sub(1),
        })
        .unwrap_or(scene.slice_indices[axis_index].saturating_add(128))
}

fn readback_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    output_path: &Path,
) -> anyhow::Result<()> {
    let padded_bytes_per_row = ((width * 4 + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
        / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trxviz_headless_readback"),
        size: padded_bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| anyhow!("failed to receive GPU readback status"))?
        .map_err(|err| anyhow!("failed to map render output: {err}"))?;

    let mapped = buffer_slice.get_mapped_range();
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * width as usize * 4;
        rgba[dst_offset..dst_offset + width as usize * 4]
            .copy_from_slice(&mapped[src_offset..src_offset + width as usize * 4]);
    }
    drop(mapped);
    output_buffer.unmap();

    image::save_buffer(output_path, &rgba, width, height, ColorType::Rgba8)
        .with_context(|| format!("saving PNG to {}", output_path.display()))?;
    Ok(())
}

fn ensure_export_tube_geometry(workflow: &mut HeadlessWorkflowState) -> anyhow::Result<()> {
    for draw in workflow.runtime.scene_plan.streamline_draws.clone() {
        if !draw.visible {
            continue;
        }
        let fingerprint = workflow_streamline_fingerprint(&draw);
        let record = workflow
            .execution_cache
            .node_runs
            .entry(draw.node_uuid)
            .or_default();
        if record.last_success_fingerprint == Some(fingerprint)
            && workflow
                .execution_cache
                .tube_geometry_cache
                .get(&draw.node_uuid)
                .is_some_and(|cache| cache.fingerprint == fingerprint)
        {
            continue;
        }
        apply_job_result(
            &mut workflow.execution_cache,
            draw.node_uuid,
            fingerprint,
            run_workflow_job(WorkflowJobPayload::TubeGeometry(draw)).map_err(|err| anyhow!(err))?,
        );
    }
    Ok(())
}

fn compute_scene_bounds(scene: &HeadlessScene, workflow: &HeadlessWorkflowState) -> SceneBounds {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    let mut expand = |point: Vec3| {
        min = min.min(point);
        max = max.max(point);
    };

    for nifti in &scene.nifti_files {
        for x in [0.0, nifti.volume.dims[0] as f32] {
            for y in [0.0, nifti.volume.dims[1] as f32] {
                for z in [0.0, nifti.volume.dims[2] as f32] {
                    expand(nifti.volume.voxel_to_world(Vec3::new(x, y, z)));
                }
            }
        }
    }

    for surface in &scene.gifti_surfaces {
        expand(surface.data.bbox_min);
        expand(surface.data.bbox_max);
    }

    for draw in &workflow.runtime.scene_plan.streamline_draws {
        if !draw.visible {
            continue;
        }
        let subset = crate::workflow::materialize_flow_gpu(draw.flow.clone());
        for position in &subset.positions {
            expand(Vec3::from(*position));
        }
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        if let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        {
            for (mesh, _) in &cache.meshes {
                for vertex in &mesh.vertices {
                    expand(Vec3::from(vertex.position));
                }
            }
        }
    }

    if min.is_finite() && max.is_finite() {
        SceneBounds { min, max }
    } else {
        let half = Vec3::splat((scene.volume_extent * 0.5).max(1.0));
        SceneBounds {
            min: scene.volume_center - half,
            max: scene.volume_center + half,
        }
    }
}

fn build_glb_scene(
    scene: &HeadlessScene,
    workflow: &HeadlessWorkflowState,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
    render_3d: &WorkflowRender3D,
    options: &HeadlessSceneExportOptions,
) -> anyhow::Result<Vec<u8>> {
    let mut builder = GlbBuilder::new();
    let scene_bounds = if options.view == HeadlessView::InflatedStage {
        compute_render_bounds(scene, render_data)
    } else {
        compute_scene_bounds(scene, workflow)
    };
    let scene_center = (scene_bounds.min + scene_bounds.max) * 0.5;
    let scene_radius = ((scene_bounds.max - scene_bounds.min) * 0.5)
        .length()
        .max(1.0);

    match options.view {
        HeadlessView::InflatedStage => {
            for (draw_index, draw) in workflow
                .runtime
                .scene_plan
                .stage_surface_draws
                .iter()
                .enumerate()
            {
                let Some(surface) = scene
                    .gifti_surfaces
                    .iter()
                    .find(|surface| surface.id == draw.source_id)
                else {
                    continue;
                };
                let colors = surface_vertex_colors_for_export(surface.data.as_ref(), draw);
                let positions = surface
                    .data
                    .vertices
                    .iter()
                    .map(|position| gltf_point(*position))
                    .collect::<Vec<_>>();
                let normals = surface
                    .data
                    .normals
                    .iter()
                    .map(|normal| gltf_vector(*normal))
                    .collect::<Vec<_>>();
                let material = builder.add_unlit_vertex_color_material(
                    format!("stage_surface_material_{draw_index}"),
                    draw.opacity,
                    false,
                );
                let mesh = builder.add_mesh(
                    format!("stage_surface_mesh_{}", surface.name),
                    &positions,
                    Some(&normals),
                    Some(&colors),
                    None,
                    &surface.data.indices,
                    material,
                    false,
                )?;
                for (panel_index, model_matrix) in stage_instance_model_matrices(
                    draw.structure,
                    surface.data.bbox_min,
                    surface.data.bbox_max,
                )
                .into_iter()
                .enumerate()
                {
                    builder.add_mesh_node(
                        format!(
                            "stage_surface_{}_{}_{}",
                            surface.name, draw_index, panel_index
                        ),
                        mesh,
                        gltf_transform(model_matrix),
                    );
                }
            }
        }
        _ => {
            for (draw_index, draw) in workflow.runtime.scene_plan.surface_draws.iter().enumerate() {
                let Some(surface) = scene
                    .gifti_surfaces
                    .iter()
                    .find(|surface| surface.id == draw.source_id)
                else {
                    continue;
                };
                let colors = surface_vertex_colors_for_export(surface.data.as_ref(), draw);
                let positions = surface
                    .data
                    .vertices
                    .iter()
                    .map(|position| gltf_point(*position))
                    .collect::<Vec<_>>();
                let normals = surface
                    .data
                    .normals
                    .iter()
                    .map(|normal| gltf_vector(*normal))
                    .collect::<Vec<_>>();
                let material = builder.add_vertex_color_material(
                    format!("surface_material_{draw_index}"),
                    draw.opacity,
                    false,
                    gloss_to_roughness(draw.gloss).max(0.22),
                    if draw.opacity < 0.999 { 0.12 } else { 0.08 },
                );
                let mesh = builder.add_mesh(
                    format!("surface_mesh_{}", surface.name),
                    &positions,
                    Some(&normals),
                    Some(&colors),
                    None,
                    &surface.data.indices,
                    material,
                    false,
                )?;
                builder.add_mesh_node(
                    format!("surface_{}_{}", surface.name, draw_index),
                    mesh,
                    gltf_transform(Mat4::from_cols_array_2d(&draw.model_matrix)),
                );
            }
        }
    }

    for draw in &workflow.runtime.scene_plan.bundle_draws {
        let fingerprint = workflow_bundle_display_fingerprint(
            draw,
            draw.boundary_field_node_uuid.and_then(|uuid| {
                workflow
                    .execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            }),
        );
        let Some(cache) = workflow
            .execution_cache
            .bundle_surface_mesh_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        else {
            continue;
        };
        for (component_index, (mesh, label)) in cache.meshes.iter().enumerate() {
            add_bundle_mesh_to_glb(&mut builder, draw, mesh, label, component_index)?;
        }
    }

    for draw in &workflow.runtime.scene_plan.streamline_draws {
        if !draw.visible {
            continue;
        }
        let fingerprint = workflow_streamline_fingerprint(draw);
        let Some(cache) = workflow
            .execution_cache
            .tube_geometry_cache
            .get(&draw.node_uuid)
            .filter(|cache| cache.fingerprint == fingerprint)
        else {
            continue;
        };
        let positions = cache
            .vertices
            .iter()
            .map(|vertex| gltf_point(vertex.position))
            .collect::<Vec<_>>();
        let normals = cache
            .vertices
            .iter()
            .map(|vertex| gltf_vector(vertex.normal))
            .collect::<Vec<_>>();
        let colors = cache
            .vertices
            .iter()
            .map(|vertex| vertex.color)
            .collect::<Vec<_>>();
        let alpha = colors
            .iter()
            .fold(1.0f32, |acc, color| acc.min(color[3]))
            .clamp(0.0, 1.0);
        let material = builder.add_vertex_color_material(
            format!("streamline_material_{}", draw.draw_id),
            alpha,
            false,
            0.32,
            0.16,
        );
        let mesh = builder.add_mesh(
            format!("streamline_mesh_{}", draw.label),
            &positions,
            Some(&normals),
            Some(&colors),
            None,
            &cache.indices,
            material,
            false,
        )?;
        builder.add_mesh_node(
            format!("streamlines_{}", draw.label),
            mesh,
            glam::Mat4::IDENTITY,
        );
    }

    if options.include_slices && options.view != HeadlessView::InflatedStage {
        for volume in &render_data.volume_draws {
            if volume.opacity <= 0.001 {
                continue;
            }
            let Some(nifti) = scene
                .nifti_files
                .iter()
                .find(|nifti| nifti.id == volume.file_id)
            else {
                continue;
            };
            for axis_index in 0..3 {
                if !scene.slice_visible[axis_index] {
                    continue;
                }
                add_slice_plane_to_glb(
                    &mut builder,
                    &nifti.volume,
                    volume,
                    axis_index,
                    scene.slice_indices[axis_index],
                    nifti.name.as_str(),
                )?;
            }
        }
    }

    if options.include_lights {
        add_lighting_rig_to_glb(&mut builder, render_3d, camera, scene_center, scene_radius);
    }

    if options.include_camera {
        let aspect = options.width as f32 / options.height.max(1) as f32;
        builder.add_camera_node("scene_camera".to_string(), camera, aspect);
    }

    let mut extras = Map::new();
    extras.insert(
        "trxviz_background".to_string(),
        match &render_3d.background {
            crate::lighting::WorkflowBackground3D::Solid { color } => {
                json!({ "mode": "solid", "color": color })
            }
            crate::lighting::WorkflowBackground3D::VerticalGradient { top, bottom } => {
                json!({ "mode": "vertical_gradient", "top": top, "bottom": bottom })
            }
        },
    );
    builder.scene_extras = Some(Value::Object(extras));
    builder.finish()
}

fn add_bundle_mesh_to_glb(
    builder: &mut GlbBuilder,
    draw: &crate::workflow::BundleDrawPlan,
    mesh: &BundleMesh,
    label: &str,
    component_index: usize,
) -> anyhow::Result<()> {
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| gltf_point(vertex.position))
        .collect::<Vec<_>>();
    let normals = mesh
        .vertices
        .iter()
        .map(|vertex| gltf_vector(vertex.normal))
        .collect::<Vec<_>>();
    let colors = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.color)
        .collect::<Vec<_>>();
    let material = if matches!(
        draw.build_mode,
        crate::workflow::BundleSurfaceBuildMode::Streamtubes
    ) {
        builder.add_unlit_vertex_color_material(
            format!("bundle_material_{}_{}", draw.draw_id, component_index),
            draw.opacity,
            true,
        )
    } else {
        builder.add_vertex_color_material(
            format!("bundle_material_{}_{}", draw.draw_id, component_index),
            draw.opacity,
            true,
            0.38,
            0.10,
        )
    };
    let mesh_index = builder.add_mesh(
        format!("bundle_mesh_{}_{}", draw.label, component_index),
        &positions,
        Some(&normals),
        Some(&colors),
        None,
        &mesh.indices,
        material,
        true,
    )?;
    builder.add_mesh_node(
        format!("bundle_{}_{}", label, component_index),
        mesh_index,
        glam::Mat4::IDENTITY,
    );
    Ok(())
}

fn add_slice_plane_to_glb(
    builder: &mut GlbBuilder,
    volume: &NiftiVolume,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
    volume_name: &str,
) -> anyhow::Result<()> {
    let corners = match axis_index {
        0 => volume.axial_slice_corners(slice_index),
        1 => volume.coronal_slice_corners(slice_index),
        _ => volume.sagittal_slice_corners(slice_index),
    };
    let positions = corners
        .into_iter()
        .map(|corner| gltf_point(corner.to_array()))
        .collect::<Vec<_>>();
    let normal = gltf_vector(slice_plane_normal(axis_index).to_array());
    let normals = vec![normal; 4];
    let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let png = bake_slice_png(volume, draw, axis_index, slice_index)?;
    let texture = builder.add_png_texture(
        format!(
            "{}_slice_texture_{}_{}",
            volume_name, axis_index, slice_index
        ),
        &png,
    );
    let material = builder.add_textured_material(
        format!(
            "{}_slice_material_{}_{}",
            volume_name, axis_index, slice_index
        ),
        draw.opacity,
        true,
        true,
        texture,
    );
    let mesh = builder.add_mesh(
        format!("{}_slice_mesh_{}_{}", volume_name, axis_index, slice_index),
        &positions,
        Some(&normals),
        None,
        Some(&texcoords),
        &indices,
        material,
        true,
    )?;
    builder.add_mesh_node(
        format!("{}_slice_{}_{}", volume_name, axis_index, slice_index),
        mesh,
        glam::Mat4::IDENTITY,
    );
    Ok(())
}

fn slice_plane_normal(axis_index: usize) -> Vec3 {
    match axis_index {
        0 => Vec3::Z,
        1 => Vec3::Y,
        _ => Vec3::X,
    }
}

fn surface_vertex_colors_for_export(
    surface: &GiftiSurfaceData,
    draw: &crate::workflow::SurfaceDrawPlan,
) -> Vec<[f32; 4]> {
    if !draw.vertex_rgba.is_empty() {
        return draw.vertex_rgba.clone();
    }
    bake_surface_vertex_colors(surface, draw)
}

fn bake_surface_vertex_colors(
    surface: &GiftiSurfaceData,
    draw: &crate::workflow::SurfaceDrawPlan,
) -> Vec<[f32; 4]> {
    let default = [draw.color[0], draw.color[1], draw.color[2], 1.0];
    let Some(scalars) = &draw.projection_scalars else {
        return vec![default; surface.vertices.len()];
    };

    scalars
        .iter()
        .map(|scalar| {
            let denom = (draw.range_max - draw.range_min).max(1e-6);
            let t = ((*scalar - draw.range_min) / denom).clamp(0.0, 1.0);
            let map_alpha = draw.map_opacity * if t >= draw.map_threshold { 1.0 } else { 0.0 };
            let map_rgb = surface_colormap_rgb(t, draw.projection_colormap);
            [
                draw.color[0] * (1.0 - map_alpha) + map_rgb[0] * map_alpha,
                draw.color[1] * (1.0 - map_alpha) + map_rgb[1] * map_alpha,
                draw.color[2] * (1.0 - map_alpha) + map_rgb[2] * map_alpha,
                1.0,
            ]
        })
        .collect()
}

fn surface_colormap_rgb(t: f32, cmap: crate::renderer::mesh_renderer::SurfaceColormap) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    match cmap {
        crate::renderer::mesh_renderer::SurfaceColormap::BlueWhiteRed => {
            if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0]
            } else {
                let s = (1.0 - t) * 2.0;
                [1.0, s, s]
            }
        }
        crate::renderer::mesh_renderer::SurfaceColormap::Viridis => interpolate_colormap(
            t,
            &[
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
            ],
        ),
        crate::renderer::mesh_renderer::SurfaceColormap::Inferno => interpolate_colormap(
            t,
            &[
                [0.001, 0.000, 0.014],
                [0.125, 0.047, 0.290],
                [0.302, 0.073, 0.488],
                [0.511, 0.121, 0.561],
                [0.709, 0.212, 0.486],
                [0.865, 0.316, 0.347],
                [0.962, 0.471, 0.212],
                [0.988, 0.683, 0.139],
                [0.978, 0.893, 0.306],
            ],
        ),
    }
}

fn interpolate_colormap(t: f32, colors: &[[f32; 3]]) -> [f32; 3] {
    if colors.len() == 1 {
        return colors[0];
    }
    let x = t * (colors.len() - 1) as f32;
    let i = x.floor().clamp(0.0, (colors.len() - 2) as f32) as usize;
    let f = x.fract();
    [
        colors[i][0] + (colors[i + 1][0] - colors[i][0]) * f,
        colors[i][1] + (colors[i + 1][1] - colors[i][1]) * f,
        colors[i][2] + (colors[i + 1][2] - colors[i][2]) * f,
    ]
}

fn bake_slice_png(
    volume: &NiftiVolume,
    draw: &VolumeDrawInfo,
    axis_index: usize,
    slice_index: usize,
) -> anyhow::Result<Vec<u8>> {
    let (width, height) = match axis_index {
        0 => (volume.dims[0] as u32, volume.dims[1] as u32),
        1 => (volume.dims[0] as u32, volume.dims[2] as u32),
        _ => (volume.dims[1] as u32, volume.dims[2] as u32),
    };
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    let lo = draw.window_center - draw.window_width * 0.5;
    let hi = draw.window_center + draw.window_width * 0.5;

    for row in 0..height as usize {
        for col in 0..width as usize {
            let value = match axis_index {
                0 => {
                    volume.data
                        [col + row * volume.dims[0] + slice_index * volume.dims[0] * volume.dims[1]]
                }
                1 => {
                    volume.data
                        [col + slice_index * volume.dims[0] + row * volume.dims[0] * volume.dims[1]]
                }
                _ => {
                    volume.data
                        [slice_index + col * volume.dims[0] + row * volume.dims[0] * volume.dims[1]]
                }
            };
            let t = ((value - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
            let rgb = volume_colormap_rgb(t, draw.colormap);
            let dst = (row * width as usize + col) * 4;
            rgba[dst] = float_channel(rgb[0]);
            rgba[dst + 1] = float_channel(rgb[1]);
            rgba[dst + 2] = float_channel(rgb[2]);
            rgba[dst + 3] = float_channel(draw.opacity);
        }
    }

    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png).write_image(
        &rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(png)
}

fn volume_colormap_rgb(t: f32, colormap: u32) -> [f32; 3] {
    match colormap {
        1 => [
            clamp01(t * 2.5),
            clamp01(t * 2.5 - 1.0),
            clamp01(t * 5.0 - 4.0),
        ],
        2 => [t, 1.0 - t, 1.0],
        3 => [1.0, t, 0.0],
        4 => [0.0, t, 1.0],
        _ => [t, t, t],
    }
}

fn gloss_to_roughness(gloss: f32) -> f32 {
    (1.0 - gloss.clamp(0.0, 1.0) * 0.9).clamp(0.05, 1.0)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn float_channel(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn gltf_point(point: [f32; 3]) -> [f32; 3] {
    (gltf_axis_conversion() * Vec3::from(point)).to_array()
}

fn gltf_vector(vector: [f32; 3]) -> [f32; 3] {
    (gltf_axis_conversion() * Vec3::from(vector))
        .normalize_or_zero()
        .to_array()
}

fn gltf_axis_conversion() -> glam::Mat3 {
    GLTF_AXIS_CONVERSION
}

fn gltf_transform(transform: Mat4) -> Mat4 {
    let basis = Mat4::from_mat3(gltf_axis_conversion());
    basis * transform * basis.inverse()
}

fn add_lighting_rig_to_glb(
    builder: &mut GlbBuilder,
    render_3d: &WorkflowRender3D,
    camera: &OrbitCamera,
    scene_center: Vec3,
    scene_radius: f32,
) {
    use crate::lighting::SceneLightingPreset;

    let eye = camera.eye();
    let forward = (camera.center - eye).normalize_or_zero();
    let right = forward.cross(Vec3::Z).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let rig_distance = scene_radius * 2.2;

    let (headlight_power, key_power, fill_power, rim_power, overhead_power, backfill_power) =
        match render_3d.lighting_preset {
            SceneLightingPreset::Flat => (9000.0, 4200.0, 2800.0, 0.0, 1800.0, 1200.0),
            SceneLightingPreset::Soft => (13000.0, 6500.0, 4200.0, 2200.0, 2800.0, 1800.0),
            SceneLightingPreset::Studio => (16500.0, 9000.0, 5200.0, 3200.0, 3600.0, 2400.0),
        };

    builder.add_spot_light(
        "camera_headlight".to_string(),
        eye,
        camera.center,
        55_f32.to_radians(),
        38_f32.to_radians(),
        headlight_power,
    );

    let key_pos = scene_center + right * rig_distance * 0.9 + up * rig_distance * 0.7
        - forward * rig_distance * 0.55;
    builder.add_spot_light(
        "key_light".to_string(),
        key_pos,
        scene_center,
        70_f32.to_radians(),
        48_f32.to_radians(),
        key_power,
    );

    let fill_pos = scene_center - right * rig_distance * 1.1 + up * rig_distance * 0.35
        - forward * rig_distance * 0.25;
    builder.add_point_light("fill_light".to_string(), fill_pos, fill_power);

    if rim_power > 0.0 {
        let rim_pos = scene_center - right * rig_distance * 0.8
            + up * rig_distance * 0.55
            + forward * rig_distance * 0.95;
        builder.add_spot_light(
            "rim_light".to_string(),
            rim_pos,
            scene_center,
            65_f32.to_radians(),
            42_f32.to_radians(),
            rim_power,
        );
    }

    let overhead_pos = scene_center + Vec3::Z * rig_distance * 1.5;
    builder.add_point_light("overhead_fill".to_string(), overhead_pos, overhead_power);

    let backfill_pos = scene_center + forward * rig_distance * 1.2 + up * rig_distance * 0.15;
    builder.add_point_light("back_fill".to_string(), backfill_pos, backfill_power);
}

struct GlbBuilder {
    bin: Vec<u8>,
    accessors: Vec<Value>,
    buffer_views: Vec<Value>,
    materials: Vec<Value>,
    meshes: Vec<Value>,
    nodes: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    scene_nodes: Vec<usize>,
    scene_extras: Option<Value>,
    extensions_used: BTreeSet<String>,
    extensions_required: BTreeSet<String>,
}

impl GlbBuilder {
    fn new() -> Self {
        Self {
            bin: Vec::new(),
            accessors: Vec::new(),
            buffer_views: Vec::new(),
            materials: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            cameras: Vec::new(),
            lights: Vec::new(),
            scene_nodes: Vec::new(),
            scene_extras: None,
            extensions_used: BTreeSet::new(),
            extensions_required: BTreeSet::new(),
        }
    }

    fn add_vertex_color_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
        roughness: f32,
        emissive_strength: f32,
    ) -> usize {
        let alpha_mode = if alpha < 0.999 { "BLEND" } else { "OPAQUE" };
        let material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": alpha_mode,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
            "emissiveFactor": [emissive_strength, emissive_strength, emissive_strength],
        });
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_unlit_vertex_color_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
    ) -> usize {
        self.extensions_used
            .insert("KHR_materials_unlit".to_string());
        self.extensions_required
            .insert("KHR_materials_unlit".to_string());

        let material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": if alpha < 0.999 { "BLEND" } else { "OPAQUE" },
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "extensions": {
                "KHR_materials_unlit": {}
            }
        });
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_textured_material(
        &mut self,
        name: String,
        alpha: f32,
        double_sided: bool,
        unlit: bool,
        texture_index: usize,
    ) -> usize {
        let mut material = json!({
            "name": name,
            "doubleSided": double_sided,
            "alphaMode": if alpha < 0.999 { "BLEND" } else { "OPAQUE" },
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, alpha],
                "baseColorTexture": { "index": texture_index },
                "metallicFactor": 0.0,
                "roughnessFactor": if unlit { 1.0 } else { 0.8 },
            }
        });
        if unlit {
            self.extensions_used
                .insert("KHR_materials_unlit".to_string());
            self.extensions_required
                .insert("KHR_materials_unlit".to_string());
            material["extensions"] = json!({ "KHR_materials_unlit": {} });
        }
        self.materials.push(material);
        self.materials.len() - 1
    }

    fn add_png_texture(&mut self, name: String, png_bytes: &[u8]) -> usize {
        let buffer_view = self.push_bytes(png_bytes, None);
        self.images.push(json!({
            "name": name,
            "bufferView": buffer_view,
            "mimeType": "image/png",
        }));
        self.textures.push(json!({
            "source": self.images.len() - 1,
        }));
        self.textures.len() - 1
    }

    fn add_mesh(
        &mut self,
        name: String,
        positions: &[[f32; 3]],
        normals: Option<&[[f32; 3]]>,
        colors: Option<&[[f32; 4]]>,
        texcoords: Option<&[[f32; 2]]>,
        indices: &[u32],
        material: usize,
        double_sided: bool,
    ) -> anyhow::Result<usize> {
        let mut attributes = Map::new();
        attributes.insert(
            "POSITION".to_string(),
            Value::from(self.add_accessor_vec3_f32(positions, Some(34962), true)),
        );
        if let Some(normals) = normals {
            attributes.insert(
                "NORMAL".to_string(),
                Value::from(self.add_accessor_vec3_f32(normals, Some(34962), false)),
            );
        }
        if let Some(colors) = colors {
            attributes.insert(
                "COLOR_0".to_string(),
                Value::from(self.add_accessor_vec4_f32(colors, Some(34962))),
            );
        }
        if let Some(texcoords) = texcoords {
            attributes.insert(
                "TEXCOORD_0".to_string(),
                Value::from(self.add_accessor_vec2_f32(texcoords, Some(34962))),
            );
        }
        let indices_accessor = self.add_accessor_u32(indices, Some(34963));
        self.meshes.push(json!({
            "name": name,
            "primitives": [{
                "attributes": Value::Object(attributes),
                "indices": indices_accessor,
                "material": material,
                "mode": 4
            }]
        }));
        let _ = double_sided;
        Ok(self.meshes.len() - 1)
    }

    fn add_mesh_node(&mut self, name: String, mesh_index: usize, transform: glam::Mat4) {
        self.nodes.push(json!({
            "name": name,
            "mesh": mesh_index,
            "matrix": transform.to_cols_array(),
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_camera_node(&mut self, name: String, camera: &OrbitCamera, aspect: f32) {
        self.cameras.push(json!({
            "name": name,
            "type": "perspective",
            "perspective": {
                "aspectRatio": aspect,
                "yfov": camera.fov_y,
                "znear": camera.near,
                "zfar": camera.far,
            }
        }));
        let transform = camera_node_transform(camera.eye(), camera.center, Vec3::Z);
        self.nodes.push(json!({
            "name": name,
            "camera": self.cameras.len() - 1,
            "matrix": transform.to_cols_array(),
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_point_light(&mut self, name: String, position: Vec3, intensity: f32) {
        self.extensions_used
            .insert("KHR_lights_punctual".to_string());
        self.lights.push(json!({
            "name": name,
            "type": "point",
            "intensity": intensity,
            "color": [1.0, 1.0, 1.0],
            "range": 0.0,
        }));
        let position = gltf_axis_conversion() * position;
        self.nodes.push(json!({
            "name": name,
            "translation": position.to_array(),
            "extensions": {
                "KHR_lights_punctual": {
                    "light": self.lights.len() - 1
                }
            }
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_spot_light(
        &mut self,
        name: String,
        position: Vec3,
        target: Vec3,
        outer_cone_angle: f32,
        inner_cone_angle: f32,
        intensity: f32,
    ) {
        self.extensions_used
            .insert("KHR_lights_punctual".to_string());
        self.lights.push(json!({
            "name": name,
            "type": "spot",
            "intensity": intensity,
            "color": [1.0, 1.0, 1.0],
            "range": 0.0,
            "spot": {
                "innerConeAngle": inner_cone_angle,
                "outerConeAngle": outer_cone_angle,
            }
        }));
        let transform = camera_node_transform(position, target, Vec3::Z);
        self.nodes.push(json!({
            "name": name,
            "matrix": transform.to_cols_array(),
            "extensions": {
                "KHR_lights_punctual": {
                    "light": self.lights.len() - 1
                }
            }
        }));
        self.scene_nodes.push(self.nodes.len() - 1);
    }

    fn add_accessor_vec3_f32(
        &mut self,
        data: &[[f32; 3]],
        target: Option<u32>,
        include_bounds: bool,
    ) -> usize {
        let bytes = bytemuck::cast_slice(data);
        let buffer_view = self.push_bytes(bytes, target);
        let mut accessor = json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC3",
        });
        if include_bounds && !data.is_empty() {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for value in data {
                for axis in 0..3 {
                    min[axis] = min[axis].min(value[axis]);
                    max[axis] = max[axis].max(value[axis]);
                }
            }
            accessor["min"] = json!(min);
            accessor["max"] = json!(max);
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn add_accessor_vec4_f32(&mut self, data: &[[f32; 4]], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC4",
        }));
        self.accessors.len() - 1
    }

    fn add_accessor_vec2_f32(&mut self, data: &[[f32; 2]], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC2",
        }));
        self.accessors.len() - 1
    }

    fn add_accessor_u32(&mut self, data: &[u32], target: Option<u32>) -> usize {
        let buffer_view = self.push_bytes(bytemuck::cast_slice(data), target);
        self.accessors.push(json!({
            "bufferView": buffer_view,
            "componentType": 5125,
            "count": data.len(),
            "type": "SCALAR",
        }));
        self.accessors.len() - 1
    }

    fn push_bytes(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let mut buffer_view = json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
        });
        if let Some(target) = target {
            buffer_view["target"] = Value::from(target);
        }
        self.buffer_views.push(buffer_view);
        self.buffer_views.len() - 1
    }

    fn finish(self) -> anyhow::Result<Vec<u8>> {
        let mut root = json!({
            "asset": {
                "version": "2.0",
                "generator": "trxviz-cli",
            },
            "scene": 0,
            "scenes": [{
                "nodes": self.scene_nodes,
            }],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{
                "byteLength": self.bin.len(),
            }],
        });
        if let Some(extras) = self.scene_extras {
            root["scenes"][0]["extras"] = extras;
        }
        if !self.images.is_empty() {
            root["images"] = Value::Array(self.images);
        }
        if !self.textures.is_empty() {
            root["textures"] = Value::Array(self.textures);
        }
        if !self.cameras.is_empty() {
            root["cameras"] = Value::Array(self.cameras);
        }
        if !self.lights.is_empty() {
            root["extensions"] = json!({
                "KHR_lights_punctual": {
                    "lights": self.lights
                }
            });
        }
        if !self.extensions_used.is_empty() {
            root["extensionsUsed"] =
                Value::Array(self.extensions_used.into_iter().map(Value::from).collect());
        }
        if !self.extensions_required.is_empty() {
            root["extensionsRequired"] = Value::Array(
                self.extensions_required
                    .into_iter()
                    .map(Value::from)
                    .collect(),
            );
        }

        let mut json_bytes = serde_json::to_vec(&root)?;
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let mut bin = self.bin;
        while bin.len() % 4 != 0 {
            bin.push(0);
        }

        let total_length = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        Ok(glb)
    }
}

fn camera_node_transform(eye: Vec3, target: Vec3, up: Vec3) -> glam::Mat4 {
    let eye = gltf_axis_conversion() * eye;
    let target = gltf_axis_conversion() * target;
    let up = (gltf_axis_conversion() * up).normalize_or_zero();
    let forward = (target - eye).normalize_or_zero();
    let right = forward.cross(up).normalize_or_zero();
    let corrected_up = (-forward).cross(right).normalize_or_zero();
    glam::Mat4::from_cols(
        right.extend(0.0),
        corrected_up.extend(0.0),
        (-forward).extend(0.0),
        eye.extend(1.0),
    )
}

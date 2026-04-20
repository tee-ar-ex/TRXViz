//! Headless rendering entrypoints for project JSON and loose-asset scene capture.

mod bake;
mod export_glb;
mod gpu_context;
mod readback;
mod render_2d;
mod render_3d;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use glam::Vec3;
use trx_rs::{AnyTrxFile, ConversionOptions};

use self::export_glb::{build_glb_scene, compute_scene_bounds};
use self::gpu_context::{build_gpu_resources, create_gpu_context};
use self::render_2d::render_scene2d_to_png;
use self::render_3d::{
    build_camera, build_render_data, compute_render_bounds, render_scene3d_to_png,
};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{
    FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use crate::data::nifti_data::NiftiVolume;
use crate::data::odx_data::OdxScene;
use crate::data::orientation_field::BoundaryGlyphColorMode;
use crate::data::parcellation_data::{ParcellationVolume, guess_label_table_path};
use crate::data::trx_data::{RenderStyle, TrxGpuData};
use crate::renderer::mesh_renderer::MeshDrawStyle;
use crate::scene::{
    HeadlessScene, HeadlessWorkflowState, LoadedGiftiSurface, LoadedParcellationSource,
    LoadedStreamlineSource, direct_streamline_import_warnings,
};
use crate::units::Millimeters;
use crate::workflow::{
    BundleSurfacePlan, CachedBoundaryField, CachedBundleSurfaceMeshes, CachedDerivedStreamline,
    CachedSurfaceQuery, CachedSurfaceStreamlineMap, CachedTubeGeometry, LoadedParcellation,
    ParcellationAsset, WorkflowAssetDocument, WorkflowJobOutput,
    WorkflowJobPayload, WorkflowNodeUuid, add_default_nodes_for_asset, ensure_node_uuids,
    evaluate_scene_plan, load_workflow_project_from_path, mark_expensive_success,
    resolve_document_asset_paths, run_workflow_job, save_streamline_plan,
    set_default_odx_fixel_3d_visibility, set_default_odx_fixel_dpf,
    set_default_odx_volume_dpv, workflow_boundary_plan_fingerprint,
    workflow_bundle_display_fingerprint, workflow_bundle_plan_fingerprint,
    workflow_reactive_streamline_fingerprint, workflow_streamline_fingerprint,
    workflow_surface_projection_fingerprint, workflow_surface_query_fingerprint,
};

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
    fixel_2d_line_width: f32,
    fixel_2d_slab_half_width_mm: Millimeters,
    fixel_2d_opacity: f32,
    fixel_2d_colormap_code: u32,
    fixel_2d_scalar_range: [f32; 2],
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
                let warnings = odx_scene.glyph_warnings().to_vec();
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                scene.odx_files.push(crate::data::loaded_files::LoadedOdx {
                    id,
                    name,
                    path,
                    scene: Arc::new(odx_scene),
                    warnings,
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
            warnings: odx_arc.glyph_warnings().to_vec(),
            visible: true,
        });
        let dpv_names: Vec<String> = odx_arc.dpv_names().iter().map(|s| s.to_string()).collect();
        let dpf_names: Vec<String> = odx_arc
            .dataset()
            .dpf_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
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
        let _ = set_default_odx_fixel_dpf(&mut workflow.document, id, &dpf_names);
        let _ = set_default_odx_volume_dpv(&mut workflow.document, id, &dpv_names);
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
                plan.dps_field.as_ref().map(|field| field.as_str()),
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

//! Headless rendering entrypoints for project JSON and loose-asset scene capture.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use glam::Vec3;
use image::ColorType;
use pollster::block_on;
use trx_rs::{AnyTrxFile, ConversionOptions};

use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap};
use crate::data::nifti_data::NiftiVolume;
use crate::data::orientation_field::BoundaryGlyphColorMode;
use crate::data::parcellation_data::{ParcellationVolume, guess_label_table_path};
use crate::data::trx_data::{RenderStyle, TrxGpuData};
use crate::lighting::SceneLightingParams;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};
use crate::scene::{HeadlessScene, HeadlessWorkflowState, LoadedGiftiSurface, LoadedParcellationSource, LoadedStreamlineSource};
use crate::workflow::{
    BundleSurfacePlan, CachedBoundaryField, CachedBundleSurfaceMeshes, CachedDerivedStreamline,
    CachedSurfaceQuery, CachedSurfaceStreamlineMap, CachedTubeGeometry, LoadedParcellation,
    ParcellationAsset, WorkflowAssetDocument, WorkflowJobOutput, WorkflowJobPayload,
    WorkflowNodeUuid, add_default_nodes_for_asset, ensure_node_uuids, evaluate_scene_plan,
    load_workflow_project_from_path, mark_expensive_success, resolve_document_asset_paths,
    run_workflow_job, save_streamline_plan,
    workflow_boundary_plan_fingerprint, workflow_bundle_display_fingerprint,
    workflow_bundle_plan_fingerprint, workflow_reactive_streamline_fingerprint,
    workflow_streamline_fingerprint, workflow_surface_projection_fingerprint,
    workflow_surface_query_fingerprint,
};

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct HeadlessRenderOptions {
    pub width: u32,
    pub height: u32,
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
            target: None,
            azimuth_deg: None,
            elevation_deg: None,
            distance: None,
        }
    }
}

#[derive(Default)]
pub struct AssetArgs {
    pub trx_paths: Vec<PathBuf>,
    pub nifti_paths: Vec<PathBuf>,
    pub surface_paths: Vec<PathBuf>,
    pub parcellation_paths: Vec<PathBuf>,
}

struct HeadlessRenderData {
    surface_draws: Vec<(usize, MeshDrawStyle)>,
    volume_draws: Vec<VolumeDrawInfo>,
    streamline_draws: Vec<StreamlineDrawInfo>,
    bundle_draws: Vec<BundleDrawInfo>,
    any_visible_streamlines: bool,
    glyph_visible: bool,
    glyph_color_mode: BoundaryGlyphColorMode,
    glyph_density_3d_step: u32,
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

struct SceneBounds {
    min: Vec3,
    max: Vec3,
}

struct GpuSceneResources {
    streamlines: AllStreamlineResources,
    slices: AllSliceResources,
    meshes: MeshResources,
    glyphs: GlyphResources,
    bounds: SceneBounds,
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

fn render_loaded_scene(
    mut scene: HeadlessScene,
    mut workflow: HeadlessWorkflowState,
    output_path: &Path,
    options: &HeadlessRenderOptions,
) -> anyhow::Result<()> {
    execute_workflow_to_completion(&scene, &mut workflow)?;
    let gpu = create_gpu_context()?;
    let mut resources =
        build_gpu_resources(&gpu.device, &gpu.queue, &scene, &workflow).context("building GPU resources")?;
    let render_data = build_render_data(&scene, &workflow);
    let camera = build_camera(&resources.bounds, options, options.width as f32 / options.height as f32);
    if render_data.glyph_visible {
        scene.boundary_field = workflow
            .runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible)
            .and_then(|draw| workflow.execution_cache.boundary_field_cache.get(&draw.build_node_uuid))
            .map(|cache| cache.field.clone());
    }
    render_scene3d_to_png(
        &gpu.device,
        &gpu.queue,
        &mut resources,
        &render_data,
        &camera,
        scene.slice_visible,
        options.width,
        options.height,
        output_path,
    )
}

fn load_project_state(project_path: &Path) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut project =
        load_workflow_project_from_path(project_path).map_err(|err| anyhow!(err))?;
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
        }
    }

    let workflow = HeadlessWorkflowState {
        document: project.document,
        project_path: Some(project_path.to_path_buf()),
        ..Default::default()
    };
    Ok((scene, workflow))
}

fn load_asset_args_state(args: &AssetArgs) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut scene = HeadlessScene::default();
    let mut workflow = HeadlessWorkflowState::default();

    for path in &args.trx_paths {
        let any = AnyTrxFile::load(path).map_err(|err| anyhow!(err.to_string()))?;
        let source = load_streamline_source_from_any(any)?;
        let id = apply_loaded_trx(&mut scene, path.clone(), source, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Streamlines {
                id,
                path: path.clone(),
                imported: false,
            },
            Some(
                scene.trx_files
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

    if workflow.document.assets.is_empty() {
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
        })
        .map_err(|err| anyhow!(err.to_string()))
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
    let LoadedStreamlineSource { data, backing } = source;
    let is_first = scene.trx_files.is_empty()
        && scene.nifti_files.is_empty()
        && scene.gifti_surfaces.is_empty();
    scene.volume_center = data.center();
    scene.volume_extent = data.extent();
    if is_first {
        scene.slice_world_offsets = [scene.volume_center.z, scene.volume_center.y, scene.volume_center.x];
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
        scene.slice_world_offsets = [scene.volume_center.z, scene.volume_center.y, scene.volume_center.x];
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
        &scene.gifti_surfaces,
        &scene.parcellations,
        &mut workflow.display_runtimes,
        &mut workflow.next_draw_id,
        &mut workflow.execution_cache,
        false,
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

        for plan in workflow.runtime.scene_plan.reactive_streamline_plans.clone() {
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            let record = workflow.execution_cache.node_runs.entry(plan.node_uuid).or_default();
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
            let record = workflow.execution_cache.node_runs.entry(plan.node_uuid).or_default();
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
            let record = workflow.execution_cache.node_runs.entry(plan.node_uuid).or_default();
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
            let record = workflow.execution_cache.node_runs.entry(draw.node_uuid).or_default();
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
            let record = workflow.execution_cache.node_runs.entry(draw.node_uuid).or_default();
            if record.last_success_fingerprint == Some(fingerprint) {
                continue;
            }
            let plan = BundleSurfacePlan {
                build_node_uuid: draw.build_node_uuid,
                label: draw.label.clone(),
                flow: draw.flow.clone(),
                per_group: draw.per_group,
                voxel_size_mm: draw.voxel_size_mm,
                threshold: draw.threshold,
                smooth_sigma: draw.smooth_sigma,
                min_component_volume_mm3: draw.min_component_volume_mm3,
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
            cache.derived_streamline_cache
                .insert(node_uuid, CachedDerivedStreamline { flow });
            mark_expensive_success(record, fingerprint, "reactive streamlines".to_string());
        }
        WorkflowJobOutput::SurfaceQuery(flow) => {
            cache.surface_query_cache
                .insert(node_uuid, CachedSurfaceQuery { flow });
            mark_expensive_success(record, fingerprint, "surface query".to_string());
        }
        WorkflowJobOutput::SurfaceMap(map) => {
            cache.surface_streamline_map_cache.insert(
                node_uuid,
                CachedSurfaceStreamlineMap { map },
            );
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
                CachedBundleSurfaceMeshes { fingerprint, meshes },
            );
            mark_expensive_success(record, fingerprint, summary);
        }
        WorkflowJobOutput::BoundaryField { field } => {
            let summary = field
                .as_ref()
                .map(|_| "Boundary field".to_string())
                .unwrap_or_else(|| "No boundary field".to_string());
            if let Some(field) = field {
                cache.boundary_field_cache.insert(
                    node_uuid,
                    CachedBoundaryField { fingerprint, field },
                );
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

    let mut slices = AllSliceResources { entries: Vec::new() };
    for nifti in &scene.nifti_files {
        let slice_resources = SliceResources::new(device, queue, TARGET_FORMAT, &nifti.volume);
        slice_resources.update_slice(queue, SliceAxis::Axial, scene.slice_indices[0], &nifti.volume);
        slice_resources.update_slice(queue, SliceAxis::Coronal, scene.slice_indices[1], &nifti.volume);
        slice_resources.update_slice(queue, SliceAxis::Sagittal, scene.slice_indices[2], &nifti.volume);
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
        if let Some(scalars) = &draw.projection_scalars {
            meshes.update_surface_scalars(queue, draw.source_id, scalars);
        }
    }

    let mut streamlines = AllStreamlineResources { entries: Vec::new() };
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
        if let Some(cache) = workflow.execution_cache.boundary_field_cache.get(&draw.build_node_uuid) {
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

    if !bounds_min.is_finite() || !bounds_max.is_finite() {
        bounds_min = scene.volume_center - Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
        bounds_max = scene.volume_center + Vec3::splat(scene.volume_extent.max(1.0) * 0.5);
    }

    Ok(GpuSceneResources {
        streamlines,
        slices,
        meshes,
        glyphs,
        bounds: SceneBounds {
            min: bounds_min,
            max: bounds_max,
        },
    })
}

fn build_render_data(_scene: &HeadlessScene, workflow: &HeadlessWorkflowState) -> HeadlessRenderData {
    let surface_draws = workflow
        .runtime
        .scene_plan
        .surface_draws
        .iter()
        .map(|draw| {
            (
                draw.source_id,
                MeshDrawStyle {
                    color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                    scalar_min: draw.range_min,
                    scalar_max: draw.range_max,
                    scalar_enabled: draw.show_projection_map,
                    colormap: draw.projection_colormap,
                    gloss: draw.gloss,
                    map_opacity: draw.map_opacity,
                    map_threshold: draw.map_threshold,
                },
            )
        })
        .collect();
    let volume_draws = workflow
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
        .collect::<Vec<_>>();
    let streamline_draws = workflow
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
        .collect::<Vec<_>>();
    let bundle_draws = workflow
        .runtime
        .scene_plan
        .bundle_draws
        .iter()
        .map(|draw| BundleDrawInfo {
            file_id: draw.draw_id,
            opacity: draw.opacity,
        })
        .collect::<Vec<_>>();
    let glyph_draw = workflow
        .runtime
        .scene_plan
        .boundary_glyph_draws
        .iter()
        .find(|draw| draw.visible);

    HeadlessRenderData {
        any_visible_streamlines: streamline_draws.iter().any(|draw| draw.visible),
        surface_draws,
        volume_draws,
        streamline_draws,
        bundle_draws,
        glyph_visible: glyph_draw.is_some() && !workflow.execution_cache.boundary_field_cache.is_empty(),
        glyph_color_mode: glyph_draw
            .map(|draw| draw.color_mode)
            .unwrap_or(BoundaryGlyphColorMode::DirectionRgb),
        glyph_density_3d_step: glyph_draw
            .map(|draw| draw.density_3d_step as u32)
            .unwrap_or(1),
    }
}

fn build_camera(bounds: &SceneBounds, options: &HeadlessRenderOptions, aspect: f32) -> OrbitCamera {
    let center = options.target.unwrap_or((bounds.min + bounds.max) * 0.5);
    let radius = ((bounds.max - bounds.min) * 0.5).length().max(1.0);
    let mut camera = OrbitCamera::new(center, fit_distance(radius, aspect));
    camera.yaw = options
        .azimuth_deg
        .unwrap_or(45.0)
        .to_radians();
    camera.pitch = options
        .elevation_deg
        .unwrap_or(25.0)
        .to_radians();
    if let Some(distance) = options.distance {
        camera.distance = distance.max(0.1);
    }
    camera
}

fn fit_distance(radius: f32, aspect: f32) -> f32 {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let half_y = fov_y * 0.5;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let limiting_half_angle = half_y.min(half_x).max(0.1);
    (radius / limiting_half_angle.sin()) * 1.1
}

fn render_scene3d_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    render_data: &HeadlessRenderData,
    camera: &OrbitCamera,
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
    let lighting = SceneLightingParams::default();

    for volume in &render_data.volume_draws {
        if let Some((_, slice)) = resources.slices.entries.iter().find(|(id, _)| *id == volume.file_id) {
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
                3,
                0.0,
                0.0,
                aux,
                lighting,
            );
        }
    }
    for (surface_id, style) in &render_data.surface_draws {
        resources.meshes.update_surface_uniforms(
            queue,
            *surface_id,
            0,
            view_proj,
            style,
            camera_pos,
            lighting,
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
        );
    }
    if render_data.glyph_visible {
        resources.glyphs.update_uniforms(
            queue,
            0,
            view_proj,
            3,
            0.0,
            0.0,
            render_data.glyph_color_mode,
            render_data.glyph_density_3d_step,
            lighting,
        );
    }

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trxviz_headless_encoder") });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trxviz_headless_pass"),
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

        render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);

        for volume in &render_data.volume_draws {
            if let Some((_, slice)) = resources.slices.entries.iter().find(|(id, _)| *id == volume.file_id) {
                render_pass.set_pipeline(&slice.pipeline);
                render_pass.set_bind_group(0, &slice.bind_groups[0], &[]);
                render_pass.set_index_buffer(slice.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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
                            render_pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..resource.num_tube_indices, 0, 0..1);
                        }
                    } else {
                        render_pass.set_pipeline(&resource.pipeline);
                        render_pass.set_vertex_buffer(0, resource.position_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, resource.color_buffer.slice(..));
                        render_pass.set_vertex_buffer(2, resource.tangent_buffer.slice(..));
                        render_pass.set_index_buffer(resource.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..resource.num_indices, 0, 0..1);
                    }
                }
            }
        }

        if !render_data.surface_draws.is_empty() {
            resources.meshes.paint_opaque(render_pass, 0, &render_data.surface_draws);
        }
        if !render_data.bundle_draws.is_empty() {
            let bundle_draws = render_data
                .bundle_draws
                .iter()
                .map(|draw| (draw.file_id, draw.opacity))
                .collect::<Vec<_>>();
            resources.meshes.paint_bundle_opaque(render_pass, &bundle_draws);
            resources
                .meshes
                .paint_bundle_transparent(render_pass, &bundle_draws, camera_dir);
        }
        if !render_data.surface_draws.is_empty() {
            resources
                .meshes
                .paint_transparent(render_pass, 0, &render_data.surface_draws, camera_dir);
        }
        if render_data.glyph_visible {
            resources.glyphs.paint(render_pass, 0, false);
        }
    }

    let padded_bytes_per_row =
        ((width * 4 + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1) / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trxviz_headless_readback"),
        size: padded_bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
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

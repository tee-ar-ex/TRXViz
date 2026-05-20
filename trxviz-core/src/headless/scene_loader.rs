use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use glam::Vec3;

use crate::asset_loader::{AssetLoader, LoadedAsset, load_asset};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{
    FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use crate::data::nifti_data::NiftiVolume;
use crate::data::odx_data::OdxScene;
use crate::data::parcellation_data::ParcellationVolume;
use crate::scene::{
    HeadlessScene, HeadlessWorkflowState, LoadedGiftiSurface, LoadedParcellationSource,
    LoadedStreamlineSource,
};
use crate::workflow::{
    LoadedParcellation, ParcellationAsset, WorkflowAssetDocument, add_default_nodes_for_asset,
    load_workflow_project_from_path, resolve_document_asset_paths,
    set_default_odx_fixel_3d_visibility, set_default_odx_fixel_dpf, set_default_odx_volume_dpv,
};

use super::AssetArgs;

pub(super) fn load_project_state(
    project_path: &Path,
) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut project = load_workflow_project_from_path(project_path).map_err(|err| anyhow!(err))?;
    resolve_document_asset_paths(&mut project.document, project_path);

    let mut scene = HeadlessScene::default();
    for asset in project.document.assets.clone() {
        match asset {
            WorkflowAssetDocument::Streamlines { id, path, imported } => {
                let source = LoadedStreamlineSource::load(&path)?;
                let _ = imported;
                apply_loaded_trx(&mut scene, path, source, Some(id));
            }
            WorkflowAssetDocument::Volume { id, path } => {
                let volume = NiftiVolume::load(&path)?;
                apply_loaded_nifti(&mut scene, path, volume, Some(id));
            }
            WorkflowAssetDocument::Cifti { .. } => {}
            WorkflowAssetDocument::Surface { id, path } => {
                let surface = GiftiSurfaceData::load(&path)?;
                apply_loaded_surface(&mut scene, path, surface, Some(id));
            }
            WorkflowAssetDocument::Parcellation {
                id,
                path,
                label_table_path,
            } => {
                let source = LoadedParcellationSource {
                    data: ParcellationVolume::load(&path, label_table_path.as_deref())?,
                    label_table_path,
                };
                apply_loaded_parcellation(&mut scene, path, source, Some(id));
            }
            WorkflowAssetDocument::Odx { id, path } => {
                let odx_scene = OdxScene::load(&path)
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
    }

    let workflow = HeadlessWorkflowState {
        document: project.document,
        project_path: Some(project_path.to_path_buf()),
        ..Default::default()
    };
    Ok((scene, workflow))
}

pub(super) fn load_asset_args_state(
    args: &AssetArgs,
) -> anyhow::Result<(HeadlessScene, HeadlessWorkflowState)> {
    let mut scene = HeadlessScene::default();
    let mut workflow = HeadlessWorkflowState::default();

    for path in &args.tractogram_paths {
        let LoadedAsset::Streamlines(source) = load_asset(path)? else {
            bail!("expected streamline asset for {}", path.display());
        };
        let imported = !matches!(source.backing, StreamlineBacking::Native(_));
        let id = apply_loaded_trx(&mut scene, path.clone(), source, None);
        let trx_asset = scene.trx_files.iter().find(|asset| asset.id == id);
        let limit = trx_asset
            .map(|asset| asset.data.nb_streamlines.min(30_000))
            .unwrap_or(30_000);
        let has_groups = trx_asset
            .map(|asset| !asset.data.groups.is_empty())
            .unwrap_or(false);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Streamlines {
                id,
                path: path.clone(),
                imported,
            },
            Some(limit),
            has_groups,
        );
    }

    for path in &args.nifti_paths {
        let LoadedAsset::Volume(volume) = load_asset(path)? else {
            bail!("expected NIfTI volume for {}", path.display());
        };
        let id = apply_loaded_nifti(&mut scene, path.clone(), volume, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Volume {
                id,
                path: path.clone(),
            },
            None,
            false,
        );
    }

    for path in &args.surface_paths {
        let LoadedAsset::Surface(surface) = load_asset(path)? else {
            bail!("expected GIFTI surface for {}", path.display());
        };
        let id = apply_loaded_surface(&mut scene, path.clone(), surface, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Surface {
                id,
                path: path.clone(),
            },
            None,
            false,
        );
    }

    for path in &args.parcellation_paths {
        let LoadedAsset::Parcellation(source) = load_asset(path)? else {
            bail!("expected parcellation for {}", path.display());
        };
        let label_table_path = source.label_table_path.clone();
        let id = apply_loaded_parcellation(&mut scene, path.clone(), source, None);
        register_asset_default_nodes(
            &mut workflow.document,
            WorkflowAssetDocument::Parcellation {
                id,
                path: path.clone(),
                label_table_path,
            },
            None,
            false,
        );
    }

    for path in &args.odx_paths {
        let LoadedAsset::Odx(odx_scene) =
            load_asset(path).with_context(|| format!("loading ODX file {}", path.display()))?
        else {
            bail!("expected ODX asset for {}", path.display());
        };
        let show_fixel_3d_by_default = odx_scene.glyph_source_kind().is_none();
        let dims = odx_scene.dimensions();
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
            false,
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
    streamline_has_groups: bool,
) {
    document.assets.push(asset.clone());
    let pos = crate::workflow::suggest_asset_branch_origin(document);
    let _ = add_default_nodes_for_asset(
        document,
        &asset,
        pos,
        streamline_limit,
        streamline_has_groups,
    );
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

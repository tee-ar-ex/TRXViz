use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use trx_rs::{
    AnyTrxFile, ConcatenateOptions, ConversionOptions, DType, Format, concatenate_any_trx,
    header_from_reference,
};
use trxviz_core::data::gifti_data::GiftiSurfaceData;
use trxviz_core::data::loaded_files::{
    FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use trxviz_core::data::nifti_data::NiftiVolume;
use trxviz_core::data::orientation_field::BoundaryContactField;
use trxviz_core::data::parcellation_data::{ParcellationVolume, guess_label_table_path};
use trxviz_core::data::trx_data::TrxGpuData;
use trxviz_core::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use trxviz_core::renderer::glyph_renderer::GlyphResources;
use trxviz_core::renderer::mesh_renderer::{MeshResources, SurfaceColormap};
use trxviz_core::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};

use super::state::{
    ImportDialogState, LoadedGiftiSurface, LoadedParcellationSource, LoadedStreamlineSource,
    MergeStreamlinesDialogState, WorkerMessage,
};
use super::workflow::{self, LoadedParcellation, ParcellationAsset, WorkflowAssetDocument};

impl super::TrxVizApp {
    fn loaded_streamline_source_from_any(
        any: AnyTrxFile,
    ) -> Result<LoadedStreamlineSource, String> {
        TrxGpuData::from_any_trx(&any)
            .map(|data| LoadedStreamlineSource {
                data,
                backing: StreamlineBacking::Native(Arc::new(any)),
                warnings: Vec::new(),
            })
            .map_err(|e| e.to_string())
    }

    fn allocate_file_id(&mut self, explicit_id: Option<FileId>) -> FileId {
        if let Some(id) = explicit_id {
            self.scene.next_file_id = self.scene.next_file_id.max(id + 1);
            id
        } else {
            let id = self.scene.next_file_id;
            self.scene.next_file_id += 1;
            id
        }
    }

    fn register_workflow_asset(
        &mut self,
        asset: WorkflowAssetDocument,
        add_default_nodes: bool,
        streamline_limit: Option<usize>,
    ) {
        self.workflow.document.assets.push(asset.clone());
        if add_default_nodes {
            let pos = workflow::suggest_asset_branch_origin(&self.workflow.document);
            let branch = workflow::add_default_nodes_for_asset(
                &mut self.workflow.document,
                &asset,
                pos,
                streamline_limit,
            );
            self.workflow.selection = Some(branch.primary_selection);
            self.workflow.graph_focus_request = Some(egui::Rect::from_min_max(
                egui::pos2(branch.bounds.min.x, branch.bounds.min.y),
                egui::pos2(branch.bounds.max.x, branch.bounds.max.y),
            ));
        }
    }

    pub(super) fn begin_load_trx(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading TRX".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = AnyTrxFile::load(&path)
                .map_err(|e| e.to_string())
                .and_then(Self::loaded_streamline_source_from_any);
            let _ = tx.send(WorkerMessage::TrxLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_nifti(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading NIfTI".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = NiftiVolume::load(&path).map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::NiftiLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_gifti_surface(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading GIFTI".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = GiftiSurfaceData::load(&path).map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::GiftiLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_parcellation(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading parcellation".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let label_table_path = guess_label_table_path(&path);
            let result = ParcellationVolume::load(&path, label_table_path.as_deref())
                .map(|data| LoadedParcellationSource {
                    data,
                    label_table_path,
                })
                .map_err(|err| err.to_string());
            let _ = tx.send(WorkerMessage::ParcellationLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_import_streamlines(&mut self, state: &ImportDialogState) {
        let Some(path) = state.source_path.clone() else {
            return;
        };
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let reference_path = state.reference_path.clone();
        let vtk_coordinate_mode = state.vtk_coordinate_mode;
        let label = path
            .file_name()
            .map(|n| format!("Importing {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Importing streamlines".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let options = match reference_path
                .as_deref()
                .map(header_from_reference)
                .transpose()
            {
                Ok(header) => ConversionOptions {
                    header,
                    vtk_coordinate_mode,
                    ..ConversionOptions::default()
                },
                Err(err) => {
                    let _ = tx.send(WorkerMessage::ImportedStreamlinesLoaded {
                        job_id,
                        path,
                        result: Err(err.to_string()),
                    });
                    return;
                }
            };
            let warnings = trxviz_core::scene::direct_streamline_import_warnings(&path, &options);
            let result = match trx_rs::read_tractogram(&path, &options) {
                Ok(tractogram) => TrxGpuData::from_tractogram(&tractogram)
                    .map(|data| LoadedStreamlineSource {
                        data,
                        backing: StreamlineBacking::Imported(Arc::new(tractogram)),
                        warnings,
                    })
                    .map_err(|e| e.to_string()),
                Err(err) => Err(err.to_string()),
            };
            let _ = tx.send(WorkerMessage::ImportedStreamlinesLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_import_streamlines_path(&mut self, path: PathBuf) {
        let mut state = ImportDialogState::default();
        state.source_path = Some(path.clone());
        state.detected_format = trx_rs::detect_format(&path).ok();
        self.begin_import_streamlines(&state);
    }

    pub(super) fn begin_merge_streamlines(&mut self, state: &MergeStreamlinesDialogState) {
        let Some(output_path) = state.output_path.clone() else {
            return;
        };
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let rows = state.rows.clone();
        let options = ConcatenateOptions {
            delete_dps: state.delete_dps,
            delete_dpv: state.delete_dpv,
            delete_groups: state.delete_groups,
            positions_dtype: state.positions_dtype.map(DType::from),
            input_group_names: state
                .rows
                .iter()
                .map(|row| {
                    let trimmed = row.group_name.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .collect(),
        };
        let label = output_path
            .file_name()
            .map(|n| format!("Creating {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Creating merged streamlines".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = create_merged_streamline_source(&rows, &output_path, &options);
            let _ = tx.send(WorkerMessage::MergedStreamlinesCreated {
                job_id,
                path: output_path,
                result,
            });
        });
    }

    pub(super) fn apply_loaded_trx(
        &mut self,
        path: PathBuf,
        source: LoadedStreamlineSource,
        rs: &egui_wgpu::RenderState,
    ) {
        self.apply_loaded_trx_with_options(path, source, rs, None, true);
    }

    pub(super) fn apply_loaded_trx_with_options(
        &mut self,
        path: PathBuf,
        source: LoadedStreamlineSource,
        rs: &egui_wgpu::RenderState,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        let LoadedStreamlineSource {
            data,
            backing,
            warnings,
        } = source;
        let imported = matches!(backing, StreamlineBacking::Imported(_));
        let is_first = self.scene.trx_files.is_empty()
            && self.scene.nifti_files.is_empty()
            && self.scene.gifti_surfaces.is_empty();
        self.viewport.volume_center = data.center();
        self.viewport.volume_extent = data.extent();
        if is_first {
            self.viewport.camera_3d = OrbitCamera::new(
                self.viewport.volume_center,
                self.viewport.volume_extent * 0.8,
            );
            self.reset_slice_cameras();
        }

        let data = Arc::new(data);

        let id = self.allocate_file_id(explicit_id);

        {
            let mut renderer = rs.renderer.write();
            if renderer.callback_resources.get::<MeshResources>().is_none() {
                let mr = MeshResources::new(&rs.device, rs.target_format);
                renderer.callback_resources.insert(mr);
            }
            if renderer
                .callback_resources
                .get::<GlyphResources>()
                .is_none()
            {
                let gr = GlyphResources::new(&rs.device, rs.target_format);
                renderer.callback_resources.insert(gr);
            }
        }

        if is_first {
            self.viewport.slice_world_offsets = [
                self.viewport.volume_center.z,
                self.viewport.volume_center.y,
                self.viewport.volume_center.x,
            ];
        }

        let max_streamlines = data.nb_streamlines.min(30_000);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file.trx".to_string());

        let trx = LoadedTrx {
            id,
            name,
            path: path.clone(),
            data,
            backing: Some(backing),
            import_warnings: warnings,
        };

        self.scene.trx_files.push(trx);
        if register_workflow_asset {
            let asset = WorkflowAssetDocument::Streamlines {
                id,
                path: path.clone(),
                imported,
            };
            self.register_workflow_asset(asset, true, Some(max_streamlines));
        }
        self.error_msg = None;
        self.status_msg = None;
    }

    pub(super) fn apply_loaded_nifti(
        &mut self,
        path: PathBuf,
        vol: NiftiVolume,
        rs: &egui_wgpu::RenderState,
    ) {
        self.apply_loaded_nifti_with_options(path, vol, rs, None, true);
    }

    pub(super) fn apply_loaded_nifti_with_options(
        &mut self,
        path: PathBuf,
        vol: NiftiVolume,
        rs: &egui_wgpu::RenderState,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        let first_nifti = self.scene.nifti_files.is_empty();
        let slice_indices = [vol.dims[2] / 2, vol.dims[1] / 2, vol.dims[0] / 2];
        let is_first = self.scene.nifti_files.is_empty()
            && self.scene.trx_files.is_empty()
            && self.scene.gifti_surfaces.is_empty();
        if is_first {
            self.viewport.volume_center = vol.voxel_to_world(Vec3::new(
                vol.dims[0] as f32 / 2.0,
                vol.dims[1] as f32 / 2.0,
                vol.dims[2] as f32 / 2.0,
            ));
            self.viewport.volume_extent = (vol.voxel_to_world(Vec3::new(
                vol.dims[0] as f32,
                vol.dims[1] as f32,
                vol.dims[2] as f32,
            )) - vol.voxel_to_world(Vec3::ZERO))
            .length();
            self.viewport.camera_3d = OrbitCamera::new(
                self.viewport.volume_center,
                self.viewport.volume_extent * 0.8,
            );
        }

        let slice_resources = SliceResources::new(&rs.device, &rs.queue, rs.target_format, &vol);
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Axial,
            self.viewport.slice_indices[0],
            &vol,
        );
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Coronal,
            self.viewport.slice_indices[1],
            &vol,
        );
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Sagittal,
            self.viewport.slice_indices[2],
            &vol,
        );

        let id = self.allocate_file_id(explicit_id);

        {
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                all.entries.push((id, slice_resources));
            } else {
                renderer.callback_resources.insert(AllSliceResources {
                    entries: vec![(id, slice_resources)],
                });
            }
        }

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "volume.nii".to_string());
        self.scene.nifti_files.push(LoadedNifti {
            id,
            name,
            volume: vol,
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            z_order: self.scene.nifti_files.len() as i32,
            window_center: 0.5,
            window_width: 1.0,
            visible: true,
        });
        if register_workflow_asset {
            self.register_workflow_asset(
                WorkflowAssetDocument::Volume {
                    id,
                    path: path.clone(),
                },
                true,
                None,
            );
        }
        if first_nifti {
            self.viewport.slice_indices = slice_indices;
            self.reset_slice_view();
        } else {
            self.viewport.slices_dirty = false;
        }
        self.error_msg = None;
        self.status_msg = None;
    }

    pub(super) fn apply_loaded_gifti_surface(
        &mut self,
        path: PathBuf,
        surface: GiftiSurfaceData,
        rs: &egui_wgpu::RenderState,
    ) {
        self.apply_loaded_gifti_surface_with_options(path, surface, rs, None, true);
    }

    pub(super) fn apply_loaded_gifti_surface_with_options(
        &mut self,
        path: PathBuf,
        surface: GiftiSurfaceData,
        rs: &egui_wgpu::RenderState,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        let first_scene_asset = self.scene.trx_files.is_empty()
            && self.scene.nifti_files.is_empty()
            && self.scene.gifti_surfaces.is_empty()
            && self.scene.parcellations.is_empty();
        let id = self.allocate_file_id(explicit_id);
        let mut renderer = rs.renderer.write();
        if renderer.callback_resources.get::<MeshResources>().is_none() {
            renderer
                .callback_resources
                .insert(MeshResources::new(&rs.device, rs.target_format));
        }
        let mesh_resources = renderer
            .callback_resources
            .get_mut::<MeshResources>()
            .expect("MeshResources inserted");
        mesh_resources.add_surface(id, &rs.device, &surface);
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
        self.scene.gifti_surfaces.push(LoadedGiftiSurface {
            id,
            name,
            path: path.clone(),
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
            projection_colormap: SurfaceColormap::Inferno,
            auto_range: true,
            range_min: 0.0,
            range_max: 1.0,
        });
        if register_workflow_asset {
            self.register_workflow_asset(
                WorkflowAssetDocument::Surface {
                    id,
                    path: path.clone(),
                },
                true,
                None,
            );
        }
        if let Some((center, extent)) = initial_surface_view {
            self.viewport.volume_center = center;
            self.viewport.volume_extent = extent;
            self.viewport.camera_3d = OrbitCamera::new(center, extent * 0.8);
            self.reset_slice_cameras();
            self.viewport.slice_world_offsets = [center.z, center.y, center.x];
        }
        self.error_msg = None;
        self.status_msg = None;
    }

    pub(super) fn apply_loaded_parcellation(
        &mut self,
        path: PathBuf,
        source: LoadedParcellationSource,
    ) {
        self.apply_loaded_parcellation_with_options(path, source, None, true);
    }

    pub(super) fn apply_loaded_parcellation_with_options(
        &mut self,
        path: PathBuf,
        source: LoadedParcellationSource,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        let id = self.allocate_file_id(explicit_id);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "parcellation.nii.gz".to_string());
        self.scene.parcellations.push(LoadedParcellation {
            asset: ParcellationAsset {
                id,
                name,
                path: path.clone(),
                data: Arc::new(source.data),
                label_table_path: source.label_table_path.clone(),
                visible: true,
            },
        });
        if register_workflow_asset {
            self.register_workflow_asset(
                WorkflowAssetDocument::Parcellation {
                    id,
                    path,
                    label_table_path: source.label_table_path,
                },
                true,
                None,
            );
        }
        self.error_msg = None;
        self.status_msg = None;
    }

    pub(super) fn reset_slice_cameras(&mut self) {
        let half_extents = self
            .scene
            .nifti_files
            .first()
            .map(|n| n.volume.slice_half_extents())
            .unwrap_or([self.viewport.volume_extent * 0.5; 3]);
        self.viewport.slice_cameras = [
            OrthoSliceCamera::new(
                SliceAxis::Axial,
                self.viewport.volume_center,
                half_extents[0] * 2.0,
            ),
            OrthoSliceCamera::new(
                SliceAxis::Coronal,
                self.viewport.volume_center,
                half_extents[1] * 2.0,
            ),
            OrthoSliceCamera::new(
                SliceAxis::Sagittal,
                self.viewport.volume_center,
                half_extents[2] * 2.0,
            ),
        ];
    }

    pub(crate) fn reset_slice_view(&mut self) {
        let Some(nf) = self.scene.nifti_files.first() else {
            return;
        };
        let vol = &nf.volume;
        let world_center = vol.voxel_to_world(Vec3::new(
            vol.dims[0] as f32 / 2.0,
            vol.dims[1] as f32 / 2.0,
            vol.dims[2] as f32 / 2.0,
        ));
        let half_extents = vol.slice_half_extents();
        self.viewport.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, world_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Coronal, world_center, half_extents[1] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Sagittal, world_center, half_extents[2] * 2.0),
        ];
        self.viewport.slice_world_offsets = [world_center.z, world_center.y, world_center.x];
        self.viewport.slices_dirty = true;
    }

    pub(crate) fn reset_slice_view_to_boundary_field(&mut self, field: &BoundaryContactField) {
        let size = Vec3::new(
            field.grid.dims[0] as f32,
            field.grid.dims[1] as f32,
            field.grid.dims[2] as f32,
        ) * field.grid.voxel_size_mm;
        let center = field.grid.origin_ras + 0.5 * size;
        let axial_extent = size.x.max(size.y);
        let coronal_extent = size.x.max(size.z);
        let sagittal_extent = size.y.max(size.z);

        self.viewport.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, center, axial_extent),
            OrthoSliceCamera::new(SliceAxis::Coronal, center, coronal_extent),
            OrthoSliceCamera::new(SliceAxis::Sagittal, center, sagittal_extent),
        ];
        self.viewport.slice_world_offsets = [center.z, center.y, center.x];
    }
}

fn create_merged_streamline_source(
    rows: &[super::state::MergeStreamlineRowState],
    output_path: &std::path::Path,
    options: &ConcatenateOptions,
) -> Result<LoadedStreamlineSource, String> {
    let tempdir = tempfile::TempDir::new().map_err(|err| err.to_string())?;
    let mut owned_inputs = Vec::new();
    for (idx, row) in rows.iter().enumerate() {
        let Some(path) = row.source_path.as_ref() else {
            continue;
        };
        let format = row
            .detected_format
            .or_else(|| trx_rs::detect_format(path).ok())
            .ok_or_else(|| format!("Unsupported streamline input: {}", path.display()))?;
        let any = match format {
            Format::Trx => AnyTrxFile::load(path).map_err(|err| err.to_string())?,
            Format::Trk => {
                return Err(format!(
                    "TrackVis input is not accepted for merge here; convert {} to .trx first",
                    path.display()
                ));
            }
            Format::Tck | Format::Vtk | Format::TinyTrack => {
                let header = row
                    .reference_path
                    .as_deref()
                    .map(header_from_reference)
                    .transpose()
                    .map_err(|err| err.to_string())?;
                let tractogram = trx_rs::read_tractogram(
                    path,
                    &ConversionOptions {
                        header,
                        vtk_coordinate_mode: row.vtk_coordinate_mode,
                        ..ConversionOptions::default()
                    },
                )
                .map_err(|err| err.to_string())?;
                let temp_path = tempdir.path().join(format!("merge_input_{idx}.trx"));
                trx_rs::write_tractogram(&temp_path, &tractogram, &ConversionOptions::default())
                    .map_err(|err| err.to_string())?;
                AnyTrxFile::load(&temp_path).map_err(|err| err.to_string())?
            }
        };
        owned_inputs.push(any);
    }

    if owned_inputs.len() < 2 {
        return Err("Choose at least two supported streamline inputs.".to_string());
    }

    let refs: Vec<&AnyTrxFile> = owned_inputs.iter().collect();
    let merged = concatenate_any_trx(&refs, options).map_err(|err| err.to_string())?;
    merged.save(output_path).map_err(|err| err.to_string())?;
    let loaded = AnyTrxFile::load(output_path).map_err(|err| err.to_string())?;
    super::TrxVizApp::loaded_streamline_source_from_any(loaded)
}

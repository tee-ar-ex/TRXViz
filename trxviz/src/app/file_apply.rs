use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use trxviz_core::asset_loader::LoadedAsset;
use trxviz_core::data::gifti_data::GiftiSurfaceData;
use trxviz_core::data::loaded_files::{
    FileId, LoadedCifti, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use trxviz_core::data::odx_data::OdxScene;
use trxviz_core::data::orientation_field::BoundaryContactField;
use trxviz_core::renderer::background_renderer::BackgroundResources;
use trxviz_core::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use trxviz_core::renderer::glyph_renderer::GlyphResources;
use trxviz_core::renderer::mesh_renderer::{MeshResources, SurfaceColormap};
use trxviz_core::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};

use crate::app::callbacks::OdxFixelResources;
use crate::app::state::{LoadedParcellationSource, LoadedStreamlineSource};
use crate::app::workflow::{self, LoadedParcellation, ParcellationAsset, WorkflowAssetDocument};

impl super::TrxVizApp {
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
            self.workflow.document.selection = self.workflow.selection;
            self.workflow.graph_focus_request = Some(egui::Rect::from_min_max(
                egui::pos2(branch.bounds.min.x, branch.bounds.min.y),
                egui::pos2(branch.bounds.max.x, branch.bounds.max.y),
            ));
            self.rebuild_workflow_editor_from_document();
            self.mark_workflow_semantic_edit(0.0);
        }
    }

    pub(super) fn apply_loaded_asset(
        &mut self,
        path: PathBuf,
        asset: LoadedAsset,
        rs: &egui_wgpu::RenderState,
    ) {
        match asset {
            LoadedAsset::Streamlines(source) => self.apply_loaded_trx(path, source, rs),
            LoadedAsset::Volume(volume) => self.apply_loaded_nifti(path, volume, rs),
            LoadedAsset::Cifti(cifti) => self.apply_loaded_cifti(path, cifti),
            LoadedAsset::Surface(surface) => self.apply_loaded_gifti_surface(path, surface, rs),
            LoadedAsset::Parcellation(source) => self.apply_loaded_parcellation(path, source),
            LoadedAsset::Odx(scene) => self.apply_loaded_odx(path, scene, rs),
        }
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
        self.viewport
            .set_volume_bounds(data.center(), data.extent());
        if is_first {
            *self.viewport.camera_3d_mut() = OrbitCamera::new(
                self.viewport.volume_center(),
                self.viewport.volume_extent() * 0.8,
            );
            self.reset_slice_cameras();
        }

        let data = Arc::new(data);
        let id = self.allocate_file_id(explicit_id);

        {
            let mut renderer = rs.renderer.write();
            if renderer
                .callback_resources
                .get::<BackgroundResources>()
                .is_none()
            {
                renderer
                    .callback_resources
                    .insert(BackgroundResources::new(&rs.device, rs.target_format));
            }
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
            self.viewport.set_slice_world_offsets([
                self.viewport.volume_center().z,
                self.viewport.volume_center().y,
                self.viewport.volume_center().x,
            ]);
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
        vol: trxviz_core::data::nifti_data::NiftiVolume,
        rs: &egui_wgpu::RenderState,
    ) {
        self.apply_loaded_nifti_with_options(path, vol, rs, None, true);
    }

    pub(super) fn apply_loaded_nifti_with_options(
        &mut self,
        path: PathBuf,
        vol: trxviz_core::data::nifti_data::NiftiVolume,
        rs: &egui_wgpu::RenderState,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        // Clear any prior error before deciding whether to set a new
        // one for this drop.
        self.error_msg = None;

        // Workflow-produced in-memory volumes (CIFTI subcortical, pyAFQ
        // probmap, ODX DPV) and any loaded ODX scene should also count
        // as "scene already has content" so dropping a NIfTI on top is
        // treated as additive — don't re-anchor the camera or jump the
        // slice plane away from what the user is currently looking at.
        let workflow_has_volumes = !self
            .workflow
            .runtime
            .scene_plan
            .volume_draws
            .is_empty();
        let has_odx = self.scene.odx_scene.is_some();
        let first_nifti =
            self.scene.nifti_files.is_empty() && !workflow_has_volumes && !has_odx;
        let slice_indices = [vol.dims[2] / 2, vol.dims[1] / 2, vol.dims[0] / 2];
        let is_first = self.scene.nifti_files.is_empty()
            && self.scene.trx_files.is_empty()
            && self.scene.gifti_surfaces.is_empty()
            && !workflow_has_volumes
            && !has_odx;

        // Warn loudly if the new NIfTI's RAS bounding box doesn't
        // overlap any existing volume. Without this, the dropped quad
        // is silently rendered far from the camera and the user has
        // no idea why "the slice didn't show up".
        if !first_nifti {
            let new_box = ras_aabb(vol.dims, vol.voxel_to_ras);
            let mut overlaps = false;
            for nf in &self.scene.nifti_files {
                if aabb_overlap(new_box, ras_aabb(nf.volume.dims, nf.volume.voxel_to_ras)) {
                    overlaps = true;
                    break;
                }
            }
            if !overlaps {
                if let Some(odx) = &self.scene.odx_scene {
                    let dims = odx.dimensions();
                    let dims = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
                    if aabb_overlap(new_box, ras_aabb(dims, odx.voxel_to_ras())) {
                        overlaps = true;
                    }
                }
            }
            if !overlaps {
                for draw in &self.workflow.runtime.scene_plan.volume_draws {
                    if let trxviz_core::workflow::VolumeBacking::InMemory { scalars, .. } =
                        &draw.source
                        && aabb_overlap(
                            new_box,
                            ras_aabb(scalars.dims, scalars.voxel_to_ras),
                        )
                    {
                        overlaps = true;
                        break;
                    }
                }
            }
            if !overlaps {
                self.error_msg = Some(format!(
                    "Dropped volume '{}' has no RAS overlap with the existing \
                     scene — its slice quad will render far from the camera. \
                     Check that the volumes are in the same coordinate space.",
                    path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "(unnamed)".into())
                ));
            }
        }
        if is_first {
            let volume_center = vol.voxel_to_world(Vec3::new(
                vol.dims[0] as f32 / 2.0,
                vol.dims[1] as f32 / 2.0,
                vol.dims[2] as f32 / 2.0,
            ));
            let volume_extent = (vol.voxel_to_world(Vec3::new(
                vol.dims[0] as f32,
                vol.dims[1] as f32,
                vol.dims[2] as f32,
            )) - vol.voxel_to_world(Vec3::ZERO))
            .length();
            self.viewport
                .set_volume_bounds(volume_center, volume_extent);
            *self.viewport.camera_3d_mut() = OrbitCamera::new(
                self.viewport.volume_center(),
                self.viewport.volume_extent() * 0.8,
            );
        }

        let slice_resources = SliceResources::new(&rs.device, &rs.queue, rs.target_format, &vol);
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Axial,
            self.viewport.slice_index(0),
            &vol,
        );
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Coronal,
            self.viewport.slice_index(1),
            &vol,
        );
        slice_resources.update_slice(
            &rs.queue,
            SliceAxis::Sagittal,
            self.viewport.slice_index(2),
            &vol,
        );

        let id = self.allocate_file_id(explicit_id);

        {
            use trxviz_core::renderer::slice_renderer::SliceResourceKind;
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                all.entries
                    .push((id, SliceResourceKind::Scalar(slice_resources)));
            } else {
                renderer.callback_resources.insert(AllSliceResources {
                    entries: vec![(id, SliceResourceKind::Scalar(slice_resources))],
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
            self.viewport.set_slice_indices(slice_indices);
            self.reset_slice_view();
        } else {
            self.viewport.clear_slices_dirty();
        }
        // Preserve the no-overlap warning set above; otherwise clear
        // any leftover error from a prior load.
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
        if renderer
            .callback_resources
            .get::<BackgroundResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(BackgroundResources::new(&rs.device, rs.target_format));
        }
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
        self.scene
            .gifti_surfaces
            .push(trxviz_core::scene::LoadedGiftiSurface {
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
            self.viewport.set_volume_bounds(center, extent);
            *self.viewport.camera_3d_mut() = OrbitCamera::new(center, extent * 0.8);
            self.reset_slice_cameras();
            self.viewport
                .set_slice_world_offsets([center.z, center.y, center.x]);
        }
        self.error_msg = None;
        self.status_msg = None;
    }

    pub(super) fn apply_loaded_cifti(&mut self, path: PathBuf, cifti: LoadedCifti) {
        self.apply_loaded_cifti_with_options(path, cifti, None, true);
    }

    pub(super) fn apply_loaded_cifti_with_options(
        &mut self,
        path: PathBuf,
        mut cifti: LoadedCifti,
        explicit_id: Option<FileId>,
        register_workflow_asset: bool,
    ) {
        let id = self.allocate_file_id(explicit_id.or(Some(cifti.id)).filter(|id| *id != 0));
        cifti.id = id;
        cifti.name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "cifti.nii".to_string());
        cifti.path = path.clone();
        let intent = cifti.data.intent;
        self.scene.cifti_files.push(cifti);
        if register_workflow_asset {
            self.register_workflow_asset(
                WorkflowAssetDocument::Cifti { id, path, intent },
                true,
                None,
            );
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

    pub(super) fn apply_loaded_odx(
        &mut self,
        path: PathBuf,
        scene: OdxScene,
        rs: &egui_wgpu::RenderState,
    ) {
        self.workflow.uploaded_odx_glyph_resource_key = None;
        self.workflow.uploaded_fixel_3d_fingerprint = 0;
        self.workflow.uploaded_fixel_2d_fingerprint = 0;

        let is_first = self.scene.trx_files.is_empty()
            && self.scene.nifti_files.is_empty()
            && self.scene.gifti_surfaces.is_empty();

        if !scene.centers_ras().is_empty() {
            let (mut min, mut max) = (
                Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            );
            for c in scene.centers_ras() {
                let p = Vec3::from_array(*c);
                min = min.min(p);
                max = max.max(p);
            }
            let center = (min + max) * 0.5;
            let extent = (max - min).length().max(1.0);
            self.viewport.set_volume_bounds(center, extent);
            if is_first {
                *self.viewport.camera_3d_mut() = OrbitCamera::new(center, extent * 0.8);
                self.reset_slice_cameras();
                self.viewport
                    .set_slice_world_offsets([center.z, center.y, center.x]);
            }
        }

        let dims = scene.dimensions();
        if is_first && self.scene.nifti_files.is_empty() {
            let mut mid = [
                (dims[2] / 2) as usize,
                (dims[1] / 2) as usize,
                (dims[0] / 2) as usize,
            ];
            let lookup = scene.ijk_lookup();
            if !lookup.is_empty() {
                for axis in 0..3 {
                    let mut vals: Vec<u32> = lookup.iter().map(|ijk| ijk[axis]).collect();
                    vals.sort_unstable();
                    let median = vals[vals.len() / 2] as usize;
                    let slot = match axis {
                        2 => 0,
                        1 => 1,
                        _ => 2,
                    };
                    mid[slot] = median;
                }
            }
            self.viewport.set_slice_indices(mid);
            let affine = scene.voxel_to_ras();
            let w0 = affine.transform_point3(Vec3::new(0.0, 0.0, mid[0] as f32));
            let w1 = affine.transform_point3(Vec3::new(0.0, mid[1] as f32, 0.0));
            let w2 = affine.transform_point3(Vec3::new(mid[2] as f32, 0.0, 0.0));
            self.viewport.set_slice_world_offsets([w0.z, w1.y, w2.x]);
        }

        let scene = Arc::new(scene);
        if scene.has_glyph_field() {
            let current_axial = self.viewport.slice_index(0) as u32;
            if let Some(nonempty_axial) = scene.nearest_nonempty_slice(2, current_axial) {
                self.viewport.set_slice_index(0, nonempty_axial as usize);
                let w = scene.voxel_to_ras().transform_point3(Vec3::new(
                    0.0,
                    0.0,
                    nonempty_axial as f32,
                ));
                self.viewport.set_slice_world_offset(0, w.z);
            }
        }
        if scene.glyph_source_kind() == Some(trxviz_core::data::odx_data::OdxGlyphSourceKind::Odf)
            && let Some(rows_per_chunk) = scene
                .odf_rows_per_chunk(rs.device.limits().max_storage_buffer_binding_size as usize)
        {
            scene.prewarm_odf_slice_metadata(
                2,
                self.viewport.slice_index(0) as u32,
                rows_per_chunk,
            );
        }
        let fixel_instances = scene.all_fixels();

        {
            let mut renderer = rs.renderer.write();
            if renderer
                .callback_resources
                .get::<GlyphResources>()
                .is_none()
            {
                renderer
                    .callback_resources
                    .insert(GlyphResources::new(&rs.device, rs.target_format));
            }
            if renderer
                .callback_resources
                .get::<OdxFixelResources>()
                .is_none()
            {
                renderer
                    .callback_resources
                    .insert(OdxFixelResources::new(&rs.device, rs.target_format));
            }
            if let Some(fr) = renderer.callback_resources.get_mut::<OdxFixelResources>() {
                fr.resources_3d.set_fixels(&rs.device, &fixel_instances);
                fr.resources_2d.set_fixels(&rs.device, &fixel_instances);
            }
        }

        self.scene.odx_scene = Some(scene.clone());
        {
            let mut renderer = rs.renderer.write();
            self.ensure_active_odx_glyph_resources(
                &mut renderer.callback_resources,
                &rs.device,
                &rs.queue,
            );
        }

        let display_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "ODX".to_string());

        let id = self.allocate_file_id(None);
        self.scene
            .odx_files
            .push(trxviz_core::data::loaded_files::LoadedOdx {
                id,
                name: display_name,
                path: path.clone(),
                scene: scene.clone(),
                warnings: scene.glyph_warnings().to_vec(),
                visible: true,
            });
        let show_fixel_3d_by_default = scene.glyph_source_kind().is_none();
        let dpf_names: Vec<String> = scene
            .dataset()
            .dpf_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let dpv_names: Vec<String> = scene.dpv_names().iter().map(|s| s.to_string()).collect();
        self.register_workflow_asset(WorkflowAssetDocument::Odx { id, path }, true, None);
        let mut workflow_changed = workflow::set_default_odx_fixel_3d_visibility(
            &mut self.workflow.document,
            id,
            show_fixel_3d_by_default,
        );
        workflow_changed |=
            workflow::set_default_odx_fixel_dpf(&mut self.workflow.document, id, &dpf_names);
        workflow_changed |=
            workflow::set_default_odx_volume_dpv(&mut self.workflow.document, id, &dpv_names);
        if workflow_changed {
            self.rebuild_workflow_editor_from_document();
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
            .unwrap_or([self.viewport.volume_extent() * 0.5; 3]);
        *self.viewport.slice_cameras_mut() = [
            OrthoSliceCamera::new(
                SliceAxis::Axial,
                self.viewport.volume_center(),
                half_extents[0] * 2.0,
            ),
            OrthoSliceCamera::new(
                SliceAxis::Coronal,
                self.viewport.volume_center(),
                half_extents[1] * 2.0,
            ),
            OrthoSliceCamera::new(
                SliceAxis::Sagittal,
                self.viewport.volume_center(),
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
        *self.viewport.slice_cameras_mut() = [
            OrthoSliceCamera::new(SliceAxis::Axial, world_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Coronal, world_center, half_extents[1] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Sagittal, world_center, half_extents[2] * 2.0),
        ];
        self.viewport
            .set_slice_world_offsets([world_center.z, world_center.y, world_center.x]);
        self.viewport.mark_slices_dirty();
    }

    pub(crate) fn reset_slice_view_to_boundary_field(&mut self, field: &BoundaryContactField) {
        let size = Vec3::new(
            field.grid.dims[0] as f32,
            field.grid.dims[1] as f32,
            field.grid.dims[2] as f32,
        ) * field.grid.voxel_size_mm.0;
        let center = field.grid.origin_ras + 0.5 * size;
        let axial_extent = size.x.max(size.y);
        let coronal_extent = size.x.max(size.z);
        let sagittal_extent = size.y.max(size.z);

        *self.viewport.slice_cameras_mut() = [
            OrthoSliceCamera::new(SliceAxis::Axial, center, axial_extent),
            OrthoSliceCamera::new(SliceAxis::Coronal, center, coronal_extent),
            OrthoSliceCamera::new(SliceAxis::Sagittal, center, sagittal_extent),
        ];
        self.viewport
            .set_slice_world_offsets([center.z, center.y, center.x]);
    }
}

/// RAS-space axis-aligned bounding box of a voxel grid. Walks the 8
/// corners of the (i,j,k) cube through the voxel-to-RAS affine and
/// takes the min/max — handles arbitrary orientation, not just
/// axis-aligned affines.
fn ras_aabb(dims: [usize; 3], voxel_to_ras: glam::Mat4) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &x in &[0.0, dims[0] as f32] {
        for &y in &[0.0, dims[1] as f32] {
            for &z in &[0.0, dims[2] as f32] {
                let p = voxel_to_ras.transform_point3(Vec3::new(x, y, z));
                min = min.min(p);
                max = max.max(p);
            }
        }
    }
    (min, max)
}

fn aabb_overlap(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> bool {
    a.0.x <= b.1.x
        && a.1.x >= b.0.x
        && a.0.y <= b.1.y
        && a.1.y >= b.0.y
        && a.0.z <= b.1.z
        && a.1.z >= b.0.z
}

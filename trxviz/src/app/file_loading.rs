use std::path::{Path, PathBuf};
use std::sync::Arc;

use trx_rs::{
    AnyTrxFile, ConcatenateOptions, ConversionOptions, DType, Format, concatenate_any_trx,
    header_from_reference,
};
use trxviz_core::asset_loader::{
    LoadedAsset, is_dsistudio_odx_input, load_asset, load_odx_with_reference_affine,
};
use trxviz_core::data::loaded_files::{FileId, StreamlineBacking};
use trxviz_core::data::trx_data::TrxGpuData;
use trxviz_core::scene::LoadedStreamlineSource;

use super::state::{ImportDialogState, MergeStreamlinesDialogState, WorkerMessage};

impl super::TrxVizApp {
    pub(super) fn loaded_streamline_source_from_any(
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

    pub(super) fn allocate_file_id(&mut self, explicit_id: Option<FileId>) -> FileId {
        if let Some(id) = explicit_id {
            self.scene.next_file_id = self.scene.next_file_id.max(id + 1);
            id
        } else {
            let id = self.scene.next_file_id;
            self.scene.next_file_id += 1;
            id
        }
    }

    fn begin_load_registered_asset(&mut self, path: PathBuf, fallback_label: &str) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| fallback_label.to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = load_asset(&path).map_err(|err| err.to_string());
            let _ = tx.send(WorkerMessage::AssetLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_trx(&mut self, path: PathBuf) {
        self.begin_load_registered_asset(path, "Loading streamlines");
    }

    pub(super) fn begin_load_nifti(&mut self, path: PathBuf) {
        self.begin_load_registered_asset(path, "Loading NIfTI");
    }

    pub(super) fn begin_load_cifti(&mut self, path: PathBuf) {
        self.begin_load_registered_asset(path, "Loading CIFTI");
    }

    pub(super) fn begin_load_gifti_surface(&mut self, path: PathBuf) {
        self.begin_load_registered_asset(path, "Loading GIFTI");
    }

    pub(super) fn begin_load_parcellation(&mut self, path: PathBuf) {
        self.begin_load_registered_asset(path, "Loading parcellation");
    }

    pub(super) fn begin_load_odx(&mut self, path: PathBuf, reference_affine_path: Option<PathBuf>) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading ODX".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = load_odx_with_reference_affine(&path, reference_affine_path.as_deref())
                .map(LoadedAsset::Odx)
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::AssetLoaded {
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
}

pub(super) fn needs_reference_affine_recovery(path: &Path, err: &str) -> bool {
    is_dsistudio_odx_input(path)
        && err.contains("DSI Studio file has no spatial affine ('trans' field)")
}

fn create_merged_streamline_source(
    rows: &[super::state::MergeStreamlineRowState],
    output_path: &Path,
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

#[cfg(test)]
mod tests {
    use super::needs_reference_affine_recovery;
    use std::path::Path;

    #[test]
    fn missing_trans_error_for_fibgz_requests_recovery() {
        assert!(needs_reference_affine_recovery(
            Path::new("sample.fib.gz"),
            "DSI Studio file has no spatial affine ('trans' field). Convert it first"
        ));
    }

    #[test]
    fn unrelated_error_does_not_request_recovery() {
        assert!(!needs_reference_affine_recovery(
            Path::new("sample.fib.gz"),
            "some other load error"
        ));
    }

    #[test]
    fn non_dsistudio_path_does_not_request_recovery() {
        assert!(!needs_reference_affine_recovery(
            Path::new("sample.odx"),
            "DSI Studio file has no spatial affine ('trans' field)"
        ));
    }
}

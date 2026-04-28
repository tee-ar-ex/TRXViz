use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use trx_rs::{AnyTrxFile, ConversionOptions, Format};

use crate::data::cifti::LoadedCifti as LoadedCiftiData;
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{LoadedCifti, StreamlineBacking};
use crate::data::nifti_data::NiftiVolume;
use crate::data::odx_data::OdxScene;
use crate::data::parcellation_data::{ParcellationVolume, guess_label_table_path};
use crate::data::trx_data::TrxGpuData;
use crate::scene::{
    LoadedParcellationSource, LoadedStreamlineSource, direct_streamline_import_warnings,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    Streamlines,
    ImportedStreamlines,
    Volume,
    Cifti,
    Surface,
    Parcellation,
    Odx,
    Unsupported,
}

pub enum LoadedAsset {
    Streamlines(LoadedStreamlineSource),
    Volume(NiftiVolume),
    Cifti(LoadedCifti),
    Surface(GiftiSurfaceData),
    Parcellation(LoadedParcellationSource),
    Odx(OdxScene),
}

pub trait AssetLoader: Sized + Send + 'static {
    const EXTENSIONS: &'static [&'static str];

    fn load(path: &Path) -> Result<Self>;
    fn kind() -> AssetKind;
}

impl AssetLoader for LoadedStreamlineSource {
    const EXTENSIONS: &'static [&'static str] = &["trx", "trk", "tck", "vtk", "tt", "trk.gz"];

    fn load(path: &Path) -> Result<Self> {
        match trx_rs::detect_format(path).map_err(|err| anyhow!(err.to_string()))? {
            Format::Trx => {
                let any = AnyTrxFile::load(path).map_err(|err| anyhow!(err.to_string()))?;
                load_streamline_source_from_any(any)
            }
            Format::Trk | Format::Tck | Format::Vtk | Format::TinyTrack => {
                let options = ConversionOptions::default();
                let warnings = direct_streamline_import_warnings(path, &options);
                let tractogram = trx_rs::read_tractogram(path, &options)
                    .map_err(|err| anyhow!(err.to_string()))?;
                let data = TrxGpuData::from_tractogram(&tractogram)
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(LoadedStreamlineSource {
                    data,
                    backing: StreamlineBacking::Imported(Arc::new(tractogram)),
                    warnings,
                })
            }
        }
    }

    fn kind() -> AssetKind {
        AssetKind::Streamlines
    }
}

impl AssetLoader for NiftiVolume {
    const EXTENSIONS: &'static [&'static str] = &["nii", "nii.gz"];

    fn load(path: &Path) -> Result<Self> {
        NiftiVolume::load(path).map_err(|err| anyhow!(err.to_string()))
    }

    fn kind() -> AssetKind {
        AssetKind::Volume
    }
}

impl AssetLoader for LoadedCifti {
    const EXTENSIONS: &'static [&'static str] =
        &["dscalar.nii", "dlabel.nii", "dtseries.nii", "pscalar.nii"];

    fn load(path: &Path) -> Result<Self> {
        let data = LoadedCiftiData::load(path).map_err(|err| anyhow!(err.to_string()))?;
        Ok(LoadedCifti {
            id: 0,
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "cifti".to_string()),
            path: path.to_path_buf(),
            data: Arc::new(data),
            visible: true,
        })
    }

    fn kind() -> AssetKind {
        AssetKind::Cifti
    }
}

impl AssetLoader for GiftiSurfaceData {
    const EXTENSIONS: &'static [&'static str] = &["gii", "gifti"];

    fn load(path: &Path) -> Result<Self> {
        GiftiSurfaceData::load(path).map_err(|err| anyhow!(err.to_string()))
    }

    fn kind() -> AssetKind {
        AssetKind::Surface
    }
}

impl AssetLoader for LoadedParcellationSource {
    const EXTENSIONS: &'static [&'static str] = &["nii", "nii.gz"];

    fn load(path: &Path) -> Result<Self> {
        let label_table_path = guess_label_table_path(path);
        ParcellationVolume::load(path, label_table_path.as_deref())
            .map(|data| LoadedParcellationSource {
                data,
                label_table_path,
            })
            .map_err(|err| anyhow!(err.to_string()))
    }

    fn kind() -> AssetKind {
        AssetKind::Parcellation
    }
}

impl AssetLoader for OdxScene {
    const EXTENSIONS: &'static [&'static str] =
        &["odx", "odxd", "fib.gz", "fz", "pam5", "mif", "mif.gz"];

    fn load(path: &Path) -> Result<Self> {
        load_odx_with_reference_affine(path, None)
    }

    fn kind() -> AssetKind {
        AssetKind::Odx
    }
}

pub fn load_asset(path: &Path) -> Result<LoadedAsset> {
    match detect_asset_kind(path) {
        AssetKind::Streamlines | AssetKind::ImportedStreamlines => Ok(LoadedAsset::Streamlines(
            LoadedStreamlineSource::load(path)?,
        )),
        AssetKind::Volume => Ok(LoadedAsset::Volume(NiftiVolume::load(path)?)),
        AssetKind::Cifti => Ok(LoadedAsset::Cifti(LoadedCifti::load(path)?)),
        AssetKind::Surface => Ok(LoadedAsset::Surface(GiftiSurfaceData::load(path)?)),
        AssetKind::Parcellation => Ok(LoadedAsset::Parcellation(LoadedParcellationSource::load(
            path,
        )?)),
        AssetKind::Odx => Ok(LoadedAsset::Odx(OdxScene::load(path)?)),
        AssetKind::Unsupported => {
            bail!("Unsupported asset type for {}", path.display());
        }
    }
}

pub fn detect_asset_kind(path: &Path) -> AssetKind {
    if is_odx_compatible(path) {
        return AssetKind::Odx;
    }

    match trx_rs::detect_format(path) {
        Ok(Format::Trx) => AssetKind::Streamlines,
        Ok(Format::Trk | Format::Tck | Format::Vtk | Format::TinyTrack) => {
            AssetKind::ImportedStreamlines
        }
        Err(_) => classify_non_streamline_path(path),
    }
}

pub fn load_odx_with_reference_affine(
    path: &Path,
    reference_affine_path: Option<&Path>,
) -> Result<OdxScene> {
    use odx_rs::cli_support::{LoadDatasetOptions, load_dataset};

    if let Ok(scene) = OdxScene::open(path) {
        return Ok(scene);
    }

    let options = LoadDatasetOptions {
        sh_path: None,
        fixel_dir: None,
        reference_affine: reference_affine_path,
        mapmri_tensor_path: None,
        mapmri_uvec_path: None,
        preserve_nifti_affine: false,
    };
    let (dataset, _format) = load_dataset(path, options).map_err(|err| anyhow!(err.to_string()))?;
    OdxScene::from_dataset(dataset).map_err(|err| anyhow!(err.to_string()))
}

pub fn is_dsistudio_odx_input(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    file_name.ends_with(".fib.gz") || file_name.ends_with(".fz")
}

fn load_streamline_source_from_any(any: AnyTrxFile) -> Result<LoadedStreamlineSource> {
    TrxGpuData::from_any_trx(&any)
        .map(|data| LoadedStreamlineSource {
            data,
            backing: StreamlineBacking::Native(Arc::new(any)),
            warnings: Vec::new(),
        })
        .map_err(|err| anyhow!(err.to_string()))
}

fn classify_non_streamline_path(path: &Path) -> AssetKind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "gz" if stem.ends_with(".nii") => classify_nifti_like(path, &file_name),
        "nii" => classify_nifti_like(path, &file_name),
        "gii" | "gifti" => AssetKind::Surface,
        _ => AssetKind::Unsupported,
    }
}

fn classify_nifti_like(path: &Path, file_name: &str) -> AssetKind {
    if [
        ".dscalar.nii",
        ".dlabel.nii",
        ".dtseries.nii",
        ".pscalar.nii",
    ]
    .iter()
    .any(|suffix| file_name.ends_with(suffix))
    {
        return AssetKind::Cifti;
    }
    if guess_label_table_path(path).is_some()
        || ["parcel", "parc", "atlas", "label", "seg", "segmentation"]
            .iter()
            .any(|needle| file_name.contains(needle))
    {
        AssetKind::Parcellation
    } else {
        AssetKind::Volume
    }
}

fn is_odx_compatible(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let lower = file_name.to_ascii_lowercase();

    if path.is_dir() {
        if path.join("header.json").exists() {
            return true;
        }
        let has_index = path.join("index.mif").exists() || path.join("index.nii.gz").exists();
        let has_dirs =
            path.join("directions.mif").exists() || path.join("directions.nii.gz").exists();
        return has_index && has_dirs;
    }

    lower.ends_with(".odx")
        || lower.ends_with(".odxd")
        || lower.ends_with(".fib.gz")
        || lower.ends_with(".fz")
        || lower.ends_with(".pam5")
        || lower.ends_with(".mif")
        || lower.ends_with(".mif.gz")
}

#[cfg(test)]
mod tests {
    use super::{AssetKind, detect_asset_kind, is_dsistudio_odx_input};
    use std::path::Path;

    #[test]
    fn detects_cifti_compound_suffixes() {
        assert_eq!(
            detect_asset_kind(Path::new("map.dscalar.nii")),
            AssetKind::Cifti
        );
        assert_eq!(
            detect_asset_kind(Path::new("labels.dlabel.nii")),
            AssetKind::Cifti
        );
        assert_eq!(
            detect_asset_kind(Path::new("series.dtseries.nii")),
            AssetKind::Cifti
        );
    }

    #[test]
    fn detects_odx_compatible_extensions_before_streamline_logic() {
        assert_eq!(
            detect_asset_kind(Path::new("subject.fib.gz")),
            AssetKind::Odx
        );
        assert_eq!(detect_asset_kind(Path::new("subject.pam5")), AssetKind::Odx);
    }

    #[test]
    fn detects_streamline_vs_imported_streamline_paths() {
        assert_eq!(
            detect_asset_kind(Path::new("bundle.trx")),
            AssetKind::Streamlines
        );
        assert_eq!(
            detect_asset_kind(Path::new("bundle.trk.gz")),
            AssetKind::ImportedStreamlines
        );
    }

    #[test]
    fn detects_gifti_and_plain_nifti_paths() {
        assert_eq!(
            detect_asset_kind(Path::new("surface.surf.gii")),
            AssetKind::Surface
        );
        assert_eq!(
            detect_asset_kind(Path::new("bold.nii.gz")),
            AssetKind::Volume
        );
    }

    #[test]
    fn dsistudio_detection_matches_fib_and_fz() {
        assert!(is_dsistudio_odx_input(Path::new("sample.fib.gz")));
        assert!(is_dsistudio_odx_input(Path::new("sample.fz")));
        assert!(!is_dsistudio_odx_input(Path::new("sample.odx")));
    }
}

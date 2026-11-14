use std::path::PathBuf;
use std::sync::Arc;

use crate::data::cifti::LoadedCifti as LoadedCiftiData;
use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::TrxGpuData;
use trx_rs::{AnyTrxFile, Tractogram};

pub type FileId = usize;

#[derive(Clone)]
pub enum StreamlineBacking {
    Native(Arc<AnyTrxFile>),
    Imported(Arc<Tractogram>),
    Derived(Arc<Tractogram>),
}

/// All per-TRX-file state: data, rendering options, filters, bundle mesh, etc.
pub struct LoadedTrx {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<TrxGpuData>,
    pub backing: Option<StreamlineBacking>,
    pub import_warnings: Vec<String>,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum VolumeColormap {
    #[default]
    Grayscale,
    Hot,
    Cool,
    RedYellow,
    BlueLightblue,
    Inferno,
    Magma,
    Plasma,
    Viridis,
}

impl VolumeColormap {
    pub fn as_u32(&self) -> u32 {
        match self {
            VolumeColormap::Grayscale => 0,
            VolumeColormap::Hot => 1,
            VolumeColormap::Cool => 2,
            VolumeColormap::RedYellow => 3,
            VolumeColormap::BlueLightblue => 4,
            VolumeColormap::Inferno => 5,
            VolumeColormap::Magma => 6,
            VolumeColormap::Plasma => 7,
            VolumeColormap::Viridis => 8,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            VolumeColormap::Grayscale => "Grayscale",
            VolumeColormap::Hot => "Hot",
            VolumeColormap::Cool => "Cool",
            VolumeColormap::RedYellow => "Red-Yellow",
            VolumeColormap::BlueLightblue => "Blue-Lightblue",
            VolumeColormap::Inferno => "Inferno",
            VolumeColormap::Magma => "Magma",
            VolumeColormap::Plasma => "Plasma",
            VolumeColormap::Viridis => "Viridis",
        }
    }

    pub const ALL: &[VolumeColormap] = &[
        VolumeColormap::Grayscale,
        VolumeColormap::Hot,
        VolumeColormap::Cool,
        VolumeColormap::RedYellow,
        VolumeColormap::BlueLightblue,
        VolumeColormap::Inferno,
        VolumeColormap::Magma,
        VolumeColormap::Plasma,
        VolumeColormap::Viridis,
    ];
}

/// How a scalar volume is rendered in the 3D viewport.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum VolumeRender3DMode {
    /// Traditional orthogonal 2D slice quads in 3D space.
    #[default]
    Off,
    /// 3D Maximum Intensity Projection (MIP) ray-marched through the volume.
    Mip,
    /// 3D Front-to-Back Alpha-Composited Direct Volume Rendering (DVR) with ray-gradient edge cues.
    DirectVolume,
}

impl VolumeRender3DMode {
    pub fn as_u32(&self) -> u32 {
        match self {
            VolumeRender3DMode::Off => 0,
            VolumeRender3DMode::Mip => 1,
            VolumeRender3DMode::DirectVolume => 2,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            VolumeRender3DMode::Off => "2D Slices Only",
            VolumeRender3DMode::Mip => "3D MIP (Max Intensity)",
            VolumeRender3DMode::DirectVolume => "3D Volume (Direct Glow)",
        }
    }

    pub const ALL: &[VolumeRender3DMode] = &[
        VolumeRender3DMode::Off,
        VolumeRender3DMode::Mip,
        VolumeRender3DMode::DirectVolume,
    ];
}

#[allow(dead_code)]
pub struct LoadedNifti {
    pub id: FileId,
    pub name: String,
    pub volume: NiftiVolume,
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub z_order: i32,
    pub window_center: f32,
    pub window_width: f32,
    pub visible: bool,
}

pub struct LoadedCifti {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<LoadedCiftiData>,
    pub visible: bool,
}

/// Loaded ODX (ODF/fixel/DPV) file asset.
pub struct LoadedOdx {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub scene: Arc<crate::data::odx_data::OdxScene>,
    pub warnings: Vec<String>,
    pub visible: bool,
}

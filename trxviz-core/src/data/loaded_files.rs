use std::path::PathBuf;
use std::sync::Arc;

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VolumeColormap {
    Grayscale,
    Hot,
    Cool,
    RedYellow,
    BlueLightblue,
}

impl VolumeColormap {
    pub fn as_u32(&self) -> u32 {
        match self {
            VolumeColormap::Grayscale => 0,
            VolumeColormap::Hot => 1,
            VolumeColormap::Cool => 2,
            VolumeColormap::RedYellow => 3,
            VolumeColormap::BlueLightblue => 4,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            VolumeColormap::Grayscale => "Grayscale",
            VolumeColormap::Hot => "Hot",
            VolumeColormap::Cool => "Cool",
            VolumeColormap::RedYellow => "Red-Yellow",
            VolumeColormap::BlueLightblue => "Blue-Lightblue",
        }
    }

    pub const ALL: &[VolumeColormap] = &[
        VolumeColormap::Grayscale,
        VolumeColormap::Hot,
        VolumeColormap::Cool,
        VolumeColormap::RedYellow,
        VolumeColormap::BlueLightblue,
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

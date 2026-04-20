use std::path::Path;

use trx_rs::Format;
use trxviz_core::asset_loader::{AssetKind, detect_asset_kind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DroppedPathKind {
    OpenTrx,
    ImportTractogram(Format),
    OpenNifti,
    OpenCifti,
    OpenParcellation,
    OpenGifti,
    OpenOdx,
    Unsupported,
}

pub(super) fn classify_dropped_path(path: &Path) -> DroppedPathKind {
    match detect_asset_kind(path) {
        AssetKind::Odx => DroppedPathKind::OpenOdx,
        AssetKind::Streamlines => DroppedPathKind::OpenTrx,
        AssetKind::ImportedStreamlines => match trx_rs::detect_format(path) {
            Ok(format @ (Format::Trk | Format::Tck | Format::Vtk | Format::TinyTrack)) => {
                DroppedPathKind::ImportTractogram(format)
            }
            _ => DroppedPathKind::Unsupported,
        },
        AssetKind::Volume => DroppedPathKind::OpenNifti,
        AssetKind::Cifti => DroppedPathKind::OpenCifti,
        AssetKind::Surface => DroppedPathKind::OpenGifti,
        AssetKind::Parcellation => DroppedPathKind::OpenParcellation,
        AssetKind::Unsupported => DroppedPathKind::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::{DroppedPathKind, classify_dropped_path};
    use trx_rs::Format;

    #[test]
    fn trk_paths_are_classified_as_imports() {
        assert_eq!(
            classify_dropped_path(std::path::Path::new("sample.trk.gz")),
            DroppedPathKind::ImportTractogram(Format::Trk)
        );
    }
}

pub(super) fn tri_axis_value(p: glam::Vec3, axis_index: usize) -> f32 {
    match axis_index {
        0 => p.z,
        1 => p.y,
        _ => p.x,
    }
}

pub(super) fn intersect_edge_with_slice(
    p0: glam::Vec3,
    p1: glam::Vec3,
    axis_index: usize,
    slice_pos: f32,
    eps: f32,
) -> Option<glam::Vec3> {
    let c0 = tri_axis_value(p0, axis_index);
    let c1 = tri_axis_value(p1, axis_index);
    let d0 = c0 - slice_pos;
    let d1 = c1 - slice_pos;

    // Coplanar edge: skip to avoid degenerate full-triangle artifacts.
    if d0.abs() <= eps && d1.abs() <= eps {
        return None;
    }
    if d0.abs() <= eps {
        return Some(p0);
    }
    if d1.abs() <= eps {
        return Some(p1);
    }
    if d0 * d1 > 0.0 {
        return None;
    }
    let t = d0 / (d0 - d1);
    Some(p0 + (p1 - p0) * t)
}

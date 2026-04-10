use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use glam::{Mat4, Vec3, Vec4};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

#[derive(Clone, Debug)]
pub struct ParcelLabel {
    pub id: u32,
    pub name: String,
    pub color: [f32; 4],
}

#[derive(Clone)]
pub struct ParcellationVolume {
    pub labels: Vec<u32>,
    pub dims: [usize; 3],
    pub voxel_to_ras: Mat4,
    pub world_to_voxel: Mat4,
    pub label_table: BTreeMap<u32, ParcelLabel>,
}

impl ParcellationVolume {
    pub fn load(path: &Path, label_table_path: Option<&Path>) -> anyhow::Result<Self> {
        let obj = ReaderOptions::new().read_file(path)?;
        let header = obj.header();
        let voxel_to_ras = nifti_voxel_to_ras(header)?;
        let world_to_voxel = voxel_to_ras.inverse();

        let dims = [
            header.dim[1] as usize,
            header.dim[2] as usize,
            header.dim[3] as usize,
        ];

        let volume = obj.into_volume();
        let array = volume.into_ndarray::<i32>()?;
        let raw = array
            .as_slice_memory_order()
            .context("Parcellation array is not contiguous")?;
        let labels: Vec<u32> = raw.iter().map(|&value| value.max(0) as u32).collect();

        let mut label_table = if let Some(path) = label_table_path {
            parse_label_table(path)?
        } else {
            BTreeMap::new()
        };

        for label in labels.iter().copied().filter(|label| *label != 0) {
            label_table.entry(label).or_insert_with(|| ParcelLabel {
                id: label,
                name: format!("Label {label}"),
                color: generated_label_color(label),
            });
        }

        Ok(Self {
            labels,
            dims,
            voxel_to_ras,
            world_to_voxel,
            label_table,
        })
    }

    pub fn label_name(&self, label: u32) -> String {
        self.label_table
            .get(&label)
            .map(|entry| entry.name.clone())
            .unwrap_or_else(|| format!("Label {label}"))
    }

    pub fn label_color(&self, label: u32) -> [f32; 4] {
        self.label_table
            .get(&label)
            .map(|entry| entry.color)
            .unwrap_or_else(|| generated_label_color(label))
    }

    pub fn voxel_to_world(&self, voxel: Vec3) -> Vec3 {
        let world = self.voxel_to_ras * voxel.extend(1.0);
        Vec3::new(world.x, world.y, world.z)
    }

    pub fn sample_label_world(&self, world: Vec3) -> Option<u32> {
        let voxel = self.world_to_voxel * world.extend(1.0);
        let i = voxel.x.round() as isize;
        let j = voxel.y.round() as isize;
        let k = voxel.z.round() as isize;
        self.label_at(i, j, k)
    }

    pub fn streamline_hits_labels(&self, points: &[[f32; 3]], labels: &BTreeSet<u32>) -> bool {
        if labels.is_empty() {
            return false;
        }
        points.iter().any(|point| {
            self.sample_label_world(Vec3::from(*point))
                .is_some_and(|label| labels.contains(&label))
        })
    }

    pub fn streamline_avoids_labels(&self, points: &[[f32; 3]], labels: &BTreeSet<u32>) -> bool {
        !self.streamline_hits_labels(points, labels)
    }

    pub fn streamline_end_hits_labels(
        &self,
        points: &[[f32; 3]],
        labels: &BTreeSet<u32>,
        endpoint_count: usize,
    ) -> bool {
        if labels.is_empty() || points.is_empty() {
            return false;
        }
        let first = self
            .sample_label_world(Vec3::from(points[0]))
            .is_some_and(|label| labels.contains(&label));
        let last = self
            .sample_label_world(Vec3::from(*points.last().unwrap()))
            .is_some_and(|label| labels.contains(&label));
        match endpoint_count {
            0 => false,
            1 => first || last,
            _ => first && last,
        }
    }

    pub fn crop_streamline_inside(
        &self,
        points: &[[f32; 3]],
        labels: &BTreeSet<u32>,
    ) -> Vec<Vec<[f32; 3]>> {
        self.crop_streamline(points, labels, true)
    }

    pub fn crop_streamline_outside(
        &self,
        points: &[[f32; 3]],
        labels: &BTreeSet<u32>,
    ) -> Vec<Vec<[f32; 3]>> {
        self.crop_streamline(points, labels, false)
    }

    pub fn nearest_slice_index(
        &self,
        axis_index: usize,
        slice_pos: f32,
        world_center: Vec3,
    ) -> Option<usize> {
        let probe = match axis_index {
            0 => Vec3::new(world_center.x, world_center.y, slice_pos),
            1 => Vec3::new(world_center.x, slice_pos, world_center.z),
            _ => Vec3::new(slice_pos, world_center.y, world_center.z),
        };
        let voxel = self.world_to_voxel * probe.extend(1.0);
        let coord = match axis_index {
            0 => voxel.z,
            1 => voxel.y,
            _ => voxel.x,
        };
        let idx = coord.round() as isize;
        let max = match axis_index {
            0 => self.dims[2] as isize,
            1 => self.dims[1] as isize,
            _ => self.dims[0] as isize,
        };
        (0..max).contains(&idx).then_some(idx as usize)
    }

    pub fn slice_contour_segments(
        &self,
        axis_index: usize,
        slice_index: usize,
        labels: &BTreeSet<u32>,
    ) -> Vec<([Vec3; 2], [f32; 4])> {
        if labels.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        match axis_index {
            0 => {
                for j in 0..self.dims[1] {
                    for i in 0..self.dims[0] {
                        let label = self.label_at(i as isize, j as isize, slice_index as isize);
                        if !label.is_some_and(|value| labels.contains(&value)) {
                            continue;
                        }
                        let color = self.label_color(label.unwrap());
                        self.push_cell_edges_axial(slice_index, i, j, labels, color, &mut segments);
                    }
                }
            }
            1 => {
                for k in 0..self.dims[2] {
                    for i in 0..self.dims[0] {
                        let label = self.label_at(i as isize, slice_index as isize, k as isize);
                        if !label.is_some_and(|value| labels.contains(&value)) {
                            continue;
                        }
                        let color = self.label_color(label.unwrap());
                        self.push_cell_edges_coronal(
                            slice_index,
                            i,
                            k,
                            labels,
                            color,
                            &mut segments,
                        );
                    }
                }
            }
            _ => {
                for k in 0..self.dims[2] {
                    for j in 0..self.dims[1] {
                        let label = self.label_at(slice_index as isize, j as isize, k as isize);
                        if !label.is_some_and(|value| labels.contains(&value)) {
                            continue;
                        }
                        let color = self.label_color(label.unwrap());
                        self.push_cell_edges_sagittal(
                            slice_index,
                            j,
                            k,
                            labels,
                            color,
                            &mut segments,
                        );
                    }
                }
            }
        }
        segments
    }

    fn crop_streamline(
        &self,
        points: &[[f32; 3]],
        labels: &BTreeSet<u32>,
        keep_inside: bool,
    ) -> Vec<Vec<[f32; 3]>> {
        if points.len() < 2 || labels.is_empty() {
            return Vec::new();
        }

        let state = |point: [f32; 3]| {
            self.sample_label_world(Vec3::from(point))
                .is_some_and(|label| labels.contains(&label))
        };

        let mut segments = Vec::new();
        let mut current = Vec::new();
        let mut prev_inside = state(points[0]);

        if prev_inside == keep_inside {
            current.push(points[0]);
        }

        for window in points.windows(2) {
            let a = window[0];
            let b = window[1];
            let next_inside = state(b);
            if prev_inside != next_inside {
                let boundary = [
                    (a[0] + b[0]) * 0.5,
                    (a[1] + b[1]) * 0.5,
                    (a[2] + b[2]) * 0.5,
                ];
                if prev_inside == keep_inside {
                    current.push(boundary);
                } else {
                    current.push(boundary);
                }
                if current.len() >= 2 {
                    segments.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                if next_inside == keep_inside {
                    current.push(boundary);
                }
            }
            if next_inside == keep_inside {
                current.push(b);
            }
            prev_inside = next_inside;
        }

        if current.len() >= 2 {
            segments.push(current);
        }

        segments
    }

    fn push_cell_edges_axial(
        &self,
        k: usize,
        i: usize,
        j: usize,
        labels: &BTreeSet<u32>,
        color: [f32; 4],
        out: &mut Vec<([Vec3; 2], [f32; 4])>,
    ) {
        let corners = [
            self.voxel_to_world(Vec3::new(i as f32 - 0.5, j as f32 - 0.5, k as f32)),
            self.voxel_to_world(Vec3::new(i as f32 + 0.5, j as f32 - 0.5, k as f32)),
            self.voxel_to_world(Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32)),
            self.voxel_to_world(Vec3::new(i as f32 - 0.5, j as f32 + 0.5, k as f32)),
        ];

        let neighbors = [
            self.label_at(i as isize, j as isize - 1, k as isize),
            self.label_at(i as isize + 1, j as isize, k as isize),
            self.label_at(i as isize, j as isize + 1, k as isize),
            self.label_at(i as isize - 1, j as isize, k as isize),
        ];
        let edges = [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ];
        for (neighbor, edge) in neighbors.into_iter().zip(edges) {
            if !neighbor.is_some_and(|label| labels.contains(&label)) {
                out.push((edge, color));
            }
        }
    }

    fn push_cell_edges_coronal(
        &self,
        j: usize,
        i: usize,
        k: usize,
        labels: &BTreeSet<u32>,
        color: [f32; 4],
        out: &mut Vec<([Vec3; 2], [f32; 4])>,
    ) {
        let corners = [
            self.voxel_to_world(Vec3::new(i as f32 - 0.5, j as f32, k as f32 - 0.5)),
            self.voxel_to_world(Vec3::new(i as f32 + 0.5, j as f32, k as f32 - 0.5)),
            self.voxel_to_world(Vec3::new(i as f32 + 0.5, j as f32, k as f32 + 0.5)),
            self.voxel_to_world(Vec3::new(i as f32 - 0.5, j as f32, k as f32 + 0.5)),
        ];

        let neighbors = [
            self.label_at(i as isize, j as isize, k as isize - 1),
            self.label_at(i as isize + 1, j as isize, k as isize),
            self.label_at(i as isize, j as isize, k as isize + 1),
            self.label_at(i as isize - 1, j as isize, k as isize),
        ];
        let edges = [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ];
        for (neighbor, edge) in neighbors.into_iter().zip(edges) {
            if !neighbor.is_some_and(|label| labels.contains(&label)) {
                out.push((edge, color));
            }
        }
    }

    fn push_cell_edges_sagittal(
        &self,
        i: usize,
        j: usize,
        k: usize,
        labels: &BTreeSet<u32>,
        color: [f32; 4],
        out: &mut Vec<([Vec3; 2], [f32; 4])>,
    ) {
        let corners = [
            self.voxel_to_world(Vec3::new(i as f32, j as f32 - 0.5, k as f32 - 0.5)),
            self.voxel_to_world(Vec3::new(i as f32, j as f32 + 0.5, k as f32 - 0.5)),
            self.voxel_to_world(Vec3::new(i as f32, j as f32 + 0.5, k as f32 + 0.5)),
            self.voxel_to_world(Vec3::new(i as f32, j as f32 - 0.5, k as f32 + 0.5)),
        ];

        let neighbors = [
            self.label_at(i as isize, j as isize, k as isize - 1),
            self.label_at(i as isize, j as isize + 1, k as isize),
            self.label_at(i as isize, j as isize, k as isize + 1),
            self.label_at(i as isize, j as isize - 1, k as isize),
        ];
        let edges = [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
        ];
        for (neighbor, edge) in neighbors.into_iter().zip(edges) {
            if !neighbor.is_some_and(|label| labels.contains(&label)) {
                out.push((edge, color));
            }
        }
    }

    fn label_at(&self, i: isize, j: isize, k: isize) -> Option<u32> {
        if i < 0 || j < 0 || k < 0 {
            return None;
        }
        let i = i as usize;
        let j = j as usize;
        let k = k as usize;
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return None;
        }
        let idx = i + self.dims[0] * (j + self.dims[1] * k);
        self.labels.get(idx).copied()
    }
}

pub fn parse_label_table(path: &Path) -> anyhow::Result<BTreeMap<u32, ParcelLabel>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read label table {}", path.display()))?;
    let mut labels = BTreeMap::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(label) =
            parse_csv_label_line(trimmed).or_else(|| parse_whitespace_label_line(trimmed))
        {
            labels.insert(label.id, label);
        }
    }

    if labels.is_empty() {
        bail!("No labels were parsed from {}", path.display());
    }

    Ok(labels)
}

fn parse_csv_label_line(line: &str) -> Option<ParcelLabel> {
    let parts: Vec<_> = line.split(',').map(str::trim).collect();
    if parts.len() < 2 {
        return None;
    }
    let id = parts[0].parse::<u32>().ok()?;
    let name = parts[1].trim_matches('"').to_string();
    let color = if parts.len() >= 6 {
        [
            parts[2].parse::<u8>().ok()? as f32 / 255.0,
            parts[3].parse::<u8>().ok()? as f32 / 255.0,
            parts[4].parse::<u8>().ok()? as f32 / 255.0,
            parts[5].parse::<u8>().ok()? as f32 / 255.0,
        ]
    } else {
        generated_label_color(id)
    };
    Some(ParcelLabel { id, name, color })
}

fn parse_whitespace_label_line(line: &str) -> Option<ParcelLabel> {
    let parts: Vec<_> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let id = parts[0].parse::<u32>().ok()?;
    let (name, color) = if parts.len() >= 5
        && parts[parts.len() - 3].parse::<u8>().is_ok()
        && parts[parts.len() - 2].parse::<u8>().is_ok()
        && parts[parts.len() - 1].parse::<u8>().is_ok()
    {
        let name = parts[1..parts.len() - 3].join(" ");
        let color = [
            parts[parts.len() - 3].parse::<u8>().ok()? as f32 / 255.0,
            parts[parts.len() - 2].parse::<u8>().ok()? as f32 / 255.0,
            parts[parts.len() - 1].parse::<u8>().ok()? as f32 / 255.0,
            1.0,
        ];
        (name, color)
    } else if parts.len() >= 6
        && parts[parts.len() - 4].parse::<u8>().is_ok()
        && parts[parts.len() - 3].parse::<u8>().is_ok()
        && parts[parts.len() - 2].parse::<u8>().is_ok()
        && parts[parts.len() - 1].parse::<u8>().is_ok()
    {
        let name = parts[1..parts.len() - 4].join(" ");
        let color = [
            parts[parts.len() - 4].parse::<u8>().ok()? as f32 / 255.0,
            parts[parts.len() - 3].parse::<u8>().ok()? as f32 / 255.0,
            parts[parts.len() - 2].parse::<u8>().ok()? as f32 / 255.0,
            parts[parts.len() - 1].parse::<u8>().ok()? as f32 / 255.0,
        ];
        (name, color)
    } else {
        (parts[1..].join(" "), generated_label_color(id))
    };
    Some(ParcelLabel { id, name, color })
}

pub fn guess_label_table_path(volume_path: &Path) -> Option<PathBuf> {
    let parent = volume_path.parent()?;
    let stem = volume_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("parcellation");
    let candidates = [
        format!("{stem}.txt"),
        format!("{stem}.lut"),
        format!("{stem}.csv"),
        format!("{stem}.tsv"),
    ];
    candidates
        .into_iter()
        .map(|name| parent.join(name))
        .find(|path| path.exists())
}

fn generated_label_color(label: u32) -> [f32; 4] {
    let x = label.wrapping_mul(0x9e3779b9);
    let r = ((x & 0xff) as f32 / 255.0).clamp(0.2, 0.95);
    let g = (((x >> 8) & 0xff) as f32 / 255.0).clamp(0.2, 0.95);
    let b = (((x >> 16) & 0xff) as f32 / 255.0).clamp(0.2, 0.95);
    [r, g, b, 0.95]
}

fn nifti_voxel_to_ras(header: &nifti::NiftiHeader) -> anyhow::Result<Mat4> {
    if header.qform_code > 0 {
        let qfac = if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        Ok(quatern_to_mat44(
            header.quatern_b,
            header.quatern_c,
            header.quatern_d,
            header.quatern_x,
            header.quatern_y,
            header.quatern_z,
            header.pixdim[1],
            header.pixdim[2],
            header.pixdim[3],
            qfac,
        ))
    } else if header.sform_code > 0 {
        let sx = header.srow_x;
        let sy = header.srow_y;
        let sz = header.srow_z;
        Ok(Mat4::from_cols(
            Vec4::new(sx[0], sy[0], sz[0], 0.0),
            Vec4::new(sx[1], sy[1], sz[1], 0.0),
            Vec4::new(sx[2], sy[2], sz[2], 0.0),
            Vec4::new(sx[3], sy[3], sz[3], 1.0),
        ))
    } else {
        bail!("NIfTI header has neither valid qform nor sform code");
    }
}

fn quatern_to_mat44(
    qb: f32,
    qc: f32,
    qd: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    qfac: f32,
) -> Mat4 {
    let mut a = 1.0f64 - (qb * qb + qc * qc + qd * qd) as f64;
    let mut b = qb as f64;
    let mut c = qc as f64;
    let mut d = qd as f64;

    if a < 1e-7 {
        let s = 1.0 / (b * b + c * c + d * d).sqrt();
        a = 0.0;
        b *= s;
        c *= s;
        d *= s;
    } else {
        a = a.sqrt();
    }

    let xd = if dx > 0.0 { dx as f64 } else { 1.0 };
    let yd = if dy > 0.0 { dy as f64 } else { 1.0 };
    let zd_abs = if dz > 0.0 { dz as f64 } else { 1.0 };
    let zd = if qfac < 0.0 { -zd_abs } else { zd_abs };

    let r00 = ((a * a + b * b - c * c - d * d) * xd) as f32;
    let r01 = ((2.0 * (b * c - a * d)) * yd) as f32;
    let r02 = ((2.0 * (b * d + a * c)) * zd) as f32;
    let r10 = ((2.0 * (b * c + a * d)) * xd) as f32;
    let r11 = ((a * a + c * c - b * b - d * d) * yd) as f32;
    let r12 = ((2.0 * (c * d - a * b)) * zd) as f32;
    let r20 = ((2.0 * (b * d - a * c)) * xd) as f32;
    let r21 = ((2.0 * (c * d + a * b)) * yd) as f32;
    let r22 = ((a * a + d * d - c * c - b * b) * zd) as f32;

    Mat4::from_cols(
        Vec4::new(r00, r10, r20, 0.0),
        Vec4::new(r01, r11, r21, 0.0),
        Vec4::new(r02, r12, r22, 0.0),
        Vec4::new(qx, qy, qz, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whitespace_label_table_lines() {
        let line = "42 SuperiorTemporal 120 180 220";
        let label = parse_whitespace_label_line(line).unwrap();
        assert_eq!(label.id, 42);
        assert_eq!(label.name, "SuperiorTemporal");
        assert!(label.color[0] > 0.0);
    }

    #[test]
    fn parses_csv_label_table_lines() {
        let line = "17,\"Left-Hippocampus\",20,120,200,255";
        let label = parse_csv_label_line(line).unwrap();
        assert_eq!(label.id, 17);
        assert_eq!(label.name, "Left-Hippocampus");
        assert_eq!(label.color[3], 1.0);
    }
}

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::Path;

use anyhow::{Context, bail};
use base64::Engine;
use glam::Vec3;
use roxmltree::{Document, Node};

const NIFTI_INTENT_POINTSET: i32 = 1008;
const NIFTI_INTENT_TRIANGLE: i32 = 1009;

#[derive(Clone)]
pub struct GiftiSurfaceData {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub bbox_min: Vec3,
    pub bbox_max: Vec3,
}

impl GiftiSurfaceData {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let xml = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read GIFTI file {}", path.display()))?;
        let doc = Document::parse_with_options(
            &xml,
            roxmltree::ParsingOptions {
                allow_dtd: true,
                ..Default::default()
            },
        )
        .context("Failed to parse GIFTI XML")?;

        let mut pointset: Option<Vec<[f32; 3]>> = None;
        let mut triangles: Option<Vec<[u32; 3]>> = None;

        for da in doc.descendants().filter(|n| n.has_tag_name("DataArray")) {
            let intent = parse_intent(
                da.attribute("Intent")
                    .ok_or_else(|| anyhow::anyhow!("DataArray missing Intent attribute"))?,
            )?;

            if intent != NIFTI_INTENT_POINTSET && intent != NIFTI_INTENT_TRIANGLE {
                continue;
            }

            let datatype = parse_datatype(
                da.attribute("DataType")
                    .ok_or_else(|| anyhow::anyhow!("DataArray missing DataType attribute"))?,
            )?;
            let encoding = parse_encoding(
                da.attribute("Encoding")
                    .ok_or_else(|| anyhow::anyhow!("DataArray missing Encoding attribute"))?,
            )?;
            let endian = parse_endian(da.attribute("Endian").unwrap_or("LittleEndian"))?;
            let row_major = parse_row_major(
                da.attribute("ArrayIndexingOrder")
                    .unwrap_or("RowMajorOrder"),
            )?;
            let dims = parse_dims(da)?;
            let values = parse_data_values(da, datatype, encoding, endian)?;

            if intent == NIFTI_INTENT_POINTSET {
                if dims.len() < 2 {
                    bail!("POINTSET DataArray has invalid dimensions");
                }
                let nrows = dims[0];
                let ncols = dims[1];
                if ncols != 3 {
                    bail!("POINTSET DataArray must be Nx3, got {} columns", ncols);
                }
                let reordered = reorder_2d(&values, nrows, ncols, row_major)
                    .context("Failed to reorder POINTSET values")?;
                let mut verts = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    let base = r * 3;
                    verts.push([reordered[base], reordered[base + 1], reordered[base + 2]]);
                }
                let c_ras = parse_c_ras_meta(da);
                if c_ras != [0.0, 0.0, 0.0] {
                    for p in &mut verts {
                        p[0] += c_ras[0];
                        p[1] += c_ras[1];
                        p[2] += c_ras[2];
                    }
                }
                pointset = Some(verts);
            } else {
                if dims.len() < 2 {
                    bail!("TRIANGLE DataArray has invalid dimensions");
                }
                let nrows = dims[0];
                let ncols = dims[1];
                if ncols != 3 {
                    bail!("TRIANGLE DataArray must be Nx3, got {} columns", ncols);
                }
                let reordered = reorder_2d(&values, nrows, ncols, row_major)
                    .context("Failed to reorder TRIANGLE values")?;
                let mut tris = Vec::with_capacity(nrows);
                for r in 0..nrows {
                    let base = r * 3;
                    let a = reordered[base] as i64;
                    let b = reordered[base + 1] as i64;
                    let c = reordered[base + 2] as i64;
                    if a < 0 || b < 0 || c < 0 {
                        bail!("TRIANGLE DataArray contains negative indices");
                    }
                    tris.push([a as u32, b as u32, c as u32]);
                }
                triangles = Some(tris);
            }
        }

        let vertices = pointset.context("No NIFTI_INTENT_POINTSET DataArray found")?;
        let tris = triangles.context("No NIFTI_INTENT_TRIANGLE DataArray found")?;
        if vertices.is_empty() {
            bail!("POINTSET DataArray has zero vertices");
        }

        let mut tris = tris;
        orient_triangles_consistently(&mut tris);
        orient_surface_outward(&vertices, &mut tris);

        let mut indices = Vec::with_capacity(tris.len() * 3);
        let max_idx = vertices.len() as u32;
        for tri in tris {
            if tri[0] >= max_idx || tri[1] >= max_idx || tri[2] >= max_idx {
                bail!("Triangle index out of bounds for POINTSET vertex count");
            }
            indices.extend_from_slice(&tri);
        }
        let normals = compute_vertex_normals(&vertices, &indices);

        let mut bbox_min = Vec3::splat(f32::INFINITY);
        let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);
        for p in &vertices {
            let v = Vec3::from(*p);
            bbox_min = bbox_min.min(v);
            bbox_max = bbox_max.max(v);
        }

        Ok(Self {
            vertices,
            normals,
            indices,
            bbox_min,
            bbox_max,
        })
    }
}

#[derive(Clone, Copy)]
enum DataType {
    Int32,
    UInt32,
    Float32,
    Float64,
}

#[derive(Clone, Copy)]
enum Encoding {
    Ascii,
    Base64Binary,
    GZipBase64Binary,
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

fn parse_intent(intent: &str) -> anyhow::Result<i32> {
    if let Ok(v) = intent.parse::<i32>() {
        return Ok(v);
    }
    match intent {
        "NIFTI_INTENT_POINTSET" => Ok(NIFTI_INTENT_POINTSET),
        "NIFTI_INTENT_TRIANGLE" => Ok(NIFTI_INTENT_TRIANGLE),
        _ => bail!("Unsupported DataArray intent: {intent}"),
    }
}

fn parse_datatype(dtype: &str) -> anyhow::Result<DataType> {
    if let Ok(v) = dtype.parse::<i32>() {
        return match v {
            8 => Ok(DataType::Int32),
            16 => Ok(DataType::Float32),
            64 => Ok(DataType::Float64),
            768 => Ok(DataType::UInt32),
            _ => bail!("Unsupported GIFTI DataType code: {dtype}"),
        };
    }
    match dtype {
        "NIFTI_TYPE_INT32" => Ok(DataType::Int32),
        "NIFTI_TYPE_UINT32" => Ok(DataType::UInt32),
        "NIFTI_TYPE_FLOAT32" => Ok(DataType::Float32),
        "NIFTI_TYPE_FLOAT64" => Ok(DataType::Float64),
        _ => bail!("Unsupported GIFTI DataType: {dtype}"),
    }
}

fn parse_encoding(enc: &str) -> anyhow::Result<Encoding> {
    match enc {
        "ASCII" => Ok(Encoding::Ascii),
        "Base64Binary" => Ok(Encoding::Base64Binary),
        "GZipBase64Binary" => Ok(Encoding::GZipBase64Binary),
        "ExternalFileBinary" => bail!("ExternalFileBinary GIFTI encoding is not supported"),
        _ => bail!("Unsupported GIFTI Encoding: {enc}"),
    }
}

fn parse_endian(endian: &str) -> anyhow::Result<Endian> {
    match endian {
        "LittleEndian" => Ok(Endian::Little),
        "BigEndian" => Ok(Endian::Big),
        "Undefined" => Ok(Endian::Little),
        _ => bail!("Unsupported GIFTI Endian value: {endian}"),
    }
}

fn parse_row_major(order: &str) -> anyhow::Result<bool> {
    match order {
        "RowMajorOrder" => Ok(true),
        "ColumnMajorOrder" => Ok(false),
        "Undefined" => Ok(true),
        _ => bail!("Unsupported ArrayIndexingOrder value: {order}"),
    }
}

fn parse_dims(da: Node<'_, '_>) -> anyhow::Result<Vec<usize>> {
    let ndim: usize = da
        .attribute("Dimensionality")
        .ok_or_else(|| anyhow::anyhow!("DataArray missing Dimensionality"))?
        .parse()
        .context("Invalid Dimensionality attribute")?;
    if ndim == 0 {
        bail!("DataArray Dimensionality cannot be zero");
    }
    let mut dims = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let key = format!("Dim{i}");
        let d: usize = da
            .attribute(key.as_str())
            .ok_or_else(|| anyhow::anyhow!("DataArray missing {key}"))?
            .parse()
            .with_context(|| format!("Invalid {key}"))?;
        dims.push(d);
    }
    Ok(dims)
}

fn parse_data_values(
    da: Node<'_, '_>,
    dtype: DataType,
    encoding: Encoding,
    endian: Endian,
) -> anyhow::Result<Vec<f64>> {
    let data_node = da
        .children()
        .find(|n| n.has_tag_name("Data"))
        .ok_or_else(|| anyhow::anyhow!("DataArray missing Data child element"))?;
    let data_text = data_node.text().unwrap_or("");

    match encoding {
        Encoding::Ascii => parse_ascii_values(data_text),
        Encoding::Base64Binary | Encoding::GZipBase64Binary => {
            let bytes = decode_binary_payload(data_text, encoding)?;
            decode_binary_values(&bytes, dtype, endian)
        }
    }
}

fn parse_ascii_values(data_text: &str) -> anyhow::Result<Vec<f64>> {
    let mut out = Vec::new();
    for tok in data_text.split_whitespace() {
        let v = tok
            .parse::<f64>()
            .with_context(|| format!("Invalid ASCII numeric token in GIFTI data: {tok}"))?;
        out.push(v);
    }
    Ok(out)
}

fn decode_binary_payload(data_text: &str, encoding: Encoding) -> anyhow::Result<Vec<u8>> {
    let compact: String = data_text
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    let raw = base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .context("Failed to decode Base64 GIFTI data")?;
    if matches!(encoding, Encoding::GZipBase64Binary) {
        // gifticlib labels compressed payloads as "GZipBase64Binary", but in
        // practice files may contain zlib streams (0x78 0x9c) instead of a gzip
        // wrapper. Accept both to match real-world GIFTI variants.
        let mut out = Vec::new();
        let gzip_result = {
            let mut decoder = flate2::read::GzDecoder::new(raw.as_slice());
            decoder.read_to_end(&mut out)
        };
        if gzip_result.is_ok() {
            return Ok(out);
        }

        out.clear();
        let zlib_result = {
            let mut decoder = flate2::read::ZlibDecoder::new(raw.as_slice());
            decoder.read_to_end(&mut out)
        };
        if zlib_result.is_ok() {
            return Ok(out);
        }

        out.clear();
        let mut deflate_decoder = flate2::read::DeflateDecoder::new(raw.as_slice());
        deflate_decoder
            .read_to_end(&mut out)
            .context("Failed to decompress GIFTI compressed data (gzip/zlib/deflate)")?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

fn decode_binary_values(bytes: &[u8], dtype: DataType, endian: Endian) -> anyhow::Result<Vec<f64>> {
    let elem_size = match dtype {
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4usize,
        DataType::Float64 => 8usize,
    };
    if !bytes.len().is_multiple_of(elem_size) {
        bail!(
            "Binary GIFTI data has invalid length {} for element size {}",
            bytes.len(),
            elem_size
        );
    }

    let mut out = Vec::with_capacity(bytes.len() / elem_size);
    match dtype {
        DataType::Int32 => {
            for chunk in bytes.chunks_exact(4) {
                let v = match endian {
                    Endian::Little => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    Endian::Big => i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                };
                out.push(v as f64);
            }
        }
        DataType::UInt32 => {
            for chunk in bytes.chunks_exact(4) {
                let v = match endian {
                    Endian::Little => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    Endian::Big => u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                };
                out.push(v as f64);
            }
        }
        DataType::Float32 => {
            for chunk in bytes.chunks_exact(4) {
                let bits = match endian {
                    Endian::Little => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    Endian::Big => u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                };
                out.push(f32::from_bits(bits) as f64);
            }
        }
        DataType::Float64 => {
            for chunk in bytes.chunks_exact(8) {
                let bits = match endian {
                    Endian::Little => u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]),
                    Endian::Big => u64::from_be_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]),
                };
                out.push(f64::from_bits(bits));
            }
        }
    }
    Ok(out)
}

fn reorder_2d(
    values: &[f64],
    rows: usize,
    cols: usize,
    row_major: bool,
) -> anyhow::Result<Vec<f32>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow::anyhow!("rows*cols overflows for DataArray"))?;
    if values.len() != expected {
        bail!(
            "DataArray value count mismatch: expected {}, got {}",
            expected,
            values.len()
        );
    }
    let mut out = vec![0.0f32; expected];
    if row_major {
        for (i, v) in values.iter().enumerate() {
            out[i] = *v as f32;
        }
    } else {
        for c in 0..cols {
            for r in 0..rows {
                out[r * cols + c] = values[c * rows + r] as f32;
            }
        }
    }
    Ok(out)
}

fn parse_c_ras_meta(da: Node<'_, '_>) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    let Some(meta_node) = da.children().find(|n| n.has_tag_name("MetaData")) else {
        return out;
    };

    for md in meta_node.children().filter(|n| n.has_tag_name("MD")) {
        let name = md
            .children()
            .find(|n| n.has_tag_name("Name"))
            .and_then(|n| n.text())
            .unwrap_or("");
        let value = md
            .children()
            .find(|n| n.has_tag_name("Value"))
            .and_then(|n| n.text())
            .unwrap_or("");
        let parsed = value.parse::<f32>().unwrap_or(0.0);
        match name {
            "VolGeomC_R" => out[0] = parsed,
            "VolGeomC_A" => out[1] = parsed,
            "VolGeomC_S" => out[2] = parsed,
            _ => {}
        }
    }

    out
}

fn compute_vertex_normals(vertices: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let ia = tri[0] as usize;
        let ib = tri[1] as usize;
        let ic = tri[2] as usize;
        if ia >= vertices.len() || ib >= vertices.len() || ic >= vertices.len() {
            continue;
        }
        let a = Vec3::from(vertices[ia]);
        let b = Vec3::from(vertices[ib]);
        let c = Vec3::from(vertices[ic]);
        let n = (b - a).cross(c - a);
        normals[ia] += n;
        normals[ib] += n;
        normals[ic] += n;
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or_zero().into())
        .collect()
}

fn orient_triangles_consistently(triangles: &mut [[u32; 3]]) {
    if triangles.len() < 2 {
        return;
    }

    let mut edge_map: HashMap<(u32, u32), Vec<(usize, bool)>> = HashMap::new();
    for (tri_index, tri) in triangles.iter().copied().enumerate() {
        for (start, end) in triangle_edges(tri) {
            let key = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            let same_as_canonical = start < end;
            edge_map
                .entry(key)
                .or_default()
                .push((tri_index, same_as_canonical));
        }
    }

    let mut visited = vec![false; triangles.len()];
    let mut should_flip = vec![false; triangles.len()];
    let mut queue = VecDeque::new();

    for seed in 0..triangles.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        queue.push_back(seed);

        while let Some(current) = queue.pop_front() {
            for (start, end) in triangle_edges(triangles[current]) {
                let key = if start < end {
                    (start, end)
                } else {
                    (end, start)
                };
                let Some(neighbors) = edge_map.get(&key) else {
                    continue;
                };
                if neighbors.len() != 2 {
                    continue;
                }

                let mut current_sign = None;
                let mut neighbor = None;
                for (tri_index, sign) in neighbors {
                    if *tri_index == current {
                        current_sign = Some(*sign);
                    } else {
                        neighbor = Some((*tri_index, *sign));
                    }
                }
                let (neighbor_index, neighbor_sign) = match neighbor {
                    Some(value) => value,
                    None => continue,
                };
                let current_sign = match current_sign {
                    Some(value) => value,
                    None => continue,
                };

                let neighbor_flip = if current_sign == neighbor_sign {
                    !should_flip[current]
                } else {
                    should_flip[current]
                };
                if visited[neighbor_index] {
                    continue;
                }
                visited[neighbor_index] = true;
                should_flip[neighbor_index] = neighbor_flip;
                queue.push_back(neighbor_index);
            }
        }
    }

    for (tri, flip) in triangles.iter_mut().zip(should_flip) {
        if flip {
            tri.swap(1, 2);
        }
    }
}

fn orient_surface_outward(vertices: &[[f32; 3]], triangles: &mut [[u32; 3]]) {
    if triangles.is_empty() || vertices.is_empty() {
        return;
    }

    let centroid = vertices
        .iter()
        .fold(Vec3::ZERO, |acc, vertex| acc + Vec3::from(*vertex))
        / vertices.len() as f32;
    let orientation_score = triangles.iter().fold(0.0f32, |acc, tri| {
        let a = Vec3::from(vertices[tri[0] as usize]);
        let b = Vec3::from(vertices[tri[1] as usize]);
        let c = Vec3::from(vertices[tri[2] as usize]);
        let normal = (b - a).cross(c - a);
        let face_center = (a + b + c) / 3.0;
        acc + normal.dot(face_center - centroid)
    });

    if orientation_score < 0.0 {
        for tri in triangles {
            tri.swap(1, 2);
        }
    }
}

fn triangle_edges(tri: [u32; 3]) -> [(u32, u32); 3] {
    [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
}

#[cfg(test)]
mod tests {
    use super::{orient_surface_outward, orient_triangles_consistently, triangle_edges};
    use glam::Vec3;

    #[test]
    fn fixes_shared_edge_winding() {
        let mut tris = vec![[0, 1, 2], [1, 3, 2]];
        orient_triangles_consistently(&mut tris);

        let tri0_shared = triangle_edges(tris[0])
            .into_iter()
            .find(|edge| (edge.0 == 1 && edge.1 == 2) || (edge.0 == 2 && edge.1 == 1))
            .unwrap();
        let tri1_shared = triangle_edges(tris[1])
            .into_iter()
            .find(|edge| (edge.0 == 1 && edge.1 == 2) || (edge.0 == 2 && edge.1 == 1))
            .unwrap();

        assert_eq!(tri0_shared, (1, 2));
        assert_eq!(tri1_shared, (2, 1));
    }

    #[test]
    fn flips_inward_surface_outward() {
        let vertices = vec![
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let mut tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];

        orient_surface_outward(&vertices, &mut tris);

        let centroid = vertices
            .iter()
            .fold(Vec3::ZERO, |acc, vertex| acc + Vec3::from(*vertex))
            / vertices.len() as f32;
        let score = tris.iter().fold(0.0f32, |acc, tri| {
            let a = Vec3::from(vertices[tri[0] as usize]);
            let b = Vec3::from(vertices[tri[1] as usize]);
            let c = Vec3::from(vertices[tri[2] as usize]);
            let normal = (b - a).cross(c - a);
            let face_center = (a + b + c) / 3.0;
            acc + normal.dot(face_center - centroid)
        });

        assert!(score > 0.0);
    }
}

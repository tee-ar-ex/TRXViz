use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, bail};
use flate2::read::GzDecoder;
use glam::{Mat4, Vec4};
use roxmltree::{Document, Node};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CiftiIntent {
    DenseScalar,
    DenseSeries,
    ParcelScalar,
    DenseLabel,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CiftiStructure {
    CortexLeft,
    CortexRight,
    Subcortical,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ScalarKind {
    Continuous,
    Label,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LabelEntry {
    pub key: i32,
    pub name: String,
    pub rgba: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScalarMetadata {
    pub map_name: String,
    pub suggested_range: Option<(f32, f32)>,
    pub series_index: Option<usize>,
    pub series_value: Option<f32>,
    pub label_table: Vec<LabelEntry>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceScalars {
    pub structure: Option<CiftiStructure>,
    pub source_surface_id: Option<usize>,
    pub vertex_count: usize,
    pub values: Vec<f32>,
    pub kind: ScalarKind,
    pub metadata: ScalarMetadata,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeScalars {
    pub dims: [usize; 3],
    pub voxel_to_ras: Mat4,
    pub values: Vec<f32>,
    pub kind: ScalarKind,
    pub metadata: ScalarMetadata,
}

#[derive(Clone, Debug)]
pub struct LoadedCifti {
    pub intent: CiftiIntent,
    pub map_count: usize,
    pub left_scalars: Vec<Option<SurfaceScalars>>,
    pub right_scalars: Vec<Option<SurfaceScalars>>,
    pub subcortical_scalars: Vec<Option<VolumeScalars>>,
}

impl LoadedCifti {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = read_file_bytes(path)
            .with_context(|| format!("Failed to read CIFTI container {}", path.display()))?;
        let nifti = parse_nifti2_container(&bytes)?;
        let intent = parse_cifti_intent(nifti.header.intent_code as u16)?;
        let xml_text = nifti
            .extensions
            .iter()
            .find(|extension| extension.ecode == 32)
            .and_then(|extension| std::str::from_utf8(&extension.data).ok())
            .map(|text| text.trim_end_matches('\0').trim().to_string())
            .filter(|text| text.contains("<CIFTI"))
            .ok_or_else(|| anyhow::anyhow!("Missing CIFTI XML extension"))?;
        let xml = parse_cifti_xml(&xml_text)?;
        let shape = squeeze_trailing_singletons_dims(&nifti.shape)?;
        if shape.len() != 2 {
            bail!("CIFTI matrix must be 2D after squeezing singleton dimensions");
        }

        let (brain_axis_dim, brain_axis) = xml
            .axes
            .iter()
            .find_map(|(dim, axis)| match axis {
                CiftiAxis::BrainModels(_) | CiftiAxis::Parcels(_) => Some((*dim, axis)),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("Missing brain-model or parcel axis in CIFTI XML"))?;
        let (map_axis_dim, map_axis) = xml
            .axes
            .iter()
            .find_map(|(dim, axis)| {
                if *dim == brain_axis_dim {
                    None
                } else {
                    Some((*dim, axis))
                }
            })
            .ok_or_else(|| anyhow::anyhow!("Missing map axis in CIFTI XML"))?;

        if brain_axis_dim > 1 || map_axis_dim > 1 {
            bail!("Only 2D CIFTI matrices are supported");
        }

        let map_count = shape[map_axis_dim];
        let grayordinate_count = shape[brain_axis_dim];
        match brain_axis {
            CiftiAxis::BrainModels(models) if models.grayordinate_count() != grayordinate_count => {
                bail!(
                    "Brain-model grayordinate count {} does not match matrix shape {}",
                    models.grayordinate_count(),
                    grayordinate_count
                );
            }
            CiftiAxis::Parcels(parcels) if parcels.parcels.len() != grayordinate_count => {
                bail!(
                    "Parcel count {} does not match matrix shape {}",
                    parcels.parcels.len(),
                    grayordinate_count
                );
            }
            _ => {}
        }

        let mut left_scalars = Vec::with_capacity(map_count);
        let mut right_scalars = Vec::with_capacity(map_count);
        let mut subcortical_scalars = Vec::with_capacity(map_count);
        for map_index in 0..map_count {
            let map_values = extract_map_values(&nifti.data, &shape, map_axis_dim, map_index)?;
            let metadata = metadata_for_map(map_axis, map_index, &map_values);
            match brain_axis {
                CiftiAxis::BrainModels(models) => {
                    let expanded = expand_dense_brain_models(models, &map_values, &metadata)?;
                    left_scalars.push(expanded.left);
                    right_scalars.push(expanded.right);
                    subcortical_scalars.push(expanded.subcortical);
                }
                CiftiAxis::Parcels(parcels) => {
                    let expanded = expand_parcels(parcels, &map_values, &metadata)?;
                    left_scalars.push(expanded.left);
                    right_scalars.push(expanded.right);
                    subcortical_scalars.push(expanded.subcortical);
                }
                _ => bail!("Unsupported CIFTI brain axis"),
            }
        }

        Ok(Self {
            intent,
            map_count,
            left_scalars,
            right_scalars,
            subcortical_scalars,
        })
    }
}

fn read_file_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader: Box<dyn Read> = if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn parse_cifti_intent(intent_code: u16) -> anyhow::Result<CiftiIntent> {
    match intent_code {
        3002 => Ok(CiftiIntent::DenseSeries),
        3006 => Ok(CiftiIntent::DenseScalar),
        3007 => Ok(CiftiIntent::DenseLabel),
        3008 => Ok(CiftiIntent::ParcelScalar),
        other => bail!("Unsupported CIFTI intent code {other}"),
    }
}

#[derive(Clone, Debug)]
struct ParsedExtension {
    ecode: i32,
    data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ParsedNifti2 {
    header: ParsedNifti2Header,
    shape: Vec<usize>,
    data: Vec<f32>,
    extensions: Vec<ParsedExtension>,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct ParsedNifti2Header {
    datatype: u16,
    dim: [i64; 8],
    vox_offset: usize,
    scl_slope: f64,
    scl_inter: f64,
    pixdim: [f64; 8],
    qform_code: i32,
    sform_code: i32,
    quatern_b: f64,
    quatern_c: f64,
    quatern_d: f64,
    qoffset_x: f64,
    qoffset_y: f64,
    qoffset_z: f64,
    srow_x: [f64; 4],
    srow_y: [f64; 4],
    srow_z: [f64; 4],
    intent_code: i32,
}

fn parse_nifti2_container(bytes: &[u8]) -> anyhow::Result<ParsedNifti2> {
    if bytes.len() < 544 {
        bail!("File is too short to be a NIfTI-2 container");
    }
    let sizeof_hdr_le = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let sizeof_hdr_be = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let little_endian = match (sizeof_hdr_le, sizeof_hdr_be) {
        (540, _) => true,
        (_, 540) => false,
        _ => bail!("Invalid NIfTI-2 header size"),
    };
    let header = ParsedNifti2Header {
        datatype: read_u16(bytes, 12, little_endian)?,
        dim: read_i64_array::<8>(bytes, 16, little_endian)?,
        vox_offset: read_i64(bytes, 168, little_endian)? as usize,
        scl_slope: read_f64(bytes, 176, little_endian)?,
        scl_inter: read_f64(bytes, 184, little_endian)?,
        pixdim: read_f64_array::<8>(bytes, 104, little_endian)?,
        qform_code: read_i32(bytes, 344, little_endian)?,
        sform_code: read_i32(bytes, 348, little_endian)?,
        quatern_b: read_f64(bytes, 352, little_endian)?,
        quatern_c: read_f64(bytes, 360, little_endian)?,
        quatern_d: read_f64(bytes, 368, little_endian)?,
        qoffset_x: read_f64(bytes, 376, little_endian)?,
        qoffset_y: read_f64(bytes, 384, little_endian)?,
        qoffset_z: read_f64(bytes, 392, little_endian)?,
        srow_x: read_f64_array::<4>(bytes, 400, little_endian)?,
        srow_y: read_f64_array::<4>(bytes, 432, little_endian)?,
        srow_z: read_f64_array::<4>(bytes, 464, little_endian)?,
        intent_code: read_i32(bytes, 504, little_endian)?,
    };
    if header.dim[0] < 6 {
        bail!("CIFTI-2 NIfTI container must expose matrix dimensions in dim[5] and dim[6]");
    }
    let shape = vec![header.dim[5] as usize, header.dim[6] as usize];
    let data_len = shape.iter().product::<usize>();
    let extensions = parse_extensions(bytes, header.vox_offset, little_endian)?;
    let data = parse_nifti2_data(bytes, &header, data_len, little_endian)?;
    Ok(ParsedNifti2 {
        header,
        shape,
        data,
        extensions,
    })
}

fn parse_extensions(
    bytes: &[u8],
    vox_offset: usize,
    little_endian: bool,
) -> anyhow::Result<Vec<ParsedExtension>> {
    if vox_offset < 544 || bytes.len() < vox_offset {
        bail!("Invalid NIfTI-2 vox_offset");
    }
    let extender = &bytes[540..544];
    if extender[0] == 0 {
        return Ok(Vec::new());
    }
    let mut cursor = 544usize;
    let mut extensions = Vec::new();
    while cursor + 8 <= vox_offset {
        let esize = read_i32(bytes, cursor, little_endian)? as usize;
        let ecode = read_i32(bytes, cursor + 4, little_endian)?;
        if esize < 8 || cursor + esize > vox_offset {
            break;
        }
        extensions.push(ParsedExtension {
            ecode,
            data: bytes[cursor + 8..cursor + esize].to_vec(),
        });
        cursor += esize;
    }
    Ok(extensions)
}

fn parse_nifti2_data(
    bytes: &[u8],
    header: &ParsedNifti2Header,
    data_len: usize,
    little_endian: bool,
) -> anyhow::Result<Vec<f32>> {
    let slope = if header.scl_slope == 0.0 { 1.0 } else { header.scl_slope };
    let inter = header.scl_inter;
    let size_of = datatype_size(header.datatype)?;
    let end = header.vox_offset + size_of * data_len;
    if end > bytes.len() {
        bail!("NIfTI-2 data section is truncated");
    }
    let data_bytes = &bytes[header.vox_offset..end];
    let mut values = Vec::with_capacity(data_len);
    for index in 0..data_len {
        let offset = index * size_of;
        let raw = match header.datatype {
            2 => data_bytes[offset] as f64,
            4 => read_i16(data_bytes, offset, little_endian)? as f64,
            8 => read_i32(data_bytes, offset, little_endian)? as f64,
            16 => read_f32(data_bytes, offset, little_endian)? as f64,
            64 => read_f64(data_bytes, offset, little_endian)?,
            256 => (data_bytes[offset] as i8) as f64,
            512 => read_u16(data_bytes, offset, little_endian)? as f64,
            768 => read_u32(data_bytes, offset, little_endian)? as f64,
            1024 => read_i64(data_bytes, offset, little_endian)? as f64,
            1280 => read_u64(data_bytes, offset, little_endian)? as f64,
            other => bail!("Unsupported CIFTI datatype code {other}"),
        };
        values.push((raw * slope + inter) as f32);
    }
    Ok(values)
}

fn datatype_size(datatype: u16) -> anyhow::Result<usize> {
    match datatype {
        2 | 256 => Ok(1),
        4 | 512 => Ok(2),
        8 | 16 | 768 => Ok(4),
        64 | 1024 | 1280 => Ok(8),
        other => bail!("Unsupported CIFTI datatype code {other}"),
    }
}

fn squeeze_trailing_singletons_dims(shape: &[usize]) -> anyhow::Result<Vec<usize>> {
    let shape: Vec<usize> = shape
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, dim)| (index < 2 || dim != 1).then_some(dim))
        .collect();
    if shape.len() < 2 {
        bail!("CIFTI matrix has fewer than two dimensions");
    }
    Ok(shape)
}

fn extract_map_values(
    data: &[f32],
    shape: &[usize],
    map_axis_dim: usize,
    map_index: usize,
) -> anyhow::Result<Vec<f32>> {
    if shape.len() != 2 {
        bail!("CIFTI matrix shape must be 2D");
    }
    let dim0 = shape[0];
    let dim1 = shape[1];
    if map_axis_dim == 0 {
        Ok((0..dim1)
            .map(|other| data[map_index + dim0 * other])
            .collect())
    } else {
        let start = dim0 * map_index;
        let end = start + dim0;
        data.get(start..end)
            .map(|slice| slice.to_vec())
            .ok_or_else(|| anyhow::anyhow!("Map index {} exceeds matrix shape", map_index + 1))
    }
}

fn metadata_for_map(axis: &CiftiAxis, map_index: usize, map_values: &[f32]) -> ScalarMetadata {
    match axis {
        CiftiAxis::Scalars(names) => ScalarMetadata {
            map_name: names
                .get(map_index)
                .cloned()
                .unwrap_or_else(|| format!("map {}", map_index + 1)),
            suggested_range: Some(robust_range(map_values)),
            series_index: None,
            series_value: None,
            label_table: Vec::new(),
        },
        CiftiAxis::Series(series) => ScalarMetadata {
            map_name: format!("t={:.3}", series.start + series.step * map_index as f32),
            suggested_range: Some(robust_range(map_values)),
            series_index: Some(map_index),
            series_value: Some(series.start + series.step * map_index as f32),
            label_table: Vec::new(),
        },
        CiftiAxis::Labels(maps) => {
            let map = maps.get(map_index);
            ScalarMetadata {
                map_name: map
                    .map(|entry| entry.map_name.clone())
                    .unwrap_or_else(|| format!("label {}", map_index + 1)),
                suggested_range: None,
                series_index: None,
                series_value: None,
                label_table: map.map(|entry| entry.label_table.clone()).unwrap_or_default(),
            }
        }
        _ => ScalarMetadata {
            map_name: format!("map {}", map_index + 1),
            suggested_range: Some(robust_range(map_values)),
            series_index: None,
            series_value: None,
            label_table: Vec::new(),
        },
    }
}

#[derive(Clone, Debug)]
struct ExpandedMap {
    left: Option<SurfaceScalars>,
    right: Option<SurfaceScalars>,
    subcortical: Option<VolumeScalars>,
}

fn expand_dense_brain_models(
    models: &BrainModelsAxis,
    map_values: &[f32],
    metadata: &ScalarMetadata,
) -> anyhow::Result<ExpandedMap> {
    let mut left = models.left_vertices.map(|count| vec![f32::NAN; count]);
    let mut right = models.right_vertices.map(|count| vec![f32::NAN; count]);
    let mut subcortical = models
        .volume_dims
        .map(|dims| vec![f32::NAN; dims[0] * dims[1] * dims[2]]);
    for model in &models.models {
        let start = model.index_offset;
        let end = start + model.index_count;
        let values = map_values
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("Brain model slice exceeds map length"))?;
        match model.kind {
            BrainModelKind::Surface(CiftiStructure::CortexLeft) => {
                if let Some(buffer) = left.as_mut() {
                    for (value, vertex) in values.iter().zip(model.vertex_indices.iter()) {
                        if let Some(slot) = buffer.get_mut(*vertex) {
                            *slot = *value;
                        }
                    }
                }
            }
            BrainModelKind::Surface(CiftiStructure::CortexRight) => {
                if let Some(buffer) = right.as_mut() {
                    for (value, vertex) in values.iter().zip(model.vertex_indices.iter()) {
                        if let Some(slot) = buffer.get_mut(*vertex) {
                            *slot = *value;
                        }
                    }
                }
            }
            BrainModelKind::Voxels => {
                if let (Some(buffer), Some(dims)) = (subcortical.as_mut(), models.volume_dims) {
                    for (value, ijk) in values.iter().zip(model.voxel_indices.iter()) {
                        let flat = ijk[0] + dims[0] * (ijk[1] + dims[1] * ijk[2]);
                        if let Some(slot) = buffer.get_mut(flat) {
                            *slot = *value;
                        }
                    }
                }
            }
            BrainModelKind::Surface(CiftiStructure::Subcortical) => {}
        }
    }
    Ok(ExpandedMap {
        left: left.map(|values| SurfaceScalars {
            structure: Some(CiftiStructure::CortexLeft),
            source_surface_id: None,
            vertex_count: values.len(),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
        right: right.map(|values| SurfaceScalars {
            structure: Some(CiftiStructure::CortexRight),
            source_surface_id: None,
            vertex_count: values.len(),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
        subcortical: subcortical.map(|values| VolumeScalars {
            dims: models.volume_dims.unwrap_or([0; 3]),
            voxel_to_ras: models.volume_to_ras.unwrap_or(Mat4::IDENTITY),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
    })
}

fn expand_parcels(
    parcels: &ParcelsAxis,
    map_values: &[f32],
    metadata: &ScalarMetadata,
) -> anyhow::Result<ExpandedMap> {
    let mut left = parcels.left_vertices.map(|count| vec![f32::NAN; count]);
    let mut right = parcels.right_vertices.map(|count| vec![f32::NAN; count]);
    let mut subcortical = parcels
        .volume_dims
        .map(|dims| vec![f32::NAN; dims[0] * dims[1] * dims[2]]);
    for (parcel_index, parcel) in parcels.parcels.iter().enumerate() {
        let Some(&value) = map_values.get(parcel_index) else {
            bail!("Parcel map is shorter than parcel axis");
        };
        if let Some(buffer) = left.as_mut() {
            for &vertex in &parcel.left_vertices {
                if let Some(slot) = buffer.get_mut(vertex) {
                    *slot = value;
                }
            }
        }
        if let Some(buffer) = right.as_mut() {
            for &vertex in &parcel.right_vertices {
                if let Some(slot) = buffer.get_mut(vertex) {
                    *slot = value;
                }
            }
        }
        if let (Some(buffer), Some(dims)) = (subcortical.as_mut(), parcels.volume_dims) {
            for ijk in &parcel.voxel_indices {
                let flat = ijk[0] + dims[0] * (ijk[1] + dims[1] * ijk[2]);
                if let Some(slot) = buffer.get_mut(flat) {
                    *slot = value;
                }
            }
        }
    }
    Ok(ExpandedMap {
        left: left.map(|values| SurfaceScalars {
            structure: Some(CiftiStructure::CortexLeft),
            source_surface_id: None,
            vertex_count: values.len(),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
        right: right.map(|values| SurfaceScalars {
            structure: Some(CiftiStructure::CortexRight),
            source_surface_id: None,
            vertex_count: values.len(),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
        subcortical: subcortical.map(|values| VolumeScalars {
            dims: parcels.volume_dims.unwrap_or([0; 3]),
            voxel_to_ras: parcels.volume_to_ras.unwrap_or(Mat4::IDENTITY),
            values,
            kind: scalar_kind_from_metadata(metadata),
            metadata: metadata.clone(),
        }),
    })
}

fn scalar_kind_from_metadata(metadata: &ScalarMetadata) -> ScalarKind {
    if metadata.label_table.is_empty() {
        ScalarKind::Continuous
    } else {
        ScalarKind::Label
    }
}

#[derive(Clone, Debug)]
struct CiftiXml {
    axes: HashMap<usize, CiftiAxis>,
}

#[derive(Clone, Debug)]
enum CiftiAxis {
    BrainModels(BrainModelsAxis),
    Parcels(ParcelsAxis),
    Scalars(Vec<String>),
    Series(SeriesAxis),
    Labels(Vec<LabelMap>),
}

#[derive(Clone, Debug)]
struct LabelMap {
    map_name: String,
    label_table: Vec<LabelEntry>,
}

#[derive(Clone, Copy, Debug)]
struct SeriesAxis {
    start: f32,
    step: f32,
}

#[derive(Clone, Debug)]
struct BrainModelsAxis {
    models: Vec<BrainModel>,
    left_vertices: Option<usize>,
    right_vertices: Option<usize>,
    volume_dims: Option<[usize; 3]>,
    volume_to_ras: Option<Mat4>,
}

impl BrainModelsAxis {
    fn grayordinate_count(&self) -> usize {
        self.models.iter().map(|model| model.index_count).sum()
    }
}

#[derive(Clone, Debug)]
struct BrainModel {
    kind: BrainModelKind,
    index_offset: usize,
    index_count: usize,
    vertex_indices: Vec<usize>,
    voxel_indices: Vec<[usize; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrainModelKind {
    Surface(CiftiStructure),
    Voxels,
}

#[derive(Clone, Debug)]
struct ParcelsAxis {
    parcels: Vec<Parcel>,
    left_vertices: Option<usize>,
    right_vertices: Option<usize>,
    volume_dims: Option<[usize; 3]>,
    volume_to_ras: Option<Mat4>,
}

#[derive(Clone, Debug)]
struct Parcel {
    left_vertices: Vec<usize>,
    right_vertices: Vec<usize>,
    voxel_indices: Vec<[usize; 3]>,
}

fn parse_cifti_xml(xml_text: &str) -> anyhow::Result<CiftiXml> {
    let document = Document::parse(xml_text).context("Failed to parse CIFTI XML")?;
    let root = document
        .descendants()
        .find(|node| node.has_tag_name("CIFTI"))
        .ok_or_else(|| anyhow::anyhow!("CIFTI XML root not found"))?;
    let matrix = root
        .children()
        .find(|node| node.has_tag_name("Matrix"))
        .ok_or_else(|| anyhow::anyhow!("CIFTI Matrix node not found"))?;
    let mut axes = HashMap::new();
    for axis_node in matrix.children().filter(|node| node.has_tag_name("MatrixIndicesMap")) {
        let dim = attr_usize(axis_node, "AppliesToMatrixDimension")?;
        let axis_type = axis_node
            .attribute("IndicesMapToDataType")
            .ok_or_else(|| anyhow::anyhow!("MatrixIndicesMap missing IndicesMapToDataType"))?;
        let axis = match axis_type {
            "CIFTI_INDEX_TYPE_BRAIN_MODELS" => {
                CiftiAxis::BrainModels(parse_brain_models_axis(axis_node)?)
            }
            "CIFTI_INDEX_TYPE_PARCELS" => CiftiAxis::Parcels(parse_parcels_axis(axis_node)?),
            "CIFTI_INDEX_TYPE_SCALARS" => CiftiAxis::Scalars(parse_scalar_axis(axis_node)),
            "CIFTI_INDEX_TYPE_SERIES" => CiftiAxis::Series(parse_series_axis(axis_node)?),
            "CIFTI_INDEX_TYPE_LABELS" => CiftiAxis::Labels(parse_label_axis(axis_node)),
            other => bail!("Unsupported CIFTI axis type {other}"),
        };
        axes.insert(dim, axis);
    }
    Ok(CiftiXml { axes })
}

fn parse_brain_models_axis(node: Node<'_, '_>) -> anyhow::Result<BrainModelsAxis> {
    let mut models = Vec::new();
    let mut left_vertices = None;
    let mut right_vertices = None;
    let mut volume_dims = None;
    let mut volume_to_ras = None;
    for child in node.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "BrainModel" => {
                let structure_name = child
                    .attribute("BrainStructure")
                    .ok_or_else(|| anyhow::anyhow!("BrainModel missing BrainStructure"))?;
                let kind = match child
                    .attribute("ModelType")
                    .ok_or_else(|| anyhow::anyhow!("BrainModel missing ModelType"))?
                {
                    "CIFTI_MODEL_TYPE_SURFACE" => {
                        BrainModelKind::Surface(map_structure(structure_name))
                    }
                    "CIFTI_MODEL_TYPE_VOXELS" => BrainModelKind::Voxels,
                    other => bail!("Unsupported BrainModel type {other}"),
                };
                let index_offset = attr_usize(child, "IndexOffset")?;
                let index_count = attr_usize(child, "IndexCount")?;
                let surface_number_of_vertices =
                    child.attribute("SurfaceNumberOfVertices").and_then(|value| value.parse().ok());
                match kind {
                    BrainModelKind::Surface(CiftiStructure::CortexLeft) => {
                        left_vertices = left_vertices.or(surface_number_of_vertices);
                    }
                    BrainModelKind::Surface(CiftiStructure::CortexRight) => {
                        right_vertices = right_vertices.or(surface_number_of_vertices);
                    }
                    _ => {}
                }
                let vertex_indices = child
                    .children()
                    .find(|entry| entry.has_tag_name("VertexIndices"))
                    .map(parse_index_list)
                    .transpose()?
                    .unwrap_or_default();
                let voxel_indices = child
                    .children()
                    .find(|entry| entry.has_tag_name("VoxelIndicesIJK"))
                    .map(parse_triplet_list)
                    .transpose()?
                    .unwrap_or_default();
                models.push(BrainModel {
                    kind,
                    index_offset,
                    index_count,
                    vertex_indices,
                    voxel_indices,
                });
            }
            "Volume" => {
                volume_dims = child
                    .attribute("VolumeDimensions")
                    .map(parse_dims_attr)
                    .transpose()?;
            }
            "TransformationMatrixVoxelIndicesIJKtoXYZ" => {
                volume_to_ras = Some(parse_volume_transform(child)?);
                volume_dims = volume_dims.or_else(|| {
                    child
                        .parent()
                        .and_then(|parent| parent.attribute("VolumeDimensions"))
                        .map(parse_dims_attr)
                        .transpose()
                        .ok()
                        .flatten()
                });
            }
            _ => {}
        }
    }
    if let Some(dims) = node.attribute("VolumeDimensions") {
        volume_dims = Some(parse_dims_attr(dims)?);
    }
    Ok(BrainModelsAxis {
        models,
        left_vertices,
        right_vertices,
        volume_dims,
        volume_to_ras,
    })
}

fn parse_parcels_axis(node: Node<'_, '_>) -> anyhow::Result<ParcelsAxis> {
    let mut parcels = Vec::new();
    let mut left_vertices = None;
    let mut right_vertices = None;
    let mut volume_dims = None;
    let mut volume_to_ras = None;
    for child in node.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "Surface" => {
                let structure = map_structure(
                    child
                        .attribute("BrainStructure")
                        .ok_or_else(|| anyhow::anyhow!("Surface missing BrainStructure"))?,
                );
                let count = attr_usize(child, "SurfaceNumberOfVertices")?;
                match structure {
                    CiftiStructure::CortexLeft => left_vertices = Some(count),
                    CiftiStructure::CortexRight => right_vertices = Some(count),
                    CiftiStructure::Subcortical => {}
                }
            }
            "Parcel" => parcels.push(parse_parcel(child)?),
            "Volume" => {
                volume_dims = Some(parse_dims_attr(
                    child
                        .attribute("VolumeDimensions")
                        .ok_or_else(|| anyhow::anyhow!("Volume missing VolumeDimensions"))?,
                )?);
            }
            "TransformationMatrixVoxelIndicesIJKtoXYZ" => {
                volume_to_ras = Some(parse_volume_transform(child)?);
            }
            _ => {}
        }
    }
    Ok(ParcelsAxis {
        parcels,
        left_vertices,
        right_vertices,
        volume_dims,
        volume_to_ras,
    })
}

fn parse_parcel(node: Node<'_, '_>) -> anyhow::Result<Parcel> {
    let mut left_vertices = Vec::new();
    let mut right_vertices = Vec::new();
    let mut voxel_indices = Vec::new();
    for child in node.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "Vertices" => {
                let structure = map_structure(
                    child
                        .attribute("BrainStructure")
                        .ok_or_else(|| anyhow::anyhow!("Vertices missing BrainStructure"))?,
                );
                let vertices = parse_index_list(child)?;
                match structure {
                    CiftiStructure::CortexLeft => left_vertices.extend(vertices),
                    CiftiStructure::CortexRight => right_vertices.extend(vertices),
                    CiftiStructure::Subcortical => {}
                }
            }
            "VoxelIndicesIJK" => voxel_indices.extend(parse_triplet_list(child)?),
            _ => {}
        }
    }
    Ok(Parcel {
        left_vertices,
        right_vertices,
        voxel_indices,
    })
}

fn parse_scalar_axis(node: Node<'_, '_>) -> Vec<String> {
    node.children()
        .filter(|child| child.has_tag_name("NamedMap"))
        .map(|named_map| {
            named_map
                .children()
                .find(|child| child.has_tag_name("MapName"))
                .and_then(|entry| entry.text())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("map")
                .to_string()
        })
        .collect()
}

fn parse_label_axis(node: Node<'_, '_>) -> Vec<LabelMap> {
    node.children()
        .filter(|child| child.has_tag_name("NamedMap"))
        .map(|named_map| {
            let map_name = named_map
                .children()
                .find(|child| child.has_tag_name("MapName"))
                .and_then(|entry| entry.text())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("label")
                .to_string();
            let label_table = named_map
                .children()
                .find(|child| child.has_tag_name("LabelTable"))
                .map(|table| {
                    table
                        .children()
                        .filter(|entry| entry.has_tag_name("Label"))
                        .filter_map(|entry| {
                            Some(LabelEntry {
                                key: entry.attribute("Key")?.parse().ok()?,
                                name: entry.text().unwrap_or("").trim().to_string(),
                                rgba: [
                                    entry.attribute("Red")?.parse().ok()?,
                                    entry.attribute("Green")?.parse().ok()?,
                                    entry.attribute("Blue")?.parse().ok()?,
                                    entry.attribute("Alpha")?.parse().ok()?,
                                ],
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            LabelMap {
                map_name,
                label_table,
            }
        })
        .collect()
}

fn parse_series_axis(node: Node<'_, '_>) -> anyhow::Result<SeriesAxis> {
    Ok(SeriesAxis {
        start: node
            .attribute("SeriesStart")
            .ok_or_else(|| anyhow::anyhow!("Series axis missing SeriesStart"))?
            .parse()
            .context("Invalid SeriesStart")?,
        step: node
            .attribute("SeriesStep")
            .ok_or_else(|| anyhow::anyhow!("Series axis missing SeriesStep"))?
            .parse()
            .context("Invalid SeriesStep")?,
    })
}

fn parse_index_list(node: Node<'_, '_>) -> anyhow::Result<Vec<usize>> {
    node.text()
        .unwrap_or("")
        .split_whitespace()
        .map(|value| value.parse().context("Invalid vertex index"))
        .collect()
}

fn parse_triplet_list(node: Node<'_, '_>) -> anyhow::Result<Vec<[usize; 3]>> {
    let values: Vec<usize> = node
        .text()
        .unwrap_or("")
        .split_whitespace()
        .map(|value| value.parse().context("Invalid voxel index"))
        .collect::<Result<_, _>>()?;
    if values.len() % 3 != 0 {
        bail!("VoxelIndicesIJK must have a multiple-of-3 length");
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn parse_dims_attr(value: &str) -> anyhow::Result<[usize; 3]> {
    let dims: Vec<usize> = value
        .split(',')
        .flat_map(|part| part.split_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().context("Invalid volume dimension"))
        .collect::<Result<_, _>>()?;
    if dims.len() != 3 {
        bail!("VolumeDimensions must contain three integers");
    }
    Ok([dims[0], dims[1], dims[2]])
}

fn parse_volume_transform(node: Node<'_, '_>) -> anyhow::Result<Mat4> {
    let values: Vec<f32> = node
        .text()
        .unwrap_or("")
        .split_whitespace()
        .map(|value| value.parse().context("Invalid transform value"))
        .collect::<Result<_, _>>()?;
    if values.len() != 16 {
        bail!("TransformationMatrixVoxelIndicesIJKtoXYZ must contain 16 values");
    }
    Ok(Mat4::from_cols(
        Vec4::new(values[0], values[4], values[8], values[12]),
        Vec4::new(values[1], values[5], values[9], values[13]),
        Vec4::new(values[2], values[6], values[10], values[14]),
        Vec4::new(values[3], values[7], values[11], values[15]),
    ))
}

fn attr_usize(node: Node<'_, '_>, name: &str) -> anyhow::Result<usize> {
    node.attribute(name)
        .ok_or_else(|| anyhow::anyhow!("Missing {name}"))
        .and_then(|value| value.parse().context(format!("Invalid {name}")))
}

fn map_structure(value: &str) -> CiftiStructure {
    let lower = value.to_ascii_lowercase();
    if lower.contains("cortex_left") {
        CiftiStructure::CortexLeft
    } else if lower.contains("cortex_right") {
        CiftiStructure::CortexRight
    } else {
        CiftiStructure::Subcortical
    }
}

fn robust_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((finite.len() as f32) * 0.02).floor() as usize;
    let hi_idx = ((finite.len() as f32) * 0.98).floor() as usize;
    let lo = finite[lo_idx.min(finite.len() - 1)];
    let hi = finite[hi_idx.min(finite.len() - 1)].max(lo + 1e-6);
    (lo, hi)
}

fn read_u16(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<u16> {
    Ok(if little_endian {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into()?)
    } else {
        u16::from_be_bytes(bytes[offset..offset + 2].try_into()?)
    })
}

fn read_i16(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<i16> {
    Ok(if little_endian {
        i16::from_le_bytes(bytes[offset..offset + 2].try_into()?)
    } else {
        i16::from_be_bytes(bytes[offset..offset + 2].try_into()?)
    })
}

fn read_u32(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<u32> {
    Ok(if little_endian {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into()?)
    } else {
        u32::from_be_bytes(bytes[offset..offset + 4].try_into()?)
    })
}

fn read_i32(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<i32> {
    Ok(if little_endian {
        i32::from_le_bytes(bytes[offset..offset + 4].try_into()?)
    } else {
        i32::from_be_bytes(bytes[offset..offset + 4].try_into()?)
    })
}

fn read_u64(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<u64> {
    Ok(if little_endian {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into()?)
    } else {
        u64::from_be_bytes(bytes[offset..offset + 8].try_into()?)
    })
}

fn read_i64(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<i64> {
    Ok(if little_endian {
        i64::from_le_bytes(bytes[offset..offset + 8].try_into()?)
    } else {
        i64::from_be_bytes(bytes[offset..offset + 8].try_into()?)
    })
}

fn read_f32(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<f32> {
    Ok(if little_endian {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into()?)
    } else {
        f32::from_be_bytes(bytes[offset..offset + 4].try_into()?)
    })
}

fn read_f64(bytes: &[u8], offset: usize, little_endian: bool) -> anyhow::Result<f64> {
    Ok(if little_endian {
        f64::from_le_bytes(bytes[offset..offset + 8].try_into()?)
    } else {
        f64::from_be_bytes(bytes[offset..offset + 8].try_into()?)
    })
}

fn read_i64_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    little_endian: bool,
) -> anyhow::Result<[i64; N]> {
    let mut values = [0i64; N];
    for (index, value) in values.iter_mut().enumerate() {
        *value = read_i64(bytes, offset + index * 8, little_endian)?;
    }
    Ok(values)
}

fn read_f64_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    little_endian: bool,
) -> anyhow::Result<[f64; N]> {
    let mut values = [0.0f64; N];
    for (index, value) in values.iter_mut().enumerate() {
        *value = read_f64(bytes, offset + index * 8, little_endian)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{CiftiIntent, LoadedCifti, ScalarKind};
    use std::path::PathBuf;

    fn sample_path(file_name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../CiftiLib/example/data")
            .join(file_name)
    }

    #[test]
    fn loads_dense_scalar_example() {
        let cifti = LoadedCifti::load(&sample_path("ones.dscalar.nii")).expect("load dscalar");
        assert_eq!(cifti.intent, CiftiIntent::DenseScalar);
        assert!(cifti.map_count >= 1);
        assert!(cifti.left_scalars.iter().any(|entry| entry.is_some()));
        assert!(cifti.right_scalars.iter().any(|entry| entry.is_some()));
    }

    #[test]
    fn loads_dense_series_example_with_series_metadata() {
        let cifti = LoadedCifti::load(&sample_path(
            "Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii",
        ))
        .expect("load dtseries");
        assert_eq!(cifti.intent, CiftiIntent::DenseSeries);
        let first = cifti
            .left_scalars
            .first()
            .and_then(|entry| entry.as_ref())
            .expect("left cortex map");
        assert!(first.metadata.series_index.is_some());
        assert!(first.metadata.series_value.is_some());
    }

    #[test]
    fn loads_dense_label_example_with_label_table() {
        let cifti = LoadedCifti::load(&sample_path(
            "Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii",
        ))
        .expect("load dlabel");
        assert_eq!(cifti.intent, CiftiIntent::DenseLabel);
        let first = cifti
            .left_scalars
            .first()
            .and_then(|entry| entry.as_ref())
            .expect("left cortex labels");
        assert_eq!(first.kind, ScalarKind::Label);
        assert!(!first.metadata.label_table.is_empty());
    }
}

struct ComputeParams {
    full_bins: u32,
    source_bins: u32,
    ncoeffs: u32,
    slice_count: u32,
}

struct FlatF32 {
    values: array<f32>,
}

struct FlatU32 {
    values: array<u32>,
}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var<storage, read> coeffs: FlatF32;
@group(0) @binding(2) var<storage, read> transform: FlatF32;
@group(0) @binding(3) var<storage, read> slice_indices: FlatU32;
@group(0) @binding(4) var<storage, read_write> amplitudes: FlatF32;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.slice_count * params.full_bins;
    let flat_index = gid.x;
    if flat_index >= total {
        return;
    }

    let slice_slot = flat_index / params.full_bins;
    let dir_index = flat_index % params.full_bins;
    let compact_index = slice_indices.values[slice_slot];

    var source_dir_index = dir_index;
    if params.source_bins != params.full_bins {
        source_dir_index = dir_index % params.source_bins;
    }

    let coeff_base = compact_index * params.ncoeffs;
    let transform_base = source_dir_index * params.ncoeffs;
    var value = 0.0;
    for (var coeff_index = 0u; coeff_index < params.ncoeffs; coeff_index = coeff_index + 1u) {
        value = value
            + transform.values[transform_base + coeff_index]
                * coeffs.values[coeff_base + coeff_index];
    }

    amplitudes.values[compact_index * params.full_bins + dir_index] = max(value, 0.0);
}

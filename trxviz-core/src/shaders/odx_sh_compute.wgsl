const ROW_WORKGROUP_SIZE: u32 = 64u;

struct ComputeParams {
    full_bins: u32,
    source_bins: u32,
    ncoeffs: u32,
    slice_count: u32,
    flags: vec4<u32>,
    scales: vec4<f32>,
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

var<workgroup> partial_mins: array<f32, ROW_WORKGROUP_SIZE>;
var<workgroup> partial_sums: array<f32, ROW_WORKGROUP_SIZE>;
var<workgroup> partial_valid: array<u32, ROW_WORKGROUP_SIZE>;

fn is_finite_scalar(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

fn reduce_min(local_id: u32) {
    var stride = ROW_WORKGROUP_SIZE / 2u;
    loop {
        if local_id < stride {
            partial_mins[local_id] = min(partial_mins[local_id], partial_mins[local_id + stride]);
        }
        workgroupBarrier();
        if stride == 1u {
            break;
        }
        stride = stride / 2u;
    }
}

fn reduce_sum_and_valid(local_id: u32) {
    var stride = ROW_WORKGROUP_SIZE / 2u;
    loop {
        if local_id < stride {
            partial_sums[local_id] = partial_sums[local_id] + partial_sums[local_id + stride];
            partial_valid[local_id] = partial_valid[local_id] * partial_valid[local_id + stride];
        }
        workgroupBarrier();
        if stride == 1u {
            break;
        }
        stride = stride / 2u;
    }
}

fn reduce_max(local_id: u32) {
    var stride = ROW_WORKGROUP_SIZE / 2u;
    loop {
        if local_id < stride {
            partial_sums[local_id] = max(partial_sums[local_id], partial_sums[local_id + stride]);
        }
        workgroupBarrier();
        if stride == 1u {
            break;
        }
        stride = stride / 2u;
    }
}

@compute @workgroup_size(64)
fn cs_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let slice_slot = workgroup_id.x;
    if slice_slot >= params.slice_count {
        return;
    }
    let lid = local_id.x;
    let compact_index = slice_indices.values[slice_slot];

    var local_min = 1e30;
    for (var dir_index = lid; dir_index < params.full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
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

        let clamped = max(value, 0.0);
        amplitudes.values[compact_index * params.full_bins + dir_index] = clamped;
        if is_finite_scalar(clamped) {
            local_min = min(local_min, clamped);
        }
    }

    partial_mins[lid] = local_min;
    workgroupBarrier();
    reduce_min(lid);

    let row_min =
        select(0.0, partial_mins[0], params.flags.x != 0u && is_finite_scalar(partial_mins[0]));

    var local_sum = 0.0;
    var local_valid = 1u;
    for (var dir_index = lid; dir_index < params.full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = compact_index * params.full_bins + dir_index;
        let raw = amplitudes.values[output_offset];
        let adjusted = select(raw, raw - row_min, params.flags.x != 0u && is_finite_scalar(raw));
        amplitudes.values[output_offset] = adjusted;
        if is_finite_scalar(adjusted) {
            local_sum = local_sum + adjusted;
        } else {
            local_valid = 0u;
        }
    }

    partial_sums[lid] = local_sum;
    partial_valid[lid] = local_valid;
    workgroupBarrier();
    reduce_sum_and_valid(lid);

    let target_peak_length_mm = params.scales.x;
    let should_normalize =
        params.flags.y != 0u
        && partial_valid[0] != 0u
        && target_peak_length_mm > 0.0;
    if !should_normalize {
        return;
    }

    var row_peak = 0.0;
    for (var dir_index = lid; dir_index < params.full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = compact_index * params.full_bins + dir_index;
        row_peak = max(row_peak, amplitudes.values[output_offset]);
    }
    partial_sums[lid] = row_peak;
    workgroupBarrier();
    reduce_max(lid);
    let norm_peak = partial_sums[0];
    if !is_finite_scalar(norm_peak) || norm_peak <= 0.0 {
        return;
    }
    let scale = target_peak_length_mm / norm_peak;

    for (var dir_index = lid; dir_index < params.full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = compact_index * params.full_bins + dir_index;
        amplitudes.values[output_offset] = amplitudes.values[output_offset] * scale;
    }
}

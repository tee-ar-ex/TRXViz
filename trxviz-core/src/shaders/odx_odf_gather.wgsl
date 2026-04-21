const ROW_WORKGROUP_SIZE: u32 = 64u;

struct ComputeParams {
    counts: vec4<u32>,
    scales: vec4<f32>,
    flags: vec4<u32>,
}

struct FlatF32 {
    values: array<f32>,
}

struct WorkItem {
    local_row: u32,
    output_row: u32,
}

struct WorkItems {
    values: array<WorkItem>,
}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var<storage, read> source_amplitudes: FlatF32;
@group(0) @binding(2) var<storage, read> work_items: WorkItems;
@group(0) @binding(3) var<storage, read_write> output_amplitudes: FlatF32;

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
    let full_bins = params.counts.x;
    let source_bins = params.counts.y;
    let work_item_count = params.counts.z;
    let work_item_index = workgroup_id.x;
    if work_item_index >= work_item_count {
        return;
    }
    let lid = local_id.x;
    let item = work_items.values[work_item_index];

    var local_min = 1e30;
    for (var dir_index = lid; dir_index < full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        var source_dir_index = dir_index;
        if source_bins != full_bins {
            source_dir_index = dir_index % source_bins;
        }
        let source_offset = item.local_row * source_bins + source_dir_index;
        let output_offset = item.output_row * full_bins + dir_index;
        let value = source_amplitudes.values[source_offset];
        output_amplitudes.values[output_offset] = value;
        if is_finite_scalar(value) {
            local_min = min(local_min, value);
        }
    }

    partial_mins[lid] = local_min;
    workgroupBarrier();
    reduce_min(lid);

    let row_min =
        select(0.0, partial_mins[0], params.flags.x != 0u && is_finite_scalar(partial_mins[0]));

    var local_sum = 0.0;
    var local_valid = 1u;
    for (var dir_index = lid; dir_index < full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = item.output_row * full_bins + dir_index;
        let raw = output_amplitudes.values[output_offset];
        let adjusted = select(raw, raw - row_min, params.flags.x != 0u && is_finite_scalar(raw));
        output_amplitudes.values[output_offset] = adjusted;
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

    let target_peak_length_mm = params.scales.y;
    let should_normalize =
        params.flags.y != 0u
        && partial_valid[0] != 0u
        && target_peak_length_mm > 0.0;
    if !should_normalize {
        return;
    }

    var row_peak = 0.0;
    for (var dir_index = lid; dir_index < full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = item.output_row * full_bins + dir_index;
        row_peak = max(row_peak, output_amplitudes.values[output_offset]);
    }
    partial_sums[lid] = row_peak;
    workgroupBarrier();
    reduce_max(lid);
    let norm_peak = partial_sums[0];
    if !is_finite_scalar(norm_peak) || norm_peak <= 0.0 {
        return;
    }
    let scale = target_peak_length_mm / norm_peak;

    for (var dir_index = lid; dir_index < full_bins; dir_index = dir_index + ROW_WORKGROUP_SIZE) {
        let output_offset = item.output_row * full_bins + dir_index;
        output_amplitudes.values[output_offset] = output_amplitudes.values[output_offset] * scale;
    }
}

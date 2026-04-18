struct ComputeParams {
    counts: vec4<u32>,
    scales: vec4<f32>,
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

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_bins = params.counts.x;
    let source_bins = params.counts.y;
    let work_item_count = params.counts.z;
    let flat_index = gid.x;
    let total = work_item_count * full_bins;
    if flat_index >= total {
        return;
    }

    let work_item_index = flat_index / full_bins;
    let dir_index = flat_index % full_bins;
    let item = work_items.values[work_item_index];
    var source_dir_index = dir_index;
    if source_bins != full_bins {
        source_dir_index = dir_index % source_bins;
    }

    let amp_norm = max(params.scales.x, 1e-6);
    let source_offset = item.local_row * source_bins + source_dir_index;
    let output_offset = item.output_row * full_bins + dir_index;
    output_amplitudes.values[output_offset] = source_amplitudes.values[source_offset] / amp_norm;
}

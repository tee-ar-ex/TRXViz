// tractography_prob.wgsl — GPU probabilistic tractography, sparse SH data.
//
// One workgroup = one seed.  Workgroup size: (32, 1, 1).
// Lane 0 handles all sequential decisions; all 32 lanes cooperate on ODF
// evaluation (each lane covers n_dirs/32 directions, strided).
//
// Coordinate system: seeds and output points are in ODX voxel space.
// The Rust host converts seed RAS→vox before upload and vox→RAS on readback.
//
// Output layout (per seed):
//   out_points  [ seed * 2 * max_pts * 3 + dir * max_pts * 3 + step * 3 + xyz ]
//     dir=0 : backward segment (max_pts slots)
//     dir=1 : forward  segment (max_pts slots)
//   out_lengths [ seed * 2 + 0 ] = backward_len
//   out_lengths [ seed * 2 + 1 ] = forward_len

// ── constants ────────────────────────────────────────────────────────
const THR_X: u32 = 32u;
const INVALID_VOXEL: u32 = 0xFFFFFFFFu;
// Workgroup ODF buffer must be >= n_dirs.  Detail-2 icosphere has 162 verts;
// we size for 256 to leave headroom without wasting memory.
const WG_ODF_SIZE: u32 = 256u;

// ── parameter struct (16-byte padded for var<uniform>) ───────────────
struct TractographyParams {
    batch_size:               u32,
    ncoeffs:                  u32,
    n_dirs:                   u32,
    dimx:                     u32,
    dimy:                     u32,
    dimz:                     u32,
    max_points:               u32,
    batch_offset:             u32,
    step_size_vox:            f32,
    max_angle_cos:            f32,
    relative_peak_threshold:  f32,
    rng_seed:                 u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,  // 16-byte boundary
}

// ── bindings: group 0 (static + per-batch input) ─────────────────────
@group(0) @binding(0) var<uniform>      params:       TractographyParams;
@group(0) @binding(1) var<storage, read> seeds_vox:   array<f32>;  // [batch * 3]
@group(0) @binding(2) var<storage, read> sh_coeffs:   array<f32>;  // [NB_VOXELS * ncoeffs]
@group(0) @binding(3) var<storage, read> b_matrix:    array<f32>;  // [n_dirs * ncoeffs]
@group(0) @binding(4) var<storage, read> lut:         array<u32>;  // [dimx*dimy*dimz]
@group(0) @binding(5) var<storage, read> sphere_verts: array<f32>; // [n_dirs * 3]

// ── bindings: group 1 (output per batch) ─────────────────────────────
@group(1) @binding(0) var<storage, read_write> out_points:  array<f32>; // [batch * 2 * max_pts * 3]
@group(1) @binding(1) var<storage, read_write> out_lengths: array<u32>; // [batch * 2]

// ── workgroup shared memory ───────────────────────────────────────────
var<workgroup> wg_odf:  array<f32, 256>;  // ODF PMF for current step
var<workgroup> wg_pt:   array<f32, 3>;    // current tracking point (xyz)
var<workgroup> wg_dir:  array<f32, 3>;    // chosen direction (xyz)
var<workgroup> wg_stop: u32;              // 1 = stop tracking

// ── LCG RNG (reproducible, matches cpu_tractography.rs) ──────────────
fn lcg_step(state: u32) -> u32 {
    return state * 1664525u + 1013904223u;
}

fn lcg_f32(state: ptr<function, u32>) -> f32 {
    *state = lcg_step(*state);
    return f32(*state >> 8u) / f32(1u << 24u);
}

// ── helpers ───────────────────────────────────────────────────────────
fn load_sphere_vert(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sphere_verts[base], sphere_verts[base + 1u], sphere_verts[base + 2u]);
}

fn lut_index(xi: i32, yi: i32, zi: i32) -> u32 {
    return u32(xi) * params.dimy * params.dimz
         + u32(yi) * params.dimz
         + u32(zi);
}

// ── ODF evaluation at voxel-space point ───────────────────────────────
// All 32 threads call this; each thread fills wg_odf[t] for t = tidx, tidx+32, …
// Returns true if total trilinear weight is large enough (point is in-mask).
fn eval_odf(pt: vec3<f32>, tidx: u32) -> bool {
    let ix0 = i32(floor(pt.x));
    let iy0 = i32(floor(pt.y));
    let iz0 = i32(floor(pt.z));
    let wx1 = pt.x - f32(ix0);  let wx0 = 1.0 - wx1;
    let wy1 = pt.y - f32(iy0);  let wy0 = 1.0 - wy1;
    let wz1 = pt.z - f32(iz0);  let wz0 = 1.0 - wz1;

    let dimx = i32(params.dimx);
    let dimy = i32(params.dimy);
    let dimz = i32(params.dimz);

    // Share total weight via wg_odf slot [n_dirs] (just past ODF values)
    // Thread 0 will compute total_w at the end.

    var t = i32(tidx);
    loop {
        if (t >= i32(params.n_dirs)) { break; }

        var odf_t: f32 = 0.0;
        var total_w: f32 = 0.0;

        for (var dx = 0; dx < 2; dx++) {
        for (var dy = 0; dy < 2; dy++) {
        for (var dz = 0; dz < 2; dz++) {
            let xi = ix0 + dx;
            let yi = iy0 + dy;
            let zi = iz0 + dz;
            if (xi < 0 || yi < 0 || zi < 0 ||
                xi >= dimx || yi >= dimy || zi >= dimz) { continue; }

            let compact = lut[lut_index(xi, yi, zi)];
            if (compact == INVALID_VOXEL) { continue; }

            let wx = select(wx0, wx1, dx == 1);
            let wy = select(wy0, wy1, dy == 1);
            let wz = select(wz0, wz1, dz == 1);
            let w = wx * wy * wz;
            if (w <= 0.0) { continue; }

            total_w += w;
            // Dot sh_coeffs[compact * K .. +K] with b_matrix[t * K .. +K]
            let sh_base = compact * params.ncoeffs;
            let b_base  = u32(t) * params.ncoeffs;
            var dot_val: f32 = 0.0;
            for (var k = 0u; k < params.ncoeffs; k++) {
                dot_val += sh_coeffs[sh_base + k] * b_matrix[b_base + k];
            }
            odf_t += w * max(0.0, dot_val);
        } } }

        // Store ODF value; zero if outside mask
        wg_odf[u32(t)] = select(0.0, odf_t, total_w >= 0.01);
        t += i32(THR_X);
    }
    workgroupBarrier();

    // Lane 0 checks if there's any ODF mass to track through
    // (approximate mask check: if all ODF values are zero, we're outside)
    if (tidx == 0u) {
        var any_nonzero = false;
        for (var i = 0u; i < params.n_dirs; i++) {
            if (wg_odf[i] > 0.0) { any_nonzero = true; break; }
        }
        wg_stop = select(0u, 1u, !any_nonzero);
    }
    workgroupBarrier();
    return wg_stop == 0u;
}

// ── direction sampling (lane 0 only) ─────────────────────────────────
// Reads wg_odf, applies angle mask and relative threshold, CDF-samples.
// Writes result to wg_dir; sets wg_stop=1 if no valid direction found.
// prev_dir is ignored when is_start=true.
fn sample_dir_lane0(prev_dir: vec3<f32>, is_start: bool, rng: ptr<function, u32>) {
    let n_dirs = params.n_dirs;

    // 1. Find max ODF value
    var max_val: f32 = 0.0;
    for (var t = 0u; t < n_dirs; t++) {
        max_val = max(max_val, wg_odf[t]);
    }
    if (max_val <= 0.0) { wg_stop = 1u; return; }

    // 2. Relative threshold + optional angle mask
    let thresh = max_val * params.relative_peak_threshold;
    var total: f32 = 0.0;
    for (var t = 0u; t < n_dirs; t++) {
        var v = wg_odf[t];
        if (v < thresh) { v = 0.0; }
        if (!is_start && v > 0.0) {
            let sv = load_sphere_vert(t);
            let dot_abs = abs(prev_dir.x * sv.x + prev_dir.y * sv.y + prev_dir.z * sv.z);
            if (dot_abs < params.max_angle_cos) { v = 0.0; }
        }
        wg_odf[t] = v;
        total += v;
    }
    if (total <= 0.0) { wg_stop = 1u; return; }

    // 3. CDF sampling
    let r = lcg_f32(rng) * total;
    var cumsum: f32 = 0.0;
    var chosen = n_dirs - 1u;
    for (var t = 0u; t < n_dirs; t++) {
        cumsum += wg_odf[t];
        if (cumsum >= r) { chosen = t; break; }
    }

    // 4. Choose direction, flip to match prev_dir hemisphere
    var sv = load_sphere_vert(chosen);
    if (!is_start) {
        let d = prev_dir.x * sv.x + prev_dir.y * sv.y + prev_dir.z * sv.z;
        if (d < 0.0) { sv = -sv; }
    }

    // Normalize
    let len = sqrt(sv.x * sv.x + sv.y * sv.y + sv.z * sv.z);
    if (len > 0.0) { sv /= len; }

    wg_dir[0] = sv.x;
    wg_dir[1] = sv.y;
    wg_dir[2] = sv.z;
    wg_stop = 0u;
}

// ── track one half (backward or forward) ─────────────────────────────
// tidx: thread index (0..31)
// seed_vox: starting point in voxel space
// init_dir: first step direction
// out_base: base index into out_points for this seed+direction (in f32 units)
// Returns number of points written (including the seed point at step 0)
fn track_half(
    tidx: u32,
    seed_vox: vec3<f32>,
    init_dir: vec3<f32>,
    out_base: u32,
    rng: ptr<function, u32>,
) -> u32 {
    let max_pts = params.max_points;
    var npts: u32 = 0u;

    // Write seed point (only written for the forward half to avoid duplicates)
    if (tidx == 0u) {
        wg_pt[0] = seed_vox.x + init_dir.x * params.step_size_vox;
        wg_pt[1] = seed_vox.y + init_dir.y * params.step_size_vox;
        wg_pt[2] = seed_vox.z + init_dir.z * params.step_size_vox;
        wg_dir[0] = init_dir.x;
        wg_dir[1] = init_dir.y;
        wg_dir[2] = init_dir.z;
        wg_stop = 0u;
    }
    workgroupBarrier();

    var direction = init_dir;

    loop {
        if (npts >= max_pts) { break; }

        let pt = vec3<f32>(wg_pt[0], wg_pt[1], wg_pt[2]);

        // All 32 threads evaluate ODF at current point
        let in_mask = eval_odf(pt, tidx);

        // Lane 0 samples direction
        if (tidx == 0u) {
            if (!in_mask) {
                wg_stop = 1u;
            } else {
                let prev_dir = vec3<f32>(wg_dir[0], wg_dir[1], wg_dir[2]);
                sample_dir_lane0(prev_dir, false, rng);
            }
        }
        workgroupBarrier();

        if (wg_stop != 0u) { break; }

        // Lane 0 writes current point and advances
        if (tidx == 0u) {
            let off = out_base + npts * 3u;
            out_points[off]      = wg_pt[0];
            out_points[off + 1u] = wg_pt[1];
            out_points[off + 2u] = wg_pt[2];
            // Advance
            let step = params.step_size_vox;
            wg_pt[0] += wg_dir[0] * step;
            wg_pt[1] += wg_dir[1] * step;
            wg_pt[2] += wg_dir[2] * step;
        }
        workgroupBarrier();

        npts += 1u;
    }

    return npts;
}

// ── kernel entry point ────────────────────────────────────────────────
@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
) {
    let seed_idx = global_id.x / THR_X;   // which seed in this batch
    let tidx     = local_id.x;            // lane within workgroup (0..31)

    if (seed_idx >= params.batch_size) { return; }

    // Load seed in voxel space
    let seed_base = seed_idx * 3u;
    let seed_vox = vec3<f32>(
        seeds_vox[seed_base],
        seeds_vox[seed_base + 1u],
        seeds_vox[seed_base + 2u],
    );

    // Per-seed RNG seed (lane 0 only uses it, but all initialize for clarity)
    var rng = lcg_step(params.rng_seed ^ (params.batch_offset + seed_idx));
    rng = lcg_step(rng);  // warm up

    // Output base offsets (f32 index into out_points)
    let pts_per_dir = params.max_points;
    let back_base = seed_idx * 2u * pts_per_dir * 3u;               // dir=0
    let fwd_base  = seed_idx * 2u * pts_per_dir * 3u + pts_per_dir * 3u; // dir=1

    // ── sample initial direction at seed ──────────────────────────────
    let in_mask_seed = eval_odf(seed_vox, tidx);
    if (tidx == 0u) {
        if (!in_mask_seed) {
            wg_stop = 1u;
        } else {
            let zero_dir = vec3<f32>(0.0, 0.0, 0.0);
            sample_dir_lane0(zero_dir, true, &rng);
        }
    }
    workgroupBarrier();

    if (wg_stop != 0u) {
        if (tidx == 0u) {
            out_lengths[seed_idx * 2u]      = 0u;
            out_lengths[seed_idx * 2u + 1u] = 0u;
        }
        return;
    }

    let init_dir = vec3<f32>(wg_dir[0], wg_dir[1], wg_dir[2]);

    // ── backward half ─────────────────────────────────────────────────
    let neg_dir = vec3<f32>(-init_dir.x, -init_dir.y, -init_dir.z);
    let back_len = track_half(tidx, seed_vox, neg_dir, back_base, &rng);

    // ── forward half ──────────────────────────────────────────────────
    let fwd_len = track_half(tidx, seed_vox, init_dir, fwd_base, &rng);

    // ── write lengths ─────────────────────────────────────────────────
    if (tidx == 0u) {
        out_lengths[seed_idx * 2u]      = back_len;
        out_lengths[seed_idx * 2u + 1u] = fwd_len;
    }
}

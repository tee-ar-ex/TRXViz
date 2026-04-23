// dipy_ptt.wgsl — GPU Parallel Transport Tractography (Aydogan & Shi 2021).
//
// One workgroup = one seed.  Workgroup size: (32, 1, 1).
// Lane 0 handles all sequential decisions (frame state, RNG draws,
// rejection sampling). All 32 lanes cooperate on the FOD-at-direction
// inner loop (each evaluates a slice of the trilinear neighbor sum).
//
// Same coordinate / output conventions as `tractography_prob.wgsl`:
//   - Seeds and output points are in ODX voxel space (host converts
//     RAS↔vox).
//   - out_points layout: [seed * 2 * max_pts * 3 + dir * max_pts * 3 + step * 3 + xyz]
//     dir=0 = backward segment, dir=1 = forward segment.
//   - out_lengths layout: [seed * 2 + dir] = points written.
//
// Algorithm (per nibrary `algorithm_ptt_*.cpp`, simplified for v1):
//   1. Initialize a parallel transport frame (PTF) at the seed: random
//      orthonormal axes (F[0..2]) plus a random (k1, k2) curvature.
//      Rejection-sample initial candidates against an estimated
//      posteriorMax.
//   2. Walk: closed-form propagation of (p, F) along the chosen
//      (k1, k2) curve for one step (`prep_propagator` × frame).
//   3. Per step: pick a new (k1, k2) candidate uniformly in the disc
//      of radius `max_curvature`. Evaluate "data support" by integrating
//      FOD amplitude along a probe arc of length `probe_length` sampled
//      at `probe_quality` points (and `probe_count` circumferential
//      offsets at `probe_radius`). Accept by rejection sampling against
//      `posteriorMax` re-estimated each step.
//   4. Bidirectional from seed: forward branch, then flip the frame
//      (negate F[0], F[1], k1) and backward branch.
//
// Differences vs. nibrary / GPUStreamlines (documented in
// `docs/ptt-implementation-notes.md`):
//   - **Sampling**: pure rejection sampling (DIPY's choice) — no CDF
//     mesh, no disc-vertex precomputation. Simpler shader, ~no host-side
//     setup. Easy to swap to CDF sampling later if needed.
//   - **FOD-at-direction**: trxviz stores SH coefficients + B-matrix
//     rows for sphere vertices (no analytic SH basis evaluation in the
//     shader). For PTT we need the FOD amplitude at an *arbitrary*
//     direction (the frame tangent at each probe point). We approximate
//     by snapping to the nearest sphere vertex and using its precomputed
//     B-matrix row. Quantization error is bounded by the sphere
//     density (detail-2 icosphere ≈ 162 verts, max ~6° gap).
//   - **Symmetric FOD only**: trxviz SH data is even-order. nibrary's
//     asymmetric branch is dead weight here.
//   - **Workgroup size (32, 1, 1)**: matches trxviz's prob shader.
//     GPUStreamlines uses (32, 2, 1) — 2 streamlines per workgroup —
//     for slightly higher SIMT utilization but more complex indexing.
//   - **RNG**: LCG (matches trxviz cpu_yeh / cpu_dipy / prob shader).
//     GPUStreamlines uses Philox; we don't need parallel-friendly
//     properties since all RNG draws happen on lane 0.
//   - **Rejection of per-step masks** (limiting/roa/term): same as
//     existing GPU prob path. Post-hoc filters (roi/end/no_end/
//     hausdorff) are applied CPU-side after readback.

// ── constants ────────────────────────────────────────────────────────
const THR_X: u32 = 32u;
const INVALID_VOXEL: u32 = 0xFFFFFFFFu;
const EPS: f32 = 1e-6;
const K_SMALL: f32 = 1e-4;

// Compensation factor when extrapolating posteriorMax from sampled
// candidates — matches nibrary `DEFAULT_PTT_MAXPOSTESTCOMPENS = 2`
// (algorithm_ptt_params.h:29). Keeps rejection sampling efficient even
// when the true max is underestimated by the small candidate batch.
const POSTERIOR_MAX_COMPENS: f32 = 2.0;

// Number of candidate samples used to estimate posteriorMax per step.
// nibrary's `propMaxEstTrials` (default 20). Each estimate evaluates a
// full probe — expensive but rare (every step in v1; nibrary throttles
// to every `maxEstInterval` steps via `useLegacySampling=false`).
const PROP_MAX_EST_TRIALS: u32 = 20u;
// Initial estimate at seed uses more samples since the frame is
// brand-new and posterior shape is unknown. nibrary default 100.
const INIT_MAX_EST_TRIALS: u32 = 100u;

// ── parameter struct (16-byte padded for var<uniform>) ───────────────
//
// Field layout matches the host-side params packing in
// `gpu/dipy.rs::run_gpu_dipy_ptt`. Same shape as TractographyParams
// for the prob shader, plus PTT-specific knobs at the end.
struct PttParams {
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
    rng_seed:                 u32,

    // PTT-specific (DipyDirectionGetter::Ptt fields).
    probe_length_vox:         f32,   // probe arc length (voxel units)
    probe_quality:            u32,   // arc-length samples per probe (>= 2)
    probe_radius_vox:         f32,   // circumferential probe radius (voxel units; 0 = degenerate)
    probe_count:              u32,   // circumferential samples (>= 1)
    max_curvature_per_vox:    f32,   // max k = sqrt(k1²+k2²); voxel units (1 / min_radius)
    data_support_exponent:    f32,
    min_data_support:         f32,
    rejection_sampling_max_try: u32,

    // 19 useful slots + 5 pads = 24 u32 = 96 bytes (16-byte aligned).
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32,
}

// ── bindings ─────────────────────────────────────────────────────────
//
// PTT uses a precomputed FOD-amplitude buffer (`fod_amp[v, d] =
// max(0, sh[v] · b_matrix[d])`, baked on the host) instead of the
// raw SH coefficients + B-matrix that the prob shader uses. This
// turns the per-FOD-lookup cost from 8 corners × ncoeffs ≈ 360 mults
// into 8 corner reads of one channel — ~4.5× speedup per call.
//
// Layout: 5 slots vs prob's 6 (sh + b → fod_amp).
@group(0) @binding(0) var<uniform>      params:       PttParams;
@group(0) @binding(1) var<storage, read> seeds_vox:   array<f32>;  // [batch * 3]
@group(0) @binding(2) var<storage, read> fod_amp:     array<f32>;  // [NB_VOXELS * n_dirs]
@group(0) @binding(3) var<storage, read> lut:         array<u32>;  // [dimx*dimy*dimz]
@group(0) @binding(4) var<storage, read> sphere_verts: array<f32>; // [n_dirs * 3]

@group(1) @binding(0) var<storage, read_write> out_points:  array<f32>;
@group(1) @binding(1) var<storage, read_write> out_lengths: array<u32>;

// ── workgroup shared memory ───────────────────────────────────────────
//
// Two roles for workgroup memory:
//   1. PTF state (frame + position + k1/k2 + posteriorMax + control flags).
//      Lane 0 owns these between candidate-evaluation rounds.
//   2. Per-lane candidate exchange buffers: each of the 32 lanes
//      evaluates its own candidate per round and writes the result here
//      so lane 0 can reduce (find max for posteriorMax estimation, find
//      first acceptable for rejection sampling).
//
// This is the key change vs. v0.1: candidate evaluation is now 32-way
// parallel instead of lane-0-sequential. Each lane derives its own RNG
// stream, picks its own (k1, k2) and (during init) its own random frame,
// runs its own probe likelihood eval (calc_data_support_local), and
// posts the result. Lane 0 then scans the 32-element exchange to make
// the accept/reject decision.
//
// Total workgroup memory: ~600 bytes. Well under any wgpu limit
// (the 16 KB minimum guarantee).

// All workgroup arrays are sized for BLOCK_Y=2 streamlines per
// workgroup. Each streamline is identified by `tidy ∈ {0, 1}` and
// indexes its own slice. The helpers below compute the base offset
// for clarity at call sites.
const BLOCK_Y: u32 = 2u;

// PTF state (lane-0 owned, per streamline).
var<workgroup> wg_frame:    array<f32, 18>;  // BLOCK_Y * 9 (F0|F1|F2)
var<workgroup> wg_p:        array<f32, 6>;   // BLOCK_Y * 3
var<workgroup> wg_k1:       array<f32, 2>;
var<workgroup> wg_k2:       array<f32, 2>;
var<workgroup> wg_post_max: array<f32, 2>;
var<workgroup> wg_stop:     array<u32, 2>;   // 1 = halt streamline
// Initial-frame snapshot, used to flip the frame for the backward half.
var<workgroup> wg_init_frame:    array<f32, 18>;
var<workgroup> wg_init_k1:       array<f32, 2>;
var<workgroup> wg_init_k2:       array<f32, 2>;
var<workgroup> wg_init_post_max: array<f32, 2>;

// Per-lane candidate exchange (32 entries × 2 streamlines).
var<workgroup> wg_cand_lh:    array<f32, 64>;
var<workgroup> wg_cand_k1:    array<f32, 64>;
var<workgroup> wg_cand_k2:    array<f32, 64>;
// Per-lane random frame (init phase only). 32 lanes × 9 floats × 2 streamlines.
var<workgroup> wg_cand_frame: array<f32, 576>;
// Initial-direction hint (set by lane 0 before init_frame_batched).
var<workgroup> wg_init_dir: array<f32, 6>;

// CDF sampling on the disc (per streamline).
var<workgroup> wg_disc_vert_lh: array<f32, 48>;  // BLOCK_Y * DISC_VERT_CNT
var<workgroup> wg_face_cdf:     array<f32, 62>;  // BLOCK_Y * DISC_FACE_CNT

// Index helpers — keep base-offset arithmetic out of call sites.
fn frame_base(tidy: u32) -> u32 { return tidy * 9u; }
fn p_base(tidy: u32) -> u32 { return tidy * 3u; }
fn cand_base(tidy: u32) -> u32 { return tidy * 32u; }
fn cand_frame_base(tidy: u32) -> u32 { return tidy * 288u; }
fn init_dir_base(tidy: u32) -> u32 { return tidy * 3u; }
fn disc_vert_base(tidy: u32) -> u32 { return tidy * 24u; }
fn face_cdf_base(tidy: u32) -> u32 { return tidy * 31u; }

// ── disc mesh constants (lifted from cuslines/wgsl_shaders/disc.wgsl) ─
//
// 24 unit-disc vertices + 31 triangle faces, used to build a CDF over
// (k1, k2) curvature candidates. Direct translation from nibrary's
// disc.h. Vertices are scaled by `max_curvature_per_vox` at sample time.
//
// SAMPLING_QUALITY=2 — matches DIPY's default; nibrary supports more
// resolutions but they're rarely tuned.
const DISC_VERT_CNT: u32 = 24u;
const DISC_FACE_CNT: u32 = 31u;

const DISC_VERT: array<f32, 48> = array<f32, 48>(
    -0.99680788, -0.07983759,
    -0.94276539,  0.33345677,
    -0.87928469, -0.47629658,
    -0.72856617,  0.68497542,
    -0.60006556, -0.79995082,
    -0.54129995, -0.02761342,
    -0.39271207,  0.37117272,
    -0.39217391,  0.91989110,
    -0.36362884, -0.40757367,
    -0.22391316, -0.97460910,
    -0.00130022,  0.53966106,
     0.00000000,  0.00000000,
     0.00973999,  0.99995257,
     0.01606516, -0.54289908,
     0.21342395, -0.97695968,
     0.38192071, -0.38666136,
     0.38897094,  0.37442837,
     0.40696681,  0.91344295,
     0.54387161, -0.01477123,
     0.59119367, -0.80652963,
     0.73955688,  0.67309406,
     0.87601150, -0.48229022,
     0.94617928,  0.32364298,
     0.99585368, -0.09096944,
);

const DISC_FACE: array<u32, 93> = array<u32, 93>(
     9u,  8u,  4u,
    11u, 16u, 10u,
     5u,  8u, 11u,
     5u,  1u,  0u,
    18u, 16u, 11u,
    11u, 15u, 18u,
    13u,  8u,  9u,
    11u,  8u, 13u,
    13u, 15u, 11u,
    22u, 18u, 23u,
    22u, 20u, 16u,
    16u, 18u, 22u,
    16u, 20u, 17u,
    12u, 10u, 17u,
    17u, 10u, 16u,
    15u, 19u, 21u,
    23u, 18u, 21u,
    21u, 18u, 15u,
     2u,  4u,  8u,
     2u,  5u,  0u,
     8u,  5u,  2u,
     7u, 10u, 12u,
     6u,  7u,  3u,
    10u,  7u,  6u,
     3u,  1u,  6u,
     1u,  5u,  6u,
    11u, 10u,  6u,
     6u,  5u, 11u,
    14u, 19u, 15u,
    15u, 13u, 14u,
    14u, 13u,  9u,
);

// ── LCG RNG (matches the prob shader / cpu_yeh / cpu_dipy) ───────────
fn lcg_step(state: u32) -> u32 {
    return state * 1664525u + 1013904223u;
}
fn lcg_f32(state: ptr<function, u32>) -> f32 {
    *state = lcg_step(*state);
    return f32(*state >> 8u) / f32(1u << 24u);
}
// Symmetric uniform in [-1, 1).
fn lcg_sym(state: ptr<function, u32>) -> f32 {
    return lcg_f32(state) * 2.0 - 1.0;
}

// ── helpers ───────────────────────────────────────────────────────────
fn lut_index(xi: i32, yi: i32, zi: i32) -> u32 {
    return u32(xi) * params.dimy * params.dimz
         + u32(yi) * params.dimz
         + u32(zi);
}

fn load_sphere_vert(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sphere_verts[base], sphere_verts[base + 1u], sphere_verts[base + 2u]);
}

// Find the sphere vertex closest to a unit direction. Returns the
// vertex index. The "sphere is symmetric" so we use abs(dot) — the
// vertex on the opposite hemisphere is equivalent.
//
// Lane-0 only. Sequential O(n_dirs) scan. n_dirs ≈ 162 for detail-2.
fn nearest_vertex(dir: vec3<f32>) -> u32 {
    var best: u32 = 0u;
    var best_abs: f32 = -1.0;
    for (var t = 0u; t < params.n_dirs; t++) {
        let sv = load_sphere_vert(t);
        let d = abs(sv.x * dir.x + sv.y * dir.y + sv.z * dir.z);
        if (d > best_abs) {
            best_abs = d;
            best = t;
        }
    }
    return best;
}

// Evaluate FOD amplitude at a point + direction in voxel space.
//
// **Pure function** — no workgroup state, no barriers. Each lane calls
// it independently with its own (pt, dir). Critical for 32-way
// candidate parallelism: the candidate-evaluation loop has each lane
// running its own probe, each probe doing its own ~4 FOD evals.
//
// Approach: find the nearest sphere vertex to `dir`, then trilinearly
// interpolate the precomputed FOD amplitude at that vertex across the
// 8 voxel neighbors of `pt`. Returns 0.0 if outside mask.
//
// Rationale: trxviz stores SH coefficients + a B-matrix with rows for
// sphere vertices, not an analytic SH basis evaluator in the shader.
// Snapping to the nearest sphere vertex incurs a small angular
// quantization (≈6° at detail-2 ≈ 162 verts) but avoids reimplementing
// the SH basis on the GPU. Same approach GPUStreamlines uses
// (precomputed `dataf` channels per sphere vertex).
//
// Cost per call: nearest_vertex (O(n_dirs) sphere scan ≈ 162 dot
// products = ~810 ops) + 8 trilinear corners × 1 lookup = ~820 ops
// total. Down from ~3690 ops in the SH-dot version. The sphere scan
// is now the dominant cost; future optimizations could precompute a
// neighbor-grid acceleration structure on the sphere.
fn fod_at_dir_local(pt: vec3<f32>, dir: vec3<f32>) -> f32 {
    let vidx = nearest_vertex(dir);
    let n_dirs = params.n_dirs;

    let ix0 = i32(floor(pt.x));
    let iy0 = i32(floor(pt.y));
    let iz0 = i32(floor(pt.z));
    let wx1 = pt.x - f32(ix0); let wx0 = 1.0 - wx1;
    let wy1 = pt.y - f32(iy0); let wy0 = 1.0 - wy1;
    let wz1 = pt.z - f32(iz0); let wz0 = 1.0 - wz1;

    let dimx = i32(params.dimx);
    let dimy = i32(params.dimy);
    let dimz = i32(params.dimz);

    var amp: f32 = 0.0;
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
        // Single lookup — the heavy SH-dot was baked into fod_amp by
        // the host. amp[compact * n_dirs + vidx] is already
        // max(0, sh · b_row).
        amp += w * fod_amp[compact * n_dirs + vidx];
    } } }

    // Outside-mask threshold (matches eval_odf in prob shader).
    return select(0.0, amp, total_w >= 0.01);
}

// ── PTT propagator ───────────────────────────────────────────────────
//
// Direct port of nibrary `PTF::prepPropagator` (ptf.h:95-130). Computes
// the 3×3 transformation that propagates the frame (p, F) by arc-length
// `t` along the curve of constant curvature (k1, k2).
//
// Returns the 9 entries of the matrix flattened row-major:
//   [PP[0]..PP[8]] used as in `walk_frame`:
//     dp = PP[0]·F0 + PP[1]·F1 + PP[2]·F2
//     T  = PP[3]·F0 + PP[4]·F1 + PP[5]·F2
//     N2 = PP[6]·F0 + PP[7]·F1 + PP[8]·F2
fn prep_propagator(k1c: f32, k2c: f32, t: f32) -> array<f32, 9> {
    var pp: array<f32, 9>;

    if (abs(k1c) < K_SMALL && abs(k2c) < K_SMALL) {
        // Straight line segment — degenerate.
        pp[0] = t;   pp[1] = 0.0; pp[2] = 0.0;
        pp[3] = 1.0; pp[4] = 0.0; pp[5] = 0.0;
        pp[6] = 0.0; pp[7] = 0.0; pp[8] = 1.0;
        return pp;
    }

    // Clamp to avoid div-by-zero in the kk = 1/k² term.
    var k1 = k1c;
    var k2 = k2c;
    if (abs(k1) < K_SMALL) { k1 = sign(k1c) * K_SMALL; if (k1 == 0.0) { k1 = K_SMALL; } }
    if (abs(k2) < K_SMALL) { k2 = sign(k2c) * K_SMALL; if (k2 == 0.0) { k2 = K_SMALL; } }

    let k = sqrt(k1 * k1 + k2 * k2);
    let sinkt = sin(k * t);
    let coskt = cos(k * t);
    let kk = 1.0 / (k * k);

    pp[0] = sinkt / k;
    pp[1] = k1 * (1.0 - coskt) * kk;
    pp[2] = k2 * (1.0 - coskt) * kk;
    pp[3] = coskt;
    pp[4] = k1 * sinkt / k;
    pp[5] = k2 * sinkt / k;
    pp[6] = -pp[5];
    pp[7] = k1 * k2 * (coskt - 1.0) * kk;
    pp[8] = (k1 * k1 + k2 * k2 * coskt) * kk;
    return pp;
}

// Apply propagator `pp` to (p, F) — port of `PTF::walk` (ptf.h:132-157).
// Updates `p` in place and returns the new frame F' = (T, N1, N2).
//
// The new tangent T = PP[3]·F0 + PP[4]·F1 + PP[5]·F2 (normalized).
// The new N2     = PP[6]·F0 + PP[7]·F1 + PP[8]·F2.
// N1 is reconstructed orthogonal to T and N2 via cross products.
fn walk_frame(
    p: ptr<function, vec3<f32>>,
    f0: ptr<function, vec3<f32>>,
    f1: ptr<function, vec3<f32>>,
    f2: ptr<function, vec3<f32>>,
    pp: ptr<function, array<f32, 9>>,
) {
    let p0 = *p;
    let a0 = *f0;
    let a1 = *f1;
    let a2 = *f2;
    let m = *pp;

    let dp = m[0] * a0 + m[1] * a1 + m[2] * a2;
    var t  = m[3] * a0 + m[4] * a1 + m[5] * a2;
    var n2 = m[6] * a0 + m[7] * a1 + m[8] * a2;

    let tn = length(t);
    if (tn > EPS) { t = t / tn; }

    // n1 = n2 × t (right-handed); then renormalize n1; then n2 = t × n1.
    var n1 = vec3<f32>(
        n2.y * t.z - n2.z * t.y,
        n2.z * t.x - n2.x * t.z,
        n2.x * t.y - n2.y * t.x,
    );
    let nn1 = length(n1);
    if (nn1 > EPS) { n1 = n1 / nn1; }

    n2 = vec3<f32>(
        t.y * n1.z - t.z * n1.y,
        t.z * n1.x - t.x * n1.z,
        t.x * n1.y - t.y * n1.x,
    );

    *p = p0 + dp;
    *f0 = t;
    *f1 = n1;
    *f2 = n2;
}

// ── PTT data support (probe likelihood) ──────────────────────────────
//
// Port of nibrary `PTF::calcDataSupport` symmetric branch (ptf.cpp:131-213).
// Builds a probe arc of `probe_quality` points along the candidate
// (k1, k2) curve, optionally with `probe_count` circumferential samples
// at `probe_radius` around each, and sums FOD amplitudes.
//
// Performance note: the probe is sampled lane-0-sequentially; each
// FOD eval still uses all 32 lanes for the trilinear sum. With
// probe_quality=4, probe_count=1, this is ~4 FOD evals per candidate.
// With probe_count=4 (DIPY-recommended for asymmetric noise rejection),
// it's ~16. Keep these modest.
//
// Returns the integrated likelihood. Caller raises to
// `data_support_exponent` before comparing.
// **Pure function** — no workgroup state, no barriers. Each lane calls
// it independently with its own (k1, k2) candidate. This is what makes
// 32-way candidate parallelism possible.
fn calc_data_support_local(
    p_in: vec3<f32>,
    f0_in: vec3<f32>,
    f1_in: vec3<f32>,
    f2_in: vec3<f32>,
    k1c: f32, k2c: f32,
) -> f32 {
    if (params.probe_quality < 2u) { return 0.0; }

    let probe_step = params.probe_length_vox / f32(params.probe_quality - 1u);
    let angular_sep = 2.0 * 3.14159265358979 / max(1.0, f32(params.probe_count));

    var p = p_in;
    var f0 = f0_in;
    var f1 = f1_in;
    var f2 = f2_in;

    // First sample: at the starting frame's tangent, no offset.
    var likelihood: f32 = 0.0;
    if (params.probe_count <= 1u) {
        likelihood += fod_at_dir_local(p, f0);
    } else {
        var acc: f32 = 0.0;
        for (var c = 0u; c < params.probe_count; c++) {
            let theta = f32(c) * angular_sep;
            let off = f1 * (params.probe_radius_vox * cos(theta))
                    + f2 * (params.probe_radius_vox * sin(theta));
            acc += fod_at_dir_local(p + off, f0);
        }
        likelihood += acc;
    }

    // Walk along arc, sample at each subsequent quality point.
    var pp = prep_propagator(k1c, k2c, probe_step);
    for (var q = 1u; q < params.probe_quality; q++) {
        walk_frame(&p, &f0, &f1, &f2, &pp);
        if (params.probe_count <= 1u) {
            likelihood += fod_at_dir_local(p, f0);
        } else {
            var acc: f32 = 0.0;
            for (var c = 0u; c < params.probe_count; c++) {
                let theta = f32(c) * angular_sep;
                let off = f1 * (params.probe_radius_vox * cos(theta))
                        + f2 * (params.probe_radius_vox * sin(theta));
                acc += fod_at_dir_local(p + off, f0);
            }
            likelihood += acc;
        }
    }

    let total_samples = f32(params.probe_quality * max(1u, params.probe_count));
    return likelihood / total_samples;
}

// ── random orthonormal frame around an initial tangent ────────────────
//
// Port of nibrary `RandomDoer::getARandomMovingFrame`. Given an initial
// tangent direction, build two random orthonormal normals.
//
// Approach: pick an arbitrary axis not parallel to dir, cross to get N1,
// cross again to get N2. Random rotation of N1/N2 around dir to
// distribute uniformly.
fn random_frame_around(
    dir: vec3<f32>,
    rng: ptr<function, u32>,
) -> array<vec3<f32>, 3> {
    var f0 = dir;
    let dn = length(f0);
    if (dn > EPS) { f0 = f0 / dn; }

    // Pick a reference axis least aligned with f0.
    var axis: vec3<f32>;
    if (abs(f0.x) < abs(f0.y) && abs(f0.x) < abs(f0.z)) {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else if (abs(f0.y) < abs(f0.z)) {
        axis = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        axis = vec3<f32>(0.0, 0.0, 1.0);
    }

    // First normal.
    var n1 = vec3<f32>(
        f0.y * axis.z - f0.z * axis.y,
        f0.z * axis.x - f0.x * axis.z,
        f0.x * axis.y - f0.y * axis.x,
    );
    let nn1 = length(n1);
    if (nn1 > EPS) { n1 = n1 / nn1; }

    // Random rotation around f0 to distribute uniformly.
    let theta = lcg_f32(rng) * 2.0 * 3.14159265358979;
    let c = cos(theta);
    let s = sin(theta);
    // Rodrigues rotation of n1 around f0 by theta.
    let n1r = n1 * c
        + vec3<f32>(
            f0.y * n1.z - f0.z * n1.y,
            f0.z * n1.x - f0.x * n1.z,
            f0.x * n1.y - f0.y * n1.x,
        ) * s;
    n1 = normalize(n1r);

    let n2 = vec3<f32>(
        f0.y * n1.z - f0.z * n1.y,
        f0.z * n1.x - f0.x * n1.z,
        f0.x * n1.y - f0.y * n1.x,
    );

    var out: array<vec3<f32>, 3>;
    out[0] = f0;
    out[1] = n1;
    out[2] = n2;
    return out;
}

// Pick a uniform random point in the disc of radius `max_curvature`.
// Port of nibrary `RandomDoer::getARandomPointWithinDisk`.
//
// Inverse-CDF (sqrt for area uniformity) + uniform angle.
fn random_in_disc(
    rng: ptr<function, u32>,
) -> vec2<f32> {
    let r = sqrt(lcg_f32(rng)) * params.max_curvature_per_vox;
    let theta = lcg_f32(rng) * 2.0 * 3.14159265358979;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// ── per-attempt initialization (32-way parallel) ─────────────────────
//
// Find an initial frame + (k1, k2) for a seed point. Each round, all 32
// lanes generate their own random frame + (k1, k2) and score it; lane 0
// then scans for the best. We use ceil(INIT_MAX_EST_TRIALS/32) rounds.
//
// The winning lane's frame is reconstructed from `wg_cand_frame[winner * 9 ..]`.
//
// On success: writes wg_frame, wg_k1, wg_k2, wg_post_max, wg_p; sets wg_stop=0.
// On failure (best support below min_data_support): wg_stop=1.
fn init_frame_batched(
    seed: vec3<f32>,
    init_dir_hint: vec3<f32>,
    rng_lane: ptr<function, u32>,
    tidx: u32,
    tidy: u32,
) {
    var post_max: f32 = 0.0;
    var best_lane: u32 = 0u;
    var best_lh: f32 = -1.0;

    let cb = cand_base(tidy);
    let cfb = cand_frame_base(tidy);
    let fb_out = frame_base(tidy);
    let pb = p_base(tidy);

    let n_rounds = (INIT_MAX_EST_TRIALS + THR_X - 1u) / THR_X;
    for (var round = 0u; round < n_rounds; round++) {
        let fr = random_frame_around(init_dir_hint, rng_lane);
        let kk = random_in_disc(rng_lane);
        let lh = calc_data_support_local(seed, fr[0], fr[1], fr[2], kk.x, kk.y);

        wg_cand_lh[cb + tidx] = lh;
        wg_cand_k1[cb + tidx] = kk.x;
        wg_cand_k2[cb + tidx] = kk.y;
        let fb = cfb + tidx * 9u;
        wg_cand_frame[fb + 0u] = fr[0].x; wg_cand_frame[fb + 1u] = fr[0].y; wg_cand_frame[fb + 2u] = fr[0].z;
        wg_cand_frame[fb + 3u] = fr[1].x; wg_cand_frame[fb + 4u] = fr[1].y; wg_cand_frame[fb + 5u] = fr[1].z;
        wg_cand_frame[fb + 6u] = fr[2].x; wg_cand_frame[fb + 7u] = fr[2].y; wg_cand_frame[fb + 8u] = fr[2].z;
        workgroupBarrier();

        if (tidx == 0u) {
            for (var t = 0u; t < THR_X; t++) {
                let v = wg_cand_lh[cb + t];
                if (v > post_max) { post_max = v; }
                if (v > best_lh) {
                    best_lh = v;
                    best_lane = t;
                }
            }
        }
        workgroupBarrier();
    }

    if (tidx == 0u) {
        post_max = pow(post_max * POSTERIOR_MAX_COMPENS, params.data_support_exponent);

        let support = pow(best_lh, params.data_support_exponent);
        if (best_lh <= 0.0 || support < params.min_data_support) {
            wg_stop[tidy] = 1u;
            return;
        }

        let fb_in = cfb + best_lane * 9u;
        for (var i = 0u; i < 9u; i++) {
            wg_frame[fb_out + i] = wg_cand_frame[fb_in + i];
        }
        wg_k1[tidy] = wg_cand_k1[cb + best_lane];
        wg_k2[tidy] = wg_cand_k2[cb + best_lane];
        wg_post_max[tidy] = post_max;
        wg_p[pb + 0u] = seed.x; wg_p[pb + 1u] = seed.y; wg_p[pb + 2u] = seed.z;
        wg_stop[tidy] = 0u;
    }
}

// ── per-step candidate pick (CDF-on-disc sampling) ───────────────────
//
// Port of nibrary's `sampleFromCDF` (algorithm_ptt_propagate.cpp:50-124).
// Replaces the rejection-sampling picker — converges on each step with
// far fewer FOD evaluations because the CDF concentrates samples in
// high-likelihood regions of the (k1, k2) disc.
//
// Algorithm:
//   1. Each of 24 lanes evaluates the data-support likelihood at one
//      disc vertex (k1, k2) = DISC_VERT[i] × max_curvature. Lanes 24-31
//      sit idle (~25% wasted slots; minor cost).
//   2. Lane 0 builds a face CDF: for each of 31 triangles, sum the 3
//      vertex likelihoods (zeroed if any vertex is below
//      min_data_support). Cumulative sum.
//   3. Lane 0 inverse-CDF samples a face, then picks a uniform point
//      inside via barycentric coordinates. Evaluate that point's
//      likelihood. If above the floor, accept; else retry up to a few
//      times (CDF guides us toward density, so 1-2 tries usually suffices).
//
// Per step: 24 + ~2 = ~26 FOD evals. Vs. rejection sampling's
// 20 + ~64 = ~84 FOD evals on average. ~3× fewer evals → ~3× speedup
// at this layer.
fn pick_next_candidate_cdf(
    p: vec3<f32>,
    f0: vec3<f32>, f1: vec3<f32>, f2: vec3<f32>,
    rng_seq: ptr<function, u32>,
    tidx: u32,
    tidy: u32,
) -> u32 {
    let max_k = params.max_curvature_per_vox;
    let dvb = disc_vert_base(tidy);
    let fcb = face_cdf_base(tidy);

    // ── Step 1: each lane scores one disc vertex ──────────────────
    if (tidx < DISC_VERT_CNT) {
        let vi = tidx * 2u;
        let k1 = DISC_VERT[vi]      * max_k;
        let k2 = DISC_VERT[vi + 1u] * max_k;
        wg_disc_vert_lh[dvb + tidx] = calc_data_support_local(p, f0, f1, f2, k1, k2);
    }
    workgroupBarrier();

    // ── Step 2: lane 0 builds the face CDF ────────────────────────
    if (tidx == 0u) {
        var cum: f32 = 0.0;
        for (var fi = 0u; fi < DISC_FACE_CNT; fi++) {
            let f_base = fi * 3u;
            let v0 = DISC_FACE[f_base];
            let v1 = DISC_FACE[f_base + 1u];
            let v2 = DISC_FACE[f_base + 2u];
            let l0 = wg_disc_vert_lh[dvb + v0];
            let l1 = wg_disc_vert_lh[dvb + v1];
            let l2 = wg_disc_vert_lh[dvb + v2];

            let s0 = pow(l0, params.data_support_exponent);
            let s1 = pow(l1, params.data_support_exponent);
            let s2 = pow(l2, params.data_support_exponent);

            if (s0 >= params.min_data_support
                && s1 >= params.min_data_support
                && s2 >= params.min_data_support) {
                cum += s0 + s1 + s2;
            }
            wg_face_cdf[fcb + fi] = cum;
        }
        wg_post_max[tidy] = cum;

        if (cum <= 0.0) {
            wg_stop[tidy] = 1u;
        } else {
            wg_stop[tidy] = 0u;
        }
    }
    workgroupBarrier();

    if (wg_stop[tidy] != 0u) { return 0u; }

    if (tidx == 0u) {
        var accepted: u32 = 0u;
        let total_cdf = wg_post_max[tidy];

        for (var tries = 0u; tries < params.rejection_sampling_max_try; tries++) {
            if (accepted != 0u) { break; }

            let r = lcg_f32(rng_seq) * total_cdf;
            var face_idx: u32 = DISC_FACE_CNT - 1u;
            for (var fi = 0u; fi < DISC_FACE_CNT; fi++) {
                if (wg_face_cdf[fcb + fi] >= r) {
                    face_idx = fi;
                    break;
                }
            }

            let f_base = face_idx * 3u;
            let v0i = DISC_FACE[f_base];
            let v1i = DISC_FACE[f_base + 1u];
            let v2i = DISC_FACE[f_base + 2u];
            let v0_k1 = DISC_VERT[v0i * 2u]      * max_k;
            let v0_k2 = DISC_VERT[v0i * 2u + 1u] * max_k;
            let v1_k1 = DISC_VERT[v1i * 2u]      * max_k;
            let v1_k2 = DISC_VERT[v1i * 2u + 1u] * max_k;
            let v2_k1 = DISC_VERT[v2i * 2u]      * max_k;
            let v2_k2 = DISC_VERT[v2i * 2u + 1u] * max_k;

            var r1 = lcg_f32(rng_seq);
            var r2 = lcg_f32(rng_seq);
            if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }

            let k1t = v0_k1 + r1 * (v1_k1 - v0_k1) + r2 * (v2_k1 - v0_k1);
            let k2t = v0_k2 + r1 * (v1_k2 - v0_k2) + r2 * (v2_k2 - v0_k2);

            let lh = calc_data_support_local(p, f0, f1, f2, k1t, k2t);
            let support = pow(lh, params.data_support_exponent);
            if (support >= params.min_data_support) {
                wg_k1[tidy] = k1t;
                wg_k2[tidy] = k2t;
                accepted = 1u;
            }
        }
        wg_stop[tidy] = select(1u, 0u, accepted != 0u);
    }
    workgroupBarrier();

    return select(0u, 1u, wg_stop[tidy] == 0u);
}

// ── one branch (forward or backward) ──────────────────────────────────
//
// Caller has already set up wg_frame, wg_p, wg_k1, wg_k2, wg_post_max.
// Walks for up to max_pts steps, writing points to out_points starting
// at out_base (in f32 units). Returns the number of points written.
//
// Note: each step does PROP_MAX_EST_TRIALS + up to
// rejection_sampling_max_try probe evaluations. Each probe eval is
// probe_quality × max(1, probe_count) FOD evaluations. Each FOD eval
// is one trilinear interp over ncoeffs SH coefficients. With defaults
// (PROP_MAX=20, rej_max=100, q=4, count=1, ncoeffs=28), worst-case
// per step ≈ 120 × 4 × 28 = ~13k FLOPs in the inner SH dot. Per
// streamline ≈ 100 steps × 13k ≈ 1.3M FLOPs. Per workgroup. Plenty of
// per-streamline work to amortize GPU dispatch.
fn track_branch(
    out_base: u32,
    rng_lane: ptr<function, u32>,
    rng_seq: ptr<function, u32>,
    tidx: u32,
    tidy: u32,
) -> u32 {
    let max_pts = params.max_points;
    let pb = p_base(tidy);
    let fb = frame_base(tidy);
    var npts: u32 = 0u;

    loop {
        if (npts >= max_pts) { break; }

        let p = vec3<f32>(wg_p[pb + 0u], wg_p[pb + 1u], wg_p[pb + 2u]);
        let f0 = vec3<f32>(wg_frame[fb + 0u], wg_frame[fb + 1u], wg_frame[fb + 2u]);
        let f1 = vec3<f32>(wg_frame[fb + 3u], wg_frame[fb + 4u], wg_frame[fb + 5u]);
        let f2 = vec3<f32>(wg_frame[fb + 6u], wg_frame[fb + 7u], wg_frame[fb + 8u]);

        if (tidx == 0u) {
            let off = out_base + npts * 3u;
            out_points[off] = p.x;
            out_points[off + 1u] = p.y;
            out_points[off + 2u] = p.z;
        }
        workgroupBarrier();
        npts += 1u;

        let ok = pick_next_candidate_cdf(p, f0, f1, f2, rng_seq, tidx, tidy);
        workgroupBarrier();
        if (ok == 0u) { break; }

        if (tidx == 0u) {
            let k1c = wg_k1[tidy];
            let k2c = wg_k2[tidy];
            var pp = prep_propagator(k1c, k2c, params.step_size_vox);
            var p_mut = p;
            var f0_mut = f0;
            var f1_mut = f1;
            var f2_mut = f2;
            walk_frame(&p_mut, &f0_mut, &f1_mut, &f2_mut, &pp);
            wg_p[pb + 0u] = p_mut.x; wg_p[pb + 1u] = p_mut.y; wg_p[pb + 2u] = p_mut.z;
            wg_frame[fb + 0u] = f0_mut.x; wg_frame[fb + 1u] = f0_mut.y; wg_frame[fb + 2u] = f0_mut.z;
            wg_frame[fb + 3u] = f1_mut.x; wg_frame[fb + 4u] = f1_mut.y; wg_frame[fb + 5u] = f1_mut.z;
            wg_frame[fb + 6u] = f2_mut.x; wg_frame[fb + 7u] = f2_mut.y; wg_frame[fb + 8u] = f2_mut.z;
        }
        workgroupBarrier();
    }

    return npts;
}

// ── kernel entry point ────────────────────────────────────────────────
//
// Two streamlines per workgroup (BLOCK_Y=2). Each streamline owns one
// SIMT group of 32 lanes (tidx ∈ [0, 32)); the streamline index is
// `tidy ∈ [0, 2)`. Doubling workgroup occupancy this way tends to
// improve SM-level utilization on most GPUs.
@compute @workgroup_size(32, 2, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
) {
    let tidx = local_id.x;
    let tidy = local_id.y;
    // Each workgroup handles BLOCK_Y seeds; global_id.x is already
    // the per-lane index within a workgroup's x-dim. Seed index is
    // `workgroup_x * BLOCK_Y + tidy`.
    let workgroup_x = global_id.x / THR_X;
    let seed_idx = workgroup_x * BLOCK_Y + tidy;

    if (seed_idx >= params.batch_size) { return; }

    // Load seed in voxel space.
    let seed_base = seed_idx * 3u;
    let seed_vox = vec3<f32>(
        seeds_vox[seed_base],
        seeds_vox[seed_base + 1u],
        seeds_vox[seed_base + 2u],
    );

    // Per-seed RNG (lane 0, sequential decisions).
    var rng_seq = lcg_step(params.rng_seed ^ (params.batch_offset + seed_idx));
    rng_seq = lcg_step(rng_seq);

    // Per-lane RNG (each of the 32 lanes in tidx direction).
    var rng_lane = lcg_step(rng_seq ^ ((tidx + 1u) * 2654435761u));
    rng_lane = lcg_step(rng_lane);

    let pts_per_dir = params.max_points;
    let back_base = seed_idx * 2u * pts_per_dir * 3u;
    let fwd_base  = seed_idx * 2u * pts_per_dir * 3u + pts_per_dir * 3u;

    if (tidx == 0u) {
        out_lengths[seed_idx * 2u] = 0u;
        out_lengths[seed_idx * 2u + 1u] = 0u;
    }
    workgroupBarrier();

    // ── Pick initial direction hint (strongest FOD sphere vertex) ──
    let cb = cand_base(tidy);
    let idb = init_dir_base(tidy);
    var best_amp_local: f32 = -1.0;
    var best_t_local: u32 = 0u;
    for (var t = tidx; t < params.n_dirs; t += THR_X) {
        let sv = load_sphere_vert(t);
        let amp = fod_at_dir_local(seed_vox, sv);
        if (amp > best_amp_local) {
            best_amp_local = amp;
            best_t_local = t;
        }
    }
    wg_cand_lh[cb + tidx] = best_amp_local;
    wg_cand_k1[cb + tidx] = bitcast<f32>(best_t_local);
    workgroupBarrier();

    if (tidx == 0u) {
        var best_amp: f32 = -1.0;
        var best_t: u32 = 0u;
        for (var t = 0u; t < THR_X; t++) {
            if (wg_cand_lh[cb + t] > best_amp) {
                best_amp = wg_cand_lh[cb + t];
                best_t = bitcast<u32>(wg_cand_k1[cb + t]);
            }
        }
        if (best_amp <= 0.0) {
            wg_stop[tidy] = 1u;
        } else {
            let dir = load_sphere_vert(best_t);
            wg_init_dir[idb + 0u] = dir.x;
            wg_init_dir[idb + 1u] = dir.y;
            wg_init_dir[idb + 2u] = dir.z;
            wg_stop[tidy] = 0u;
        }
    }
    workgroupBarrier();

    if (wg_stop[tidy] != 0u) { return; }

    let init_dir = vec3<f32>(wg_init_dir[idb + 0u], wg_init_dir[idb + 1u], wg_init_dir[idb + 2u]);

    // Initialize PTT frame + (k1, k2) at the seed via batched candidate
    // evaluation (each of the 32 lanes scores its own random frame).
    init_frame_batched(seed_vox, init_dir, &rng_lane, tidx, tidy);
    workgroupBarrier();
    if (wg_stop[tidy] != 0u) { return; }

    // Save the initial frame so the backward branch can restart from
    // the seed with a flipped tangent.
    let fb = frame_base(tidy);
    if (tidx == 0u) {
        for (var i = 0u; i < 9u; i++) { wg_init_frame[fb + i] = wg_frame[fb + i]; }
        wg_init_k1[tidy] = wg_k1[tidy];
        wg_init_k2[tidy] = wg_k2[tidy];
        wg_init_post_max[tidy] = wg_post_max[tidy];
    }
    workgroupBarrier();

    // ── forward branch ──────────────────────────────────────────────
    let fwd_len = track_branch(fwd_base, &rng_lane, &rng_seq, tidx, tidy);

    // ── flip + backward branch ──────────────────────────────────────
    let pb = p_base(tidy);
    if (tidx == 0u) {
        wg_frame[fb + 0u] = -wg_init_frame[fb + 0u];
        wg_frame[fb + 1u] = -wg_init_frame[fb + 1u];
        wg_frame[fb + 2u] = -wg_init_frame[fb + 2u];
        wg_frame[fb + 3u] = -wg_init_frame[fb + 3u];
        wg_frame[fb + 4u] = -wg_init_frame[fb + 4u];
        wg_frame[fb + 5u] = -wg_init_frame[fb + 5u];
        wg_frame[fb + 6u] =  wg_init_frame[fb + 6u];
        wg_frame[fb + 7u] =  wg_init_frame[fb + 7u];
        wg_frame[fb + 8u] =  wg_init_frame[fb + 8u];
        wg_k1[tidy] = -wg_init_k1[tidy];
        wg_k2[tidy] =  wg_init_k2[tidy];
        wg_post_max[tidy] = wg_init_post_max[tidy];
        wg_p[pb + 0u] = seed_vox.x; wg_p[pb + 1u] = seed_vox.y; wg_p[pb + 2u] = seed_vox.z;
    }
    workgroupBarrier();

    let back_len = track_branch(back_base, &rng_lane, &rng_seq, tidx, tidy);

    if (tidx == 0u) {
        out_lengths[seed_idx * 2u]      = back_len;
        out_lengths[seed_idx * 2u + 1u] = fwd_len;
    }
}

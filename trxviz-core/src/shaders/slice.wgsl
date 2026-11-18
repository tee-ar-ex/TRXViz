struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos_tex: vec3<f32>,
    render_3d_mode: u32,
    window_center: f32,
    window_width: f32,
    colormap: u32,
    opacity: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.tex_coord = in.tex_coord;
    return out;
}

@vertex
fn vs_raymarch(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.tex_coord = in.tex_coord;
    return out;
}

fn colormap_hot(t: f32) -> vec3<f32> {
    let r = clamp(t * 2.5, 0.0, 1.0);
    let g = clamp(t * 2.5 - 1.0, 0.0, 1.0);
    let b = clamp(t * 5.0 - 4.0, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn colormap_cool(t: f32) -> vec3<f32> {
    return vec3<f32>(t, 1.0 - t, 1.0);
}

fn colormap_red_yellow(t: f32) -> vec3<f32> {
    return vec3<f32>(1.0, t, 0.0);
}

fn colormap_blue_lightblue(t: f32) -> vec3<f32> {
    return vec3<f32>(0.0, t, 1.0);
}

fn colormap_inferno(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.001, 0.000, 0.014);
    let c1 = vec3<f32>(0.125, 0.047, 0.290);
    let c2 = vec3<f32>(0.302, 0.073, 0.488);
    let c3 = vec3<f32>(0.511, 0.121, 0.561);
    let c4 = vec3<f32>(0.709, 0.212, 0.486);
    let c5 = vec3<f32>(0.865, 0.316, 0.347);
    let c6 = vec3<f32>(0.962, 0.471, 0.212);
    let c7 = vec3<f32>(0.988, 0.683, 0.139);
    let c8 = vec3<f32>(0.988, 0.998, 0.645);
    let x = clamp(t, 0.0, 1.0) * 8.0;
    let i = clamp(i32(floor(x)), 0, 7);
    let f = clamp(x - f32(i), 0.0, 1.0);
    let a = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i];
    let b = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i + 1];
    return mix(a, b, f);
}

fn colormap_magma(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.001, 0.000, 0.014);
    let c1 = vec3<f32>(0.113, 0.065, 0.276);
    let c2 = vec3<f32>(0.275, 0.063, 0.485);
    let c3 = vec3<f32>(0.445, 0.122, 0.507);
    let c4 = vec3<f32>(0.625, 0.185, 0.498);
    let c5 = vec3<f32>(0.805, 0.251, 0.443);
    let c6 = vec3<f32>(0.944, 0.378, 0.365);
    let c7 = vec3<f32>(0.992, 0.606, 0.419);
    let c8 = vec3<f32>(0.987, 0.991, 0.750);
    let x = clamp(t, 0.0, 1.0) * 8.0;
    let i = clamp(i32(floor(x)), 0, 7);
    let f = clamp(x - f32(i), 0.0, 1.0);
    let a = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i];
    let b = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i + 1];
    return mix(a, b, f);
}

fn colormap_plasma(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.050, 0.030, 0.528);
    let c1 = vec3<f32>(0.255, 0.014, 0.615);
    let c2 = vec3<f32>(0.417, 0.001, 0.658);
    let c3 = vec3<f32>(0.563, 0.052, 0.642);
    let c4 = vec3<f32>(0.693, 0.165, 0.565);
    let c5 = vec3<f32>(0.798, 0.280, 0.470);
    let c6 = vec3<f32>(0.881, 0.393, 0.383);
    let c7 = vec3<f32>(0.949, 0.518, 0.295);
    let c8 = vec3<f32>(0.988, 0.652, 0.211);
    let c9 = vec3<f32>(0.940, 0.975, 0.131);
    let x = clamp(t, 0.0, 1.0) * 9.0;
    let i = clamp(i32(floor(x)), 0, 8);
    let f = clamp(x - f32(i), 0.0, 1.0);
    let a = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i];
    let b = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i + 1];
    return mix(a, b, f);
}

fn colormap_viridis(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.267, 0.005, 0.329);
    let c1 = vec3<f32>(0.283, 0.141, 0.458);
    let c2 = vec3<f32>(0.254, 0.265, 0.530);
    let c3 = vec3<f32>(0.207, 0.372, 0.553);
    let c4 = vec3<f32>(0.164, 0.471, 0.558);
    let c5 = vec3<f32>(0.128, 0.567, 0.551);
    let c6 = vec3<f32>(0.135, 0.659, 0.518);
    let c7 = vec3<f32>(0.267, 0.749, 0.441);
    let c8 = vec3<f32>(0.478, 0.821, 0.318);
    let c9 = vec3<f32>(0.741, 0.873, 0.150);
    let x = clamp(t, 0.0, 1.0) * 9.0;
    let i = clamp(i32(floor(x)), 0, 8);
    let f = clamp(x - f32(i), 0.0, 1.0);
    let a = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i];
    let b = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i + 1];
    return mix(a, b, f);
}

fn apply_colormap(cmap: u32, val: f32) -> vec3<f32> {
    switch cmap {
        case 1u: { return colormap_hot(val); }
        case 2u: { return colormap_cool(val); }
        case 3u: { return colormap_red_yellow(val); }
        case 4u: { return colormap_blue_lightblue(val); }
        case 5u: { return colormap_inferno(val); }
        case 6u: { return colormap_magma(val); }
        case 7u: { return colormap_plasma(val); }
        case 8u: { return colormap_viridis(val); }
        default: { return vec3<f32>(val, val, val); }
    }
}

fn sample_trilinear(uv: vec3<f32>) -> f32 {
    let dims_u = textureDimensions(volume_texture);
    let dims = vec3<f32>(f32(dims_u.x), f32(dims_u.y), f32(dims_u.z));
    let max_idx = vec3<i32>(i32(dims_u.x) - 1, i32(dims_u.y) - 1, i32(dims_u.z) - 1);

    let coord = clamp(uv * dims - vec3<f32>(0.5), vec3<f32>(0.0), max(dims - vec3<f32>(1.0001), vec3<f32>(0.0)));
    let i0 = clamp(vec3<i32>(floor(coord)), vec3<i32>(0), max_idx);
    let i1 = min(i0 + vec3<i32>(1), max_idx);
    let f = fract(coord);

    let c000 = textureLoad(volume_texture, vec3<i32>(i0.x, i0.y, i0.z), 0).r;
    let c100 = textureLoad(volume_texture, vec3<i32>(i1.x, i0.y, i0.z), 0).r;
    let c010 = textureLoad(volume_texture, vec3<i32>(i0.x, i1.y, i0.z), 0).r;
    let c110 = textureLoad(volume_texture, vec3<i32>(i1.x, i1.y, i0.z), 0).r;
    let c001 = textureLoad(volume_texture, vec3<i32>(i0.x, i0.y, i1.z), 0).r;
    let c101 = textureLoad(volume_texture, vec3<i32>(i1.x, i0.y, i1.z), 0).r;
    let c011 = textureLoad(volume_texture, vec3<i32>(i0.x, i1.y, i1.z), 0).r;
    let c111 = textureLoad(volume_texture, vec3<i32>(i1.x, i1.y, i1.z), 0).r;

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);
    return mix(c0, c1, f.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = textureSample(volume_texture, volume_sampler, in.tex_coord).r;
    // Empty voxels stay transparent so an overlay (ROI mask, bundle map) drawn as its
    // own slice pass doesn't paint its zero background over the volumes beneath it.
    if (intensity <= 1e-6) {
        discard;
    }
    // Apply windowing
    let lo = uniforms.window_center - uniforms.window_width * 0.5;
    let hi = uniforms.window_center + uniforms.window_width * 0.5;
    let val = clamp((intensity - lo) / max(hi - lo, 0.001), 0.0, 1.0);
    let color = apply_colormap(uniforms.colormap, val);
    return vec4<f32>(color, uniforms.opacity);
}

@fragment
fn fs_raymarch(in: VertexOutput) -> @location(0) vec4<f32> {
    let o = uniforms.camera_pos_tex;
    let ray_vec = in.tex_coord - o;
    let ray_len = length(ray_vec);
    if (ray_len < 1e-6) {
        discard;
    }
    let d = ray_vec / ray_len;

    // Exact slab intersection against the unit cube [0, 1]^3 in texture space
    var inv_d = vec3<f32>(0.0);
    inv_d.x = 1.0 / select(d.x, select(-1e-6, 1e-6, d.x >= 0.0), abs(d.x) < 1e-6);
    inv_d.y = 1.0 / select(d.y, select(-1e-6, 1e-6, d.y >= 0.0), abs(d.y) < 1e-6);
    inv_d.z = 1.0 / select(d.z, select(-1e-6, 1e-6, d.z >= 0.0), abs(d.z) < 1e-6);

    let t0 = (vec3<f32>(0.0) - o) * inv_d;
    let t1 = (vec3<f32>(1.0) - o) * inv_d;
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);
    let t_enter = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    let t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    if (t_exit <= t_enter + 1e-5) {
        discard;
    }

    let num_steps: u32 = 192u;
    let dt = (t_exit - t_enter) / f32(num_steps);
    let jitter = fract(sin(dot(in.clip_position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 0.75;

    let lo = max(uniforms.window_center - uniforms.window_width * 0.5, 0.0005);
    let hi = uniforms.window_center + uniforms.window_width * 0.5;
    let inv_win = 1.0 / max(hi - lo, 0.001);

    if (uniforms.render_3d_mode == 2u) {
        // Mode 2: 3D Front-to-Back Direct Volume Rendering (DVR) with ray-derivative edge boost
        var acc_color = vec3<f32>(0.0);
        var acc_alpha: f32 = 0.0;
        var prev_val: f32 = 0.0;
        let path_scale = (t_exit - t_enter) * 160.0 / f32(num_steps);

        for (var step: u32 = 0u; step < num_steps; step = step + 1u) {
            let t = t_enter + (f32(step) + jitter) * dt;
            let pos = o + d * t;
            let raw = sample_trilinear(pos);
            if (raw > lo) {
                let norm = clamp((raw - lo) * inv_win, 0.0, 1.0);
                let val = pow(norm, 0.60);
                let edge_boost = 1.0 + 1.8 * abs(val - prev_val);
                let c = clamp(apply_colormap(uniforms.colormap, val) * (0.95 + 0.25 * edge_boost), vec3<f32>(0.0), vec3<f32>(1.0));
                let sample_alpha = clamp(pow(norm, 1.85) * edge_boost * uniforms.opacity * path_scale * 2.2, 0.0, 0.95);
                let weight = (1.0 - acc_alpha) * sample_alpha;
                acc_color = acc_color + weight * c;
                acc_alpha = acc_alpha + weight;
                prev_val = val;
                if (acc_alpha > 0.985) {
                    break;
                }
            } else {
                prev_val = 0.0;
            }
        }

        if (acc_alpha <= 0.001) {
            discard;
        }
        return vec4<f32>(acc_color / max(acc_alpha, 0.001), clamp(acc_alpha, 0.0, 1.0));
    } else {
        // Mode 1: 3D Maximum Intensity Projection (MIP)
        var max_raw: f32 = 0.0;
        for (var step: u32 = 0u; step < num_steps; step = step + 1u) {
            let t = t_enter + (f32(step) + jitter) * dt;
            let pos = o + d * t;
            let raw = sample_trilinear(pos);
            max_raw = max(max_raw, raw);
        }

        if (max_raw <= lo) {
            discard;
        }
        let val = pow(clamp((max_raw - lo) * inv_win, 0.0, 1.0), 0.55);
        let color = apply_colormap(uniforms.colormap, val);
        let alpha = clamp(pow(val, 0.65) * uniforms.opacity, 0.0, 1.0);
        return vec4<f32>(color, alpha);
    }
}

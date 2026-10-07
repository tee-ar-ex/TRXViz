struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    slab_half_width: f32,   // 0 = disabled
    slab_normal: vec3<f32>, // unit normal to slice plane
    color_mode: u32,
    slab_center: vec3<f32>, // world-space point on slice plane
    draw_step: u32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    opacity: f32,
    gloss: f32,
    scale_mul: f32,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    post_params: vec4<f32>,
    opacity_gate: vec4<f32>,
    size_gate: vec4<f32>,
}

struct Amplitudes {
    values: array<f32>,
}

struct GateSamples {
    values: array<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> amplitudes: Amplitudes;
@group(0) @binding(2) var<storage, read> opacity_samples: GateSamples;
@group(0) @binding(3) var<storage, read> size_samples: GateSamples;

struct VertexInput {
    @location(0) direction: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) scale: f32,
    @location(3) amplitude_offset: u32,
    @location(4) min_contacts: u32,
    @location(5) contact_count: u32,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) center: vec3<f32>,
    @location(4) draw_alpha: f32,
    @location(5) ndc_xy: vec2<f32>,
}

fn palette_plasma(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return clamp(
        vec3<f32>(
            0.05873234 + x * (2.176514 + x * (-2.689460 + x * (6.130348 + x * (-11.10743 + x * 5.436053)))),
            0.02333670 + x * (0.2383834 + x * (0.8639469 + x * (-1.930449 + x * (2.137246 + x * -0.6826481)))),
            0.5433248 + x * (1.465357 + x * (-4.203530 + x * (4.693453 + x * (-2.372373 + x * 0.4475437))))
        ),
        vec3<f32>(0.0), vec3<f32>(1.0)
    );
}

fn palette_viridis(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return clamp(
        vec3<f32>(
            0.2777273 + x * (-0.4872 + x * (4.61385 + x * (-14.1862 + x * (14.7864 + x * -4.07578)))),
            0.0054173 + x * (1.40495 + x * (0.33305 + x * (-3.02716 + x * (2.69389 + x * -0.90487)))),
            0.3340998 + x * (0.73916 + x * (-2.94589 + x * (5.48795 + x * (-5.52477 + x * 2.17525))))
        ),
        vec3<f32>(0.0), vec3<f32>(1.0)
    );
}

fn palette_inferno(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return clamp(
        vec3<f32>(
            0.0002189 + x * (0.1065 + x * (11.60249 + x * (-41.7039 + x * (55.0215 + x * -24.00824)))),
            0.001652 + x * (-0.69594 + x * (4.2625 + x * (-8.4392 + x * (7.7773 + x * -2.74041)))),
            -0.019480 + x * (3.9314 + x * (-15.9347 + x * (44.3551 + x * (-55.2813 + x * 22.9486))))
        ),
        vec3<f32>(0.0), vec3<f32>(1.0)
    );
}

fn palette_bwr(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    if x < 0.5 {
        let s = x * 2.0;
        return vec3<f32>(s, s, 1.0);
    }
    let s = (x - 0.5) * 2.0;
    return vec3<f32>(1.0, 1.0 - s, 1.0 - s);
}

// nilearn cold_hot pair: exact piecewise matplotlib `hot` + its
// channel-rotated "cold" twin. Mirrors renderer/colormap.rs.
fn hot_ramp(x: f32) -> vec3<f32> {
    let c = clamp(x, 0.0, 1.0);
    var r = 1.0;
    if c < 0.365079 { r = 0.0416 + c * (1.0 - 0.0416) / 0.365079; }
    let g = clamp((c - 0.365079) / (0.746032 - 0.365079), 0.0, 1.0);
    let b = clamp((c - 0.746032) / (1.0 - 0.746032), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn palette_cold_hot(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    if x < 0.5 { return hot_ramp(1.0 - 2.0 * x).zyx; }
    return hot_ramp(2.0 * x - 1.0);
}

fn palette_cold_white_hot(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    if x < 0.5 { return hot_ramp(2.0 * x).zyx; }
    return hot_ramp(2.0 - 2.0 * x);
}

// Palette codes shared with fixel.wgsl and renderer/colormap.rs
// (palette_code_rgb): 2=plasma 3=viridis 4=inferno 5=grayscale
// 6=blue-white-red 7=cold-hot 8=cold-white-hot.
fn sample_palette(mode: u32, t: f32) -> vec3<f32> {
    if mode == 2u { return palette_plasma(t); }
    if mode == 3u { return palette_viridis(t); }
    if mode == 4u { return palette_inferno(t); }
    if mode == 5u { return vec3<f32>(t, t, t); }
    if mode == 6u { return palette_bwr(t); }
    if mode == 7u { return palette_cold_hot(t); }
    if mode == 8u { return palette_cold_white_hot(t); }
    return palette_bwr(t);
}

fn gate_factor(sample: f32, params: vec4<f32>) -> f32 {
    // Treat NaN/Inf samples (our sentinel for "no scalars connected") as gate-disabled.
    // Direct `sample != sample` can be folded away under fast-math on some backends,
    // so inspect the bit pattern: exponent all-1s means NaN or Inf.
    let bits = bitcast<u32>(sample);
    if (bits & 0x7F800000u) == 0x7F800000u {
        return 1.0;
    }
    let lo = params.x;
    let hi = params.y;
    if sample <= lo {
        return params.z;
    }
    if sample >= hi {
        return params.w;
    }
    let t = clamp((sample - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    return mix(params.z, params.w, t);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let amp = amplitudes.values[in.amplitude_offset + in.vertex_index];
    let sm = select(1.0, uniforms.scale_mul, uniforms.scale_mul > 0.0);
    let size_factor = gate_factor(size_samples.values[in.instance_index], uniforms.size_gate);
    let world = in.center + in.direction * amp * in.scale * sm * size_factor;
    out.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.normal = normalize(in.direction);
    if uniforms.color_mode == 0u {
        out.color = abs(in.direction);
    } else if uniforms.color_mode == 1u {
        out.color = vec3<f32>(0.92, 0.92, 0.92);
    } else {
        let an = select(1.0, uniforms.post_params.w, uniforms.post_params.w > 0.0);
        out.color = sample_palette(uniforms.color_mode, clamp(amp / an, 0.0, 1.0));
    }
    out.center = in.center;
    out.draw_alpha = select(0.0, 1.0, in.contact_count >= in.min_contacts);
    out.draw_alpha *= gate_factor(opacity_samples.values[in.instance_index], uniforms.opacity_gate);
    if uniforms.draw_step > 1u && (in.instance_index % uniforms.draw_step) != 0u {
        out.draw_alpha = 0.0;
    }
    out.ndc_xy = out.clip_position.xy / out.clip_position.w;
    return out;
}

fn apply_depth_fade(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    if uniforms.fog_color.w < 0.5 {
        return color;
    }
    let dist = length(uniforms.camera_pos - world_pos);
    let fade = clamp(
        (dist - uniforms.fog_params.x) / max(uniforms.fog_params.y - uniforms.fog_params.x, 1e-4),
        0.0,
        1.0,
    );
    return mix(color, uniforms.fog_color.rgb, fade);
}

fn apply_post_color(color: vec3<f32>, ndc_xy: vec2<f32>) -> vec3<f32> {
    let uv = ndc_xy * 0.5 + vec2<f32>(0.5, 0.5);
    let centered = uv - vec2<f32>(0.5, 0.5);
    let radius = length(centered) * 1.41421356;
    let vignette = 1.0 - uniforms.post_params.z * smoothstep(0.2, 1.0, radius);
    let graded =
        (color * uniforms.post_params.x - vec3<f32>(0.5)) * uniforms.post_params.y + vec3<f32>(0.5);
    return clamp(graded * vignette, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.draw_alpha <= 0.0 {
        discard;
    }
    // Slab clipping: plane-distance test (works for oblique slices).
    if uniforms.slab_half_width > 0.0 {
        let dist = dot(in.center - uniforms.slab_center, uniforms.slab_normal);
        if abs(dist) > uniforms.slab_half_width {
            discard;
        }
    }

    var lit = in.color;
    if uniforms.ambient_strength < 0.999 {
        let n = normalize(in.normal);
        let view_dir = normalize(uniforms.camera_pos - in.world_pos);
        let key = max(dot(n, normalize(vec3<f32>(0.45, 0.55, 1.0))), 0.0);
        let fill = max(dot(n, normalize(vec3<f32>(-0.7, 0.2, 0.65))), 0.0);
        let head = max(dot(n, view_dir), 0.0);
        let shiny = 4.0 + uniforms.gloss * 60.0;
        let spec = pow(head, shiny) * (uniforms.specular_strength + uniforms.gloss);
        let shade = uniforms.ambient_strength
            + uniforms.key_strength * key
            + uniforms.fill_strength * fill
            + uniforms.headlight_mix * head;
        lit = in.color * shade + vec3<f32>(spec);
    }
    let faded = apply_depth_fade(lit, in.world_pos);
    return vec4<f32>(apply_post_color(faded, in.ndc_xy), uniforms.opacity);
}

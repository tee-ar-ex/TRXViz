struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    slab_half_width: f32,   // 0 = disabled
    slab_normal: vec3<f32>, // unit normal to slice plane
    draw_step: u32,
    slab_center: vec3<f32>, // world-space point on slice plane
    line_width: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    opacity: f32,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    post_params: vec4<f32>,
    color_params: vec4<f32>, // x=colormap u32-as-f32, y=scalar_min, z=scalar_max, w=reserved
    opacity_gate: vec4<f32>, // x=range_min, y=range_max, z=below, w=above
    style_params: vec4<f32>, // x=directional∈[0,1], yzw=reserved
}

// Piecewise-linear opacity gate. Below `range_min` → `below`; above
// `range_max` → `above`; linearly interpolated in between. Matches the
// semantics used in `shaders/glyph.wgsl::gate_factor`.
fn gate_factor(sample: f32, params: vec4<f32>) -> f32 {
    // NaN / ±Inf → treat as gate-disabled (full alpha) so renders don't
    // blank out when a scalar stream is missing.
    if sample != sample || sample == sample * 0.5 + 1e30 {
        return 1.0;
    }
    let range_min = params.x;
    let range_max = params.y;
    let below = params.z;
    let above = params.w;
    if sample <= range_min {
        return below;
    }
    if sample >= range_max {
        return above;
    }
    let t = (sample - range_min) / max(range_max - range_min, 1e-6);
    return mix(below, above, t);
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) center: vec3<f32>,
    @location(2) direction: vec3<f32>,
    @location(3) length: f32,
    @location(4) scalar: f32,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) draw_alpha: f32,
    @location(3) ndc_xy: vec2<f32>,
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

fn sample_palette(mode: u32, t: f32) -> vec3<f32> {
    if mode == 2u { return palette_plasma(t); }
    if mode == 3u { return palette_viridis(t); }
    if mode == 4u { return palette_inferno(t); }
    return palette_bwr(t);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let lm = select(1.0, uniforms.post_params.w, uniforms.post_params.w > 0.0);
    let eff_len = in.length * lm;
    // `directional` ∈ [0, 1]: 0 ⇒ bidirectional line through the voxel
    // centre (quad_pos.x ∈ [−1, 1]); 1 ⇒ half-arrow starting at the
    // centre and extending along the peak (quad_pos.x ∈ [0, 1]). For
    // asymmetric ODFs (full-basis descoteaux SH) each peak has a true
    // forward direction, so the half-arrow makes the asymmetry visible.
    let directional = clamp(uniforms.style_params.x, 0.0, 1.0);
    let qx = mix(in.quad_pos.x, in.quad_pos.x * 0.5 + 0.5, directional);
    let along = in.direction * eff_len * qx;
    let world = in.center + along;

    let start_t = mix(-1.0, 0.0, directional);
    let end_t = 1.0;
    let clip = uniforms.view_proj * vec4<f32>(world, 1.0);
    let clip_end = uniforms.view_proj * vec4<f32>(in.center + in.direction * eff_len * end_t, 1.0);
    let clip_start =
        uniforms.view_proj * vec4<f32>(in.center + in.direction * eff_len * start_t, 1.0);

    let ndc_a = clip_start.xy / clip_start.w;
    let ndc_b = clip_end.xy / clip_end.w;
    let screen_dir = normalize(ndc_b - ndc_a);
    let perp = vec2<f32>(-screen_dir.y, screen_dir.x);

    var final_clip = clip;
    final_clip = vec4<f32>(
        final_clip.x + perp.x * in.quad_pos.y * uniforms.line_width * final_clip.w,
        final_clip.y + perp.y * in.quad_pos.y * uniforms.line_width * final_clip.w,
        final_clip.z,
        final_clip.w,
    );

    out.clip_position = final_clip;

    let cmap = u32(uniforms.color_params.x);
    if cmap == 0u {
        out.color = abs(in.direction);
    } else {
        let lo = uniforms.color_params.y;
        let hi = uniforms.color_params.z;
        let denom = max(hi - lo, 1e-6);
        let t = clamp((in.scalar - lo) / denom, 0.0, 1.0);
        out.color = sample_palette(cmap, t);
    }
    out.center = in.center;
    out.draw_alpha = gate_factor(in.scalar, uniforms.opacity_gate);

    if uniforms.draw_step > 1u && (in.instance_index % uniforms.draw_step) != 0u {
        out.draw_alpha = 0.0;
    }
    out.ndc_xy = final_clip.xy / final_clip.w;
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
    if uniforms.slab_half_width > 0.0 {
        let dist = dot(in.center - uniforms.slab_center, uniforms.slab_normal);
        if abs(dist) > uniforms.slab_half_width {
            discard;
        }
    }

    let lit = in.color * (uniforms.ambient_strength + uniforms.key_strength);
    let faded = apply_depth_fade(lit, in.center);
    return vec4<f32>(apply_post_color(faded, in.ndc_xy), uniforms.opacity * in.draw_alpha);
}

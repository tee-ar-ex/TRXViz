struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,
    camera_pos: vec3<f32>,
    shininess: f32,
    map_opacity: f32,
    map_threshold: f32,
    scalar_min: f32,
    scalar_max: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    scalar_enabled: u32,
    vertex_color_enabled: u32,
    colormap: u32,
    _pad0: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    post_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) vertex_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) vertex_color: vec4<f32>,
    @location(4) ndc_xy: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.clip_position = uniforms.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_normal = world_normal;
    out.scalar = in.scalar;
    out.vertex_color = in.vertex_color;
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

fn colormap_blue_white_red(t: f32) -> vec3<f32> {
    if t < 0.5 {
        let s = t * 2.0;
        return vec3<f32>(s, s, 1.0);
    }
    let s = (1.0 - t) * 2.0;
    return vec3<f32>(1.0, s, s);
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
    let x = t * 9.0;
    let i = clamp(i32(floor(x)), 0, 8);
    let f = fract(x);
    let a = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i];
    let b = array<vec3<f32>, 10>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)[i + 1];
    return mix(a, b, f);
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
    let c8 = vec3<f32>(0.978, 0.893, 0.306);
    let x = t * 8.0;
    let i = clamp(i32(floor(x)), 0, 7);
    let f = fract(x);
    let a = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i];
    let b = array<vec3<f32>, 9>(c0, c1, c2, c3, c4, c5, c6, c7, c8)[i + 1];
    return mix(a, b, f);
}

fn sample_colormap(t: f32, cmap: u32) -> vec3<f32> {
    if cmap == 1u {
        return colormap_viridis(t);
    }
    if cmap == 2u {
        return colormap_inferno(t);
    }
    return colormap_blue_white_red(t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let V = normalize(uniforms.camera_pos - in.world_pos);
    var N = normalize(in.world_normal);
    // Two-sided lighting keeps meshes readable even with mixed winding.
    if dot(N, V) < 0.0 {
        N = -N;
    }
    var base_rgb = uniforms.color.rgb;
    var alpha = uniforms.color.a;
    if uniforms.vertex_color_enabled == 1u {
        base_rgb = in.vertex_color.rgb;
        alpha = uniforms.color.a * in.vertex_color.a;
    }
    if uniforms.scalar_enabled == 1u {
        let denom = max(uniforms.scalar_max - uniforms.scalar_min, 1e-6);
        let t = clamp((in.scalar - uniforms.scalar_min) / denom, 0.0, 1.0);
        let map_alpha = uniforms.map_opacity * step(uniforms.map_threshold, t);
        let map_rgb = sample_colormap(t, uniforms.colormap);
        base_rgb = mix(uniforms.color.rgb, map_rgb, map_alpha);
    }

    let key = max(dot(N, normalize(vec3<f32>(0.45, 0.55, 0.9))), 0.0);
    let fill = max(dot(N, normalize(vec3<f32>(-0.7, 0.2, 0.65))), 0.0);
    let head = max(dot(N, V), 0.0);
    let spec = pow(head, uniforms.shininess) * uniforms.specular_strength;
    let shade = uniforms.ambient_strength
        + uniforms.key_strength * key
        + uniforms.fill_strength * fill
        + uniforms.headlight_mix * head;

    let lit = base_rgb * shade + vec3<f32>(spec);
    let faded = apply_depth_fade(lit, in.world_pos);
    return vec4<f32>(apply_post_color(faded, in.ndc_xy), alpha);
}

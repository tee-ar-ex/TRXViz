struct Uniforms {
    view_proj:  mat4x4<f32>,
    camera_pos: vec3<f32>,
    opacity:    f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    post_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:    vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color:        vec4<f32>,
    @location(3) ndc_xy:       vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos     = in.position;
    out.world_normal  = normalize(in.normal);
    out.color         = in.color;
    out.ndc_xy        = out.clip_position.xy / out.clip_position.w;
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
    let V = normalize(uniforms.camera_pos - in.world_pos);
    let N = normalize(in.world_normal);
    let key = abs(dot(N, normalize(vec3<f32>(0.45, 0.55, 0.7))));
    let fill = abs(dot(N, normalize(vec3<f32>(-0.7, 0.2, 0.65))));
    let head = abs(dot(N, V));
    let spec = pow(head, 24.0) * uniforms.specular_strength;
    let shade = uniforms.ambient_strength
        + uniforms.key_strength * key
        + uniforms.fill_strength * fill
        + uniforms.headlight_mix * head;
    let lit  = in.color.rgb * shade + vec3<f32>(spec);
    let faded = apply_depth_fade(lit, in.world_pos);
    return vec4<f32>(apply_post_color(faded, in.ndc_xy), uniforms.opacity);
}

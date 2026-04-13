struct Uniforms {
    top_color: vec4<f32>,
    bottom_color: vec4<f32>,
    post_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
}

fn apply_post_color(color: vec3<f32>, ndc_xy: vec2<f32>) -> vec3<f32> {
    let exposure = uniforms.post_params.x;
    let contrast = uniforms.post_params.y;
    let vignette_strength = uniforms.post_params.z;
    let uv = ndc_xy * 0.5 + vec2<f32>(0.5, 0.5);
    let centered = uv - vec2<f32>(0.5, 0.5);
    let radius = length(centered) * 1.41421356;
    let vignette = 1.0 - vignette_strength * smoothstep(0.2, 1.0, radius);
    let graded = (color * exposure - vec3<f32>(0.5)) * contrast + vec3<f32>(0.5);
    return clamp(graded * vignette, vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    )[vertex_index];
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.ndc_xy = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let gradient_t = clamp((in.ndc_xy.y + 1.0) * 0.5, 0.0, 1.0);
    let mode = uniforms.post_params.w;
    let base = mix(uniforms.bottom_color.rgb, uniforms.top_color.rgb, gradient_t * mode);
    return vec4<f32>(apply_post_color(base, in.ndc_xy), 1.0);
}

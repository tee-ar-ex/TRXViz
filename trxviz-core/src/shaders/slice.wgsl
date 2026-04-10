struct Uniforms {
    view_proj: mat4x4<f32>,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = textureSample(volume_texture, volume_sampler, in.tex_coord).r;
    // Apply windowing
    let lo = uniforms.window_center - uniforms.window_width * 0.5;
    let hi = uniforms.window_center + uniforms.window_width * 0.5;
    let val = clamp((intensity - lo) / max(hi - lo, 0.001), 0.0, 1.0);

    var color: vec3<f32>;
    switch uniforms.colormap {
        case 1u: { color = colormap_hot(val); }
        case 2u: { color = colormap_cool(val); }
        case 3u: { color = colormap_red_yellow(val); }
        case 4u: { color = colormap_blue_lightblue(val); }
        default: { color = vec3<f32>(val, val, val); }
    }

    return vec4<f32>(color, uniforms.opacity);
}

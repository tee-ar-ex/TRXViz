// Composite slice shader: samples a precomposited 2D RGBA texture and
// outputs it directly. Compositing (multi-layer windowing, threshold,
// colormap, alpha-over) happens on the CPU in `composite_slice_into`
// and the resulting RGBA image is uploaded into this texture.

struct Uniforms {
    view_proj: mat4x4<f32>,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var slice_texture: texture_2d<f32>;
@group(0) @binding(2) var slice_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(slice_texture, slice_sampler, in.uv);
    return vec4<f32>(c.rgb, c.a * uniforms.opacity);
}

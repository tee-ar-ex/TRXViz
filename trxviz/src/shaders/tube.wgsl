struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    render_style: u32,
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
    tube_radius: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_pos = in.position;
    out.world_normal = normalize(in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if uniforms.slab_axis < 3u {
        var coord: f32;
        if uniforms.slab_axis == 0u {
            coord = in.world_pos.x;
        } else if uniforms.slab_axis == 1u {
            coord = in.world_pos.y;
        } else {
            coord = in.world_pos.z;
        }
        if coord < uniforms.slab_min || coord > uniforms.slab_max {
            discard;
        }
    }

    let normal = normalize(in.world_normal);
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    let key = max(dot(normal, normalize(vec3<f32>(0.45, 0.55, 1.0))), 0.0);
    let fill = max(dot(normal, normalize(vec3<f32>(-0.7, 0.2, 0.65))), 0.0);
    let head = max(dot(normal, view_dir), 0.0);
    let specular = pow(head, 32.0) * uniforms.specular_strength;
    let lit = uniforms.ambient_strength
        + uniforms.key_strength * key
        + uniforms.fill_strength * fill
        + uniforms.headlight_mix * head
        + specular;

    return vec4<f32>(in.color.rgb * lit, in.color.a);
}

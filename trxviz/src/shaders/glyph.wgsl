struct Uniforms {
    view_proj: mat4x4<f32>,
    slab_axis: u32,
    color_mode: u32,
    draw_step: u32,
    slab_min: f32,
    slab_max: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad1: f32,
}

struct Amplitudes {
    values: array<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> amplitudes: Amplitudes;

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
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let amp = amplitudes.values[in.amplitude_offset + in.vertex_index];
    let world = in.center + in.direction * amp * in.scale;
    out.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.normal = normalize(in.direction);
    if uniforms.color_mode == 0u {
        out.color = abs(in.direction);
    } else {
    out.color = vec3<f32>(0.92, 0.92, 0.92);
    }
    out.center = in.center;
    out.draw_alpha = select(0.0, 1.0, in.contact_count >= in.min_contacts);
    if uniforms.draw_step > 1u && (in.instance_index % uniforms.draw_step) != 0u {
        out.draw_alpha = 0.0;
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.draw_alpha <= 0.0 {
        discard;
    }
    if uniforms.slab_axis < 3u {
        var coord: f32;
        if uniforms.slab_axis == 0u {
            coord = in.center.x;
        } else if uniforms.slab_axis == 1u {
            coord = in.center.y;
        } else {
            coord = in.center.z;
        }
        if coord < uniforms.slab_min || coord > uniforms.slab_max {
            discard;
        }
    }

    var lit = in.color;
    if uniforms.ambient_strength < 0.999 {
        let n = normalize(in.normal);
        let view_dir = normalize(-in.world_pos);
        let key = max(dot(n, normalize(vec3<f32>(0.45, 0.55, 1.0))), 0.0);
        let fill = max(dot(n, normalize(vec3<f32>(-0.7, 0.2, 0.65))), 0.0);
        let head = max(dot(n, view_dir), 0.0);
        let spec = pow(head, 20.0) * uniforms.specular_strength;
        let shade = uniforms.ambient_strength
            + uniforms.key_strength * key
            + uniforms.fill_strength * fill
            + uniforms.headlight_mix * head;
        lit = in.color * shade + vec3<f32>(spec);
    }
    return vec4<f32>(lit, 0.95);
}

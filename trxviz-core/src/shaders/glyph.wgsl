struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
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
    _pad1: vec2<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    post_params: vec4<f32>,
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
    @location(5) ndc_xy: vec2<f32>,
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
        let view_dir = normalize(uniforms.camera_pos - in.world_pos);
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
    let faded = apply_depth_fade(lit, in.world_pos);
    return vec4<f32>(apply_post_color(faded, in.ndc_xy), 0.95);
}

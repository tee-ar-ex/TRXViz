struct Uniforms {
    view_proj: mat4x4<f32>,
    // camera_pos packed with render_style to avoid vec3 alignment waste
    camera_pos: vec3<f32>,
    render_style: u32,   // 0=flat, 1=illuminated, 2=tubes (unused here), 3=depth_cue
    slab_axis: u32,      // 0=X, 1=Y, 2=Z, 3=disabled
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
    @location(1) color: vec4<f32>,
    @location(2) tangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) tangent: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_pos = in.position;
    out.tangent = in.tangent;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Slab clipping
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

    if uniforms.render_style == 1u {
        // Illuminated streamlines (Zoeckler 1996)
        // Diffuse and specular computed from angle between light/view and the tangent.
        let key_dir = normalize(vec3<f32>(0.45, 0.55, 1.0));
        let fill_dir = normalize(vec3<f32>(-0.7, 0.2, 0.65));
        let t = normalize(in.tangent);
        let tk = dot(t, key_dir);
        let tf = dot(t, fill_dir);
        let key = sqrt(max(0.0, 1.0 - tk * tk));
        let fill = sqrt(max(0.0, 1.0 - tf * tf));

        let view_dir = normalize(uniforms.camera_pos - in.world_pos);
        let tv = dot(t, view_dir);
        let head = sqrt(max(0.0, 1.0 - tv * tv));
        let specular = pow(head, 16.0) * uniforms.specular_strength;

        let lit = uniforms.ambient_strength
            + uniforms.key_strength * key
            + uniforms.fill_strength * fill
            + uniforms.headlight_mix * head
            + specular;
        return vec4<f32>(in.color.rgb * lit, in.color.a);

    } else if uniforms.render_style == 3u {
        // Depth cueing: brightness fades with distance from camera
        let dist = length(uniforms.camera_pos - in.world_pos);
        // tube_radius field repurposed as depth_far when in depth-cue mode
        let depth_far = max(uniforms.tube_radius, 1.0);
        let cue = 1.0 - clamp(dist / depth_far, 0.0, 0.65);
        return vec4<f32>(in.color.rgb * cue, in.color.a);
    }

    // Flat (default)
    return in.color;
}

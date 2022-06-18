fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn saturate_v2(x: vec2<f32>) -> vec2<f32> {
    return clamp(x, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
}

fn saturate_v4(x: vec4<f32>) -> vec4<f32> {
    return clamp(x, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));
}

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] texcoord0: vec2<f32>;
    [[location(3)]] uva1: vec4<f32>;
    [[location(4)]] uva2: vec4<f32>;
    [[location(5)]] uva3: vec4<f32>;
    [[location(6)]] uva4: vec4<f32>;
    [[location(7)]] color0: vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] color0: vec4<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] texcoord0: vec2<f32>;
    [[location(3)]] texcoord1: vec2<f32>;
    [[location(4)]] eye: vec3<f32>;
    [[location(5)]] shadow0: vec4<f32>;
    // builtin("PointSize") or psize when NANOEM_IO_HAS_POINT
};

struct FragmentInput {
    [[location(0)]] color0: vec4<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] texcoord0: vec2<f32>;
    [[location(3)]] texcoord1: vec2<f32>;
    [[location(4)]] eye: vec3<f32>;
    [[location(5)]] shadow0: vec4<f32>;
};

let alpha_test_threshold: f32 = 0.005;

struct ModelParameters {
    model_matrix: mat4x4<f32>;
    model_view_matrix: mat4x4<f32>;
    model_view_projection_matrix: mat4x4<f32>;
    light_view_projection_matrix: mat4x4<f32>;
    light_color: vec4<f32>;
    light_direction: vec4<f32>;
    camera_position: vec4<f32>;
    material_ambient: vec4<f32>;
    material_diffuse: vec4<f32>;
    material_specular: vec4<f32>;
    enable_vertex_color: vec4<f32>;
    diffuse_texture_blend_factor: vec4<f32>;
    sphere_texture_blend_factor: vec4<f32>;
    toon_texture_blend_factor: vec4<f32>;
    use_texture_sampler: vec4<f32>;
    sphere_texture_type: vec4<f32>;
    shadow_map_size: vec4<f32>;
};

[[group(1), binding(0)]]
var<uniform> model_parameters: ModelParameters;
[[group(0), binding(2)]]
var diffuse_texture: texture_2d<f32>;
[[group(0), binding(3)]]
var diffuse_texture_sampler: sampler;
[[group(0), binding(4)]]
var sphere_map_texture: texture_2d<f32>;
[[group(0), binding(5)]]
var sphere_map_texture_sampler: sampler;
[[group(0), binding(6)]]
var toon_texture: texture_2d<f32>;
[[group(0), binding(7)]]
var toon_texture_sampler: sampler;

fn has_diffuse_texture() -> bool {
    return model_parameters.use_texture_sampler.x != 0.0;
}

fn has_sphere_texture() -> bool {
    return model_parameters.use_texture_sampler.y != 0.0;
}

fn has_toon_texture() -> bool {
    return model_parameters.use_texture_sampler.z != 0.0;
}

fn has_shadow_map_texture() -> bool {
    return model_parameters.use_texture_sampler.w != 0.0;
}

fn is_sphere_texture_multiply() -> bool {
    return model_parameters.sphere_texture_type.x != 0.0;
}

fn is_sphere_texture_additive() -> bool {
    return model_parameters.sphere_texture_type.z != 0.0;
}

fn is_sphere_texture_as_sub_texture() -> bool {
    return model_parameters.sphere_texture_type.y != 0.0;
}

fn coverage_alpha(frag_input: FragmentInput, rgba: vec4<f32>) -> vec4<f32> {
    var result = rgba;
    if (has_diffuse_texture()) {
        let texcoord0 = frag_input.texcoord0;
        let texelr = textureSample(diffuse_texture, diffuse_texture_sampler, texcoord0).r;
        result.a = result.a * texelr;
    }
    if (has_sphere_texture()) {
        let texcoord1 = frag_input.texcoord1;
        if (is_sphere_texture_multiply()) {
            result.a = result.a * textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        } else if (is_sphere_texture_as_sub_texture()) {
            result.a = result.a * textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        } else if (is_sphere_texture_additive()) {
            result.a = result.a * textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        }
    }
    if (has_toon_texture()) {
        let light_position = normalize(-model_parameters.light_direction.xyz);
        let normal = normalize(frag_input.normal);
        let y = 1.0 - saturate(dot(normal, light_position) * 16.0 + 0.5);
        result.a = result.a * textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, y)).a;
    }
    if (result.a - alpha_test_threshold < 0.0) {
        discard;
    }
    return result;
}

[[stage(vertex)]]
fn vs_main(vin: VertexInput) -> VertexOutput {
    let position = vec4<f32>(vin.position, 1.0);
    let normal = vec4<f32>(vin.normal, 0.0);
    let sphere = normalize(model_parameters.model_view_matrix * normal).xy * 0.5 + 0.5;
    var vout: VertexOutput;
    vout.position = model_parameters.model_view_projection_matrix * position;
    vout.normal = normal.xyz;
    vout.texcoord0 = vin.texcoord0;
    vout.texcoord1 = sphere;
    return vout;
}

[[stage(fragment)]]
fn fs_main(fin: FragmentInput) -> [[location(0)]] vec4<f32> {
    return coverage_alpha(fin, model_parameters.light_color);
}
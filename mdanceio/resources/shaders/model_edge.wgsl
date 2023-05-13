struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord0: vec2<f32>,
    @location(3) uva1: vec4<f32>,
    @location(4) uva2: vec4<f32>,
    @location(5) uva3: vec4<f32>,
    @location(6) uva4: vec4<f32>,
    @location(7) color0: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color0: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord0: vec2<f32>,
    @location(3) texcoord1: vec2<f32>,
    @location(4) eye: vec3<f32>,
    @location(5) shadow0: vec4<f32>,
};

struct FragmentInput {
    @location(0) color0: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord0: vec2<f32>,
    @location(3) texcoord1: vec2<f32>,
    @location(4) eye: vec3<f32>,
    @location(5) shadow0: vec4<f32>,
}

struct ModelUniform {
    model_matrix: mat4x4<f32>,
    model_view_matrix: mat4x4<f32>,
    model_view_projection_matrix: mat4x4<f32>,
    light_view_projection_matrix: mat4x4<f32>,
    light_color: vec4<f32>,
    light_direction: vec4<f32>,
    camera_position: vec4<f32>,
    shadow_map_size: vec4<f32>,
}

struct MaterialUniform {
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    edge_color: vec4<f32>,
    enable_vertex_color: vec4<f32>,
    diffuse_blend_factor: vec4<f32>,
    sphere_blend_factor: vec4<f32>,
    toon_blend_factor: vec4<f32>,
    use_texture_sampler: vec4<f32>,
    sphere_texture_type: vec4<f32>,
    edge_size: f32,
}

@group(0)
@binding(0)
var diffuse_texture: texture_2d<f32>;
@group(0)
@binding(1)
var diffuse_texture_sampler: sampler;
@group(0)
@binding(2)
var sphere_map_texture: texture_2d<f32>;
@group(0)
@binding(3)
var sphere_map_texture_sampler: sampler;
@group(0)
@binding(4)
var toon_texture: texture_2d<f32>;
@group(0)
@binding(5)
var toon_texture_sampler: sampler;

@group(1)
@binding(0)
var<uniform> model_uniform: ModelUniform;

@group(1)
@binding(1)
var<uniform> material_uniform: MaterialUniform;

@group(2) @binding(0)
var shadow_texture: texture_2d<f32>;
@group(2) @binding(1)
var shadow_texture_sampler: sampler;

fn has_diffuse_texture() -> bool {
    return material_uniform.use_texture_sampler.x != 0.0;
}

fn has_sphere_texture() -> bool {
    return material_uniform.use_texture_sampler.y != 0.0;
}

fn has_toon_texture() -> bool {
    return material_uniform.use_texture_sampler.z != 0.0;
}

fn has_shadow_map_texture() -> bool {
    return material_uniform.use_texture_sampler.w != 0.0;
}

fn is_sphere_texture_multiply() -> bool {
    return material_uniform.sphere_texture_type.x != 0.0;
}

fn is_sphere_texture_additive() -> bool {
    return material_uniform.sphere_texture_type.z != 0.0;
}

fn is_sphere_texture_as_sub_texture() -> bool {
    return material_uniform.sphere_texture_type.y != 0.0;
}

const alpha_test_threshold: f32 = 0.005;

const threshold_type1: f32 = 1500.0;
const threshold_type2: f32 = 8000.0;

const toon_factor: f32 = 3.0;

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
        let light_position = normalize(-model_uniform.light_direction.xyz);
        let normal = normalize(frag_input.normal);
        let y = 1.0 - saturate(dot(normal, light_position) * 16.0 + 0.5);
        result.a = result.a * textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, y)).a;
    }
    if (result.a - alpha_test_threshold < 0.0) {
        discard;
    }
    return result;
}

@vertex
fn vs_main(vin: VertexInput) -> VertexOutput {
    let position = vec4<f32>(vin.position, 1.0);
    let normal = vec4<f32>(vin.normal, 0.0);
    let sphere = normalize(model_uniform.model_view_matrix * normal).xy * 0.5 + 0.5;
    var vout: VertexOutput;
    vout.position = model_uniform.model_view_projection_matrix * position;
    vout.normal = normal.xyz;
    vout.texcoord0 = vin.texcoord0;
    vout.texcoord1 = sphere;
    return vout;
}

@fragment
fn fs_main(fin: FragmentInput) -> @location(0) vec4<f32> {
    return coverage_alpha(fin, material_uniform.edge_color);
}
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
    [[buildin(position)]] position: vec4<f32>;
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
}

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
}

@group(1)
@binding(0)
var<uniform> model_parameters: ModelParameters;
@group(0)
@binding(2)
var diffuse_texture: texture_2d<f32>;
@group(0)
@binding(3)
var diffuse_texture_sampler: sampler;
@group(0)
@binding(4)
var sphere_map_texture: texture_2d<f32>;
@group(0)
@binding(5)
var sphere_map_texture_sampler: sampler;
@group(0)
@binding(6)
var toon_texture: texture_2d<f32>;
@group(0)
@binding(7)
var toon_texture_sampler: sampler;

fn has_diffuse_texture() -> bool {
    model_parameters.use_texture_sampler.x != 0.0
}

fn has_sphere_texture() -> bool {
    model_parameters.use_texture_sampler.y != 0.0
}

fn has_toon_texture() -> bool {
    model_parameters.use_texture_sampler.z != 0.0
}

fn has_shadow_map_texture() -> bool {
    model_parameters.use_texture_sampler.w != 0.0
}

fn is_sphere_texture_multiply() -> bool {
    model_parameters.sphere_texture_type.x != 0.0
}

fn is_sphere_texture_additive() -> bool {
    model_parameters.sphere_texture_type.z != 0.0
}

fn is_sphere_texture_as_sub_texture() -> bool {
    model_parameters.sphere_texture_type.y != 0.0
}

fn coverage_alpha(input: FragmentInput, rgba: vec4<f32>) -> vec4<f32> {
    if (has_diffuse_texture()) {
        let texcoord0 = input.texcoord0;
        let texel = textureSample(diffuse_texture, diffuse_texture_sampler, texcoord0);
        rgba.a *= texel.a;
    }
    if (has_sphere_texture()) {
        let texcoord1 = input.texcoord1;
        if (is_sphere_texture_multiply()) {
            rgba.a *= textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        } 
        else if (is_sphere_texture_as_sub_texture()) {
            rgba.a *= textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        } 
        else if (is_sphere_texture_additive()) {
            rgba.a *= textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1).a;
        }
    }
    if (has_toon_texture()) {
        let light_position = normalize(-model_parameters.light_direction.xyz);
        let normal = normalize(input.normal);
        let y = 1.0 - saturate(dot(normal, light_position) * 16.0 + 0.5);
        rgba.a *= textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, y)).a;
    }
    if (rgba.a - alpha_test_threshold < 0.0) {
        discard;
    }
    return rgba;
}

@group(0)
@binding(0)
var shadow_texture: texture_2d<f32>;
@group(0)
@binding(1)
var shadow_texture_sampler: sampler;

let threshold_type1: f32 = 1500.0;
let threshold_type2: f32 = 8000.0;

fn shadow_coverage(texcoord: vec4<f32>, shadow_map_size: vec4<f32>) -> f32 {
    let receiver_depth = texcoord.z;
    let shadow_map_depth = textureSample(shadow_texture, shadow_texture_sampler, texcoord.xy).x;
    var component = saturate(receiver_depth - shadow_map_depth);
    let coverage_type_int: i32 = i32(shadow_map_size.w);
    if (coverage_type_int == 2) {
        component = saturate(component * threshold_type2 * texcoord.y - 0.3);
    } 
    else if (coverage_type_int == 1) {
        component = saturate(component * threshold_type1 - 0.3);
    }
    return 1.0 - component;
}

@stage(vertex)
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    let position = vec4<f32>(input.position, 1.0);
    let normal = vec4<f32>(input.normal, 0.0);
    let color = (model_parameters.material_diffuse.rgb * model_parameters.light_color.rgb + model_parameters.material_ambient.rgb) * select(vec3<f32>(1.0, 1.0, 1.0), input.uva1.xyz, vec3<bool>(model_parameters.enable_vertex_color.xyz));
    let sphere = normalize(model_parameters.model_view_matrix * normal).xy * 0.5 + 0.5;
    var output: VertexOutput;
    output.position = model_parameters.model_view_projection_matrix * position;
    output.normal = normal.xyz;
    output.eye = model_parameters.camera_position.xyz - (model_parameters.model_matrix * position).xyz;
    output.texcoord0 = input.texcoord0;
    output.texcoord1 = sphere;
    output.color0 = saturate_v4(vec4<f32>(color, model_parameters.material_diffuse.a));
    output.shadow0 = model_parameters.light_view_projection_matrix * position;
    // TODO: when NANOEM_IO_HAS_POINT
    return output;
}

let toon_factor: f32 = 3.0;

@stage(fragment)
fn fs_main(
    input: FragmentInput,
) -> vec4<f32> {
    var material_color = input.color0;
    if (has_diffuse_texture()) {
        let texcoord0 = input.texcoord0;
        let texel = textureSample(diffuse_texture, diffuse_texture_sampler, texcoord0);
        material_color.rgb *= (texel.rgb * model_parameters.diffuse_texture_blend_factor.rgb) * model_parameters.diffuse_texture_blend_factor.a;
        material_color.a *= texel.a;
    }
    if (has_sphere_texture()) {
        let texcoord1 = input.texcoord1;
        if (is_sphere_texture_multiply()) {
            let texel = textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
            material_color.rgb *= (texel.rgb * model_parameters.sphere_texture_blend_factor.rgb) * model_parameters.sphere_texture_blend_factor.a;
            material_color.a *= texel.a;
        }
        else if (is_sphere_texture_as_sub_texture()) {
            material_color *= textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
        }
        else if (is_sphere_texture_additive()) {
            let texel = textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
            material_color.rgb += (texel.rgb * model_parameters.sphere_texture_blend_factor.rgb) * model_parameters.sphere_texture_blend_factor.a;
            material_color.a *= texel.a;
        }
    }
    let light_position = -model_parameters.light_direction.xyz;
    if (has_shadow_map_texture()) {
        let texcoord0 = input.shadow0 / input.shadow0.w;
        if (all(saturate_v2(texcoord0.xy) == texcoord0.xy)) {
            let toon_color = textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, 1.0));
            var shadow_color = material_color;
            shadow_color.rgb *= (toon_color.rgb * model_parameters.toon_texture_blend_factor.rgb) * model_parameters.toon_texture_blend_factor.a;
            shadow_color.a *= toon_color.a;
            var coverage = shadow_coverage(texcoord0, model_parameters.shadow_map_size);
            coverage = min(saturate(dot(input.normal, light_position) * toon_factor), coverage);
            material_color = mix(shadow_color, material_color, coverage);
        }
    }
    else if (has_toon_texture()) {
        let y = 0.5 - dot(input.normal, light_position) * 0.5;
        var toon_color = textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, y));
        toon_color.rgb *= (toon_color.rgb * model_parameters.toon_texture_blend_factor.rgb) * model_parameters.toon_texture_blend_factor.a;
        toon_color.a *= toon_color.a;
        material_color *= toon_color;
    }
    if (material_color.a < 0.0) {
        discard;
    }
    let specular_power = model_parameters.material_specular.a;
    if (specular_power > 0.0) {
        let half_vector = normalize(light_position + normalize(input.eye));
        let specular_angle = max(dot(normalize(input.normal), half_vector), 0.0);
        let spec = pow(specular_angle, specular_power);
        material_color.rgb += model_parameters.material_specular.rgb * model_parameters.light_color.rgb * spec;
    }
    return saturate_v4(material_color);
}
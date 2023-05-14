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

@vertex
fn vs_main(
    vin: VertexInput,
) -> VertexOutput {
    let position = vec4<f32>(vin.position, 1.0);
    let normal = vec4<f32>(vin.normal, 0.0);
    let color = (material_uniform.diffuse.rgb * model_uniform.light_color.rgb + material_uniform.ambient.rgb) * select(vec3<f32>(1.0, 1.0, 1.0), vin.uva1.xyz, vec3<bool>(material_uniform.enable_vertex_color.xyz));
    let sphere = normalize(model_uniform.model_view_matrix * normal).xy * 0.5 + 0.5;
    var vout: VertexOutput;
    vout.position = model_uniform.model_view_projection_matrix * position;
    vout.normal = normal.xyz;
    vout.eye = model_uniform.camera_position.xyz - (model_uniform.model_matrix * position).xyz;
    vout.texcoord0 = vin.texcoord0;
    vout.texcoord1 = sphere;
    vout.color0 = saturate(vec4<f32>(color, material_uniform.diffuse.a));
    vout.shadow0 = model_uniform.light_view_projection_matrix * position;
    // TODO: when NANOEM_IO_HAS_POINT
    return vout;
}

@fragment
fn fs_main(
    fin: FragmentInput,
) -> @location(0) vec4<f32> {
    var material_color = fin.color0;
    if (has_diffuse_texture()) {
        let texcoord0 = fin.texcoord0;
        let texel = textureSample(diffuse_texture, diffuse_texture_sampler, texcoord0);
        material_color = vec4<f32>(material_color.rgb *(texel.rgb * material_uniform.diffuse_blend_factor.rgb) * material_uniform.diffuse_blend_factor.a, material_color.a * texel.a);
    }
    if (has_sphere_texture()) {
        let texcoord1 = fin.texcoord1;
        if (is_sphere_texture_multiply()) {
            let texel = textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
            material_color = vec4<f32>(material_color.rgb * (texel.rgb * material_uniform.sphere_blend_factor.rgb) * material_uniform.sphere_blend_factor.a, material_color.a * texel.a);
        }
        else if (is_sphere_texture_as_sub_texture()) {
            material_color *= textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
        }
        else if (is_sphere_texture_additive()) {
            let texel = textureSample(sphere_map_texture, sphere_map_texture_sampler, texcoord1);
            material_color = vec4<f32>(material_color.rgb + (texel.rgb * material_uniform.sphere_blend_factor.rgb) * material_uniform.sphere_blend_factor.a, material_color.a * texel.a);
        }
    }
    let light_position = -model_uniform.light_direction.xyz;
    if (has_shadow_map_texture()) {
        let texcoord0 = fin.shadow0 / fin.shadow0.w;
        let satu_texc = saturate(texcoord0.xy);
        let toon_color = textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, 1.0));
        var coverage = shadow_coverage(texcoord0, model_uniform.shadow_map_size);
        if (satu_texc.x == texcoord0.x && satu_texc.y == texcoord0.y) {
            var shadow_color = material_color;
            shadow_color = vec4<f32>(shadow_color.rgb * (toon_color.rgb * material_uniform.toon_blend_factor.rgb) * material_uniform.toon_blend_factor.a, shadow_color.a * toon_color.a);
            coverage = min(saturate(dot(fin.normal, light_position) * toon_factor), coverage);
            material_color = mix(shadow_color, material_color, coverage);
        }
    }
    else if (has_toon_texture()) {
        let y = 0.5 - dot(fin.normal, light_position) * 0.5;
        var toon_color = textureSample(toon_texture, toon_texture_sampler, vec2<f32>(0.0, y));
        toon_color = vec4<f32>(toon_color.rgb * (toon_color.rgb * material_uniform.toon_blend_factor.rgb) * material_uniform.toon_blend_factor.a, toon_color.a * toon_color.a);
        material_color *= toon_color;
    }
    if (material_color.a < 0.0) {
        discard;
    }
    let specular_power = material_uniform.specular.a;
    if (specular_power > 0.0) {
        let half_vector = normalize(light_position + normalize(fin.eye));
        let specular_angle = max(dot(normalize(fin.normal), half_vector), 0.0);
        let spec = pow(specular_angle, specular_power);
        material_color = vec4<f32>(material_color.rgb + material_uniform.specular.rgb * model_uniform.light_color.rgb * spec, material_color.a);
    }
    return saturate(material_color);
}

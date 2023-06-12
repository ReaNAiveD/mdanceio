struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct FragmentInput {
    @location(0) color: vec4<f32>,
}

struct DebugUniform {
    view_projection_matrix: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0)
@binding(0)
var<uniform> debug_uniform: DebugUniform;

@vertex
fn vs_main(vin: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = debug_uniform.view_projection_matrix * vec4<f32>(vin.position, 1.0);
    out.color = debug_uniform.color;
    return out;
}

@fragment
fn fs_main(fin: FragmentInput) -> @location(0) vec4<f32> {
    return fin.color;
}
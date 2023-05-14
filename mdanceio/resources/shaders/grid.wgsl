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
    // builtin("PointSize") or psize when NANOEM_IO_HAS_POINT
};

struct FragmentInput {
    @location(0) color0: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord0: vec2<f32>,
    @location(3) texcoord1: vec2<f32>,
    @location(4) eye: vec3<f32>,
    @location(5) shadow0: vec4<f32>,
}

struct GridParameters {
    model_view_projection_matrix: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0)
@binding(0)
var<uniform> grid_parameters: GridParameters;

@vertex
fn vs_main(vin: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = grid_parameters.model_view_projection_matrix * vec4<f32>(vin.position, 1.0);
    out.color0 = vin.color0 * grid_parameters.color;
    return out;
}

@fragment
fn fs_main(fin: FragmentInput) -> @location(0) vec4<f32> {
    return fin.color0;
}
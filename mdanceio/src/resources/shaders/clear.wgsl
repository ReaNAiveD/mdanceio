struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec4<f32>) -> VertexOutput {
    var result: VertexOutput;
    result.position = vec4<f32>(position.x, position.y, 0.5, 0.0);
    return result;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
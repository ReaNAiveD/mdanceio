@stage(vertex)
fn vs_main(position: vec4<f32>) -> vec4<f32> {
    return position;
}

@stage(fragment)
fn fs_main() -> vec4<f32> {
    return vec4<f32>(0f32, 0f32, 0f32, 0f32);
}
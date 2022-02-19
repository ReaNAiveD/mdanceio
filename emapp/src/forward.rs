use cgmath::{Vector3, Vector4};

#[repr(C, align(16))]
pub struct LineVertexUnit {
    position: Vector3<f32>,
    color: Vector4<u8>,
}
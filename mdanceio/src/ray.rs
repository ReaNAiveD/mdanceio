use cgmath::Vector3;

pub struct Ray {
    pub from: Vector3<f32>,
    pub to: Vector3<f32>,
    pub direction: Vector3<f32>,
}
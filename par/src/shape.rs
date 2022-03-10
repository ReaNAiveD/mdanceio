#[derive(Debug, Clone)]
pub struct ShapesMesh {
    points: Vec<f32>,
    triangles: Vec<u32>,
    normals: Vec<f32>,
    tcoords: Vec<f32>,
}
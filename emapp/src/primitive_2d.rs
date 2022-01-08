use cgmath::{Vector2, Vector4};

pub trait Primitive2d {
    fn stroke_line(&mut self, from: Vector2<f32>, to: Vector2<f32>, color: Vector4<f32>, thickness: f32);
    fn stroke_rect(&mut self, rect: Vector4<f32>, color: Vector4<f32>, roundness: f32, thickness: f32);
    fn fill_rect(&mut self, rect: Vector4<f32>, color: Vector4<f32>, roundness: f32);
    fn stroke_circle(&mut self, rect: Vector4<f32>, color: Vector4<f32>, thickness: f32);
    fn fill_circle(&mut self, rect: Vector4<f32>, color: Vector4<f32>);
    fn stroke_curve(&mut self, a: Vector2<f32>, c0: Vector2<f32>, c1: Vector2<f32>, b: Vector2<f32>, color: Vector4<f32>, thickness: f32);
    fn draw_tooltop(&mut self, text: &String);
}
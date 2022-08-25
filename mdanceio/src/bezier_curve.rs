use cgmath::{Vector2, Vector4};

#[derive(Debug, Clone, PartialEq)]
pub struct BezierCurve {
    parameters: Vec<Vector2<f32>>,
    c0: Vector2<u8>,
    c1: Vector2<u8>,
    interval: u32,
}

impl BezierCurve {
    const P0: Vector2<f32> = Vector2 { x: 0f32, y: 0f32 };
    const P1: Vector2<f32> = Vector2 {
        x: 127f32,
        y: 127f32,
    };

    pub fn create(&c0: &Vector2<u8>, c1: &Vector2<u8>, interval: u32) -> Self {
        let mut curve = Self {
            parameters: vec![],
            c0: c0.clone(),
            c1: c1.clone(),
            interval,
        };
        let c0f = c0.map(|v| v as f32);
        let c1f = c1.map(|v| v as f32);
        let interval_f = interval as f32;
        for i in 0..=interval {
            let t = i as f32 / interval_f;
            let it = 1f32 - t;
            curve.parameters.push((Self::P0 * it.powi(3) + c0f * t * it.powi(2) * 3f32 + c1f * t.powi(2) * it * 3f32 + Self::P1 * t.powi(3)).zip(Self::P1, |a, b| a / b));
        }
        curve
    }

    pub fn length(&self) -> usize {
        self.parameters.len()
    }

    pub fn value(&self, value: f32) -> f32 {
        let mut nearest = &Self::P1;
        for i in 0..self.length() {
            if (nearest.x - value).abs() > (self.parameters[i].x - value).abs() {
                nearest = &self.parameters[i];
            }
        }
        nearest.y
    }

    pub fn split(&self, t: f32) -> (Self, Self) {
        let tv = t.clamp(0f32, 1f32);
        let points = vec![Self::P0, self.c0.map(|u| u as f32), self.c1.map(|u| u as f32), Self::P1];
        let mut left = vec![];
        let mut right = vec![];
        Self::split_bezier_curve(&points, tv, &mut left, &mut right);
        (Self::create(&left[1].map(|u| u as u8), &left[2].map(|u| u as u8), (self.interval as f32 * tv) as u32), Self::create(&right[1].map(|u| u as u8), &right[2].map(|u| u as u8), (self.interval as f32 * (1f32 - tv)) as u32))
    }

    pub fn to_parameters(&self) -> Vector4<u8> {
        Vector4 { x: self.c0.x, y: self.c0.y, z: self.c1.x, w: self.c1.y }
    }

    pub fn c0(&self) -> Vector2<u8> {
        self.c0
    }

    pub fn c1(&self) -> Vector2<u8> {
        self.c1
    }

    fn split_bezier_curve(points: &Vec<Vector2<f32>>, t: f32, left: &mut Vec<Vector2<f32>>, right: &mut Vec<Vector2<f32>>) {
        if points.len() == 1 {
            left.push(points[0]);
            right.push(points[0]);
        } else {
            left.push(points[0]);
            right.push(points[points.len() - 1]);
            let mut new_points = vec![];
            for i in 0..points.len() - 1 {
                new_points.push(points[i] * (1f32 - t) + points[i + 1] * t);
            }
            Self::split_bezier_curve(&new_points, t, left, right);
        }
    }
}

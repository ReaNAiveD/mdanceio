use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use cgmath::{Vector2, Vector4};

pub trait Curve {
    fn value(&self, v: f32) -> f32;
}

#[derive(Debug, Clone, PartialEq)]
pub struct BezierCurve {
    points: Vec<Vector2<f32>>,
    c0: Vector2<f32>,
    c1: Vector2<f32>,
    interval: u32,
}

impl BezierCurve {
    const P0: Vector2<f32> = Vector2 { x: 0f32, y: 0f32 };
    const P1: Vector2<f32> = Vector2 { x: 1f32, y: 1f32 };

    pub fn new(c0: Vector2<f32>, c1: Vector2<f32>, interval: u32) -> Self {
        let interval = interval.max(1u32);
        let mut points = vec![];
        let c0f = c0.map(|v| v as f32);
        let c1f = c1.map(|v| v as f32);
        let interval_f = interval as f32;
        for i in 0..=interval {
            let t = i as f32 / interval_f;
            let it = 1f32 - t;
            points.push(
                Self::P0 * it.powi(3)
                    + c0f * t * it.powi(2) * 3f32
                    + c1f * t.powi(2) * it * 3f32
                    + Self::P1 * t.powi(3),
            );
        }
        points.sort_unstable_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        Self {
            points,
            c0,
            c1,
            interval,
        }
    }

    pub fn split(&self, t: f32) -> (Self, Self) {
        let t = t.clamp(0f32, 1f32);
        let points = vec![Self::P0, self.c0, self.c1, Self::P1];
        let mut left = vec![];
        let mut right = vec![];
        Self::split_bezier_curve(&points, t, &mut left, &mut right);
        let left_interval = (self.interval as f32 * t) as u32;
        let right_interval = (self.interval as f32 * (1f32 - t)) as u32;
        (
            Self::new(left[1], left[2], left_interval),
            Self::new(right[1], right[2], right_interval),
        )
    }

    pub fn from_parameters(parameters: Vector4<u8>, interval: u32) -> Self {
        let p = parameters.map(|v| v as f32 / 127f32);
        Self::new(Vector2::new(p.x, p.y), Vector2::new(p.z, p.w), interval)
    }

    pub fn to_parameters(&self) -> Vector4<u8> {
        Vector4 {
            x: self.c0.x,
            y: self.c0.y,
            z: self.c1.x,
            w: self.c1.y,
        }
        .map(|v| (v * 127f32) as u8)
    }

    fn split_bezier_curve(
        points: &Vec<Vector2<f32>>,
        t: f32,
        left: &mut Vec<Vector2<f32>>,
        right: &mut Vec<Vector2<f32>>,
    ) {
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

impl Curve for BezierCurve {
    fn value(&self, v: f32) -> f32 {
        let mut n = (self.points[0], self.points[1]);
        for point in &self.points[2..] {
            if n.1.x > v {
                break;
            }
            n = (n.1, point.clone())
        }
        if n.0.x == n.1.x {
            n.0.y
        } else {
            n.0.y + (v - n.0.x) * (n.1.y - n.0.y) / (n.1.x - n.0.x)
        }
    }
}

pub trait BezierCurveFactory {
    fn get_or_new(&self, c0: Vector2<u8>, c1: Vector2<u8>, interval: u32) -> Arc<BezierCurve>;
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CurveCacheKey {
    c0: Vector2<u8>,
    c1: Vector2<u8>,
}

#[derive(Debug)]
pub struct BezierCurveCache(RwLock<HashMap<CurveCacheKey, Arc<BezierCurve>>>);

impl BezierCurveCache {
    pub fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }
}

impl BezierCurveFactory for BezierCurveCache {
    fn get_or_new(&self, c0: Vector2<u8>, c1: Vector2<u8>, interval: u32) -> Arc<BezierCurve> {
        let interval = interval;
        let key = CurveCacheKey { c0, c1 };
        let c0 = c0.map(|v| v as f32 / 127f32);
        let c1 = c1.map(|v| v as f32 / 127f32);
        let build_new_curve = || Arc::new(BezierCurve::new(c0, c1, interval));
        match self.0.read() {
            Ok(map) => {
                if let Some(curve) = map.get(&key) {
                    if curve.interval < interval {
                        return build_new_curve();
                    } else {
                        return curve.clone();
                    }
                }
            }
            Err(_) => return build_new_curve(),
        };
        match self.0.write() {
            Ok(mut map) => map.entry(key).or_insert(build_new_curve()).clone(),
            Err(_) => build_new_curve(),
        }
    }
}

impl Clone for BezierCurveCache {
    fn clone(&self) -> Self {
        Self::new()
    }
}

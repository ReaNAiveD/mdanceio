use cgmath::Vector3;

use crate::utils::CompareElementWise;

pub struct BoundingBox {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl BoundingBox {
    const INITIAL_MIN: [f32; 3] = [f32::MAX, f32::MAX, f32::MAX];
    const INITIAL_MAX: [f32; 3] = [f32::MIN; 3];
    pub fn new() -> Self {
        Self {
            min: Self::INITIAL_MIN.into(),
            max: Self::INITIAL_MAX.into(),
        }
    }

    pub fn reset(&mut self) {
        self.min = Self::INITIAL_MIN.into();
        self.max = Self::INITIAL_MAX.into();
    }

    pub fn set(&mut self, value: Vector3<f32>) {
        self.set_min_max(value, value);
    }

    pub fn set_min_max(&mut self, min: Vector3<f32>, max: Vector3<f32>) {
        self.min = self.min.min_element_wise(min);
        self.max = self.max.max_element_wise(max);
    }

    pub fn set_other(&mut self, other: Self) {
        self.set_min_max(other.min, other.max);
    }
}

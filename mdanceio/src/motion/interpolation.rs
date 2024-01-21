use cgmath::{Vector2, Vector3, VectorSpace};
use nanoem::motion::{MotionBoneKeyframeInterpolation, MotionCameraKeyframeInterpolation};

use crate::bezier_curve::{BezierCurveFactory, Curve};

pub fn coefficient(prev_frame_index: u32, next_frame_index: u32, frame_index: u32) -> f32 {
    let interval = next_frame_index - prev_frame_index;
    if prev_frame_index == next_frame_index {
        1f32
    } else {
        (frame_index - prev_frame_index) as f32 / (interval as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KeyframeInterpolationPoint {
    // pub bezier_control_point: Vector4<u8>,
    pub control_point1: Vector2<u8>,
    pub control_point2: Vector2<u8>,
    pub is_linear: bool,
}

impl Default for KeyframeInterpolationPoint {
    fn default() -> Self {
        Self {
            control_point1: Vector2::new(20, 20),
            control_point2: Vector2::new(107, 107),
            is_linear: true,
        }
    }
}

impl From<[u8; 4]> for KeyframeInterpolationPoint {
    fn from(v: [u8; 4]) -> Self {
        Self::new(&v)
    }
}

impl KeyframeInterpolationPoint {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn is_linear_interpolation(interpolation: &[u8; 4]) -> bool {
        interpolation[0] == interpolation[1]
            && interpolation[2] == interpolation[3]
            && interpolation[0] + interpolation[2] == interpolation[1] + interpolation[3]
    }

    pub fn new(interpolation: &[u8; 4]) -> Self {
        if Self::is_linear_interpolation(interpolation) {
            Self::default()
        } else {
            Self {
                control_point1: Vector2::new(interpolation[0], interpolation[1]),
                control_point2: Vector2::new(interpolation[2], interpolation[3]),
                is_linear: false,
            }
        }
    }

    pub fn bezier_control_point(&self) -> [u8; 4] {
        [
            self.control_point1[0],
            self.control_point1[1],
            self.control_point2[0],
            self.control_point2[1],
        ]
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        Self {
            control_point1: self
                .control_point1
                .map(|v| v as f32)
                .lerp(other.control_point1.map(|v| v as f32), amount)
                .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
            control_point2: self
                .control_point2
                .map(|v| v as f32)
                .lerp(other.control_point2.map(|v| v as f32), amount)
                .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
            is_linear: self.is_linear,
        }
    }

    pub fn curve_value(
        &self,
        interval: u32,
        amount: f32,
        bezier_factory: &dyn BezierCurveFactory,
    ) -> f32 {
        if self.is_linear {
            amount
        } else {
            let curve =
                bezier_factory.get_or_new(self.control_point1, self.control_point2, interval);
            curve.value(amount)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoneKeyframeInterpolation {
    pub translation: Vector3<KeyframeInterpolationPoint>,
    pub orientation: KeyframeInterpolationPoint,
}

impl Default for BoneKeyframeInterpolation {
    fn default() -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::default(),
                y: KeyframeInterpolationPoint::default(),
                z: KeyframeInterpolationPoint::default(),
            },
            orientation: KeyframeInterpolationPoint::default(),
        }
    }
}

impl BoneKeyframeInterpolation {
    pub fn zero() -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::zero(),
                y: KeyframeInterpolationPoint::zero(),
                z: KeyframeInterpolationPoint::zero(),
            },
            orientation: KeyframeInterpolationPoint::zero(),
        }
    }

    pub fn build(interpolation: MotionBoneKeyframeInterpolation) -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::new(&interpolation.translation_x),
                y: KeyframeInterpolationPoint::new(&interpolation.translation_y),
                z: KeyframeInterpolationPoint::new(&interpolation.translation_z),
            },
            orientation: KeyframeInterpolationPoint::new(&interpolation.orientation),
        }
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        BoneKeyframeInterpolation {
            translation: Vector3 {
                x: self.translation.x.lerp(other.translation.x, amount),
                y: self.translation.y.lerp(other.translation.y, amount),
                z: self.translation.z.lerp(other.translation.z, amount),
            },
            orientation: self.orientation.lerp(other.orientation, amount),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraKeyframeInterpolation {
    pub lookat: Vector3<KeyframeInterpolationPoint>,
    pub angle: KeyframeInterpolationPoint,
    pub fov: KeyframeInterpolationPoint,
    pub distance: KeyframeInterpolationPoint,
}

impl Default for CameraKeyframeInterpolation {
    fn default() -> Self {
        Self {
            lookat: [KeyframeInterpolationPoint::default(); 3].into(),
            angle: KeyframeInterpolationPoint::default(),
            fov: KeyframeInterpolationPoint::default(),
            distance: KeyframeInterpolationPoint::default(),
        }
    }
}

impl From<MotionCameraKeyframeInterpolation> for CameraKeyframeInterpolation {
    fn from(v: MotionCameraKeyframeInterpolation) -> Self {
        Self {
            lookat: Vector3::new(v.lookat_x.into(), v.lookat_y.into(), v.lookat_z.into()),
            angle: v.angle.into(),
            fov: v.fov.into(),
            distance: v.distance.into(),
        }
    }
}

impl CameraKeyframeInterpolation {
    pub fn zero() -> Self {
        Self {
            lookat: Vector3::new(
                KeyframeInterpolationPoint::zero(),
                KeyframeInterpolationPoint::zero(),
                KeyframeInterpolationPoint::zero(),
            ),
            angle: KeyframeInterpolationPoint::zero(),
            fov: KeyframeInterpolationPoint::zero(),
            distance: KeyframeInterpolationPoint::zero(),
        }
    }

    pub fn build(interpolation: MotionCameraKeyframeInterpolation) -> Self {
        Self {
            lookat: Vector3::new(
                KeyframeInterpolationPoint::new(&interpolation.lookat_x),
                KeyframeInterpolationPoint::new(&interpolation.lookat_y),
                KeyframeInterpolationPoint::new(&interpolation.lookat_z),
            ),
            angle: KeyframeInterpolationPoint::new(&interpolation.angle),
            fov: KeyframeInterpolationPoint::new(&interpolation.fov),
            distance: KeyframeInterpolationPoint::new(&interpolation.distance),
        }
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        Self {
            lookat: Vector3::new(
                self.lookat.x.lerp(other.lookat.x, amount),
                self.lookat.y.lerp(other.lookat.y, amount),
                self.lookat.z.lerp(other.lookat.z, amount),
            ),
            angle: self.angle.lerp(other.angle, amount),
            fov: self.fov.lerp(other.fov, amount),
            distance: self.distance.lerp(other.distance, amount),
        }
    }
}
use cgmath::{AbsDiffEq, InnerSpace, Matrix, Matrix4, Vector3, Zero};

pub trait Light {
    // TODO
    fn color(&self) -> Vector3<f32>;
    fn direction(&self) -> Vector3<f32>;
    fn get_shadow_transform(&self) -> Matrix4<f32>;
    fn ground_shadow_color(&self) -> Vector3<f32>;
    fn is_translucent_ground_shadow_enabled(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    // TODO: undo_stack
    color: Vector3<f32>,
    direction: Vector3<f32>,
    translucent: bool,
    dirty: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: Self::INITIAL_COLOR,
            direction: Self::INITIAL_DIRECTION,
            translucent: false,
            dirty: false,
        }
    }
}

impl DirectionalLight {
    pub const INITIAL_COLOR: Vector3<f32> =
        Vector3::new(154f32 / 255f32, 154f32 / 255f32, 154f32 / 255f32);
    pub const INITIAL_DIRECTION: Vector3<f32> = Vector3::new(-0.5f32, -1.0f32, 0.5f32);

    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.color = Self::INITIAL_COLOR;
        self.direction = Self::INITIAL_DIRECTION;
    }

    pub fn set_color(&mut self, value: Vector3<f32>) {
        self.color = value;
        self.dirty = true;
    }

    pub fn set_direction(&mut self, value: Vector3<f32>) {
        self.direction =
            if !value.abs_diff_eq(&Vector3::<f32>::zero(), Vector3::<f32>::default_epsilon()) {
                value
            } else {
                Self::INITIAL_DIRECTION
            };
        self.dirty = true;
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

impl Light for DirectionalLight {
    fn color(&self) -> Vector3<f32> {
        self.color
    }

    fn direction(&self) -> Vector3<f32> {
        self.direction
    }

    fn get_shadow_transform(&self) -> Matrix4<f32> {
        let position = -self.direction;
        let dot = position.dot(Vector3::unit_y());
        let m1 = (Vector3::unit_x() * dot - position.x * Vector3::unit_y()).extend(0f32);
        let m2 = (Vector3::unit_y() * dot - position.y * Vector3::unit_y()).extend(0f32);
        let m3 = (Vector3::unit_z() * dot - position.z * Vector3::unit_y()).extend(0f32);
        let m4 = Vector3::zero().extend(dot);
        Matrix4 {
            x: m1,
            y: m2,
            z: m3,
            w: m4,
        }
        .transpose()
    }

    fn ground_shadow_color(&self) -> Vector3<f32> {
        self.color()
    }

    fn is_translucent_ground_shadow_enabled(&self) -> bool {
        self.translucent
    }
}

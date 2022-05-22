use std::{cell::RefCell, collections::HashMap, rc::Rc};

use cgmath::{
    Deg, ElementWise, InnerSpace, Matrix3, Matrix4, PerspectiveFov, Quaternion, Rad, Rotation3,
    SquareMatrix, Transform, Vector3, Vector4, Zero, Vector2,
};

use crate::{bezier_curve::BezierCurve, project::Project, utils::Invert, motion::Motion};

enum TransformCoordinateType {
    Global,
    Local,
}

enum FollowingType {
    None = 1,
    Model,
    Bone,
}

pub trait Camera {
    // TODO
    fn get_view_transform(&self) -> (Matrix4<f32>, Matrix4<f32>);

    fn position(&self) -> Vector3<f32>;
    fn direction(&self) -> Vector3<f32>;
}

#[derive(Debug, Clone, Copy)]
struct MotionCameraKeyframeInterpolationSwitch {
    lookat_x: bool,
    lookat_y: bool,
    lookat_z: bool,
    angle: bool,
    fov: bool,
    distance: bool,
}

impl Default for MotionCameraKeyframeInterpolationSwitch {
    fn default() -> Self {
        Self {
            lookat_x: true,
            lookat_y: true,
            lookat_z: true,
            angle: true,
            fov: true,
            distance: true,
        }
    }
}

pub struct PerspectiveCamera {
    // TODO: undo_stack_t *m_undoStack;
    bezier_curves_data: HashMap<u64, Box<BezierCurve>>,
    keyframe_bezier_curves:
        HashMap<Rc<RefCell<nanoem::motion::MotionCameraKeyframe>>, Box<BezierCurve>>,
    outside_parent: (String, String),
    transform_coordinate_type: TransformCoordinateType,
    view_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
    position: Vector3<f32>,
    direction: Vector3<f32>,
    look_at: Vector3<f32>,
    angle: Vector3<f32>,
    distance: f32,
    fov: (i32, f32),
    bezier_control_points: nanoem::motion::MotionCameraKeyframeInterpolation,
    automatic_bezier_control_point: Vector4<u8>,
    following_type: FollowingType,
    is_linear_interpolation: MotionCameraKeyframeInterpolationSwitch,
    perspective: bool,
    locked: bool,
    dirty: bool,
}

impl PerspectiveCamera {
    pub const ANGLE_SCALE_FACTOR: Vector3<f32> = Vector3::new(-1f32, 1f32, 1f32);
    pub const INITIAL_LOOK_AT: Vector3<f32> = Vector3::new(0f32, 10f32, 0f32);
    pub const INITIAL_DISTANCE: f32 = 45f32;
    pub const INITIAL_FOV_RADIAN: f32 =
        (Self::INITIAL_FOV as f32) * 0.01745329251994329576923690768489f32;
    pub const MAX_FOV: i32 = 135;
    pub const MIN_FOV: i32 = 1;
    pub const INITIAL_FOV: i32 = 30;
    pub const DEFAULT_BEZIER_CONTROL_POINT: Vector4<u8> = Vector4::new(20, 20, 107, 107);
    pub const DEFAULT_AUTOMATIC_BEZIER_CONTROL_POINT: Vector4<u8> = Vector4::new(64, 0, 64, 127);

    pub fn new() -> Self {
        Self {
            bezier_curves_data: HashMap::new(),
            keyframe_bezier_curves: HashMap::new(),
            outside_parent: (String::default(), String::default()),
            transform_coordinate_type: TransformCoordinateType::Local,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
            position: Vector3::zero(),
            direction: Vector3::unit_z(),
            look_at: Self::INITIAL_LOOK_AT,
            angle: Vector3::zero(),
            distance: Self::INITIAL_DISTANCE,
            fov: (Self::INITIAL_FOV, Self::INITIAL_FOV_RADIAN),
            bezier_control_points: nanoem::motion::MotionCameraKeyframeInterpolation {
                lookat_x: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                lookat_y: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                lookat_z: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                angle: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                fov: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                distance: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            },
            automatic_bezier_control_point: Self::DEFAULT_AUTOMATIC_BEZIER_CONTROL_POINT,
            following_type: FollowingType::None,
            is_linear_interpolation: MotionCameraKeyframeInterpolationSwitch::default(),
            perspective: true,
            locked: false,
            dirty: false,
        }
    }

    pub fn reset(&mut self) {
        self.perspective = true;
        self.angle = Vector3::zero();
        self.look_at = Self::INITIAL_LOOK_AT;
        self.distance = Self::INITIAL_DISTANCE;
        self.fov = (Self::INITIAL_FOV, Self::INITIAL_FOV_RADIAN);
        self.set_dirty(true);
        self.bezier_control_points = nanoem::motion::MotionCameraKeyframeInterpolation {
            lookat_x: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            lookat_y: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            lookat_z: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            angle: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            fov: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            distance: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
        };
        self.is_linear_interpolation = MotionCameraKeyframeInterpolationSwitch::default();
    }

    pub fn update(&mut self, viewport_image_size: Vector2<u16>) {
        let look_at = self.bound_look_at();
        let angle = self.angle.mul_element_wise(Self::ANGLE_SCALE_FACTOR);
        let x = Quaternion::from_angle_x(Rad(angle.x));
        let y = Quaternion::from_angle_y(Rad(angle.y));
        let z = Quaternion::from_angle_z(Rad(angle.z));
        let view_orientation = Matrix3::from(z * x * y);
        self.view_matrix =
            Matrix4::from(view_orientation) * Matrix4::from_translation(-self.look_at);
        self.view_matrix[3] += Vector4::new(0.0f32, 0.0f32, self.distance, 0.0f32);
        let position = self.view_matrix.affine_invert().unwrap()[3].truncate();
        if self.distance > 0.0 {
            self.direction = (look_at - position).normalize();
        } else if self.distance < 0.0 {
            self.direction = (position - look_at).normalize();
        }
        self.position = position;
        if self.perspective {
            self.fov.1 = (Rad::from(Deg(1.0f32)).0).max(self.fov.1);
            self.projection_matrix = PerspectiveFov {
                fovy: Rad(self.fov.1),
                aspect: self.aspect_ratio(),
                near: 0.5,
                far: f32::INFINITY,
            }.into();
        } else {
            let viewport_image_size: Vector2<f32> = viewport_image_size.cast().unwrap();
            let inverse_distance = 1.0 / self.distance;
            let mut projection_matrix: Matrix4<f32> = Matrix4::identity();
            projection_matrix[0][0] = 2.0f32 * (viewport_image_size.y / viewport_image_size.x).max(1.0) * inverse_distance;
            projection_matrix[1][1] = 2.0f32 * (viewport_image_size.x / viewport_image_size.y).max(1.0) * inverse_distance;
            projection_matrix[2][2] = 2.0f32 / (self.zfar() - 0.5f32);
            self.projection_matrix = projection_matrix;
        }
    }

    pub fn synchronize_parameters(&mut self, motion: &Motion, frame_index: u32) {
        const distance_factor: f32 = -1.0f32;
        let outside_parent = ("".to_owned(), "".to_owned());
        if let Some(keyframe) = motion.find_camera_keyframe(frame_index) {

        }
    }

    pub fn bound_look_at(&self) -> Vector3<f32> {
        todo!()
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }

    pub fn aspect_ratio(&self) -> f32 {
        todo!()
    }

    pub fn zfar(&self) -> f32 {
        todo!()
    }
}

impl Camera for PerspectiveCamera {
    fn get_view_transform(&self) -> (Matrix4<f32>, Matrix4<f32>) {
        (self.view_matrix, self.projection_matrix)
    }

    fn position(&self) -> Vector3<f32> {
        self.position
    }

    fn direction(&self) -> Vector3<f32> {
        self.direction
    }
}

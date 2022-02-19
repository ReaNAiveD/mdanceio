use std::{collections::HashMap, rc::Rc, cell::RefCell};

use cgmath::{Matrix4, Vector3, Vector4, Zero};

use crate::{bezier_curve::BezierCurve, project::Project};

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
    keyframe_bezier_curves: HashMap<Rc<RefCell<nanoem::motion::MotionCameraKeyframe>>, Box<BezierCurve>>,
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
    pub const INITIAL_FOV_RADIAN: f32 = (Self::INITIAL_FOV as f32) * 0.01745329251994329576923690768489f32;
    pub const MAX_FOV: i32 = 135;
    pub const MIN_FOV: i32 = 1;
    pub const INITIAL_FOV: i32 = 30;
    pub const DEFAULT_BEZIER_CONTROL_POINT: Vector4<u8> = Vector4::new(20, 20, 107, 107);
    pub const DEFAULT_AUTOMATIC_BEZIER_CONTROL_POINT: Vector4<u8> = Vector4::new(64, 0, 64, 127);

    pub fn new(project: &Project) -> Self {
        Self {
            bezier_curves_data: HashMap::new(),
            keyframe_bezier_curves: HashMap::new(),
            outside_parent: (String::default(), String::default()),
            transform_coordinate_type: TransformCoordinateType::Local,
            view_matrix: todo!(),
            projection_matrix: todo!(),
            position: Vector3::zero(),
            direction: Vector3::unit_z(),
            look_at: Self::INITIAL_LOOK_AT,
            angle: todo!(),
            distance: Self::INITIAL_DISTANCE,
            fov: (Self::INITIAL_FOV, Self::INITIAL_FOV_RADIAN),
            bezier_control_points: nanoem::motion::MotionCameraKeyframeInterpolation{
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
        self.bezier_control_points = nanoem::motion::MotionCameraKeyframeInterpolation{
            lookat_x: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            lookat_y: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            lookat_z: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            angle: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            fov: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            distance: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
        };
        self.is_linear_interpolation = MotionCameraKeyframeInterpolationSwitch::default();
    }

    pub fn update(&mut self) {
        let look_at = self.bound_look_at();
    }

    pub fn bound_look_at(&self) {}

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }
}
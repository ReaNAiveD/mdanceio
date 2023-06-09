use std::{cell::RefCell, collections::HashMap};

use cgmath::{
    AbsDiffEq, Deg, ElementWise, InnerSpace, Matrix3, Matrix4, MetricSpace, Quaternion, Rad,
    Rotation3, SquareMatrix, Vector1, Vector2, Vector3, Vector4, VectorSpace, Zero,
};

use crate::{
    bezier_curve::{BezierCurve, Curve},
    model::{Bone, Model},
    motion::{KeyframeInterpolationPoint, Motion},
    project::Project,
    ray::Ray,
    utils::{f128_to_vec3, infinite_perspective, intersect_ray_plane, project, un_project, Invert},
};

use nanoem::motion::{MotionCameraKeyframe, MotionTrackBundle};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TransformCoordinateType {
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum FollowingType {
    None = 1,
    Model,
    Bone,
}

fn bone_pos(bone: &Bone) -> Vector3<f32> {
    (bone.matrices.skinning_transform * f128_to_vec3(bone.origin.origin).extend(1f32)).truncate()
}

pub trait Camera {
    fn get_view_transform(&self) -> (Matrix4<f32>, Matrix4<f32>);

    fn position(&self) -> Vector3<f32>;
    fn direction(&self) -> Vector3<f32>;
    fn fov(&self) -> i32;

    fn is_locked(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CurveCacheKey {
    next: [u8; 4],
    interval: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CameraKeyframeInterpolation {
    lookat_x: KeyframeInterpolationPoint,
    lookat_y: KeyframeInterpolationPoint,
    lookat_z: KeyframeInterpolationPoint,
    angle: KeyframeInterpolationPoint,
    fov: KeyframeInterpolationPoint,
    distance: KeyframeInterpolationPoint,
}

#[derive(Debug, Clone)]
pub struct PerspectiveCamera {
    // TODO: undo_stack_t *m_undoStack;
    bezier_curves_data: RefCell<HashMap<CurveCacheKey, Box<BezierCurve>>>,
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
    pub interpolation: CameraKeyframeInterpolation,
    automatic_bezier_control_point: Vector4<u8>,
    following_type: FollowingType,
    perspective: bool,
    locked: bool,
    dirty: bool,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            bezier_curves_data: RefCell::new(HashMap::new()),
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
            interpolation: CameraKeyframeInterpolation::default(),
            automatic_bezier_control_point: Self::DEFAULT_AUTOMATIC_BEZIER_CONTROL_POINT,
            following_type: FollowingType::None,
            perspective: true,
            locked: false,
            dirty: false,
        }
    }
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
        Self::default()
    }

    pub fn reset(&mut self) {
        self.perspective = true;
        self.angle = Vector3::zero();
        self.look_at = Self::INITIAL_LOOK_AT;
        self.distance = Self::INITIAL_DISTANCE;
        self.fov = (Self::INITIAL_FOV, Self::INITIAL_FOV_RADIAN);
        self.set_dirty(true);
        self.interpolation = CameraKeyframeInterpolation::default();
    }

    pub fn update(&mut self, viewport_image_size: Vector2<u32>, bound_look_at: Vector3<f32>) {
        let angle = self.angle.mul_element_wise(Self::ANGLE_SCALE_FACTOR);
        let x = Quaternion::from_angle_x(Rad(angle.x));
        let y = Quaternion::from_angle_y(Rad(angle.y));
        let z = Quaternion::from_angle_z(Rad(angle.z));
        let view_orientation = Matrix3::from(z * x * y);
        self.view_matrix =
            Matrix4::from(view_orientation) * Matrix4::from_translation(-bound_look_at);
        self.view_matrix[3] += Vector4::new(0.0f32, 0.0f32, self.distance, 0.0f32);
        let position = self.view_matrix.affine_invert().unwrap()[3].truncate();
        if self.distance > 0.0 {
            self.direction = (bound_look_at - position).normalize();
        } else if self.distance < 0.0 {
            self.direction = (position - bound_look_at).normalize();
        }
        self.position = position;
        let viewport_image_size: Vector2<f32> = viewport_image_size
            .cast()
            .unwrap()
            .map(|v: f32| v.max(1f32));
        if self.perspective {
            self.fov.1 = (Rad::from(Deg(1.0f32)).0).max(self.fov.1);
            self.projection_matrix = infinite_perspective(
                Rad(self.fov.1),
                viewport_image_size.x / viewport_image_size.y,
                0.5,
            );
        } else {
            let inverse_distance = 1.0 / self.distance;
            let mut projection_matrix: Matrix4<f32> = Matrix4::identity();
            projection_matrix[0][0] = 2.0f32
                * (viewport_image_size.y / viewport_image_size.x).max(1.0)
                * inverse_distance;
            projection_matrix[1][1] = 2.0f32
                * (viewport_image_size.x / viewport_image_size.y).max(1.0)
                * inverse_distance;
            projection_matrix[2][2] = 2.0f32 / (self.zfar() - 0.5f32);
            self.projection_matrix = projection_matrix;
        }
    }
    fn synchronize_outside_parent(
        &mut self,
        keyframe: &MotionCameraKeyframe,
        project: &Project,
        global_track_bundle: &MotionTrackBundle<()>,
    ) {
        if let Some(op) = &keyframe.outside_parent {
            if let Some(model) = global_track_bundle
                .resolve_id(op.global_model_track_index)
                .and_then(|name| project.find_model_by_name(name))
            {
                if let Some(bone_name) = global_track_bundle.resolve_id(op.global_bone_track_index)
                {
                    self.outside_parent = (model.get_name().to_owned(), bone_name.clone());
                }
            }
        }
    }

    pub fn un_projected(&self, value: &Vector3<f32>, viewport_size: Vector2<u32>) -> Vector3<f32> {
        let viewport = Vector4::new(0f32, 0f32, viewport_size.x as f32, viewport_size.y as f32);
        un_project(value, &self.view_matrix, &self.projection_matrix, &viewport).unwrap()
    }

    pub fn to_device_screen_coordinate_in_viewport(
        &self,
        value: &Vector3<f32>,
        device_scale_uniformed_viewport_layout_rect: &Vector4<u32>,
        device_scale_uniformed_viewport_image_size: &Vector2<f32>,
    ) -> Vector2<i32> {
        let viewport_rect = Vector4::new(
            0f32,
            0f32,
            device_scale_uniformed_viewport_image_size.x.into(),
            device_scale_uniformed_viewport_image_size.y.into(),
        );
        let x = (device_scale_uniformed_viewport_layout_rect.z
            - device_scale_uniformed_viewport_layout_rect.x) as f32
            * 0.5f32;
        let y = (device_scale_uniformed_viewport_layout_rect.w
            - device_scale_uniformed_viewport_layout_rect.y) as f32
            * 0.5f32;
        let coordinate = project(
            value,
            &self.view_matrix,
            &self.projection_matrix,
            &viewport_rect,
        );
        return Vector2::new(
            (x + coordinate.x) as i32,
            (device_scale_uniformed_viewport_layout_rect.w as f32 - coordinate.y - y) as i32,
        );
    }

    pub fn to_device_screen_coordinate_in_window(
        &self,
        value: &Vector3<f32>,
        device_scale_uniformed_viewport_layout_rect: &Vector4<u32>,
        device_scale_uniformed_viewport_image_size: &Vector2<f32>,
    ) -> Vector2<i32> {
        let layout_rect = Vector2::new(
            device_scale_uniformed_viewport_layout_rect.x as i32,
            device_scale_uniformed_viewport_layout_rect.y as i32,
        );
        layout_rect
            + self.to_device_screen_coordinate_in_viewport(
                value,
                device_scale_uniformed_viewport_layout_rect,
                device_scale_uniformed_viewport_image_size,
            )
    }

    // TODO: padding, rect, size used in project can be extracted
    pub fn create_ray(&self, value: Vector2<i32>, viewport_size: Vector2<u32>) -> Ray {
        let from = self.un_projected(&value.extend(0).cast().unwrap(), viewport_size);
        let to = self.un_projected(
            &value.cast().unwrap().extend(1f32 - f32::EPSILON),
            viewport_size,
        );
        Ray {
            from,
            to,
            direction: if to.distance2(from) > 0.0f32 {
                (to - from).normalize()
            } else {
                Vector3::unit_z()
            },
        }
    }

    pub fn cast_ray(
        &self,
        position: Vector2<i32>,
        viewport_size: Vector2<u32>,
    ) -> Option<Vector3<f32>> {
        let ray = self.create_ray(position, viewport_size);
        intersect_ray_plane(
            &ray.from,
            &ray.direction,
            &Vector3::zero(),
            &-self.direction(),
        )
        .map(|distance| ray.from + ray.direction * distance)
    }

    pub fn look_at(&self, active_model: Option<&Model>) -> Vector3<f32> {
        match self.following_type {
            FollowingType::None => self.look_at,
            FollowingType::Model => match active_model {
                Some(model) => {
                    if let Some(bone) = model.find_bone(Bone::NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE)
                    {
                        self.look_at + bone_pos(bone)
                    } else if let Some(bone) = model
                        .find_bone(Bone::NAME_CENTER_OFFSET_IN_JAPANESE)
                        .and_then(|bone| model.parent_bone(bone))
                    {
                        self.look_at + bone_pos(bone)
                    } else if let Some(bone) = model.bones().get(0) {
                        self.look_at + bone_pos(bone)
                    } else {
                        self.look_at
                    }
                }
                None => self.look_at,
            },
            FollowingType::Bone => match active_model.and_then(|model| model.active_bone()) {
                Some(bone) => bone_pos(bone),
                None => Vector3::zero(),
            },
        }
    }

    pub fn bound_look_at(&self, project: &Project) -> Vector3<f32> {
        (match project.resolve_bone((&self.outside_parent.0, &self.outside_parent.1)) {
            Some(bone) => bone.world_translation(),
            None => Vector3::zero(),
        }) + self.look_at(project.active_model())
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }

    pub fn set_look_at(&mut self, value: Vector3<f32>) {
        if !self.locked && !value.abs_diff_eq(&self.look_at, Vector3::<f32>::default_epsilon()) {
            self.look_at = value;
            self.dirty = true;
        }
    }

    pub fn angle(&self) -> Vector3<f32> {
        self.angle
    }

    pub fn set_angle(&mut self, value: Vector3<f32>) {
        if !self.locked && !value.abs_diff_eq(&self.angle, Vector3::<f32>::default_epsilon()) {
            self.angle = value;
            self.dirty = true;
        }
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, value: f32) {
        if !self.locked && !value.abs_diff_eq(&self.distance, f32::default_epsilon()) {
            self.distance = value;
            self.dirty = true;
        }
    }

    pub fn set_fov(&mut self, value: i32) {
        if !self.locked && value != self.fov.0 {
            self.fov.0 = value;
            self.fov.1 = Rad::from(Deg(value as f32)).0;
            self.dirty = true
        }
    }

    pub fn fov(&self) -> i32 {
        self.fov.0
    }

    pub fn fov_radians(&self) -> f32 {
        self.fov.1
    }

    pub fn set_fov_radians(&mut self, value: f32) {
        if !self.locked && !value.abs_diff_eq(&self.fov.1, f32::default_epsilon()) {
            self.fov.0 = Deg::from(Rad(value)).0 as i32;
            self.fov.1 = value;
            self.dirty = true;
        }
    }

    pub fn is_perspective(&self) -> bool {
        self.perspective
    }

    pub fn set_perspective(&mut self, value: bool) {
        if !self.locked && value != self.perspective {
            self.perspective = value;
            self.dirty = true;
        }
    }

    pub fn set_transform_coordinate_type(&mut self, value: TransformCoordinateType) {
        self.transform_coordinate_type = value;
    }

    pub fn toggle_transform_coordinate_type(&mut self) {
        self.set_transform_coordinate_type(match self.transform_coordinate_type {
            TransformCoordinateType::Global => TransformCoordinateType::Local,
            TransformCoordinateType::Local => TransformCoordinateType::Global,
        })
    }

    pub fn set_following_type(
        &mut self,
        value: FollowingType,
        viewport_image_size: Vector2<u32>,
        bound_look_at: Vector3<f32>,
    ) {
        self.following_type = value;
        self.update(viewport_image_size, bound_look_at);
    }

    pub fn zfar(&self) -> f32 {
        10000.0f32
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn update_view_projection(&mut self, view: Matrix4<f32>, projection: Matrix4<f32>) {
        self.view_matrix = view;
        self.projection_matrix = projection;
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

    fn fov(&self) -> i32 {
        self.fov.0
    }

    fn is_locked(&self) -> bool {
        self.locked
    }
}

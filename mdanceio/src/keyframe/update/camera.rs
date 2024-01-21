use cgmath::{Vector3, Vector4};
use nanoem::motion::{MotionCameraKeyframe, MotionTrack};

use crate::{camera::PerspectiveCamera, model::Model, motion::interpolation::CameraKeyframeInterpolation};

use super::updater::{KeyframeUpdater, Updatable};

pub struct CameraKeyframeBezierControlPointParameter {
    pub look_at: Vector3<Vector4<u8>>,
    pub angle: Vector4<u8>,
    pub fov: Vector4<u8>,
    pub distance: Vector4<u8>,
}

impl From<CameraKeyframeInterpolation> for CameraKeyframeBezierControlPointParameter {
    fn from(interpolation: CameraKeyframeInterpolation) -> Self {
        Self {
            look_at: interpolation
                .lookat
                .map(|point| point.bezier_control_point().into()),
            angle: interpolation.angle.bezier_control_point().into(),
            fov: interpolation.fov.bezier_control_point().into(),
            distance: interpolation.distance.bezier_control_point().into(),
        }
    }
}

pub struct CameraKeyframeLookAtBezierControlPointParameter(pub Vector3<Vector4<u8>>);

pub struct CameraKeyframeState {
    pub look_at: Vector4<f32>,
    pub angle: Vector4<f32>,
    pub distance: f32,
    pub fov: f32,
    pub bezier_param: CameraKeyframeBezierControlPointParameter,
    pub stage_index: u32,
    pub perspective: bool,
}

impl CameraKeyframeState {
    pub fn from_camera(camera: &PerspectiveCamera, active_model: Option<&Model>) -> Self {
        Self {
            look_at: camera.look_at(active_model).extend(1f32),
            angle: camera.angle().extend(0f32),
            distance: camera.distance(),
            fov: camera.fov_radians(),
            bezier_param: CameraKeyframeBezierControlPointParameter {
                look_at: Vector3 {
                    x: camera.bezier_control_point().into(),
                    y: camera.look_at_y.bezier_control_point().into(),
                    z: camera.look_at_z.bezier_control_point().into(),
                },
                angle: camera.angle.bezier_control_point().into(),
                fov: camera.fov.bezier_control_point().into(),
                distance: camera.distance.bezier_control_point().into(),
            },
            stage_index: 0,
            perspective: camera.is_perspective(),
        }
    }

    pub fn from_keyframe(keyframe: &MotionCameraKeyframe) -> Self {
        Self {
            look_at: keyframe.look_at.into(),
            angle: keyframe.angle.into(),
            distance: keyframe.distance,
            fov: keyframe.fov as f32,
            bezier_param: CameraKeyframeBezierControlPointParameter {
                look_at: Vector3 {
                    x: keyframe.interpolation.lookat_x.into(),
                    y: keyframe.interpolation.lookat_y.into(),
                    z: keyframe.interpolation.lookat_z.into(),
                },
                angle: keyframe.interpolation.angle.into(),
                fov: keyframe.interpolation.fov.into(),
                distance: keyframe.interpolation.distance.into(),
            },
            stage_index: keyframe.stage_index,
            perspective: keyframe.is_perspective_view,
        }
    }
}

pub struct CameraKeyframeOverrideInterpolation {
    pub target_frame_index: u32,
    pub look_at_params: (
        CameraKeyframeLookAtBezierControlPointParameter,
        CameraKeyframeLookAtBezierControlPointParameter,
    ),
}

pub struct CameraKeyframeUpdater {
    pub added_state: CameraKeyframeState,
    pub removed_state: Option<CameraKeyframeState>,
    pub bezier_curve_override: Option<CameraKeyframeOverrideInterpolation>,
    pub frame_index: u32,
}

impl KeyframeUpdater for CameraKeyframeUpdater {
    fn updated(&self) -> bool {
        self.removed_state.is_some()
    }

    fn selected(&self) -> bool {
        self.removed_state.is_none()
    }
}

impl Updatable for MotionTrack<MotionCameraKeyframe> {
    type Object = MotionCameraKeyframe;
    type ObjectUpdater = CameraKeyframeUpdater;

    fn apply_add(&mut self, updater: &mut Self::ObjectUpdater, object: Option<&mut Self::Object>) {}

    fn apply_remove(
        &mut self,
        updater: &mut Self::ObjectUpdater,
        object: Option<&mut Self::Object>,
    ) {
    }
}

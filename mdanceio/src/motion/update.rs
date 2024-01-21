use std::collections::HashMap;

use cgmath::{Quaternion, Vector3, Vector4};
use nanoem::motion::{
    MotionBoneKeyframe, MotionCameraKeyframe, MotionKeyframeBase, MotionMorphKeyframe,
    MotionOutsideParent, MotionTrack, MotionLightKeyframe, MotionSelfShadowKeyframe,
};

use crate::{utils::{f128_to_quat, f128_to_vec3, quat_to_f128, f128_to_vec4}, shadow_camera::CoverageMode};

use super::interpolation::{BoneKeyframeInterpolation, CameraKeyframeInterpolation};

pub trait Update {
    type Updater;

    fn update(&mut self, updater: &Option<Self::Updater>, frame_idx: u32) -> Option<Self::Updater>;
}

// Update for Track<MotionModelKeyframe> would be implemented in the future

// The following implementation is not generic, but it's not a problem for now.

pub struct BoneUpdater {
    pub translation: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub interpolation: BoneKeyframeInterpolation,
    pub stage_index: u32,
    pub enable_physics_simulation: bool,
}

impl From<MotionBoneKeyframe> for BoneUpdater {
    fn from(keyframe: MotionBoneKeyframe) -> Self {
        Self {
            translation: f128_to_vec3(keyframe.translation),
            orientation: f128_to_quat(keyframe.orientation),
            interpolation: BoneKeyframeInterpolation::build(keyframe.interpolation),
            stage_index: keyframe.stage_index,
            enable_physics_simulation: keyframe.is_physics_simulation_enabled,
        }
    }
}

impl BoneUpdater {
    pub fn into_keyframe(self, frame_index: u32) -> MotionBoneKeyframe {
        MotionBoneKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            translation: self.translation.extend(0f32).into(),
            orientation: quat_to_f128(self.orientation),
            interpolation: nanoem::motion::MotionBoneKeyframeInterpolation {
                translation_x: self.interpolation.translation.x.bezier_control_point(),
                translation_y: self.interpolation.translation.y.bezier_control_point(),
                translation_z: self.interpolation.translation.z.bezier_control_point(),
                orientation: self.interpolation.orientation.bezier_control_point(),
            },
            stage_index: self.stage_index,
            is_physics_simulation_enabled: self.enable_physics_simulation,
        }
    }
}

impl Update for MotionTrack<MotionBoneKeyframe> {
    type Updater = BoneUpdater;

    fn update(
        &mut self,
        updater: &Option<Self::Updater>,
        frame_index: u32,
    ) -> Option<Self::Updater> {
        if let Some(updater) = updater {
            let keyframe = updater.into_keyframe(frame_index);
            if let Some(old_keyframe) = self.insert_keyframe(keyframe) {
                Some(old_keyframe.into())
            } else {
                None
            }
        } else {
            let keyframe = self.remove_keyframe(frame_index);
            if let Some(keyframe) = keyframe {
                Some(keyframe.into())
            } else {
                None
            }
        }
    }
}

impl Update for MotionTrack<MotionMorphKeyframe> {
    type Updater = f32;

    fn update(
        &mut self,
        updater: &Option<Self::Updater>,
        frame_index: u32,
    ) -> Option<Self::Updater> {
        if let Some(updater) = updater {
            let keyframe = MotionMorphKeyframe {
                base: MotionKeyframeBase {
                    frame_index: frame_index,
                    annotations: HashMap::new(),
                },
                weight: *updater,
            };
            if let Some(old_keyframe) = self.insert_keyframe(keyframe) {
                Some(old_keyframe.weight)
            } else {
                None
            }
        } else {
            let keyframe = self.remove_keyframe(frame_index);
            if let Some(keyframe) = keyframe {
                Some(keyframe.weight)
            } else {
                None
            }
        }
    }
}

pub struct CameraUpdater {
    pub look_at: Vector3<f32>,
    pub angle: Vector3<f32>,
    pub distance: f32,
    pub fov: f32,
    pub interpolation: CameraKeyframeInterpolation,
    pub stage_index: u32,
    pub perspective: bool,
    pub outside_parent: Option<MotionOutsideParent>,
}

impl From<MotionCameraKeyframe> for CameraUpdater {
    fn from(keyframe: MotionCameraKeyframe) -> Self {
        Self {
            look_at: f128_to_vec3(keyframe.look_at),
            angle: f128_to_vec3(keyframe.angle),
            distance: keyframe.distance,
            fov: f32::to_radians(keyframe.fov as f32),
            interpolation: CameraKeyframeInterpolation::build(keyframe.interpolation),
            stage_index: keyframe.stage_index,
            perspective: keyframe.is_perspective_view,
            outside_parent: keyframe.outside_parent,
        }
    }
}

impl CameraUpdater {
    pub fn into_keyframe(
        self,
        frame_index: u32,
    ) -> MotionCameraKeyframe {
        MotionCameraKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            look_at: self.look_at.extend(0f32).into(),
            angle: self.angle.extend(0f32).into(),
            distance: self.distance,
            fov: self.fov.to_degrees() as i32,
            interpolation: nanoem::motion::MotionCameraKeyframeInterpolation {
                lookat_x: self.interpolation.lookat.x.bezier_control_point(),
                lookat_y: self.interpolation.lookat.y.bezier_control_point(),
                lookat_z: self.interpolation.lookat.z.bezier_control_point(),
                angle: self.interpolation.angle.bezier_control_point(),
                fov: self.interpolation.fov.bezier_control_point(),
                distance: self.interpolation.distance.bezier_control_point(),
            },
            stage_index: self.stage_index,
            is_perspective_view: self.perspective,
            outside_parent: self.outside_parent,
        }
    }
}

impl Update for MotionTrack<MotionCameraKeyframe> {
    type Updater = CameraUpdater;

    fn update(
        &mut self,
        updater: &Option<Self::Updater>,
        frame_index: u32,
    ) -> Option<Self::Updater> {
        if let Some(updater) = updater {
            let keyframe = updater.into_keyframe(frame_index);
            if let Some(old_keyframe) = self.insert_keyframe(keyframe) {
                Some(old_keyframe.into())
            } else {
                None
            }
        } else {
            let keyframe = self.remove_keyframe(frame_index);
            if let Some(keyframe) = keyframe {
                Some(keyframe.into())
            } else {
                None
            }
        }
    }
}

pub struct LightUpdater {
    pub color: Vector4<f32>,
    pub direction: Vector4<f32>,
}

impl From<MotionLightKeyframe> for LightUpdater {
    fn from(keyframe: MotionLightKeyframe) -> Self {
        Self {
            color: f128_to_vec4(keyframe.color),
            direction: f128_to_vec4(keyframe.direction),
        }
    }
}

impl LightUpdater {
    pub fn into_keyframe(self, frame_index: u32) -> MotionLightKeyframe {
        MotionLightKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            color: self.color.into(),
            direction: self.direction.into(),
        }
    }
}

impl Update for MotionTrack<MotionLightKeyframe> {
    type Updater = LightUpdater;

    fn update(
        &mut self,
        updater: &Option<Self::Updater>,
        frame_index: u32,
    ) -> Option<Self::Updater> {
        if let Some(updater) = updater {
            let keyframe = updater.into_keyframe(frame_index);
            if let Some(old_keyframe) = self.insert_keyframe(keyframe) {
                Some(old_keyframe.into())
            } else {
                None
            }
        } else {
            let keyframe = self.remove_keyframe(frame_index);
            if let Some(keyframe) = keyframe {
                Some(keyframe.into())
            } else {
                None
            }
        }
    }
}

pub struct SelfShadowUpdater {
    pub distance: f32,
    pub mode: CoverageMode,
}

impl From<MotionSelfShadowKeyframe> for SelfShadowUpdater {
    fn from(keyframe: MotionSelfShadowKeyframe) -> Self {
        Self {
            distance: keyframe.distance,
            mode: CoverageMode::from(keyframe.mode.clamp(0, 3) as u32),
        }
    }
}

impl SelfShadowUpdater {
    pub fn into_keyframe(self, frame_index: u32) -> MotionSelfShadowKeyframe {
        MotionSelfShadowKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            distance: self.distance,
            mode: u32::from(self.mode) as i32,
        }
    }
}

impl Update for MotionTrack<MotionSelfShadowKeyframe> {
    type Updater = SelfShadowUpdater;

    fn update(
        &mut self,
        updater: &Option<Self::Updater>,
        frame_index: u32,
    ) -> Option<Self::Updater> {
        if let Some(updater) = updater {
            let keyframe = updater.into_keyframe(frame_index);
            if let Some(old_keyframe) = self.insert_keyframe(keyframe) {
                Some(old_keyframe.into())
            } else {
                None
            }
        } else {
            let keyframe = self.remove_keyframe(frame_index);
            if let Some(keyframe) = keyframe {
                Some(keyframe.into())
            } else {
                None
            }
        }
    }
}



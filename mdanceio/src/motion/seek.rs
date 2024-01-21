use cgmath::{Deg, One, Quaternion, Rad, Vector3, VectorSpace, Zero};
use nanoem::motion::{
    MotionBoneKeyframe, MotionCameraKeyframe, MotionLightKeyframe, MotionMorphKeyframe,
    MotionSelfShadowKeyframe, MotionTrack,
};

use crate::{
    bezier_curve::BezierCurveFactory,
    shadow_camera::CoverageMode,
    utils::{f128_to_quat, f128_to_vec3, lerp_element_wise, lerp_f32, lerp_rad},
};

use super::interpolation::{
    BoneKeyframeInterpolation, CameraKeyframeInterpolation, KeyframeInterpolationPoint,
};

pub trait Seek {
    type Frame;

    fn find(&self, frame_index: u32) -> Option<Self::Frame>;

    fn seek(&self, frame_index: u32, curve_factory: &dyn BezierCurveFactory) -> Self::Frame;

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame;
}

// Seek for MotionTrack<MotionBoneKeyframe> would be implemented in future

// #[derive(Debug, Clone)]
// pub struct ModelFrame {
//     pub visible: bool,
//     pub constraint_states: HashMap<BoneIndex, bool>,
//     pub effect_parameters: HashMap<i32, MotionEffectParameterValue>,
//     pub outside_parents: Vec<MotionOutsideParent>,
//     pub has_edge_option: bool,
//     pub edge_scale_factor: f32,
//     pub edge_color: [f32; 4],
//     pub is_add_blending_enabled: bool,
//     pub is_physics_simulation_enabled: bool,
// }

// impl Default for ModelFrame {
//     fn default() -> Self {
//         Self {
//             visible: true,
//             constraint_states: HashMap::new(),
//             effect_parameters: HashMap::new(),
//             outside_parents: Vec::new(),
//             has_edge_option: false,
//             edge_scale_factor: 1f32,
//             edge_color: [0f32; 4],
//             is_add_blending_enabled: false,
//             is_physics_simulation_enabled: false,
//         }
//     }
// }

// impl From<&MotionModelKeyframe> for ModelFrame {
//     fn from(v: &MotionModelKeyframe) -> Self {
//         Self {
//             visible: v.visible,
//             constraint_states: v
//                 .constraint_states
//                 .iter()
//                 .filter(|state| state.bone_id >= 0)
//                 .map(|state| (state.bone_id as usize, state.enabled))
//                 .collect(),
//             effect_parameters: v
//                 .effect_parameters
//                 .iter()
//                 .map(|p| (p.parameter_id, p.value))
//                 .collect(),
//             outside_parents: v.outside_parents,
//             has_edge_option: v.has_edge_option,
//             edge_scale_factor: v.edge_scale_factor,
//             edge_color: v.edge_color,
//             is_add_blending_enabled: v.is_add_blending_enabled,
//             is_physics_simulation_enabled: v.is_physics_simulation_enabled,
//         }
//     }
// }

// impl Seek for MotionTrack<MotionModelKeyframe> {
//     type Frame = ModelFrame;

//     fn find(&self, frame_index: u32) -> Option<Self::Frame> {
//         self.keyframes
//             .get(&frame_index)
//             .map(|keyframe| keyframe.into())
//     }

//     fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
//         if let Some(frame) = self.find(frame_index) {
//             frame
//         } else {
//             match self.search_closest(frame_index) {
//                 (Some(prev), Some(next)) => {
//                     let coef = super::interpolation::coefficient(
//                         prev.base.frame_index,
//                         next.base.frame_index,
//                         frame_index,
//                     );
//                     ModelFrame {
//                         visible: prev.visible,
//                         constraint_states: prev
//                             .constraint_states
//                             .iter()
//                             .filter(|state| state.bone_id >= 0)
//                             .map(|state| (state.bone_id as usize, state.enabled))
//                             .collect(),
//                         effect_parameters: prev
//                             .effect_parameters
//                             .iter()
//                             .map(|p| (p.parameter_id, p.value))
//                             .collect(),
//                         outside_parents: prev.outside_parents.clone(),
//                         has_edge_option: prev.has_edge_option,
//                         edge_scale_factor: prev.edge_scale_factor,
//                         edge_color: prev.edge_color,
//                         is_add_blending_enabled: prev.is_add_blending_enabled,
//                         is_physics_simulation_enabled: prev.is_physics_simulation_enabled,
//                     }
//                 }
//                 (Some(prev_frame), None) => prev_frame.into(),
//                 _ => ModelFrame::default(),
//             }
//         }
//     }

//     fn seek_precisely(
//         &self,
//         frame_index: u32,
//         amount: f32,
//         curve_factory: &dyn BezierCurveFactory,
//     ) -> Self::Frame {
//         if let Some(frame0) = self.seek(frame_index, curve_factory) {
//             if amount > 0f32 {
//                 if let Some(frame1) = self.seek(frame_index, curve_factory) {
//                     Some(ModelFrame {
//                         visible: frame0.visible,
//                         constraint_states: frame0.constraint_states,
//                         effect_parameters: frame0.effect_parameters,
//                         outside_parents: frame0.outside_parents,
//                         has_edge_option: frame0.has_edge_option,
//                         edge_scale_factor: frame0.edge_scale_factor,
//                         edge_color: frame0.edge_color,
//                         is_add_blending_enabled: frame0.is_add_blending_enabled,
//                         is_physics_simulation_enabled: frame0.is_physics_simulation_enabled,
//                     })
//                 } else {
//                     Some(frame0)
//                 }
//             } else {
//                 Some(frame0)
//             }
//         } else {
//             None
//         }

//         fn lerp_effect_parameters(
//             prev: &Vec<MotionEffectParameter>,
//             next: &Vec<MotionEffectParameter>,
//             coef: f32,
//         ) -> HashMap<i32, MotionEffectParameterValue> {
//             let build_map = |vec: &Vec<MotionEffectParameter>| {vec

//                 .iter()
//                 .map(|p| (p.parameter_id, p.value))
//                 .collect()};

//             let next: HashMap<_, _> = build_map(next);
//             let mut result = HashMap::new();
//             for prev_param in prev {
//                 if let Some(next_param) = next.get(&prev_param.parameter_id) {
//                     result.insert(
//                         prev_param.parameter_id,
//                         prev_param.value.lerp_or_first(*next_param, coef),
//                     );
//                 } else {
//                     result.insert(prev_param.parameter_id, prev_param.value);
//                 }
//             }
//             result
//         }
//     }
// }

#[derive(Debug, Clone, Copy)]
pub struct BoneFrameTransform {
    pub translation: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub interpolation: BoneKeyframeInterpolation,
    pub local_transform_mix: Option<f32>,
    pub enable_physics: bool,
    pub disable_physics: bool,
}

impl BoneFrameTransform {
    pub fn mixed_translation(&self, local_user_translation: Vector3<f32>) -> Vector3<f32> {
        if let Some(coef) = self.local_transform_mix {
            local_user_translation.lerp(self.translation, coef)
        } else {
            self.translation
        }
    }

    pub fn mixed_orientation(&self, local_user_orientation: Quaternion<f32>) -> Quaternion<f32> {
        if let Some(coef) = self.local_transform_mix {
            local_user_orientation.slerp(self.orientation, coef)
        } else {
            self.orientation
        }
    }
}

impl Default for BoneFrameTransform {
    fn default() -> Self {
        Self {
            translation: Vector3::zero(),
            orientation: Quaternion::one(),
            interpolation: BoneKeyframeInterpolation::default(),
            local_transform_mix: None,
            enable_physics: false,
            disable_physics: false,
        }
    }
}

impl Seek for MotionTrack<MotionBoneKeyframe> {
    type Frame = BoneFrameTransform;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| BoneFrameTransform {
                translation: f128_to_vec3(keyframe.translation),
                orientation: f128_to_quat(keyframe.orientation),
                interpolation: BoneKeyframeInterpolation::build(keyframe.interpolation),
                local_transform_mix: None,
                enable_physics: keyframe.is_physics_simulation_enabled,
                disable_physics: false,
            })
    }

    fn seek(&self, frame_index: u32, bezier_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
            let coef = super::interpolation::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let next_translation = f128_to_vec3(next_frame.translation);
            let next_orientation = f128_to_quat(next_frame.orientation);
            let prev_enabled = prev_frame.is_physics_simulation_enabled;
            let next_enabled = next_frame.is_physics_simulation_enabled;
            if prev_enabled && !next_enabled {
                BoneFrameTransform {
                    translation: next_translation,
                    orientation: next_orientation,
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    local_transform_mix: Some(coef),
                    enable_physics: false,
                    disable_physics: true,
                }
            } else {
                let prev_translation = f128_to_vec3(prev_frame.translation);
                let prev_orientation = f128_to_quat(prev_frame.orientation);
                let translation_interpolation = Vector3::new(
                    &next_frame.interpolation.translation_x,
                    &next_frame.interpolation.translation_y,
                    &next_frame.interpolation.translation_z,
                )
                .map(KeyframeInterpolationPoint::new);
                let amounts = translation_interpolation
                    .map(|interpolation| interpolation.curve_value(interval, coef, bezier_factory));
                let translation = lerp_element_wise(prev_translation, next_translation, amounts);
                let orientation_interpolation =
                    KeyframeInterpolationPoint::new(&next_frame.interpolation.orientation);
                let amount = orientation_interpolation.curve_value(interval, coef, bezier_factory);
                let orientation = prev_orientation.slerp(next_orientation, amount);
                BoneFrameTransform {
                    translation,
                    orientation,
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    local_transform_mix: None,
                    enable_physics: prev_enabled && next_enabled,
                    disable_physics: false,
                }
            }
        } else {
            BoneFrameTransform::default()
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        let f0 = Self::seek(self, frame_index, curve_factory);
        if amount > 0f32 {
            let f1 = Self::seek(self, frame_index + 1, curve_factory);
            let local_transform_mix = match (f0.local_transform_mix, f1.local_transform_mix) {
                (Some(a0), Some(a1)) => Some(lerp_f32(a0, a1, amount)),
                (None, Some(a1)) => Some(amount * a1),
                (Some(a0), None) => Some((1f32 - amount) * a0),
                _ => None,
            };
            BoneFrameTransform {
                translation: f0.translation.lerp(f1.translation, amount),
                orientation: f0.orientation.slerp(f1.orientation, amount),
                interpolation: f0.interpolation.lerp(f1.interpolation, amount),
                local_transform_mix,
                enable_physics: f0.enable_physics && f1.enable_physics,
                disable_physics: f0.disable_physics || f1.disable_physics,
            }
        } else {
            f0
        }
    }
}

impl Seek for MotionTrack<MotionMorphKeyframe> {
    type Frame = f32;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| keyframe.weight)
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = super::interpolation::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            lerp_f32(prev_frame.weight, next_frame.weight, coef)
        } else {
            0f32
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        let w0 = self.seek(frame_index, curve_factory);
        if amount > 0f32 {
            let w1 = self.seek(frame_index, curve_factory);
            lerp_f32(w0, w1, amount)
        } else {
            w0
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraTransform {
    pub lookat: Vector3<f32>,
    pub angle: Vector3<f32>,
    pub fov: Rad<f32>,
    pub distance: f32,
    pub perspective: bool,
    pub outside_parent: Option<(i32, i32)>,
    pub interpolation: CameraKeyframeInterpolation,
}

impl From<&MotionCameraKeyframe> for CameraTransform {
    fn from(v: &MotionCameraKeyframe) -> Self {
        Self {
            lookat: f128_to_vec3(v.look_at),
            angle: f128_to_vec3(v.angle),
            fov: Deg(v.fov as f32).into(),
            distance: v.distance,
            perspective: v.is_perspective_view,
            outside_parent: v
                .outside_parent
                .map(|op| (op.global_model_track_index, op.global_bone_track_index)),
            interpolation: v.interpolation.into(),
        }
    }
}

impl Seek for MotionTrack<MotionCameraKeyframe> {
    type Frame = Option<CameraTransform>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, bezier_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
            let coef = super::interpolation::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let prev_interpolation: CameraKeyframeInterpolation = prev_frame.interpolation.into();
            let interpolation: CameraKeyframeInterpolation = next_frame.interpolation.into();
            let lookat_amount = interpolation
                .lookat
                .map(|v| v.curve_value(interval, coef, bezier_factory));
            let lookat = lerp_element_wise(
                f128_to_vec3(prev_frame.look_at),
                f128_to_vec3(next_frame.look_at),
                lookat_amount,
            );
            let angle = f128_to_vec3(prev_frame.angle).lerp(
                f128_to_vec3(next_frame.angle),
                interpolation
                    .angle
                    .curve_value(interval, coef, bezier_factory),
            );
            let prev_fov: Rad<f32> = Deg(prev_frame.fov as f32).into();
            let next_fov: Rad<f32> = Deg(prev_frame.fov as f32).into();
            let fov = Rad(lerp_f32(prev_fov.0, next_fov.0, coef));
            let distance = lerp_f32(
                prev_frame.distance,
                next_frame.distance,
                interpolation
                    .distance
                    .curve_value(interval, coef, bezier_factory),
            );
            let perspective = prev_frame.is_perspective_view;
            let outside_parent = prev_frame
                .outside_parent
                .map(|op| (op.global_model_track_index, op.global_bone_track_index));
            Some(CameraTransform {
                lookat,
                angle,
                fov,
                distance,
                perspective,
                outside_parent,
                interpolation: prev_interpolation.lerp(interpolation, coef),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(CameraTransform {
                        lookat: frame0.lookat.lerp(frame1.lookat, amount),
                        angle: frame0.angle.lerp(frame1.angle, amount),
                        fov: lerp_rad(frame0.fov, frame1.fov, amount),
                        distance: lerp_f32(frame0.distance, frame1.distance, amount),
                        perspective: frame0.perspective,
                        outside_parent: frame0.outside_parent,
                        interpolation: frame0.interpolation,
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LightFrame {
    pub color: Vector3<f32>,
    pub direction: Vector3<f32>,
}

impl From<&MotionLightKeyframe> for LightFrame {
    fn from(v: &MotionLightKeyframe) -> Self {
        Self {
            color: f128_to_vec3(v.color),
            direction: f128_to_vec3(v.direction),
        }
    }
}

impl Seek for MotionTrack<MotionLightKeyframe> {
    type Frame = Option<LightFrame>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = super::interpolation::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let prev: LightFrame = prev_frame.into();
            let next: LightFrame = next_frame.into();
            Some(LightFrame {
                color: prev.color.lerp(next.color, coef),
                direction: prev.direction.lerp(next.direction, coef),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(LightFrame {
                        color: frame0.color.lerp(frame1.color, amount),
                        direction: frame0.direction.lerp(frame1.direction, amount),
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SelfShadowParam {
    pub distance: f32,
    pub coverage: CoverageMode,
}

impl From<&MotionSelfShadowKeyframe> for SelfShadowParam {
    fn from(v: &MotionSelfShadowKeyframe) -> Self {
        Self {
            distance: v.distance,
            coverage: (v.mode as u32).into(),
        }
    }
}

impl Seek for MotionTrack<MotionSelfShadowKeyframe> {
    type Frame = Option<SelfShadowParam>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = super::interpolation::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            Some(SelfShadowParam {
                distance: lerp_f32(prev_frame.distance, next_frame.distance, coef),
                coverage: (prev_frame.mode as u32).into(),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(SelfShadowParam {
                        distance: lerp_f32(frame0.distance, frame1.distance, amount),
                        coverage: frame0.coverage,
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

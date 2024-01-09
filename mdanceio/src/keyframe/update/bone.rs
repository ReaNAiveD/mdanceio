use std::collections::HashMap;

use cgmath::{Quaternion, Vector3, Vector4};
use nanoem::motion::{
    MotionBoneKeyframe, MotionBoneKeyframeInterpolation, MotionKeyframeBase, MotionTrack,
};

use crate::{
    bezier_curve::BezierCurve,
    model::Bone,
    motion::{BoneKeyframeInterpolation, KeyframeBound, KeyframeInterpolationPoint},
    utils::{f128_to_quat, f128_to_vec4},
};

use super::updater::{KeyframeUpdater, Updatable, AddKeyframe, RemoveKeyframe};

pub struct BoneKeyframeBezierControlPointParameter {
    pub translation: Vector3<Vector4<u8>>,
    pub orientation: Vector4<u8>,
}

impl From<BoneKeyframeInterpolation> for BoneKeyframeBezierControlPointParameter {
    fn from(v: BoneKeyframeInterpolation) -> Self {
        Self {
            translation: v
                .translation
                .map(|point| point.bezier_control_point().into()),
            orientation: v.orientation.bezier_control_point().into(),
        }
    }
}

pub struct BoneKeyframeTranslationBezierControlPointParameter {
    pub x: Vector4<u8>,
    pub y: Vector4<u8>,
    pub z: Vector4<u8>,
}

pub struct BoneKeyframeState {
    pub translation: Vector4<f32>,
    pub orientation: Quaternion<f32>,
    pub stage_index: u32,
    pub bezier_param: BoneKeyframeBezierControlPointParameter,
    pub enable_physics_simulation: bool,
}

impl BoneKeyframeState {
    pub fn from_bone(bone: &Bone, enable_physics_simulation: bool) -> Self {
        Self {
            translation: bone.local_user_translation.extend(1f32),
            orientation: bone.local_user_orientation,
            stage_index: 0,
            bezier_param: bone.interpolation.into(),
            enable_physics_simulation,
        }
    }

    pub fn from_keyframe(keyframe: &MotionBoneKeyframe) -> Self {
        Self {
            translation: f128_to_vec4(keyframe.translation),
            orientation: f128_to_quat(keyframe.orientation),
            stage_index: keyframe.stage_index,
            bezier_param: BoneKeyframeBezierControlPointParameter {
                translation: Vector3 {
                    x: keyframe.interpolation.translation_x.into(),
                    y: keyframe.interpolation.translation_y.into(),
                    z: keyframe.interpolation.translation_z.into(),
                },
                orientation: keyframe.interpolation.orientation.into(),
            },
            enable_physics_simulation: keyframe.is_physics_simulation_enabled,
        }
    }

    pub fn to_keyframe(&self, frame_index: u32) -> MotionBoneKeyframe {
        MotionBoneKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            translation: self.translation.into(),
            orientation: self.orientation.into(),
            interpolation: MotionBoneKeyframeInterpolation {
                translation_x: self.bezier_param.translation.x.into(),
                translation_y: self.bezier_param.translation.y.into(),
                translation_z: self.bezier_param.translation.z.into(),
                orientation: self.bezier_param.orientation.into(),
            },
            stage_index: self.stage_index,
            is_physics_simulation_enabled: self.enable_physics_simulation,
        }
    }
}

pub struct BoneKeyframeOverrideInterpolation {
    pub target_frame_index: u32,
    pub translation_params: (
        BoneKeyframeTranslationBezierControlPointParameter,
        BoneKeyframeTranslationBezierControlPointParameter,
    ),
}

pub struct BoneKeyframeUpdater {
    pub name: String,
    pub added_state: BoneKeyframeState, 
    pub removed_state: Option<BoneKeyframeState>,
    pub bezier_curve_override: Option<BoneKeyframeOverrideInterpolation>,
    pub was_dirty: bool,
    pub frame_index: u32,
}

impl KeyframeUpdater for BoneKeyframeUpdater {
    fn updated(&self) -> bool {
        self.removed_state.is_some()
    }

    fn selected(&self) -> bool {
        self.removed_state.is_none()
    }
}

pub struct BoneKeyframeUpdaterArg {
    pub enable_bezier_curve_adjustment: bool,
    pub enable_physics_simulation: bool,
}

impl Updatable for MotionTrack<MotionBoneKeyframe> {
    type Object = Bone;
    type ObjectUpdater = BoneKeyframeUpdater;


    fn apply_add(&mut self, updater: &mut Self::ObjectUpdater, object: Option<&mut Self::Object>) {
        let old = self.insert_keyframe(updater.added_state.to_keyframe(updater.frame_index));
        if old.is_none() && updater.updated() {
            log::warn!("No existing keyframe when update bone keyframe")
        }
        if !updater.updated() && updater.selected() {
            // TODO: add keyframe to motion selection
        }
        if let Some(bone) = object {
            if bone.states.dirty {
                updater.was_dirty = true;
                bone.states.dirty = false;
            }
        }
    }

    fn apply_remove(
        &mut self,
        updater: &mut Self::ObjectUpdater,
        object: Option<&mut Self::Object>,
    ) {
        if let Some(old_state) = &updater.removed_state {
            let _ = self.insert_keyframe(old_state.to_keyframe(updater.frame_index));
        } else {
            let _ = self.remove_keyframe(updater.frame_index);
        }
        if updater.was_dirty {
            if let Some(bone) = object {
                bone.states.dirty = true;
            }
        }
    }
}

impl AddKeyframe for MotionTrack<MotionBoneKeyframe> {
    type Object = Bone;
    type Args = BoneKeyframeUpdaterArg;
    type ObjectUpdater = BoneKeyframeUpdater;

    fn build_updater_add(
        &self,
        object: &Self::Object,
        bound: &KeyframeBound,
        args: Self::Args,
    ) -> Self::ObjectUpdater {
        let name = object.name.clone();
        let mut new_state = BoneKeyframeState::from_bone(object, args.enable_physics_simulation);
        let old_state;
        let bezier_curve_override;
        if let Some(keyframe) = self.keyframes.get(&bound.current) {
            old_state = Some(BoneKeyframeState::from_keyframe(keyframe));
            bezier_curve_override = None;
        } else {
            old_state = None;
            if let Some(prev_frame_index) = bound.previous {
                let prev_keyframe = self.keyframes.get(&prev_frame_index).unwrap();
                let movable = object.origin.flags.is_movable;
                let prev_interpolation_translation_x: Vector4<u8> =
                    prev_keyframe.interpolation.translation_x.into();
                let prev_interpolation_translation_y: Vector4<u8> =
                    prev_keyframe.interpolation.translation_y.into();
                let prev_interpolation_translation_z: Vector4<u8> =
                    prev_keyframe.interpolation.translation_z.into();
                let get_new_interpolation =
                    |prev_interpolation_value: Vector4<u8>,
                     new_state_interpolation: &mut Vector4<u8>| {
                        if args.enable_bezier_curve_adjustment && movable {
                            if KeyframeInterpolationPoint::is_linear_interpolation(
                                &prev_keyframe.interpolation.translation_x,
                            ) {
                                Bone::DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT.into()
                            } else if bound.next.is_some() && bound.next.unwrap() > prev_frame_index
                            {
                                let next_frame_index = bound.next.unwrap();
                                let interval = next_frame_index - prev_frame_index;
                                let bezier_curve = BezierCurve::from_parameters(
                                    prev_interpolation_value,
                                    interval,
                                );
                                let amount =
                                    (bound.current - prev_frame_index) as f32 / (interval as f32);
                                let pair = bezier_curve.split(amount);
                                *new_state_interpolation = pair.1.to_parameters();
                                pair.0.to_parameters()
                            } else {
                                prev_interpolation_value
                            }
                        } else {
                            prev_interpolation_value
                        }
                    };
                let new_interpolation_translation_x = get_new_interpolation(
                    prev_interpolation_translation_x,
                    &mut new_state.bezier_param.translation.x,
                );
                let new_interpolation_translation_y = get_new_interpolation(
                    prev_interpolation_translation_y,
                    &mut new_state.bezier_param.translation.y,
                );
                let new_interpolation_translation_z = get_new_interpolation(
                    prev_interpolation_translation_z,
                    &mut new_state.bezier_param.translation.z,
                );
                bezier_curve_override = Some(BoneKeyframeOverrideInterpolation {
                    target_frame_index: prev_frame_index,
                    translation_params: (
                        BoneKeyframeTranslationBezierControlPointParameter {
                            x: new_interpolation_translation_x,
                            y: new_interpolation_translation_y,
                            z: new_interpolation_translation_z,
                        },
                        BoneKeyframeTranslationBezierControlPointParameter {
                            x: prev_interpolation_translation_x,
                            y: prev_interpolation_translation_y,
                            z: prev_interpolation_translation_z,
                        },
                    ),
                })
            } else {
                bezier_curve_override = None
            }
        }
        BoneKeyframeUpdater {
            name,
            added_state: new_state,
            removed_state: old_state,
            bezier_curve_override,
            was_dirty: false,
            frame_index: bound.current,
        }
    }
}

impl RemoveKeyframe for MotionTrack<MotionBoneKeyframe> {
    type ObjectKeyframe = MotionBoneKeyframe;
    type ObjectUpdater = BoneKeyframeUpdater;

    fn build_updater_remove(&self, keyframe: &MotionBoneKeyframe) -> BoneKeyframeUpdater {
        let name = &self.name;
        let new_state = BoneKeyframeState::from_keyframe(keyframe);
        BoneKeyframeUpdater {
            name: name.clone(),
            added_state: new_state,
            removed_state: None,
            bezier_curve_override: None, // join bezier curve not supported
            was_dirty: false,
            frame_index: keyframe.base.frame_index,
        }
    }
}

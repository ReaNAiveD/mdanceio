use std::{cell::RefCell, rc::Rc};

use crate::motion::{
    Motion, MotionAccessoryKeyframe, MotionBoneKeyframe, MotionCameraKeyframe,
    MotionEffectParameter, MotionLightKeyframe, MotionModelKeyframe,
    MotionModelKeyframeConstraintState, MotionOutsideParent, MotionSelfShadowKeyframe,
};

struct MutableBaseKeyframe {
    is_reference: bool,
    is_in_motion: bool,
}

impl MutableBaseKeyframe {
    fn can_delete(&self) -> bool {
        self.is_in_motion == false
    }
}

struct MutableMotionEffectParameter {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionEffectParameter>>,
}

struct MutableMotionOutsideParent {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionOutsideParent>>,
}

struct MutableMotionAccessoryKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionAccessoryKeyframe>>,
    num_allocated_effect_parameters: usize,
}

struct MutableMotionBoneKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionBoneKeyframe>>,
}

struct MutableMotionCameraKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionCameraKeyframe>>,
}

struct MutableMotionLightKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionLightKeyframe>>,
}

struct MutableMotionModelKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionModelKeyframe>>,
    num_allocated_constraint_states: usize,
    num_allocated_effect_parameters: usize,
    num_allocated_outside_parents: usize,
}

struct MutableMotionModelKeyframeConstraintState {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionModelKeyframeConstraintState>>,
}

struct MutableMotionMorphKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionModelKeyframe>>,
}

struct MutableMotionSelfShadowKeyframe {
    base: MutableBaseKeyframe,
    origin: Rc<RefCell<MotionSelfShadowKeyframe>>,
}

struct MutableMotion {
    origin: Rc<RefCell<Motion>>,
    is_reference: bool,
    num_allocated_accessory_keyframes: usize,
    num_allocated_bone_keyframes: usize,
    num_allocated_camera_keyframes: usize,
    num_allocated_light_keyframes: usize,
    num_allocated_model_keyframes: usize,
    num_allocated_morph_keyframes: usize,
    num_allocated_self_shadow_keyframes: usize,
}

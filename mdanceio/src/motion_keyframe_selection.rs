use nanoem::motion::{MotionBoneKeyframe, MotionMorphKeyframe, MotionAccessoryKeyframe, MotionCameraKeyframe, MotionLightKeyframe, MotionModelKeyframe, MotionSelfShadowKeyframe};

pub trait MotionKeyframeSelectionCommon<T> {
    fn contains(&self, keyframe: &T) -> bool;
    fn get_all(&self) -> (Vec<&T>, u32);
    fn add(&mut self, keyframe: &T);
    fn remove(&mut self, keyframe: &T);
    fn add_keyframes(&mut self, start: u32, end: u32);
}

pub trait MotionKeyframeSelection {
    fn contains_accessory_keyframe(&self, keyframe: &MotionAccessoryKeyframe) -> bool;
    fn get_all_accessory_keyframes(&self) -> (Vec<&MotionAccessoryKeyframe>, u32);
    fn add_accessory_keyframe(&mut self, keyframe: &MotionAccessoryKeyframe);
    fn remove_accessory_keyframe(&mut self, keyframe: &MotionAccessoryKeyframe);
    fn contains_bone_keyframe(&self, keyframe: &MotionBoneKeyframe) -> bool;
    fn get_all_bone_keyframes(&self) -> (Vec<&MotionBoneKeyframe>, u32);
    fn add_bone_keyframe(&mut self, keyframe: &MotionBoneKeyframe);
    fn remove_bone_keyframe(&mut self, keyframe: &MotionBoneKeyframe);
    fn contains_camera_keyframe(&self, keyframe: &MotionCameraKeyframe) -> bool;
    fn get_all_camera_keyframes(&self) -> (Vec<&MotionCameraKeyframe>, u32);
    fn add_camera_keyframe(&mut self, keyframe: &MotionCameraKeyframe);
    fn remove_camera_keyframe(&mut self, keyframe: &MotionCameraKeyframe);
    fn contains_light_keyframe(&self, keyframe: &MotionLightKeyframe) -> bool;
    fn get_all_light_keyframes(&self) -> (Vec<&MotionLightKeyframe>, u32);
    fn add_light_keyframe(&mut self, keyframe: &MotionLightKeyframe);
    fn remove_light_keyframe(&mut self, keyframe: &MotionLightKeyframe);
    fn contains_model_keyframe(&self, keyframe: &MotionModelKeyframe) -> bool;
    fn get_all_model_keyframes(&self) -> (Vec<&MotionModelKeyframe>, u32);
    fn add_model_keyframe(&mut self, keyframe: &MotionModelKeyframe);
    fn remove_model_keyframe(&mut self, keyframe: &MotionModelKeyframe);
    fn contains_morph_keyframe(&self, keyframe: &MotionMorphKeyframe) -> bool;
    fn get_all_morph_keyframes(&self) -> (Vec<&MotionMorphKeyframe>, u32);
    fn add_morph_keyframe(&mut self, keyframe: &MotionMorphKeyframe);
    fn remove_morph_keyframe(&mut self, keyframe: &MotionMorphKeyframe);
    fn contains_self_shadow_keyframe(&self, keyframe: &MotionSelfShadowKeyframe) -> bool;
    fn get_all_self_shadow_keyframes(&self) -> (Vec<&MotionSelfShadowKeyframe>, u32);
    fn add_self_shadow_keyframe(&mut self, keyframe: &MotionSelfShadowKeyframe);
    fn remove_self_shadow_keyframe(&mut self, keyframe: &MotionSelfShadowKeyframe);
    fn add_all_keyframes(&mut self, flags: u32);
    fn has_all_keyframes(&self, flags: u32);
    fn clear_all_keyframes(&mut self, flags: u32);
    fn add_accessory_keyframes(&mut self, start: u32, end: u32);
    fn add_camera_keyframes(&mut self, start: u32, end: u32);
    fn add_light_keyframes(&mut self, start: u32, end: u32);
    fn add_model_keyframes(&mut self, start: u32, end: u32);
    fn add_self_shadow_keyframes(&mut self, start: u32, end: u32);
    fn add_bone_keyframes(&mut self, bone: &MotionBoneKeyframe, start: u32, end: u32);
    fn add_morph_keyframes(&mut self, bone: &MotionMorphKeyframe, start: u32, end: u32);
}

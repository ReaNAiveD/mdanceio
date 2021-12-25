use std::{rc::{Weak, Rc}, cell::RefCell, collections::HashMap};

use crate::common::{UserData, F128};

struct MotionTrack {
    id: i32,
    name: String,
    keyframes: HashMap<i32, MotionKeyframeObject>,
}

struct Motion {
    annotations: HashMap<String, String>,
    target_model_name: String,
    accessory_keyframes: Vec<MotionAccessoryKeyframe>,
    bone_keyframes: Vec<MotionBoneKeyframe>,
    morph_keyframes: Vec<MotionMorphKeyframe>,
    camera_keyframes: Vec<MotionCameraKeyframe>,
    light_keyframes: Vec<MotionLightKeyframe>,
    model_keyframes: Vec<MotionModelKeyframe>,
    self_shadow_keyframes: Vec<MotionSelfShadowKeyframe>,
    local_bone_motion_track_allocated_id: i32,
    local_bone_motion_track_bundle: HashMap<MotionTrack, char>,
    local_morph_motion_track_allocated_id: i32,
    local_morph_motion_track_bundle: HashMap<MotionTrack, char>,
    global_motion_track_allocated_id: i32,
    global_motion_track_bundle: HashMap<MotionTrack, char>,
    typ: i32,
    max_frame_index: u32,
    preferred_fps: f32,
    user_data: Rc<RefCell<UserData>>,
}

enum MotionEffectParameterValue {
    BOOL(bool),
    INT(i32),
    FLOAT(f32),
    VECTOR4(F128),
}

enum MotionParentKeyframe {
    ACCESSARY(Weak<RefCell<MotionAccessoryKeyframe>>),
    CAMERA(Weak<RefCell<MotionCameraKeyframe>>),
    MODEL(Weak<RefCell<MotionModelKeyframe>>),
}

struct MotionEffectParameter {
    parameter_id: i32,
    keyframe: MotionParentKeyframe,
    value: MotionEffectParameterValue,
}

struct MotionOutsideParent {
    keyframe: MotionParentKeyframe,
    global_model_track_index: i32,
    global_bone_track_index: i32,
    local_bone_track_index: i32,
}

struct MotionKeyframeObject {
    index: i32,
    frame_index: u32,
    is_selected: bool,
    user_data: Rc<RefCell<UserData>>,
    annotations: HashMap<String, String>,
}

struct MotionAccessoryKeyframe {
    base: MotionKeyframeObject,
    translation: F128,
    orientation: F128,
    scale_factor: f32,
    opacity: f32,
    is_add_blending_enabled: bool,
    is_shadow_enabled: bool,
    visible: bool,
    effect_parameters: Vec<MotionEffectParameter>,
    accessary_id: i32,
    outside_parent: Weak<RefCell<MotionOutsideParent>>,
}

const MOTION_BONE_KEYFRAME_INTERPOLATION_TYPE_MAX_ENUM: usize = 4;

struct MotionBoneKeyframe {
    base: MotionKeyframeObject,
    translation: F128,
    orientation: F128,
    interplation: [[u8; 4]; MOTION_BONE_KEYFRAME_INTERPOLATION_TYPE_MAX_ENUM],
    bone_id: i32,
    stage_index: u32,
    is_physics_simulation_enabled: bool,
}

struct MotionCameraKeyframe {
    base: MotionKeyframeObject,
    look_at: F128,
    angle: F128,
    distance: f32,
    fov: i32,
    interplation: [[u8; 4]; MOTION_BONE_KEYFRAME_INTERPOLATION_TYPE_MAX_ENUM],
    is_perspective_view: bool,
    stage_index: u32,
    outside_parent: Weak<RefCell<MotionOutsideParent>>
}

struct MotionLightKeyframe {
    base: MotionKeyframeObject,
    color: F128,
    direction: F128,
}

struct MotionModelKeyframeConstraintState {
    bone_id: i32,
    enabled: bool,
}

struct MotionModelKeyframe {
    base: MotionKeyframeObject,
    visible: bool,
    constraint_states: Vec<MotionModelKeyframeConstraintState>,
    effect_parameters: Vec<MotionEffectParameter>,
    outside_parents: Vec<Weak<RefCell<MotionOutsideParent>>>,
    has_edge_option: bool,
    edge_scale_factor: f32,
    edge_color: F128,
    is_add_blending_enabled: bool,
    is_physics_simulation_enabled: bool,
}

struct MotionMorphKeyframe {
    base: MotionKeyframeObject,
    weight: f32,
    morph_id: i32,
}

struct MotionSelfShadowKeyframe {
    base: MotionKeyframeObject,
    distance: f32,
    mode: i32,
}
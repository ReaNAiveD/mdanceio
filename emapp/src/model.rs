use std::collections::HashMap;

use cgmath::{Matrix4, Quaternion, Vector3, Vector4};
use nanoem::{motion::Motion, model::{ModelBone, ModelRigidBody}, common::LanguageType};

#[derive(Debug, Clone, Copy)]
struct Matrices {
    world_transform: Matrix4<f32>,
    local_transform: Matrix4<f32>,
    normal_transform: Matrix4<f32>,
    skinning_transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Copy)]
struct BezierControlPoints {
    translation_x: Vector4<u8>,
    translation_y: Vector4<u8>,
    translation_z: Vector4<u8>,
    orientation: Vector4<u8>,
}

#[derive(Debug, Clone, Copy)]
struct LinearInterpolationEnable {
    translation_x: bool,
    translation_y: bool,
    translation_z: bool,
    orientation: bool,
}

struct FrameTransform {
    translation: Vector3<f32>,
    orientation: Quaternion<f32>,
    bezier_control_points: BezierControlPoints,
    enable_linear_interpolation: LinearInterpolationEnable,
}

#[derive(Debug, Clone)]
struct Bone {
    name: String,
    canonical_name: String,
    matrices: Matrices,
    local_orientation: Quaternion<f32>,
    local_inherent_orientation: Quaternion<f32>,
    local_morph_orientation: Quaternion<f32>,
    local_user_orientation: Quaternion<f32>,
    constraint_joint_orientation: Quaternion<f32>,
    local_translation: Vector3<f32>,
    local_inherent_translation: Vector3<f32>,
    local_morph_translation: Vector3<f32>,
    local_user_translation: Vector3<f32>,
    bezier_control_points: BezierControlPoints,
    states: u32,
}

impl Bone {
    const DEFAULT_BAZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
    const NAME_ROOT_PARENT_IN_JAPANESE: &'static [u8] = &[
        0xe5, 0x85, 0xa8, 0xe3, 0x81, 0xa6, 0xe3, 0x81, 0xae, 0xe8, 0xa6, 0xaa, 0x0,
    ];
    const NAME_CENTER_IN_JAPANESE: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0,
    ];
    const NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE: &'static [u8] = &[
        0xe6, 0x93, 0x8d, 0xe4, 0xbd, 0x9c, 0xe4, 0xb8, 0xad, 0xe5, 0xbf, 0x83, 0,
    ];
    const NAME_CENTER_OFFSET_IN_JAPANESE: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0xe5, 0x85, 0x88, 0,
    ];
    const NAME_LEFT_IN_JAPANESE: &'static [u8] = &[0xe5, 0xb7, 0xa6, 0x0];
    const NAME_RIGHT_IN_JAPANESE: &'static [u8] = &[0xe5, 0x8f, 0xb3, 0x0];
    const NAME_DESTINATION_IN_JAPANESE: &'static [u8] = &[0xe5, 0x85, 0x88, 0x0];
    const LEFT_KNEE_IN_JAPANESE: &'static [u8] =
        &[0xe5, 0xb7, 0xa6, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];
    const RIGHT_KNEE_IN_JAPANESE: &'static [u8] =
        &[0xe5, 0x8f, 0xb3, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];

    // fn synchronize_transform(motion: &mut Motion, model_bone: &ModelBone, model_rigid_body: &ModelRigidBody, frame_index: u32, transform: &FrameTransform) {
    //     let name = model_bone.get_name(LanguageType::Japanese).unwrap();
    //     if let Some(Keyframe) = motion.find_bone_keyframe_object(name, index)
    // }
}

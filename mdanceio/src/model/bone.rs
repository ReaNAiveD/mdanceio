use cgmath::{
    AbsDiffEq, ElementWise, Euler, InnerSpace, Matrix4, One, Quaternion, Rad, SquareMatrix,
    Vector3, Vector4, VectorSpace, Zero,
};

use crate::{
    motion::{KeyframeInterpolationPoint, Motion},
    physics_engine::PhysicsEngine,
    utils::{f128_to_quat, f128_to_vec3, mat4_truncate},
};

use super::{model::RigidBody, NanoemBone};

#[derive(Debug, Clone, Copy)]
pub struct Matrices {
    pub world_transform: Matrix4<f32>,
    pub local_transform: Matrix4<f32>,
    pub normal_transform: Matrix4<f32>,
    pub skinning_transform: Matrix4<f32>,
}

impl Default for Matrices {
    fn default() -> Self {
        Self {
            world_transform: Matrix4::identity(),
            local_transform: Matrix4::identity(),
            normal_transform: Matrix4::identity(),
            skinning_transform: Matrix4::identity(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoneKeyframeInterpolation {
    pub translation_x: KeyframeInterpolationPoint,
    pub translation_y: KeyframeInterpolationPoint,
    pub translation_z: KeyframeInterpolationPoint,
    pub orientation: KeyframeInterpolationPoint,
}

impl BoneKeyframeInterpolation {
    pub fn build(interpolation: nanoem::motion::MotionBoneKeyframeInterpolation) -> Self {
        Self {
            translation_x: KeyframeInterpolationPoint::build(interpolation.translation_x),
            translation_y: KeyframeInterpolationPoint::build(interpolation.translation_y),
            translation_z: KeyframeInterpolationPoint::build(interpolation.translation_z),
            orientation: KeyframeInterpolationPoint::build(interpolation.orientation),
        }
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        BoneKeyframeInterpolation {
            translation_x: self.translation_x.lerp(other.translation_x, amount),
            translation_y: self.translation_y.lerp(other.translation_y, amount),
            translation_z: self.translation_z.lerp(other.translation_z, amount),
            orientation: self.orientation.lerp(other.orientation, amount),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BoneFrameTransform {
    pub translation: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub interpolation: BoneKeyframeInterpolation,
}

impl Default for BoneFrameTransform {
    fn default() -> Self {
        Self {
            translation: Vector3::zero(),
            orientation: Quaternion::one(),
            interpolation: BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoneStates {
    pub dirty: bool,
    pub editing_masked: bool,
}

#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub canonical_name: String,
    pub matrices: Matrices,
    pub local_orientation: Quaternion<f32>,
    pub local_inherent_orientation: Quaternion<f32>,
    pub local_morph_orientation: Quaternion<f32>,
    pub local_user_orientation: Quaternion<f32>,
    pub constraint_joint_orientation: Quaternion<f32>,
    pub local_translation: Vector3<f32>,
    pub local_inherent_translation: Vector3<f32>,
    pub local_morph_translation: Vector3<f32>,
    pub local_user_translation: Vector3<f32>,
    pub interpolation: BoneKeyframeInterpolation,
    pub states: BoneStates,
    pub origin: NanoemBone,
}

impl Bone {
    pub const DEFAULT_BEZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    pub const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
    pub const NAME_ROOT_PARENT_IN_JAPANESE: &'static str = "全ての親";
    pub const NAME_CENTER_IN_JAPANESE: &'static str = "センター";
    pub const NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE: &'static str = "操作中心";
    pub const NAME_CENTER_OFFSET_IN_JAPANESE: &'static str = "セコター先";
    pub const NAME_LEFT_IN_JAPANESE: &'static str = "左";
    pub const NAME_RIGHT_IN_JAPANESE: &'static str = "右";
    pub const NAME_DESTINATION_IN_JAPANESE: &'static str = "先";
    pub const LEFT_KNEE_IN_JAPANESE: &'static str = "左ひざ";
    pub const RIGHT_KNEE_IN_JAPANESE: &'static str = "右ひざ";

    pub fn from_nanoem(bone: &NanoemBone, language_type: nanoem::common::LanguageType) -> Self {
        let mut name = bone.get_name(language_type).to_owned();
        let mut canonical_name = bone
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Bone{}", bone.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        Self {
            name,
            canonical_name,
            matrices: Matrices {
                world_transform: Matrix4::identity(),
                local_transform: Matrix4::identity(),
                normal_transform: Matrix4::identity(),
                skinning_transform: Matrix4::identity(),
            },
            local_orientation: Quaternion::one(),
            local_inherent_orientation: Quaternion::one(),
            local_morph_orientation: Quaternion::one(),
            local_user_orientation: Quaternion::one(),
            constraint_joint_orientation: Quaternion::one(),
            local_translation: Vector3::zero(),
            local_inherent_translation: Vector3::zero(),
            local_morph_translation: Vector3::zero(),
            local_user_translation: Vector3::zero(),
            interpolation: BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
            },
            states: BoneStates::default(),
            origin: bone.clone(),
        }
    }

    fn translate(v: &Vector3<f32>, m: &Matrix4<f32>) -> Matrix4<f32> {
        m * Matrix4::from_translation(*v)
    }

    fn shrink_3x3(m: &Matrix4<f32>) -> Matrix4<f32> {
        let mut result = *m;
        result[3] = Vector4::new(0f32, 0f32, 0f32, 1f32);
        result
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        rigid_body: Option<&mut RigidBody>,
        frame_index: u32,
        amount: f32,
        physics_engine: &mut PhysicsEngine,
    ) {
        let t0 = self.synchronize_transform(motion, rigid_body, frame_index, physics_engine);
        if amount > 0f32 {
            let t1 = self.synchronize_transform(motion, None, frame_index + 1, physics_engine);
            self.local_user_translation = t0.translation.lerp(t1.translation, amount);
            self.local_user_orientation = t0.orientation.slerp(t1.orientation, amount);
            self.interpolation = t0.interpolation.lerp(t1.interpolation, amount);
        } else {
            self.local_user_translation = t0.translation;
            self.local_user_orientation = t0.orientation;
            self.interpolation = t0.interpolation;
        }
    }

    fn synchronize_transform(
        &self,
        motion: &Motion,
        rigid_body: Option<&mut RigidBody>,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) -> BoneFrameTransform {
        if let Some(keyframe) = motion.find_bone_keyframe(&self.canonical_name, frame_index) {
            BoneFrameTransform {
                translation: f128_to_vec3(keyframe.translation),
                orientation: f128_to_quat(keyframe.orientation),
                interpolation: BoneKeyframeInterpolation::build(keyframe.interpolation),
            }
        } else if let (Some(prev_frame), Some(next_frame)) = motion
            .opaque
            .search_closest_bone_keyframes(&self.canonical_name, frame_index)
        {
            let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let prev_translation = f128_to_vec3(prev_frame.translation);
            let next_translation = f128_to_vec3(next_frame.translation);
            let prev_orientation = f128_to_quat(prev_frame.orientation);
            let next_orientation = f128_to_quat(next_frame.orientation);
            let prev_enabled = prev_frame.is_physics_simulation_enabled;
            let next_enabled = next_frame.is_physics_simulation_enabled;
            let frame_transform = if prev_enabled && !next_enabled && rigid_body.is_some() {
                BoneFrameTransform {
                    translation: self.local_user_translation.lerp(next_translation, coef),
                    orientation: self.local_user_orientation.slerp(next_orientation, coef),
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                }
            } else {
                BoneFrameTransform {
                    translation: Vector3::new(
                        motion.lerp_value_interpolation(
                            &next_frame.interpolation.translation_x,
                            prev_translation.x,
                            next_translation.x,
                            interval,
                            coef,
                        ),
                        motion.lerp_value_interpolation(
                            &next_frame.interpolation.translation_y,
                            prev_translation.y,
                            next_translation.y,
                            interval,
                            coef,
                        ),
                        motion.lerp_value_interpolation(
                            &next_frame.interpolation.translation_z,
                            prev_translation.z,
                            next_translation.z,
                            interval,
                            coef,
                        ),
                    ),
                    orientation: motion.slerp_interpolation(
                        &next_frame.interpolation.orientation,
                        &prev_orientation,
                        &next_orientation,
                        interval,
                        coef,
                    ),
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                }
            };
            if prev_enabled && next_enabled {
                if let Some(rigid_body) = rigid_body {
                    rigid_body.disable_kinematic(physics_engine);
                }
            }
            frame_transform
        } else {
            BoneFrameTransform::default()
        }
    }

    pub fn update_local_transform(
        &mut self,
        parent_origin_world_transform: Option<(Vector3<f32>, Matrix4<f32>)>,
    ) {
        if self
            .local_translation
            .abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
            && self
                .local_orientation
                .abs_diff_eq(&Quaternion::one(), Quaternion::<f32>::default_epsilon())
        {
            self.update_local_transform_to(
                parent_origin_world_transform,
                &Vector3::zero(),
                &Quaternion::one(),
            );
        } else {
            let translation = self.local_translation;
            let orientation = self.local_orientation;
            self.update_local_transform_to(
                parent_origin_world_transform,
                &translation,
                &orientation,
            );
        }
    }

    pub fn update_local_transform_to(
        &mut self,
        parent_origin_world_transform: Option<(Vector3<f32>, Matrix4<f32>)>,
        translation: &Vector3<f32>,
        orientation: &Quaternion<f32>,
    ) {
        let local_transform = Matrix4::from_translation(*translation) * Matrix4::from(*orientation);
        let bone_origin = f128_to_vec3(self.origin.origin);
        if let Some((parent_origin, parent_world_transform)) = parent_origin_world_transform {
            let offset = bone_origin - parent_origin;
            let offset_matrix = Matrix4::from_translation(offset);
            let local_transform_with_offset = offset_matrix * local_transform;
            self.matrices.world_transform = parent_world_transform * local_transform_with_offset;
        } else {
            let offset_matrix = Matrix4::from_translation(bone_origin);
            self.matrices.world_transform = offset_matrix * local_transform;
        }
        self.matrices.local_transform = local_transform;
        self.matrices.skinning_transform =
            Self::translate(&-bone_origin, &self.matrices.world_transform);
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn update_local_morph_transform(
        &mut self,
        morph: &nanoem::model::ModelMorphBone,
        weight: f32,
    ) {
        self.local_morph_translation =
            Vector3::zero().lerp(f128_to_vec3(morph.translation), weight);
        self.local_morph_orientation =
            Quaternion::one().slerp(f128_to_quat(morph.orientation), weight);
    }

    pub fn update_local_orientation(
        &mut self,
        parent_inherent_bone_and_is_constraint_joint_bone_active: Option<(&Self, bool)>,
        effector_bone_local_user_orientation: Option<Quaternion<f32>>,
        is_constraint_joint_bone_active: bool,
    ) {
        if self.origin.flags.has_inherent_orientation {
            let mut orientation = Quaternion::<f32>::one();
            if let Some((parent_bone, parent_is_constraint_joint_bone_active)) =
                parent_inherent_bone_and_is_constraint_joint_bone_active
            {
                if parent_bone.origin.flags.has_local_inherent {
                    orientation = Quaternion::<f32>::from(mat4_truncate(
                        parent_bone.matrices.local_transform,
                    )) * orientation;
                } else if parent_is_constraint_joint_bone_active {
                    orientation = parent_bone.constraint_joint_orientation * orientation;
                } else if parent_bone.origin.flags.has_inherent_orientation {
                    orientation = parent_bone.local_inherent_orientation * orientation;
                } else {
                    orientation = parent_bone.local_user_orientation * orientation;
                }
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                if let Some(effector_bone_local_user_orientation) =
                    effector_bone_local_user_orientation
                {
                    orientation =
                        orientation.slerp(effector_bone_local_user_orientation, coefficient);
                } else {
                    orientation = Quaternion::one().slerp(orientation, coefficient);
                }
            }
            let local_orientation = if is_constraint_joint_bone_active {
                self.constraint_joint_orientation * self.local_morph_orientation * orientation
            } else {
                self.local_morph_orientation * self.local_user_orientation * orientation
            };
            self.local_orientation = local_orientation;
            self.local_inherent_orientation = orientation;
        } else if is_constraint_joint_bone_active {
            self.local_orientation =
                (self.constraint_joint_orientation * self.local_morph_orientation).normalize();
        } else {
            self.local_orientation =
                (self.local_morph_orientation * self.local_user_orientation).normalize();
        }
    }

    fn update_local_translation(&mut self, parent_inherent_bone: Option<&Bone>) {
        let mut translation = self.local_user_translation;
        if self.origin.flags.has_inherent_translation {
            if let Some(parent_bone) = parent_inherent_bone {
                if parent_bone.origin.flags.has_local_inherent {
                    translation += parent_bone.matrices.local_transform[3].truncate();
                } else if parent_bone.origin.flags.has_inherent_translation {
                    translation += parent_bone.local_inherent_translation;
                } else {
                    translation += parent_bone
                        .local_translation
                        .mul_element_wise(parent_bone.local_morph_translation);
                }
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                translation *= coefficient;
            }
            self.local_inherent_translation = translation;
        }
        translation += self.local_morph_translation;
        self.local_translation = translation;
    }

    pub fn update_skinning_transform(&mut self, skinning_transform: Matrix4<f32>) {
        self.matrices.skinning_transform = skinning_transform;
        self.matrices.world_transform =
            Self::translate(&f128_to_vec3(self.origin.origin), &skinning_transform);
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn apply_all_local_transform(
        &mut self,
        parent_inherent_bone_and_is_constraint_joint_bone_active: Option<(&Self, bool)>,
        parent_origin_world_transform: Option<(Vector3<f32>, Matrix4<f32>)>,
        effector_bone_local_user_orientation: Option<Quaternion<f32>>,
        is_constraint_joint_bone_active: bool,
    ) {
        self.update_local_orientation(
            parent_inherent_bone_and_is_constraint_joint_bone_active,
            effector_bone_local_user_orientation,
            is_constraint_joint_bone_active,
        );
        self.update_local_translation(
            parent_inherent_bone_and_is_constraint_joint_bone_active.map(|v| v.0),
        );
        self.update_local_transform(parent_origin_world_transform);
        // We deprecate constraint embedded in bone. All constraints saved in model.constraints
        // self.solve_constraint(constraint, num_iterations, bones)
    }

    pub fn apply_outside_parent_transform(&mut self, outside_parent_bone: &Bone) {
        let inv_origin = -f128_to_vec3(self.origin.origin);
        let out = Self::translate(&inv_origin, &self.matrices.world_transform);
        self.matrices.world_transform = out * outside_parent_bone.matrices.world_transform;
        let out = Self::translate(&inv_origin, &self.matrices.world_transform);
        self.matrices.local_transform = out;
        self.matrices.skinning_transform = out;
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn reset_local_transform(&mut self) {
        self.local_orientation = Quaternion::one();
        self.local_inherent_orientation = Quaternion::one();
        self.local_translation = Vector3::zero();
        self.local_inherent_translation = Vector3::zero();
        self.interpolation = BoneKeyframeInterpolation {
            translation_x: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            translation_y: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            translation_z: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            orientation: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
        };
    }

    pub fn reset_morph_transform(&mut self) {
        self.local_morph_orientation = Quaternion::one();
        self.local_morph_translation = Vector3::zero();
    }

    pub fn reset_user_transform(&mut self) {
        self.local_user_orientation = Quaternion::one();
        self.local_user_translation = Vector3::zero();
        self.states.dirty = false;
    }

    pub fn world_transform_origin(&self) -> Vector3<f32> {
        self.matrices.world_transform[3].truncate()
    }

    pub fn has_unit_x_constraint(&self) -> bool {
        self.canonical_name == Self::LEFT_KNEE_IN_JAPANESE
            || self.canonical_name == Self::RIGHT_KNEE_IN_JAPANESE
    }

    pub fn constrain_orientation(
        orientation: Quaternion<f32>,
        upper_limit: &Vector3<f32>,
        lower_limit: &Vector3<f32>,
    ) -> Quaternion<f32> {
        let mut euler = Euler::from(orientation);
        euler.x = Rad(euler.x.0.clamp(lower_limit.x, upper_limit.x));
        euler.y = Rad(euler.y.0.clamp(lower_limit.y, upper_limit.y));
        euler.z = Rad(euler.z.0.clamp(lower_limit.z, upper_limit.z));
        euler.into()
    }
}

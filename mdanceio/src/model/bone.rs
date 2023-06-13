use std::collections::{HashMap, HashSet};

use cgmath::{
    AbsDiffEq, ElementWise, Euler, InnerSpace, Matrix4, One, Quaternion, Rad, Rotation3,
    SquareMatrix, Vector3, Vector4, VectorSpace, Zero,
};

use crate::{
    motion::{BoneKeyframeInterpolation, Motion},
    physics_engine::PhysicsEngine,
    utils::{f128_to_quat, f128_to_vec3, mat4_truncate},
};

use super::{
    constraint::{Constraint, ConstraintJoint, ConstraintSet},
    BoneIndex, ConstraintIndex, NanoemBone, NanoemConstraint, NanoemConstraintJoint, RigidBody,
};

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

#[derive(Debug, Clone, Copy)]
pub struct Inherent<T> {
    pub bone: BoneIndex,
    pub value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct ConstraintJointBind {
    pub constraint: ConstraintIndex,
    pub orientation: Quaternion<f32>,
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
    pub handle: BoneIndex,
    pub matrices: Matrices,
    pub local_orientation: Quaternion<f32>,
    pub local_translation: Vector3<f32>,
    pub local_morph_orientation: Quaternion<f32>,
    pub local_morph_translation: Vector3<f32>,
    pub local_user_orientation: Quaternion<f32>,
    pub local_user_translation: Vector3<f32>,
    pub local_inherent_orientation: Option<Inherent<Quaternion<f32>>>,
    pub local_inherent_translation: Option<Inherent<Vector3<f32>>>,
    pub constraint_joint: Option<ConstraintJointBind>,
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
        let mut local_inherent_orientation = None;
        if bone.has_inherent_orientation() {
            let parent_index = bone.parent_inherent_bone_index as usize;
            local_inherent_orientation = Some(Inherent {
                bone: parent_index,
                value: Quaternion::one(),
            });
        }
        let mut local_inherent_translation = None;
        if bone.has_inherent_translation() {
            let parent_index = bone.parent_inherent_bone_index as usize;
            local_inherent_translation = Some(Inherent {
                bone: parent_index,
                value: Vector3::zero(),
            });
        }
        Self {
            name,
            canonical_name,
            handle: bone.base.index,
            matrices: Matrices::default(),
            local_orientation: Quaternion::one(),
            local_morph_orientation: Quaternion::one(),
            local_user_orientation: Quaternion::one(),
            local_translation: Vector3::zero(),
            local_morph_translation: Vector3::zero(),
            local_user_translation: Vector3::zero(),
            local_inherent_orientation,
            local_inherent_translation,
            constraint_joint: None,
            interpolation: BoneKeyframeInterpolation::zero(),
            states: BoneStates::default(),
            origin: bone.clone(),
        }
    }

    pub fn empty(handle: usize) -> Self {
        Bone {
            name: "".to_owned(),
            canonical_name: "".to_owned(),
            handle,
            matrices: Matrices::default(),
            local_orientation: Quaternion::one(),
            local_inherent_orientation: None,
            local_morph_orientation: Quaternion::one(),
            local_user_orientation: Quaternion::one(),
            local_translation: Vector3::zero(),
            local_inherent_translation: None,
            local_morph_translation: Vector3::zero(),
            local_user_translation: Vector3::zero(),
            constraint_joint: None,
            interpolation: BoneKeyframeInterpolation::default(),
            states: BoneStates::default(),
            origin: NanoemBone {
                base: nanoem::model::ModelObject { index: handle },
                name_ja: "".to_owned(),
                name_en: "".to_owned(),
                constraint: None,
                parent_bone_index: -1,
                parent_inherent_bone_index: -1,
                effector_bone_index: -1,
                target_bone_index: -1,
                global_bone_index: -1,
                stage_index: -1,
                ..Default::default()
            },
        }
    }

    pub fn constraint_joint_orientation(&self) -> Quaternion<f32> {
        self.constraint_joint
            .map(|joint| joint.orientation)
            .unwrap_or(Quaternion::one())
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
        let transform = motion.find_bone_transform(&self.canonical_name, frame_index, amount);
        self.local_user_translation = transform.mixed_translation(self.local_user_translation);
        self.local_user_orientation = transform.mixed_orientation(self.local_user_orientation);
        self.interpolation = transform.interpolation;
        if let Some(rigid_body) = rigid_body {
            if transform.enable_physics && rigid_body.is_from_simulation() {
                rigid_body.disable_kinematic(physics_engine);
            }
            if transform.disable_physics {
                rigid_body.enable_kinematic(physics_engine);
            }
        }
    }

    pub fn update_matrices(&mut self, parent_bone: Option<&Self>) {
        if self
            .local_translation
            .abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
            && self
                .local_orientation
                .abs_diff_eq(&Quaternion::one(), Quaternion::<f32>::default_epsilon())
        {
            self.update_matrices_to(parent_bone, &Vector3::zero(), &Quaternion::one());
        } else {
            let translation = self.local_translation;
            let orientation = self.local_orientation;
            self.update_matrices_to(parent_bone, &translation, &orientation);
        }
    }

    pub fn update_matrices_to(
        &mut self,
        parent_bone: Option<&Self>,
        translation: &Vector3<f32>,
        orientation: &Quaternion<f32>,
    ) {
        let local_transform = Matrix4::from_translation(*translation) * Matrix4::from(*orientation);
        let bone_origin = f128_to_vec3(self.origin.origin);
        if let Some(parent) = parent_bone {
            let offset = bone_origin - f128_to_vec3(parent.origin.origin);
            let offset_matrix = Matrix4::from_translation(offset);
            let local_transform_with_offset = offset_matrix * local_transform;
            self.matrices.world_transform =
                parent.matrices.world_transform * local_transform_with_offset;
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
        parent_inherent_bone: Option<&Self>,
        effector_bone: Option<&Self>,
    ) {
        if let Some(local_inherent_orientation) = &mut self.local_inherent_orientation {
            let mut orientation = Quaternion::<f32>::one();
            if let Some(inherent_bone) = parent_inherent_bone {
                orientation = if inherent_bone.origin.flags.has_local_inherent {
                    Quaternion::<f32>::from(mat4_truncate(inherent_bone.matrices.local_transform))
                } else if let Some(joint) = &inherent_bone.constraint_joint {
                    joint.orientation
                } else if let Some(inherent) = &inherent_bone.local_inherent_orientation {
                    inherent.value
                } else {
                    inherent_bone.local_user_orientation
                } * orientation;
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                if let Some(effector_bone) = effector_bone {
                    orientation =
                        orientation.slerp(effector_bone.local_user_orientation, coefficient);
                } else {
                    orientation = Quaternion::one().slerp(orientation, coefficient);
                }
            }
            let local_orientation = if let Some(joint) = &self.constraint_joint {
                joint.orientation * self.local_morph_orientation * orientation
            } else {
                self.local_morph_orientation * self.local_user_orientation * orientation
            };
            self.local_orientation = local_orientation;
            local_inherent_orientation.value = orientation;
        } else if let Some(joint) = &mut self.constraint_joint {
            self.local_orientation = (joint.orientation * self.local_morph_orientation).normalize();
        } else {
            self.local_orientation =
                (self.local_morph_orientation * self.local_user_orientation).normalize();
        }
    }

    fn update_local_translation(&mut self, parent_inherent_bone: Option<&Self>) {
        let mut translation = self.local_user_translation;
        if let Some(local_inherent_translation) = &mut self.local_inherent_translation {
            if let Some(inherent_bone) = parent_inherent_bone {
                if inherent_bone.origin.flags.has_local_inherent {
                    translation += inherent_bone.matrices.local_transform[3].truncate();
                } else if let Some(inherent) = &inherent_bone.local_inherent_translation {
                    translation += inherent.value;
                } else {
                    translation += inherent_bone
                        .local_translation
                        .mul_element_wise(inherent_bone.local_morph_translation);
                }
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                translation *= coefficient;
            }
            local_inherent_translation.value = translation;
        }
        translation += self.local_morph_translation;
        self.local_translation = translation;
    }

    pub fn update_matrices_by_skinning(&mut self, skinning_transform: Matrix4<f32>) {
        self.matrices.skinning_transform = skinning_transform;
        self.matrices.world_transform =
            Self::translate(&f128_to_vec3(self.origin.origin), &skinning_transform);
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn apply_local_transform(
        &mut self,
        parent_bone: Option<&Bone>,
        parent_inherent_bone: Option<&Self>,
        effector_bone: Option<&Self>,
    ) {
        self.update_local_orientation(parent_inherent_bone, effector_bone);
        self.update_local_translation(parent_inherent_bone);
        self.update_matrices(parent_bone);
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
        if let Some(local_inherent_orientation) = &mut self.local_inherent_orientation {
            local_inherent_orientation.value = Quaternion::one();
        }
        self.local_translation = Vector3::zero();
        if let Some(local_inherent_translation) = &mut self.local_inherent_translation {
            local_inherent_translation.value = Vector3::zero();
        }
        self.interpolation = BoneKeyframeInterpolation::default();
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

    pub fn world_translation(&self) -> Vector3<f32> {
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

#[derive(Debug, Clone)]
pub struct BoneSet {
    bones: Vec<Bone>,
    bones_by_name: HashMap<String, BoneIndex>,
    inherent_bones: HashMap<BoneIndex, HashSet<BoneIndex>>,
    parent_bone_tree: HashMap<BoneIndex, Vec<BoneIndex>>,
    constraints: ConstraintSet,
    active_bone_pair: (Option<BoneIndex>, Option<BoneIndex>),
}

impl BoneSet {
    pub fn new(
        origin_bones: &[NanoemBone],
        origin_constraints: &[NanoemConstraint],
        language_type: nanoem::common::LanguageType,
    ) -> Self {
        let mut inherent_bones = HashMap::new();
        let mut bones_by_name = HashMap::new();
        let mut bones = origin_bones
            .iter()
            .map(|bone| Bone::from_nanoem(bone, language_type))
            .collect::<Vec<_>>();
        for bone in &bones {
            if bone.origin.has_inherent_orientation() || bone.origin.has_inherent_translation() {
                let parent_index = bone.origin.parent_inherent_bone_index as usize;
                if let Some(parent_bone) = origin_bones.get(parent_index) {
                    inherent_bones
                        .entry(parent_bone.base.index)
                        .or_insert(HashSet::new())
                        .insert(bone.handle);
                }
            }
            for language in nanoem::common::LanguageType::all() {
                bones_by_name.insert(bone.origin.get_name(*language).to_owned(), bone.handle);
            }
        }
        let mut parent_bone_tree = HashMap::new();
        for bone in &bones {
            if let Some(parent_bone) = origin_bones.get(bone.origin.parent_bone_index as usize) {
                parent_bone_tree
                    .entry(parent_bone.base.index)
                    .or_insert(vec![])
                    .push(bone.handle);
            }
        }
        let constraints = ConstraintSet::new(origin_constraints, &bones, language_type);
        Self::init_with_constraints(&mut bones, &constraints);
        Self {
            bones,
            bones_by_name,
            inherent_bones,
            parent_bone_tree,
            constraints,
            active_bone_pair: (None, None),
        }
    }

    fn init_with_constraints(bones: &mut [Bone], constraints: &ConstraintSet) {
        for bone in bones {
            if let Some(constraint) = constraints.find_by_joint(bone.handle) {
                if constraint.enabled() {
                    bone.constraint_joint = Some(ConstraintJointBind {
                        constraint: constraint.origin.base.index,
                        orientation: Quaternion::one(),
                    });
                } else {
                    bone.constraint_joint = None;
                }
            } else {
                bone.constraint_joint = None;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bones.len()
    }

    pub fn get(&self, bone: BoneIndex) -> Option<&Bone> {
        self.bones.get(bone)
    }

    pub fn find(&self, name: &str) -> Option<&Bone> {
        self.bones_by_name
            .get(name)
            .and_then(|idx| self.bones.get(*idx))
    }

    pub fn get_mut(&mut self, bone: BoneIndex) -> Option<&mut Bone> {
        self.bones.get_mut(bone)
    }

    pub fn find_mut(&mut self, name: &str) -> Option<&mut Bone> {
        self.bones_by_name
            .get(name)
            .and_then(|idx| self.bones.get_mut(*idx))
    }

    pub fn try_get(&self, idx: i32) -> Option<&Bone> {
        usize::try_from(idx).ok().and_then(|bone| self.get(bone))
    }

    pub fn try_get_mut(&mut self, idx: i32) -> Option<&mut Bone> {
        usize::try_from(idx)
            .ok()
            .and_then(|bone| self.get_mut(bone))
    }

    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
    }

    pub fn find_constraint(&self, bone_name: &str) -> Option<&Constraint> {
        self.bones_by_name
            .get(bone_name)
            .and_then(|bone| self.constraints.find_by_target(*bone))
    }

    pub fn find_mut_constraint(&mut self, bone_name: &str) -> Option<&mut Constraint> {
        self.bones_by_name
            .get(bone_name)
            .and_then(|bone| self.constraints.find_mut_by_target(*bone))
    }

    pub fn parent_of(&self, bone: BoneIndex) -> Option<&Bone> {
        self.get(bone)
            .and_then(|bone| usize::try_from(bone.origin.parent_bone_index).ok())
            .and_then(|idx| self.bones.get(idx))
    }

    pub fn parent_inherent_of(&self, bone: BoneIndex) -> Option<&Bone> {
        self.get(bone)
            .and_then(|bone| usize::try_from(bone.origin.parent_inherent_bone_index).ok())
            .and_then(|idx| self.bones.get(idx))
    }

    pub fn effector_of(&self, bone: BoneIndex) -> Option<&Bone> {
        self.get(bone)
            .and_then(|bone| usize::try_from(bone.origin.effector_bone_index).ok())
            .and_then(|idx| self.bones.get(idx))
    }

    pub fn target_of(&self, bone: BoneIndex) -> Option<&Bone> {
        self.get(bone)
            .and_then(|bone| usize::try_from(bone.origin.target_bone_index).ok())
            .and_then(|idx| self.bones.get(idx))
    }

    pub fn iter_idx(&self) -> impl Iterator<Item = BoneIndex> {
        0..self.bones.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Bone> {
        self.bones.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Bone> {
        self.bones.iter_mut()
    }

    pub fn get_active(&self) -> Option<&Bone> {
        self.active_bone_pair.0.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.0 != bone_idx {
            self.active_bone_pair.0 = bone_idx;
        }
    }

    pub fn get_active_outside_parent_subject_bone(&self) -> Option<&Bone> {
        self.active_bone_pair.1.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active_outside_parent_subject_bone(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.1 != bone_idx {
            self.active_bone_pair.1 = bone_idx;
        }
    }

    pub fn has_any_dirty_bone(&self) -> bool {
        self.bones
            .iter()
            .map(|bone| bone.states.dirty)
            .fold(false, |a, b| a | b)
    }
}

impl BoneSet {
    pub fn apply_local_transform(&mut self, bone: BoneIndex) {
        let parent_bone = self.parent_of(bone).cloned();
        let parent_inherent_bone = self.parent_inherent_of(bone).cloned();
        let effector_bone = self.effector_of(bone).cloned();
        if let Some(bone) = self.get_mut(bone) {
            bone.apply_local_transform(
                parent_bone.as_ref(),
                parent_inherent_bone.as_ref(),
                effector_bone.as_ref(),
            );
        }
        if let Some(constraint) = self.constraints.find_by_target(bone).cloned() {
            self.solve_constraint(&constraint);
        }
    }

    pub fn reset_local_transform(&mut self) {
        for bone in &mut self.bones {
            bone.reset_local_transform();
        }
    }

    pub fn update_matrices(&mut self, bone: BoneIndex) {
        let parent_bone = self.parent_of(bone).cloned();
        if let Some(bone) = self.get_mut(bone) {
            bone.update_matrices(parent_bone.as_ref());
        }
    }

    pub fn update_matrices_to(
        &mut self,
        bone: BoneIndex,
        translation: &Vector3<f32>,
        orientation: &Quaternion<f32>,
    ) {
        let parent_bone = self.parent_of(bone).cloned();
        if let Some(bone) = self.get_mut(bone) {
            bone.update_matrices_to(parent_bone.as_ref(), translation, orientation);
        }
    }

    pub fn solve_constraint(&mut self, constraint: &Constraint) {
        if constraint.states.enabled {
            let num_iterations = constraint.origin.num_iterations;
            let target_bone = self
                .try_get(constraint.origin.target_bone_index)
                .cloned()
                .unwrap();
            for iter_idx in 0..num_iterations {
                self.solve_constraint_iteration(&target_bone, iter_idx as usize, constraint);
            }
        } else {
            for joint in &constraint.origin.joints {
                if let Some(joint_bone) = self.try_get_mut(joint.bone_index) {
                    joint_bone.constraint_joint = None;
                }
            }
        }
    }

    fn solve_constraint_iteration(
        &mut self,
        target_bone: &Bone,
        iter_idx: usize,
        constraint: &Constraint,
    ) {
        for joint_idx in 0..constraint.origin.joints.len() {
            self.solve_constraint_joint(
                &constraint.origin.joints[0..=joint_idx],
                target_bone,
                iter_idx,
                constraint,
            );
        }
    }

    fn solve_constraint_joint(
        &mut self,
        joints: &[NanoemConstraintJoint],
        target_bone: &Bone,
        iter_idx: usize,
        constraint: &Constraint,
    ) {
        let joint = joints.last().unwrap();
        let joint_bone = self.try_get(joint.bone_index).unwrap();
        let effector_bone = self
            .try_get(constraint.origin.effector_bone_index)
            .cloned()
            .unwrap();
        let angle_limit = Rad(constraint.origin.angle_limit);
        if let Some(mut joint_result) = ConstraintJoint::solve_axis_angle(
            joint_bone.matrices.world_transform,
            effector_bone.world_translation().extend(1f32),
            target_bone.world_translation().extend(1f32),
        ) {
            if joint_bone.origin.flags.has_fixed_axis {
                let axis = f128_to_vec3(joint_bone.origin.fixed_axis);
                if axis.magnitude2().gt(&f32::EPSILON) {
                    joint_result.axis = axis;
                }
            }
            let new_angle_limit = angle_limit * (joints.len() as f32);
            joint_result.angle = if new_angle_limit > joint_result.angle {
                joint_result.angle
            } else {
                new_angle_limit
            };
            let orientation = Quaternion::from_axis_angle(joint_result.axis, joint_result.angle);
            let mut mixed_orientation = if iter_idx == 0 {
                orientation * joint_bone.local_user_orientation
            } else {
                joint_bone.constraint_joint_orientation() * orientation
            };
            if joint.has_angle_limit {
                let upper_limit = f128_to_vec3(joint.upper_limit);
                let lower_limit = f128_to_vec3(joint.lower_limit);
                mixed_orientation =
                    Bone::constrain_orientation(mixed_orientation, &upper_limit, &lower_limit)
            }
            let joint_bone = self.try_get_mut(joint.bone_index).unwrap();
            joint_bone.constraint_joint.as_mut().unwrap().orientation = mixed_orientation;

            for upper_joint in joints.iter().rev() {
                let upper_joint_bone = self.try_get(upper_joint.bone_index).unwrap();
                let translation = upper_joint_bone.local_translation;
                let orientation = upper_joint_bone.constraint_joint_orientation();
                self.update_matrices_to(upper_joint_bone.handle, &translation, &orientation);
            }
            let joint_bone = self.try_get(joint.bone_index).unwrap();
            joint_result.set_transform(joint_bone.matrices.world_transform);

            self.update_matrices(effector_bone.handle);
        }
    }
}

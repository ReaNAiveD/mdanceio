use std::collections::{HashMap, HashSet};

use cgmath::{AbsDiffEq, InnerSpace, Matrix3, Matrix4, One, Quaternion, Vector3, Vector4, Zero, Rad};

use crate::utils::Invert;

use super::{Bone, BoneIndex, ConstraintIndex, NanoemBone, NanoemConstraint};

#[derive(Debug, Clone, Copy)]
pub struct ConstraintJoint {
    pub orientation: Quaternion<f32>,
    pub translation: Vector3<f32>,
    pub target_direction: Vector3<f32>,
    pub effector_direction: Vector3<f32>,
    pub axis: Vector3<f32>,
    // 弧度
    pub angle: Rad<f32>,
}

impl Default for ConstraintJoint {
    fn default() -> Self {
        Self {
            orientation: Quaternion::one(),
            translation: Vector3::zero(),
            target_direction: Vector3::zero(),
            effector_direction: Vector3::zero(),
            axis: Vector3::zero(),
            angle: Rad(0f32),
        }
    }
}

impl ConstraintJoint {
    pub fn set_transform(&mut self, v: Matrix4<f32>) {
        self.orientation = (Matrix3 {
            x: v.x.truncate(),
            y: v.y.truncate(),
            z: v.z.truncate(),
        })
        .into();
        self.translation = v[3].truncate();
    }

    pub fn solve_axis_angle(
        transform: Matrix4<f32>,
        effector_position: Vector4<f32>,
        target_position: Vector4<f32>,
    ) -> Option<Self> {
        let inv_transform = transform.affine_invert().unwrap();
        let inv_effector_position = (inv_transform * effector_position).truncate();
        let inv_target_position = (inv_transform * target_position).truncate();
        if inv_effector_position.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
            || inv_target_position.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
        {
            return None;
        }
        let effector_direction = inv_effector_position.normalize();
        let target_direction = inv_target_position.normalize();
        let mut axis = effector_direction.cross(target_direction);
        if axis.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon()) {
            return None;
        }
        let z = effector_direction
            .dot(target_direction)
            .clamp(-1.0f32, 1.0f32);
        axis = axis.normalize();
        if z.abs() <= f32::default_epsilon() {
            return None;
        }
        let angle = Rad(z.acos());
        Some(Self {
            target_direction,
            effector_direction,
            axis,
            angle,
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintStates {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: String,
    pub canonical_name: String,
    // pub joint_iteration_result: Vec<Vec<ConstraintJoint>>,
    // pub effector_iteration_result: Vec<Vec<ConstraintJoint>>,
    pub states: ConstraintStates,
    pub origin: NanoemConstraint,
}

impl Constraint {
    pub fn from_nanoem(
        constraint: &NanoemConstraint,
        bone: Option<&NanoemBone>,
        language_type: nanoem::common::LanguageType,
    ) -> Self {
        let mut name = if let Some(bone) = bone {
            bone.get_name(language_type).to_owned()
        } else {
            "".to_owned()
        };
        let mut canonical_name = if let Some(bone) = bone {
            bone.get_name(nanoem::common::LanguageType::default())
                .to_owned()
        } else {
            "".to_owned()
        };
        if canonical_name.is_empty() {
            canonical_name = format!("Constraint{}", constraint.get_target_bone_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        // let joint_iteration_result =
        //     vec![
        //         vec![ConstraintJoint::default(); constraint.num_iterations as usize];
        //         constraint.joints.len()
        //     ];
        // let effector_iteration_result =
        //     vec![
        //         vec![ConstraintJoint::default(); constraint.num_iterations as usize];
        //         constraint.joints.len()
        //     ];
        Self {
            name,
            canonical_name,
            // joint_iteration_result,
            // effector_iteration_result,
            states: ConstraintStates { enabled: true },
            origin: constraint.clone(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.states.enabled
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
    target_to_constraint: HashMap<BoneIndex, ConstraintIndex>,
    joint_to_constraint: HashMap<BoneIndex, ConstraintIndex>,
    constraint_effector_bones: HashSet<BoneIndex>,
}

impl ConstraintSet {
    pub fn new(
        origin: &[NanoemConstraint],
        bones: &[Bone],
        language_type: nanoem::common::LanguageType,
    ) -> Self {
        let mut constraints = vec![];
        let mut target_to_constraint = HashMap::new();
        let mut joint_to_constraint = HashMap::new();
        let mut constraint_effector_bones = HashSet::new();
        let mut all_origin = origin.to_vec();
        for bone in bones {
            if let Some(mut constraint) = bone.origin.constraint.clone() {
                constraint.target_bone_index = bone.origin.base.index as i32;
                all_origin.push(constraint);
            }
        }
        all_origin.sort_unstable_by_key(|c| c.target_bone_index);
        for (idx, constraint) in all_origin.iter_mut().enumerate() {
            constraint.base.index = idx;
        }
        let get_bone = |idx: i32| usize::try_from(idx).ok().and_then(|idx| bones.get(idx));
        for constraint in &all_origin {
            let target_bone = get_bone(constraint.target_bone_index);
            constraints.push(Constraint::from_nanoem(
                constraint,
                target_bone.map(|bone| &bone.origin),
                language_type,
            ));
            for joint in &constraint.joints {
                if let Some(bone) = get_bone(joint.bone_index) {
                    joint_to_constraint.insert(bone.origin.base.index, constraint.base.index);
                }
            }
            if let Some(effector_bone) = get_bone(constraint.effector_bone_index) {
                constraint_effector_bones.insert(effector_bone.origin.base.index);
            }
            if let Some(target_bone) = target_bone {
                target_to_constraint.insert(target_bone.handle, constraint.base.index);
            }
        }
        Self {
            constraints,
            target_to_constraint,
            joint_to_constraint,
            constraint_effector_bones,
        }
    }

    pub fn get(&self, constraint: ConstraintIndex) -> Option<&Constraint> {
        self.constraints.get(constraint)
    }

    pub fn get_mut(&mut self, constraint: ConstraintIndex) -> Option<&mut Constraint> {
        self.constraints.get_mut(constraint)
    }

    pub fn iter_idx(&self) -> impl Iterator<Item = ConstraintIndex> {
        0..self.constraints.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Constraint> {
        self.constraints.iter_mut()
    }

    pub fn find_by_joint(&self, bone: BoneIndex) -> Option<&Constraint> {
        self.joint_to_constraint
            .get(&bone)
            .and_then(|idx| self.constraints.get(*idx))
    }

    pub fn find_mut_by_joint(&mut self, bone: BoneIndex) -> Option<&mut Constraint> {
        self.joint_to_constraint
            .get(&bone)
            .and_then(|idx| self.constraints.get_mut(*idx))
    }

    pub fn find_by_target(&self, bone: BoneIndex) -> Option<&Constraint> {
        self.target_to_constraint
            .get(&bone)
            .and_then(|idx| self.constraints.get(*idx))
    }

    pub fn find_mut_by_target(&mut self, bone: BoneIndex) -> Option<&mut Constraint> {
        self.target_to_constraint
            .get(&bone)
            .and_then(|idx| self.constraints.get_mut(*idx))
    }

    pub fn is_active(&self, constraint: ConstraintIndex) -> bool {
        self.constraints
            .get(constraint)
            .map(|constraint| constraint.states.enabled)
            .unwrap_or(false)
    }
}

use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
};

use cgmath::{Vector3, Zero};
use nalgebra::Isometry3;
use nanoem::{common::LanguageType, model::ModelRigidBodyTransformType};
use rapier3d::prelude::RigidBodyType;

use crate::{
    physics_engine::{PhysicsEngine, RigidBodyFollowBone},
    utils::{
        f128_to_isometry, f128_to_vec3, from_isometry, mat4_truncate, to_isometry, to_na_vec3,
        Invert,
    },
};

use super::{bone::BoneSet, Bone, BoneIndex, NanoemJoint, NanoemRigidBody, RigidBodyIndex};

type RapierRB = rapier3d::dynamics::RigidBody;
type RapierRBHandle = rapier3d::dynamics::RigidBodyHandle;
type RapierCollider = rapier3d::geometry::Collider;

#[derive(Debug, Clone, Copy, Default)]
pub struct RigidBodyStates {
    pub enabled: bool,
    pub all_forces_should_reset: bool,
    pub editing_masked: bool,
}

#[derive(Debug)]
pub struct RigidBody {
    // TODO: physics engine and shape mesh and engine rigid_body
    physics_rb: Option<RapierRBHandle>,
    initial_world_transform: nalgebra::Isometry3<f32>,
    global_torque_force: (Vector3<f32>, bool),
    global_velocity_force: (Vector3<f32>, bool),
    local_torque_force: (Vector3<f32>, bool),
    local_velocity_force: (Vector3<f32>, bool),
    pub name: String,
    pub canonical_name: String,
    pub states: RigidBodyStates,
    pub origin: NanoemRigidBody,
    pub is_morph: bool,
}

impl RigidBody {
    pub fn from_nanoem(
        rigid_body: &NanoemRigidBody,
        language: nanoem::common::LanguageType,
        is_morph: bool,
        bones: &BoneSet,
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        // TODO: set physics engine and bind to engine rigid_body
        let mut name = rigid_body.get_name(language).to_owned();
        let mut canonical_name = rigid_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("RigidBody{}", rigid_body.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let orientation = rigid_body.orientation;
        let origin = rigid_body.origin;
        let mut world_transform = f128_to_isometry(origin, orientation);
        let mut initial_world_transform = world_transform;
        if rigid_body.is_bone_relative {
            if let Some(bone) = bones.try_get(rigid_body.bone_index) {
                let bone_origin = bone.origin.origin;
                let offset = nalgebra::Isometry3::translation(
                    bone_origin[0],
                    bone_origin[1],
                    bone_origin[2],
                );
                world_transform = offset * world_transform;
                initial_world_transform = world_transform;
            }
        }

        let inner_rigid_body = Self::build_physics_rb(rigid_body, world_transform, is_morph);
        let rigid_body_handle = physics_engine.rigid_body_set.insert(inner_rigid_body);
        Self::build_physics_collider(rigid_body).map(|collider| {
            physics_engine.collider_set.insert_with_parent(
                collider,
                rigid_body_handle,
                &mut physics_engine.rigid_body_set,
            )
        });
        Self {
            global_torque_force: (Vector3::zero(), false),
            global_velocity_force: (Vector3::zero(), false),
            local_torque_force: (Vector3::zero(), false),
            local_velocity_force: (Vector3::zero(), false),
            name,
            canonical_name,
            states: RigidBodyStates {
                enabled: true,
                ..Default::default()
            },
            origin: rigid_body.clone(),
            physics_rb: Some(rigid_body_handle),
            initial_world_transform,
            is_morph,
        }
    }

    pub fn enable(&mut self, physics_engine: &mut PhysicsEngine) {
        if !self.states.enabled {
            let world_transform = self.initial_world_transform;
            let inner_rigid_body =
                Self::build_physics_rb(&self.origin, world_transform, self.is_morph);
            let rigid_body_handle = physics_engine.rigid_body_set.insert(inner_rigid_body);
            Self::build_physics_collider(&self.origin).map(|collider| {
                physics_engine.collider_set.insert_with_parent(
                    collider,
                    rigid_body_handle,
                    &mut physics_engine.rigid_body_set,
                )
            });
            self.physics_rb = Some(rigid_body_handle);
            self.states.enabled = true;
        }
    }

    pub fn disable(&mut self, physics_engine: &mut PhysicsEngine) {
        if self.states.enabled {
            if let Some(handle) = self.physics_rb {
                physics_engine.remove_rb(handle);
            }
            self.states.enabled = false;
        }
    }

    pub fn mark_all_forces_reset(&mut self) {
        self.states.all_forces_should_reset = true;
    }

    pub fn add_global_torque_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.global_torque_force.0 += value * weight;
        self.global_torque_force.1 = true;
    }

    pub fn add_global_velocity_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.global_velocity_force.0 += value * weight;
        self.global_velocity_force.1 = true;
    }

    pub fn add_local_torque_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.local_torque_force.0 += value * weight;
        self.local_torque_force.1 = true;
    }

    pub fn add_local_velocity_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.local_velocity_force.0 += value * weight;
        self.local_velocity_force.1 = true;
    }

    pub fn initialize_transform_feedback(
        &mut self,
        bone: Option<&Bone>,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = self
            .physics_rb
            .and_then(|handle| physics_engine.rigid_body_set.get_mut(handle))
        {
            if let Some(bone) = bone {
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                physics_rigid_body
                    .set_position(skinning_transform * self.initial_world_transform, true);
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            }
        }
    }

    pub fn enable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rb) = physics_engine.get_rb_mut(self.physics_rb) {
            if !physics_rb.is_kinematic() {
                physics_rb.set_body_type(RigidBodyType::KinematicPositionBased, true);
            }
        }
    }

    pub fn disable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rb) = physics_engine.get_rb_mut(self.physics_rb) {
            if physics_rb.is_kinematic() {
                physics_rb.set_body_type(RigidBodyType::Dynamic, true);
            }
        }
    }

    pub fn apply_forces(&mut self, bone: Option<&Bone>, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rb) = physics_engine.get_rb_mut(self.physics_rb) {
            if self.states.all_forces_should_reset {
                physics_rb.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rb.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rb.reset_forces(true);
            } else {
                if self.global_torque_force.1 {
                    physics_rb.apply_torque_impulse(to_na_vec3(self.global_torque_force.0), true);
                    self.global_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_torque_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rb.position())
                        .rotation;
                        physics_rb.apply_torque_impulse(
                            local_orientation * to_na_vec3(self.local_torque_force.0),
                            true,
                        );
                        self.local_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                    }
                }
                if self.global_velocity_force.1 {
                    physics_rb.apply_impulse(to_na_vec3(self.global_velocity_force.0), true);
                    self.global_velocity_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_velocity_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rb.position())
                        .rotation;
                        physics_rb.apply_impulse(
                            local_orientation * to_na_vec3(self.local_velocity_force.0),
                            true,
                        );
                        self.local_velocity_force = (Vector3::new(0f32, 0f32, 0f32), false);
                    }
                }
            }
            self.states.all_forces_should_reset = false;
        }
    }

    pub fn synchronize_to_simulation(&mut self, bone: &Bone, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine.get_rb_mut(self.physics_rb) {
            if self.is_to_simulation() || physics_rigid_body.is_kinematic() {
                let initial_transform = self.initial_world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                let world_transform = skinning_transform * initial_transform;
                physics_rigid_body.set_position(world_transform, true);
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            }
        }
    }

    pub fn synchronize_from_simulation(
        &mut self,
        bone: &mut Bone,
        parent_bone: Option<&Bone>,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine.get_rb_mut(self.physics_rb) {
            if self.is_from_simulation() && !physics_rigid_body.is_kinematic() {
                let initial_transform = self.initial_world_transform;
                let mut world_transform = *physics_rigid_body.position();
                if follow_type == RigidBodyFollowBone::Perform
                    && self.origin.transform_type
                        == ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone
                {
                    let local_transform = to_isometry(bone.matrices.local_transform);
                    world_transform =
                        nalgebra::Isometry3::from(local_transform.translation.inverse())
                            * world_transform;
                    physics_rigid_body.set_position(world_transform, true);
                }
                let skinning_transform =
                    from_isometry(world_transform * initial_transform.inverse());
                bone.update_matrices_by_skinning(skinning_transform);
                if let Some(parent_bone) = parent_bone {
                    let offset =
                        f128_to_vec3(self.origin.origin) - f128_to_vec3(parent_bone.origin.origin);
                    let local_transform = parent_bone
                        .matrices
                        .world_transform
                        .affine_invert()
                        .unwrap()
                        * bone.matrices.world_transform;
                    bone.local_user_translation = local_transform[3].truncate() - offset;
                    bone.local_user_orientation = mat4_truncate(local_transform).into();
                } else {
                    let local_transform = bone.matrices.world_transform;
                    bone.local_user_translation =
                        local_transform[3].truncate() - f128_to_vec3(bone.origin.origin);
                    bone.local_user_orientation = mat4_truncate(local_transform).into();
                }
                physics_rigid_body.wake_up(false);
            }
        }
    }

    pub fn is_kinematic(&self, physics_engine: &PhysicsEngine) -> bool {
        physics_engine
            .get_rb(self.physics_rb)
            .map(|rigid_body| rigid_body.is_kinematic())
            .unwrap_or(true)
    }

    pub fn is_to_simulation(&self) -> bool {
        matches!(
            self.origin.get_transform_type(),
            ModelRigidBodyTransformType::FromBoneToSimulation
        )
    }

    pub fn is_from_simulation(&self) -> bool {
        matches!(
            self.origin.get_transform_type(),
            ModelRigidBodyTransformType::FromSimulationToBone
                | ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone
        )
    }
}

impl RigidBody {
    pub fn build_physics_rb(
        origin_rb: &NanoemRigidBody,
        world_transform: Isometry3<f32>,
        is_morph: bool,
    ) -> RapierRB {
        let rigid_body_builder =
            rapier3d::dynamics::RigidBodyBuilder::new(RigidBodyType::KinematicPositionBased)
                .position(world_transform)
                .angular_damping(origin_rb.angular_damping)
                .linear_damping(origin_rb.linear_damping);
        let mut rigid_body = rigid_body_builder.build();
        if matches!(
            origin_rb.transform_type,
            ModelRigidBodyTransformType::FromBoneToSimulation
        ) || is_morph
        {
            rigid_body.wake_up(true);
        }
        rigid_body
    }

    fn build_physics_collider(origin_rb: &NanoemRigidBody) -> Option<RapierCollider> {
        let size = origin_rb.size;
        let collider_builder = match origin_rb.shape_type {
            nanoem::model::ModelRigidBodyShapeType::Unknown => None,
            nanoem::model::ModelRigidBodyShapeType::Sphere => {
                Some(rapier3d::geometry::ColliderBuilder::ball(size[0]))
            }
            nanoem::model::ModelRigidBodyShapeType::Box => Some(
                rapier3d::geometry::ColliderBuilder::cuboid(size[0], size[1], size[2]),
            ),
            nanoem::model::ModelRigidBodyShapeType::Capsule => Some(
                rapier3d::geometry::ColliderBuilder::capsule_y(size[1] / 2f32, size[0]),
            ),
        };
        let mass = match origin_rb.transform_type {
            ModelRigidBodyTransformType::FromBoneToSimulation => 0f32,
            _ => origin_rb.mass,
        };
        let mut group = rapier3d::geometry::Group::from_bits_truncate(
            0x1u32 << origin_rb.collision_group_id.clamp(0, 15),
        );
        if matches!(
            origin_rb.transform_type,
            ModelRigidBodyTransformType::FromBoneToSimulation
        ) {
            group |= rapier3d::geometry::Group::GROUP_2;
        }
        collider_builder.map(|builder| {
            builder
                .mass(mass)
                .friction(origin_rb.friction)
                .restitution(origin_rb.restitution)
                .collision_groups(rapier3d::geometry::InteractionGroups::new(
                    group,
                    rapier3d::geometry::Group::from_bits_truncate(
                        (origin_rb.collision_mask & 0xffff) as u32,
                    ),
                ))
                .build()
        })
    }
}

pub struct RigidBodySet {
    rigid_bodies: Vec<RigidBody>,
    bone_to_rigid_bodies: HashMap<BoneIndex, RigidBodyIndex>,
    bone_bound_rigid_bodies: HashMap<BoneIndex, RigidBodyIndex>,
}

impl RigidBodySet {
    pub fn new(
        rigid_bodies: &[NanoemRigidBody],
        bones: &BoneSet,
        morph_bones: &HashSet<BoneIndex>,
        language_type: LanguageType,
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        let rigid_bodies: Vec<RigidBody> = rigid_bodies
            .iter()
            .map(|rigid_body| {
                let is_dynamic = !matches!(
                    rigid_body.get_transform_type(),
                    nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation
                );
                let is_morph = if let Some(bone) = bones.try_get(rigid_body.get_bone_index()) {
                    is_dynamic && morph_bones.contains(&bone.handle)
                } else {
                    false
                };
                RigidBody::from_nanoem(rigid_body, language_type, is_morph, bones, physics_engine)
            })
            .collect();
        let mut bone_to_rigid_bodies = HashMap::new();
        let mut bone_bound_rigid_bodies = HashMap::new();
        for (handle, rigid_body) in rigid_bodies.iter().enumerate() {
            if let Some(bone) = bones.try_get(rigid_body.origin.bone_index) {
                bone_to_rigid_bodies.insert(bone.handle, handle);
                if rigid_body.is_from_simulation() {
                    bone_bound_rigid_bodies.insert(bone.handle, handle);
                }
            }
        }
        Self {
            rigid_bodies,
            bone_to_rigid_bodies,
            bone_bound_rigid_bodies,
        }
    }

    pub fn get(&self, handle: RigidBodyIndex) -> Option<&RigidBody> {
        self.rigid_bodies.get(handle)
    }

    pub fn get_mut(&mut self, handle: RigidBodyIndex) -> Option<&mut RigidBody> {
        self.rigid_bodies.get_mut(handle)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut RigidBody> {
        self.rigid_bodies.iter_mut()
    }

    pub fn find_mut_by_bone(&mut self, bone: BoneIndex) -> Option<&mut RigidBody> {
        self.bone_to_rigid_bodies
            .get(&bone)
            .and_then(|idx| self.rigid_bodies.get_mut(*idx))
    }

    pub fn find_mut_by_bone_bound(&mut self, bone: BoneIndex) -> Option<&mut RigidBody> {
        self.bone_bound_rigid_bodies
            .get(&bone)
            .and_then(|idx| self.rigid_bodies.get_mut(*idx))
    }
}

impl RigidBodySet {
    pub fn synchronize_to_simulation(
        &mut self,
        bones: &BoneSet,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in self.rigid_bodies.iter_mut() {
            let bone = bones.try_get(rigid_body.origin.bone_index);
            rigid_body.apply_forces(bone, physics_engine);
            if let Some(bone) = bone {
                rigid_body.synchronize_to_simulation(bone, physics_engine);
            }
        }
    }

    pub fn synchronize_from_simulation(
        &mut self,
        bones: &mut BoneSet,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in self.rigid_bodies.iter_mut() {
            if let Some(bone) = bones.try_get(rigid_body.origin.bone_index) {
                let parent_bone = bones.try_get(bone.origin.parent_bone_index).cloned();
                rigid_body.synchronize_from_simulation(
                    bones.get_mut(bone.handle).unwrap(),
                    parent_bone.as_ref(),
                    follow_type,
                    physics_engine,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JointStates {
    pub enabled: bool,
    pub editing_masked: bool,
}

pub struct Joint {
    // TODO: physics engine and shape mesh and engine rigid_body
    name: String,
    canonical_name: String,
    physics_joint: Option<rapier3d::dynamics::ImpulseJointHandle>,
    states: JointStates,
    origin: NanoemJoint,
}

impl Joint {
    pub fn from_nanoem(
        joint: &NanoemJoint,
        language: nanoem::common::LanguageType,
        rigid_bodies: &RigidBodySet,
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        let mut name = joint.get_name(language).to_owned();
        let mut canonical_name = joint
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Joint{}", joint.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let orientation = joint.orientation;
        let origin = joint.origin;
        let world_transform = f128_to_isometry(origin, orientation);
        let rigid_body_a = usize::try_from(joint.rigid_body_a_index)
            .ok()
            .and_then(|idx| rigid_bodies.get(idx));
        let rigid_body_b = usize::try_from(joint.rigid_body_b_index)
            .ok()
            .and_then(|idx| rigid_bodies.get(idx));
        let physics_joint = if let (Some(rigid_body_a), Some(rigid_body_b)) =
            (rigid_body_a, rigid_body_b)
        {
            let mut physics_joint = rapier3d::dynamics::GenericJoint::default();
            let local_frame_a = rigid_body_a.initial_world_transform.inverse() * world_transform;
            let local_frame_b = rigid_body_b.initial_world_transform.inverse() * world_transform;
            physics_joint.set_local_frame1(local_frame_a);
            physics_joint.set_local_frame2(local_frame_b);

            fn limit(
                joint: &mut rapier3d::dynamics::GenericJoint,
                axis: rapier3d::dynamics::JointAxis,
                min: f32,
                max: f32,
                max_limit: f32,
                stiffness: f32,
            ) {
                if max - min < max_limit && max - min > 0f32 {
                    joint.set_limits(axis, [min, max]);
                    if stiffness > 0f32 {
                        joint.set_motor(axis, 0f32, 0f32, stiffness, 1f32);
                    }
                } else {
                    joint.lock_axes(axis.into());
                }
            }

            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::X,
                joint.linear_lower_limit[0],
                joint.linear_upper_limit[0],
                100.,
                joint.linear_stiffness[0],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::Y,
                joint.linear_lower_limit[1],
                joint.linear_upper_limit[1],
                100.,
                joint.linear_stiffness[1],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::Z,
                joint.linear_lower_limit[2],
                joint.linear_upper_limit[2],
                100.,
                joint.linear_stiffness[2],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngX,
                joint.angular_lower_limit[0],
                joint.angular_upper_limit[0],
                PI * 2.,
                joint.angular_stiffness[0],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngY,
                joint.angular_lower_limit[1],
                joint.angular_upper_limit[1],
                PI * 2.,
                joint.angular_stiffness[1],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngZ,
                joint.angular_lower_limit[2],
                joint.angular_upper_limit[2],
                PI * 2.,
                joint.angular_stiffness[2],
            );
            log::info!("Joint {:?}, {:?}", joint.name_ja, physics_joint);
            Some(physics_engine.impulse_joint_set.insert(
                rigid_body_a.physics_rb.unwrap(),
                rigid_body_b.physics_rb.unwrap(),
                physics_joint,
                true,
            ))
        } else {
            None
        };

        Self {
            name,
            canonical_name,
            physics_joint,
            states: JointStates::default(),
            origin: joint.clone(),
        }
    }

    pub fn enable(&mut self) {
        if !self.states.enabled {
            // TODO: add to physics engine
            self.states.enabled = true;
        }
    }

    pub fn disable(&mut self) {
        if self.states.enabled {
            // TODO: remove from physics engine
            self.states.enabled = false;
        }
    }
}

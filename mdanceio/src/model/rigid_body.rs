use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
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

use super::{bone::BoneSet, Bone, BoneIndex, NanoemRigidBody, RigidBodyIndex};

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
    pub physics_rb: Option<RapierRBHandle>,
    pub initial_world_transform: nalgebra::Isometry3<f32>,
    prev_world_transform: Cell<Option<nalgebra::Isometry3<f32>>>,
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
        if rigid_body.is_bone_relative {
            if let Some(bone) = bones.try_get(rigid_body.bone_index) {
                let bone_origin = bone.origin.origin;
                let offset = nalgebra::Isometry3::translation(
                    bone_origin[0],
                    bone_origin[1],
                    bone_origin[2],
                );
                world_transform = offset * world_transform;
            }
        }
        let initial_world_transform = world_transform;

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
            prev_world_transform: Cell::new(None),
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

    pub fn initialize_simulation(
        &mut self,
        bone: Option<&Bone>,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rb) = self
            .physics_rb
            .and_then(|handle| physics_engine.rigid_body_set.get_mut(handle))
        {
            if let Some(bone) = bone {
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                physics_rb.set_position(skinning_transform * self.initial_world_transform, true);
            } else {
                physics_rb.set_position(self.initial_world_transform, true);
            }
            physics_rb.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rb.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rb.reset_forces(true);
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
        self.synchronize_to_simulation_by_lerp(bone, physics_engine, 1f32)
    }

    pub fn synchronize_to_simulation_by_lerp(
        &self,
        bone: &Bone,
        physics_engine: &mut PhysicsEngine,
        amount: f32,
    ) {
        if let Some(physics_rigid_body) = physics_engine.get_rb_mut(self.physics_rb) {
            if self.is_to_simulation() || physics_rigid_body.is_kinematic() {
                let initial_transform = self.initial_world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                let mut world_transform = skinning_transform * initial_transform;
                if amount != 1f32 {
                    if let Some(prev) = self.prev_world_transform.get() {
                        world_transform = prev.lerp_slerp(&world_transform, amount);
                    }
                }
                self.prev_world_transform.set(Some(world_transform));
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
        let group = rapier3d::geometry::Group::from_bits_truncate(
            0x1u32 << origin_rb.collision_group_id.clamp(0, 15),
        );
        // let orientation = origin_rb.orientation;
        // let pos = f128_to_isometry([0., 0., 0., 0.], orientation);
        collider_builder.map(|builder| {
            builder
                // .position(pos)
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

    pub fn try_get(&self, idx: i32) -> Option<&RigidBody> {
        usize::try_from(idx).ok().and_then(|idx| self.get(idx))
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
    pub fn apply_forces(&mut self, bones: &BoneSet, physics_engine: &mut PhysicsEngine) {
        for rigid_body in self.rigid_bodies.iter_mut() {
            let bone = bones.try_get(rigid_body.origin.bone_index);
            rigid_body.apply_forces(bone, physics_engine);
        }
    }

    pub fn synchronize_to_simulation_by_lerp(
        &self,
        bones: &BoneSet,
        physics_engine: &mut PhysicsEngine,
        amount: f32,
    ) {
        for rigid_body in self.rigid_bodies.iter() {
            let bone = bones.try_get(rigid_body.origin.bone_index);
            if let Some(bone) = bone {
                rigid_body.synchronize_to_simulation_by_lerp(bone, physics_engine, amount);
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

#[test]
fn test_joint_with_kinematic() {
    use nalgebra::{vector, Isometry3, Rotation};
    use rapier3d::prelude::{
        BroadPhase, CCDSolver, ColliderBuilder, ColliderSet, ImpulseJointSet,
        IntegrationParameters, IslandManager, JointAxesMask, JointAxis, MultibodyJointSet,
        NarrowPhase, PhysicsPipeline, RigidBodyBuilder, RigidBodyType,
    };
    let a_pos = Isometry3::new(vector![0.0, -1.0, 0.0], vector![0.0, 0.0, 0.0]);
    let rigid_body_a = RigidBodyBuilder::new(RigidBodyType::KinematicPositionBased)
        .position(a_pos)
        .additional_mass(1f32)
        .build();
    let b_pos = Isometry3::new(vector![0.0, 1.0, 0.0], vector![0.0, 0.0, 0.0]);
    let rigid_body_b = RigidBodyBuilder::new(RigidBodyType::Dynamic)
        .position(b_pos)
        .additional_mass(1f32)
        .build();
    let mut rbs = rapier3d::dynamics::RigidBodySet::new();
    let rb_a = rbs.insert(rigid_body_a);
    let rb_b = rbs.insert(rigid_body_b);
    let mut collider_set = ColliderSet::new();
    let collider_a = ColliderBuilder::ball(0.5).restitution(0.7).build();
    collider_set.insert_with_parent(collider_a, rb_a, &mut rbs);
    let collider_b = ColliderBuilder::ball(0.5).restitution(0.7).build();
    collider_set.insert_with_parent(collider_b, rb_b, &mut rbs);
    let joint_pos = Isometry3::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]);
    let mut physics_joint = rapier3d::dynamics::GenericJoint::default();
    physics_joint.set_local_frame1(a_pos.inverse() * joint_pos);
    physics_joint.set_local_frame2(b_pos.inverse() * joint_pos);
    physics_joint
        .lock_axes(JointAxesMask::X | JointAxesMask::Y | JointAxesMask::Z | JointAxesMask::ANG_Y);
    physics_joint.set_limits(JointAxis::AngX, [-20f32.to_radians(), 20f32.to_radians()]);
    physics_joint.set_limits(JointAxis::AngZ, [-20f32.to_radians(), 20f32.to_radians()]);
    let mut impulse_joint_set = ImpulseJointSet::new();
    impulse_joint_set.insert(rb_a, rb_b, physics_joint, true);
    println!("{:?}", rbs.get(rb_a).unwrap().position());
    println!("{:?}", rbs.get(rb_b).unwrap().position());
    let mut pipeline = PhysicsPipeline::new();
    let gravity = vector![0.0, -9.81, 0.0];
    let integration_parameters = IntegrationParameters::default();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = BroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();
    println!("Stepping...");
    pipeline.step(
        &gravity,
        &integration_parameters,
        &mut island_manager,
        &mut broad_phase,
        &mut narrow_phase,
        &mut rbs,
        &mut collider_set,
        &mut impulse_joint_set,
        &mut multibody_joint_set,
        &mut ccd_solver,
        None,
        &(),
        &(),
    );
    println!("{:?}", rbs.get(rb_a).unwrap().position());
    println!("{:?}", rbs.get(rb_b).unwrap().position());
    rbs.get_mut(rb_a)
        .unwrap()
        .set_translation(vector![0.0, -4.0, 1.0], true);
    println!("Stepping...");
    pipeline.step(
        &gravity,
        &integration_parameters,
        &mut island_manager,
        &mut broad_phase,
        &mut narrow_phase,
        &mut rbs,
        &mut collider_set,
        &mut impulse_joint_set,
        &mut multibody_joint_set,
        &mut ccd_solver,
        None,
        &(),
        &(),
    );
    println!("{:?}", rbs.get(rb_a).unwrap().position());
    println!("{:?}", rbs.get(rb_b).unwrap().position());
    rbs.get_mut(rb_a).unwrap().set_rotation(
        Rotation::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.2f32).into(),
        true,
    );
    println!("Stepping...");
    pipeline.step(
        &gravity,
        &integration_parameters,
        &mut island_manager,
        &mut broad_phase,
        &mut narrow_phase,
        &mut rbs,
        &mut collider_set,
        &mut impulse_joint_set,
        &mut multibody_joint_set,
        &mut ccd_solver,
        None,
        &(),
        &(),
    );
    println!("{:?}", rbs.get(rb_a).unwrap().position());
    println!("{:?}", rbs.get(rb_b).unwrap().position());

    // rbs.get_mut(rb_a)
    //     .unwrap()
    //     .set_translation(vector![0.0, -10.0, 0.0], true);
    rbs.get_mut(rb_a).unwrap().set_rotation(
        Rotation::from_axis_angle(&nalgebra::Vector3::y_axis(), -0.8f32).into(),
        true,
    );
    println!("Stepping...");
    pipeline.step(
        &gravity,
        &integration_parameters,
        &mut island_manager,
        &mut broad_phase,
        &mut narrow_phase,
        &mut rbs,
        &mut collider_set,
        &mut impulse_joint_set,
        &mut multibody_joint_set,
        &mut ccd_solver,
        None,
        &(),
        &(),
    );
    println!("{:?}", rbs.get(rb_a).unwrap().position());
    println!("{:?}", rbs.get(rb_b).unwrap().position());
}

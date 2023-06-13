use std::f32::consts::PI;

use nanoem::common::LanguageType;
use rapier3d::prelude::JointAxesMask;

use crate::{physics_engine::PhysicsEngine, utils::f128_to_isometry};

use super::{rigid_body::RigidBodySet, NanoemJoint};

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
        let rigid_body_a = rigid_bodies.try_get(joint.rigid_body_a_index);
        let rigid_body_b = rigid_bodies.try_get(joint.rigid_body_b_index);
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
                        joint.set_motor(axis, 0f32, 0f32, stiffness, 10f32);
                    } else if JointAxesMask::ANG_AXES.contains(axis.into()) {
                        joint.set_motor(axis, 0f32, 0f32, 10000., 10000.);
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

pub struct JointSet {
    joints: Vec<Joint>,
}

impl JointSet {
    pub fn new(
        joints: &[NanoemJoint],
        language_type: LanguageType,
        rigid_bodies: &RigidBodySet,
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        let joints = joints
            .iter()
            .map(|joint| Joint::from_nanoem(joint, language_type, rigid_bodies, physics_engine))
            .collect();
        Self { joints }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Joint> {
        self.joints.iter_mut()
    }
}

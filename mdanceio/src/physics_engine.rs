use cgmath::Matrix4;
use rapier3d::prelude::{
    BroadPhase, CCDSolver, ColliderSet, DebugRenderMode, DebugRenderPipeline, DebugRenderStyle,
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase,
    PhysicsPipeline, RigidBody, RigidBodyHandle, RigidBodySet,
};

use crate::graphics::physics_debug::PhysicsDrawerBuilder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigidBodyFollowBone {
    Skip,
    Perform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationTiming {
    Before,
    After,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationMode {
    Disable,
    EnableAnytime,
    EnablePlaying,
    EnableTracing,
}

pub struct PhysicsEngine {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    gravity: nalgebra::Vector3<f32>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    pub simulation_mode: SimulationMode,
    dt_residual: f32,
    debug_render_pipeline: DebugRenderPipeline,
    debug_drawer_builder: Option<PhysicsDrawerBuilder>,
}

impl Default for PhysicsEngine {
    fn default() -> Self {
        let mut it = IntegrationParameters::default();
        it.set_inv_dt(120f32);
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            gravity: nalgebra::vector![0f32, -9.8f32, 0f32],
            integration_parameters: it,
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            simulation_mode: SimulationMode::Disable,
            dt_residual: 0f32,
            debug_render_pipeline: DebugRenderPipeline::new(
                DebugRenderStyle::default(),
                DebugRenderMode::IMPULSE_JOINTS | DebugRenderMode::COLLIDER_SHAPES | DebugRenderMode::RIGID_BODY_AXES,
            ),
            debug_drawer_builder: None,
        }
    }
}

impl PhysicsEngine {
    pub fn new(debug_drawer_builder: Option<PhysicsDrawerBuilder>) -> Self {
        Self {
            debug_drawer_builder,
            ..Default::default()
        }
    }

    pub fn step(&mut self, delta: f32) {
        let dt = self.integration_parameters.dt;
        let it_count = (delta + self.dt_residual).div_euclid(dt) as u32;
        self.dt_residual = (delta + self.dt_residual).rem_euclid(dt);
        for _ in 0..it_count {
            self.physics_pipeline.step(
                &self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                &mut self.ccd_solver,
                None,
                &(),
                &(),
            )
        }
    }

    pub fn reset(&mut self) {}

    pub fn remove_rb(&mut self, handle: RigidBodyHandle) -> Option<RigidBody> {
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        )
    }

    pub fn get_rb(&self, handle: Option<RigidBodyHandle>) -> Option<&RigidBody> {
        handle.and_then(|handle| self.rigid_body_set.get(handle))
    }

    pub fn get_rb_mut(&mut self, handle: Option<RigidBodyHandle>) -> Option<&mut RigidBody> {
        handle.and_then(|handle| self.rigid_body_set.get_mut(handle))
    }

    pub fn debug_draw(
        &mut self,
        view_projection: Matrix4<f32>,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if let Some(drawer_builder) = &mut self.debug_drawer_builder {
            let mut drawer = drawer_builder.build(view_projection, view, device, queue);
            self.debug_render_pipeline.render_joints(
                &mut drawer,
                &self.rigid_body_set,
                &self.impulse_joint_set,
                &self.multibody_joint_set,
            );
            self.debug_render_pipeline.render_colliders(
                &mut drawer,
                &self.rigid_body_set,
                &self.collider_set,
            );
            self.debug_render_pipeline.render_rigid_bodies(&mut drawer, &self.rigid_body_set);
        }
    }
}

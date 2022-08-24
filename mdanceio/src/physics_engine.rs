use rapier3d::prelude::{
    BroadPhase, CCDSolver, ColliderSet, ImpulseJointSet, IntegrationParameters, IslandManager,
    MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};

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
}

impl Default for PhysicsEngine {
    fn default() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            gravity: nalgebra::vector![0f32, -9.8f32, 0f32],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            simulation_mode: SimulationMode::Disable,
        }
    }
}

impl PhysicsEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(&mut self, delta: f32) {
        self.integration_parameters.dt = delta;
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
            &(),
            &(),
        )
    }

    pub fn reset(&mut self) {}
}

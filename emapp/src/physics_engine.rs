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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugDrawType {
    Wireframe,
    Aabb,
    ContactPoints,
    Constraints,
    ConstraintLimits,
}

pub struct PhysicsEngine {
    // TODO
}
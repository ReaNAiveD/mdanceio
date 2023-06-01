pub type NanoemModel = nanoem::model::Model;
pub type NanoemVertex = nanoem::model::ModelVertex;
pub type NanoemBone = nanoem::model::ModelBone;
pub type NanoemMaterial = nanoem::model::ModelMaterial;
pub type NanoemMorph = nanoem::model::ModelMorph;
pub type NanoemConstraint = nanoem::model::ModelConstraint;
pub type NanoemConstraintJoint = nanoem::model::ModelConstraintJoint;
pub type NanoemLabel = nanoem::model::ModelLabel;
pub type NanoemRigidBody = nanoem::model::ModelRigidBody;
pub type NanoemJoint = nanoem::model::ModelJoint;
pub type NanoemSoftBody = nanoem::model::ModelSoftBody;
pub type NanoemTexture = nanoem::model::ModelTexture;
pub type VertexIndex = usize;
pub type BoneIndex = usize;
pub type MaterialIndex = usize;
pub type MorphIndex = usize;
pub type ConstraintIndex = usize;
pub type LabelIndex = usize;
pub type RigidBodyIndex = usize;
pub type JointIndex = usize;
pub type SoftBodyIndex = usize;

pub mod model;

pub use model::{Bone, Material, Model, Morph, Vertex, VertexUnit};

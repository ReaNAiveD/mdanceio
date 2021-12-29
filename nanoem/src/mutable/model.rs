use std::{cell::RefCell, rc::Rc};

use crate::model::{
    Model, ModelBone, ModelConstraint, ModelConstraintJoint, ModelJoint, ModelLabel,
    ModelLabelItem, ModelMaterial, ModelMorph, ModelMorphBone, ModelMorphFlip, ModelMorphGroup,
    ModelMorphImpulse, ModelMorphMaterial, ModelMorphUv, ModelMorphVertex, ModelRigidBody,
    ModelSoftBody, ModelSoftBodyAnchor, ModelTexture, ModelVertex,
};

struct MutableBaseModelObject {
    is_reference: bool,
    is_in_model: bool,
}

impl MutableBaseModelObject {
    fn can_delete(&self) -> bool {
        self.is_in_model == false
    }
}

struct MutableBaseLabelObject {
    is_reference: bool,
    is_in_label: bool,
}

impl MutableBaseLabelObject {
    fn can_delete(&self) -> bool {
        self.is_in_label == false
    }
}

struct MutableBaseMorphObject {
    is_reference: bool,
    is_in_morph: bool,
}

impl MutableBaseMorphObject {
    fn can_delete(&self) -> bool {
        self.is_in_morph == false
    }
}

struct MutableBaseSoftBodyObject {
    is_reference: bool,
    is_in_soft_body: bool,
}

impl MutableBaseSoftBodyObject {
    fn can_delete(&self) -> bool {
        self.is_in_soft_body == false
    }
}

struct MutableModel {
    origin: Rc<RefCell<Model>>,
    is_in_reference: bool,
    num_allocated_vertices: usize,
    num_allocated_vertex_indices: usize,
    num_allocated_materials: usize,
    num_allocated_bones: usize,
    num_allocated_constraints: usize,
    num_allocated_textures: usize,
    num_allocated_morphs: usize,
    num_allocated_labels: usize,
    num_allocated_rigid_bodies: usize,
    num_allocated_joints: usize,
    num_allocated_soft_bodies: usize,
}

struct MutableModelVertex {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelVertex>>,
}

struct MutableModelMaterial {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelMaterial>>,
}

struct MutableModelBone {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelBone>>,
}

struct MutableModelConstraintJoint {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelConstraintJoint>>,
}

struct MutableModelConstraint {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelConstraint>>,
}

struct MutableModelMorphGroup {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphGroup>>,
}

struct MutableModelMorphVertex {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphVertex>>,
}

struct MutableModelMorphBone {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphBone>>,
}

struct MutableModelMorphUv {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphUv>>,
}

struct MutableModelMorphMaterial {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphMaterial>>,
}

struct MutableModelMorphFlip {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphFlip>>,
}

struct MutableModelMorphImpulse {
    base: MutableBaseMorphObject,
    origin: Rc<RefCell<ModelMorphImpulse>>,
}

struct MutableModelMorph {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelMorph>>,
}

struct MutableModelLabelItem {
    base: MutableBaseLabelObject,
    origin: Rc<RefCell<ModelLabelItem>>,
}

struct MutableModelLabel {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelLabel>>,
    num_allocated_items: usize,
}

struct MutableModelRigidBody {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelRigidBody>>,
}

struct MutableModelJoint {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelJoint>>,
}

struct MutableModelSoftBodyAnchor {
    base: MutableBaseSoftBodyObject,
    origin: Rc<RefCell<ModelSoftBodyAnchor>>,
}

struct MutableModelSoftBody {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelSoftBody>>,
    num_allocated_anchors: usize,
    num_allocated_pin_vertex_indices: usize,
}

struct MutableModelTexture {
    base: MutableBaseModelObject,
    origin: Rc<RefCell<ModelTexture>>,
}

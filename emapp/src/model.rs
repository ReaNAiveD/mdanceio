use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::{Rc, Weak},
};

use bytemuck::{Pod, Zeroable};
use cgmath::{ElementWise, Matrix4, Quaternion, SquareMatrix, Vector3, Vector4, Zero};
use nanoem::model::ModelFormatType;
use par::shape::ShapesMesh;
use wgpu::{AddressMode, Buffer};

use crate::{
    bounding_box::BoundingBox,
    camera::Camera,
    drawable::{DrawType, Drawable},
    effect::{Effect, IEffect},
    error::Error,
    forward::LineVertexUnit,
    image_loader::Image,
    image_view::ImageView,
    internal::LinearDrawer,
    model_object_selection::ModelObjectSelection,
    model_program_bundle::{ModelProgramBundle, ObjectTechnique},
    pass,
    project::Project,
    undo::UndoStack,
    uri::Uri,
};

pub type NanoemModel = nanoem::model::Model<
    Vertex,
    Material,
    Bone,
    Constraint,
    Morph,
    Label,
    RigidBody,
    Joint,
    SoftBody,
>;
pub type NanoemVertex = nanoem::model::ModelVertex<Vertex>;
pub type NanoemBone = nanoem::model::ModelBone<Bone, Constraint>;
pub type NanoemMaterial = nanoem::model::ModelMaterial<Material>;
pub type NanoemMorph = nanoem::model::ModelMorph<Morph>;
pub type NanoemConstraint = nanoem::model::ModelConstraint<Constraint>;
pub type NanoemConstraintJoint = nanoem::model::ModelConstraintJoint<()>;
pub type NanoemLabel = nanoem::model::ModelLabel<Label, Bone, Constraint, Morph>;
pub type NanoemRigidBody = nanoem::model::ModelRigidBody<RigidBody>;
pub type NanoemJoint = nanoem::model::ModelJoint<Joint>;
pub type NanoemSoftBody = nanoem::model::ModelSoftBody<SoftBody>;

pub trait SkinDeformer {
    // TODO
}

pub struct BindPose {
    // TODO
}

pub trait Gizmo {
    // TODO
}

pub trait VertexWeightPainter {
    // TODO
}

pub enum AxisType {
    None,
    Center,
    X,
    Y,
    Z,
}

pub enum EditActionType {
    None,
    SelectModelObject,
    PaintVertexWeight,
    CreateTriangleVertices,
    CreateParentBone,
    CreateTargetBone,
}

pub enum TransformCoordinateType {
    Global,
    Local,
}

pub enum ResetType {
    TranslationAxisX,
    TranslationAxisY,
    TranslationAxisZ,
    Orientation,
    OrientationAngleX,
    OrientationAngleY,
    OrientationAngleZ,
}

struct LoadingImageItem {
    file_uri: Uri,
    filename: String,
    wrap: AddressMode,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexUnit {
    position: [f32; 4],
    normal: [f32; 4],
    texcoord: [f32; 4],
    edge: [f32; 4],
    uva: [[f32; 4]; 4],
    weights: [f32; 4],
    indices: [f32; 4],
    info: [f32; 4], /* type,vertexIndex,edgeSize,padding */
}

impl From<VertexSimd> for VertexUnit {
    fn from(simd: VertexSimd) -> Self {
        Self {
            position: simd.origin.into(),
            normal: simd.normal.into(),
            texcoord: simd.texcoord.into(),
            edge: simd.origin.into(),
            uva: [
                simd.origin_uva[0].into(),
                simd.origin_uva[1].into(),
                simd.origin_uva[2].into(),
                simd.origin_uva[3].into(),
            ],
            weights: simd.weights.into(),
            indices: simd.indices.into(),
            info: simd.info.into(),
        }
    }
}

pub struct NewModelDescription {
    name: HashMap<nanoem::common::LanguageType, String>,
    comment: HashMap<nanoem::common::LanguageType, String>,
}

pub enum ImportFileType {
    None,
    WaveFrontObj,
    DirectX,
    Metasequoia,
}

pub struct ImportDescription {
    file_uri: Uri,
    name: HashMap<nanoem::common::LanguageType, String>,
    comment: HashMap<nanoem::common::LanguageType, String>,
    transform: Matrix4<f32>,
    file_type: ImportFileType,
}

pub struct ExportDescription {
    transform: Matrix4<f32>,
}

struct ParallelSkinningTaskData {
    draw_type: DrawType,
    edge_size_scale_factor: f32,
    bone_indices: HashMap<Rc<RefCell<NanoemMaterial>>, HashMap<i32, i32>>,
    output: u8,
    materials: Rc<RefCell<[NanoemMaterial]>>,
    vertices: Rc<RefCell<[NanoemMaterial]>>,
    num_vertices: usize,
}

struct DrawArrayBuffer {
    vertices: Vec<LineVertexUnit>,
    buffer: Buffer,
}

struct DrawIndexedBuffer {
    vertices: Vec<LineVertexUnit>,
    active_indices: Vec<u32>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    active_index_buffer: Buffer,
    color: Vector4<f32>,
}

struct OffscreenPassiveRenderTargetEffect {
    passive_effect: Rc<RefCell<dyn IEffect>>,
    enabled: bool,
}

pub struct Model {
    handle: u16,
    // camera: Rc<RefCell<dyn Camera>>,
    // selection: Rc<RefCell<dyn ModelObjectSelection>>,
    // drawer: Box<LinearDrawer>,
    // skin_deformer: Rc<RefCell<dyn SkinDeformer>>,
    // gizmo: Rc<RefCell<dyn Gizmo>>,
    // vertex_weight_painter: Rc<RefCell<dyn VertexWeightPainter>>,
    // offscreen_passive_render_target_effects: HashMap<String, OffscreenPassiveRenderTargetEffect>,
    // draw_all_vertex_normals: DrawArrayBuffer,
    // draw_all_vertex_points: DrawArrayBuffer,
    // draw_all_vertex_faces: DrawIndexedBuffer,
    // draw_all_vertex_weights: DrawIndexedBuffer,
    // draw_rigid_body: HashMap<Rc<RefCell<ShapesMesh>>, DrawIndexedBuffer>,
    // draw_joint: HashMap<Rc<RefCell<ShapesMesh>>, DrawIndexedBuffer>,
    opaque: Box<NanoemModel>,
    // undo_stack: Box<UndoStack>,
    // editing_undo_stack: Box<UndoStack>,
    // active_morph_ptr: HashMap<nanoem::model::ModelMorphCategory, Rc<RefCell<NanoemMorph>>>,
    // active_constraint_ptr: Rc<RefCell<NanoemConstraint>>,
    // active_material_ptr: Rc<RefCell<NanoemMaterial>>,
    // hovered_bone_ptr: Rc<RefCell<NanoemBone>>,
    // vertex_buffer_data: Vec<u8>,
    // face_states: Vec<u32>,
    // active_bone_pair_ptr: (Rc<RefCell<NanoemBone>>, Rc<RefCell<NanoemBone>>),
    // active_effect_pair_ptr: (Rc<RefCell<dyn IEffect>>, Rc<RefCell<dyn IEffect>>),
    // screen_image: Image,
    // loading_image_items: Vec<LoadingImageItem>,
    // image_map: HashMap<String, Image>,
    bone_index_hash_map: HashMap<*const RefCell<NanoemMaterial>, HashMap<i32, i32>>,
    bones: HashMap<String, Weak<RefCell<NanoemBone>>>,
    morphs: HashMap<String, Weak<RefCell<NanoemMorph>>>,
    constraints: HashMap<*const RefCell<NanoemBone>, Weak<RefCell<NanoemConstraint>>>,
    // redo_bone_names: Vec<String>,
    // redo_morph_names: Vec<String>,
    // outside_parents: HashMap<Rc<RefCell<NanoemBone>>, (String, String)>,
    // image_uris: HashMap<String, Uri>,
    // attachment_uris: HashMap<String, Uri>,
    // bone_bound_rigid_bodies: HashMap<Rc<RefCell<NanoemBone>>, Rc<RefCell<NanoemRigidBody>>>,
    constraint_joint_bones: HashMap<*const RefCell<NanoemBone>, Weak<RefCell<NanoemConstraint>>>,
    inherent_bones: HashMap<*const RefCell<NanoemBone>, HashSet<*const RefCell<NanoemBone>>>,
    constraint_effector_bones: HashSet<*const RefCell<NanoemBone>>,
    parent_bone_tree: HashMap<*const RefCell<NanoemBone>, Vec<Weak<RefCell<NanoemBone>>>>,
    shared_fallback_bone: Rc<RefCell<Bone>>,
    // bounding_box: BoundingBox,
    // // UserData m_userData;
    // annotations: HashMap<String, String>,
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffer: wgpu::Buffer,
    // edge_color: Vector4<f32>,
    // transform_axis_type: AxisType,
    // edit_action_type: EditActionType,
    // transform_coordinate_type: TransformCoordinateType,
    // file_uri: Uri,
    name: String,
    comment: String,
    canonical_name: String,
    // states: u32,
    // edge_size_scale_factor: f32,
    opacity: f32,
    // // void *m_dispatchParallelTaskQueue
    count_vertex_skinning_needed: i32,
    stage_vertex_buffer_index: i32,
}

impl Model {
    pub const INITIAL_WORLD_MATRIX: Matrix4<f32> = Matrix4::new(
        1f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32,
        1f32,
    );
    pub const DEFAULT_CM_SCALE_FACTOR: f32 = 0.1259496f32;
    pub const DEFAULT_MODEL_CORRECTION_HEIGHT: f32 = -2f32;
    pub const PMX_FORMAT_EXTENSION: &'static str = "pmx";
    pub const PMD_FORMAT_EXTENSION: &'static str = "pmd";

    pub const DRAW_BONE_CONNECTION_THICKNESS: f32 = 1.0f32;
    pub const DRAW_VERTEX_NORMAL_SCALE_FACTOR: f32 = 0.0f32;
    pub const MAX_BONE_UNIFORMS: i32 = 55;

    pub fn new(project: &Project, handle: u16) -> Self {
        Self {
            handle,
            // camera: todo!(),
            // selection: todo!(),
            // drawer: todo!(),
            // skin_deformer: todo!(),
            // gizmo: todo!(),
            // vertex_weight_painter: todo!(),
            // offscreen_passive_render_target_effects: todo!(),
            // draw_all_vertex_normals: todo!(),
            // draw_all_vertex_points: todo!(),
            // draw_all_vertex_faces: todo!(),
            // draw_all_vertex_weights: todo!(),
            // draw_rigid_body: todo!(),
            // draw_joint: todo!(),
            opaque: todo!(),
            // undo_stack: todo!(),
            // editing_undo_stack: todo!(),
            // active_morph_ptr: todo!(),
            // active_constraint_ptr: todo!(),
            // active_material_ptr: todo!(),
            // hovered_bone_ptr: todo!(),
            // vertex_buffer_data: todo!(),
            // face_states: todo!(),
            // active_bone_pair_ptr: todo!(),
            // active_effect_pair_ptr: todo!(),
            // screen_image: todo!(),
            // loading_image_items: todo!(),
            // image_map: todo!(),
            bone_index_hash_map: HashMap::new(),
            bones: HashMap::new(),
            morphs: HashMap::new(),
            constraints: HashMap::new(),
            // redo_bone_names: todo!(),
            // redo_morph_names: todo!(),
            // outside_parents: todo!(),
            // image_uris: todo!(),
            // attachment_uris: todo!(),
            // bone_bound_rigid_bodies: todo!(),
            constraint_joint_bones: HashMap::new(),
            inherent_bones: HashMap::new(),
            constraint_effector_bones: HashSet::new(),
            parent_bone_tree: HashMap::new(),
            shared_fallback_bone: Rc::new(RefCell::new(Bone::new(
                "SharedFallbackBone",
                "SharedFallbackBone",
            ))),
            // bounding_box: todo!(),
            // annotations: todo!(),
            vertex_buffers: todo!(),
            index_buffer: todo!(),
            // edge_color: todo!(),
            // transform_axis_type: todo!(),
            // edit_action_type: todo!(),
            // transform_coordinate_type: todo!(),
            // file_uri: todo!(),
            name: "".to_owned(),
            comment: "".to_owned(),
            canonical_name: "".to_owned(),
            // states: todo!(),
            // edge_size_scale_factor: todo!(),
            opacity: 1.0f32,
            count_vertex_skinning_needed: 0,
            stage_vertex_buffer_index: 0,
        }
    }

    pub fn new_from_bytes(bytes: &[u8], project: &Project, handle: u16, device: &wgpu::Device) -> Result<Self, Error> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        let mut nanoem_model = Box::new(NanoemModel::default());
        match nanoem_model.load_from_buffer(&mut buffer) {
            Ok(_) => {
                let opaque = nanoem_model;
                let language = project.parse_language();
                let mut name = opaque.get_name(language).to_owned();
                let comment = opaque.get_comment(language).to_owned();
                let canonical_name = opaque
                    .get_name(nanoem::common::LanguageType::default())
                    .to_owned();
                if name.is_empty() {
                    name = canonical_name.clone();
                }

                let shared_fallback_bone = Rc::new(RefCell::new(Bone::new(
                    "SharedFallbackBone",
                    "SharedFallbackBone",
                )));
                
                let vertices = opaque.get_all_vertex_objects();
                for vertex in vertices {
                    let _ = Vertex::new_bind(&mut vertex.borrow_mut(), &opaque);
                }
                let materials = opaque.get_all_material_objects();
                let indices = opaque.get_all_vertex_indices();
                let mut index_offset = 0;
                for material_rc in materials {
                    let material = &mut material_rc.borrow_mut();
                    Material::new_bind(material, project.shared_fallback_image(), language);
                    let num_indices = material.get_num_vertex_indices();
                    for i in index_offset..(index_offset + num_indices) {
                        let vertex_index = indices[i].borrow();
                        if let Some(vertex) = vertices.get(vertex_index.clone() as usize) {
                            if let Some(user_data) = vertex.borrow_mut().get_user_data() {
                                user_data.borrow_mut().set_material(material_rc);
                            }
                        }
                    }
                    index_offset += num_indices;
                }
                let mut constraint_joint_bones = HashMap::new();
                let mut constraint_effector_bones = HashSet::new();
                let mut inherent_bones = HashMap::new();
                let mut bones = HashMap::new();
                for bone_rc in opaque.get_all_bone_objects() {
                    {
                        let bone = &mut bone_rc.borrow_mut();
                        Bone::new_bind(bone, language);
                    }
                    let bone = &bone_rc.borrow();
                    if let Some(constraint) = bone.get_constraint_object() {
                        let target_bone = opaque
                            .get_one_bone_object(constraint.borrow().get_target_bone_index());
                        Constraint::new_bind(&mut constraint.borrow_mut(), target_bone, language);
                        for joint in constraint.borrow().get_all_joint_objects() {
                            if let Some(bone) = opaque
                                .get_one_bone_object(joint.borrow().get_bone_index())
                            {
                                constraint_joint_bones
                                    .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                            }
                        }
                        if let Some(effector_bone) = opaque
                            .get_one_bone_object(constraint.borrow().get_effector_bone_index())
                        {
                            constraint_effector_bones
                                .insert(Rc::downgrade(effector_bone).into_raw());
                        }
                    }
                    if bone.has_inherent_orientation() || bone.has_inherent_translation() {
                        if let Some(parent_bone) = opaque
                            .get_one_bone_object(bone.get_parent_inherent_bone_index())
                        {
                            inherent_bones
                                .entry(Rc::downgrade(parent_bone).into_raw())
                                .or_insert(HashSet::new())
                                .insert(Rc::downgrade(bone_rc).into_raw());
                        }
                    }
                    for language in nanoem::common::LanguageType::all() {
                        bones.insert(
                            bone.get_name(language.clone()).to_owned(),
                            Rc::downgrade(bone_rc),
                        );
                    }
                }
                let mut parent_bone_tree = HashMap::new();
                for bone in opaque.get_all_bone_objects() {
                    if let Some(parent_bone) = opaque
                        .get_one_bone_object(bone.borrow().get_parent_bone_index())
                    {
                        parent_bone_tree
                            .entry(Rc::downgrade(parent_bone).into_raw())
                            .or_insert(vec![])
                            .push(Rc::downgrade(bone));
                    }
                }
                if let Some(first_bone) = opaque.get_all_bone_objects().get(0) {
                    // TODO: set into selection
                }
                let nanoem_constraints = opaque.get_all_constraint_objects();
                let mut constraints = HashMap::new();
                if nanoem_constraints.len() > 0 {
                    for constraint in nanoem_constraints {
                        let target_bone = opaque
                            .get_one_bone_object(constraint.borrow().get_target_bone_index());
                        Constraint::new_bind(&mut constraint.borrow_mut(), target_bone, language);
                        for joint in constraint.borrow().get_all_joint_objects() {
                            if let Some(bone) = opaque
                                .get_one_bone_object(joint.borrow().get_bone_index())
                            {
                                constraint_joint_bones
                                    .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                            }
                        }
                        if let Some(effector_bone) = opaque
                            .get_one_bone_object(constraint.borrow().get_effector_bone_index())
                        {
                            constraint_effector_bones
                                .insert(Rc::downgrade(effector_bone).into_raw());
                        }
                        if let Some(target_bone) = target_bone {
                            constraints.insert(
                                Rc::downgrade(&target_bone).into_raw(),
                                Rc::downgrade(constraint),
                            );
                        }
                    }
                } else {
                    for bone in opaque.get_all_bone_objects() {
                        if let Some(constraint) = bone.borrow().get_constraint_object() {
                            constraints
                                .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                        }
                    }
                }
                let mut bone_set: HashSet<*const RefCell<NanoemBone>> = HashSet::new();
                let mut morphs = HashMap::new();
                for morph in opaque.get_all_morph_objects() {
                    Morph::new_bind(&mut morph.borrow_mut(), language);
                    if let nanoem::model::ModelMorphType::Vertex = morph.borrow().get_type() {
                        if let nanoem::model::ModelMorphU::VERTICES(morph_vertices) = morph.borrow().get_u()
                        {
                            for morph_vertex in morph_vertices {
                                if let Some(vertex) = opaque
                                    .get_one_vertex_object(morph_vertex.get_vertex_index())
                                {
                                    for bone_index in vertex.borrow().get_bone_indices() {
                                        if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                            bone_set.insert(Rc::downgrade(bone).into_raw());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let category = morph.borrow().get_category();
                    // TODO: set active category
                    for language in nanoem::common::LanguageType::all() {
                        morphs.insert(
                            morph.borrow().get_name(language.clone()).to_owned(),
                            Rc::downgrade(morph),
                        );
                    }
                }
                for label in opaque.get_all_label_objects() {
                    Label::new_bind(&mut label.borrow_mut(), language);
                }
                for rigid_body in opaque.get_all_rigid_body_objects() {
                    let is_dynamic =
                        if let nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation =
                            rigid_body.borrow().get_transform_type()
                        {
                            false
                        } else {
                            true
                        };
                    let is_morph = if let Some(bone) = opaque
                        .get_one_bone_object(rigid_body.borrow().get_bone_index())
                    {
                        is_dynamic && bone_set.contains(&Rc::downgrade(bone).into_raw())
                    } else {
                        false
                    };
                    RigidBody::new_bind(&mut rigid_body.borrow_mut(), language);
                    // TODO: initializeTransformFeedback
                }
                for joint in opaque.get_all_joint_objects() {
                    Joint::new_bind(&mut joint.borrow_mut(), language);
                }
                for soft_body in opaque.get_all_soft_body_objects() {
                    SoftBody::new_bind(&mut soft_body.borrow_mut(), language);
                }
                for nanoem_vertex in opaque.get_all_vertex_objects() {
                    if let Some(vertex) = nanoem_vertex.borrow().get_user_data() {
                        vertex
                            .borrow_mut()
                            .setup_bone_binding(&nanoem_vertex.borrow(), &opaque, shared_fallback_bone.clone());
                    }
                }
                // split_bones_per_material();
                
                let mut offset: usize = 0;
                let mut unique_bone_index_per_material = 0;
                let mut references: HashMap<i32, HashSet<*const RefCell<NanoemVertex>>> = HashMap::new();
                let mut index_hash = HashMap::new();
                let materials = opaque.get_all_material_objects();
                let vertices = opaque.get_all_vertex_objects();
                let indices = opaque.get_all_vertex_indices();
                let mut bone_index_hash_map = HashMap::new();
                let mut count_vertex_skinning_needed = 0;
                for material in materials {
                    let num_indices = material.borrow().get_num_vertex_indices();
                    for j in offset..offset + num_indices {
                        let vertex_index = &indices[j];
                        let vertex = &vertices[*vertex_index.borrow() as usize];
                        for bone_index in vertex.borrow().get_bone_indices() {
                            if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                let bone_index = bone.borrow().get_index();
                                if bone_index >= 0 {
                                    if !index_hash.contains_key(&bone_index) {
                                        index_hash.insert(bone_index, unique_bone_index_per_material);
                                        unique_bone_index_per_material += 1;
                                    }
                                    references
                                        .entry(bone_index)
                                        .or_insert(HashSet::new())
                                        .insert(Rc::downgrade(vertex).into_raw());
                                }
                            }
                        }
                    }
                    if !index_hash.is_empty() {
                        if references.len() > Self::MAX_BONE_UNIFORMS as usize {
                            let mut vertex_list: Vec<Rc<RefCell<NanoemVertex>>> = vec![];
                            let mut bone_vertex_list: Vec<(i32, Vec<Rc<RefCell<NanoemVertex>>>)> = vec![];
                            for vertex_reference in &references {
                                vertex_list.clear();
                                for vertex in vertex_reference.1 {
                                    let vertex = unsafe { Weak::from_raw(*vertex) };
                                    vertex_list.push(vertex.upgrade().unwrap())
                                }
                                bone_vertex_list.push((*vertex_reference.0, vertex_list.clone()));
                            }
                            bone_vertex_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                            index_hash.clear();
                            for j in 0..Self::MAX_BONE_UNIFORMS {
                                let pair = &bone_vertex_list[j as usize];
                                index_hash.insert(pair.0, j);
                            }
                            for pair in &mut bone_vertex_list {
                                let all_vertices = &mut pair.1;
                                for nanoem_vertex in all_vertices {
                                    if let Some(vertex) = nanoem_vertex.borrow_mut().get_user_data() {
                                        vertex.borrow_mut().set_skinning_enabled(true);
                                    }
                                    count_vertex_skinning_needed += 1;
                                }
                            }
                        }
                        bone_index_hash_map
                            .insert(Rc::downgrade(material).as_ptr(), index_hash.clone());
                    }
                    for it in &mut references {
                        it.1.clear()
                    }
                    offset += num_indices;
                    unique_bone_index_per_material = 0;
                    index_hash.clear();
                    references.clear();
                }
                
                let vertices = opaque.get_all_vertex_objects();
                log::trace!("Len(vertices): {}", vertices.len());
                let mut vertex_buffer_data: Vec<VertexUnit> = vec![];
                for vertex in vertices {
                    let vertex = vertex.borrow();
                    let vertex = vertex.get_user_data().as_ref().unwrap();
                    vertex_buffer_data.push(vertex.clone().borrow().simd.clone().into());
                }
                log::trace!("Len(vertex_buffer): {}", vertex_buffer_data.len());
                let vertex_buffer_even = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor{
                    label: Some(format!("Model/{}/VertexBuffer/Even", canonical_name).as_str()),
                    contents: bytemuck::cast_slice(&vertex_buffer_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let vertex_buffer_odd = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor{
                    label: Some(format!("Model/{}/VertexBuffer/Odd", canonical_name).as_str()),
                    contents: bytemuck::cast_slice(&vertex_buffer_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let vertex_buffers = [vertex_buffer_even, vertex_buffer_odd];
                let indices = opaque.get_all_vertex_indices();
                let index_buffer_data: Vec<u32> = indices.iter().map(|rc| rc.borrow().clone()).collect();
                log::trace!("Len(index_buffer): {}", index_buffer_data.len());
                let index_buffer = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                    label: Some(format!("Model/{}/IndexBuffer", canonical_name).as_str()),
                    contents: bytemuck::cast_slice(&index_buffer_data),
                    usage: wgpu::BufferUsages::INDEX,
                });

                Ok(Self {
                    handle,
                    opaque,
                    bone_index_hash_map,
                    bones,
                    morphs,
                    constraints,
                    constraint_joint_bones,
                    inherent_bones,
                    constraint_effector_bones,
                    parent_bone_tree,
                    shared_fallback_bone,
                    vertex_buffers,
                    index_buffer,
                    name,
                    comment,
                    canonical_name,
                    opacity: 1.0f32,
                    count_vertex_skinning_needed,
                    stage_vertex_buffer_index: 0,
                })
            }
            Err(status) => Err(Error::from_nanoem("Cannot load the model: ", status)),
        }
    }

    pub fn loadable_extensions() -> Vec<&'static str> {
        vec![Self::PMD_FORMAT_EXTENSION, Self::PMX_FORMAT_EXTENSION]
    }

    pub fn is_loadable_extension(extension: &str) -> bool {
        Self::loadable_extensions()
            .iter()
            .any(|ext| ext.to_lowercase().eq(extension))
    }

    pub fn uri_has_loadable_extension(uri: &Uri) -> bool {
        if let Some(ext) = uri.absolute_path_extension() {
            Self::is_loadable_extension(ext)
        } else {
            false
        }
    }

    pub fn generate_new_model_data(
        desc: &NewModelDescription,
    ) -> Result<Vec<u8>, nanoem::common::Status> {
        let mut buffer = nanoem::common::MutableBuffer::create()?;
        let mut model = NanoemModel::default();
        {
            model.set_additional_uv_size(0);
            model.set_codec_type(nanoem::common::CodecType::Utf16);
            model.set_format_type(ModelFormatType::Pmx2_0);
            for language in nanoem::common::LanguageType::all() {
                model.set_name(
                    desc.name.get(language).unwrap_or(&"".to_string()),
                    language.clone(),
                );
                model.set_comment(
                    desc.comment.get(language).unwrap_or(&"".to_string()),
                    language.clone(),
                );
            }
        }
        let mut center_bone = NanoemBone::default();
        center_bone.set_name(
            &Bone::NAME_CENTER_IN_JAPANESE.to_string(),
            nanoem::common::LanguageType::Japanese,
        );
        center_bone.set_name(&"Center".to_string(), nanoem::common::LanguageType::English);
        center_bone.set_visible(true);
        center_bone.set_movable(true);
        center_bone.set_rotatable(true);
        center_bone.set_user_handleable(true);
        let center_bone_rc = Rc::from(RefCell::from(center_bone));
        model.insert_bone(&center_bone_rc, -1)?;
        {
            let mut root_label = nanoem::model::ModelLabel::default();
            root_label.set_name(&"Root".to_string(), nanoem::common::LanguageType::Japanese);
            root_label.set_name(&"Root".to_string(), nanoem::common::LanguageType::English);
            root_label.insert_item_object(
                &nanoem::model::ModelLabelItem::create_from_bone_object(center_bone_rc),
                -1,
            );
            root_label.set_special(true);
            model.insert_label(Rc::new(RefCell::new(root_label)), -1);
        }
        {
            let mut expression_label = nanoem::model::ModelLabel::default();
            expression_label.set_name(
                &Label::NAME_EXPRESSION_IN_JAPANESE.to_string(),
                nanoem::common::LanguageType::Japanese,
            );
            expression_label.set_name(
                &"Expression".to_string(),
                nanoem::common::LanguageType::English,
            );
            expression_label.set_special(true);
            model.insert_label(Rc::new(RefCell::new(expression_label)), -1);
        }
        model.save_to_buffer(&mut buffer)?;
        Ok(buffer.get_data())
    }

    pub fn load(&mut self, bytes: &[u8], project: &Project) -> Result<(), Error> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        let mut nanoem_model = Box::new(NanoemModel::default());
        match nanoem_model.load_from_buffer(&mut buffer) {
            Ok(_) => {
                self.opaque = nanoem_model;
                let language = project.parse_language();
                self.name = self.opaque.get_name(language).to_owned();
                self.comment = self.opaque.get_comment(language).to_owned();
                self.canonical_name = self
                    .opaque
                    .get_name(nanoem::common::LanguageType::default())
                    .to_owned();
                if self.name.is_empty() {
                    self.name = self.canonical_name.clone();
                }
                Ok(())
            }
            Err(status) => Err(Error::from_nanoem("Cannot load the model: ", status)),
        }
    }

    pub fn upload(&mut self) {
        // TODO
    }

    pub fn initialize_all_staging_vertex_buffers(&mut self, device: &wgpu::Device) {
        let vertices = self.opaque.get_all_vertex_objects();
        let mut vertex_buffer_data: Vec<VertexUnit> = vec![];
        for vertex in vertices {
            let vertex = vertex.borrow();
            let vertex = vertex.get_user_data().as_ref().unwrap();
            vertex_buffer_data.push(vertex.clone().borrow().simd.clone().into());
        }
        let vertex_buffer_even = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor{
            label: Some(format!("Model/{}/VertexBuffer/Even", self.get_canonical_name()).as_str()),
            contents: bytemuck::cast_slice(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_buffer_odd = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor{
            label: Some(format!("Model/{}/VertexBuffer/Odd", self.get_canonical_name()).as_str()),
            contents: bytemuck::cast_slice(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.vertex_buffers = [vertex_buffer_even, vertex_buffer_odd];
    }

    pub fn initialize_staging_index_buffer(&mut self, device: &wgpu::Device) {
        let indices = self.opaque.get_all_vertex_indices();
        let index_buffer_data: Vec<u32> = indices.iter().map(|rc| rc.borrow().clone()).collect();
        log::trace!("Len(index_buffer): {}", index_buffer_data.len());
        self.index_buffer = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
            label: Some(format!("Model/{}/IndecBuffer", self.get_canonical_name()).as_str()),
            contents: bytemuck::cast_slice(&index_buffer_data),
            usage: wgpu::BufferUsages::INDEX,
        });
    }

    pub fn create_all_images(&mut self) {
        // TODO: 创建所有材质贴图并绑定到Material上
    }

    pub fn setup_all_bindings(&mut self, project: &Project) {
        let language = project.parse_language();
        let vertices = self.opaque.get_all_vertex_objects();
        for vertex in vertices {
            let _ = Vertex::new_bind(&mut vertex.borrow_mut(), &self.opaque);
        }
        let materials = self.opaque.get_all_material_objects();
        let indices = self.opaque.get_all_vertex_indices();
        let mut index_offset = 0;
        for material_rc in materials {
            let material = &mut material_rc.borrow_mut();
            Material::new_bind(material, project.shared_fallback_image(), language);
            let num_indices = material.get_num_vertex_indices();
            for i in index_offset..(index_offset + num_indices) {
                let vertex_index = indices[i].borrow();
                if let Some(vertex) = vertices.get(vertex_index.clone() as usize) {
                    if let Some(user_data) = vertex.borrow_mut().get_user_data() {
                        user_data.borrow_mut().set_material(material_rc);
                    }
                }
            }
            index_offset += num_indices;
        }
        for bone_rc in self.opaque.get_all_bone_objects() {
            let bone = &mut bone_rc.borrow_mut();
            Bone::new_bind(bone, language);
            if let Some(constraint) = bone.get_constraint_object() {
                let target_bone = self
                    .opaque
                    .get_one_bone_object(constraint.borrow().get_target_bone_index());
                Constraint::new_bind(&mut constraint.borrow_mut(), target_bone, language);
                for joint in constraint.borrow().get_all_joint_objects() {
                    if let Some(bone) = self
                        .opaque
                        .get_one_bone_object(joint.borrow().get_bone_index())
                    {
                        self.constraint_joint_bones
                            .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                    }
                }
                if let Some(effector_bone) = self
                    .opaque
                    .get_one_bone_object(constraint.borrow().get_effector_bone_index())
                {
                    self.constraint_effector_bones
                        .insert(Rc::downgrade(effector_bone).into_raw());
                }
            }
            if bone.has_inherent_orientation() || bone.has_inherent_translation() {
                if let Some(parent_bone) = self
                    .opaque
                    .get_one_bone_object(bone.get_parent_inherent_bone_index())
                {
                    self.inherent_bones
                        .entry(Rc::downgrade(parent_bone).into_raw())
                        .or_insert(HashSet::new())
                        .insert(Rc::downgrade(bone_rc).into_raw());
                }
            }
            for language in nanoem::common::LanguageType::all() {
                self.bones.insert(
                    bone.get_name(language.clone()).to_owned(),
                    Rc::downgrade(bone_rc),
                );
            }
        }
        for bone in self.opaque.get_all_bone_objects() {
            if let Some(parent_bone) = self
                .opaque
                .get_one_bone_object(bone.borrow().get_parent_bone_index())
            {
                self.parent_bone_tree
                    .entry(Rc::downgrade(parent_bone).into_raw())
                    .or_insert(vec![])
                    .push(Rc::downgrade(bone));
            }
        }
        if let Some(first_bone) = self.opaque.get_all_bone_objects().get(0) {
            // TODO: set into selection
        }
        let constraints = self.opaque.get_all_constraint_objects();
        if constraints.len() > 0 {
            for constraint in constraints {
                let target_bone = self
                    .opaque
                    .get_one_bone_object(constraint.borrow().get_target_bone_index());
                Constraint::new_bind(&mut constraint.borrow_mut(), target_bone, language);
                for joint in constraint.borrow().get_all_joint_objects() {
                    if let Some(bone) = self
                        .opaque
                        .get_one_bone_object(joint.borrow().get_bone_index())
                    {
                        self.constraint_joint_bones
                            .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                    }
                }
                if let Some(effector_bone) = self
                    .opaque
                    .get_one_bone_object(constraint.borrow().get_effector_bone_index())
                {
                    self.constraint_effector_bones
                        .insert(Rc::downgrade(effector_bone).into_raw());
                }
                if let Some(target_bone) = target_bone {
                    self.constraints.insert(
                        Rc::downgrade(&target_bone).into_raw(),
                        Rc::downgrade(constraint),
                    );
                }
            }
        } else {
            for bone in self.opaque.get_all_bone_objects() {
                if let Some(constraint) = bone.borrow().get_constraint_object() {
                    self.constraints
                        .insert(Rc::downgrade(bone).into_raw(), Rc::downgrade(constraint));
                }
            }
        }
        let mut bone_set: HashSet<*const RefCell<NanoemBone>> = HashSet::new();
        for morph in self.opaque.get_all_morph_objects() {
            Morph::new_bind(&mut morph.borrow_mut(), language);
            if let nanoem::model::ModelMorphType::Vertex = morph.borrow().get_type() {
                if let nanoem::model::ModelMorphU::VERTICES(morph_vertices) = morph.borrow().get_u()
                {
                    for morph_vertex in morph_vertices {
                        if let Some(vertex) = self
                            .opaque
                            .get_one_vertex_object(morph_vertex.get_vertex_index())
                        {
                            for bone_index in vertex.borrow().get_bone_indices() {
                                if let Some(bone) = self.opaque.get_one_bone_object(bone_index) {
                                    bone_set.insert(Rc::downgrade(bone).into_raw());
                                }
                            }
                        }
                    }
                }
            }
            let category = morph.borrow().get_category();
            // TODO: set active category
            for language in nanoem::common::LanguageType::all() {
                self.morphs.insert(
                    morph.borrow().get_name(language.clone()).to_owned(),
                    Rc::downgrade(morph),
                );
            }
        }
        for label in self.opaque.get_all_label_objects() {
            Label::new_bind(&mut label.borrow_mut(), language);
        }
        for rigid_body in self.opaque.get_all_rigid_body_objects() {
            let is_dynamic =
                if let nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation =
                    rigid_body.borrow().get_transform_type()
                {
                    false
                } else {
                    true
                };
            let is_morph = if let Some(bone) = self
                .opaque
                .get_one_bone_object(rigid_body.borrow().get_bone_index())
            {
                is_dynamic && bone_set.contains(&Rc::downgrade(bone).into_raw())
            } else {
                false
            };
            RigidBody::new_bind(&mut rigid_body.borrow_mut(), language);
            // TODO: initializeTransformFeedback
        }
        for joint in self.opaque.get_all_joint_objects() {
            Joint::new_bind(&mut joint.borrow_mut(), language);
        }
        for soft_body in self.opaque.get_all_soft_body_objects() {
            SoftBody::new_bind(&mut soft_body.borrow_mut(), language);
        }
        for nanoem_vertex in self.opaque.get_all_vertex_objects() {
            if let Some(vertex) = nanoem_vertex.borrow().get_user_data() {
                vertex
                    .borrow_mut()
                    .setup_bone_binding(&nanoem_vertex.borrow(), &self.opaque, self.shared_fallback_bone.clone());
            }
        }
        self.split_bones_per_material();
    }

    pub fn split_bones_per_material(&mut self) {
        let mut offset: usize = 0;
        let mut unique_bone_index_per_material = 0;
        let mut references: HashMap<i32, HashSet<*const RefCell<NanoemVertex>>> = HashMap::new();
        let mut index_hash = HashMap::new();
        let materials = self.opaque.get_all_material_objects();
        let vertices = self.opaque.get_all_vertex_objects();
        let indices = self.opaque.get_all_vertex_indices();
        self.count_vertex_skinning_needed = 0;
        for material in materials {
            let num_indices = material.borrow().get_num_vertex_indices();
            for j in offset..offset + num_indices {
                let vertex_index = &indices[j];
                let vertex = &vertices[*vertex_index.borrow() as usize];
                for bone_index in vertex.borrow().get_bone_indices() {
                    if let Some(bone) = self.opaque.get_one_bone_object(bone_index) {
                        let bone_index = bone.borrow().get_index();
                        if bone_index >= 0 {
                            if !index_hash.contains_key(&bone_index) {
                                index_hash.insert(bone_index, unique_bone_index_per_material);
                                unique_bone_index_per_material += 1;
                            }
                            references
                                .entry(bone_index)
                                .or_insert(HashSet::new())
                                .insert(Rc::downgrade(vertex).into_raw());
                        }
                    }
                }
            }
            if !index_hash.is_empty() {
                if references.len() > Self::MAX_BONE_UNIFORMS as usize {
                    let mut vertex_list: Vec<Rc<RefCell<NanoemVertex>>> = vec![];
                    let mut bone_vertex_list: Vec<(i32, Vec<Rc<RefCell<NanoemVertex>>>)> = vec![];
                    for vertex_reference in &references {
                        vertex_list.clear();
                        for vertex in vertex_reference.1 {
                            let vertex = unsafe { Weak::from_raw(*vertex) };
                            vertex_list.push(vertex.upgrade().unwrap())
                        }
                        bone_vertex_list.push((*vertex_reference.0, vertex_list.clone()));
                    }
                    bone_vertex_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                    index_hash.clear();
                    for j in 0..Self::MAX_BONE_UNIFORMS {
                        let pair = &bone_vertex_list[j as usize];
                        index_hash.insert(pair.0, j);
                    }
                    for pair in &mut bone_vertex_list {
                        let all_vertices = &mut pair.1;
                        for nanoem_vertex in all_vertices {
                            if let Some(vertex) = nanoem_vertex.borrow_mut().get_user_data() {
                                vertex.borrow_mut().set_skinning_enabled(true);
                            }
                            self.count_vertex_skinning_needed += 1;
                        }
                    }
                }
                self.bone_index_hash_map
                    .insert(Rc::downgrade(material).as_ptr(), index_hash.clone());
            }
            for it in &mut references {
                it.1.clear()
            }
            offset += num_indices;
            unique_bone_index_per_material = 0;
            index_hash.clear();
            references.clear();
        }
    }

    fn create_image() {}

    pub fn update_diffuse_image<'a, 'b: 'a>(
        material: &'a mut NanoemMaterial,
        model: &'b NanoemModel,
        mode: &mut wgpu::AddressMode,
        flags: &mut u32,
    ) {
        *mode = wgpu::AddressMode::Repeat;
        *flags = 0;
        if let Some(diffuse_texture) = material.get_diffuse_texture_object(model) {
            let path = diffuse_texture.borrow().get_path();
        }
    }

    pub fn find_bone(&self, name: &String) -> Option<Rc<RefCell<NanoemBone>>> {
        self.bones.get(name).map(|rc| rc.upgrade()).flatten()
    }

    pub fn shared_fallback_bone(&self) -> &Rc<RefCell<Bone>> {
        &self.shared_fallback_bone
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_canonical_name(&self) -> &String {
        &self.canonical_name
    }

    pub fn is_add_blend_enabled(&self) -> bool {
        // TODO: isAddBlendEnabled
        true
    }

    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    pub fn world_transform(&self, initial: &Matrix4<f32>) -> Matrix4<f32> {
        initial.clone()
    }
}

impl Drawable for Model {
    fn draw(
        &self,
        view: &wgpu::TextureView,
        typ: DrawType,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue, 
        adapter_info: wgpu::AdapterInfo,
    ) {
        if self.is_visible() {
            match typ {
                DrawType::Color => self.draw_color(view, project, device, queue, adapter_info),
                DrawType::Edge => todo!(),
                DrawType::GroundShadow => todo!(),
                DrawType::ShadowMap => todo!(),
                DrawType::ScriptExternalColor => todo!(),
            }
        }
    }

    fn is_visible(&self) -> bool {
        // TODO: isVisible
        true
    }
}

impl Model {
    fn draw_color(
        &self,
        view: &wgpu::TextureView,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue, 
        adapter_info: wgpu::AdapterInfo,
    ) {
        let mut index_offset = 0usize;
        let model_ref = &self.opaque;
        let materials = model_ref.get_all_material_objects();
        for nanoem_material in materials {
            let num_indices = nanoem_material.borrow().get_num_vertex_indices();
            log::trace!("Render next Material, Index count: {}; Offset: {}", num_indices, index_offset);
            let buffer = pass::Buffer::new(
                num_indices,
                index_offset,
                &self.vertex_buffers[1 - self.stage_vertex_buffer_index as usize],
                &self.index_buffer,
                true,
            );
            if let Some(material) = nanoem_material.borrow().get_user_data() {
                if material.borrow().is_visible() {
                    // TODO: get technique by discovery
                    let mut technique =
                        ObjectTechnique::new(nanoem_material.borrow().is_point_draw_enabled());
                    let technique_type = technique.technique_type();
                    while let Some((pass, shader)) = technique.execute(device) {
                        pass.set_global_parameters(self, project);
                        pass.set_camera_parameters(
                            project.active_camera(),
                            &Self::INITIAL_WORLD_MATRIX,
                            self,
                        );
                        pass.set_light_parameters(project.global_light(), false);
                        pass.set_all_model_parameters(self, project);
                        pass.set_material_parameters(
                            &nanoem_material.borrow(),
                            technique_type,
                            project.shared_fallback_image(),
                        );
                        pass.set_shadow_map_parameters(
                            project.shadow_camera(),
                            &Self::INITIAL_WORLD_MATRIX,
                            project,
                            adapter_info.backend,
                            technique_type,
                            project.shared_fallback_image(),
                        );
                        pass.execute(
                            &buffer,
                            view,
                            Some(&project.viewport_primary_depth_view()),
                            shader,
                            technique_type,
                            device,
                            queue,
                            self,
                            project,
                        );
                    }
                    // if (!technique->hasNextScriptCommand() && !scriptExternalColor) {
                    // technique->resetScriptCommandState();
                    // technique->resetScriptExternalColor();
                    // }
                }
            }
            index_offset += num_indices;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Matrices {
    world_transform: Matrix4<f32>,
    local_transform: Matrix4<f32>,
    normal_transform: Matrix4<f32>,
    skinning_transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Copy)]
struct BezierControlPoints {
    translation_x: Vector4<u8>,
    translation_y: Vector4<u8>,
    translation_z: Vector4<u8>,
    orientation: Vector4<u8>,
}

#[derive(Debug, Clone, Copy)]
struct LinearInterpolationEnable {
    translation_x: bool,
    translation_y: bool,
    translation_z: bool,
    orientation: bool,
}

struct FrameTransform {
    translation: Vector3<f32>,
    orientation: Quaternion<f32>,
    bezier_control_points: BezierControlPoints,
    enable_linear_interpolation: LinearInterpolationEnable,
}

#[derive(Debug, Clone)]
pub struct Bone {
    name: String,
    canonical_name: String,
    matrices: Matrices,
    local_orientation: Quaternion<f32>,
    local_inherent_orientation: Quaternion<f32>,
    local_morph_orientation: Quaternion<f32>,
    local_user_orientation: Quaternion<f32>,
    constraint_joint_orientation: Quaternion<f32>,
    local_translation: Vector3<f32>,
    local_inherent_translation: Vector3<f32>,
    local_morph_translation: Vector3<f32>,
    local_user_translation: Vector3<f32>,
    bezier_control_points: BezierControlPoints,
    states: u32,
}

impl Bone {
    const DEFAULT_BAZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
    const NAME_ROOT_PARENT_IN_JAPANESE: &'static [u8] = &[
        0xe5, 0x85, 0xa8, 0xe3, 0x81, 0xa6, 0xe3, 0x81, 0xae, 0xe8, 0xa6, 0xaa, 0x0,
    ];
    const NAME_CENTER_IN_JAPANESE_UTF8: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0,
    ];
    const NAME_CENTER_IN_JAPANESE: &'static str = "センター";
    const NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE: &'static [u8] = &[
        0xe6, 0x93, 0x8d, 0xe4, 0xbd, 0x9c, 0xe4, 0xb8, 0xad, 0xe5, 0xbf, 0x83, 0,
    ];
    const NAME_CENTER_OFFSET_IN_JAPANESE: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0xe5, 0x85, 0x88, 0,
    ];
    const NAME_LEFT_IN_JAPANESE: &'static [u8] = &[0xe5, 0xb7, 0xa6, 0x0];
    const NAME_RIGHT_IN_JAPANESE: &'static [u8] = &[0xe5, 0x8f, 0xb3, 0x0];
    const NAME_DESTINATION_IN_JAPANESE: &'static [u8] = &[0xe5, 0x85, 0x88, 0x0];
    const LEFT_KNEE_IN_JAPANESE: &'static [u8] =
        &[0xe5, 0xb7, 0xa6, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];
    const RIGHT_KNEE_IN_JAPANESE: &'static [u8] =
        &[0xe5, 0x8f, 0xb3, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];

    const PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_X: u32 = 1u32 << 1;
    const PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_Y: u32 = 1u32 << 2;
    const PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_Z: u32 = 1u32 << 3;
    const PRIVATE_STATE_LINEAR_INTERPOLATION_ORIENTATION: u32 = 1u32 << 4;
    const PRIVATE_STATE_DIRTY: u32 = 1u32 << 5;
    const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 6;
    const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    const PRIVATE_STATE_INITIAL_VALUE: u32 = Self::PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_X
        | Self::PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_Y
        | Self::PRIVATE_STATE_LINEAR_INTERPOLATION_TRANSLATION_Z
        | Self::PRIVATE_STATE_LINEAR_INTERPOLATION_ORIENTATION;

    pub fn new(name: &str, canonical_name: &str) -> Self {
        Self {
            name: name.to_owned(),
            canonical_name: canonical_name.to_owned(),
            matrices: Matrices {
                world_transform: Matrix4::identity(),
                local_transform: Matrix4::identity(),
                normal_transform: Matrix4::identity(),
                skinning_transform: Matrix4::identity(),
            },
            local_orientation: Quaternion::new(1f32, 0f32, 0f32, 0f32),
            local_inherent_orientation: Quaternion::new(1f32, 0f32, 0f32, 0f32),
            local_morph_orientation: Quaternion::new(1f32, 0f32, 0f32, 0f32),
            local_user_orientation: Quaternion::new(1f32, 0f32, 0f32, 0f32),
            constraint_joint_orientation: Quaternion::new(1f32, 0f32, 0f32, 0f32),
            local_translation: Vector3::zero(),
            local_inherent_translation: Vector3::zero(),
            local_morph_translation: Vector3::zero(),
            local_user_translation: Vector3::zero(),
            bezier_control_points: BezierControlPoints {
                translation_x: Vector4::new(0, 0, 0, 0),
                translation_y: Vector4::new(0, 0, 0, 0),
                translation_z: Vector4::new(0, 0, 0, 0),
                orientation: Vector4::new(0, 0, 0, 0),
            },
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }
    }

    pub fn new_bind(
        bone: &mut NanoemBone,
        language_type: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
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
        let result = Rc::new(RefCell::new(Self::new(&name, &canonical_name)));
        bone.set_user_data(&result);
        result
    }

    // fn synchronize_transform(motion: &mut Motion, model_bone: &ModelBone, model_rigid_body: &ModelRigidBody, frame_index: u32, transform: &FrameTransform) {
    //     let name = model_bone.get_name(LanguageType::Japanese).unwrap();
    //     if let Some(Keyframe) = motion.find_bone_keyframe_object(name, index)
    // }
}

pub struct ConstraintJoint {
    orientation: Quaternion<f32>,
    translation: Vector3<f32>,
    target_direction: Vector3<f32>,
    effector_direction: Vector3<f32>,
    axis: Vector3<f32>,
    angle: f32,
}

pub struct Constraint {
    name: String,
    canonical_name: String,
    joint_iteration_result: HashMap<*const RefCell<NanoemConstraintJoint>, Vec<ConstraintJoint>>,
    effector_iteration_result: HashMap<*const RefCell<NanoemConstraintJoint>, Vec<ConstraintJoint>>,
    states: u32,
}

impl Constraint {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = Self::PRIVATE_STATE_ENABLED;

    pub fn new_bind(
        constraint: &mut NanoemConstraint,
        bone: Option<&Rc<RefCell<NanoemBone>>>,
        language_type: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        let mut name = if let Some(bone) = bone {
            bone.borrow().get_name(language_type).to_owned()
        } else {
            "".to_owned()
        };
        let mut canonical_name = if let Some(bone) = bone {
            bone.borrow()
                .get_name(nanoem::common::LanguageType::default())
                .to_owned()
        } else {
            "".to_owned()
        };
        if canonical_name.is_empty() {
            canonical_name = format!("Constraint{}", constraint.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let mut joint_iteration_result: HashMap<
            *const RefCell<NanoemConstraintJoint>,
            Vec<ConstraintJoint>,
        > = HashMap::new();
        let mut effector_iteration_result: HashMap<
            *const RefCell<NanoemConstraintJoint>,
            Vec<ConstraintJoint>,
        > = HashMap::new();
        for joint in constraint.get_all_joint_objects() {
            joint_iteration_result.insert(Rc::downgrade(joint).into_raw(), vec![]);
            effector_iteration_result.insert(Rc::downgrade(joint).into_raw(), vec![]);
        }
        let result = Rc::new(RefCell::new(Self {
            name,
            canonical_name,
            joint_iteration_result,
            effector_iteration_result,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }));
        constraint.set_user_data(&result);
        result
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VertexSimd {
    // may use simd128
    pub origin: Vector4<f32>,
    pub normal: Vector4<f32>,
    pub texcoord: Vector4<f32>,
    pub info: Vector4<f32>,
    pub indices: Vector4<f32>,
    pub delta: Vector4<f32>,
    pub weights: Vector4<f32>,
    pub origin_uva: [Vector4<f32>; 4],
    pub delta_uva: [Vector4<f32>; 5],
}

#[derive(Clone)]
pub struct Vertex {
    material: Option<Weak<RefCell<NanoemMaterial>>>,
    soft_body: Option<Weak<RefCell<NanoemSoftBody>>>,
    bones: [Option<Weak<RefCell<Bone>>>; 4],
    states: u32,
    pub simd: VertexSimd,
}

impl Vertex {
    const PRIVATE_STATE_SKINNING_ENABLED: u32 = 1 << 1;
    const PRIVATE_STATE_EDITING_MASKED: u32 = 1 << 2;
    const PRIVATE_STATE_INITIAL_VALUE: u32 = 0;

    fn new_bind(vertex: &mut NanoemVertex, model: &NanoemModel) -> Rc<RefCell<Self>> {
        let direction = Vector4::new(1f32, 1f32, 1f32, 1f32);
        let texcoord = vertex.get_tex_coord();
        let bone_indices: [f32; 4] = vertex
            .get_bone_indices()
            .map(|idx| model.get_one_bone_object(idx))
            .map(|orc| orc.map_or(-1f32, |rc| rc.borrow().get_index() as f32));
        let simd = VertexSimd {
            origin: vertex.get_origin().into(),
            normal: vertex.get_normal().into(),
            texcoord: Vector4::new(
                texcoord[0].fract(),
                texcoord[1].fract(),
                texcoord[2],
                texcoord[3],
            ),
            info: Vector4::new(
                vertex.get_edge_size(),
                i32::from(vertex.get_type()) as f32,
                vertex.get_index() as f32,
                1f32,
            ),
            indices: bone_indices.into(),
            delta: Vector4::zero(),
            weights: vertex.get_bone_weights().into(),
            origin_uva: vertex.get_additional_uv().map(|uv| uv.into()),
            delta_uva: [
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
            ],
        };
        let result = Rc::new(RefCell::new(Self {
            material: None,
            soft_body: None,
            bones: [None, None, None, None],
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            simd,
        }));
        vertex.set_user_data(result.clone());
        result
    }

    pub fn set_material(&mut self, material: &Rc<RefCell<NanoemMaterial>>) {
        self.material = Some(Rc::downgrade(material))
    }

    fn get_bone_from_vertex_by_index(
        vertex: &NanoemVertex,
        model: &NanoemModel,
        index: usize,
        fallback_bone: Rc<RefCell<Bone>>
    ) -> Rc<RefCell<Bone>> {
        model
            .get_one_bone_object(vertex.get_bone_indices()[index])
            .map(|bone| bone.borrow().get_user_data().map(|rc| rc.clone()))
            .flatten()
            .unwrap_or(fallback_bone)
    }

    pub fn setup_bone_binding(&mut self, vertex: &NanoemVertex, model: &NanoemModel,
        fallback_bone: Rc<RefCell<Bone>>) {
        match vertex.get_type() {
            nanoem::model::ModelVertexType::UNKNOWN => {}
            nanoem::model::ModelVertexType::BDEF1 => {
                self.bones[0] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 0, fallback_bone.clone(),
                )));
            }
            nanoem::model::ModelVertexType::BDEF2 | nanoem::model::ModelVertexType::SDEF => {
                self.bones[0] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 0, fallback_bone.clone(),
                )));
                self.bones[1] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 1, fallback_bone.clone(),
                )));
            }
            nanoem::model::ModelVertexType::BDEF4 | nanoem::model::ModelVertexType::QDEF => {
                self.bones[0] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 0, fallback_bone.clone(),
                )));
                self.bones[1] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 1, fallback_bone.clone(),
                )));
                self.bones[2] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 2, fallback_bone.clone(),
                )));
                self.bones[3] = Some(Rc::downgrade(&Self::get_bone_from_vertex_by_index(
                    vertex, model, 3, fallback_bone.clone(),
                )));
            }
        }
    }

    pub fn set_skinning_enabled(&mut self, value: bool) {
        self.states = if value {
            self.states | Self::PRIVATE_STATE_SKINNING_ENABLED
        } else {
            self.states & !Self::PRIVATE_STATE_SKINNING_ENABLED
        }
    }
}

pub struct MaterialColor {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub diffuse_opacity: f32,
    pub specular_power: f32,
    pub diffuse_texture_blend_factor: Vector4<f32>,
    pub sphere_texture_blend_factor: Vector4<f32>,
    pub toon_texture_blend_factor: Vector4<f32>,
}

impl MaterialColor {
    pub fn new_reset(v: f32) -> Self {
        Self {
            ambient: Vector3::new(v, v, v),
            diffuse: Vector3::new(v, v, v),
            specular: Vector3::new(v, v, v),
            diffuse_opacity: v,
            specular_power: v.max(Material::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: Vector4::new(v, v, v, v),
            sphere_texture_blend_factor: Vector4::new(v, v, v, v),
            toon_texture_blend_factor: Vector4::new(v, v, v, v),
        }
    }
}

struct MaterialBlendColor {
    base: MaterialColor,
    add: MaterialColor,
    mul: MaterialColor,
}

pub struct MaterialEdge {
    pub color: Vector3<f32>,
    pub opacity: f32,
    pub size: f32,
}

impl MaterialEdge {
    pub fn new_reset(v: f32) -> Self {
        Self {
            color: Vector3::new(v, v, v),
            opacity: v,
            size: v,
        }
    }
}

struct MaterialBlendEdge {
    base: MaterialEdge,
    add: MaterialEdge,
    mul: MaterialEdge,
}

pub struct Material {
    // TODO
    color: MaterialBlendColor,
    edge: MaterialBlendEdge,
    effect: Option<Effect>,
    diffuse_image: Option<Rc<wgpu::Texture>>,
    sphere_map_image: Option<Rc<wgpu::Texture>>,
    toon_image: Option<Rc<wgpu::Texture>>,
    name: String,
    canonical_name: String,
    fallback_image: wgpu::TextureView,
    index_hash: HashMap<u32, u32>,
    toon_color: Vector4<f32>,
    states: u32,
}

impl Material {
    pub const PRIVATE_STATE_VISIBLE: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_DISPLAY_DIFFUSE_TEXTURE_UV_MESH_ENABLED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_DISPLAY_SPHERE_MAP_TEXTURE_UV_MESH_ENABLED: u32 = 1u32 << 3;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;
    pub const PRIVATE_STATE_INITIAL_VALUE: u32 =
        Self::PRIVATE_STATE_VISIBLE | Self::PRIVATE_STATE_DISPLAY_DIFFUSE_TEXTURE_UV_MESH_ENABLED;
    pub const MINIUM_SPECULAR_POWER: f32 = 0.1f32;

    pub fn new_bind(
        material: &mut NanoemMaterial,
        fallback: &wgpu::Texture,
        language_type: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        let mut name = material.get_name(language_type).to_owned();
        let mut canonical_name = material
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Material{}", material.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            color: MaterialBlendColor {
                base: MaterialColor {
                    ambient: Vector4::from(material.get_ambient_color()).truncate(),
                    diffuse: Vector4::from(material.get_diffuse_color()).truncate(),
                    specular: Vector4::from(material.get_specular_color()).truncate(),
                    diffuse_opacity: material.get_diffuse_opacity(),
                    specular_power: material.get_specular_power(),
                    diffuse_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    sphere_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    toon_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                },
                add: MaterialColor::new_reset(0f32),
                mul: MaterialColor::new_reset(1f32),
            },
            edge: MaterialBlendEdge {
                base: MaterialEdge {
                    color: Vector4::from(material.get_edge_color()).truncate(),
                    opacity: material.get_edge_opacity(),
                    size: material.get_edge_size(),
                },
                add: MaterialEdge::new_reset(0f32),
                mul: MaterialEdge::new_reset(1f32),
            },
            effect: None,
            diffuse_image: None,
            sphere_map_image: None,
            toon_image: None,
            name,
            canonical_name,
            fallback_image: fallback.create_view(&wgpu::TextureViewDescriptor::default()),
            index_hash: HashMap::new(),
            toon_color: Vector4::new(1f32, 1f32, 1f32, 1f32),
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }));
        material.set_user_data(result.clone());
        result
    }

    pub fn is_visible(&self) -> bool {
        // TODO: isVisible
        true
    }

    pub fn color(&self) -> MaterialColor {
        MaterialColor {
            ambient: self
                .color
                .base
                .ambient
                .mul_element_wise(self.color.mul.ambient)
                + self.color.add.ambient,
            diffuse: self
                .color
                .base
                .diffuse
                .mul_element_wise(self.color.mul.diffuse)
                + self.color.add.diffuse,
            specular: self
                .color
                .base
                .specular
                .mul_element_wise(self.color.mul.specular)
                + self.color.add.specular,
            diffuse_opacity: self.color.base.diffuse_opacity * self.color.mul.diffuse_opacity
                + self.color.add.diffuse_opacity,
            specular_power: (self.color.base.specular_power * self.color.mul.specular_power
                + self.color.add.specular_power)
                .min(Self::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: self
                .color
                .base
                .diffuse_texture_blend_factor
                .mul_element_wise(self.color.mul.diffuse_texture_blend_factor)
                + self.color.add.diffuse_texture_blend_factor,
            sphere_texture_blend_factor: self
                .color
                .base
                .sphere_texture_blend_factor
                .mul_element_wise(self.color.mul.sphere_texture_blend_factor)
                + self.color.add.sphere_texture_blend_factor,
            toon_texture_blend_factor: self
                .color
                .base
                .toon_texture_blend_factor
                .mul_element_wise(self.color.mul.toon_texture_blend_factor)
                + self.color.add.toon_texture_blend_factor,
        }
    }

    pub fn edge(&self) -> MaterialEdge {
        MaterialEdge {
            color: self.edge.base.color.mul_element_wise(self.edge.mul.color) + self.edge.add.color,
            opacity: self.edge.base.opacity * self.edge.mul.opacity + self.edge.add.opacity,
            size: self.edge.base.size * self.edge.mul.size + self.edge.add.size,
        }
    }

    pub fn diffuse_image(&self) -> Option<&wgpu::Texture> {
        self.diffuse_image.as_ref().map(|rc|rc.as_ref())
    }

    pub fn sphere_map_image(&self) -> Option<&wgpu::Texture> {
        self.sphere_map_image.as_ref().map(|rc|rc.as_ref())
    }

    pub fn toon_image(&self) -> Option<&wgpu::Texture> {
        self.toon_image.as_ref().map(|rc|rc.as_ref())
    }

    pub fn set_diffuse_image(&mut self, texture: Rc<wgpu::Texture>) {
        self.diffuse_image = Some(texture.clone());
    }

    pub fn set_sphere_map_image(&mut self, texture: Rc<wgpu::Texture>) {
        self.sphere_map_image = Some(texture.clone());
    }

    pub fn set_toon_image(&mut self, texture: Rc<wgpu::Texture>) {
        self.toon_image = Some(texture.clone());
    }
}

pub struct Morph {
    name: String,
    canonical_name: String,
    weight: f32,
    dirty: bool,
}

impl Morph {
    pub fn new_bind(
        morph: &mut NanoemMorph,
        language: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        let mut name = morph.get_name(language).to_owned();
        let mut canonical_name = morph
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Morph{}", morph.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            name,
            canonical_name,
            weight: 0f32,
            dirty: false,
        }));
        morph.set_user_data(&result);
        result
    }
}

pub struct Label {
    name: String,
    canonical_name: String,
}

impl Label {
    const NAME_EXPRESSION_IN_JAPANESE_UTF8: &'static [u8] =
        &[0xe8, 0xa1, 0xa8, 0xe6, 0x83, 0x85, 0x0];
    const NAME_EXPRESSION_IN_JAPANESE: &'static str = "表情";

    pub fn new_bind(
        label: &mut NanoemLabel,
        language: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        let mut name = label.get_name(language).to_owned();
        let mut canonical_name = label
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", label.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            name,
            canonical_name,
        }));
        label.set_user_data(&result);
        result
    }
}

pub struct RigidBody {
    // TODO: physics engine and shape mesh and engine rigid_body
    global_torque_force: (Vector3<f32>, bool),
    global_velocity_force: (Vector3<f32>, bool),
    local_torque_force: (Vector3<f32>, bool),
    local_velocity_force: (Vector3<f32>, bool),
    name: String,
    canonical_name: String,
    states: u32,
}

impl RigidBody {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_ALL_FORCES_SHOULD_RESET: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 3;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn new_bind(
        rigid_body: &mut NanoemRigidBody,
        language: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        // TODO: set physics engine and bind to engine rigid_body
        let mut name = rigid_body.get_name(language).to_owned();
        let mut canonical_name = rigid_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", rigid_body.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            global_torque_force: (Vector3::zero(), false),
            global_velocity_force: (Vector3::zero(), false),
            local_torque_force: (Vector3::zero(), false),
            local_velocity_force: (Vector3::zero(), false),
            name,
            canonical_name,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }));
        rigid_body.set_user_data(&result);
        result
    }
}

pub struct Joint {
    // TODO: physics engine and shape mesh and engine rigid_body
    name: String,
    canonical_name: String,
    states: u32,
}

impl Joint {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn new_bind(
        joint: &mut NanoemJoint,
        language: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        // TODO: should resolve physic engine
        let mut name = joint.get_name(language).to_owned();
        let mut canonical_name = joint
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", joint.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            name,
            canonical_name,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }));
        joint.set_user_data(&result);
        result
    }
}

pub struct SoftBody {
    // TODO: physics engine and shape mesh and engine soft_body
    name: String,
    canonical_name: String,
    states: u32,
}

impl SoftBody {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn new_bind(
        soft_body: &mut NanoemSoftBody,
        language: nanoem::common::LanguageType,
    ) -> Rc<RefCell<Self>> {
        // TODO: should resolve physic engine
        let mut name = soft_body.get_name(language).to_owned();
        let mut canonical_name = soft_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", soft_body.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let result = Rc::new(RefCell::new(Self {
            name,
            canonical_name,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
        }));
        soft_body.set_user_data(&result);
        result
    }
}

pub struct VisualizationClause {
    // TODO
}

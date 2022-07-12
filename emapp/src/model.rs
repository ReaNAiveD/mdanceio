use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    f32::consts::PI,
    rc::{Rc, Weak},
};

use bytemuck::{Pod, Zeroable};
use cgmath::{
    AbsDiffEq, Deg, ElementWise, InnerSpace, Matrix3, Matrix4, Quaternion, Rad, Rotation3,
    SquareMatrix, Vector3, Vector4, VectorSpace, Zero,
};
use nalgebra::Isometry;
use nanoem::{
    model::{ModelMorphCategory, ModelRigidBodyTransformType},
    motion::{MotionBoneKeyframe, MotionModelKeyframe, MotionTrackBundle},
};
use wgpu::{AddressMode, Buffer};

use crate::{
    bounding_box::BoundingBox,
    camera::{Camera, PerspectiveCamera},
    deformer::Deformer,
    drawable::{DrawType, Drawable},
    effect::{Effect, IEffect},
    error::Error,
    forward::LineVertexUnit,
    internal::LinearDrawer,
    model_object_selection::ModelObjectSelection,
    model_program_bundle::{
        EdgeTechnique, GroundShadowTechnique, ModelProgramBundle, ObjectTechnique, ZplotTechnique,
    },
    motion::{KeyframeInterpolationPoint, Motion},
    pass,
    physics_engine::{PhysicsEngine, RigidBodyFollowBone, SimulationMode, SimulationTiming},
    project::Project,
    technique::Technique,
    undo::UndoStack,
    uri::Uri,
    utils::{
        f128_to_quat, f128_to_vec3, f128_to_vec4, from_na_mat4, lerp_f32, mat4_truncate,
        to_isometry, to_na_mat4, to_na_vec3, CompareElementWise, Invert,
    },
};

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
pub type VertexIndex = usize;
pub type BoneIndex = usize;
pub type MaterialIndex = usize;
pub type MorphIndex = usize;
pub type ConstraintIndex = usize;
pub type LabelIndex = usize;
pub type RigidBodyIndex = usize;
pub type JointIndex = usize;
pub type SoftBodyIndex = usize;

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

#[derive(Debug, Clone, Copy)]
pub struct ModelMorphUsage {
    pub eyebrow: Option<MorphIndex>,
    pub eye: Option<MorphIndex>,
    pub lip: Option<MorphIndex>,
    pub other: Option<MorphIndex>,
}

#[derive(Debug, Clone, Copy, Default, Hash)]
struct ModelStates {
    visible: bool,
    uploaded: bool,
    vertex_shader_skinning: bool,
    show_all_bones: bool,
    show_all_vertex_points: bool,
    show_all_vertex_faces: bool,
    show_all_rigid_body_shapes: bool,
    show_all_rigid_body_shapes_color_by_shape: bool,
    compute_shader_skinning: bool,
    morph_weight_focus: bool,
    enable_shadow: bool,
    enable_shadow_map: bool,
    enable_add_blend: bool,
    show_all_joint_shapes: bool,
    dirty: bool,
    dirty_staging_buffer: bool,
    dirty_morph: bool,
    physics_simulation: bool,
    enable_ground_shadow: bool,
    show_all_material_shapes: bool,
    show_all_vertex_weights: bool,
    blending_vertex_weights_enabled: bool,
    show_all_vertex_normals: bool,
}

pub struct Model {
    // camera: Rc<RefCell<dyn Camera>>,
    // selection: Rc<RefCell<dyn ModelObjectSelection>>,
    // drawer: Box<LinearDrawer>,
    skin_deformer: Deformer,
    // gizmo: Rc<RefCell<dyn Gizmo>>,
    // vertex_weight_painter: Rc<RefCell<dyn VertexWeightPainter>>,
    // offscreen_passive_render_target_effects: HashMap<String, OffscreenPassiveRenderTargetEffect>,
    // draw_all_vertex_normals: DrawArrayBuffer,
    // draw_all_vertex_points: DrawArrayBuffer,
    // draw_all_vertex_faces: DrawIndexedBuffer,
    // draw_all_vertex_weights: DrawIndexedBuffer,
    // draw_rigid_body: HashMap<Rc<RefCell<TriMesh>>, DrawIndexedBuffer>,
    // draw_joint: HashMap<Rc<RefCell<TriMesh>>, DrawIndexedBuffer>,
    opaque: Box<NanoemModel>,
    vertices: Vec<Vertex>,
    vertex_indices: Vec<u32>,
    materials: Vec<Material>,
    bones: Vec<Bone>,
    constraints: Vec<Constraint>,
    morphs: Vec<Morph>,
    labels: Vec<Label>,
    rigid_bodies: Vec<RigidBody>,
    joints: Vec<Joint>,
    soft_bodies: Vec<SoftBody>,
    // undo_stack: Box<UndoStack>,
    // editing_undo_stack: Box<UndoStack>,
    active_morph: ModelMorphUsage,
    // active_constraint_ptr: Rc<RefCell<NanoemConstraint>>,
    // active_material_ptr: Rc<RefCell<NanoemMaterial>>,
    // hovered_bone_ptr: Rc<RefCell<NanoemBone>>,
    // vertex_buffer_data: Vec<u8>,
    // face_states: Vec<u32>,
    active_bone_pair: (Option<BoneIndex>, Option<BoneIndex>),
    // active_effect_pair_ptr: (Rc<RefCell<dyn IEffect>>, Rc<RefCell<dyn IEffect>>),
    // screen_image: Image,
    // loading_image_items: Vec<LoadingImageItem>,
    // image_map: HashMap<String, Image>,
    bone_index_hash_map: HashMap<MaterialIndex, HashMap<BoneIndex, usize>>,
    bones_by_name: HashMap<String, BoneIndex>,
    morphs_by_name: HashMap<String, MorphIndex>,
    bone_to_constraints: HashMap<BoneIndex, ConstraintIndex>,
    // redo_bone_names: Vec<String>,
    // redo_morph_names: Vec<String>,
    outside_parents: HashMap<BoneIndex, (String, String)>,
    // image_uris: HashMap<String, Uri>,
    // attachment_uris: HashMap<String, Uri>,
    bone_bound_rigid_bodies: HashMap<BoneIndex, RigidBodyIndex>,
    constraint_joint_bones: HashMap<BoneIndex, ConstraintIndex>,
    inherent_bones: HashMap<BoneIndex, HashSet<BoneIndex>>,
    constraint_effector_bones: HashSet<BoneIndex>,
    parent_bone_tree: HashMap<BoneIndex, Vec<BoneIndex>>,
    pub shared_fallback_bone: Bone,
    bounding_box: BoundingBox,
    // // UserData m_userData;
    // annotations: HashMap<String, String>,
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffer: wgpu::Buffer,
    edge_color: Vector4<f32>,
    // transform_axis_type: AxisType,
    // edit_action_type: EditActionType,
    // transform_coordinate_type: TransformCoordinateType,
    // file_uri: Uri,
    name: String,
    comment: String,
    canonical_name: String,
    states: ModelStates,
    edge_size_scale_factor: f32,
    opacity: f32,
    // // void *m_dispatchParallelTaskQueue
    count_vertex_skinning_needed: i32,
    stage_vertex_buffer_index: usize,
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
    pub const MAX_BONE_UNIFORMS: usize = 55;

    pub fn new_from_bytes(
        bytes: &[u8],
        language_type: nanoem::common::LanguageType,
        fallback_texture: &wgpu::Texture,
        physics_engine: &mut PhysicsEngine,
        camera: &dyn Camera,
        device: &wgpu::Device,
    ) -> Result<Self, Error> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        match NanoemModel::load_from_buffer(&mut buffer) {
            Ok(nanoem_model) => {
                let opaque = Box::new(nanoem_model);
                let mut name = opaque.get_name(language_type).to_owned();
                let comment = opaque.get_comment(language_type).to_owned();
                let canonical_name = opaque
                    .get_name(nanoem::common::LanguageType::default())
                    .to_owned();
                if name.is_empty() {
                    name = canonical_name.clone();
                }

                // TODO: 共享fallback骨骼
                // let shared_fallback_bone = Rc::new(RefCell::new(Bone::new(
                //     "SharedFallbackBone",
                //     "SharedFallbackBone",
                // )));

                let mut vertices = opaque
                    .vertices
                    .iter()
                    .map(|vertex| Vertex::from_nanoem(vertex))
                    .collect::<Vec<_>>();
                let indices = opaque.vertex_indices.clone();
                let materials = opaque
                    .materials
                    .iter()
                    .map(|material| {
                        Material::from_nanoem(material, fallback_texture, language_type)
                    })
                    .collect::<Vec<_>>();
                let mut index_offset = 0;
                for nanoem_material in &opaque.materials {
                    let num_indices = nanoem_material.get_num_vertex_indices();
                    for i in index_offset..(index_offset + num_indices) {
                        let vertex_index = indices[i];
                        if let Some(vertex) = vertices.get_mut(vertex_index as usize) {
                            vertex.set_material(nanoem_material.get_index());
                        }
                    }
                    index_offset += num_indices;
                }
                let mut constraint_joint_bones = HashMap::new();
                let mut constraint_effector_bones = HashSet::new();
                let mut inherent_bones = HashMap::new();
                let mut bones_by_name = HashMap::new();
                let bones = opaque
                    .bones
                    .iter()
                    .map(|bone| Bone::from_nanoem(bone, language_type))
                    .collect::<Vec<_>>();
                for bone in &opaque.bones {
                    if bone.has_inherent_orientation() || bone.has_inherent_translation() {
                        if let Some(parent_bone) =
                            opaque.get_one_bone_object(bone.get_parent_inherent_bone_index())
                        {
                            inherent_bones
                                .entry(parent_bone.base.index)
                                .or_insert(HashSet::new())
                                .insert(bone.base.index);
                        }
                    }
                    for language in nanoem::common::LanguageType::all() {
                        bones_by_name
                            .insert(bone.get_name(language.clone()).to_owned(), bone.base.index);
                    }
                }
                let mut parent_bone_tree = HashMap::new();
                for bone in &opaque.bones {
                    if let Some(parent_bone) = opaque.get_one_bone_object(bone.parent_bone_index) {
                        parent_bone_tree
                            .entry(parent_bone.base.index)
                            .or_insert(vec![])
                            .push(bone.base.index);
                    }
                }
                if let Some(first_bone) = opaque.bones.get(0) {
                    // TODO: set into selection
                }
                let mut constraints = vec![];
                let mut bone_to_constraints = HashMap::new();
                for nanoem_constraint in &opaque.constraints {
                    let target_bone =
                        opaque.get_one_bone_object(nanoem_constraint.get_target_bone_index());
                    constraints.push(Constraint::from_nanoem(
                        &nanoem_constraint,
                        target_bone,
                        language_type,
                    ));
                    for joint in &nanoem_constraint.joints {
                        if let Some(bone) = opaque.get_one_bone_object(joint.bone_index) {
                            constraint_joint_bones
                                .insert(bone.base.index, nanoem_constraint.base.index);
                        }
                    }
                    if let Some(effector_bone) =
                        opaque.get_one_bone_object(nanoem_constraint.effector_bone_index)
                    {
                        constraint_effector_bones.insert(effector_bone.base.index);
                    }
                    if let Some(target_bone) = target_bone {
                        bone_to_constraints
                            .insert(target_bone.base.index, nanoem_constraint.base.index);
                    }
                }
                for bone in &opaque.bones {
                    if let Some(mut nanoem_constraint) = bone.constraint.clone() {
                        nanoem_constraint.base.index = constraints.len();
                        nanoem_constraint.target_bone_index = bone.base.index as i32;
                        let target_bone_index = nanoem_constraint.get_target_bone_index();
                        constraints.push(Constraint::from_nanoem(
                            &nanoem_constraint,
                            opaque.get_one_bone_object(target_bone_index),
                            language_type,
                        ));
                        for joint in &nanoem_constraint.joints {
                            if let Some(bone) = opaque.get_one_bone_object(joint.get_bone_index()) {
                                constraint_joint_bones
                                    .insert(bone.base.index, nanoem_constraint.base.index);
                            }
                        }
                        if let Some(effector_bone) =
                            opaque.get_one_bone_object(nanoem_constraint.effector_bone_index)
                        {
                            constraint_effector_bones.insert(effector_bone.base.index);
                        }
                        if opaque.constraints.len() > 0 {
                            bone_to_constraints
                                .insert(bone.base.index, nanoem_constraint.base.index);
                        }
                    }
                }
                let mut bone_set: HashSet<BoneIndex> = HashSet::new();
                let mut morphs_by_name = HashMap::new();
                let morphs = opaque
                    .morphs
                    .iter()
                    .map(|morph| Morph::from_nanoem(&morph, language_type))
                    .collect::<Vec<_>>();
                for (index, morph) in opaque.morphs.iter().enumerate() {
                    if let nanoem::model::ModelMorphType::Vertex(morph_vertices) = morph.get_type()
                    {
                        for morph_vertex in morph_vertices {
                            if let Some(vertex) =
                                opaque.get_one_vertex_object(morph_vertex.vertex_index)
                            {
                                for bone_index in vertex.get_bone_indices() {
                                    if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                        bone_set.insert(bone.base.index);
                                    }
                                }
                            }
                        }
                    }
                    let category = morph.category;
                    for language in nanoem::common::LanguageType::all() {
                        morphs_by_name
                            .insert(morph.get_name(*language).to_owned(), morph.base.index);
                    }
                }
                let get_active_morph = |category: ModelMorphCategory| {
                    opaque
                        .morphs
                        .iter()
                        .enumerate()
                        .filter(|(index, morph)| morph.category == category)
                        .next()
                        .map(|(idx, _)| idx)
                };
                let active_morph = ModelMorphUsage {
                    eyebrow: get_active_morph(ModelMorphCategory::Eyebrow),
                    eye: get_active_morph(ModelMorphCategory::Eye),
                    lip: get_active_morph(ModelMorphCategory::Lip),
                    other: get_active_morph(ModelMorphCategory::Other),
                };
                let labels = opaque
                    .labels
                    .iter()
                    .map(|label| Label::from_nanoem(label, language_type))
                    .collect();
                let rigid_bodies = opaque
                    .rigid_bodies
                    .iter()
                    .map(|rigid_body| {
                        let is_dynamic =
                            if let nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation =
                                rigid_body.get_transform_type()
                            {
                                false
                            } else {
                                true
                            };
                        let is_morph = if let Some(bone) =
                            opaque.get_one_bone_object(rigid_body.get_bone_index())
                        {
                            is_dynamic && bone_set.contains(&bone.base.index)
                        } else {
                            false
                        };
                        // TODO: initializeTransformFeedback
                        RigidBody::from_nanoem(rigid_body, language_type, is_morph, &bones, physics_engine)})
                    .collect();
                let joints = opaque
                    .joints
                    .iter()
                    .map(|joint| Joint::from_nanoem(joint, language_type))
                    .collect();
                let soft_bodies = opaque
                    .soft_bodies
                    .iter()
                    .map(|soft_body| SoftBody::from_nanoem(soft_body, language_type))
                    .collect();

                let shared_fallback_bone = Bone {
                    name: "".to_owned(),
                    canonical_name: "".to_owned(),
                    matrices: Matrices {
                        world_transform: Matrix4::identity(),
                        local_transform: Matrix4::identity(),
                        normal_transform: Matrix4::identity(),
                        skinning_transform: Matrix4::identity(),
                    },
                    local_orientation: Quaternion::zero(),
                    local_inherent_orientation: Quaternion::zero(),
                    local_morph_orientation: Quaternion::zero(),
                    local_user_orientation: Quaternion::zero(),
                    constraint_joint_orientation: Quaternion::zero(),
                    local_translation: Vector3::zero(),
                    local_inherent_translation: Vector3::zero(),
                    local_morph_translation: Vector3::zero(),
                    local_user_translation: Vector3::zero(),
                    interpolation: BoneKeyframeInterpolation::default(),
                    states: BoneStates::default(),
                    origin: NanoemBone {
                        base: nanoem::model::ModelObject { index: usize::MAX },
                        name_ja: "".to_owned(),
                        name_en: "".to_owned(),
                        constraint: None,
                        parent_bone_index: -1,
                        parent_inherent_bone_index: -1,
                        effector_bone_index: -1,
                        target_bone_index: -1,
                        global_bone_index: -1,
                        stage_index: -1,
                        ..Default::default()
                    },
                };
                // split_bones_per_material();

                let edge_size_scale_factor = 1.0f32;

                let mut offset: usize = 0;
                let mut unique_bone_index_per_material = 0usize;
                let mut references: HashMap<usize, HashSet<VertexIndex>> = HashMap::new();
                let mut index_hash = HashMap::new();
                let mut bone_index_hash_map = HashMap::new();
                let mut count_vertex_skinning_needed = 0;
                for material in &opaque.materials {
                    let num_indices = material.num_vertex_indices;
                    for j in offset..offset + num_indices {
                        let vertex_index = &opaque.vertex_indices[j];
                        let vertex = &opaque.vertices[*vertex_index as usize];
                        for bone_index in vertex.bone_indices {
                            if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                let bone_index = bone.base.index;
                                if !index_hash.contains_key(&bone_index) {
                                    index_hash.insert(bone_index, unique_bone_index_per_material);
                                    unique_bone_index_per_material += 1;
                                }
                                references
                                    .entry(bone_index)
                                    .or_insert(HashSet::new())
                                    .insert(vertex.base.index);
                            }
                        }
                    }
                    if !index_hash.is_empty() {
                        if references.len() > Self::MAX_BONE_UNIFORMS {
                            let mut vertex_list: Vec<VertexIndex> = vec![];
                            let mut bone_vertex_list: Vec<(usize, Vec<VertexIndex>)> = vec![];
                            for vertex_reference in &references {
                                vertex_list.clear();
                                for vertex in vertex_reference.1 {
                                    vertex_list.push(*vertex)
                                }
                                bone_vertex_list.push((*vertex_reference.0, vertex_list.clone()));
                            }
                            bone_vertex_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                            index_hash.clear();
                            for j in 0..Self::MAX_BONE_UNIFORMS {
                                let pair = &bone_vertex_list[j];
                                index_hash.insert(pair.0, j);
                            }
                            for pair in &mut bone_vertex_list {
                                let all_vertices = &mut pair.1;
                                for vertex_index in all_vertices {
                                    vertices
                                        .get_mut(*vertex_index)
                                        .map(|vertex| vertex.set_skinning_enabled(true));
                                    count_vertex_skinning_needed += 1;
                                }
                            }
                        }
                        bone_index_hash_map.insert(material.base.index, index_hash.clone());
                    }
                    for it in &mut references {
                        it.1.clear()
                    }
                    offset += num_indices;
                    unique_bone_index_per_material = 0;
                    index_hash.clear();
                    references.clear();
                }

                log::trace!("Len(vertices): {}", vertices.len());
                let mut vertex_buffer_data: Vec<VertexUnit> = vec![];
                for vertex in &vertices {
                    vertex_buffer_data.push(vertex.simd.clone().into());
                }
                log::trace!("Len(vertex_buffer): {}", vertex_buffer_data.len());
                let edge_size = Self::internal_edge_size(&bones, camera, edge_size_scale_factor);
                let skin_deformer = Deformer::new(
                    &vertex_buffer_data,
                    &vertices,
                    &bones,
                    &shared_fallback_bone,
                    &morphs,
                    edge_size,
                    device,
                );
                let vertex_buffer_even = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(format!("Model/{}/VertexBuffer/Even", canonical_name).as_str()),
                    size: vertices.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: true,
                });
                let vertex_buffer_odd = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(format!("Model/{}/VertexBuffer/Odd", canonical_name).as_str()),
                    size: vertices.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: true,
                });
                let vertex_buffers = [vertex_buffer_even, vertex_buffer_odd];
                log::trace!("Len(index_buffer): {}", &opaque.vertex_indices.len());
                let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(format!("Model/{}/IndexBuffer", canonical_name).as_str()),
                        contents: bytemuck::cast_slice(&opaque.vertex_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                );
                let mut stage_vertex_buffer_index = 0;
                skin_deformer.execute(&vertex_buffers[stage_vertex_buffer_index], device);
                stage_vertex_buffer_index = 1 - stage_vertex_buffer_index;

                Ok(Self {
                    opaque,
                    skin_deformer,
                    bone_index_hash_map,
                    bones,
                    morphs,
                    constraints,
                    vertices,
                    vertex_indices: indices,
                    materials,
                    labels,
                    rigid_bodies,
                    joints,
                    soft_bodies,
                    bones_by_name,
                    morphs_by_name,
                    bone_to_constraints,
                    active_bone_pair: (None, None),
                    active_morph,
                    outside_parents: HashMap::new(),
                    constraint_joint_bones,
                    inherent_bones,
                    constraint_effector_bones,
                    parent_bone_tree,
                    bone_bound_rigid_bodies: HashMap::new(),
                    // shared_fallback_bone,
                    vertex_buffers,
                    index_buffer,
                    shared_fallback_bone,
                    name,
                    comment,
                    canonical_name,
                    opacity: 1.0f32,
                    count_vertex_skinning_needed,
                    stage_vertex_buffer_index,
                    edge_color: Vector4::new(0f32, 0f32, 0f32, 1f32),
                    edge_size_scale_factor,
                    bounding_box: BoundingBox::new(),
                    states: ModelStates {
                        physics_simulation: true,
                        enable_ground_shadow: true,
                        uploaded: true,
                        dirty: true,
                        visible: true,
                        ..Default::default()
                    },
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
        let mut model = NanoemModel {
            version: nanoem::model::ModelFormatVersion::Pmx2_0,
            codec_type: nanoem::model::CodecType::Utf16Le,
            additional_uv_size: 0u8,
            name_ja: "".to_owned(),
            name_en: "".to_owned(),
            comment_ja: "".to_owned(),
            comment_en: "".to_owned(),
            vertices: vec![],
            vertex_indices: vec![],
            materials: vec![],
            bones: vec![],
            constraints: vec![],
            textures: vec![],
            morphs: vec![],
            labels: vec![],
            rigid_bodies: vec![],
            joints: vec![],
            soft_bodies: vec![],
        };
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
        let center_bone = center_bone;
        let center_bone = model.insert_bone(center_bone, -1)?;
        {
            let mut root_label = nanoem::model::ModelLabel {
                base: nanoem::model::ModelObject { index: 0 },
                name_ja: "Root".to_string(),
                name_en: "Root".to_string(),
                is_special: true,
                items: vec![],
            };
            root_label.insert_item_object(
                nanoem::model::ModelLabelItem::create_from_bone_object(center_bone),
                -1,
            );
            model.insert_label(root_label, -1);
        }
        {
            let mut expression_label = nanoem::model::ModelLabel {
                base: nanoem::model::ModelObject { index: 0 },
                name_ja: Label::NAME_EXPRESSION_IN_JAPANESE.to_string(),
                name_en: "Expression".to_string(),
                is_special: true,
                items: vec![],
            };
            model.insert_label(expression_label, -1);
        }
        model.save_to_buffer(&mut buffer)?;
        Ok(buffer.get_data())
    }

    pub fn upload(&mut self) {
        // TODO
        self.states.uploaded = true;
    }

    pub fn create_all_images(&mut self, texture_lut: &HashMap<String, Rc<wgpu::Texture>>) {
        // TODO: 创建所有材质贴图并绑定到Material上
        for material in &mut self.materials {
            material.diffuse_image = material
                .origin
                .get_diffuse_texture_object(&self.opaque.textures)
                .map(|texture_object| texture_lut.get(&texture_object.path))
                .flatten()
                .map(|rc| rc.clone());
            material.sphere_map_image = material
                .origin
                .get_sphere_map_texture_object(&self.opaque.textures)
                .map(|texture_object| texture_lut.get(&texture_object.path))
                .flatten()
                .map(|rc| rc.clone());
            material.toon_image = material
                .origin
                .get_toon_texture_object(&self.opaque.textures)
                .map(|texture_object| texture_lut.get(&texture_object.path))
                .flatten()
                .map(|rc| rc.clone());
        }
    }

    fn create_image() {}

    pub fn create_all_bone_bounds_rigid_bodies(&mut self) {
        for (handle, rigid_body) in self.rigid_bodies.iter().enumerate() {
            if rigid_body.origin.transform_type != ModelRigidBodyTransformType::FromBoneToSimulation
            {
                if let Ok(bone) = usize::try_from(rigid_body.origin.bone_index) {
                    self.bone_bound_rigid_bodies.insert(bone, handle);
                }
            }
        }
    }

    pub fn clear_all_bone_bounds_rigid_bodies(&mut self) {
        self.bone_bound_rigid_bodies.clear();
    }

    pub fn initialize_all_rigid_bodies_transform_feedback(
        &mut self,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                rigid_body.initialize_transform_feedback(bone, physics_engine);
            }
        }
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        amount: f32,
        timing: SimulationTiming,
        physics_engine: &mut PhysicsEngine,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        let mut visible = true;
        if timing == SimulationTiming::Before {
            if let Some(keyframe) = motion.find_model_keyframe(frame_index) {
                self.edge_color = f128_to_vec4(keyframe.edge_color);
                self.edge_size_scale_factor = keyframe.edge_scale_factor;
                // TODO: do something if having active effect
                visible = keyframe.visible;
                self.set_physics_simulation_enabled(keyframe.is_physics_simulation_enabled);
                self.set_visible(visible);
                self.synchronize_all_constraint_states(
                    keyframe,
                    &motion.opaque.local_bone_motion_track_bundle,
                );
                self.synchronize_all_outside_parents(
                    keyframe,
                    &motion.opaque.global_motion_track_bundle,
                );
            } else {
                if let (Some(prev_frame), Some(next_frame)) =
                    motion.opaque.search_closest_model_keyframes(frame_index)
                {
                    let coef = Motion::coefficient(
                        prev_frame.base.frame_index,
                        next_frame.base.frame_index,
                        frame_index,
                    );
                    self.edge_color = f128_to_vec4(prev_frame.edge_color)
                        .lerp(f128_to_vec4(next_frame.edge_color), coef);
                    self.edge_size_scale_factor = lerp_f32(
                        prev_frame.edge_scale_factor,
                        next_frame.edge_scale_factor,
                        coef,
                    );
                    // TODO: do something if having active effect
                }
                visible = self.states.visible
            }
        }
        if visible {
            match timing {
                SimulationTiming::Before => {
                    self.bounding_box.reset();
                    self.reset_all_materials();
                    self.reset_all_bone_local_transform();
                    self.synchronize_morph_motion(motion, frame_index, amount);
                    self.synchronize_bone_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        physics_engine,
                        outside_parent_bone_map,
                    );
                    self.solve_all_constraints();
                    self.synchronize_all_rigid_body_kinematics(motion, frame_index, physics_engine);
                    self.synchronize_all_rigid_bodies_transform_feedback_to_simulation(
                        physics_engine,
                    );
                }
                SimulationTiming::After => {
                    self.synchronize_bone_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        physics_engine,
                        outside_parent_bone_map,
                    );
                }
            }
        }
    }

    fn synchronize_all_constraint_states(
        &mut self,
        keyframe: &MotionModelKeyframe,
        local_bone_motion_track_bundle: &MotionTrackBundle<MotionBoneKeyframe>,
    ) {
        for state in &keyframe.constraint_states {
            if let Some(constraint) = local_bone_motion_track_bundle
                .resolve_id(state.bone_id)
                .and_then(|name| self.bones_by_name.get(name))
                .and_then(|bone_idx| self.bone_to_constraints.get(bone_idx))
                .and_then(|constraint_idx| self.constraints.get_mut(*constraint_idx))
            {
                constraint.states.enabled = state.enabled;
            }
        }
    }

    fn synchronize_all_outside_parents(
        &mut self,
        keyframe: &MotionModelKeyframe,
        global_motion_track_bundle: &MotionTrackBundle<()>,
    ) {
        self.outside_parents.clear();
        for op in &keyframe.outside_parents {
            if let Some(bound_bone) = global_motion_track_bundle
                .resolve_id(op.local_bone_track_index)
                .and_then(|subject_bone_name| self.bones_by_name.get(subject_bone_name))
                .and_then(|idx| self.bones.get(*idx))
            {
                if let Some(target_object_name) =
                    global_motion_track_bundle.resolve_id(op.global_model_track_index)
                {
                    if let Some(target_bone_name) =
                        global_motion_track_bundle.resolve_id(op.global_bone_track_index)
                    {
                        // TODO: verify model and bone exists
                        self.outside_parents.insert(
                            bound_bone.origin.base.index,
                            (target_object_name.clone(), target_bone_name.clone()),
                        );
                    }
                }
            }
        }
    }

    fn synchronize_all_rigid_body_kinematics(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                let bone_name = bone.canonical_name.clone();
                if let (Some(prev_frame), Some(next_frame)) = (
                    motion.find_bone_keyframe(&bone_name, frame_index),
                    motion.find_bone_keyframe(&bone_name, frame_index + 1),
                ) {
                    if prev_frame.is_physics_simulation_enabled
                        && !next_frame.is_physics_simulation_enabled
                    {
                        rigid_body.enable_kinematic(physics_engine);
                    }
                } else {
                    if let (Some(prev_frame), Some(next_frame)) = motion
                        .opaque
                        .search_closest_bone_keyframes(&bone_name, frame_index)
                    {
                        if prev_frame.is_physics_simulation_enabled
                            && !next_frame.is_physics_simulation_enabled
                        {
                            rigid_body.enable_kinematic(physics_engine);
                        }
                    }
                }
            }
        }
    }

    fn synchronize_all_rigid_bodies_transform_feedback_to_simulation(
        &mut self,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            let bone = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx));
            rigid_body.apply_all_forces(bone, physics_engine);
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                rigid_body.synchronize_transform_feedback_to_simulation(bone, physics_engine);
            }
        }
    }

    pub fn synchronize_all_rigid_bodies_transform_feedback_from_simulation(
        &mut self,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some((bone_idx, bone)) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx).map(|bone| (idx, bone)))
            {
                let parent_bone = usize::try_from(bone.origin.parent_bone_index)
                    .ok()
                    .and_then(|idx| self.bones.get(idx))
                    .map(|bone| bone.clone());
                rigid_body.synchronize_transform_feedback_from_simulation(
                    self.bones.get_mut(bone_idx).unwrap(),
                    (&parent_bone).as_ref(),
                    follow_type,
                    physics_engine,
                );
            }
        }
    }

    fn synchronize_bone_motion(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        amount: f32,
        timing: SimulationTiming,
        physics_engine: &mut PhysicsEngine,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        if let SimulationTiming::Before = timing {
            for bone in &mut self.bones {
                let rigid_body = self
                    .bone_bound_rigid_bodies
                    .get(&bone.origin.base.index)
                    .and_then(|idx| self.rigid_bodies.get_mut(*idx));
                bone.synchronize_motion(motion, rigid_body, frame_index, amount, physics_engine);
            }
        }
        self.apply_all_bones_transform(timing, physics_engine, outside_parent_bone_map);
    }

    fn synchronize_morph_motion(&mut self, motion: &Motion, frame_index: u32, amount: f32) {
        if !self.states.dirty_morph {
            self.reset_all_morphs();
            for morph in &mut self.morphs {
                let name = morph.canonical_name.to_string();
                morph.synchronize_motion(motion, &name, frame_index, amount);
            }
            self.deform_all_morphs(true);
            for morph in &mut self.morphs {
                morph.dirty = false;
            }
            self.states.dirty_morph = true;
        }
    }

    fn solve_all_constraints(&mut self) {
        for constraint in &mut self
            .constraints
            .iter_mut()
            .filter(|constraint| constraint.states.enabled)
        {
            let num_iterations = constraint.origin.num_iterations;
            let angle_limit = constraint.origin.angle_limit;
            let effector_bone_index = usize::try_from(constraint.origin.effector_bone_index).ok();
            let target_bone_index = usize::try_from(constraint.origin.target_bone_index).ok();
            let effector_bone_position = effector_bone_index
                .and_then(|idx| self.bones.get(idx))
                .map(|bone| bone.world_transform_origin().extend(1f32));
            let target_bone_position = target_bone_index
                .and_then(|idx| self.bones.get(idx))
                .map(|bone| bone.world_transform_origin().extend(1f32));
            if let (Some(effector_bone_position), Some(target_bone_position)) =
                (effector_bone_position, target_bone_position)
            {
                for i in 0..num_iterations {
                    for j in 0..constraint.origin.joints.len() {
                        let joint = &constraint.origin.joints[j];
                        if let Some(joint_bone) = usize::try_from(joint.bone_index)
                            .ok()
                            .and_then(|idx| self.bones.get(idx))
                        {
                            if let Some(joint_result) = constraint
                                .joint_iteration_result
                                .get_mut(joint.base.index)
                                .and_then(|results| results.get_mut(i as usize))
                            {
                                if !joint_result.solve_axis_angle(
                                    &joint_bone.matrices.world_transform,
                                    &effector_bone_position,
                                    &target_bone_position,
                                ) {
                                    let new_angle_limit = angle_limit * (j as f32 + 1f32);
                                    let has_unit_x_constraint = joint_bone.has_unit_x_constraint();
                                    if i == 0 && has_unit_x_constraint {
                                        joint_result.axis = Vector3::unit_x();
                                    }
                                    joint_result
                                        .set_transform(&joint_bone.matrices.world_transform);

                                    let orientation = Quaternion::from_axis_angle(
                                        joint_result.axis,
                                        Rad(joint_result.angle.min(new_angle_limit)),
                                    );
                                    let mut mixed_orientation = if i == 0 {
                                        orientation * joint_bone.local_orientation
                                    } else {
                                        joint_bone.constraint_joint_orientation * orientation
                                    };
                                    if has_unit_x_constraint {
                                        let lower_limit =
                                            Vector3::new(0.5f32.to_radians(), 0f32, 0f32);
                                        let upper_limit =
                                            Vector3::new(180f32.to_radians(), 0f32, 0f32);
                                        mixed_orientation = Bone::constraint_orientation(
                                            mixed_orientation,
                                            &upper_limit,
                                            &lower_limit,
                                        );
                                    }
                                    usize::try_from(joint.bone_index)
                                        .ok()
                                        .and_then(|idx| self.bones.get_mut(idx))
                                        .map(|joint_bone| {
                                            joint_bone.constraint_joint_orientation =
                                                mixed_orientation
                                        });

                                    for k in (0..=j).rev() {
                                        let upper_joint = &constraint.origin.joints[k];
                                        let upper_joint_bone_index =
                                            usize::try_from(upper_joint.bone_index).ok();
                                        let parent_origin_world_transform = upper_joint_bone_index
                                            .and_then(|idx| self.bones.get(idx))
                                            .and_then(|upper_joint_bone| {
                                                usize::try_from(
                                                    upper_joint_bone.origin.parent_bone_index,
                                                )
                                                .ok()
                                            })
                                            .and_then(|idx| self.bones.get(idx))
                                            .map(|bone| {
                                                (
                                                    f128_to_vec3(bone.origin.origin),
                                                    bone.matrices.world_transform,
                                                )
                                            });
                                        if let Some(upper_joint_bone) = upper_joint_bone_index
                                            .and_then(|idx| self.bones.get_mut(idx))
                                        {
                                            let translation = upper_joint_bone.local_translation;
                                            let orientation =
                                                upper_joint_bone.constraint_joint_orientation;
                                            upper_joint_bone.update_local_transform_to(
                                                parent_origin_world_transform,
                                                &translation,
                                                &orientation,
                                            )
                                        }
                                    }

                                    let parent_origin_world_transform = effector_bone_index
                                        .and_then(|idx| self.bones.get(idx))
                                        .and_then(|effector_bone| {
                                            usize::try_from(effector_bone.origin.parent_bone_index)
                                                .ok()
                                        })
                                        .and_then(|idx| self.bones.get(idx))
                                        .map(|bone| {
                                            (
                                                f128_to_vec3(bone.origin.origin),
                                                bone.matrices.world_transform,
                                            )
                                        });
                                    effector_bone_index
                                        .and_then(|idx| self.bones.get_mut(idx))
                                        .map(|effector_bone| {
                                            effector_bone.update_local_transform(
                                                parent_origin_world_transform,
                                            )
                                        });
                                    if let Some(effector_result) = constraint
                                        .effector_iteration_result
                                        .get_mut(joint.base.index)
                                        .and_then(|results| results.get_mut(i as usize))
                                    {
                                        effector_bone_index
                                            .and_then(|idx| self.bones.get(idx))
                                            .map(|effector_bone| {
                                                effector_result.set_transform(
                                                    &effector_bone.matrices.world_transform,
                                                )
                                            });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn reset_all_morph_deform_states(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) {
        let mut active_morphs = HashSet::new();
        active_morphs.insert(self.active_morph.eyebrow);
        active_morphs.insert(self.active_morph.eye);
        active_morphs.insert(self.active_morph.lip);
        active_morphs.insert(self.active_morph.other);
        for morph_idx in 0..self.morphs.len() {
            let morph = self.morphs.get(morph_idx).unwrap();
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Bone(children) => {
                    for child in children {
                        if let Some((idx, target_bone)) = usize::try_from(child.bone_index)
                            .ok()
                            .and_then(|idx| self.bones.get_mut(idx).map(|bone| (idx, bone)))
                        {
                            let rigid_body = self
                                .bone_bound_rigid_bodies
                                .get(&idx)
                                .and_then(|idx| self.rigid_bodies.get_mut(*idx));
                            target_bone.reset_morph_transform();
                            target_bone.synchronize_motion(
                                motion,
                                rigid_body,
                                frame_index,
                                0f32,
                                physics_engine,
                            );
                        }
                    }
                }
                nanoem::model::ModelMorphType::Flip(children) => {
                    for target_morph_index in children
                        .iter()
                        .map(|child| usize::try_from(child.morph_index).ok())
                        .collect::<Vec<_>>()
                    {
                        if let Some((idx, target_morph)) = target_morph_index
                            .and_then(|idx| self.morphs.get_mut(idx).map(|morph| (idx, morph)))
                        {
                            if !active_morphs.contains(&Some(idx)) {
                                let name = target_morph.canonical_name.clone();
                                target_morph.synchronize_motion(motion, &name, frame_index, 0f32);
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Group(children) => {
                    for target_morph_index in children
                        .iter()
                        .map(|child| usize::try_from(child.morph_index).ok())
                        .collect::<Vec<_>>()
                    {
                        if let Some((idx, target_morph)) = target_morph_index
                            .and_then(|idx| self.morphs.get_mut(idx).map(|morph| (idx, morph)))
                        {
                            if !active_morphs.contains(&Some(idx)) {
                                let name = target_morph.canonical_name.clone();
                                target_morph.synchronize_motion(motion, &name, frame_index, 0f32);
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Material(children) => {
                    for child in children {
                        if let Some(target_material) = usize::try_from(child.material_index)
                            .ok()
                            .and_then(|idx| self.materials.get_mut(idx))
                        {
                            target_material.reset();
                        }
                    }
                }
                nanoem::model::ModelMorphType::Vertex(_)
                | nanoem::model::ModelMorphType::Texture(_)
                | nanoem::model::ModelMorphType::Uva1(_)
                | nanoem::model::ModelMorphType::Uva2(_)
                | nanoem::model::ModelMorphType::Uva3(_)
                | nanoem::model::ModelMorphType::Uva4(_)
                | nanoem::model::ModelMorphType::Impulse(_) => {}
            }
        }
    }

    fn apply_all_bones_transform(
        &mut self,
        timing: SimulationTiming,
        physics_engine: &mut PhysicsEngine,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        if timing == SimulationTiming::Before {
            self.bounding_box.reset();
        }
        // TODO: Here nanoem use a ordered bone. If any sort to bone happened, I will change logic here.
        for idx in 0..self.bones.len() {
            let bone = &self.bones[idx];
            if (bone.origin.flags.is_affected_by_physics_simulation
                && timing == SimulationTiming::After)
                || (!bone.origin.flags.is_affected_by_physics_simulation
                    && timing == SimulationTiming::Before)
            {
                let is_constraint_joint_bone_active = Some(true)
                    == self
                        .constraint_joint_bones
                        .get(&bone.origin.base.index)
                        .and_then(|idx| self.constraints.get(*idx))
                        .map(|constraint| constraint.states.enabled);
                let parent_bone = usize::try_from(bone.origin.parent_bone_index)
                    .ok()
                    .and_then(|idx| self.bones.get(idx))
                    .map(|b| (b.clone()));
                let parent_bone_and_is_constraint_joint_bone_active =
                    parent_bone.as_ref().map(|bone| {
                        (
                            bone,
                            Some(true)
                                == self
                                    .constraint_joint_bones
                                    .get(&bone.origin.base.index)
                                    .and_then(|idx| self.constraints.get(*idx))
                                    .map(|constraint| constraint.states.enabled),
                        )
                    });
                let effector_bone_local_user_orientation =
                    usize::try_from(bone.origin.effector_bone_index)
                        .ok()
                        .and_then(|idx| self.bones.get(idx))
                        .map(|bone| bone.local_user_orientation);
                let outside_parent_bone = self
                    .outside_parents
                    .get(&idx)
                    .and_then(|op_path| outside_parent_bone_map.get(op_path));
                self.bones[idx].apply_all_local_transform(
                    parent_bone_and_is_constraint_joint_bone_active,
                    effector_bone_local_user_orientation,
                    is_constraint_joint_bone_active,
                );
                if let Some(outside_parent_bone) = outside_parent_bone {
                    self.bones[idx].apply_outside_parent_transform(outside_parent_bone);
                }
                self.bounding_box
                    .set(self.bones[idx].world_transform_origin());
            }
        }
    }

    fn reset_all_bone_transforms(&mut self) {
        for bone in &mut self.bones {
            bone.reset_local_transform();
            bone.reset_morph_transform();
            bone.reset_user_transform();
        }
    }

    fn reset_all_bone_local_transform(&mut self) {
        for bone in &mut self.bones {
            bone.reset_local_transform();
        }
    }

    fn reset_all_bone_morph_transform(&mut self) {
        for bone in &mut self.bones {
            bone.reset_local_transform();
            bone.reset_morph_transform();
        }
    }

    fn reset_all_materials(&mut self) {
        for material in &mut self.materials {
            material.reset();
        }
    }

    fn reset_all_morphs(&mut self) {
        for morph in &mut self.morphs {
            morph.reset();
        }
    }

    fn reset_all_vertices(&mut self) {
        for vertex in &mut self.vertices {
            vertex.reset();
        }
    }

    pub fn set_physics_simulation_enabled(&mut self, value: bool) {
        if self.states.physics_simulation != value {
            self.set_all_physics_objects_enabled(value && self.states.visible);
            self.states.physics_simulation = value;
        }
    }

    pub fn set_visible(&mut self, value: bool) {
        if self.states.visible != value {
            self.set_all_physics_objects_enabled(value & self.states.physics_simulation);
            // TODO: enable effect
            self.states.visible = value;
            // TODO: publish set visible event
        }
    }

    pub fn set_all_physics_objects_enabled(&mut self, value: bool) {
        if value {
            for soft_body in &mut self.soft_bodies {
                soft_body.enable();
            }
            for rigid_body in &mut self.rigid_bodies {
                rigid_body.enable();
            }
            for joint in &mut self.joints {
                joint.enable();
            }
        } else {
            for soft_body in &mut self.soft_bodies {
                soft_body.disable();
            }
            for rigid_body in &mut self.rigid_bodies {
                rigid_body.disable();
            }
            for joint in &mut self.joints {
                joint.disable();
            }
        }
    }

    pub fn deform_all_morphs(&mut self, check_dirty: bool) {
        for morph_idx in 0..self.morphs.len() {
            self.pre_deform_morph(morph_idx);
        }
        for morph_idx in 0..self.morphs.len() {
            self.deform_morph(morph_idx, check_dirty);
        }
    }

    fn pre_deform_morph(&mut self, morph_idx: MorphIndex) {
        if let Some(morph) = self.morphs.get(morph_idx) {
            let weight = morph.weight;
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Group(children) => {
                    for (target_morph_idx, child_weight) in children
                        .iter()
                        .map(|child| (usize::try_from(child.morph_index).ok(), child.weight))
                        .collect::<Vec<_>>()
                    {
                        if let Some(target_morph) =
                            target_morph_idx.and_then(|idx| self.morphs.get_mut(idx))
                        {
                            if let nanoem::model::ModelMorphType::Flip(_) = target_morph.origin.typ
                            {
                                target_morph.set_forced_weight(weight * child_weight);
                                self.pre_deform_morph(target_morph_idx.unwrap());
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Flip(children) => {
                    if weight > 0f32 && children.len() > 0 {
                        let target_idx = (((children.len() + 1) as f32 * weight) as usize - 1)
                            .clamp(0, children.len() - 1);
                        let child = &children[target_idx];
                        let child_weight = child.weight;
                        usize::try_from(child.morph_index)
                            .ok()
                            .and_then(|idx| self.morphs.get_mut(idx))
                            .map(|morph| morph.set_weight(weight));
                    }
                }
                nanoem::model::ModelMorphType::Material(children) => {
                    for child in children {
                        if let Some(material) = usize::try_from(child.material_index)
                            .ok()
                            .and_then(|idx| self.materials.get_mut(idx))
                        {
                            material.reset_deform();
                        } else {
                            for material in &mut self.materials {
                                material.reset_deform();
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Vertex(_)
                | nanoem::model::ModelMorphType::Bone(_)
                | nanoem::model::ModelMorphType::Texture(_)
                | nanoem::model::ModelMorphType::Uva1(_)
                | nanoem::model::ModelMorphType::Uva2(_)
                | nanoem::model::ModelMorphType::Uva3(_)
                | nanoem::model::ModelMorphType::Uva4(_)
                | nanoem::model::ModelMorphType::Impulse(_) => {}
            }
        }
    }

    fn deform_morph(&mut self, morph_idx: MorphIndex, check_dirty: bool) {
        if let Some(morph) = self.morphs.get_mut(morph_idx) {
            if !check_dirty || (check_dirty && morph.dirty) {
                let weight = morph.weight;
                match &morph.origin.typ {
                    nanoem::model::ModelMorphType::Group(children) => {
                        for (child_weight, target_morph_idx) in children
                            .iter()
                            .map(|child| (child.weight, usize::try_from(child.morph_index).ok()))
                            .collect::<Vec<_>>()
                        {
                            if let Some(target_morph) =
                                target_morph_idx.and_then(|idx| self.morphs.get_mut(idx))
                            {
                                if let nanoem::model::ModelMorphType::Flip(_) =
                                    target_morph.origin.typ
                                {
                                    target_morph.set_forced_weight(weight * child_weight);
                                    self.deform_morph(target_morph_idx.unwrap(), false);
                                }
                            }
                        }
                    }

                    nanoem::model::ModelMorphType::Flip(children) => {}
                    nanoem::model::ModelMorphType::Impulse(children) => {
                        for child in children {
                            if let Some(rigid_body) = usize::try_from(child.rigid_body_index)
                                .ok()
                                .and_then(|idx| self.rigid_bodies.get_mut(idx))
                            {
                                let torque = f128_to_vec3(child.torque);
                                let velocity = f128_to_vec3(child.velocity);
                                if torque.abs_diff_eq(
                                    &Vector3::zero(),
                                    Vector3::<f32>::default_epsilon(),
                                ) && velocity.abs_diff_eq(
                                    &Vector3::zero(),
                                    Vector3::<f32>::default_epsilon(),
                                ) {
                                    rigid_body.mark_all_forces_reset();
                                } else if child.is_local {
                                    rigid_body.add_local_torque_force(torque, weight);
                                    rigid_body.add_local_velocity_force(velocity, weight);
                                } else {
                                    rigid_body.add_global_torque_force(torque, weight);
                                    rigid_body.add_global_velocity_force(velocity, weight);
                                }
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Material(children) => {
                        for child in children {
                            if let Some(material) = usize::try_from(child.material_index)
                                .ok()
                                .and_then(|idx| self.materials.get_mut(idx))
                            {
                                material.deform(child, weight);
                            } else {
                                for material in &mut self.materials {
                                    material.deform(child, weight);
                                }
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Bone(children) => {
                        for child in children {
                            if let Some(bone) = usize::try_from(child.bone_index)
                                .ok()
                                .and_then(|idx| self.bones.get_mut(idx))
                            {
                                bone.update_local_morph_transform(child, weight);
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Vertex(children) => {
                        for child in children {
                            if let Some(vertex) = usize::try_from(child.vertex_index)
                                .ok()
                                .and_then(|idx| self.vertices.get_mut(idx))
                            {
                                vertex.deform(child, weight);
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Texture(children)
                    | nanoem::model::ModelMorphType::Uva1(children)
                    | nanoem::model::ModelMorphType::Uva2(children)
                    | nanoem::model::ModelMorphType::Uva3(children)
                    | nanoem::model::ModelMorphType::Uva4(children) => {
                        for child in children {
                            if let Some(vertex) = usize::try_from(child.vertex_index)
                                .ok()
                                .and_then(|idx| self.vertices.get_mut(idx))
                            {
                                vertex.deform_uv(
                                    child,
                                    morph.origin.typ.uv_index().unwrap(),
                                    weight,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn perform_all_bones_transform(
        &mut self,
        physics_engine: &mut PhysicsEngine,
        physics_simulation_time_step: f32,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        self.apply_all_bones_transform(
            SimulationTiming::Before,
            physics_engine,
            outside_parent_bone_map,
        );
        self.solve_all_constraints();
        if physics_engine.simulation_mode == SimulationMode::EnableAnytime {
            self.synchronize_all_rigid_bodies_transform_feedback_to_simulation(physics_engine);
            physics_engine.step(physics_simulation_time_step);
            self.synchronize_all_rigid_bodies_transform_feedback_from_simulation(
                RigidBodyFollowBone::Skip,
                physics_engine,
            );
        }
        self.apply_all_bones_transform(
            SimulationTiming::After,
            physics_engine,
            outside_parent_bone_map,
        );
        self.mark_staging_vertex_buffer_dirty();
        // TODO: handle owned camera
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn vertices_len(&self) -> usize {
        self.vertices.len()
    }

    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    pub fn active_bone(&self) -> Option<&Bone> {
        self.active_bone_pair.0.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active_bone(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.0 != bone_idx {
            // TODO: publish set event
            self.active_bone_pair.0 = bone_idx;
        }
    }

    pub fn morphs(&self) -> &[Morph] {
        &self.morphs
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn active_outside_parent_subject_bone(&self) -> Option<&Bone> {
        self.active_bone_pair.1.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active_outside_parent_subject_bone(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.1 != bone_idx {
            self.active_bone_pair.1 = bone_idx;
        }
    }

    pub fn find_bone(&self, name: &str) -> Option<&Bone> {
        self.bones_by_name
            .get(name)
            .and_then(|index| self.bones.get(*index))
    }

    pub fn find_bone_mut(&mut self, name: &str) -> Option<&mut Bone> {
        self.bones_by_name
            .get(name)
            .and_then(|index| self.bones.get_mut(*index))
    }

    pub fn find_morph(&self, name: &str) -> Option<&Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|index| self.morphs.get(*index))
    }

    pub fn find_morph_mut(&mut self, name: &str) -> Option<&mut Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|index| self.morphs.get_mut(*index))
    }

    pub fn parent_bone(&self, bone: &Bone) -> Option<&Bone> {
        usize::try_from(bone.origin.parent_bone_index)
            .ok()
            .and_then(|idx| self.bones.get(idx))
    }

    fn internal_edge_size(bones: &[Bone], camera: &dyn Camera, edge_size_scale_factor: f32) -> f32 {
        if bones.len() > 1 {
            let bone = &bones[1];
            let bone_position = bone.world_transform_origin();
            (bone_position - camera.position()).magnitude()
                * (camera.fov() as f32 / 30f32).clamp(0f32, 1f32)
                * 0.001f32
                * edge_size_scale_factor
        } else {
            0f32
        }
    }

    pub fn edge_size(&self, camera: &dyn Camera) -> f32 {
        Self::internal_edge_size(&self.bones, camera, self.edge_size_scale_factor)
    }

    pub fn edge_size_scale_factor(&self) -> f32 {
        self.edge_size_scale_factor
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_canonical_name(&self) -> &str {
        &self.canonical_name
    }

    pub fn is_add_blend_enabled(&self) -> bool {
        self.states.enable_add_blend
    }

    pub fn is_physics_simulation_enabled(&self) -> bool {
        self.states.physics_simulation
    }

    pub fn is_staging_vertex_buffer_dirty(&self) -> bool {
        self.states.dirty_staging_buffer
    }

    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    pub fn world_transform(&self, initial: &Matrix4<f32>) -> Matrix4<f32> {
        initial.clone()
    }

    pub fn contains_bone(&self, name: &str) -> bool {
        self.bones_by_name.contains_key(name)
    }

    pub fn contains_morph(&self, name: &str) -> bool {
        self.morphs_by_name.contains_key(name)
    }
}

impl Drawable for Model {
    fn draw(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        typ: DrawType,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) {
        if self.is_visible() {
            match typ {
                DrawType::Color | DrawType::ScriptExternalColor => {
                    self.draw_color(color_view, depth_view, project, device, queue, adapter_info)
                }
                DrawType::Edge => {
                    let edge_size_scale_factor = self.edge_size(project.active_camera());
                    if edge_size_scale_factor > 0f32 {
                        self.draw_edge(
                            edge_size_scale_factor,
                            color_view,
                            depth_view,
                            project,
                            device,
                            queue,
                            adapter_info,
                        );
                    }
                }
                DrawType::GroundShadow => {
                    if self.states.enable_ground_shadow {
                        self.draw_ground_shadow(
                            color_view,
                            depth_view,
                            project,
                            device,
                            queue,
                            adapter_info,
                        )
                    }
                }
                DrawType::ShadowMap => {
                    if self.states.enable_shadow_map {
                        self.draw_shadow_map(
                            color_view,
                            depth_view,
                            project,
                            device,
                            queue,
                            adapter_info,
                        )
                    }
                }
            }
        }
    }

    fn is_visible(&self) -> bool {
        self.states.visible
    }
}

impl Model {
    pub fn mark_staging_vertex_buffer_dirty(&mut self) {
        self.states.dirty_staging_buffer = true;
    }

    pub fn update_staging_vertex_buffer(&mut self, device: &wgpu::Device) {
        if self.states.dirty_staging_buffer {
            self.skin_deformer
                .execute(&self.vertex_buffers[self.stage_vertex_buffer_index], device);
            self.stage_vertex_buffer_index = 1 - self.stage_vertex_buffer_index;
            self.states.dirty_morph = false;
            self.states.dirty_staging_buffer = false;
        }
    }

    fn draw_color(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) {
        let mut index_offset = 0usize;
        for material in &self.materials {
            let num_indices = material.origin.num_vertex_indices;
            log::trace!(
                "Render next Material, Index count: {}; Offset: {}",
                num_indices,
                index_offset
            );
            let buffer = pass::Buffer::new(
                num_indices,
                index_offset,
                &self.vertex_buffers[1 - self.stage_vertex_buffer_index as usize],
                &self.index_buffer,
                true,
            );
            if material.is_visible() {
                // TODO: get technique by discovery
                let mut technique =
                    ObjectTechnique::new(device, material.origin.flags.is_point_draw_enabled);
                let technique_type = technique.technique_type();
                while let Some(pass) = technique.execute(device) {
                    pass.set_global_parameters(self, project);
                    pass.set_camera_parameters(
                        project.active_camera(),
                        &Self::INITIAL_WORLD_MATRIX,
                        self,
                    );
                    pass.set_light_parameters(project.global_light(), false);
                    pass.set_all_model_parameters(self, project);
                    pass.set_material_parameters(
                        &material,
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
                        color_view,
                        depth_view,
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
            index_offset += num_indices;
        }
    }

    fn draw_edge(
        &self,
        edge_size_scale_factor: f32,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) {
        let mut index_offset = 0usize;
        for material in &self.materials {
            let num_indices = material.origin.num_vertex_indices;
            if material.origin.flags.is_edge_enabled
                && !material.origin.flags.is_line_draw_enabled
                && !material.origin.flags.is_point_draw_enabled
            {
                let buffer = pass::Buffer::new(
                    num_indices,
                    index_offset,
                    &self.vertex_buffers[1 - self.stage_vertex_buffer_index as usize],
                    &self.index_buffer,
                    true,
                );
                if material.is_visible() {
                    let mut technique = EdgeTechnique::new(device);
                    let technique_type = technique.technique_type();
                    while let Some(pass) = technique.execute(device) {
                        pass.set_global_parameters(self, project);
                        pass.set_camera_parameters(
                            project.active_camera(),
                            &Self::INITIAL_WORLD_MATRIX,
                            self,
                        );
                        pass.set_light_parameters(project.global_light(), false);
                        pass.set_all_model_parameters(self, project);
                        pass.set_material_parameters(
                            material,
                            technique_type,
                            project.shared_fallback_image(),
                        );
                        pass.set_edge_parameters(
                            material,
                            edge_size_scale_factor,
                            project.shared_fallback_image(),
                        );
                        pass.execute(
                            &buffer,
                            color_view,
                            depth_view,
                            technique_type,
                            device,
                            queue,
                            self,
                            project,
                        );
                    }
                    // if (!technique->hasNextScriptCommand()) {
                    // technique->resetScriptCommandState();
                    // }
                }
            }
            index_offset += num_indices;
        }
    }

    fn draw_ground_shadow(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) {
        let mut index_offset = 0usize;
        let world = project.global_light().get_shadow_transform();
        for material in &self.materials {
            let num_indices = material.origin.num_vertex_indices;
            if material.origin.flags.is_casting_shadow_enabled
                && !material.origin.flags.is_point_draw_enabled
            {
                let buffer = pass::Buffer::new(
                    num_indices,
                    index_offset,
                    &self.vertex_buffers[1 - self.stage_vertex_buffer_index as usize],
                    &self.index_buffer,
                    true,
                );
                if material.is_visible() {
                    let mut technique = GroundShadowTechnique::new(device);
                    let technique_type = technique.technique_type();
                    while let Some(pass) = technique.execute(device) {
                        pass.set_global_parameters(self, project);
                        pass.set_camera_parameters(project.active_camera(), &world, self);
                        pass.set_light_parameters(project.global_light(), false);
                        pass.set_all_model_parameters(self, project);
                        pass.set_material_parameters(
                            material,
                            technique_type,
                            project.shared_fallback_image(),
                        );
                        pass.set_ground_shadow_parameters(
                            project.global_light(),
                            project.active_camera(),
                            &world,
                            project.shared_fallback_image(),
                        );
                        pass.execute(
                            &buffer,
                            color_view,
                            depth_view,
                            technique_type,
                            device,
                            queue,
                            self,
                            project,
                        );
                    }
                    // if (!technique->hasNextScriptCommand()) {
                    // technique->resetScriptCommandState();
                    // }
                }
            }
            index_offset += num_indices;
        }
    }

    fn draw_shadow_map(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) {
        let mut index_offset = 0usize;
        for material in &self.materials {
            let num_indices = material.origin.num_vertex_indices;
            if material.is_casting_shadow_map_enabled() && !material.is_point_draw_enabled() {
                let buffer = pass::Buffer::new(
                    num_indices,
                    index_offset,
                    &self.vertex_buffers[1 - self.stage_vertex_buffer_index as usize],
                    &self.index_buffer,
                    true,
                );
                if material.is_visible() {
                    let mut technique = ZplotTechnique::new(device);
                    let technique_type = technique.technique_type();
                    while let Some(pass) = technique.execute(device) {
                        pass.set_global_parameters(self, project);
                        pass.set_camera_parameters(
                            project.active_camera(),
                            &Self::INITIAL_WORLD_MATRIX,
                            self,
                        );
                        pass.set_light_parameters(project.global_light(), false);
                        pass.set_all_model_parameters(self, project);
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
                            color_view,
                            depth_view,
                            technique_type,
                            device,
                            queue,
                            self,
                            project,
                        );
                        // TODO: process technique hasNextScriptCommand
                    }
                }
            }
            index_offset += num_indices;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Matrices {
    pub world_transform: Matrix4<f32>,
    pub local_transform: Matrix4<f32>,
    pub normal_transform: Matrix4<f32>,
    pub skinning_transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoneKeyframeInterpolation {
    pub translation_x: KeyframeInterpolationPoint,
    pub translation_y: KeyframeInterpolationPoint,
    pub translation_z: KeyframeInterpolationPoint,
    pub orientation: KeyframeInterpolationPoint,
}

impl BoneKeyframeInterpolation {
    pub fn build(interpolation: nanoem::motion::MotionBoneKeyframeInterpolation) -> Self {
        Self {
            translation_x: KeyframeInterpolationPoint::build(interpolation.translation_x),
            translation_y: KeyframeInterpolationPoint::build(interpolation.translation_y),
            translation_z: KeyframeInterpolationPoint::build(interpolation.translation_z),
            orientation: KeyframeInterpolationPoint::build(interpolation.orientation),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BezierControlPoints {
    translation_x: Vector4<u8>,
    translation_y: Vector4<u8>,
    translation_z: Vector4<u8>,
    orientation: Vector4<u8>,
}

struct BoneFrameTransform {
    pub translation: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub interpolation: BoneKeyframeInterpolation,
}

impl Default for BoneFrameTransform {
    fn default() -> Self {
        Self {
            translation: Vector3::zero(),
            orientation: Quaternion::zero(),
            interpolation: BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: Bone::DEFAULT_BEZIER_CONTROL_POINT.into(),
                    is_linear_interpolation: true,
                },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoneStates {
    pub dirty: bool,
    pub editing_masked: bool,
}

#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub canonical_name: String,
    pub matrices: Matrices,
    pub local_orientation: Quaternion<f32>,
    pub local_inherent_orientation: Quaternion<f32>,
    pub local_morph_orientation: Quaternion<f32>,
    pub local_user_orientation: Quaternion<f32>,
    pub constraint_joint_orientation: Quaternion<f32>,
    pub local_translation: Vector3<f32>,
    pub local_inherent_translation: Vector3<f32>,
    pub local_morph_translation: Vector3<f32>,
    pub local_user_translation: Vector3<f32>,
    pub interpolation: BoneKeyframeInterpolation,
    pub states: BoneStates,
    pub origin: NanoemBone,
}

impl Bone {
    pub const DEFAULT_BEZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    pub const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
    const NAME_ROOT_PARENT_IN_JAPANESE_UTF8: &'static [u8] = &[
        0xe5, 0x85, 0xa8, 0xe3, 0x81, 0xa6, 0xe3, 0x81, 0xae, 0xe8, 0xa6, 0xaa, 0x0,
    ];
    pub const NAME_ROOT_PARENT_IN_JAPANESE: &'static str = "全ての親";
    const NAME_CENTER_IN_JAPANESE_UTF8: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0,
    ];
    pub const NAME_CENTER_IN_JAPANESE: &'static str = "センター";
    const NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE_UTF8: &'static [u8] = &[
        0xe6, 0x93, 0x8d, 0xe4, 0xbd, 0x9c, 0xe4, 0xb8, 0xad, 0xe5, 0xbf, 0x83, 0,
    ];
    pub const NAME_CENTER_OF_VIEWPOINT_IN_JAPANESE: &'static str = "操作中心";
    const NAME_CENTER_OFFSET_IN_JAPANESE_UTF8: &'static [u8] = &[
        0xe3, 0x82, 0xbb, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xbf, 0xe3, 0x83, 0xbc, 0xe5, 0x85, 0x88, 0,
    ];
    pub const NAME_CENTER_OFFSET_IN_JAPANESE: &'static str = "セコター先";
    const NAME_LEFT_IN_JAPANESE_UTF8: &'static [u8] = &[0xe5, 0xb7, 0xa6, 0x0];
    pub const NAME_LEFT_IN_JAPANESE: &'static str = "左";
    const NAME_RIGHT_IN_JAPANESE_UTF8: &'static [u8] = &[0xe5, 0x8f, 0xb3, 0x0];
    pub const NAME_RIGHT_IN_JAPANESE: &'static str = "右";
    const NAME_DESTINATION_IN_JAPANESE_UTF8: &'static [u8] = &[0xe5, 0x85, 0x88, 0x0];
    pub const NAME_DESTINATION_IN_JAPANESE: &'static str = "先";
    const LEFT_KNEE_IN_JAPANESE_UTF8: &'static [u8] =
        &[0xe5, 0xb7, 0xa6, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];
    pub const LEFT_KNEE_IN_JAPANESE: &'static str = "左ひざ";
    const RIGHT_KNEE_IN_JAPANESE_UTF8: &'static [u8] =
        &[0xe5, 0x8f, 0xb3, 0xe3, 0x81, 0xb2, 0xe3, 0x81, 0x96, 0x0];
    pub const RIGHT_KNEE_IN_JAPANESE: &'static str = "右ひざ";

    pub fn from_nanoem(bone: &NanoemBone, language_type: nanoem::common::LanguageType) -> Self {
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
            interpolation: BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: Vector4::new(0u8, 0u8, 0u8, 0u8),
                    is_linear_interpolation: true,
                },
            },
            states: BoneStates::default(),
            origin: bone.clone(),
        }
    }

    fn translate(v: &Vector3<f32>, m: &Matrix4<f32>) -> Matrix4<f32> {
        let mut result = *m;
        result[3] = Vector4::new(1f32, 1f32, 1f32, 1f32);
        result[3] = result * v.extend(0f32);
        result
    }

    fn shrink_3x3(m: &Matrix4<f32>) -> Matrix4<f32> {
        let mut result = *m;
        result[3] = Vector4::new(0f32, 0f32, 0f32, 1f32);
        return result;
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        rigid_body: Option<&mut RigidBody>,
        frame_index: u32,
        amount: f32,
        physics_engine: &mut PhysicsEngine,
    ) {
        let t0 = self.synchronize_transform(motion, rigid_body, frame_index, physics_engine);
        if amount > 0f32 {
            let t1 = self.synchronize_transform(motion, None, frame_index + 1, physics_engine);
            self.local_user_translation = t0.translation.lerp(t1.translation, amount);
            self.local_user_orientation = t0.orientation.slerp(t1.orientation, amount);
            self.interpolation = BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: t0
                        .interpolation
                        .translation_x
                        .bezier_control_point
                        .map(|v| v as f32)
                        .lerp(
                            t1.interpolation
                                .translation_x
                                .bezier_control_point
                                .map(|v| v as f32),
                            amount,
                        )
                        .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
                    is_linear_interpolation: t0.interpolation.translation_x.is_linear_interpolation,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: t0
                        .interpolation
                        .translation_y
                        .bezier_control_point
                        .map(|v| v as f32)
                        .lerp(
                            t1.interpolation
                                .translation_y
                                .bezier_control_point
                                .map(|v| v as f32),
                            amount,
                        )
                        .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
                    is_linear_interpolation: t0.interpolation.translation_y.is_linear_interpolation,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: t0
                        .interpolation
                        .translation_z
                        .bezier_control_point
                        .map(|v| v as f32)
                        .lerp(
                            t1.interpolation
                                .translation_z
                                .bezier_control_point
                                .map(|v| v as f32),
                            amount,
                        )
                        .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
                    is_linear_interpolation: t0.interpolation.translation_z.is_linear_interpolation,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: t0
                        .interpolation
                        .orientation
                        .bezier_control_point
                        .map(|v| v as f32)
                        .lerp(
                            t1.interpolation
                                .orientation
                                .bezier_control_point
                                .map(|v| v as f32),
                            amount,
                        )
                        .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
                    is_linear_interpolation: t0.interpolation.orientation.is_linear_interpolation,
                },
            };
        } else {
            self.local_user_translation = t0.translation;
            self.local_user_orientation = t0.orientation;
            self.interpolation = BoneKeyframeInterpolation {
                translation_x: KeyframeInterpolationPoint {
                    bezier_control_point: t0.interpolation.translation_x.bezier_control_point,
                    is_linear_interpolation: t0.interpolation.translation_x.is_linear_interpolation,
                },
                translation_y: KeyframeInterpolationPoint {
                    bezier_control_point: t0.interpolation.translation_y.bezier_control_point,
                    is_linear_interpolation: t0.interpolation.translation_y.is_linear_interpolation,
                },
                translation_z: KeyframeInterpolationPoint {
                    bezier_control_point: t0.interpolation.translation_z.bezier_control_point,
                    is_linear_interpolation: t0.interpolation.translation_z.is_linear_interpolation,
                },
                orientation: KeyframeInterpolationPoint {
                    bezier_control_point: t0.interpolation.orientation.bezier_control_point,
                    is_linear_interpolation: t0.interpolation.orientation.is_linear_interpolation,
                },
            };
        }
    }

    fn synchronize_transform(
        &self,
        motion: &Motion,
        rigid_body: Option<&mut RigidBody>,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) -> BoneFrameTransform {
        if let Some(keyframe) = motion.find_bone_keyframe(&self.canonical_name, frame_index) {
            return BoneFrameTransform {
                translation: f128_to_vec3(keyframe.translation),
                orientation: f128_to_quat(keyframe.orientation),
                interpolation: BoneKeyframeInterpolation::build(keyframe.interpolation),
            };
        } else {
            if let (Some(prev_frame), Some(next_frame)) = motion
                .opaque
                .search_closest_bone_keyframes(&self.canonical_name, frame_index)
            {
                let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
                let coef = Motion::coefficient(
                    prev_frame.base.frame_index,
                    next_frame.base.frame_index,
                    frame_index,
                );
                let prev_translation = f128_to_vec3(prev_frame.translation);
                let next_translation = f128_to_vec3(next_frame.translation);
                let prev_orientation = f128_to_quat(prev_frame.orientation);
                let next_orientation = f128_to_quat(next_frame.orientation);
                let prev_enabled = prev_frame.is_physics_simulation_enabled;
                let next_enabled = next_frame.is_physics_simulation_enabled;
                let frame_transform = if prev_enabled && !next_enabled && rigid_body.is_some() {
                    BoneFrameTransform {
                        translation: self.local_user_translation.lerp(next_translation, coef),
                        orientation: self.local_user_orientation.slerp(next_orientation, coef),
                        interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    }
                } else {
                    BoneFrameTransform {
                        translation: Vector3::new(
                            motion.lerp_value_interpolation(
                                &next_frame.interpolation.translation_x,
                                prev_translation.x,
                                next_translation.x,
                                interval,
                                coef,
                            ),
                            motion.lerp_value_interpolation(
                                &next_frame.interpolation.translation_y,
                                prev_translation.y,
                                next_translation.y,
                                interval,
                                coef,
                            ),
                            motion.lerp_value_interpolation(
                                &next_frame.interpolation.translation_z,
                                prev_translation.z,
                                next_translation.z,
                                interval,
                                coef,
                            ),
                        ),
                        orientation: motion.slerp_interpolation(
                            &next_frame.interpolation.orientation,
                            &prev_orientation,
                            &next_orientation,
                            interval,
                            coef,
                        ),
                        interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    }
                };
                if prev_enabled && next_enabled {
                    if let Some(rigid_body) = rigid_body {
                        rigid_body.disable_kinematic(physics_engine);
                    }
                }
                return frame_transform;
            } else {
                return BoneFrameTransform::default();
            }
        }
    }

    pub fn update_local_transform(
        &mut self,
        parent_origin_world_transform: Option<(Vector3<f32>, Matrix4<f32>)>,
    ) {
        if self
            .local_translation
            .abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
            && self
                .local_orientation
                .abs_diff_eq(&Quaternion::zero(), Quaternion::<f32>::default_epsilon())
        {
            self.update_local_transform_to(
                parent_origin_world_transform,
                &Vector3::zero(),
                &Quaternion::zero(),
            );
        } else {
            let translation = self.local_translation;
            let orientation = self.local_orientation;
            self.update_local_transform_to(
                parent_origin_world_transform,
                &translation,
                &orientation,
            );
        }
    }

    pub fn update_local_transform_to(
        &mut self,
        parent_origin_world_transform: Option<(Vector3<f32>, Matrix4<f32>)>,
        translation: &Vector3<f32>,
        orientation: &Quaternion<f32>,
    ) {
        let translation_matrix = Matrix4 {
            x: Vector4::unit_x(),
            y: Vector4::unit_y(),
            z: Vector4::unit_z(),
            w: translation.extend(1f32),
        };
        let local_transform = Matrix4::<f32>::from(*orientation) * translation_matrix;
        let bone_origin = f128_to_vec3(self.origin.origin);
        if let Some((parent_origin, parent_world_transform)) = parent_origin_world_transform {
            let offset = bone_origin - parent_origin;
            let offset_matrix = Matrix4 {
                x: Vector4::unit_x(),
                y: Vector4::unit_y(),
                z: Vector4::unit_z(),
                w: offset.extend(1f32),
            };
            let parent_world_transform = parent_world_transform;
            let local_transform_with_offset = local_transform * offset_matrix;
            self.matrices.world_transform = local_transform_with_offset * parent_world_transform;
        } else {
            let offset_matrix = Matrix4 {
                x: Vector4::unit_x(),
                y: Vector4::unit_y(),
                z: Vector4::unit_z(),
                w: bone_origin.extend(1f32),
            };
            self.matrices.world_transform = local_transform * offset_matrix;
        }
        self.matrices.local_transform = local_transform;
        self.matrices.skinning_transform =
            Self::translate(&-bone_origin, &self.matrices.world_transform);
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn update_local_morph_transform(
        &mut self,
        morph: &nanoem::model::ModelMorphBone,
        weight: f32,
    ) {
        self.local_morph_translation =
            Vector3::zero().lerp(f128_to_vec3(morph.translation), weight);
        self.local_morph_orientation =
            Quaternion::zero().slerp(f128_to_quat(morph.orientation), weight);
    }

    pub fn update_local_orientation(
        &mut self,
        parent_bone_and_is_constraint_joint_bone_active: Option<(&Self, bool)>,
        effector_bone_local_user_orientation: Option<Quaternion<f32>>,
        is_constraint_joint_bone_active: bool,
    ) {
        // let is_constraint_joint_bone_active = Some(true)
        //     == model
        //         .constraint_joint_bones
        //         .get(&self.origin.base.index)
        //         .and_then(|idx| model.constraints.get(*idx))
        //         .map(|constraint| constraint.states.enabled);
        if self.origin.flags.has_inherent_orientation {
            let mut orientation = Quaternion::<f32>::zero();
            // if let Some(parent_bone) = usize::try_from(self.origin.parent_bone_index)
            //     .ok()
            //     .and_then(|idx| model.bones.get(idx))
            if let Some((parent_bone, parent_is_constraint_joint_bone_active)) =
                parent_bone_and_is_constraint_joint_bone_active
            {
                if parent_bone.origin.flags.has_local_inherent {
                    orientation = Quaternion::<f32>::from(mat4_truncate(
                        parent_bone.matrices.local_transform,
                    )) * orientation;
                } else if parent_is_constraint_joint_bone_active
                // } else if let Some(true) = model
                //     .constraint_joint_bones
                //     .get(&parent_bone.origin.base.index)
                //     .and_then(|idx| model.constraints.get(*idx))
                //     .map(|constraint| constraint.states.enabled)
                {
                    orientation = parent_bone.constraint_joint_orientation * orientation;
                } else {
                    if parent_bone.origin.flags.has_inherent_orientation {
                        orientation = parent_bone.local_inherent_orientation * orientation;
                    } else {
                        orientation = parent_bone.local_user_orientation * orientation;
                    }
                }
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                if let Some(effector_bone_local_user_orientation) =
                    effector_bone_local_user_orientation
                {
                    orientation =
                        orientation.slerp(effector_bone_local_user_orientation, coefficient);
                } else {
                    orientation = Quaternion::zero().slerp(orientation, coefficient);
                }
            }
            let local_orientation = if is_constraint_joint_bone_active {
                (self.constraint_joint_orientation * self.local_morph_orientation * orientation)
                    .normalize()
            } else {
                (self.local_morph_orientation * self.local_user_orientation * orientation)
                    .normalize()
            };
            self.local_orientation = local_orientation;
            self.local_inherent_orientation = orientation;
        } else if is_constraint_joint_bone_active {
            self.local_orientation =
                (self.constraint_joint_orientation * self.local_morph_orientation).normalize();
        } else {
            self.local_orientation =
                (self.local_morph_orientation * self.local_user_orientation).normalize();
        }
    }

    fn update_local_translation(&mut self, parent_bone: Option<&Bone>) {
        let mut translation = self.local_user_translation;
        if self.origin.flags.has_inherent_translation {
            if let Some(parent_bone) = parent_bone {
                if parent_bone.origin.flags.has_local_inherent {
                    translation += parent_bone.matrices.local_transform[3].truncate();
                } else if parent_bone.origin.flags.has_inherent_translation {
                    translation += parent_bone.local_inherent_translation;
                } else {
                    translation += parent_bone
                        .local_translation
                        .mul_element_wise(parent_bone.local_morph_translation);
                }
            }
            let coefficient = self.origin.inherent_coefficient;
            if (coefficient - 1f32).abs() > 0.0f32 {
                translation *= coefficient;
            }
            self.local_inherent_translation = translation;
        }
        translation += self.local_morph_translation;
        self.local_translation = translation;
    }

    pub fn update_skinning_transform(&mut self, skinning_transform: Matrix4<f32>) {
        self.matrices.skinning_transform = skinning_transform;
        self.matrices.world_transform =
            Self::translate(&f128_to_vec3(self.origin.origin), &skinning_transform);
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn apply_all_local_transform(
        &mut self,
        parent_bone_and_is_constraint_joint_bone_active: Option<(&Self, bool)>,
        effector_bone_local_user_orientation: Option<Quaternion<f32>>,
        is_constraint_joint_bone_active: bool,
    ) {
        self.update_local_orientation(
            parent_bone_and_is_constraint_joint_bone_active,
            effector_bone_local_user_orientation,
            is_constraint_joint_bone_active,
        );
        self.update_local_translation(parent_bone_and_is_constraint_joint_bone_active.map(|v| v.0));
        self.update_local_transform(parent_bone_and_is_constraint_joint_bone_active.map(|v| {
            (
                f128_to_vec3(v.0.origin.origin),
                v.0.matrices.world_transform,
            )
        }));
        // We deprecate constraint embedded in bone. All constraints saved in model.constraints
        // self.solve_constraint(constraint, num_iterations, bones)
    }

    pub fn apply_outside_parent_transform(&mut self, outside_parent_bone: &Bone) {
        let inv_origin = -f128_to_vec3(self.origin.origin);
        let out = Self::translate(&inv_origin, &self.matrices.world_transform);
        self.matrices.world_transform = outside_parent_bone.matrices.world_transform * out;
        let out = Self::translate(&inv_origin, &self.matrices.world_transform);
        self.matrices.local_transform = out;
        self.matrices.skinning_transform = out;
        self.matrices.normal_transform = Self::shrink_3x3(&self.matrices.world_transform);
    }

    pub fn reset_local_transform(&mut self) {
        self.local_orientation = Quaternion::zero();
        self.local_inherent_orientation = Quaternion::zero();
        self.local_translation = Vector3::zero();
        self.local_inherent_translation = Vector3::zero();
        self.interpolation = BoneKeyframeInterpolation {
            translation_x: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            translation_y: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            translation_z: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
            orientation: KeyframeInterpolationPoint {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            },
        };
    }

    pub fn reset_morph_transform(&mut self) {
        self.local_morph_orientation = Quaternion::zero();
        self.local_morph_translation = Vector3::zero();
    }

    pub fn reset_user_transform(&mut self) {
        self.local_user_orientation = Quaternion::zero();
        self.local_user_translation = Vector3::zero();
        self.states.dirty = false;
    }

    pub fn world_transform_origin(&self) -> Vector3<f32> {
        self.matrices.world_transform[3].truncate()
    }

    pub fn has_unit_x_constraint(&self) -> bool {
        &self.canonical_name == Self::LEFT_KNEE_IN_JAPANESE
            || &self.canonical_name == Self::RIGHT_KNEE_IN_JAPANESE
    }

    pub fn constraint_orientation(
        orientation: Quaternion<f32>,
        upper_limit: &Vector3<f32>,
        lower_limit: &Vector3<f32>,
    ) -> Quaternion<f32> {
        let rad_90_deg = 90f32.to_radians();
        let matrix: Matrix3<f32> = orientation.into();
        if lower_limit.x > -rad_90_deg && upper_limit.x < rad_90_deg {
            let radians = Vector3::new(
                matrix[1][2].asin(),
                (-matrix[0][2]).atan2(matrix[2][2]),
                (-matrix[1][0]).atan2(matrix[1][1]),
            );
            let r = radians.clamp_element_wise(*lower_limit, *upper_limit);
            let x = Quaternion::from_axis_angle(Vector3::unit_x(), Rad(r.x));
            let y = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(r.y));
            let z = Quaternion::from_axis_angle(Vector3::unit_z(), Rad(r.z));
            z * x * y
        } else if lower_limit.y > -rad_90_deg && upper_limit.y < rad_90_deg {
            let radians = Vector3::new(
                (-matrix[2][1]).atan2(matrix[2][2]),
                matrix[2][0].asin(),
                (-matrix[1][0]).atan2(matrix[0][0]),
            );
            let r = radians.clamp_element_wise(*lower_limit, *upper_limit);
            let x = Quaternion::from_axis_angle(Vector3::unit_x(), Rad(r.x));
            let y = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(r.y));
            let z = Quaternion::from_axis_angle(Vector3::unit_z(), Rad(r.z));
            x * y * z
        } else {
            let radians = Vector3::new(
                (-matrix[2][1]).atan2(matrix[1][1]),
                (-matrix[0][2]).atan2(matrix[0][0]),
                matrix[0][1].asin(),
            );
            let r = radians.clamp_element_wise(*lower_limit, *upper_limit);
            let x = Quaternion::from_axis_angle(Vector3::unit_x(), Rad(r.x));
            let y = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(r.y));
            let z = Quaternion::from_axis_angle(Vector3::unit_z(), Rad(r.z));
            y * z * x
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstraintJoint {
    orientation: Quaternion<f32>,
    translation: Vector3<f32>,
    target_direction: Vector3<f32>,
    effector_direction: Vector3<f32>,
    axis: Vector3<f32>,
    angle: f32,
}

impl ConstraintJoint {
    pub fn set_transform(&mut self, v: &Matrix4<f32>) {
        self.orientation = (Matrix3 {
            x: v.x.truncate(),
            y: v.y.truncate(),
            z: v.z.truncate(),
        })
        .into();
        self.translation = v[3].truncate();
    }

    pub fn solve_axis_angle(
        &mut self,
        transform: &Matrix4<f32>,
        effector_position: &Vector4<f32>,
        target_position: &Vector4<f32>,
    ) -> bool {
        let inv_transform = transform.affine_invert().unwrap();
        let inv_effector_position = (inv_transform * effector_position).truncate();
        let inv_target_position = (inv_transform * target_position).truncate();
        if inv_effector_position.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
            || inv_target_position.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon())
        {
            return true;
        }
        let effector_direction = inv_effector_position.normalize();
        let target_direction = inv_target_position.normalize();
        let axis = effector_direction.cross(target_direction);
        self.effector_direction = effector_direction;
        self.target_direction = target_direction;
        self.axis = axis;
        if axis.abs_diff_eq(&Vector3::zero(), Vector3::<f32>::default_epsilon()) {
            return true;
        }
        let z = effector_direction
            .dot(target_direction)
            .clamp(-1.0f32, 1.0f32);
        self.axis = axis.normalize();
        if z.abs() <= f32::default_epsilon() {
            return true;
        }
        self.angle = z.acos();
        return false;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintStates {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    name: String,
    canonical_name: String,
    joint_iteration_result: Vec<Vec<ConstraintJoint>>,
    effector_iteration_result: Vec<Vec<ConstraintJoint>>,
    pub states: ConstraintStates,
    pub origin: NanoemConstraint,
}

impl Constraint {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = Self::PRIVATE_STATE_ENABLED;

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
            canonical_name = format!("Constraint{}", constraint.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let mut joint_iteration_result = vec![vec![]; constraint.joints.len()];
        let mut effector_iteration_result = vec![vec![]; constraint.joints.len()];
        Self {
            name,
            canonical_name,
            joint_iteration_result,
            effector_iteration_result,
            states: ConstraintStates { enabled: true },
            origin: constraint.clone(),
        }
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
    material: Option<MaterialIndex>,
    soft_body: Option<SoftBodyIndex>,
    bones: [Option<BoneIndex>; 4],
    states: u32,
    pub simd: VertexSimd,
    pub origin: NanoemVertex,
}

impl Vertex {
    const PRIVATE_STATE_SKINNING_ENABLED: u32 = 1 << 1;
    const PRIVATE_STATE_EDITING_MASKED: u32 = 1 << 2;
    const PRIVATE_STATE_INITIAL_VALUE: u32 = 0;

    fn from_nanoem(vertex: &NanoemVertex) -> Self {
        let direction = Vector4::new(1f32, 1f32, 1f32, 1f32);
        let texcoord = vertex.get_tex_coord();
        let bone_indices: [f32; 4] = vertex.get_bone_indices().map(|idx| idx as f32);
        let mut bones = [None; 4];
        match vertex.typ {
            nanoem::model::ModelVertexType::UNKNOWN => {}
            nanoem::model::ModelVertexType::BDEF1 => {
                bones[0] = vertex
                    .bone_indices
                    .get(0)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
            }
            nanoem::model::ModelVertexType::BDEF2 | nanoem::model::ModelVertexType::SDEF => {
                bones[0] = vertex
                    .bone_indices
                    .get(0)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
                bones[1] = vertex
                    .bone_indices
                    .get(1)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
            }
            nanoem::model::ModelVertexType::BDEF4 | nanoem::model::ModelVertexType::QDEF => {
                bones[0] = vertex
                    .bone_indices
                    .get(0)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
                bones[1] = vertex
                    .bone_indices
                    .get(1)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
                bones[2] = vertex
                    .bone_indices
                    .get(2)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
                bones[3] = vertex
                    .bone_indices
                    .get(3)
                    .map(|idx| if *idx >= 0 { Some(*idx as usize) } else { None })
                    .flatten();
            }
        }
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
                vertex.edge_size,
                i32::from(vertex.typ) as f32,
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
        Self {
            material: None,
            soft_body: None,
            bones,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            simd,
            origin: vertex.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.simd.delta = Vector4::zero();
        self.simd.delta_uva = [Vector4::zero(); 5];
    }

    pub fn deform(&mut self, morph: &nanoem::model::ModelMorphVertex, weight: f32) {
        self.simd.delta = self.simd.delta + f128_to_vec4(morph.position) * weight;
    }

    pub fn deform_uv(&mut self, morph: &nanoem::model::ModelMorphUv, uv_idx: usize, weight: f32) {
        self.simd.delta_uva[uv_idx].add_assign_element_wise(f128_to_vec4(morph.position) * weight);
    }

    pub fn set_material(&mut self, material_idx: MaterialIndex) {
        self.material = Some(material_idx)
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

#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialStates {
    pub visible: bool,
    pub display_diffuse_texture_uv_mesh_enabled: bool,
    pub display_sphere_map_texture_uv_mesh_enabled: bool,
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
    states: MaterialStates,
    origin: NanoemMaterial,
}

impl Material {
    pub const PRIVATE_STATE_VISIBLE: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_DISPLAY_DIFFUSE_TEXTURE_UV_MESH_ENABLED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_DISPLAY_SPHERE_MAP_TEXTURE_UV_MESH_ENABLED: u32 = 1u32 << 3;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;
    pub const PRIVATE_STATE_INITIAL_VALUE: u32 =
        Self::PRIVATE_STATE_VISIBLE | Self::PRIVATE_STATE_DISPLAY_DIFFUSE_TEXTURE_UV_MESH_ENABLED;
    pub const MINIUM_SPECULAR_POWER: f32 = 0.1f32;

    pub fn from_nanoem(
        material: &NanoemMaterial,
        fallback: &wgpu::Texture,
        language_type: nanoem::common::LanguageType,
    ) -> Self {
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
        Self {
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
            states: MaterialStates {
                visible: true,
                display_diffuse_texture_uv_mesh_enabled: true,
                ..Default::default()
            },
            origin: material.clone(),
        }
    }

    pub fn reset(&mut self) {
        let material = &self.origin;
        self.color.base.ambient = Vector4::from(material.get_ambient_color()).truncate();
        self.color.base.diffuse = Vector4::from(material.get_diffuse_color()).truncate();
        self.color.base.specular = Vector4::from(material.get_specular_color()).truncate();
        self.color.base.diffuse_opacity = material.get_diffuse_opacity();
        self.color.base.specular_power = material
            .get_specular_power()
            .max(Self::MINIUM_SPECULAR_POWER);
        self.color.base.diffuse_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.sphere_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.toon_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.edge.base.color = Vector4::from(material.get_edge_color()).truncate();
        self.edge.base.opacity = material.get_edge_opacity();
        self.edge.base.size = material.get_edge_size();
    }

    pub fn reset_deform(&mut self) {
        self.color.mul = MaterialColor::new_reset(1f32);
        self.color.add = MaterialColor::new_reset(0f32);
        self.edge.mul = MaterialEdge::new_reset(1f32);
        self.edge.add = MaterialEdge::new_reset(0f32);
    }

    pub fn deform(&mut self, morph: &nanoem::model::ModelMorphMaterial, weight: f32) {
        const ONE_V4: Vector4<f32> = Vector4 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
            w: 1f32,
        };
        const ONE_V3: Vector3<f32> = Vector3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        };
        let diffuse_texture_blend_factor = f128_to_vec4(morph.diffuse_texture_blend);
        let sphere_texture_blend_factor = f128_to_vec4(morph.sphere_map_texture_blend);
        // TODO: nanoem use sphere_map_texture_blend, it may be a mistake
        let toon_texture_blend_factor = f128_to_vec4(morph.toon_texture_blend);
        match morph.operation {
            nanoem::model::ModelMorphMaterialOperationType::Multiply => {
                self.color.mul.ambient.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.mul.diffuse.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.mul.specular.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.mul.diffuse_opacity = lerp_f32(
                    self.color.mul.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.mul.specular_power =
                    lerp_f32(self.color.mul.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .mul
                    .diffuse_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .mul
                    .sphere_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .mul
                    .toon_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .mul
                    .color
                    .mul_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.mul.opacity = lerp_f32(self.edge.mul.opacity, morph.edge_opacity, weight);
                self.edge.mul.size = lerp_f32(self.edge.mul.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Add => {
                self.color.add.ambient.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.add.diffuse.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.add.specular.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.add.diffuse_opacity = lerp_f32(
                    self.color.add.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.add.specular_power =
                    lerp_f32(self.color.add.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .add
                    .diffuse_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .add
                    .sphere_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .add
                    .toon_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .add
                    .color
                    .add_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.add.opacity = lerp_f32(self.edge.add.opacity, morph.edge_opacity, weight);
                self.edge.add.size = lerp_f32(self.edge.add.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Unknown => {}
        }
    }

    pub fn is_visible(&self) -> bool {
        self.states.visible
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
        self.diffuse_image.as_ref().map(|rc| rc.as_ref())
    }

    pub fn sphere_map_image(&self) -> Option<&wgpu::Texture> {
        self.sphere_map_image.as_ref().map(|rc| rc.as_ref())
    }

    pub fn toon_image(&self) -> Option<&wgpu::Texture> {
        self.toon_image.as_ref().map(|rc| rc.as_ref())
    }

    pub fn spheremap_texture_type(&self) -> nanoem::model::ModelMaterialSphereMapTextureType {
        self.origin.sphere_map_texture_type
    }

    pub fn is_culling_disabled(&self) -> bool {
        self.origin.flags.is_culling_disabled
    }

    pub fn is_casting_shadow_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_enabled
    }
    pub fn is_casting_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_map_enabled
    }
    pub fn is_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_shadow_map_enabled
    }
    pub fn is_edge_enabled(&self) -> bool {
        self.origin.flags.is_edge_enabled
    }
    pub fn is_vertex_color_enabled(&self) -> bool {
        self.origin.flags.is_vertex_color_enabled
    }
    pub fn is_point_draw_enabled(&self) -> bool {
        self.origin.flags.is_point_draw_enabled
    }
    pub fn is_line_draw_enabled(&self) -> bool {
        self.origin.flags.is_line_draw_enabled
    }
}

pub struct Morph {
    pub name: String,
    pub canonical_name: String,
    weight: f32,
    pub dirty: bool,
    pub origin: NanoemMorph,
}

impl Morph {
    pub fn from_nanoem(morph: &NanoemMorph, language: nanoem::common::LanguageType) -> Self {
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
        Self {
            name,
            canonical_name,
            weight: 0f32,
            dirty: false,
            origin: morph.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.weight = 0f32;
        self.dirty = false;
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn set_weight(&mut self, value: f32) {
        self.dirty = self.weight.abs() > f32::EPSILON || value.abs() > f32::EPSILON;
        self.weight = value;
    }

    pub fn set_forced_weight(&mut self, value: f32) {
        self.dirty = false;
        self.weight = value;
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        name: &str,
        frame_index: u32,
        amount: f32,
    ) {
        let w0 = self.synchronize_weight(motion, name, frame_index);
        if amount > 0f32 {
            let w1 = self.synchronize_weight(motion, name, frame_index + 1);
            self.set_weight(lerp_f32(w0, w1, amount));
        } else {
            self.set_weight(w0);
        }
    }

    fn synchronize_weight(&mut self, motion: &Motion, name: &str, frame_index: u32) -> f32 {
        if let Some(keyframe) = motion.find_morph_keyframe(name, frame_index) {
            keyframe.weight
        } else {
            if let (Some(prev_frame), Some(next_frame)) = motion
                .opaque
                .search_closest_morph_keyframes(name, frame_index)
            {
                let coef = Motion::coefficient(
                    prev_frame.base.frame_index,
                    next_frame.base.frame_index,
                    frame_index,
                );
                lerp_f32(prev_frame.weight, next_frame.weight, coef)
            } else {
                0f32
            }
        }
    }
}

pub struct Label {
    name: String,
    canonical_name: String,
    origin: NanoemLabel,
}

impl Label {
    const NAME_EXPRESSION_IN_JAPANESE_UTF8: &'static [u8] =
        &[0xe8, 0xa1, 0xa8, 0xe6, 0x83, 0x85, 0x0];
    const NAME_EXPRESSION_IN_JAPANESE: &'static str = "表情";

    pub fn from_nanoem(label: &NanoemLabel, language: nanoem::common::LanguageType) -> Self {
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
        Self {
            name,
            canonical_name,
            origin: label.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RigidBodyStates {
    pub enabled: bool,
    pub all_forces_should_reset: bool,
    pub editing_masked: bool,
}

pub struct RigidBody {
    // TODO: physics engine and shape mesh and engine rigid_body
    physics_rigid_body: rapier3d::dynamics::RigidBodyHandle,
    initial_world_transform: nalgebra::Isometry3<f32>,
    global_torque_force: (Vector3<f32>, bool),
    global_velocity_force: (Vector3<f32>, bool),
    local_torque_force: (Vector3<f32>, bool),
    local_velocity_force: (Vector3<f32>, bool),
    name: String,
    canonical_name: String,
    states: RigidBodyStates,
    origin: NanoemRigidBody,
}

impl RigidBody {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_ALL_FORCES_SHOULD_RESET: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 3;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn from_nanoem(
        rigid_body: &NanoemRigidBody,
        language: nanoem::common::LanguageType,
        is_morph: bool,
        bones: &[Bone],
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
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
        let orientation = rigid_body.orientation.0;
        let origin = rigid_body.origin.0;
        let mut world_transform = nalgebra::Isometry3::new(
            nalgebra::vector![origin[0], origin[1], origin[2]],
            nalgebra::vector![orientation[0], orientation[1], orientation[2]],
        );
        let mut initial_world_transform = world_transform;
        if rigid_body.is_bone_relative {
            if let Some(bone) = usize::try_from(rigid_body.bone_index)
                .ok()
                .and_then(|idx| bones.get(idx))
            {
                let bone_origin = bone.origin.origin.0;
                let offset = nalgebra::Isometry3::translation(
                    bone_origin[0],
                    bone_origin[1],
                    bone_origin[2],
                );
                world_transform = offset * world_transform;
                initial_world_transform = world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                world_transform = world_transform * skinning_transform;
            }
        }
        let rigid_body_builder = rapier3d::dynamics::RigidBodyBuilder::new(
            rapier3d::dynamics::RigidBodyType::KinematicPositionBased,
        )
        .position(world_transform)
        .angular_damping(rigid_body.angular_damping)
        .linear_damping(rigid_body.linear_damping);
        let size = rigid_body.size.0;
        let mut collider_builder = match rigid_body.shape_type {
            nanoem::model::ModelRigidBodyShapeType::Unknown => (None, f32::INFINITY),
            nanoem::model::ModelRigidBodyShapeType::Sphere => (
                Some(rapier3d::geometry::ColliderBuilder::ball(size[0])),
                4f32 * PI * size[0].powi(3) / 3f32,
            ),
            nanoem::model::ModelRigidBodyShapeType::Box => (
                Some(rapier3d::geometry::ColliderBuilder::cuboid(
                    size[0], size[1], size[2],
                )),
                size[0] * size[1] * size[2],
            ),
            nanoem::model::ModelRigidBodyShapeType::Capsule => (
                Some(rapier3d::geometry::ColliderBuilder::capsule_y(
                    size[1] / 2f32,
                    size[0],
                )),
                PI * size[0] * size[0] * (4f32 / 3f32 * size[0] + size[1]),
            ),
        };
        let mut mass = rigid_body.mass;
        if let nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation =
            rigid_body.transform_type
        {
            mass = 0f32;
        }
        collider_builder.0 = collider_builder.0.map(|builder| {
            builder
                .density(mass / collider_builder.1)
                .friction(rigid_body.friction)
                .restitution(rigid_body.restitution)
                .collision_groups(rapier3d::geometry::InteractionGroups::new(
                    0x1 << rigid_body.collision_group_id.clamp(0, 15),
                    (rigid_body.collision_mask & 0xffff) as u32,
                ))
        });
        let rigid_body_handle = physics_engine
            .rigid_body_set
            .insert(rigid_body_builder.build());
        if let Some(collider_builder) = &collider_builder.0 {
            let collider_handle = physics_engine.collider_set.insert_with_parent(
                collider_builder.build(),
                rigid_body_handle,
                &mut physics_engine.rigid_body_set,
            );
        }
        Self {
            global_torque_force: (Vector3::zero(), false),
            global_velocity_force: (Vector3::zero(), false),
            local_torque_force: (Vector3::zero(), false),
            local_velocity_force: (Vector3::zero(), false),
            name,
            canonical_name,
            states: RigidBodyStates::default(),
            origin: rigid_body.clone(),
            physics_rigid_body: rigid_body_handle,
            initial_world_transform,
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
        bone: &Bone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            let skinning_transform = to_isometry(bone.matrices.skinning_transform);
            physics_rigid_body
                .set_position(self.initial_world_transform * skinning_transform, true);
            physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rigid_body.reset_forces(true);
        }
    }

    pub fn enable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if !physics_rigid_body.is_kinematic() {
                physics_rigid_body
                    .set_body_type(rapier3d::dynamics::RigidBodyType::KinematicVelocityBased);
            }
        }
    }

    pub fn disable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if physics_rigid_body.is_kinematic() {
                physics_rigid_body.set_body_type(rapier3d::dynamics::RigidBodyType::Dynamic);
            }
        }
    }

    pub fn apply_all_forces(&mut self, bone: Option<&Bone>, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if self.states.all_forces_should_reset {
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            } else {
                if self.global_torque_force.1 {
                    physics_rigid_body
                        .apply_torque_impulse(to_na_vec3(self.global_torque_force.0), true);
                    self.global_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_torque_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rigid_body.position())
                        .rotation;
                        physics_rigid_body.apply_torque_impulse(
                            local_orientation * to_na_vec3(self.local_torque_force.0),
                            true,
                        );
                        self.local_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                    }
                }
                if self.global_velocity_force.1 {
                    physics_rigid_body
                        .apply_impulse(to_na_vec3(self.global_velocity_force.0), true);
                    self.global_velocity_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_velocity_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rigid_body.position())
                        .rotation;
                        physics_rigid_body.apply_torque_impulse(
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

    pub fn synchronize_transform_feedback_to_simulation(
        &mut self,
        bone: &Bone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if self.origin.transform_type == ModelRigidBodyTransformType::FromBoneToSimulation
                || physics_rigid_body.is_kinematic()
            {
                let initial_transform = self.initial_world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                let world_transform = initial_transform * skinning_transform;
                physics_rigid_body.set_position(world_transform, true);
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            }
        }
    }

    pub fn synchronize_transform_feedback_from_simulation(
        &mut self,
        bone: &mut Bone,
        parent_bone: Option<&Bone>,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if (self.origin.transform_type == ModelRigidBodyTransformType::FromSimulationToBone
                || self.origin.transform_type
                    == ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone)
                && !physics_rigid_body.is_kinematic()
            {
                let initial_transform = self.initial_world_transform;
                let mut world_transform = physics_rigid_body.position().clone();
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
                    from_na_mat4((world_transform * initial_transform.inverse()).to_homogeneous());
                bone.update_skinning_transform(skinning_transform);
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
            .rigid_body_set
            .get(self.physics_rigid_body)
            .unwrap()
            .is_kinematic()
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
    states: JointStates,
    origin: NanoemJoint,
}

impl Joint {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn from_nanoem(joint: &NanoemJoint, language: nanoem::common::LanguageType) -> Self {
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
        Self {
            name,
            canonical_name,
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

#[derive(Debug, Clone, Copy, Default)]
pub struct SoftBodyStates {
    pub enabled: bool,
    pub editing_masked: bool,
}

pub struct SoftBody {
    // TODO: physics engine and shape mesh and engine soft_body
    name: String,
    canonical_name: String,
    pub states: SoftBodyStates,
    origin: NanoemSoftBody,
}

impl SoftBody {
    pub fn from_nanoem(soft_body: &NanoemSoftBody, language: nanoem::common::LanguageType) -> Self {
        // TODO: should resolve physic engine
        let mut name = soft_body.get_name(language).to_owned();
        let mut canonical_name = soft_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", soft_body.base.index);
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        Self {
            name,
            canonical_name,
            states: SoftBodyStates::default(),
            origin: soft_body.clone(),
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

pub struct VisualizationClause {
    // TODO
}

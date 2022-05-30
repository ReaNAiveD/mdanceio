use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::{Rc, Weak},
};

use bytemuck::{Pod, Zeroable};
use cgmath::{ElementWise, Matrix4, Quaternion, SquareMatrix, Vector3, Vector4, Zero};
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
type VertexIndex = usize;
type BoneIndex = usize;
type MaterialIndex = usize;
type MorphIndex = usize;
type ConstraintIndex = usize;
type LabelIndex = usize;
type RigidBodyIndex = usize;
type JointIndex = usize;
type SoftBodyIndex = usize;

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
    bone_index_hash_map: HashMap<MaterialIndex, HashMap<BoneIndex, usize>>,
    bones_by_name: HashMap<String, BoneIndex>,
    morphs_by_name: HashMap<String, MorphIndex>,
    bone_to_constraints: HashMap<BoneIndex, ConstraintIndex>,
    // redo_bone_names: Vec<String>,
    // redo_morph_names: Vec<String>,
    // outside_parents: HashMap<Rc<RefCell<NanoemBone>>, (String, String)>,
    // image_uris: HashMap<String, Uri>,
    // attachment_uris: HashMap<String, Uri>,
    // bone_bound_rigid_bodies: HashMap<Rc<RefCell<NanoemBone>>, Rc<RefCell<NanoemRigidBody>>>,
    constraint_joint_bones: HashMap<BoneIndex, ConstraintIndex>,
    inherent_bones: HashMap<BoneIndex, HashSet<BoneIndex>>,
    constraint_effector_bones: HashSet<BoneIndex>,
    parent_bone_tree: HashMap<BoneIndex, Vec<BoneIndex>>,
    // shared_fallback_bone: Rc<RefCell<Bone>>,
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
    pub const MAX_BONE_UNIFORMS: usize = 55;

    pub fn new_from_bytes(
        bytes: &[u8],
        project: &Project,
        handle: u16,
        device: &wgpu::Device,
    ) -> Result<Self, Error> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        match NanoemModel::load_from_buffer(&mut buffer) {
            Ok(nanoem_model) => {
                let opaque = Box::new(nanoem_model);
                let language = project.parse_language();
                let mut name = opaque.get_name(language).to_owned();
                let comment = opaque.get_comment(language).to_owned();
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
                        Material::from_nanoem(material, project.shared_fallback_image(), language)
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
                    .map(|bone| Bone::from_nanoem(bone, language))
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
                        language,
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
                            language,
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
                    .map(|morph| Morph::from_nanoem(&morph, language))
                    .collect::<Vec<_>>();
                for morph in &opaque.morphs {
                    if let nanoem::model::ModelMorphType::Vertex = morph.get_type() {
                        if let nanoem::model::ModelMorphU::VERTICES(morph_vertices) = &morph.morphs {
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
                    }
                    let category = morph.category;
                    // TODO: set active category
                    for language in nanoem::common::LanguageType::all() {
                        morphs_by_name
                            .insert(morph.get_name(*language).to_owned(), morph.base.index);
                    }
                }
                let labels = opaque
                    .labels
                    .iter()
                    .map(|label| Label::from_nanoem(label, language))
                    .collect();
                let rigid_bodies = opaque
                    .rigid_bodies
                    .iter()
                    .map(|rigid_body| RigidBody::from_nanoem(rigid_body, language))
                    .collect();
                for rigid_body in &opaque.rigid_bodies {
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
                }
                let joints = opaque
                    .joints
                    .iter()
                    .map(|joint| Joint::from_nanoem(joint, language))
                    .collect();
                let soft_bodies = opaque
                    .soft_bodies
                    .iter()
                    .map(|soft_body| SoftBody::from_nanoem(soft_body, language))
                    .collect();
                // split_bones_per_material();

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

                let vertices = vertices;
                log::trace!("Len(vertices): {}", vertices.len());
                let mut vertex_buffer_data: Vec<VertexUnit> = vec![];
                for vertex in &vertices {
                    vertex_buffer_data.push(vertex.simd.clone().into());
                }
                log::trace!("Len(vertex_buffer): {}", vertex_buffer_data.len());
                let vertex_buffer_even = wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(format!("Model/{}/VertexBuffer/Even", canonical_name).as_str()),
                        contents: bytemuck::cast_slice(&vertex_buffer_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                );
                let vertex_buffer_odd = wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(format!("Model/{}/VertexBuffer/Odd", canonical_name).as_str()),
                        contents: bytemuck::cast_slice(&vertex_buffer_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                );
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

                Ok(Self {
                    handle,
                    opaque,
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
                    constraint_joint_bones,
                    inherent_bones,
                    constraint_effector_bones,
                    parent_bone_tree,
                    // shared_fallback_bone,
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
    }

    pub fn create_all_images(&mut self, texture_lut: &HashMap<String, Rc<wgpu::Texture>>) {
        // TODO: 创建所有材质贴图并绑定到Material上
        for material in &mut self.materials {
            material.diffuse_image = material.origin.get_diffuse_texture_object(&self.opaque.textures).map(|texture_object| texture_lut.get(&texture_object.path)).flatten().map(|rc| rc.clone());
            material.sphere_map_image = material.origin.get_sphere_map_texture_object(&self.opaque.textures).map(|texture_object| texture_lut.get(&texture_object.path)).flatten().map(|rc| rc.clone());
            material.toon_image = material.origin.get_toon_texture_object(&self.opaque.textures).map(|texture_object| texture_lut.get(&texture_object.path)).flatten().map(|rc| rc.clone());
        }
    }

    fn create_image() {}

    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    pub fn active_bone(&self) -> Option<&Bone> {
        todo!()
    }

    pub fn find_bone(&self, name: &str) -> Option<&Bone> {
        self.bones_by_name
            .get(name)
            .map(|index| self.bones.get(*index))
            .flatten()
    }

    pub fn parent_bone(&self, bone: &Bone) -> Option<&Bone> {
        todo!()
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
                        view,
                        Some(&project.viewport_primary_depth_view()),
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
    pub origin: NanoemBone,
}

impl Bone {
    const DEFAULT_BAZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
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
            bezier_control_points: BezierControlPoints {
                translation_x: Vector4::new(0, 0, 0, 0),
                translation_y: Vector4::new(0, 0, 0, 0),
                translation_z: Vector4::new(0, 0, 0, 0),
                orientation: Vector4::new(0, 0, 0, 0),
            },
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            origin: bone.clone(),
        }
    }

    // fn synchronize_transform(motion: &mut Motion, model_bone: &ModelBone, model_rigid_body: &ModelRigidBody, frame_index: u32, transform: &FrameTransform) {
    //     let name = model_bone.get_name(LanguageType::Japanese).unwrap();
    //     if let Some(Keyframe) = motion.find_bone_keyframe_object(name, index)
    // }

    pub fn skinning_transform(&self) -> Matrix4<f32> {
        todo!()
    }

    pub fn world_transform_origin(&self) -> Vector3<f32> {
        todo!()
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

#[derive(Debug, Clone)]
pub struct Constraint {
    name: String,
    canonical_name: String,
    joint_iteration_result: Vec<Vec<ConstraintJoint>>,
    effector_iteration_result: Vec<Vec<ConstraintJoint>>,
    states: u32,
    origin: NanoemConstraint,
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
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
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
    origin: NanoemVertex,
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
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            origin: material.clone(),
        }
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
    name: String,
    canonical_name: String,
    weight: f32,
    dirty: bool,
    origin: NanoemMorph,
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

pub struct RigidBody {
    // TODO: physics engine and shape mesh and engine rigid_body
    global_torque_force: (Vector3<f32>, bool),
    global_velocity_force: (Vector3<f32>, bool),
    local_torque_force: (Vector3<f32>, bool),
    local_velocity_force: (Vector3<f32>, bool),
    name: String,
    canonical_name: String,
    states: u32,
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
        Self {
            global_torque_force: (Vector3::zero(), false),
            global_velocity_force: (Vector3::zero(), false),
            local_torque_force: (Vector3::zero(), false),
            local_velocity_force: (Vector3::zero(), false),
            name,
            canonical_name,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            origin: rigid_body.clone(),
        }
    }
}

pub struct Joint {
    // TODO: physics engine and shape mesh and engine rigid_body
    name: String,
    canonical_name: String,
    states: u32,
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
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            origin: joint.clone(),
        }
    }
}

pub struct SoftBody {
    // TODO: physics engine and shape mesh and engine soft_body
    name: String,
    canonical_name: String,
    states: u32,
    origin: NanoemSoftBody,
}

impl SoftBody {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

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
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            origin: soft_body.clone(),
        }
    }
}

pub struct VisualizationClause {
    // TODO
}

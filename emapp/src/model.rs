use std::{
    cell::{RefCell, Ref},
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3, Vector4};
use nanoem::model::{ModelConstraint, ModelMaterial, ModelVertex, ModelFormatType};
use par::shape::ShapesMesh;
use wgpu::{AddressMode, Buffer, PipelineLayoutDescriptor};

use crate::{
    bounding_box::BoundingBox, camera::Camera, drawable::DrawType, effect::IEffect,
    forward::LineVertexUnit, image_loader::Image, internal::LinearDrawer,
    model_object_selection::ModelObjectSelection, undo::UndoStack, uri::Uri, project::Project,
};

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

pub struct VertexUnit {
    // TODO
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
    bone_indices: HashMap<Rc<RefCell<ModelMaterial>>, HashMap<i32, i32>>,
    output: u8,
    materials: Rc<RefCell<[ModelMaterial]>>,
    vertices: Rc<RefCell<[ModelVertex]>>,
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
    passiveEffect: Rc<RefCell<dyn IEffect>>,
    enabled: bool,
}

pub struct Model {
    handle: u16,
    camera: Rc<RefCell<dyn Camera>>,
    selection: Rc<RefCell<dyn ModelObjectSelection>>,
    drawer: Box<LinearDrawer>,
    skin_deformer: Rc<RefCell<dyn SkinDeformer>>,
    gizmo: Rc<RefCell<dyn Gizmo>>,
    vertex_weight_painter: Rc<RefCell<dyn VertexWeightPainter>>,
    offscreen_passive_render_target_effects: HashMap<String, OffscreenPassiveRenderTargetEffect>,
    draw_all_vertex_normals: DrawArrayBuffer,
    draw_all_vertex_points: DrawArrayBuffer,
    draw_all_vertex_faces: DrawIndexedBuffer,
    draw_all_vertex_weights: DrawIndexedBuffer,
    draw_rigid_body: HashMap<Rc<RefCell<ShapesMesh>>, DrawIndexedBuffer>,
    draw_joint: HashMap<Rc<RefCell<ShapesMesh>>, DrawIndexedBuffer>,
    opaque: Rc<RefCell<nanoem::model::Model>>,
    undo_stack: Box<UndoStack>,
    editing_undo_stack: Box<UndoStack>,
    active_morph_ptr:
        HashMap<nanoem::model::ModelMorphCategory, Rc<RefCell<nanoem::model::ModelMorph>>>,
    active_constraint_ptr: Rc<RefCell<nanoem::model::ModelConstraint>>,
    active_material_ptr: Rc<RefCell<nanoem::model::ModelMaterial>>,
    hovered_bone_ptr: Rc<RefCell<nanoem::model::ModelBone>>,
    vertex_buffer_data: Vec<u8>,
    face_states: Vec<u32>,
    active_bone_pair_ptr: (
        Rc<RefCell<nanoem::model::ModelBone>>,
        Rc<RefCell<nanoem::model::ModelBone>>,
    ),
    active_effect_pair_ptr: (Rc<RefCell<dyn IEffect>>, Rc<RefCell<dyn IEffect>>),
    screen_image: Image,
    loading_image_items: Vec<LoadingImageItem>,
    image_map: HashMap<String, Image>,
    bone_index_hash_map: HashMap<Rc<RefCell<nanoem::model::ModelMaterial>>, HashMap<i32, i32>>,
    bones: HashMap<String, Rc<RefCell<nanoem::model::ModelBone>>>,
    morphs: HashMap<String, Rc<RefCell<nanoem::model::ModelMorph>>>,
    constraints:
        HashMap<Rc<RefCell<nanoem::model::ModelBone>>, Rc<RefCell<nanoem::model::ModelConstraint>>>,
    redo_bone_names: Vec<String>,
    redo_morph_names: Vec<String>,
    outside_parents: HashMap<Rc<RefCell<nanoem::model::ModelBone>>, (String, String)>,
    image_uris: HashMap<String, Uri>,
    attachment_uris: HashMap<String, Uri>,
    bone_bound_rigid_bodies:
        HashMap<Rc<RefCell<nanoem::model::ModelBone>>, Rc<RefCell<nanoem::model::ModelRigidBody>>>,
    constraint_joint_bones:
        HashMap<Rc<RefCell<nanoem::model::ModelBone>>, Rc<RefCell<ModelConstraint>>>,
    inherent_bones:
        HashMap<Rc<RefCell<nanoem::model::ModelBone>>, HashSet<nanoem::model::ModelBone>>,
    constraint_effect_bones: HashSet<Rc<RefCell<nanoem::model::ModelBone>>>,
    parent_bone_tree:
        HashMap<Rc<RefCell<nanoem::model::ModelBone>>, Vec<Rc<RefCell<nanoem::model::ModelBone>>>>,
    shared_fallback_bone: Rc<RefCell<Bone>>,
    bounding_box: BoundingBox,
    // UserData m_userData;
    annotations: HashMap<String, String>,
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffer: wgpu::Buffer,
    edge_color: Vector4<f32>,
    transform_axis_type: AxisType,
    edit_action_type: EditActionType,
    transform_coordinate_type: TransformCoordinateType,
    file_uri: Uri,
    name: String,
    comment: String,
    canonical_name: String,
    states: u32,
    edge_size_scale_factor: f32,
    opacity: f32,
    // void *m_dispatchParallelTaskQueue
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

    pub fn new(project: &Project, handle: u16) -> Self {
        Self {
            handle,
            camera: todo!(),
            selection: todo!(),
            drawer: todo!(),
            skin_deformer: todo!(),
            gizmo: todo!(),
            vertex_weight_painter: todo!(),
            offscreen_passive_render_target_effects: todo!(),
            draw_all_vertex_normals: todo!(),
            draw_all_vertex_points: todo!(),
            draw_all_vertex_faces: todo!(),
            draw_all_vertex_weights: todo!(),
            draw_rigid_body: todo!(),
            draw_joint: todo!(),
            opaque: todo!(),
            undo_stack: todo!(),
            editing_undo_stack: todo!(),
            active_morph_ptr: todo!(),
            active_constraint_ptr: todo!(),
            active_material_ptr: todo!(),
            hovered_bone_ptr: todo!(),
            vertex_buffer_data: todo!(),
            face_states: todo!(),
            active_bone_pair_ptr: todo!(),
            active_effect_pair_ptr: todo!(),
            screen_image: todo!(),
            loading_image_items: todo!(),
            image_map: todo!(),
            bone_index_hash_map: todo!(),
            bones: todo!(),
            morphs: todo!(),
            constraints: todo!(),
            redo_bone_names: todo!(),
            redo_morph_names: todo!(),
            outside_parents: todo!(),
            image_uris: todo!(),
            attachment_uris: todo!(),
            bone_bound_rigid_bodies: todo!(),
            constraint_joint_bones: todo!(),
            inherent_bones: todo!(),
            constraint_effect_bones: todo!(),
            parent_bone_tree: todo!(),
            shared_fallback_bone: todo!(),
            bounding_box: todo!(),
            annotations: todo!(),
            vertex_buffers: todo!(),
            index_buffer: todo!(),
            edge_color: todo!(),
            transform_axis_type: todo!(),
            edit_action_type: todo!(),
            transform_coordinate_type: todo!(),
            file_uri: todo!(),
            name: todo!(),
            comment: todo!(),
            canonical_name: todo!(),
            states: todo!(),
            edge_size_scale_factor: todo!(),
            opacity: todo!(),
            count_vertex_skinning_needed: todo!(),
            stage_vertex_buffer_index: todo!(),
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
        let mut model = nanoem::model::Model::default();
        {
            model.set_additional_uv_size(0);
            model.set_codec_type(nanoem::common::CodecType::Utf16);
            model.set_format_type(ModelFormatType::Pmx2_0);
            for language in nanoem::common::LanguageType::all() {
                model.set_name(desc.name.get(language).unwrap_or(&"".to_string()), language.clone());
                model.set_comment(desc.comment.get(language).unwrap_or(&"".to_string()), language.clone());
            }
        }
        let mut center_bone = nanoem::model::ModelBone::default();
        center_bone.set_name(&Bone::NAME_CENTER_IN_JAPANESE.to_string(), nanoem::common::LanguageType::Japanese);
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
            root_label.insert_item_object(&nanoem::model::ModelLabelItem::create_from_bone_object(center_bone_rc), -1);
            root_label.set_special(true);
            model.insert_label(&root_label, -1);
        }
        {
            let mut expression_label = nanoem::model::ModelLabel::default();
            expression_label.set_name(&Label::NAME_EXPRESSION_IN_JAPANESE.to_string(), nanoem::common::LanguageType::Japanese);
            expression_label.set_name(&"Expression".to_string(), nanoem::common::LanguageType::English);
            expression_label.set_special(true);
            model.insert_label(&expression_label, -1);
        }
        model.save_to_buffer(&mut buffer)?;
        Ok(buffer.get_data())
    }

    pub fn find_bone(&self, name: &String) -> Option<Rc<RefCell<nanoem::model::ModelBone>>> {
        self.bones.get(name).map(|rc| rc.clone())
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_canonical_name(&self) -> &String {
        &self.canonical_name
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
struct Bone {
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

    // fn synchronize_transform(motion: &mut Motion, model_bone: &ModelBone, model_rigid_body: &ModelRigidBody, frame_index: u32, transform: &FrameTransform) {
    //     let name = model_bone.get_name(LanguageType::Japanese).unwrap();
    //     if let Some(Keyframe) = motion.find_bone_keyframe_object(name, index)
    // }
}

struct Label {
    // TODO
}

impl Label {
    const NAME_EXPRESSION_IN_JAPANESE_UTF8: &'static [u8] = &[0xe8, 0xa1, 0xa8, 0xe6, 0x83, 0x85, 0x0];
    const NAME_EXPRESSION_IN_JAPANESE: &'static str = "表情";
}

pub struct VisualizationClause {
    // TODO
}

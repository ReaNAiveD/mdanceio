use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{
    common::{Buffer, LanguageType, Status, UserData, F128},
    utils::fourcc,
};

struct Info {
    codec_type: u8,
    additional_uv_size: u8,
    vertex_index_size: u8,
    texture_index_size: u8,
    material_index_size: u8,
    bone_index_size: u8,
    morph_index_size: u8,
    rigid_body_index_size: u8,
}

pub enum ModelFormatType {
    Unknown = -1,
    Pmd1_0,
    Pmx2_0,
    Pmx2_1,
}

impl From<i32> for ModelFormatType {
    fn from(value: i32) -> Self {
        match value {
            10 => ModelFormatType::Pmd1_0,
            20 => ModelFormatType::Pmx2_0,
            21 => ModelFormatType::Pmx2_1,
            _ => ModelFormatType::Unknown,
        }
    }
}

pub enum CodecType {
    Unknown = -1,
    Sjis,
    Utf8,
    Utf16,
}

impl From<i32> for CodecType {
    fn from(value: i32) -> Self {
        match value {
            0 => CodecType::Sjis,
            1 => CodecType::Utf8,
            2 => CodecType::Utf16,
            _ => CodecType::Unknown,
        }
    }
}

pub struct Model {
    version: f32,
    info_length: u8,
    info: Info,
    name_ja: String,
    name_en: String,
    comment_ja: String,
    comment_en: String,
    vertices: Vec<ModelVertex>,
    vertex_indices: Vec<u32>,
    materials: Vec<ModelMaterial>,
    bones: Vec<Rc<RefCell<ModelBone>>>,
    ordered_bones: Vec<Rc<RefCell<ModelBone>>>,
    constraints: Vec<ModelConstraint>,
    textures: Vec<ModelTexture>,
    morphs: Vec<Rc<RefCell<ModelMorph>>>,
    labels: Vec<ModelLabel>,
    rigid_bodies: Vec<ModelRigidBody>,
    joints: Vec<ModelJoint>,
    soft_bodies: Vec<ModelSoftBody>,
    user_data: Option<Rc<RefCell<UserData>>>,
}

impl Model {
    fn get_string_pmx(&self, buffer: &mut Buffer) -> Result<String, Status> {
        let length = buffer.read_len()?;
        let src = buffer.read_buffer(length)?;
        let codec = if self.info.codec_type == 1u8 {
            encoding_rs::UTF_8
        } else {
            encoding_rs::UTF_16LE
        };
        // TODO: need bom removal or not?
        let (cow, encoding_used, had_errors) = codec.decode(src);
        if had_errors {
            return Err(Status::ErrorDecodeUnicodeStringFailed);
        }
        Ok(cow.into())
    }

    fn parse_vertex_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_vertices = buffer.read_len()?;
        if num_vertices > 0 {
            self.vertices.clear();
            for i in 0..num_vertices {
                self.vertices.push(ModelVertex::parse_pmx(self, buffer)?);
                self.vertices[i].base.index = i as i32;
            }
        }
        Ok(())
    }

    fn parse_vertex_index_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let vertex_index_size = self.info.vertex_index_size as usize;
        let num_vertex_indices = buffer.read_len()?;
        let num_vertices = self.vertices.len();
        if (num_vertex_indices == 0 && num_vertices > 0) || num_vertex_indices % 3 != 0 {
            Err(Status::ErrorModelFaceCorrupted)
        } else {
            self.vertex_indices.clear();
            for _ in 0..num_vertex_indices {
                let vertex_index = buffer.read_integer(vertex_index_size)? as u32;
                self.vertex_indices
                    .push(if vertex_index < num_vertices as u32 {
                        vertex_index
                    } else {
                        0
                    })
            }
            Ok(())
        }
    }

    fn parse_texture_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_textures = buffer.read_len()?;
        if num_textures > 0 {
            self.textures.clear();
            for i in 0..num_textures {
                self.textures.push(ModelTexture::parse_pmx(self, buffer)?);
                self.textures[i].base.index = i as i32;
            }
        }
        Ok(())
    }

    fn parse_material_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_materials = buffer.read_len()?;
        if num_materials > 0 {
            self.materials.clear();
            for i in 0..num_materials {
                self.materials.push(ModelMaterial::parse_pmx(self, buffer)?);
                self.materials[i].base.index = i as i32;
            }
        }
        Ok(())
    }

    fn parse_bone_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_bones = buffer.read_len()?;
        if num_bones > 0 {
            self.bones.clear();
            for i in 0..num_bones {
                let mut bone = ModelBone::parse_pmx(self, buffer)?;
                bone.base.index = i as i32;
                // bone.constraint = bone.constraint.map(|mut b| {b.target_bone_index = i as i32; b});
                if let Some(ref mut constraint) = bone.constraint {
                    constraint.target_bone_index = i as i32;
                }
                self.bones.push(Rc::new(RefCell::new(bone)));
                self.ordered_bones.push(self.bones[i].clone());
            }
        }
        Ok(())
    }

    fn parse_morph_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_morphs = buffer.read_len()?;
        if num_morphs > 0 {
            self.morphs.clear();
            for i in 0..num_morphs {
                let mut morph = ModelMorph::parse_pmx(self, buffer)?;
                morph.base.index = i as i32;
                self.morphs.push(Rc::new(RefCell::new(morph)));
            }
        }
        Ok(())
    }

    fn parse_label_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_labels = buffer.read_len()?;
        if num_labels > 0 {
            self.labels.clear();
            for i in 0..num_labels {
                let mut label = ModelLabel::parse_pmx(self, buffer)?;
                label.base.index = i as i32;
                self.labels.push(label);
            }
        }
        Ok(())
    }

    fn parse_rigid_body_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_rigid_bodies = buffer.read_len()?;
        if num_rigid_bodies > 0 {
            self.rigid_bodies.clear();
            for i in 0..num_rigid_bodies {
                let mut rigid_body = ModelRigidBody::parse_pmx(self, buffer)?;
                rigid_body.base.index = i as i32;
                self.rigid_bodies.push(rigid_body);
            }
        }
        Ok(())
    }

    fn parse_joint_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_joints = buffer.read_len()?;
        if num_joints > 0 {
            self.joints.clear();
            for i in 0..num_joints {
                let mut joint = ModelJoint::parse_pmx(self, buffer)?;
                joint.base.index = i as i32;
                self.joints.push(joint);
            }
        }
        Ok(())
    }

    fn parse_soft_body_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_soft_bodies = buffer.read_len()?;
        if num_soft_bodies > 0 {
            self.soft_bodies.clear();
            for i in 0..num_soft_bodies {
                let mut soft_body = ModelSoftBody::parse_pmx(self, buffer)?;
                soft_body.base.index = i as i32;
                self.soft_bodies.push(soft_body);
            }
        }
        Ok(())
    }

    fn parse_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        self.parse_vertex_block_pmx(buffer)?;
        self.parse_vertex_index_block_pmx(buffer)?;
        self.parse_texture_block_pmx(buffer)?;
        self.parse_material_block_pmx(buffer)?;
        self.parse_bone_block_pmx(buffer)?;
        self.parse_morph_block_pmx(buffer)?;
        self.parse_label_block_pmx(buffer)?;
        self.parse_rigid_body_block_pmx(buffer)?;
        self.parse_joint_block_pmx(buffer)?;
        if self.version > 2.0f32 && !buffer.is_end() {
            self.parse_soft_body_block_pmx(buffer)?;
        }
        if buffer.is_end() {
            Ok(())
        } else {
            Err(Status::ErrorBufferNotEnd)
        }
    }

    fn load_from_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let signature = buffer.read_u32_little_endian()?;
        if signature == fourcc('P' as u8, 'M' as u8, 'X' as u8, ' ' as u8)
            || signature == fourcc('P' as u8, 'M' as u8, 'X' as u8, 0xA0u8)
        {
            self.version = buffer.read_f32_little_endian()?;
            self.info_length = buffer.read_byte()?;
            if self.info_length == 8u8 {
                self.info.codec_type = buffer.read_byte()?;
                self.info.additional_uv_size = buffer.read_byte()?;
                self.info.vertex_index_size = buffer.read_byte()?;
                self.info.texture_index_size = buffer.read_byte()?;
                self.info.material_index_size = buffer.read_byte()?;
                self.info.bone_index_size = buffer.read_byte()?;
                self.info.morph_index_size = buffer.read_byte()?;
                self.info.rigid_body_index_size = buffer.read_byte()?;
                self.name_ja = self.get_string_pmx(buffer)?;
                self.name_en = self.get_string_pmx(buffer)?;
                self.comment_ja = self.get_string_pmx(buffer)?;
                self.comment_en = self.get_string_pmx(buffer)?;
                self.parse_pmx(buffer)?;
            }
            Ok(())
        } else {
            Err(Status::ErrorInvalidSignature)
        }
    }

    fn load_from_buffer(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let result = self.load_from_pmx(buffer);
        if result.err() == Some(Status::ErrorInvalidSignature) {
            Err(Status::ErrorNoSupportForPMD)
        } else {
            result
        }
    }

    pub fn get_format_type(&self) -> ModelFormatType {
        ((self.version * 10f32) as i32).into()
    }

    pub fn get_codec_type(&self) -> CodecType {
        let major_version = self.version as i32;
        if major_version >= 2 && self.info.codec_type != 0 {
            CodecType::Utf8
        } else if major_version >= 2 {
            CodecType::Utf16
        } else if major_version == 1 {
            CodecType::Sjis
        } else {
            CodecType::Unknown
        }
    }

    pub fn get_additional_uv_size(&self) -> usize {
        self.info.additional_uv_size.into()
    }

    pub fn get_name(&self, language_type: LanguageType) -> String {
        match language_type {
            LanguageType::Japanese => self.name_ja.clone(),
            LanguageType::English => self.name_en.clone(),
            LanguageType::Unknown => "".into(),
        }
    }

    pub fn get_comment(&self, language_type: LanguageType) -> String {
        match language_type {
            LanguageType::Japanese => self.comment_ja.clone(),
            LanguageType::English => self.comment_en.clone(),
            LanguageType::Unknown => "".into(),
        }
    }

    pub fn get_all_vertex_objects(&self) -> &Vec<ModelVertex> {
        &self.vertices
    }

    pub fn get_all_vertex_indices(&self) -> &Vec<u32> {
        &self.vertex_indices
    }

    pub fn get_all_material_objects(&self) -> &Vec<ModelMaterial> {
        &self.materials
    }

    pub fn get_all_bone_objects(&self) -> Vec<Weak<RefCell<ModelBone>>> {
        self.bones.iter().map(|rc| Rc::downgrade(&rc)).collect()
    }

    pub fn get_all_ordered_bone_object(&self) -> Vec<Weak<RefCell<ModelBone>>> {
        self.ordered_bones
            .iter()
            .map(|rc| Rc::downgrade(&rc))
            .collect()
    }

    pub fn get_all_constraint_objects(&self) -> &Vec<ModelConstraint> {
        &self.constraints
    }

    pub fn get_all_texture_objects(&self) -> &Vec<ModelTexture> {
        &self.textures
    }

    pub fn get_all_morph_objects(&self) -> Vec<Weak<RefCell<ModelMorph>>> {
        self.morphs.iter().map(|rc| Rc::downgrade(&rc)).collect()
    }

    pub fn get_all_label_objects(&self) -> &Vec<ModelLabel> {
        &self.labels
    }

    pub fn get_all_rigid_body_objects(&self) -> &Vec<ModelRigidBody> {
        &self.rigid_bodies
    }

    pub fn get_all_joint_objects(&self) -> &Vec<ModelJoint> {
        &self.joints
    }

    pub fn get_all_soft_body_objects(&self) -> &Vec<ModelSoftBody> {
        &self.soft_bodies
    }

    pub fn get_user_data(&self) -> Option<Rc<RefCell<UserData>>> {
        self.user_data.clone()
    }

    pub fn set_user_data(&mut self, user_data: Rc<RefCell<UserData>>) {
        self.user_data = Some(user_data.clone())
    }

    fn get_one_bone_object(&self, index: i32) -> Option<Rc<RefCell<ModelBone>>> {
        if index < 0 {
            None
        } else {
            self.bones.get(index as usize).map(|rc| rc.clone())
        }
    }

    fn get_one_morph_object(&self, index: i32) -> Option<Rc<RefCell<ModelMorph>>> {
        if index < 0 {
            None
        } else {
            self.morphs.get(index as usize).map(|rc| rc.clone())
        }
    }
}

struct ModelObject {
    index: i32,
    user_data: Option<Rc<RefCell<UserData>>>,
}

pub enum ModelVertexType {
    UNKNOWN,
    BDEF1,
    BDEF2,
    BDEF4,
    SDEF,
    QDEF,
}

impl From<i32> for ModelVertexType {
    fn from(v: i32) -> Self {
        match v {
            0 => Self::BDEF1,
            1 => Self::BDEF2,
            2 => Self::BDEF4,
            3 => Self::SDEF,
            4 => Self::QDEF,
            _ => Self::UNKNOWN,
        }
    }
}

impl Default for ModelVertexType {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

pub struct ModelVertex {
    base: ModelObject,
    origin: F128,
    normal: F128,
    uv: F128,
    additional_uv: [F128; 4],
    typ: ModelVertexType,
    num_bone_indices: usize,
    bone_indices: [i32; 4],
    num_bone_weights: usize,
    bone_weights: F128,
    sdef_c: F128,
    sdef_r0: F128,
    sdef_r1: F128,
    edge_size: f32,
    bone_weight_origin: u8,
}

impl ModelVertex {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelVertex, Status> {
        let mut vertex = ModelVertex {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            origin: buffer.read_f32_3_little_endian()?,
            normal: buffer.read_f32_3_little_endian()?,
            uv: F128([
                buffer.read_f32_little_endian()?,
                buffer.read_f32_little_endian()?,
                0.0f32,
                0.0f32,
            ]),
            additional_uv: <[F128; 4]>::default(),
            typ: ModelVertexType::from(buffer.read_byte()? as i32),
            num_bone_indices: usize::default(),
            bone_indices: <[i32; 4]>::default(),
            num_bone_weights: usize::default(),
            bone_weights: F128::default(),
            sdef_c: F128::default(),
            sdef_r0: F128::default(),
            sdef_r1: F128::default(),
            edge_size: f32::default(),
            bone_weight_origin: u8::default(),
        };
        for i in 0..parent_model.info.additional_uv_size {
            vertex.additional_uv[i as usize] = buffer.read_f32_4_little_endian()?;
        }
        let bone_index_size = parent_model.info.bone_index_size;
        match vertex.typ {
            ModelVertexType::UNKNOWN => return Err(Status::ErrorModelVertexCorrupted),
            ModelVertexType::BDEF1 => {
                vertex.bone_indices[0] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_weights.0[0] = 1.0f32;
                vertex.num_bone_indices = 1;
                vertex.num_bone_weights = 1;
            }
            ModelVertexType::BDEF2 => {
                vertex.bone_indices[0] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_indices[1] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_weights.0[0] = buffer.read_clamped_little_endian()?;
                vertex.bone_weights.0[1] = 1.0f32 - vertex.bone_weights.0[0];
                vertex.num_bone_indices = 2;
                vertex.num_bone_weights = 2;
            }
            ModelVertexType::BDEF4 | ModelVertexType::QDEF => {
                vertex.bone_indices[0] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_indices[1] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_indices[2] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_indices[3] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_weights = buffer.read_f32_4_little_endian()?;
                vertex.num_bone_indices = 4;
                vertex.num_bone_weights = 4;
            }
            ModelVertexType::SDEF => {
                vertex.bone_indices[0] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_indices[1] = buffer.read_integer_nullable(bone_index_size as usize)?;
                vertex.bone_weights.0[0] = buffer.read_clamped_little_endian()?;
                vertex.bone_weights.0[1] = 1.0f32 - vertex.bone_weights.0[0];
                vertex.num_bone_indices = 2;
                vertex.num_bone_weights = 2;
                vertex.sdef_c = buffer.read_f32_3_little_endian()?;
                vertex.sdef_c.0[3] = 1.0f32;
                vertex.sdef_r0 = buffer.read_f32_3_little_endian()?;
                vertex.sdef_r0.0[3] = 1.0f32;
                vertex.sdef_r1 = buffer.read_f32_3_little_endian()?;
                vertex.sdef_r1.0[3] = 1.0f32;
            }
        }
        vertex.edge_size = buffer.read_f32_little_endian()?;
        return Ok(vertex);
    }
}

#[derive(Default)]
pub struct ModelMaterialFlags {
    is_culling_disabled: bool,
    is_casting_shadow_enabled: bool,
    is_casting_shadow_map_enabled: bool,
    is_shadow_map_enabled: bool,
    is_edge_enabled: bool,
    is_vertex_color_enabled: bool,
    is_point_draw_enabled: bool,
    is_line_draw_enabled: bool,
}

impl ModelMaterialFlags {
    fn from_u8(value: u8) -> ModelMaterialFlags {
        return ModelMaterialFlags {
            is_culling_disabled: value % 2 != 0,
            is_casting_shadow_enabled: (value / 2) % 2 != 0,
            is_casting_shadow_map_enabled: (value / 4) % 2 != 0,
            is_shadow_map_enabled: (value / 8) % 2 != 0,
            is_edge_enabled: (value / 16) % 2 != 0,
            is_vertex_color_enabled: (value / 32) % 2 != 0,
            is_point_draw_enabled: (value / 64) % 2 != 0,
            is_line_draw_enabled: (value / 128) % 2 != 0,
        };
    }
}

pub enum ModelMaterialSpheremapTextureType {
    Unknown = -1,
    TypeNone,
    TypeMultiply,
    TypeAdd,
    TypeSubTexture,
}

impl Default for ModelMaterialSpheremapTextureType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl From<i32> for ModelMaterialSpheremapTextureType {
    fn from(v: i32) -> Self {
        match v {
            0 => Self::TypeNone,
            1 => Self::TypeMultiply,
            2 => Self::TypeAdd,
            3 => Self::TypeSubTexture,
            _ => Self::Unknown,
        }
    }
}

pub struct ModelMaterial {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    diffuse_color: F128,
    diffuse_opacity: f32,
    specular_power: f32,
    specular_color: F128,
    ambient_color: F128,
    edge_color: F128,
    edge_opacity: f32,
    edge_size: f32,
    diffuse_texture_index: i32,
    sphere_map_texture_index: i32,
    toon_texture_index: i32,
    sphere_map_texture_type: ModelMaterialSpheremapTextureType,
    is_toon_shared: bool,
    num_vertex_indices: usize,
    flags: ModelMaterialFlags,
    sphere_map_texture_sph: Option<Box<ModelTexture>>,
    sphere_map_texture_spa: Option<Box<ModelTexture>>,
    diffuse_texture: Option<Box<ModelTexture>>,
    clob: String,
}

impl ModelMaterial {
    pub fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelMaterial, Status> {
        let mut error: Option<Status> = None;
        let texture_index_size = parent_model.info.texture_index_size;
        let mut material = ModelMaterial {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            diffuse_color: buffer.read_f32_3_little_endian()?,
            diffuse_opacity: buffer.read_f32_little_endian()?,
            specular_color: buffer.read_f32_3_little_endian()?,
            specular_power: buffer.read_f32_little_endian()?,
            ambient_color: buffer.read_f32_3_little_endian()?,
            flags: ModelMaterialFlags::from_u8(buffer.read_byte()?),
            edge_color: buffer.read_f32_3_little_endian()?,
            edge_opacity: buffer.read_f32_little_endian()?,
            edge_size: buffer.read_f32_little_endian()?,
            diffuse_texture_index: buffer.read_integer_nullable(texture_index_size.into())?,
            sphere_map_texture_index: buffer.read_integer_nullable(texture_index_size.into())?,
            toon_texture_index: i32::default(),
            sphere_map_texture_type: ModelMaterialSpheremapTextureType::default(),
            is_toon_shared: bool::default(),
            num_vertex_indices: usize::default(),
            sphere_map_texture_sph: None,
            sphere_map_texture_spa: None,
            diffuse_texture: None,
            clob: String::default(),
        };
        if material.flags.is_point_draw_enabled {
            material.flags.is_casting_shadow_enabled = false;
            material.flags.is_casting_shadow_map_enabled = false;
            material.flags.is_shadow_map_enabled = false;
        } else if material.flags.is_line_draw_enabled {
            material.flags.is_edge_enabled = false;
        }
        let sphere_map_texture_type_raw = buffer.read_byte()?;
        let sphere_map_texture_type = if sphere_map_texture_type_raw == 0xffu8 {
            ModelMaterialSpheremapTextureType::TypeNone
        } else {
            ModelMaterialSpheremapTextureType::from(sphere_map_texture_type_raw as i32)
        };
        match sphere_map_texture_type {
            ModelMaterialSpheremapTextureType::Unknown => {
                error = Some(Status::ErrorModelMaterialCorrupted)
            }
            ModelMaterialSpheremapTextureType::TypeNone
            | ModelMaterialSpheremapTextureType::TypeMultiply
            | ModelMaterialSpheremapTextureType::TypeAdd
            | ModelMaterialSpheremapTextureType::TypeSubTexture => {
                material.sphere_map_texture_type = sphere_map_texture_type;
            }
        }
        material.is_toon_shared = buffer.read_byte()? != 0u8;
        if material.is_toon_shared {
            material.toon_texture_index = buffer.read_byte()? as i32;
        } else {
            material.toon_texture_index =
                buffer.read_integer_nullable(texture_index_size as usize)? as i32;
        }
        material.clob = parent_model.get_string_pmx(buffer)?;
        material.num_vertex_indices = buffer.read_i32_little_endian()? as usize;
        if let Some(err) = error {
            Err(err)
        } else {
            Ok(material)
        }
    }
}

enum ModelBoneType {
    Rotatable,
    RotatableAndMovable,
    ConstraintEffector,
    Unknown,
    ConstraintJoint,
    InherentOrientationJoint,
    ConstraintRoot,
    Invisible,
    FixedAxis,
    InherentOrientationEffector,
}

#[derive(Default)]
struct ModelBoneFlags {
    has_destination_bone_index: bool,
    is_rotatable: bool,
    is_movable: bool,
    is_visible: bool,
    is_user_handleable: bool,
    has_constraint: bool,
    has_local_inherent: bool,
    has_inherent_orientation: bool,
    has_inherent_translation: bool,
    has_fixed_axis: bool,
    has_local_axes: bool,
    is_affected_by_physics_simulation: bool,
    has_external_parent_bone: bool,
}

impl ModelBoneFlags {
    /// Original Definition
    /// struct nanoem_model_bone_flags_t {
    ///     unsigned int has_destination_bone_index : 1;
    ///     unsigned int is_rotateable : 1;
    ///     unsigned int is_movable : 1;
    ///     unsigned int is_visible : 1;
    ///     unsigned int is_user_handleable : 1;
    ///     unsigned int has_constraint : 1;
    ///     unsigned int padding_1 : 1;
    ///     unsigned int has_local_inherent : 1;
    ///     unsigned int has_inherent_orientation : 1;
    ///     unsigned int has_inherent_translation : 1;
    ///     unsigned int has_fixed_axis : 1;
    ///     unsigned int has_local_axes : 1;
    ///     unsigned int is_affected_by_physics_simulation : 1;
    ///     unsigned int has_external_parent_bone : 1;
    ///     unsigned int padding_2 : 2;
    /// } flags;
    fn from_raw(u: u16) -> ModelBoneFlags {
        ModelBoneFlags {
            has_destination_bone_index: u % 2 != 0,
            is_rotatable: (u / 2) % 2 != 0,
            is_movable: (u / 4) % 2 != 0,
            is_visible: (u / 8) % 2 != 0,
            is_user_handleable: (u / 16) % 2 != 0,
            has_constraint: (u / 32) % 2 != 0,
            has_local_inherent: (u / 128) % 2 != 0,
            has_inherent_orientation: (u / 256) % 2 != 0,
            has_inherent_translation: (u / 512) % 2 != 0,
            has_fixed_axis: (u / 1024) % 2 != 0,
            has_local_axes: (u / 2048) % 2 != 0,
            is_affected_by_physics_simulation: (u / 4096) % 2 != 0,
            has_external_parent_bone: (u / 8192) % 2 != 0,
        }
    }
}

#[test]
fn test_model_bone_flags_from_value() {
    let f = ModelBoneFlags::from_raw(33);
    assert_eq!(true, f.has_destination_bone_index);
    assert_eq!(true, f.has_constraint);
    assert_eq!(false, f.has_inherent_translation);
}

pub struct ModelBone {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    constraint: Option<Box<ModelConstraint>>,
    origin: F128,
    destination_origin: F128,
    fixed_axis: F128,
    local_x_axis: F128,
    local_z_axis: F128,
    inherent_coefficient: f32,
    parent_bone_index: i32,
    parent_inherent_bone_index: i32,
    effector_bone_index: i32,
    target_bone_index: i32,
    global_bone_index: i32,
    stage_index: i32,
    typ: ModelBoneType,
    flags: ModelBoneFlags,
}

impl ModelBone {
    fn compare_pmx(&self, other: &Self) -> i32 {
        if self.flags.is_affected_by_physics_simulation
            == other.flags.is_affected_by_physics_simulation
        {
            if self.stage_index == other.stage_index {
                return self.base.index - other.base.index;
            }
            return self.stage_index - other.stage_index;
        }
        return if other.flags.is_affected_by_physics_simulation {
            -1
        } else {
            1
        };
    }

    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelBone, Status> {
        let bone_index_size = parent_model.info.bone_index_size;
        let mut bone = ModelBone {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            origin: buffer.read_f32_3_little_endian()?,
            destination_origin: F128::default(),
            fixed_axis: F128::default(),
            local_x_axis: F128::default(),
            local_z_axis: F128::default(),
            inherent_coefficient: f32::default(),
            parent_bone_index: buffer.read_integer_nullable(bone_index_size as usize)?,
            parent_inherent_bone_index: i32::default(),
            effector_bone_index: i32::default(),
            target_bone_index: i32::default(),
            global_bone_index: i32::default(),
            stage_index: buffer.read_i32_little_endian()?,
            typ: ModelBoneType::Unknown,
            flags: ModelBoneFlags::from_raw(buffer.read_u16_little_endian()?),
            constraint: None,
        };
        if bone.flags.has_destination_bone_index {
            bone.target_bone_index = buffer.read_integer_nullable(bone_index_size as usize)?;
        } else {
            bone.destination_origin = buffer.read_f32_3_little_endian()?;
        }
        if bone.flags.has_inherent_orientation || bone.flags.has_inherent_translation {
            bone.parent_inherent_bone_index =
                buffer.read_integer_nullable(bone_index_size as usize)?;
            bone.inherent_coefficient = buffer.read_f32_little_endian()?;
        } else {
            bone.inherent_coefficient = 1.0f32;
        }
        if bone.flags.has_fixed_axis {
            bone.fixed_axis = buffer.read_f32_3_little_endian()?;
        }
        if bone.flags.has_local_axes {
            bone.local_x_axis = buffer.read_f32_3_little_endian()?;
            bone.local_z_axis = buffer.read_f32_3_little_endian()?;
        } else {
            bone.local_x_axis.0[0] = 1.0f32;
            bone.local_z_axis.0[2] = 1.0f32;
        }
        if bone.flags.has_external_parent_bone {
            bone.global_bone_index = buffer.read_i32_little_endian()?;
        }
        if bone.flags.has_constraint {
            bone.constraint = Some(Box::new(ModelConstraint::parse_pmx(parent_model, buffer)?))
        }
        Ok(bone)
    }

    // fn create(model: &Model) -> Result<&ModelVertex, Status> {
    //     vertex = ModelVertex {
    //         base: ModelObject {
    //             index: -1,
    //             user_data: model.user_data.clone(),
    //         },
    //         origin: todo!(),
    //         normal: todo!(),
    //         uv: todo!(),
    //         additional_uv: todo!(),
    //         typ: -1,
    //         num_bone_indices: todo!(),
    //         bone_indices: todo!(),
    //         num_bone_weights: todo!(),
    //         bone_weights: todo!(),
    //         sdef_c: todo!(),
    //         sdef_r0: todo!(),
    //         sdef_r1: todo!(),
    //         edge_size: todo!(),
    //         bone_weight_origin: todo!(),
    //     };
    // }
}

struct ModelConstraintJoint {
    base: ModelObject,
    bone_index: i32,
    has_angle_limit: bool,
    lower_limit: F128,
    upper_limit: F128,
}

pub struct ModelConstraint {
    base: ModelObject,
    effector_bone_index: i32,
    target_bone_index: i32,
    num_iterations: i32,
    angle_limit: f32,
    joints: Vec<ModelConstraintJoint>,
}

impl ModelConstraint {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelConstraint, Status> {
        let bone_index_size = parent_model.info.bone_index_size as usize;
        let mut constraint = ModelConstraint {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            effector_bone_index: buffer.read_integer_nullable(bone_index_size)?,
            target_bone_index: i32::default(),
            num_iterations: buffer.read_i32_little_endian()?,
            angle_limit: buffer.read_f32_little_endian()?,
            joints: vec![],
        };
        let num_joints = buffer.read_len()?;
        for i in 0..num_joints {
            let mut joint = ModelConstraintJoint {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                bone_index: buffer.read_integer_nullable(bone_index_size)?,
                has_angle_limit: buffer.read_byte()? != (0 as u8),
                lower_limit: F128::default(),
                upper_limit: F128::default(),
            };
            if joint.has_angle_limit {
                joint.lower_limit = buffer.read_f32_3_little_endian()?;
                joint.upper_limit = buffer.read_f32_3_little_endian()?;
            }
            constraint.joints.push(joint);
        }
        Ok(constraint)
    }
}

pub struct ModelMorphBone {
    base: ModelObject,
    bone_index: i32,
    translation: F128,
    orientation: F128,
}

impl ModelMorphBone {
    fn parse_pmx(
        bone_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphBone>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphBone {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                bone_index: buffer.read_integer_nullable(bone_index_size)?,
                translation: buffer.read_f32_3_little_endian()?,
                orientation: buffer.read_f32_3_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub struct ModelMorphGroup {
    base: ModelObject,
    morph_index: i32,
    weight: f32,
}

impl ModelMorphGroup {
    fn parse_pmx(
        morph_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphGroup>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphGroup {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                morph_index: buffer.read_integer_nullable(morph_index_size)?,
                weight: buffer.read_f32_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub struct ModelMorphFlip {
    base: ModelObject,
    morph_index: i32,
    weight: f32,
}

impl ModelMorphFlip {
    fn parse_pmx(
        morph_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphFlip>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for i in 0..num_objects {
            let item = ModelMorphFlip {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                morph_index: buffer.read_integer_nullable(morph_index_size)?,
                weight: buffer.read_f32_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub struct ModelMorphImpulse {
    base: ModelObject,
    rigid_body_index: i32,
    is_local: bool,
    velocity: F128,
    torque: F128,
}

impl ModelMorphImpulse {
    fn parse_pmx(
        rigid_body_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphImpulse>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphImpulse {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                rigid_body_index: buffer.read_integer_nullable(rigid_body_index_size)?,
                is_local: buffer.read_byte()? != 0,
                velocity: buffer.read_f32_3_little_endian()?,
                torque: buffer.read_f32_3_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub enum ModelMorphMaterialOperationType {
    Unknown = -1,
    Multiply,
    Add,
}

impl From<u8> for ModelMorphMaterialOperationType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelMorphMaterialOperationType::Multiply,
            1 => ModelMorphMaterialOperationType::Add,
            _ => ModelMorphMaterialOperationType::Unknown,
        }
    }
}

pub struct ModelMorphMaterial {
    base: ModelObject,
    material_index: i32,
    operation: ModelMorphMaterialOperationType,
    diffuse_color: F128,
    diffuse_opacity: f32,
    specular_color: F128,
    specular_power: f32,
    ambient_color: F128,
    edge_color: F128,
    edge_opacity: f32,
    edge_size: f32,
    diffuse_texture_blend: F128,
    sphere_map_texture_blend: F128,
    toon_texture_blend: F128,
}

impl ModelMorphMaterial {
    fn parse_pmx(
        material_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphMaterial>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphMaterial {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                material_index: buffer.read_integer_nullable(material_index_size)?,
                operation: buffer.read_byte()?.into(),
                diffuse_color: buffer.read_f32_3_little_endian()?,
                diffuse_opacity: buffer.read_f32_little_endian()?,
                specular_color: buffer.read_f32_3_little_endian()?,
                specular_power: buffer.read_f32_little_endian()?,
                ambient_color: buffer.read_f32_3_little_endian()?,
                edge_color: buffer.read_f32_3_little_endian()?,
                edge_opacity: buffer.read_f32_little_endian()?,
                edge_size: buffer.read_f32_little_endian()?,
                diffuse_texture_blend: buffer.read_f32_4_little_endian()?,
                sphere_map_texture_blend: buffer.read_f32_4_little_endian()?,
                toon_texture_blend: buffer.read_f32_4_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub struct ModelMorphUv {
    base: ModelObject,
    vertex_index: i32,
    position: F128,
}

impl ModelMorphUv {
    fn parse_pmx(
        vertex_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphUv>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphUv {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                vertex_index: buffer.read_integer(vertex_index_size)?,
                position: buffer.read_f32_4_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub struct ModelMorphVertex {
    base: ModelObject,
    vertex_index: i32,
    relative_index: i32,
    position: F128,
}

impl ModelMorphVertex {
    fn parse_pmx(
        vertex_index_size: usize,
        buffer: &mut Buffer,
    ) -> Result<Vec<ModelMorphVertex>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for _ in 0..num_objects {
            let item = ModelMorphVertex {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                vertex_index: buffer.read_integer(vertex_index_size)?,
                relative_index: -1,
                position: buffer.read_f32_3_little_endian()?,
            };
            vec.push(item);
        }
        Ok(vec)
    }
}

pub enum ModelMorphU {
    GROUPS(Vec<ModelMorphGroup>),
    VERTICES(Vec<ModelMorphVertex>),
    BONES(Vec<ModelMorphBone>),
    UVS(Vec<ModelMorphUv>),
    MATERIALS(Vec<ModelMorphMaterial>),
    FLIPS(Vec<ModelMorphFlip>),
    IMPULSES(Vec<ModelMorphImpulse>),
}

pub enum ModelMorphCategory {
    Unknown = -1,
    Base,
    Eyebrow,
    Eye,
    Lip,
    Other,
}

impl From<u8> for ModelMorphCategory {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelMorphCategory::Base,
            1 => ModelMorphCategory::Eyebrow,
            2 => ModelMorphCategory::Eye,
            3 => ModelMorphCategory::Lip,
            4 => ModelMorphCategory::Other,
            _ => ModelMorphCategory::Unknown,
        }
    }
}

pub enum ModelMorphType {
    Unknown = -1,
    Group,
    Vertex,
    Bone,
    Texture,
    Uva1,
    Uva2,
    Uva3,
    Uva4,
    Material,
    Flip,
    Impulse,
}

impl From<u8> for ModelMorphType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelMorphType::Group,
            1 => ModelMorphType::Vertex,
            2 => ModelMorphType::Bone,
            3 => ModelMorphType::Texture,
            4 => ModelMorphType::Uva1,
            5 => ModelMorphType::Uva2,
            6 => ModelMorphType::Uva3,
            7 => ModelMorphType::Uva4,
            8 => ModelMorphType::Material,
            9 => ModelMorphType::Flip,
            10 => ModelMorphType::Impulse,
            _ => ModelMorphType::Unknown,
        }
    }
}

pub struct ModelMorph {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    typ: ModelMorphType,
    category: ModelMorphCategory,
    u: ModelMorphU,
}

impl ModelMorph {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelMorph, Status> {
        let mut morph = ModelMorph {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            category: ModelMorphCategory::from(buffer.read_byte()?),
            typ: ModelMorphType::from(buffer.read_byte()?),
            u: ModelMorphU::BONES(vec![]),
        };
        match morph.typ {
            ModelMorphType::Bone => {
                morph.u = ModelMorphU::BONES(ModelMorphBone::parse_pmx(
                    parent_model.info.bone_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Flip => {
                morph.u = ModelMorphU::FLIPS(ModelMorphFlip::parse_pmx(
                    parent_model.info.morph_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Group => {
                morph.u = ModelMorphU::GROUPS(ModelMorphGroup::parse_pmx(
                    parent_model.info.morph_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Impulse => {
                morph.u = ModelMorphU::IMPULSES(ModelMorphImpulse::parse_pmx(
                    parent_model.info.rigid_body_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Material => {
                morph.u = ModelMorphU::MATERIALS(ModelMorphMaterial::parse_pmx(
                    parent_model.info.material_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Texture
            | ModelMorphType::Uva1
            | ModelMorphType::Uva2
            | ModelMorphType::Uva3
            | ModelMorphType::Uva4 => {
                morph.u = ModelMorphU::UVS(ModelMorphUv::parse_pmx(
                    parent_model.info.vertex_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Vertex => {
                morph.u = ModelMorphU::VERTICES(ModelMorphVertex::parse_pmx(
                    parent_model.info.vertex_index_size as usize,
                    buffer,
                )?)
            }
            ModelMorphType::Unknown => return Err(Status::ErrorModelMorphCorrupted),
        }
        Ok(morph)
    }
}

enum ModelLabelItemType {
    Unknown = -1,
    Bone,
    Morph,
}

impl From<u8> for ModelLabelItemType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelLabelItemType::Bone,
            1 => ModelLabelItemType::Morph,
            _ => ModelLabelItemType::Unknown,
        }
    }
}

/// TODO: need option really?
enum ModelLabelItemU {
    BONE(Option<Weak<RefCell<ModelBone>>>),
    MORPH(Option<Weak<RefCell<ModelMorph>>>),
}

struct ModelLabelItem {
    base: ModelObject,
    typ: ModelLabelItemType,
    u: ModelLabelItemU,
}

pub struct ModelLabel {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    is_special: bool,
    items: Vec<ModelLabelItem>,
}

impl ModelLabel {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelLabel, Status> {
        let bone_index_size = parent_model.info.bone_index_size as usize;
        let morph_index_size = parent_model.info.morph_index_size as usize;
        let mut label = ModelLabel {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            is_special: buffer.read_byte()? != 0,
            items: vec![],
        };
        let num_items = buffer.read_len()?;
        for _ in 0..num_items {
            let item_type = ModelLabelItemType::from(buffer.read_byte()?);
            match item_type {
                ModelLabelItemType::Bone => label.items.push(ModelLabelItem {
                    base: ModelObject {
                        index: -1,
                        user_data: None,
                    },
                    typ: ModelLabelItemType::Bone,
                    u: ModelLabelItemU::BONE(
                        parent_model
                            .get_one_bone_object(buffer.read_integer_nullable(bone_index_size)?)
                            .map(|rc| Rc::downgrade(&rc)),
                    ),
                }),
                ModelLabelItemType::Morph => label.items.push(ModelLabelItem {
                    base: ModelObject {
                        index: -1,
                        user_data: None,
                    },
                    typ: ModelLabelItemType::Morph,
                    u: ModelLabelItemU::MORPH(
                        parent_model
                            .get_one_morph_object(buffer.read_integer_nullable(morph_index_size)?)
                            .map(|rc| Rc::downgrade(&rc)),
                    ),
                }),
                ModelLabelItemType::Unknown => return Err(Status::ErrorModelLabelCorrupted),
            }
        }
        Ok(label)
    }
}

pub enum ModelRigidBodyShapeType {
    Unknown = -1,
    Sphere,
    Box,
    Capsule,
}

impl From<u8> for ModelRigidBodyShapeType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelRigidBodyShapeType::Sphere,
            1 => ModelRigidBodyShapeType::Box,
            2 => ModelRigidBodyShapeType::Capsule,
            _ => ModelRigidBodyShapeType::Unknown,
        }
    }
}

pub enum ModelRigidBodyTransformType {
    Unknown = -1,
    FromBoneToSimulation,
    FromSimulationToBone,
    FromBoneOrientationAndSimulationToBone,
}

impl From<u8> for ModelRigidBodyTransformType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelRigidBodyTransformType::FromBoneToSimulation,
            1 => ModelRigidBodyTransformType::FromSimulationToBone,
            2 => ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone,
            _ => ModelRigidBodyTransformType::Unknown,
        }
    }
}

pub struct ModelRigidBody {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    bone_index: i32,
    collision_group_id: i32,
    collision_mask: i32,
    shape_type: ModelRigidBodyShapeType,
    size: F128,
    origin: F128,
    orientation: F128,
    mass: f32,
    linear_damping: f32,
    angular_damping: f32,
    restitution: f32,
    friction: f32,
    transform_type: ModelRigidBodyTransformType,
    is_bone_relative: bool,
}

impl ModelRigidBody {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelRigidBody, Status> {
        // TODO: not process Unknown for shpe_type and transform_type
        let mut rigid_body = ModelRigidBody {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            bone_index: buffer.read_integer_nullable(parent_model.info.bone_index_size as usize)?,
            collision_group_id: buffer.read_byte()? as i32,
            collision_mask: buffer.read_i16_little_endian()? as i32,
            shape_type: buffer.read_byte()?.into(),
            size: buffer.read_f32_3_little_endian()?,
            origin: buffer.read_f32_3_little_endian()?,
            orientation: buffer.read_f32_3_little_endian()?,
            mass: buffer.read_f32_little_endian()?,
            linear_damping: buffer.read_f32_little_endian()?,
            angular_damping: buffer.read_f32_little_endian()?,
            restitution: buffer.read_f32_little_endian()?,
            friction: buffer.read_f32_little_endian()?,
            transform_type: buffer.read_byte()?.into(),
            is_bone_relative: false,
        };
        Ok(rigid_body)
    }
}

pub enum ModelJointType {
    Unknown = -1,
    Generic6dofSpringConstraint,
    Generic6dofConstraint,
    Point2pointConstraint,
    ConeTwistConstraint,
    SliderConstraint,
    HingeConstraint,
}

impl From<u8> for ModelJointType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelJointType::Generic6dofSpringConstraint,
            1 => ModelJointType::Generic6dofConstraint,
            2 => ModelJointType::Point2pointConstraint,
            3 => ModelJointType::ConeTwistConstraint,
            4 => ModelJointType::SliderConstraint,
            5 => ModelJointType::HingeConstraint,
            _ => ModelJointType::Unknown,
        }
    }
}

pub struct ModelJoint {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    rigid_body_a_index: i32,
    rigid_body_b_index: i32,
    typ: ModelJointType,
    origin: F128,
    orientation: F128,
    linear_lower_limit: F128,
    linear_upper_limit: F128,
    angular_lower_limit: F128,
    angular_upper_limit: F128,
    linear_stiffness: F128,
    angular_stiffness: F128,
}

impl ModelJoint {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelJoint, Status> {
        let rigid_body_index_size = parent_model.info.rigid_body_index_size as usize;
        let mut joint = ModelJoint {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            typ: buffer.read_byte()?.into(),
            rigid_body_a_index: buffer.read_integer_nullable(rigid_body_index_size)?,
            rigid_body_b_index: buffer.read_integer_nullable(rigid_body_index_size)?,
            origin: buffer.read_f32_3_little_endian()?,
            orientation: buffer.read_f32_3_little_endian()?,
            linear_lower_limit: buffer.read_f32_3_little_endian()?,
            linear_upper_limit: buffer.read_f32_3_little_endian()?,
            angular_lower_limit: buffer.read_f32_3_little_endian()?,
            angular_upper_limit: buffer.read_f32_3_little_endian()?,
            linear_stiffness: buffer.read_f32_3_little_endian()?,
            angular_stiffness: buffer.read_f32_3_little_endian()?,
        };
        Ok(joint)
    }
}

pub enum ModelSoftBodyShapeType {
    Unknown = -1,
    TriMesh,
    Rope,
}

impl From<u8> for ModelSoftBodyShapeType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModelSoftBodyShapeType::TriMesh,
            1 => ModelSoftBodyShapeType::Rope,
            _ => ModelSoftBodyShapeType::Unknown,
        }
    }
}

pub enum ModelSoftBodyAeroModelType {
    Unknown = -1,
    VertexPoint,
    VertexTwoSided,
    VertexOneSided,
    FaceTwoSided,
    FaceOneSided,
}

impl From<i32> for ModelSoftBodyAeroModelType {
    fn from(value: i32) -> Self {
        match value {
            0 => ModelSoftBodyAeroModelType::VertexPoint,
            1 => ModelSoftBodyAeroModelType::VertexTwoSided,
            2 => ModelSoftBodyAeroModelType::VertexOneSided,
            3 => ModelSoftBodyAeroModelType::FaceTwoSided,
            4 => ModelSoftBodyAeroModelType::FaceOneSided,
            _ => ModelSoftBodyAeroModelType::Unknown,
        }
    }
}

struct ModelSoftBodyAnchor {
    base: ModelObject,
    rigid_body_index: i32,
    vertex_index: i32,
    is_near_enabled: bool,
}

pub struct ModelSoftBody {
    base: ModelObject,
    name_ja: String,
    name_en: String,
    shape_type: ModelSoftBodyShapeType,
    material_index: i32,
    collision_group_id: u8,
    collision_mask: u16,
    flags: u8,
    bending_constraints_distance: i32,
    cluster_count: i32,
    total_mass: f32,
    collision_margin: f32,
    aero_model: ModelSoftBodyAeroModelType,
    velocity_correction_factor: f32,
    damping_coefficient: f32,
    drag_coefficient: f32,
    lift_coefficient: f32,
    pressure_coefficient: f32,
    volume_convenrsation_coefficient: f32,
    dynamic_friction_coefficient: f32,
    pose_matching_coefficient: f32,
    rigid_contact_hardness: f32,
    kinetic_contact_hardness: f32,
    soft_contact_hardness: f32,
    anchor_hardness: f32,
    soft_vs_rigid_hardness: f32,
    soft_vs_kinetic_hardness: f32,
    soft_vs_soft_hardness: f32,
    soft_vs_rigid_impulse_split: f32,
    soft_vs_kinetic_impulse_split: f32,
    soft_vs_soft_impulse_split: f32,
    velocity_solver_iterations: i32,
    positions_solver_iterations: i32,
    drift_solver_iterations: i32,
    cluster_solver_iterations: i32,
    linear_stiffness_coefficient: f32,
    angular_stiffness_coefficient: f32,
    volume_stiffness_coefficient: f32,
    anchors: Vec<ModelSoftBodyAnchor>,
    pinned_vertex_indices: Vec<u32>,
}

impl ModelSoftBody {
    fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelSoftBody, Status> {
        let material_index_size = parent_model.info.material_index_size as usize;
        let rigid_body_index_size = parent_model.info.rigid_body_index_size as usize;
        let vertex_index_size = parent_model.info.vertex_index_size as usize;
        let mut soft_body = ModelSoftBody {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            name_ja: parent_model.get_string_pmx(buffer)?,
            name_en: parent_model.get_string_pmx(buffer)?,
            shape_type: buffer.read_byte()?.into(),
            material_index: buffer.read_integer_nullable(material_index_size)?,
            collision_group_id: buffer.read_byte()?,
            collision_mask: buffer.read_u16_little_endian()?,
            flags: buffer.read_byte()?.into(),
            bending_constraints_distance: buffer.read_i32_little_endian()?,
            cluster_count: buffer.read_i32_little_endian()?,
            total_mass: buffer.read_f32_little_endian()?,
            collision_margin: buffer.read_f32_little_endian()?,
            aero_model: buffer.read_i32_little_endian()?.into(),
            velocity_correction_factor: buffer.read_f32_little_endian()?,
            damping_coefficient: buffer.read_f32_little_endian()?,
            drag_coefficient: buffer.read_f32_little_endian()?,
            lift_coefficient: buffer.read_f32_little_endian()?,
            pressure_coefficient: buffer.read_f32_little_endian()?,
            volume_convenrsation_coefficient: buffer.read_f32_little_endian()?,
            dynamic_friction_coefficient: buffer.read_f32_little_endian()?,
            pose_matching_coefficient: buffer.read_f32_little_endian()?,
            rigid_contact_hardness: buffer.read_f32_little_endian()?,
            kinetic_contact_hardness: buffer.read_f32_little_endian()?,
            soft_contact_hardness: buffer.read_f32_little_endian()?,
            anchor_hardness: buffer.read_f32_little_endian()?,
            soft_vs_rigid_hardness: buffer.read_f32_little_endian()?,
            soft_vs_kinetic_hardness: buffer.read_f32_little_endian()?,
            soft_vs_soft_hardness: buffer.read_f32_little_endian()?,
            soft_vs_rigid_impulse_split: buffer.read_f32_little_endian()?,
            soft_vs_kinetic_impulse_split: buffer.read_f32_little_endian()?,
            soft_vs_soft_impulse_split: buffer.read_f32_little_endian()?,
            velocity_solver_iterations: buffer.read_i32_little_endian()?,
            positions_solver_iterations: buffer.read_i32_little_endian()?,
            drift_solver_iterations: buffer.read_i32_little_endian()?,
            cluster_solver_iterations: buffer.read_i32_little_endian()?,
            linear_stiffness_coefficient: buffer.read_f32_little_endian()?,
            angular_stiffness_coefficient: buffer.read_f32_little_endian()?,
            volume_stiffness_coefficient: buffer.read_f32_little_endian()?,
            anchors: vec![],
            pinned_vertex_indices: vec![],
        };
        let num_anchors = buffer.read_len()?;
        for _ in 0..num_anchors {
            soft_body.anchors.push(ModelSoftBodyAnchor {
                base: ModelObject {
                    index: -1,
                    user_data: None,
                },
                rigid_body_index: buffer.read_integer_nullable(rigid_body_index_size)?,
                vertex_index: buffer.read_integer_nullable(vertex_index_size)?,
                is_near_enabled: buffer.read_byte()? != 0,
            })
        }
        let num_pin_vertex_indices = buffer.read_len()?;
        for _ in 0..num_pin_vertex_indices {
            soft_body
                .pinned_vertex_indices
                .push(buffer.read_integer(vertex_index_size)? as u32);
        }
        Ok(soft_body)
    }
}

pub struct ModelTexture {
    base: ModelObject,
    path: String,
}

impl ModelTexture {
    pub fn parse_pmx(parent_model: &Model, buffer: &mut Buffer) -> Result<ModelTexture, Status> {
        Ok(ModelTexture {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            path: parent_model.get_string_pmx(buffer)?,
        })
    }
}

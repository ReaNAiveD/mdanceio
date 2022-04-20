use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{
    common::{Buffer, CodecType, LanguageType, MutableBuffer, Status, UserData, F128},
    utils::fourcc,
};

pub static NANOEM_MODEL_OBJECT_NOT_FOUND: i32 = -1;

#[derive(Debug, Default)]
pub struct Info {
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

pub struct Model<
    VertexDataType,
    MaterialDataType,
    BoneDataType,
    ConstraintDataType,
    MorphDataType,
    LabelDataType,
    RigidBodyDataType,
    JointDataType,
    SoftBodyDataType,
> {
    version: f32,
    info_length: u8,
    info: Info,
    name_ja: String,
    name_en: String,
    comment_ja: String,
    comment_en: String,
    vertices: Vec<Rc<RefCell<ModelVertex<VertexDataType>>>>,
    vertex_indices: Vec<Rc<RefCell<u32>>>,
    materials: Vec<Rc<RefCell<ModelMaterial<MaterialDataType>>>>,
    bones: Vec<Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>>,
    ordered_bones: Vec<Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>>,
    constraints: Vec<Rc<RefCell<ModelConstraint<ConstraintDataType>>>>,
    textures: Vec<Rc<RefCell<ModelTexture>>>,
    morphs: Vec<Rc<RefCell<ModelMorph<MorphDataType>>>>,
    labels: Vec<
        Rc<RefCell<ModelLabel<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>>>,
    >,
    rigid_bodies: Vec<Rc<RefCell<ModelRigidBody<RigidBodyDataType>>>>,
    joints: Vec<Rc<RefCell<ModelJoint<JointDataType>>>>,
    soft_bodies: Vec<Rc<RefCell<ModelSoftBody<SoftBodyDataType>>>>,
    user_data: Option<Rc<RefCell<UserData>>>,
}

impl<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    > Default
    for Model<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >
{
    fn default() -> Self {
        Self {
            version: Default::default(),
            info_length: Default::default(),
            info: Default::default(),
            name_ja: Default::default(),
            name_en: Default::default(),
            comment_ja: Default::default(),
            comment_en: Default::default(),
            vertices: Default::default(),
            vertex_indices: Default::default(),
            materials: Default::default(),
            bones: Default::default(),
            ordered_bones: Default::default(),
            constraints: Default::default(),
            textures: Default::default(),
            morphs: Default::default(),
            labels: Default::default(),
            rigid_bodies: Default::default(),
            joints: Default::default(),
            soft_bodies: Default::default(),
            user_data: Default::default(),
        }
    }
}

impl<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >
    Model<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >
{
    const PMX_SIGNATURE: &'static str = "PMX ";

    fn create_empty() -> Self {
        Self {
            version: todo!(),
            info_length: todo!(),
            info: todo!(),
            name_ja: todo!(),
            name_en: todo!(),
            comment_ja: todo!(),
            comment_en: todo!(),
            vertices: todo!(),
            vertex_indices: todo!(),
            materials: todo!(),
            bones: todo!(),
            ordered_bones: todo!(),
            constraints: todo!(),
            textures: todo!(),
            morphs: todo!(),
            labels: todo!(),
            rigid_bodies: todo!(),
            joints: todo!(),
            soft_bodies: todo!(),
            user_data: todo!(),
        }
    }

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
                let mut vertex = ModelVertex::parse_pmx(self, buffer)?;
                vertex.base.index = i as i32;
                self.vertices.push(Rc::new(RefCell::new(vertex)));
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
                        Rc::new(RefCell::new(vertex_index))
                    } else {
                        Rc::new(RefCell::new(0))
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
                let mut texture = ModelTexture::parse_pmx(self, buffer)?;
                texture.base.index = i as i32;
                self.textures.push(Rc::new(RefCell::new(texture)));
            }
        }
        Ok(())
    }

    fn parse_material_block_pmx(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let num_materials = buffer.read_len()?;
        if num_materials > 0 {
            self.materials.clear();
            for i in 0..num_materials {
                let mut material = ModelMaterial::parse_pmx(self, buffer)?;
                material.base.index = i as i32;
                self.materials.push(Rc::new(RefCell::new(material)));
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
                if let Some(constraint) = &bone.constraint {
                    constraint.borrow_mut().target_bone_index = i as i32;
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
                self.labels.push(Rc::new(RefCell::new(label)));
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
                self.rigid_bodies.push(Rc::new(RefCell::new(rigid_body)));
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
                self.joints.push(Rc::new(RefCell::new(joint)));
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
                self.soft_bodies.push(Rc::new(RefCell::new(soft_body)));
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

    pub fn load_from_buffer(&mut self, buffer: &mut Buffer) -> Result<(), Status> {
        let result = self.load_from_pmx(buffer);
        if result.err() == Some(Status::ErrorInvalidSignature) {
            Err(Status::ErrorNoSupportForPMD)
        } else {
            result
        }
    }

    fn vertices_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.vertices.len() as i32)?;
        for vertex in &self.vertices {
            vertex.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn textures_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.textures.len() as i32)?;
        for texture in &self.textures {
            texture.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn materials_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.materials.len() as i32)?;
        for material in &self.materials {
            material.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn bones_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_integer(self.bones.len() as i32, if self.is_pmx() { 4 } else { 2 })?;
        for bone in &self.bones {
            bone.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn morphs_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_integer(self.morphs.len() as i32, if self.is_pmx() { 4 } else { 2 })?;
        for morph in &self.morphs {
            morph.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn labels_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        if self.is_pmx() {
            buffer.write_integer(self.labels.len() as i32, 4)?;
            for label in &self.labels {
                label.borrow().save_to_buffer(buffer, self)?;
            }
            Ok(())
        } else {
            Err(Status::ErrorNoSupportForPMD)
        }
    }

    fn rigid_bodies_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.rigid_bodies.len() as i32)?;
        for rigid_body in &self.rigid_bodies {
            rigid_body.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn joints_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.joints.len() as i32)?;
        for joint in &self.joints {
            joint.borrow().save_to_buffer(buffer, self)?;
        }
        Ok(())
    }

    fn soft_bodies_save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        if self.is_pmx21() {
            buffer.write_i32_little_endian(self.soft_bodies.len() as i32)?;
            for soft_body in &self.soft_bodies {
                soft_body.borrow().save_to_buffer(buffer, self)?;
            }
        }
        Ok(())
    }

    pub fn save_to_buffer_pmx(&mut self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        let codec_type = self.get_codec_type();
        self.info_length = 8;
        self.info.vertex_index_size = Self::get_vertex_index_size(self.vertices.len());
        self.info.texture_index_size = Self::get_object_index_size(self.textures.len());
        self.info.material_index_size = Self::get_object_index_size(self.materials.len());
        self.info.bone_index_size = Self::get_object_index_size(self.bones.len());
        self.info.morph_index_size = Self::get_object_index_size(self.morphs.len());
        self.info.rigid_body_index_size = Self::get_object_index_size(self.rigid_bodies.len());
        buffer.write_byte_array(&Self::PMX_SIGNATURE.as_bytes()[..4])?;
        buffer.write_f32_little_endian(self.version)?;
        buffer.write_byte(self.info_length)?;
        buffer.write_byte(self.info.codec_type)?;
        buffer.write_byte(self.info.additional_uv_size)?;
        buffer.write_byte(self.info.vertex_index_size)?;
        buffer.write_byte(self.info.texture_index_size)?;
        buffer.write_byte(self.info.material_index_size)?;
        buffer.write_byte(self.info.bone_index_size)?;
        buffer.write_byte(self.info.morph_index_size)?;
        buffer.write_byte(self.info.rigid_body_index_size)?;
        buffer.write_string(&self.name_ja, codec_type)?;
        buffer.write_string(&self.name_en, codec_type)?;
        buffer.write_string(&self.comment_ja, codec_type)?;
        buffer.write_string(&self.comment_en, codec_type)?;
        self.vertices_save_to_buffer(buffer)?;
        let vertex_index_size = self.info.vertex_index_size as usize;
        buffer.write_i32_little_endian(self.vertex_indices.len() as i32)?;
        for vertex_index in &self.vertex_indices {
            buffer.write_integer(vertex_index.borrow().clone() as i32, vertex_index_size)?;
        }
        self.textures_save_to_buffer(buffer)?;
        self.materials_save_to_buffer(buffer)?;
        self.bones_save_to_buffer(buffer)?;
        self.morphs_save_to_buffer(buffer)?;
        self.labels_save_to_buffer(buffer)?;
        self.rigid_bodies_save_to_buffer(buffer)?;
        self.joints_save_to_buffer(buffer)?;
        self.soft_bodies_save_to_buffer(buffer)?;
        Ok(())
    }

    pub fn save_to_buffer(&mut self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        if self.is_pmx() {
            self.save_to_buffer_pmx(buffer)
        } else {
            Err(Status::ErrorNoSupportForPMD)
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

    pub fn get_name(&self, language_type: LanguageType) -> &str {
        match language_type {
            LanguageType::Japanese => self.name_ja.as_str(),
            LanguageType::English => self.name_en.as_str(),
            LanguageType::Unknown => "",
        }
    }

    pub fn get_comment(&self, language_type: LanguageType) -> &str {
        match language_type {
            LanguageType::Japanese => self.comment_ja.as_str(),
            LanguageType::English => self.comment_en.as_str(),
            LanguageType::Unknown => "",
        }
    }

    pub fn get_all_vertex_objects(&self) -> &[Rc<RefCell<ModelVertex<VertexDataType>>>] {
        &self.vertices
    }

    pub fn get_all_vertex_indices(&self) -> &[Rc<RefCell<u32>>] {
        &self.vertex_indices
    }

    pub fn get_all_material_objects(&self) -> &[Rc<RefCell<ModelMaterial<MaterialDataType>>>] {
        &self.materials
    }

    pub fn get_all_bone_objects(
        &self,
    ) -> &[Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>] {
        &self.bones
    }

    pub fn get_all_ordered_bone_object(
        &self,
    ) -> &[Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>] {
        &self.ordered_bones
    }

    pub fn get_all_constraint_objects(
        &self,
    ) -> &[Rc<RefCell<ModelConstraint<ConstraintDataType>>>] {
        &self.constraints
    }

    pub fn get_all_texture_objects(&self) -> &[Rc<RefCell<ModelTexture>>] {
        &self.textures
    }

    pub fn get_all_morph_objects(&self) -> &[Rc<RefCell<ModelMorph<MorphDataType>>>] {
        &self.morphs
    }

    pub fn get_all_label_objects(
        &self,
    ) -> &[Rc<RefCell<ModelLabel<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>>>]
    {
        &self.labels
    }

    pub fn get_all_rigid_body_objects(&self) -> &[Rc<RefCell<ModelRigidBody<RigidBodyDataType>>>] {
        &self.rigid_bodies
    }

    pub fn get_all_joint_objects(&self) -> &[Rc<RefCell<ModelJoint<JointDataType>>>] {
        &self.joints
    }

    pub fn get_all_soft_body_objects(&self) -> &[Rc<RefCell<ModelSoftBody<SoftBodyDataType>>>] {
        &self.soft_bodies
    }

    pub fn get_user_data(&self) -> Option<Rc<RefCell<UserData>>> {
        self.user_data.clone()
    }

    pub fn set_user_data(&mut self, user_data: Rc<RefCell<UserData>>) {
        self.user_data = Some(user_data.clone())
    }

    pub fn get_one_vertex_object(
        &self,
        index: i32,
    ) -> Option<&Rc<RefCell<ModelVertex<VertexDataType>>>> {
        if index < 0 {
            None
        } else {
            self.vertices.get(index as usize)
        }
    }

    pub fn get_one_bone_object(
        &self,
        index: i32,
    ) -> Option<&Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>> {
        if index < 0 {
            None
        } else {
            self.bones.get(index as usize)
        }
    }

    pub fn get_one_morph_object(
        &self,
        index: i32,
    ) -> Option<&Rc<RefCell<ModelMorph<MorphDataType>>>> {
        if index < 0 {
            None
        } else {
            self.morphs.get(index as usize)
        }
    }

    pub fn get_one_texture_object(&self, index: i32) -> Option<&Rc<RefCell<ModelTexture>>> {
        if index < 0 {
            None
        } else {
            self.textures.get(index as usize)
        }
    }

    pub fn is_pmx(&self) -> bool {
        (self.version * 10f32) as i32 >= 20
    }

    pub fn is_pmx21(&self) -> bool {
        (self.version * 10f32) as i32 >= 21
    }

    pub fn get_vertex_index_size(size: usize) -> u8 {
        if size <= 0xff {
            return 1u8;
        } else if size <= 0xffff {
            return 2u8;
        } else {
            return 4u8;
        }
    }

    pub fn get_object_index_size(size: usize) -> u8 {
        if size <= 0x7f {
            return 1u8;
        } else if size <= 0x7fff {
            return 2u8;
        } else {
            return 4u8;
        }
    }

    pub fn set_additional_uv_size(&mut self, value: usize) {
        if (0..=4).contains(&value) {
            self.info.additional_uv_size = value as u8;
        }
    }

    pub fn set_codec_type(&mut self, value: CodecType) {
        self.info.codec_type = match value {
            CodecType::Unknown => 0,
            CodecType::Sjis => 0,
            CodecType::Utf8 => 1,
            CodecType::Utf16 => 0,
        }
    }

    pub fn set_format_type(&mut self, value: ModelFormatType) {
        self.version = match value {
            ModelFormatType::Unknown => self.version,
            ModelFormatType::Pmd1_0 => 1.0f32,
            ModelFormatType::Pmx2_0 => 2.0f32,
            ModelFormatType::Pmx2_1 => 2.1f32,
        }
    }

    pub fn set_name(&mut self, value: &String, language_type: LanguageType) {
        match language_type {
            LanguageType::Unknown => (),
            LanguageType::Japanese => self.name_ja = value.to_string(),
            LanguageType::English => self.name_en = value.to_string(),
        }
    }

    pub fn set_comment(&mut self, value: &String, language_type: LanguageType) {
        match language_type {
            LanguageType::Unknown => (),
            LanguageType::Japanese => self.comment_ja = value.to_string(),
            LanguageType::English => self.comment_en = value.to_string(),
        }
    }

    fn find_indexed_rc_offset<T>(objects: &Vec<Rc<T>>, target: &Rc<T>, index: i32) -> i32 {
        let mut offset = -1;
        if index >= 0 && objects.len() > 0 {
            if (index as usize) < objects.len()
                && objects
                    .get(index as usize)
                    .map_or(false, |rc| Rc::ptr_eq(rc, target))
            {
                offset = index;
            } else {
                for ptr in objects.iter().enumerate() {
                    if Rc::ptr_eq(target, ptr.1) {
                        offset = ptr.0 as i32;
                    }
                }
            }
        }
        offset
    }

    fn contains_indexed_rc<T>(objects: &Vec<Rc<T>>, target: &Rc<T>, index: i32) -> bool {
        Self::find_indexed_rc_offset(objects, target, index) != -1
    }

    pub fn insert_bone(
        &mut self,
        bone: &Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>,
        mut index: i32,
    ) -> Result<(), Status> {
        if Self::contains_indexed_rc(&self.bones, bone, index) {
            Err(Status::ErrorModelBoneAlreadyExists)
        } else {
            if index >= 0 && (index as usize) < self.bones.len() {
                self.bones.insert(index as usize, bone.clone());
                for bone in &self.bones[(index as usize) + 1..] {
                    bone.borrow_mut().base.index += 1;
                }
            } else {
                index = self.bones.len() as i32;
                self.bones.push(bone.clone());
            }
            bone.borrow_mut().base.index = index;
            Ok(())
        }
    }

    pub fn insert_label(
        &mut self,
        label: Rc<
            RefCell<ModelLabel<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>>,
        >,
        mut index: i32,
    ) {
        if self.labels.iter().find(|l| Rc::ptr_eq(&label, l)).is_some() {
            return;
        }
        if index >= 0 && (index as usize) < self.labels.len() {
            self.labels.insert(index as usize, label.clone());
            for label in &mut self.labels[(index as usize) + 1..] {
                label.borrow_mut().base.index += 1;
            }
        } else {
            index = self.labels.len() as i32;
            self.labels.push(label.clone());
        }
        self.labels
            .get_mut(index as usize)
            .map(|label| label.borrow_mut().base.index = index);
    }
}

fn mutable_model_object_apply_change_object_index(target: &mut i32, object_index: i32, delta: i32) {
    let dest_object_index = *target;
    if dest_object_index != NANOEM_MODEL_OBJECT_NOT_FOUND {
        if delta < 0 && dest_object_index == object_index {
            *target = NANOEM_MODEL_OBJECT_NOT_FOUND
        } else if dest_object_index >= object_index {
            *target = dest_object_index + delta
        }
    }
}

impl<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >
    Model<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >
{
    fn apply_change_all_object_indices(&mut self, vertex_index: i32, delta: i32) {
        for morph in &self.morphs {
            let mut morph_mut = morph.borrow_mut();
            match &mut morph_mut.u {
                ModelMorphU::VERTICES(vertices) => {
                    for morph_vertex in vertices {
                        mutable_model_object_apply_change_object_index(
                            &mut morph_vertex.vertex_index,
                            vertex_index,
                            delta,
                        )
                    }
                }
                ModelMorphU::UVS(uvs) => {
                    for morph_uv in uvs {
                        mutable_model_object_apply_change_object_index(
                            &mut morph_uv.vertex_index,
                            vertex_index,
                            delta,
                        )
                    }
                }
                _ => {}
            }
        }
        for soft_body in &mut self.soft_bodies {
            for anchor in &mut soft_body.borrow_mut().anchors {
                mutable_model_object_apply_change_object_index(
                    &mut anchor.vertex_index,
                    vertex_index,
                    delta,
                )
            }
        }
    }

    fn material_apply_change_all_object_indices(&mut self, material_index: i32, delta: i32) {
        for morph in &self.morphs {
            let mut morph_mut = morph.borrow_mut();
            match &mut morph_mut.u {
                ModelMorphU::MATERIALS(materials) => {
                    for morph_material in materials {
                        mutable_model_object_apply_change_object_index(
                            &mut morph_material.material_index,
                            material_index,
                            delta,
                        )
                    }
                }
                _ => {}
            }
        }
        for soft_body in &mut self.soft_bodies {
            mutable_model_object_apply_change_object_index(
                &mut soft_body.borrow_mut().material_index,
                material_index,
                delta,
            )
        }
    }

    fn bone_apply_change_all_object_indices(&mut self, bone_index: i32, delta: i32) {
        for vertex in &mut self.vertices {
            for vertex_bone_index in &mut vertex.borrow_mut().bone_indices {
                mutable_model_object_apply_change_object_index(
                    vertex_bone_index,
                    bone_index,
                    delta,
                );
            }
        }
        for constraint in &mut self.constraints {
            mutable_model_object_apply_change_object_index(
                &mut constraint.borrow_mut().effector_bone_index,
                bone_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut constraint.borrow_mut().target_bone_index,
                bone_index,
                delta,
            );
            for joint in &mut constraint.borrow_mut().joints {
                mutable_model_object_apply_change_object_index(
                    &mut joint.borrow_mut().bone_index,
                    bone_index,
                    delta,
                );
            }
        }
        for morph in &self.morphs {
            let mut morph = morph.borrow_mut();
            if let ModelMorphU::BONES(bones) = &mut morph.u {
                for bone in bones {
                    mutable_model_object_apply_change_object_index(
                        &mut bone.bone_index,
                        bone_index,
                        delta,
                    );
                }
            }
        }
        for bone in &self.bones {
            let mut bone = bone.borrow_mut();
            mutable_model_object_apply_change_object_index(
                &mut bone.parent_bone_index,
                bone_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut bone.parent_inherent_bone_index,
                bone_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut bone.effector_bone_index,
                bone_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut bone.target_bone_index,
                bone_index,
                delta,
            );
            if let Some(constraint) = &mut bone.constraint {
                let mut constraint = constraint.borrow_mut();
                mutable_model_object_apply_change_object_index(
                    &mut constraint.effector_bone_index,
                    bone_index,
                    delta,
                );
                for joint in &mut constraint.joints {
                    mutable_model_object_apply_change_object_index(
                        &mut joint.borrow_mut().bone_index,
                        bone_index,
                        delta,
                    );
                }
            }
        }
        for rigid_body in &mut self.rigid_bodies {
            mutable_model_object_apply_change_object_index(
                &mut rigid_body.borrow_mut().bone_index,
                bone_index,
                delta,
            );
        }
    }

    fn morph_apply_change_all_object_indices(&mut self, morph_index: i32, delta: i32) {
        for morph in &self.morphs {
            let mut morph = morph.borrow_mut();
            if let ModelMorphU::GROUPS(groups) = &mut morph.u {
                for group in groups {
                    mutable_model_object_apply_change_object_index(
                        &mut group.morph_index,
                        morph_index,
                        delta,
                    );
                }
            } else if let ModelMorphU::FLIPS(flips) = &mut morph.u {
                for flip in flips {
                    mutable_model_object_apply_change_object_index(
                        &mut flip.morph_index,
                        morph_index,
                        delta,
                    );
                }
            }
        }
    }

    fn rigid_body_apply_change_all_object_indices(&mut self, rigid_body_index: i32, delta: i32) {
        for morph in &self.morphs {
            let mut morph = morph.borrow_mut();
            if let ModelMorphU::IMPULSES(impulses) = &mut morph.u {
                for impulse in impulses {
                    mutable_model_object_apply_change_object_index(
                        &mut impulse.rigid_body_index,
                        rigid_body_index,
                        delta,
                    );
                }
            }
        }
        for joint in &mut self.joints {
            mutable_model_object_apply_change_object_index(
                &mut joint.borrow_mut().rigid_body_a_index,
                rigid_body_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut joint.borrow_mut().rigid_body_b_index,
                rigid_body_index,
                delta,
            );
        }
        for soft_body in &mut self.soft_bodies {
            for anchor in &mut soft_body.borrow_mut().anchors {
                mutable_model_object_apply_change_object_index(
                    &mut anchor.rigid_body_index,
                    rigid_body_index,
                    delta,
                );
            }
        }
    }

    fn texture_apply_change_all_object_indices(&mut self, texture_index: i32, delta: i32) {
        for material in &mut self.materials {
            mutable_model_object_apply_change_object_index(
                &mut material.borrow_mut().diffuse_texture_index,
                texture_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut material.borrow_mut().sphere_map_texture_index,
                texture_index,
                delta,
            );
            if !material.borrow().is_toon_shared {
                mutable_model_object_apply_change_object_index(
                    &mut material.borrow_mut().toon_texture_index,
                    texture_index,
                    delta,
                );
            }
        }
    }
}

struct ModelObject<T> {
    index: i32,
    user_data: Option<Rc<RefCell<T>>>,
}

impl<T> Default for ModelObject<T> {
    fn default() -> Self {
        Self {
            index: -1,
            user_data: None,
        }
    }
}

impl<T> Clone for ModelObject<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index.clone(),
            user_data: self.user_data.clone(),
        }
    }
}

impl<T> ModelObject<T> {
    fn get_user_data(&self) -> &Option<Rc<RefCell<T>>> {
        &self.user_data
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelVertexType> for i32 {
    fn from(value: ModelVertexType) -> Self {
        match value {
            ModelVertexType::UNKNOWN => -1,
            ModelVertexType::BDEF1 => 0,
            ModelVertexType::BDEF2 => 1,
            ModelVertexType::BDEF4 => 2,
            ModelVertexType::SDEF => 3,
            ModelVertexType::QDEF => 4,
        }
    }
}

impl Default for ModelVertexType {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

pub struct ModelVertex<T> {
    base: ModelObject<T>,
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

impl<T> ModelVertex<T> {
    fn parse_pmx<
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            T,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelVertex<T>, Status> {
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

    fn save_to_buffer<
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            T,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        buffer.write_f32_3_little_endian(self.origin)?;
        buffer.write_f32_3_little_endian(self.normal)?;
        buffer.write_f32_2_little_endian(self.uv)?;
        if parent_model.is_pmx() {
            for i in 0..parent_model.info.additional_uv_size {
                buffer.write_f32_4_little_endian(self.additional_uv[i as usize])?;
            }
            let size = parent_model.info.bone_index_size as usize;
            match self.typ {
                ModelVertexType::UNKNOWN => return Err(Status::ErrorModelVertexCorrupted),
                ModelVertexType::BDEF1 => {
                    buffer.write_byte(i32::from(self.typ) as u8)?;
                    buffer.write_integer(self.bone_indices[0], size)?;
                }
                ModelVertexType::BDEF2 => {
                    if self.bone_weights.0[0] > 0.9995f32 {
                        buffer.write_byte(i32::from(ModelVertexType::BDEF1) as u8)?;
                        buffer.write_integer(self.bone_indices[0], size)?;
                    } else if self.bone_weights.0[0] < 0.0005f32 {
                        buffer.write_byte(i32::from(ModelVertexType::BDEF1) as u8)?;
                        buffer.write_integer(self.bone_indices[1], size)?;
                    } else if self.bone_weights.0[1] > self.bone_weights.0[0] {
                        buffer.write_byte(i32::from(self.typ) as u8)?;
                        buffer.write_integer(self.bone_indices[1], size)?;
                        buffer.write_integer(self.bone_indices[0], size)?;
                        buffer.write_f32_little_endian(self.bone_weights.0[1])?;
                    } else {
                        buffer.write_byte(i32::from(self.typ) as u8)?;
                        buffer.write_integer(self.bone_indices[0], size)?;
                        buffer.write_integer(self.bone_indices[1], size)?;
                        buffer.write_f32_little_endian(self.bone_weights.0[0])?;
                    }
                }
                ModelVertexType::BDEF4 | ModelVertexType::QDEF => {
                    buffer.write_byte(i32::from(self.typ) as u8)?;
                    buffer.write_integer(self.bone_indices[0], size)?;
                    buffer.write_integer(self.bone_indices[1], size)?;
                    buffer.write_integer(self.bone_indices[2], size)?;
                    buffer.write_integer(self.bone_indices[3], size)?;
                    buffer.write_f32_4_little_endian(self.bone_weights)?;
                }
                ModelVertexType::SDEF => {
                    buffer.write_byte(i32::from(self.typ) as u8)?;
                    buffer.write_integer(self.bone_indices[0], size)?;
                    buffer.write_integer(self.bone_indices[1], size)?;
                    buffer.write_f32_little_endian(self.bone_weights.0[0])?;
                    buffer.write_f32_3_little_endian(self.sdef_c)?;
                    buffer.write_f32_3_little_endian(self.sdef_r0)?;
                    buffer.write_f32_3_little_endian(self.sdef_r1)?;
                }
            }
            buffer.write_f32_little_endian(self.edge_size)?;
        } else {
            let weight = if self.bone_weight_origin != 0 {
                self.bone_weight_origin
            } else {
                (self.bone_weights.0[0] * 100f32) as u8
            };
            buffer.write_i16_little_endian(self.bone_indices[0] as i16)?;
            buffer.write_i16_little_endian(self.bone_indices[1] as i16)?;
            buffer.write_byte(weight)?;
            buffer.write_byte(if self.edge_size != 0.0f32 { 0 } else { 1 })?;
        }
        Ok(())
    }

    pub fn get_user_data(&self) -> &Option<Rc<RefCell<T>>> {
        &self.base.user_data
    }

    pub fn set_user_data(&mut self, user_data: Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data)
    }

    pub fn get_origin(&self) -> [f32; 4] {
        self.origin.0
    }

    pub fn get_normal(&self) -> [f32; 4] {
        self.normal.0
    }

    pub fn get_tex_coord(&self) -> [f32; 4] {
        self.uv.0
    }

    pub fn get_edge_size(&self) -> f32 {
        self.edge_size
    }

    pub fn get_type(&self) -> ModelVertexType {
        self.typ
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn get_bone_indices(&self) -> [i32; 4] {
        self.bone_indices
    }

    pub fn get_bone_weights(&self) -> [f32; 4] {
        self.bone_weights.0
    }

    pub fn get_additional_uv(&self) -> [[f32; 4]; 4] {
        self.additional_uv.map(|uv| uv.0)
    }
}

#[derive(Default, Clone, Copy)]
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

impl From<ModelMaterialFlags> for u8 {
    fn from(value: ModelMaterialFlags) -> Self {
        (value.is_culling_disabled as u8) << 0
            | (value.is_casting_shadow_enabled as u8) << 1
            | (value.is_casting_shadow_map_enabled as u8) << 2
            | (value.is_shadow_map_enabled as u8) << 3
            | (value.is_edge_enabled as u8) << 4
            | (value.is_vertex_color_enabled as u8) << 5
            | (value.is_point_draw_enabled as u8) << 6
            | (value.is_line_draw_enabled as u8) << 7
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelMaterialSphereMapTextureType {
    Unknown = -1,
    TypeNone,
    TypeMultiply,
    TypeAdd,
    TypeSubTexture,
}

impl Default for ModelMaterialSphereMapTextureType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl From<i32> for ModelMaterialSphereMapTextureType {
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

impl From<ModelMaterialSphereMapTextureType> for u8 {
    fn from(v: ModelMaterialSphereMapTextureType) -> Self {
        match v {
            ModelMaterialSphereMapTextureType::Unknown => u8::MAX,
            ModelMaterialSphereMapTextureType::TypeNone => 0,
            ModelMaterialSphereMapTextureType::TypeMultiply => 1,
            ModelMaterialSphereMapTextureType::TypeAdd => 2,
            ModelMaterialSphereMapTextureType::TypeSubTexture => 3,
        }
    }
}

pub struct ModelMaterial<T> {
    base: ModelObject<T>,
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
    sphere_map_texture_type: ModelMaterialSphereMapTextureType,
    is_toon_shared: bool,
    num_vertex_indices: usize,
    flags: ModelMaterialFlags,
    sphere_map_texture_sph: Option<Rc<RefCell<ModelTexture>>>,
    sphere_map_texture_spa: Option<Rc<RefCell<ModelTexture>>>,
    diffuse_texture: Option<Rc<RefCell<ModelTexture>>>,
    clob: String,
}

impl<T> ModelMaterial<T> {
    pub fn parse_pmx<
        VertexDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            T,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelMaterial<T>, Status> {
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
            sphere_map_texture_type: ModelMaterialSphereMapTextureType::default(),
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
            ModelMaterialSphereMapTextureType::TypeNone
        } else {
            ModelMaterialSphereMapTextureType::from(sphere_map_texture_type_raw as i32)
        };
        match sphere_map_texture_type {
            ModelMaterialSphereMapTextureType::Unknown => {
                error = Some(Status::ErrorModelMaterialCorrupted)
            }
            ModelMaterialSphereMapTextureType::TypeNone
            | ModelMaterialSphereMapTextureType::TypeMultiply
            | ModelMaterialSphereMapTextureType::TypeAdd
            | ModelMaterialSphereMapTextureType::TypeSubTexture => {
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

    fn save_to_buffer<
        VertexDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            T,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_f32_3_little_endian(self.diffuse_color)?;
            buffer.write_f32_little_endian(self.diffuse_opacity)?;
            buffer.write_f32_3_little_endian(self.specular_color)?;
            buffer.write_f32_little_endian(self.specular_power)?;
            buffer.write_f32_3_little_endian(self.ambient_color)?;
            buffer.write_byte(self.flags.into())?;
            buffer.write_f32_3_little_endian(self.edge_color)?;
            buffer.write_f32_little_endian(self.edge_opacity)?;
            buffer.write_f32_little_endian(self.edge_size)?;
            let size = parent_model.info.texture_index_size as usize;
            buffer.write_integer(self.diffuse_texture_index, size)?;
            buffer.write_integer(self.sphere_map_texture_index, size)?;
            buffer.write_byte(self.sphere_map_texture_type.into())?;
            buffer.write_byte(self.is_toon_shared as u8)?;
            buffer.write_integer(self.toon_texture_index, size)?;
            buffer.write_string(&self.clob, codec_type)?;
            buffer.write_i32_little_endian(self.num_vertex_indices as i32)?;
        } else {
            buffer.write_f32_3_little_endian(self.diffuse_color)?;
            buffer.write_f32_little_endian(self.diffuse_opacity)?;
            buffer.write_f32_3_little_endian(self.specular_color)?;
            buffer.write_f32_little_endian(self.specular_power)?;
            buffer.write_f32_3_little_endian(self.ambient_color)?;
            buffer.write_byte(self.toon_texture_index as u8)?;
            buffer.write_byte(self.flags.is_edge_enabled as u8)?;
            buffer.write_i32_little_endian(self.num_vertex_indices as i32)?;
            let s = self
                .diffuse_texture
                .as_ref()
                .map(|t| t.borrow().path.clone());
            buffer.write_string(&s.unwrap_or("".to_string()), CodecType::Sjis)?;
        }
        Ok(())
    }

    pub fn get_name(&self, language_type: LanguageType) -> &str {
        match language_type {
            LanguageType::Unknown => "",
            LanguageType::English => &self.name_en,
            LanguageType::Japanese => &self.name_ja,
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn get_user_data(&self) -> &Option<Rc<RefCell<T>>> {
        self.base.get_user_data()
    }

    pub fn set_user_data(&mut self, user_data: Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data);
    }

    pub fn get_ambient_color(&self) -> [f32; 4] {
        self.ambient_color.0
    }

    pub fn get_diffuse_color(&self) -> [f32; 4] {
        self.diffuse_color.0
    }

    pub fn get_specular_color(&self) -> [f32; 4] {
        self.specular_color.0
    }

    pub fn get_diffuse_opacity(&self) -> f32 {
        self.diffuse_opacity
    }

    pub fn get_specular_power(&self) -> f32 {
        self.specular_power
    }

    pub fn get_edge_color(&self) -> [f32; 4] {
        self.edge_color.0
    }

    pub fn get_edge_opacity(&self) -> f32 {
        self.edge_opacity
    }

    pub fn get_edge_size(&self) -> f32 {
        self.edge_size
    }

    pub fn is_vertex_color_enabled(&self) -> bool {
        self.flags.is_vertex_color_enabled
    }

    pub fn is_line_draw_enabled(&self) -> bool {
        self.flags.is_line_draw_enabled
    }

    pub fn is_point_draw_enabled(&self) -> bool {
        self.flags.is_point_draw_enabled
    }

    pub fn is_culling_disabled(&self) -> bool {
        self.flags.is_culling_disabled
    }

    pub fn get_num_vertex_indices(&self) -> usize {
        self.num_vertex_indices
    }

    pub fn get_spheremap_texture_type(&self) -> ModelMaterialSphereMapTextureType {
        self.sphere_map_texture_type
    }

    pub fn get_diffuse_texture_object<
        'a,
        'b: 'a,
        VertexDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &'a self,
        parent_model: &'b Model<
            VertexDataType,
            T,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Option<&'a Rc<RefCell<ModelTexture>>> {
        let diffuse_texture_index = self.diffuse_texture_index;
        if diffuse_texture_index > -1 {
            parent_model.get_one_texture_object(diffuse_texture_index)
        } else {
            (&self.diffuse_texture).as_ref()
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

impl Default for ModelBoneType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl From<ModelBoneType> for u8 {
    fn from(v: ModelBoneType) -> Self {
        match v {
            ModelBoneType::Rotatable => 0,
            ModelBoneType::RotatableAndMovable => 1,
            ModelBoneType::ConstraintEffector => 2,
            ModelBoneType::Unknown => 3,
            ModelBoneType::ConstraintJoint => 4,
            ModelBoneType::InherentOrientationJoint => 5,
            ModelBoneType::ConstraintRoot => 6,
            ModelBoneType::Invisible => 7,
            ModelBoneType::FixedAxis => 8,
            ModelBoneType::InherentOrientationEffector => 9,
        }
    }
}

#[derive(Default, Clone, Copy)]
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

impl From<ModelBoneFlags> for u16 {
    fn from(v: ModelBoneFlags) -> Self {
        (v.has_destination_bone_index as u16)
            | (v.is_rotatable as u16) << 1
            | (v.is_movable as u16) << 2
            | (v.is_visible as u16) << 3
            | (v.is_user_handleable as u16) << 4
            | (v.has_constraint as u16) << 5
            | (v.has_local_inherent as u16) << 7
            | (v.has_inherent_orientation as u16) << 8
            | (v.has_inherent_orientation as u16) << 9
            | (v.has_fixed_axis as u16) << 10
            | (v.has_local_axes as u16) << 11
            | (v.is_affected_by_physics_simulation as u16) << 12
            | (v.has_external_parent_bone as u16) << 13
    }
}

#[test]
fn test_model_bone_flags_from_value() {
    let f = ModelBoneFlags::from_raw(33);
    assert_eq!(true, f.has_destination_bone_index);
    assert_eq!(true, f.has_constraint);
    assert_eq!(false, f.has_inherent_translation);
}

pub struct ModelBone<T, ConstraintDataType> {
    base: ModelObject<T>,
    name_ja: String,
    name_en: String,
    constraint: Option<Rc<RefCell<ModelConstraint<ConstraintDataType>>>>,
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

impl<T, ConstraintDataType> Default for ModelBone<T, ConstraintDataType> {
    fn default() -> Self {
        Self {
            base: Default::default(),
            name_ja: Default::default(),
            name_en: Default::default(),
            constraint: Default::default(),
            origin: Default::default(),
            destination_origin: Default::default(),
            fixed_axis: Default::default(),
            local_x_axis: Default::default(),
            local_z_axis: Default::default(),
            inherent_coefficient: Default::default(),
            parent_bone_index: Default::default(),
            parent_inherent_bone_index: Default::default(),
            effector_bone_index: Default::default(),
            target_bone_index: Default::default(),
            global_bone_index: Default::default(),
            stage_index: Default::default(),
            typ: Default::default(),
            flags: Default::default(),
        }
    }
}

impl<T, ConstraintDataType> ModelBone<T, ConstraintDataType> {
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

    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            T,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelBone<T, ConstraintDataType>, Status> {
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
            bone.constraint = Some(Rc::new(RefCell::new(ModelConstraint::parse_pmx(
                parent_model,
                buffer,
            )?)))
        }
        Ok(bone)
    }

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            T,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_f32_3_little_endian(self.origin)?;
            let size = parent_model.info.bone_index_size as usize;
            buffer.write_integer(self.parent_bone_index, size)?;
            buffer.write_i32_little_endian(self.stage_index)?;
            buffer.write_u16_little_endian(self.flags.into())?;
            if self.flags.has_destination_bone_index {
                buffer.write_integer(self.target_bone_index, size)?;
            } else {
                buffer.write_f32_3_little_endian(self.destination_origin)?;
            }
            if self.flags.has_inherent_translation || self.flags.has_inherent_orientation {
                buffer.write_integer(self.parent_inherent_bone_index, size)?;
                buffer.write_f32_little_endian(self.inherent_coefficient)?;
            }
            if self.flags.has_fixed_axis {
                buffer.write_f32_3_little_endian(self.fixed_axis)?;
            }
            if self.flags.has_local_axes {
                buffer.write_f32_3_little_endian(self.local_x_axis)?;
                buffer.write_f32_3_little_endian(self.local_z_axis)?;
            }
            if self.flags.has_external_parent_bone {
                buffer.write_i32_little_endian(self.global_bone_index)?;
            }
            if self.flags.has_constraint {
                if let Some(constraint) = &self.constraint {
                    constraint.borrow().save_to_buffer(buffer, parent_model)?;
                }
            }
        } else {
            buffer.write_string(&self.name_ja, CodecType::Sjis)?;
            buffer.write_i16_little_endian(self.parent_bone_index as i16)?;
            buffer.write_i16_little_endian(self.target_bone_index as i16)?;
            buffer.write_byte(self.typ.into())?;
            buffer.write_i16_little_endian(self.effector_bone_index as i16)?;
            buffer.write_f32_3_little_endian(self.origin)?;
        }
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja,
            LanguageType::English => &self.name_en,
        }
    }

    pub fn set_name(&mut self, value: &str, language: LanguageType) {
        match language {
            LanguageType::Unknown => {}
            LanguageType::Japanese => self.name_ja = value.to_owned(),
            LanguageType::English => self.name_en = value.to_owned(),
        }
    }

    pub fn get_constraint_object(
        &self,
    ) -> Option<&Rc<RefCell<ModelConstraint<ConstraintDataType>>>> {
        self.constraint.as_ref()
    }

    pub fn get_user_data(&self) -> Option<&Rc<RefCell<T>>> {
        self.base.user_data.as_ref()
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone());
    }

    pub fn set_visible(&mut self, value: bool) {
        self.flags.is_visible = value;
    }

    pub fn set_movable(&mut self, value: bool) {
        self.flags.is_movable = value;
    }

    pub fn set_rotatable(&mut self, value: bool) {
        self.flags.is_rotatable = value;
    }

    pub fn set_user_handleable(&mut self, value: bool) {
        self.flags.is_user_handleable = value;
    }

    pub fn set_constraint_enabled(&mut self, value: bool) {
        self.flags.has_constraint = value;
    }

    pub fn set_local_inherent_enabled(&mut self, value: bool) {
        self.flags.has_local_inherent = value;
    }

    pub fn set_inherent_translation_enabled(&mut self, value: bool) {
        self.flags.has_inherent_translation = value;
    }

    pub fn set_inherent_orientation_enabled(&mut self, value: bool) {
        self.flags.has_inherent_orientation = value;
    }

    pub fn set_fixed_axis_enabled(&mut self, value: bool) {
        self.flags.has_fixed_axis = value;
    }

    pub fn set_local_axes_enabled(&mut self, value: bool) {
        self.flags.has_local_axes = value;
    }

    pub fn set_affected_by_physics_simulation(&mut self, value: bool) {
        self.flags.is_affected_by_physics_simulation = value;
    }

    pub fn enable_extern_parent_bone(&mut self, value: bool) {
        self.flags.has_external_parent_bone = value;
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn has_inherent_orientation(&self) -> bool {
        self.flags.has_inherent_orientation
    }

    pub fn has_inherent_translation(&self) -> bool {
        self.flags.has_inherent_translation
    }

    pub fn get_parent_inherent_bone_index(&self) -> i32 {
        self.parent_inherent_bone_index
    }

    pub fn get_parent_bone_index(&self) -> i32 {
        self.parent_bone_index
    }
}

pub struct ModelConstraintJoint<T> {
    base: ModelObject<T>,
    bone_index: i32,
    has_angle_limit: bool,
    lower_limit: F128,
    upper_limit: F128,
}

impl<T> ModelConstraintJoint<T> {
    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let bone_index = parent_model
                .get_one_bone_object(self.bone_index)
                .map(|rc| rc.borrow().base.index)
                .unwrap_or(-1);
            buffer.write_integer(bone_index, parent_model.info.bone_index_size as usize)?;
            buffer.write_byte(self.has_angle_limit as u8)?;
            if self.has_angle_limit {
                buffer.write_f32_3_little_endian(self.lower_limit)?;
                buffer.write_f32_3_little_endian(self.upper_limit)?;
            }
        } else {
            buffer.write_i16_little_endian(self.bone_index as i16)?;
        }
        Ok(())
    }

    pub fn get_bone_index(&self) -> i32 {
        self.bone_index
    }
}

pub struct ModelConstraint<T> {
    base: ModelObject<T>,
    effector_bone_index: i32,
    target_bone_index: i32,
    num_iterations: i32,
    angle_limit: f32,
    joints: Vec<Rc<RefCell<ModelConstraintJoint<()>>>>,
}

impl<T> ModelConstraint<T> {
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            T,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelConstraint<T>, Status> {
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
            constraint.joints.push(Rc::new(RefCell::new(joint)));
        }
        Ok(constraint)
    }

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            T,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            buffer.write_integer(
                self.effector_bone_index,
                parent_model.info.bone_index_size as usize,
            )?;
            buffer.write_i32_little_endian(self.num_iterations)?;
            buffer.write_f32_little_endian(self.angle_limit)?;
            buffer.write_i32_little_endian(self.joints.len() as i32)?;
        } else {
            buffer.write_i16_little_endian(self.target_bone_index as i16)?;
            buffer.write_i16_little_endian(self.effector_bone_index as i16)?;
            buffer.write_byte(self.joints.len() as u8)?;
            buffer.write_u16_little_endian(self.num_iterations as u16)?;
            buffer.write_f32_little_endian(self.angle_limit)?;
        }
        for joint in &self.joints {
            joint.borrow().save_to_buffer(buffer, parent_model)?;
        }
        Ok(())
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone());
    }

    pub fn get_effector_bone_index(&self) -> i32 {
        self.effector_bone_index
    }

    pub fn get_target_bone_index(&self) -> i32 {
        self.target_bone_index
    }

    pub fn get_all_joint_objects(&self) -> &[Rc<RefCell<ModelConstraintJoint<()>>>] {
        &self.joints[..]
    }
}

pub struct ModelMorphBone {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        bone_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.bone_index, bone_index_size)?;
        buffer.write_f32_3_little_endian(self.translation)?;
        buffer.write_f32_4_little_endian(self.orientation)?;
        Ok(())
    }
}

pub struct ModelMorphGroup {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        morph_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.morph_index, morph_index_size)?;
        buffer.write_f32_little_endian(self.weight)?;
        Ok(())
    }
}

pub struct ModelMorphFlip {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        morph_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.morph_index, morph_index_size)?;
        buffer.write_f32_little_endian(self.weight)?;
        Ok(())
    }
}

pub struct ModelMorphImpulse {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        rigid_body_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.rigid_body_index, rigid_body_index_size)?;
        buffer.write_byte(self.is_local as u8)?;
        buffer.write_f32_3_little_endian(self.velocity)?;
        buffer.write_f32_3_little_endian(self.torque)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelMorphMaterialOperationType> for u8 {
    fn from(v: ModelMorphMaterialOperationType) -> Self {
        match v {
            ModelMorphMaterialOperationType::Unknown => Self::MAX,
            ModelMorphMaterialOperationType::Multiply => 0,
            ModelMorphMaterialOperationType::Add => 1,
        }
    }
}

pub struct ModelMorphMaterial {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        material_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.material_index, material_index_size)?;
        buffer.write_byte(self.operation.into())?;
        buffer.write_f32_3_little_endian(self.diffuse_color)?;
        buffer.write_f32_little_endian(self.diffuse_opacity)?;
        buffer.write_f32_3_little_endian(self.specular_color)?;
        buffer.write_f32_little_endian(self.specular_power)?;
        buffer.write_f32_3_little_endian(self.ambient_color)?;
        buffer.write_f32_3_little_endian(self.edge_color)?;
        buffer.write_f32_little_endian(self.edge_opacity)?;
        buffer.write_f32_little_endian(self.edge_size)?;
        buffer.write_f32_4_little_endian(self.diffuse_texture_blend)?;
        buffer.write_f32_4_little_endian(self.sphere_map_texture_blend)?;
        buffer.write_f32_4_little_endian(self.toon_texture_blend)?;
        Ok(())
    }
}

pub struct ModelMorphUv {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        vertex_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.vertex_index, vertex_index_size)?;
        buffer.write_f32_4_little_endian(self.position)?;
        Ok(())
    }
}

pub struct ModelMorphVertex {
    base: ModelObject<()>,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        vertex_index_size: usize,
    ) -> Result<(), Status> {
        buffer.write_integer(self.vertex_index, vertex_index_size)?;
        buffer.write_f32_3_little_endian(self.position)?;
        Ok(())
    }

    pub fn get_vertex_index(&self) -> i32 {
        self.vertex_index
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

impl ModelMorphU {
    pub fn len(&self) -> usize {
        match self {
            ModelMorphU::GROUPS(o) => o.len(),
            ModelMorphU::VERTICES(o) => o.len(),
            ModelMorphU::BONES(o) => o.len(),
            ModelMorphU::UVS(o) => o.len(),
            ModelMorphU::MATERIALS(o) => o.len(),
            ModelMorphU::FLIPS(o) => o.len(),
            ModelMorphU::IMPULSES(o) => o.len(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelMorphCategory> for u8 {
    fn from(v: ModelMorphCategory) -> Self {
        match v {
            ModelMorphCategory::Unknown => u8::MAX,
            ModelMorphCategory::Base => 0,
            ModelMorphCategory::Eyebrow => 1,
            ModelMorphCategory::Eye => 2,
            ModelMorphCategory::Lip => 3,
            ModelMorphCategory::Other => 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelMorphType> for u8 {
    fn from(v: ModelMorphType) -> Self {
        match v {
            ModelMorphType::Unknown => u8::MAX,
            ModelMorphType::Group => 0,
            ModelMorphType::Vertex => 1,
            ModelMorphType::Bone => 2,
            ModelMorphType::Texture => 3,
            ModelMorphType::Uva1 => 4,
            ModelMorphType::Uva2 => 5,
            ModelMorphType::Uva3 => 6,
            ModelMorphType::Uva4 => 7,
            ModelMorphType::Material => 8,
            ModelMorphType::Flip => 9,
            ModelMorphType::Impulse => 10,
        }
    }
}

pub struct ModelMorph<T> {
    base: ModelObject<T>,
    name_ja: String,
    name_en: String,
    typ: ModelMorphType,
    category: ModelMorphCategory,
    u: ModelMorphU,
}

impl<T> ModelMorph<T> {
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            T,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelMorph<T>, Status> {
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

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            T,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_byte(self.category.into())?;
            buffer.write_byte(self.typ.into())?;
            buffer.write_i32_little_endian(self.u.len() as i32)?;
            match self.typ {
                ModelMorphType::Unknown => return Err(Status::ErrorModelMorphCorrupted),
                ModelMorphType::Group => {
                    if let ModelMorphU::GROUPS(groups) = &self.u {
                        for group in groups {
                            group.save_to_buffer(
                                buffer,
                                parent_model.info.morph_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Vertex => {
                    if let ModelMorphU::VERTICES(vertices) = &self.u {
                        for vertex in vertices {
                            vertex.save_to_buffer(
                                buffer,
                                parent_model.info.vertex_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Bone => {
                    if let ModelMorphU::BONES(bones) = &self.u {
                        for bone in bones {
                            bone.save_to_buffer(
                                buffer,
                                parent_model.info.bone_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Texture
                | ModelMorphType::Uva1
                | ModelMorphType::Uva2
                | ModelMorphType::Uva3
                | ModelMorphType::Uva4 => {
                    if let ModelMorphU::UVS(uvs) = &self.u {
                        for uv in uvs {
                            uv.save_to_buffer(
                                buffer,
                                parent_model.info.vertex_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Material => {
                    if let ModelMorphU::MATERIALS(materials) = &self.u {
                        for material in materials {
                            material.save_to_buffer(
                                buffer,
                                parent_model.info.material_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Flip => {
                    if let ModelMorphU::FLIPS(flips) = &self.u {
                        for flip in flips {
                            flip.save_to_buffer(
                                buffer,
                                parent_model.info.morph_index_size as usize,
                            )?;
                        }
                    }
                }
                ModelMorphType::Impulse => {
                    if let ModelMorphU::IMPULSES(impulses) = &self.u {
                        for impulse in impulses {
                            impulse.save_to_buffer(
                                buffer,
                                parent_model.info.rigid_body_index_size as usize,
                            )?;
                        }
                    }
                }
            }
        } else {
            buffer.write_string(&self.name_ja, CodecType::Sjis)?;
            buffer.write_i32_little_endian(self.u.len() as i32)?;
            buffer.write_byte(self.category.into())?;
            match self.category {
                ModelMorphCategory::Base => {
                    if let ModelMorphU::VERTICES(vertices) = &self.u {
                        for vertex in vertices {
                            buffer.write_i32_little_endian(vertex.vertex_index)?;
                            buffer.write_f32_3_little_endian(vertex.position)?;
                        }
                    }
                }
                _ => {
                    if let ModelMorphU::VERTICES(vertices) = &self.u {
                        for vertex in vertices {
                            buffer.write_i32_little_endian(vertex.relative_index)?;
                            buffer.write_f32_3_little_endian(vertex.position)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja[..],
            LanguageType::English => &self.name_en[..],
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone())
    }

    pub fn get_type(&self) -> ModelMorphType {
        self.typ
    }

    pub fn get_category(&self) -> &ModelMorphCategory {
        &self.category
    }

    pub fn get_u(&self) -> &ModelMorphU {
        &self.u
    }
}

#[derive(Clone)]
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

impl From<ModelLabelItemType> for u8 {
    fn from(v: ModelLabelItemType) -> Self {
        match v {
            ModelLabelItemType::Unknown => Self::MAX,
            ModelLabelItemType::Bone => 0,
            ModelLabelItemType::Morph => 1,
        }
    }
}

/// TODO: need option really?
enum ModelLabelItemU<BoneDataType, ConstraintDataType, MorphDataType> {
    BONE(Weak<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>),
    MORPH(Weak<RefCell<ModelMorph<MorphDataType>>>),
}

impl<BoneDataType, ConstraintDataType, MorphDataType> Clone
    for ModelLabelItemU<BoneDataType, ConstraintDataType, MorphDataType>
{
    fn clone(&self) -> Self {
        match self {
            Self::BONE(arg0) => Self::BONE(arg0.clone()),
            Self::MORPH(arg0) => Self::MORPH(arg0.clone()),
        }
    }
}

pub struct ModelLabelItem<BoneDataType, ConstraintDataType, MorphDataType> {
    base: ModelObject<()>,
    typ: ModelLabelItemType,
    u: ModelLabelItemU<BoneDataType, ConstraintDataType, MorphDataType>,
}

impl<BoneDataType, ConstraintDataType, MorphDataType> Clone
    for ModelLabelItem<BoneDataType, ConstraintDataType, MorphDataType>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            typ: self.typ.clone(),
            u: self.u.clone(),
        }
    }
}

impl<BoneDataType, ConstraintDataType, MorphDataType>
    ModelLabelItem<BoneDataType, ConstraintDataType, MorphDataType>
{
    pub fn create_from_bone_object(
        bone: Rc<RefCell<ModelBone<BoneDataType, ConstraintDataType>>>,
    ) -> Self {
        Self {
            base: ModelObject::default(),
            typ: ModelLabelItemType::Bone,
            u: ModelLabelItemU::BONE(Rc::downgrade(&bone)),
        }
    }
}

pub struct ModelLabel<T, BoneDataType, ConstraintDataType, MorphDataType> {
    base: ModelObject<T>,
    name_ja: String,
    name_en: String,
    is_special: bool,
    items: Vec<ModelLabelItem<BoneDataType, ConstraintDataType, MorphDataType>>,
}

impl<T, BoneDataType, ConstraintDataType, MorphDataType> Default
    for ModelLabel<T, BoneDataType, ConstraintDataType, MorphDataType>
{
    fn default() -> Self {
        Self {
            base: Default::default(),
            name_ja: Default::default(),
            name_en: Default::default(),
            is_special: Default::default(),
            items: Default::default(),
        }
    }
}

impl<T, BoneDataType, ConstraintDataType, MorphDataType> Clone
    for ModelLabel<T, BoneDataType, ConstraintDataType, MorphDataType>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            name_ja: self.name_ja.clone(),
            name_en: self.name_en.clone(),
            is_special: self.is_special.clone(),
            items: self.items.clone(),
        }
    }
}

impl<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>
    ModelLabel<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>
{
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelLabel<LabelDataType, BoneDataType, ConstraintDataType, MorphDataType>, Status>
    {
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
                ModelLabelItemType::Bone => {
                    if let Some(bone) = parent_model
                        .get_one_bone_object(buffer.read_integer_nullable(bone_index_size)?)
                    {
                        label.items.push(ModelLabelItem {
                            base: ModelObject {
                                index: -1,
                                user_data: None,
                            },
                            typ: ModelLabelItemType::Bone,
                            u: ModelLabelItemU::BONE(Rc::downgrade(bone)),
                        })
                    } else {
                        return Err(Status::ErrorModelLabelItemNotFound);
                    }
                }
                ModelLabelItemType::Morph => {
                    if let Some(morph) = parent_model
                        .get_one_morph_object(buffer.read_integer_nullable(morph_index_size)?)
                    {
                        label.items.push(ModelLabelItem {
                            base: ModelObject {
                                index: -1,
                                user_data: None,
                            },
                            typ: ModelLabelItemType::Morph,
                            u: ModelLabelItemU::MORPH(Rc::downgrade(morph)),
                        })
                    } else {
                        return Err(Status::ErrorModelLabelItemNotFound);
                    }
                }
                ModelLabelItemType::Unknown => return Err(Status::ErrorModelLabelCorrupted),
            }
        }
        Ok(label)
    }

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_byte(self.is_special as u8)?;
            buffer.write_i32_little_endian(self.items.len() as i32)?;
            for item in &self.items {
                match item.typ {
                    ModelLabelItemType::Unknown => return Err(Status::ErrorModelLabelCorrupted),
                    ModelLabelItemType::Bone => {
                        if let ModelLabelItemU::BONE(bone) = &item.u {
                            if let Some(bone) = &bone.upgrade() {
                                buffer.write_byte(ModelLabelItemType::Bone.into())?;
                                buffer.write_integer(
                                    bone.borrow().base.index,
                                    parent_model.info.bone_index_size as usize,
                                )?;
                            } else {
                                return Err(Status::ErrorModelLabelCorrupted);
                            }
                        } else {
                            return Err(Status::ErrorModelLabelCorrupted);
                        }
                    }
                    ModelLabelItemType::Morph => {
                        if let ModelLabelItemU::MORPH(morph) = &item.u {
                            if let Some(morph) = &morph.upgrade() {
                                buffer.write_byte(ModelLabelItemType::Morph.into())?;
                                buffer.write_integer(
                                    morph.borrow().base.index,
                                    parent_model.info.morph_index_size as usize,
                                )?;
                            } else {
                                return Err(Status::ErrorModelLabelCorrupted);
                            }
                        } else {
                            return Err(Status::ErrorModelLabelCorrupted);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja[..],
            LanguageType::English => &self.name_en[..],
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn set_name(&mut self, value: &str, language: LanguageType) {
        match language {
            LanguageType::Unknown => (),
            LanguageType::Japanese => self.name_ja = value.to_string(),
            LanguageType::English => self.name_en = value.to_string(),
        }
    }

    pub fn set_special(&mut self, value: bool) {
        self.is_special = value;
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<LabelDataType>>) {
        self.base.user_data = Some(user_data.clone())
    }

    /// Not consider Already Existing
    pub fn insert_item_object(
        &mut self,
        item: &ModelLabelItem<BoneDataType, ConstraintDataType, MorphDataType>,
        mut index: i32,
    ) -> () {
        if index >= 0 && (index as usize) < self.items.len() {
            self.items.insert(index as usize, item.clone());
            for item in &mut self.items[(index as usize) + 1..] {
                item.base.index += 1;
            }
        } else {
            index = self.items.len() as i32;
            self.items.push(item.clone());
        }
        self.items
            .get_mut(index as usize)
            .map(|item| item.base.index = index);
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelRigidBodyShapeType> for u8 {
    fn from(v: ModelRigidBodyShapeType) -> Self {
        match v {
            ModelRigidBodyShapeType::Unknown => Self::MAX,
            ModelRigidBodyShapeType::Sphere => 0,
            ModelRigidBodyShapeType::Box => 1,
            ModelRigidBodyShapeType::Capsule => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl From<ModelRigidBodyTransformType> for u8 {
    fn from(v: ModelRigidBodyTransformType) -> Self {
        match v {
            ModelRigidBodyTransformType::Unknown => Self::MAX,
            ModelRigidBodyTransformType::FromBoneToSimulation => 0,
            ModelRigidBodyTransformType::FromSimulationToBone => 1,
            ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone => 2,
        }
    }
}

pub struct ModelRigidBody<T> {
    base: ModelObject<T>,
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

impl<T> ModelRigidBody<T> {
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            T,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelRigidBody<T>, Status> {
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

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            T,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_integer(self.bone_index, parent_model.info.bone_index_size as usize)?;
        } else {
            buffer.write_string(&self.name_ja, CodecType::Sjis)?;
            buffer.write_i16_little_endian(self.bone_index as i16)?;
        }
        buffer.write_byte(self.collision_group_id as u8)?;
        buffer.write_u16_little_endian(self.collision_mask as u16)?;
        buffer.write_byte(self.shape_type.into())?;
        buffer.write_f32_3_little_endian(self.size)?;
        buffer.write_f32_3_little_endian(self.origin)?;
        buffer.write_f32_3_little_endian(self.orientation)?;
        buffer.write_f32_little_endian(self.mass)?;
        buffer.write_f32_little_endian(self.linear_damping)?;
        buffer.write_f32_little_endian(self.angular_damping)?;
        buffer.write_f32_little_endian(self.restitution)?;
        buffer.write_f32_little_endian(self.friction)?;
        buffer.write_byte(self.transform_type.into())?;
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja,
            LanguageType::English => &self.name_en,
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn get_bone_index(&self) -> i32 {
        self.bone_index
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone());
    }

    pub fn get_transform_type(&self) -> ModelRigidBodyTransformType {
        self.transform_type
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelJointType> for u8 {
    fn from(v: ModelJointType) -> Self {
        match v {
            ModelJointType::Unknown => Self::MAX,
            ModelJointType::Generic6dofSpringConstraint => 0,
            ModelJointType::Generic6dofConstraint => 1,
            ModelJointType::Point2pointConstraint => 2,
            ModelJointType::ConeTwistConstraint => 3,
            ModelJointType::SliderConstraint => 4,
            ModelJointType::HingeConstraint => 5,
        }
    }
}

pub struct ModelJoint<T> {
    base: ModelObject<T>,
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

impl<T> ModelJoint<T> {
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            T,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelJoint<T>, Status> {
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

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            T,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let codec_type = parent_model.get_codec_type();
            let size = parent_model.info.rigid_body_index_size as usize;
            buffer.write_string(&self.name_ja, codec_type)?;
            buffer.write_string(&self.name_en, codec_type)?;
            buffer.write_byte(self.typ.into())?;
            buffer.write_integer(self.rigid_body_a_index, size)?;
            buffer.write_integer(self.rigid_body_b_index, size)?;
        } else {
            buffer.write_string(&self.name_ja, CodecType::Sjis)?;
            buffer.write_i32_little_endian(self.rigid_body_a_index)?;
            buffer.write_i32_little_endian(self.rigid_body_b_index)?;
        }
        buffer.write_f32_3_little_endian(self.origin)?;
        buffer.write_f32_3_little_endian(self.orientation)?;
        buffer.write_f32_3_little_endian(self.linear_lower_limit)?;
        buffer.write_f32_3_little_endian(self.linear_upper_limit)?;
        buffer.write_f32_3_little_endian(self.angular_lower_limit)?;
        buffer.write_f32_3_little_endian(self.angular_upper_limit)?;
        buffer.write_f32_3_little_endian(self.linear_stiffness)?;
        buffer.write_f32_3_little_endian(self.angular_stiffness)?;
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja,
            LanguageType::English => &self.name_en,
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone());
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelSoftBodyShapeType> for u8 {
    fn from(v: ModelSoftBodyShapeType) -> Self {
        match v {
            ModelSoftBodyShapeType::Unknown => Self::MAX,
            ModelSoftBodyShapeType::TriMesh => 0,
            ModelSoftBodyShapeType::Rope => 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

impl From<ModelSoftBodyAeroModelType> for i32 {
    fn from(v: ModelSoftBodyAeroModelType) -> Self {
        match v {
            ModelSoftBodyAeroModelType::Unknown => -1,
            ModelSoftBodyAeroModelType::VertexPoint => 0,
            ModelSoftBodyAeroModelType::VertexTwoSided => 1,
            ModelSoftBodyAeroModelType::VertexOneSided => 2,
            ModelSoftBodyAeroModelType::FaceTwoSided => 3,
            ModelSoftBodyAeroModelType::FaceOneSided => 4,
        }
    }
}

pub struct ModelSoftBodyAnchor {
    base: ModelObject<()>,
    rigid_body_index: i32,
    vertex_index: i32,
    is_near_enabled: bool,
}

pub struct ModelSoftBody<T> {
    base: ModelObject<T>,
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
    volume_conversation_coefficient: f32,
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

impl<T> ModelSoftBody<T> {
    fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            T,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelSoftBody<T>, Status> {
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
            volume_conversation_coefficient: buffer.read_f32_little_endian()?,
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

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            T,
        >,
    ) -> Result<(), Status> {
        let material_index_size = parent_model.info.material_index_size as usize;
        let rigid_body_index_size = parent_model.info.rigid_body_index_size as usize;
        let vertex_index_size = parent_model.info.vertex_index_size as usize;
        let codec_type = parent_model.get_codec_type();
        buffer.write_string(&self.name_ja, codec_type)?;
        buffer.write_string(&self.name_en, codec_type)?;
        buffer.write_byte(self.shape_type.into())?;
        buffer.write_integer(self.material_index, material_index_size)?;
        buffer.write_byte(self.collision_group_id as u8)?;
        buffer.write_u16_little_endian(self.collision_mask as u16)?;
        buffer.write_byte(self.flags)?;
        buffer.write_i32_little_endian(self.bending_constraints_distance)?;
        buffer.write_i32_little_endian(self.cluster_count)?;
        buffer.write_f32_little_endian(self.total_mass)?;
        buffer.write_f32_little_endian(self.collision_margin)?;
        buffer.write_i32_little_endian(self.aero_model.into())?;
        buffer.write_f32_little_endian(self.velocity_correction_factor)?;
        buffer.write_f32_little_endian(self.damping_coefficient)?;
        buffer.write_f32_little_endian(self.drag_coefficient)?;
        buffer.write_f32_little_endian(self.lift_coefficient)?;
        buffer.write_f32_little_endian(self.pressure_coefficient)?;
        buffer.write_f32_little_endian(self.volume_conversation_coefficient)?;
        buffer.write_f32_little_endian(self.dynamic_friction_coefficient)?;
        buffer.write_f32_little_endian(self.pose_matching_coefficient)?;
        buffer.write_f32_little_endian(self.rigid_contact_hardness)?;
        buffer.write_f32_little_endian(self.kinetic_contact_hardness)?;
        buffer.write_f32_little_endian(self.soft_contact_hardness)?;
        buffer.write_f32_little_endian(self.anchor_hardness)?;
        buffer.write_f32_little_endian(self.soft_vs_rigid_hardness)?;
        buffer.write_f32_little_endian(self.soft_vs_kinetic_hardness)?;
        buffer.write_f32_little_endian(self.soft_vs_soft_hardness)?;
        buffer.write_f32_little_endian(self.soft_vs_rigid_impulse_split)?;
        buffer.write_f32_little_endian(self.soft_vs_kinetic_impulse_split)?;
        buffer.write_f32_little_endian(self.soft_vs_soft_impulse_split)?;
        buffer.write_i32_little_endian(self.velocity_solver_iterations)?;
        buffer.write_i32_little_endian(self.positions_solver_iterations)?;
        buffer.write_i32_little_endian(self.drift_solver_iterations)?;
        buffer.write_i32_little_endian(self.cluster_solver_iterations)?;
        buffer.write_f32_little_endian(self.linear_stiffness_coefficient)?;
        buffer.write_f32_little_endian(self.angular_stiffness_coefficient)?;
        buffer.write_f32_little_endian(self.volume_stiffness_coefficient)?;
        buffer.write_i32_little_endian(self.anchors.len() as i32)?;
        for anchor in &self.anchors {
            buffer.write_integer(anchor.rigid_body_index, rigid_body_index_size)?;
            buffer.write_integer(anchor.vertex_index, vertex_index_size)?;
            buffer.write_byte(anchor.is_near_enabled as u8)?;
        }
        buffer.write_i32_little_endian(self.pinned_vertex_indices.len() as i32)?;
        for pinned_vertex_index in &self.pinned_vertex_indices {
            buffer.write_integer(pinned_vertex_index.clone() as i32, vertex_index_size)?;
        }
        Ok(())
    }

    pub fn get_name(&self, language: LanguageType) -> &str {
        match language {
            LanguageType::Unknown => "",
            LanguageType::Japanese => &self.name_ja,
            LanguageType::English => &self.name_en,
        }
    }

    pub fn get_index(&self) -> i32 {
        self.base.index
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<T>>) {
        self.base.user_data = Some(user_data.clone())
    }
}

pub struct ModelTexture {
    base: ModelObject<()>,
    path: String,
}

impl ModelTexture {
    pub fn parse_pmx<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
        buffer: &mut Buffer,
    ) -> Result<ModelTexture, Status> {
        Ok(ModelTexture {
            base: ModelObject {
                index: -1,
                user_data: None,
            },
            path: parent_model.get_string_pmx(buffer)?,
        })
    }

    fn save_to_buffer<
        VertexDataType,
        MaterialDataType,
        BoneDataType,
        ConstraintDataType,
        MorphDataType,
        LabelDataType,
        RigidBodyDataType,
        JointDataType,
        SoftBodyDataType,
    >(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model<
            VertexDataType,
            MaterialDataType,
            BoneDataType,
            ConstraintDataType,
            MorphDataType,
            LabelDataType,
            RigidBodyDataType,
            JointDataType,
            SoftBodyDataType,
        >,
    ) -> Result<(), Status> {
        let codec_type = parent_model.get_codec_type();
        buffer.write_string(&self.path, codec_type)
    }

    pub fn get_path(&self) -> &str {
        self.path.as_str()
    }
}

#[test]
fn test_read_pmx_resource() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut model: Model<(), (), (), (), (), (), (), (), ()> = Model {
        version: 0f32,
        info_length: 0,
        info: Info {
            codec_type: 0,
            additional_uv_size: 0,
            vertex_index_size: 0,
            texture_index_size: 0,
            material_index_size: 0,
            bone_index_size: 0,
            morph_index_size: 0,
            rigid_body_index_size: 0,
        },
        name_ja: String::default(),
        name_en: String::default(),
        comment_ja: String::default(),
        comment_en: String::default(),
        vertices: vec![],
        vertex_indices: vec![],
        materials: vec![],
        bones: vec![],
        ordered_bones: vec![],
        constraints: vec![],
        textures: vec![],
        morphs: vec![],
        labels: vec![],
        rigid_bodies: vec![],
        joints: vec![],
        soft_bodies: vec![],
        user_data: None,
    };

    let model_data = std::fs::read("test/example/Alicia/MMD/Alicia_solid.pmx")?;
    let mut buffer = Buffer::create(&model_data);
    match model.load_from_buffer(&mut buffer) {
        Ok(_) => println!("Parse PMX Success"),
        Err(e) => println!("Parse PMX with {:?}", &e),
    }
    Ok(())
}

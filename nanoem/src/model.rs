use std::{cell::RefCell, rc::Rc};

use crate::{
    common::{Buffer, LanguageType, MutableBuffer, Status, UserData, F128},
    utils::fourcc,
};

pub static NANOEM_MODEL_OBJECT_NOT_FOUND: i32 = -1;

#[derive(Debug, Clone, Copy)]
pub enum CodecType {
    Unknown(u8),
    Sjis,
    Utf8,
    Utf16Le,
}

impl Default for CodecType {
    fn default() -> Self {
        Self::Utf8
    }
}

impl CodecType {
    pub fn to_u8(self, version: &ModelFormatVersion) -> u8 {
        if version.is_pmx() {
            match self {
                CodecType::Unknown(c) => c,
                CodecType::Sjis => u8::MAX,
                CodecType::Utf8 => 1,
                CodecType::Utf16Le => 0,
            }
        } else {
            0
        }
    }

    pub fn from_u8(value: u8, version: &ModelFormatVersion) -> Self {
        if version.is_pmx() {
            match value {
                1 => CodecType::Utf8,
                0 => CodecType::Utf16Le,
                _ => CodecType::Unknown(value),
            }
        } else {
            CodecType::Sjis
        }
    }

    pub fn get_encoding(&self) -> &'static encoding_rs::Encoding {
        match self {
            CodecType::Unknown(_) => encoding_rs::UTF_8,
            CodecType::Sjis => encoding_rs::SHIFT_JIS,
            CodecType::Utf8 => encoding_rs::UTF_8,
            CodecType::Utf16Le => encoding_rs::UTF_16LE,
        }
    }

    fn get_string(&self, buffer: &mut Buffer) -> Result<String, Status> {
        let length = buffer.read_len()?;
        let src = buffer.read_buffer(length)?;
        let codec = self.get_encoding();
        // TODO: need bom removal or not?
        let (cow, encoding_used, had_errors) = codec.decode(src);
        if had_errors {
            return Err(Status::ErrorDecodeUnicodeStringFailed);
        }
        Ok(cow.into())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ModelInfo {
    pub codec_type: CodecType,
    pub additional_uv_size: u8,
    pub vertex_index_size: u8,
    pub texture_index_size: u8,
    pub material_index_size: u8,
    pub bone_index_size: u8,
    pub morph_index_size: u8,
    pub rigid_body_index_size: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelFormatVersion {
    Unknown(f32),
    Pmd1_0,
    Pmx2_0,
    Pmx2_1,
}

impl From<i32> for ModelFormatVersion {
    fn from(value: i32) -> Self {
        match value {
            10 => ModelFormatVersion::Pmd1_0,
            20 => ModelFormatVersion::Pmx2_0,
            21 => ModelFormatVersion::Pmx2_1,
            _ => ModelFormatVersion::Unknown(value as f32),
        }
    }
}

impl From<f32> for ModelFormatVersion {
    fn from(value: f32) -> Self {
        match (value * 10f32) as i32 {
            10 => ModelFormatVersion::Pmd1_0,
            20 => ModelFormatVersion::Pmx2_0,
            21 => ModelFormatVersion::Pmx2_1,
            _ => ModelFormatVersion::Unknown(value),
        }
    }
}

impl From<ModelFormatVersion> for f32 {
    fn from(v: ModelFormatVersion) -> Self {
        match v {
            ModelFormatVersion::Unknown(version) => version,
            ModelFormatVersion::Pmd1_0 => 1.0f32,
            ModelFormatVersion::Pmx2_0 => 2.0f32,
            ModelFormatVersion::Pmx2_1 => 2.1f32,
        }
    }
}

impl ModelFormatVersion {
    pub fn is_pmx(&self) -> bool {
        match self {
            ModelFormatVersion::Unknown(v) => v * 10f32 >= 20f32,
            ModelFormatVersion::Pmd1_0 => false,
            ModelFormatVersion::Pmx2_0 => true,
            ModelFormatVersion::Pmx2_1 => true,
        }
    }

    pub fn is_pmx21(&self) -> bool {
        match self {
            ModelFormatVersion::Unknown(v) => v * 10f32 >= 21f32,
            ModelFormatVersion::Pmd1_0 => false,
            ModelFormatVersion::Pmx2_0 => false,
            ModelFormatVersion::Pmx2_1 => true,
        }
    }
}

#[derive(Debug)]
pub struct Model {
    pub version: ModelFormatVersion,
    pub codec_type: CodecType,
    pub additional_uv_size: u8,
    pub name_ja: String,
    pub name_en: String,
    pub comment_ja: String,
    pub comment_en: String,
    pub vertices: Vec<ModelVertex>,
    pub vertex_indices: Vec<u32>,
    pub materials: Vec<ModelMaterial>,
    pub bones: Vec<ModelBone>,
    // pub ordered_bones: Vec<ModelBone>,
    pub constraints: Vec<ModelConstraint>,
    pub textures: Vec<ModelTexture>,
    pub morphs: Vec<ModelMorph>,
    pub labels: Vec<ModelLabel>,
    pub rigid_bodies: Vec<ModelRigidBody>,
    pub joints: Vec<ModelJoint>,
    pub soft_bodies: Vec<ModelSoftBody>,
}

impl Model {
    const PMX_SIGNATURE: &'static str = "PMX ";

    fn create_empty() -> Self {
        Self {
            version: todo!(),
            codec_type: todo!(),
            additional_uv_size: todo!(),
            name_ja: todo!(),
            name_en: todo!(),
            comment_ja: todo!(),
            comment_en: todo!(),
            vertices: todo!(),
            vertex_indices: todo!(),
            materials: todo!(),
            bones: todo!(),
            // ordered_bones: todo!(),
            constraints: todo!(),
            textures: todo!(),
            morphs: todo!(),
            labels: todo!(),
            rigid_bodies: todo!(),
            joints: todo!(),
            soft_bodies: todo!(),
        }
    }

    fn load_from_pmx(buffer: &mut Buffer) -> Result<Self, Status> {
        let signature = buffer.read_u32_little_endian()?;
        if signature == fourcc('P' as u8, 'M' as u8, 'X' as u8, ' ' as u8)
            || signature == fourcc('P' as u8, 'M' as u8, 'X' as u8, 0xA0u8)
        {
            let version = buffer.read_f32_little_endian()?.into();
            let info_length = buffer.read_byte()?;
            if info_length == 8u8 {
                let info = ModelInfo {
                    codec_type: CodecType::from_u8(buffer.read_byte()?, &version),
                    additional_uv_size: buffer.read_byte()?,
                    vertex_index_size: buffer.read_byte()?,
                    texture_index_size: buffer.read_byte()?,
                    material_index_size: buffer.read_byte()?,
                    bone_index_size: buffer.read_byte()?,
                    morph_index_size: buffer.read_byte()?,
                    rigid_body_index_size: buffer.read_byte()?,
                };
                let name_ja = info.codec_type.get_string(buffer)?;
                let name_en = info.codec_type.get_string(buffer)?;
                let comment_ja = info.codec_type.get_string(buffer)?;
                let comment_en = info.codec_type.get_string(buffer)?;
                let mut model = Self {
                    version,
                    codec_type: info.codec_type,
                    additional_uv_size: info.additional_uv_size,
                    name_ja,
                    name_en,
                    comment_ja,
                    comment_en,
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
                model.parse_pmx(buffer, &info)?;
                Ok(model)
            } else {
                Err(Status::ErrorPmxInfoCorrupted)
            }
        } else {
            Err(Status::ErrorInvalidSignature)
        }
    }

    fn get_string_pmx(&self, buffer: &mut Buffer, info: &ModelInfo) -> Result<String, Status> {
        info.codec_type.get_string(buffer)
    }

    fn parse_vertex_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_vertices = buffer.read_len()?;
        if num_vertices > 0 {
            self.vertices.clear();
            for i in 0..num_vertices {
                let vertex = ModelVertex::parse_pmx(buffer, &info, i)?;
                self.vertices.push(vertex);
            }
        }
        Ok(())
    }

    fn parse_vertex_index_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let vertex_index_size = info.vertex_index_size as usize;
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

    fn parse_texture_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_textures = buffer.read_len()?;
        if num_textures > 0 {
            self.textures.clear();
            for i in 0..num_textures {
                let texture = ModelTexture::parse_pmx(buffer, info, i)?;
                self.textures.push(texture);
            }
        }
        Ok(())
    }

    fn parse_material_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_materials = buffer.read_len()?;
        if num_materials > 0 {
            self.materials.clear();
            for i in 0..num_materials {
                let material = ModelMaterial::parse_pmx(buffer, info, i)?;
                self.materials.push(material);
            }
        }
        Ok(())
    }

    fn parse_bone_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_bones = buffer.read_len()?;
        if num_bones > 0 {
            self.bones.clear();
            for i in 0..num_bones {
                let bone = ModelBone::parse_pmx(buffer, info, i)?;
                self.bones.push(bone);
                // self.ordered_bones.push(self.bones[i].clone());
            }
        }
        Ok(())
    }

    fn parse_morph_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_morphs = buffer.read_len()?;
        if num_morphs > 0 {
            self.morphs.clear();
            for i in 0..num_morphs {
                let mut morph = ModelMorph::parse_pmx(buffer, info, i)?;
                self.morphs.push(morph);
            }
        }
        Ok(())
    }

    fn parse_label_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_labels = buffer.read_len()?;
        if num_labels > 0 {
            self.labels.clear();
            for i in 0..num_labels {
                let mut label = ModelLabel::parse_pmx(buffer, info, i)?;
                self.labels.push(label);
            }
        }
        Ok(())
    }

    fn parse_rigid_body_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_rigid_bodies = buffer.read_len()?;
        if num_rigid_bodies > 0 {
            self.rigid_bodies.clear();
            for i in 0..num_rigid_bodies {
                let mut rigid_body = ModelRigidBody::parse_pmx(buffer, info, i)?;
                self.rigid_bodies.push(rigid_body);
            }
        }
        Ok(())
    }

    fn parse_joint_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_joints = buffer.read_len()?;
        if num_joints > 0 {
            self.joints.clear();
            for i in 0..num_joints {
                let mut joint = ModelJoint::parse_pmx(buffer, info, i)?;
                self.joints.push(joint);
            }
        }
        Ok(())
    }

    fn parse_soft_body_block_pmx(
        &mut self,
        buffer: &mut Buffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let num_soft_bodies = buffer.read_len()?;
        if num_soft_bodies > 0 {
            self.soft_bodies.clear();
            for i in 0..num_soft_bodies {
                let mut soft_body = ModelSoftBody::parse_pmx(buffer, info, i)?;
                self.soft_bodies.push(soft_body);
            }
        }
        Ok(())
    }

    fn parse_pmx(&mut self, buffer: &mut Buffer, info: &ModelInfo) -> Result<(), Status> {
        self.parse_vertex_block_pmx(buffer, info)?;
        self.parse_vertex_index_block_pmx(buffer, info)?;
        self.parse_texture_block_pmx(buffer, info)?;
        self.parse_material_block_pmx(buffer, info)?;
        self.parse_bone_block_pmx(buffer, info)?;
        self.parse_morph_block_pmx(buffer, info)?;
        self.parse_label_block_pmx(buffer, info)?;
        self.parse_rigid_body_block_pmx(buffer, info)?;
        self.parse_joint_block_pmx(buffer, info)?;
        if self.version.is_pmx21() && !buffer.is_end() {
            self.parse_soft_body_block_pmx(buffer, info)?;
        }
        if buffer.is_end() {
            Ok(())
        } else {
            Err(Status::ErrorBufferNotEnd)
        }
    }

    pub fn load_from_buffer(buffer: &mut Buffer) -> Result<Self, Status> {
        let result = Self::load_from_pmx(buffer);
        if let Err(Status::ErrorInvalidSignature) = result {
            Err(Status::ErrorNoSupportForPMD)
        } else {
            result
        }
    }

    fn vertices_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.vertices.len() as i32)?;
        for vertex in &self.vertices {
            vertex.save_to_buffer(buffer, self.is_pmx(), info)?;
        }
        Ok(())
    }

    fn textures_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.textures.len() as i32)?;
        for texture in &self.textures {
            texture.save_to_buffer(buffer, info)?;
        }
        Ok(())
    }

    fn materials_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.materials.len() as i32)?;
        for material in &self.materials {
            material.save_to_buffer(buffer, self.is_pmx(), info)?;
        }
        Ok(())
    }

    fn bones_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_integer(self.bones.len() as i32, if self.is_pmx() { 4 } else { 2 })?;
        for bone in &self.bones {
            bone.save_to_buffer(buffer, self, info)?;
        }
        Ok(())
    }

    fn morphs_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_integer(self.morphs.len() as i32, if self.is_pmx() { 4 } else { 2 })?;
        for morph in &self.morphs {
            morph.save_to_buffer(buffer, self.is_pmx(), info)?;
        }
        Ok(())
    }

    fn labels_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if self.is_pmx() {
            buffer.write_integer(self.labels.len() as i32, 4)?;
            for label in &self.labels {
                label.save_to_buffer(buffer, self, info)?;
            }
            Ok(())
        } else {
            Err(Status::ErrorNoSupportForPMD)
        }
    }

    fn rigid_bodies_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.rigid_bodies.len() as i32)?;
        for rigid_body in &self.rigid_bodies {
            rigid_body.save_to_buffer(buffer, self.is_pmx(), info)?;
        }
        Ok(())
    }

    fn joints_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_i32_little_endian(self.joints.len() as i32)?;
        for joint in &self.joints {
            joint.save_to_buffer(buffer, self.is_pmx(), info)?;
        }
        Ok(())
    }

    fn soft_bodies_save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if self.is_pmx21() {
            buffer.write_i32_little_endian(self.soft_bodies.len() as i32)?;
            for soft_body in &self.soft_bodies {
                soft_body.save_to_buffer(buffer, self.is_pmx(), info)?;
            }
        }
        Ok(())
    }

    pub fn save_to_buffer_pmx(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        let encoding = self.get_codec_type().get_encoding();
        let info_length = 8;
        let info = ModelInfo {
            codec_type: self.codec_type,
            additional_uv_size: self.additional_uv_size,
            vertex_index_size: Self::get_vertex_index_size(self.vertices.len()),
            texture_index_size: Self::get_object_index_size(self.textures.len()),
            material_index_size: Self::get_object_index_size(self.materials.len()),
            bone_index_size: Self::get_object_index_size(self.bones.len()),
            morph_index_size: Self::get_object_index_size(self.morphs.len()),
            rigid_body_index_size: Self::get_object_index_size(self.rigid_bodies.len()),
        };
        buffer.write_byte_array(&Self::PMX_SIGNATURE.as_bytes()[..4])?;
        buffer.write_f32_little_endian(self.version.into())?;
        buffer.write_byte(info_length)?;
        buffer.write_byte(info.codec_type.to_u8(&self.version))?;
        buffer.write_byte(info.additional_uv_size)?;
        buffer.write_byte(info.vertex_index_size)?;
        buffer.write_byte(info.texture_index_size)?;
        buffer.write_byte(info.material_index_size)?;
        buffer.write_byte(info.bone_index_size)?;
        buffer.write_byte(info.morph_index_size)?;
        buffer.write_byte(info.rigid_body_index_size)?;
        buffer.write_string(&self.name_ja, encoding)?;
        buffer.write_string(&self.name_en, encoding)?;
        buffer.write_string(&self.comment_ja, encoding)?;
        buffer.write_string(&self.comment_en, encoding)?;
        self.vertices_save_to_buffer(buffer, &info)?;
        let vertex_index_size = info.vertex_index_size as usize;
        buffer.write_i32_little_endian(self.vertex_indices.len() as i32)?;
        for vertex_index in &self.vertex_indices {
            buffer.write_integer(vertex_index.clone() as i32, vertex_index_size)?;
        }
        self.textures_save_to_buffer(buffer, &info)?;
        self.materials_save_to_buffer(buffer, &info)?;
        self.bones_save_to_buffer(buffer, &info)?;
        self.morphs_save_to_buffer(buffer, &info)?;
        self.labels_save_to_buffer(buffer, &info)?;
        self.rigid_bodies_save_to_buffer(buffer, &info)?;
        self.joints_save_to_buffer(buffer, &info)?;
        self.soft_bodies_save_to_buffer(buffer, &info)?;
        Ok(())
    }

    pub fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        if self.is_pmx() {
            self.save_to_buffer_pmx(buffer)
        } else {
            Err(Status::ErrorNoSupportForPMD)
        }
    }

    pub fn get_format_type(&self) -> ModelFormatVersion {
        self.version
    }

    pub fn get_codec_type(&self) -> CodecType {
        self.codec_type
    }

    pub fn get_additional_uv_size(&self) -> usize {
        self.additional_uv_size.into()
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

    pub fn get_one_vertex_object(&self, index: i32) -> Option<&ModelVertex> {
        if index < 0 {
            None
        } else {
            self.vertices.get(index as usize)
        }
    }

    pub fn get_one_bone_object(&self, index: i32) -> Option<&ModelBone> {
        if index < 0 {
            None
        } else {
            self.bones.get(index as usize)
        }
    }

    pub fn get_one_morph_object(&self, index: i32) -> Option<&ModelMorph> {
        if index < 0 {
            None
        } else {
            self.morphs.get(index as usize)
        }
    }

    pub fn get_one_texture_object(&self, index: i32) -> Option<&ModelTexture> {
        if index < 0 {
            None
        } else {
            self.textures.get(index as usize)
        }
    }

    pub fn is_pmx(&self) -> bool {
        self.version.is_pmx()
    }

    pub fn is_pmx21(&self) -> bool {
        self.version.is_pmx21()
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
            self.additional_uv_size = value as u8;
        }
    }

    pub fn set_format_type(&mut self, value: ModelFormatVersion) {
        self.version = match value {
            ModelFormatVersion::Unknown(_) => self.version,
            ModelFormatVersion::Pmd1_0
            | ModelFormatVersion::Pmx2_0
            | ModelFormatVersion::Pmx2_1 => value,
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

    pub fn insert_bone<'a, 'b: 'a>(
        &'b mut self,
        mut bone: ModelBone,
        index: i32,
    ) -> Result<&'a ModelBone, Status> {
        let mut result_idx = 0usize;
        if index >= 0 && (index as usize) < self.bones.len() {
            result_idx = index as usize;
            bone.base.index = result_idx;
            self.bones.insert(result_idx, bone);
            for bone in &mut self.bones[(index as usize) + 1..] {
                bone.base.index += 1;
            }
        } else {
            result_idx = self.bones.len();
            bone.base.index = result_idx;
            self.bones.push(bone);
        }
        Ok(self.bones.get(result_idx).unwrap())
    }

    pub fn insert_label(&mut self, mut label: ModelLabel, mut index: i32) {
        if index >= 0 && (index as usize) < self.labels.len() {
            label.base.index = index as usize;
            self.labels.insert(index as usize, label);
            for label in &mut self.labels[(index as usize) + 1..] {
                label.base.index += 1;
            }
        } else {
            index = self.labels.len() as i32;
            label.base.index = index as usize;
            self.labels.push(label);
        }
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

impl Model {
    fn apply_change_all_object_indices(&mut self, vertex_index: i32, delta: i32) {
        for morph in &mut self.morphs {
            match &mut morph.morphs {
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
            for anchor in &mut soft_body.anchors {
                mutable_model_object_apply_change_object_index(
                    &mut anchor.vertex_index,
                    vertex_index,
                    delta,
                )
            }
        }
    }

    fn material_apply_change_all_object_indices(&mut self, material_index: i32, delta: i32) {
        for morph in &mut self.morphs {
            match &mut morph.morphs {
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
                &mut soft_body.material_index,
                material_index,
                delta,
            )
        }
    }

    fn bone_apply_change_all_object_indices(&mut self, bone_index: i32, delta: i32) {
        for vertex in &mut self.vertices {
            for vertex_bone_index in &mut vertex.bone_indices {
                mutable_model_object_apply_change_object_index(
                    vertex_bone_index,
                    bone_index,
                    delta,
                );
            }
        }
        for constraint in &mut self.constraints {
            mutable_model_object_apply_change_object_index(
                &mut constraint.effector_bone_index,
                bone_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut constraint.target_bone_index,
                bone_index,
                delta,
            );
            for joint in &mut constraint.joints {
                mutable_model_object_apply_change_object_index(
                    &mut joint.bone_index,
                    bone_index,
                    delta,
                );
            }
        }
        for morph in &mut self.morphs {
            if let ModelMorphU::BONES(bones) = &mut morph.morphs {
                for bone in bones {
                    mutable_model_object_apply_change_object_index(
                        &mut bone.bone_index,
                        bone_index,
                        delta,
                    );
                }
            }
        }
        for bone in &mut self.bones {
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
                mutable_model_object_apply_change_object_index(
                    &mut constraint.effector_bone_index,
                    bone_index,
                    delta,
                );
                for joint in &mut constraint.joints {
                    mutable_model_object_apply_change_object_index(
                        &mut joint.bone_index,
                        bone_index,
                        delta,
                    );
                }
            }
        }
        for rigid_body in &mut self.rigid_bodies {
            mutable_model_object_apply_change_object_index(
                &mut rigid_body.bone_index,
                bone_index,
                delta,
            );
        }
    }

    fn morph_apply_change_all_object_indices(&mut self, morph_index: i32, delta: i32) {
        for morph in &mut self.morphs {
            if let ModelMorphU::GROUPS(groups) = &mut morph.morphs {
                for group in groups {
                    mutable_model_object_apply_change_object_index(
                        &mut group.morph_index,
                        morph_index,
                        delta,
                    );
                }
            } else if let ModelMorphU::FLIPS(flips) = &mut morph.morphs {
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
        for morph in &mut self.morphs {
            if let ModelMorphU::IMPULSES(impulses) = &mut morph.morphs {
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
                &mut joint.rigid_body_a_index,
                rigid_body_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut joint.rigid_body_b_index,
                rigid_body_index,
                delta,
            );
        }
        for soft_body in &mut self.soft_bodies {
            for anchor in &mut soft_body.anchors {
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
                &mut material.diffuse_texture_index,
                texture_index,
                delta,
            );
            mutable_model_object_apply_change_object_index(
                &mut material.sphere_map_texture_index,
                texture_index,
                delta,
            );
            if !material.is_toon_shared {
                mutable_model_object_apply_change_object_index(
                    &mut material.toon_texture_index,
                    texture_index,
                    delta,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct ModelObject {
    pub index: usize,
}

impl Default for ModelObject {
    fn default() -> Self {
        Self { index: 0 }
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

#[derive(Debug, Clone)]
pub struct ModelVertex {
    pub base: ModelObject,
    pub origin: F128,
    pub normal: F128,
    pub uv: F128,
    pub additional_uv: [F128; 4],
    pub typ: ModelVertexType,
    pub num_bone_indices: usize,
    pub bone_indices: [i32; 4],
    pub num_bone_weights: usize,
    pub bone_weights: F128,
    pub sdef_c: F128,
    pub sdef_r0: F128,
    pub sdef_r1: F128,
    pub edge_size: f32,
    pub bone_weight_origin: u8,
}

impl ModelVertex {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelVertex, Status> {
        let mut vertex = ModelVertex {
            base: ModelObject { index },
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
        for i in 0..info.additional_uv_size {
            vertex.additional_uv[i as usize] = buffer.read_f32_4_little_endian()?;
        }
        let bone_index_size = info.bone_index_size;
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        buffer.write_f32_3_little_endian(self.origin)?;
        buffer.write_f32_3_little_endian(self.normal)?;
        buffer.write_f32_2_little_endian(self.uv)?;
        if is_pmx {
            for i in 0..info.additional_uv_size {
                buffer.write_f32_4_little_endian(self.additional_uv[i as usize])?;
            }
            let size = info.bone_index_size as usize;
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

    pub fn get_origin(&self) -> [f32; 4] {
        self.origin.0
    }

    pub fn get_normal(&self) -> [f32; 4] {
        self.normal.0
    }

    pub fn get_tex_coord(&self) -> [f32; 4] {
        self.uv.0
    }

    pub fn get_index(&self) -> usize {
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

#[derive(Debug, Default, Clone, Copy)]
pub struct ModelMaterialFlags {
    pub is_culling_disabled: bool,
    pub is_casting_shadow_enabled: bool,
    pub is_casting_shadow_map_enabled: bool,
    pub is_shadow_map_enabled: bool,
    pub is_edge_enabled: bool,
    pub is_vertex_color_enabled: bool,
    pub is_point_draw_enabled: bool,
    pub is_line_draw_enabled: bool,
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

#[derive(Debug, Clone, Copy)]
pub enum TextureResult<'a> {
    Texture(&'a ModelTexture),
    Index(i32),
}

#[derive(Debug, Clone)]
pub struct ModelMaterial {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub diffuse_color: F128,
    pub diffuse_opacity: f32,
    pub specular_power: f32,
    pub specular_color: F128,
    pub ambient_color: F128,
    pub edge_color: F128,
    pub edge_opacity: f32,
    pub edge_size: f32,
    pub diffuse_texture_index: i32,
    pub sphere_map_texture_index: i32,
    pub toon_texture_index: i32,
    pub sphere_map_texture_type: ModelMaterialSphereMapTextureType,
    pub is_toon_shared: bool,
    pub num_vertex_indices: usize,
    pub flags: ModelMaterialFlags,
    pub sphere_map_texture_sph: Option<ModelTexture>,
    pub sphere_map_texture_spa: Option<ModelTexture>,
    pub diffuse_texture: Option<ModelTexture>,
    pub clob: String,
}

impl ModelMaterial {
    pub fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelMaterial, Status> {
        let mut error: Option<Status> = None;
        let texture_index_size = info.texture_index_size;
        let mut material = ModelMaterial {
            base: ModelObject { index },
            name_ja: info.codec_type.get_string(buffer)?,
            name_en: info.codec_type.get_string(buffer)?,
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
        material.clob = info.codec_type.get_string(buffer)?;
        material.num_vertex_indices = buffer.read_i32_little_endian()? as usize;
        if let Some(err) = error {
            Err(err)
        } else {
            Ok(material)
        }
    }

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if is_pmx {
            let encoding = info.codec_type.get_encoding();
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_f32_3_little_endian(self.diffuse_color)?;
            buffer.write_f32_little_endian(self.diffuse_opacity)?;
            buffer.write_f32_3_little_endian(self.specular_color)?;
            buffer.write_f32_little_endian(self.specular_power)?;
            buffer.write_f32_3_little_endian(self.ambient_color)?;
            buffer.write_byte(self.flags.into())?;
            buffer.write_f32_3_little_endian(self.edge_color)?;
            buffer.write_f32_little_endian(self.edge_opacity)?;
            buffer.write_f32_little_endian(self.edge_size)?;
            let size = info.texture_index_size as usize;
            buffer.write_integer(self.diffuse_texture_index, size)?;
            buffer.write_integer(self.sphere_map_texture_index, size)?;
            buffer.write_byte(self.sphere_map_texture_type.into())?;
            buffer.write_byte(self.is_toon_shared as u8)?;
            buffer.write_integer(self.toon_texture_index, size)?;
            buffer.write_string(&self.clob, encoding)?;
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
            let s = self.diffuse_texture.as_ref().map(|t| t.path.clone());
            buffer.write_string(&s.unwrap_or("".to_string()), encoding_rs::SHIFT_JIS)?;
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

    pub fn get_index(&self) -> usize {
        self.base.index
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

    pub fn get_diffuse_texture_result(&self) -> TextureResult {
        let diffuse_texture_index = self.diffuse_texture_index;
        if diffuse_texture_index > -1 {
            TextureResult::Index(diffuse_texture_index)
        } else {
            if let Some(texture) = self.diffuse_texture.as_ref() {
                TextureResult::Texture(texture)
            } else {
                TextureResult::Index(-1)
            }
        }
    }

    pub fn get_diffuse_texture_object<'a, 'b: 'a, 'c: 'a>(&'c self, texture_lut: &'b Vec<ModelTexture>) -> Option<&'a ModelTexture> {
        match self.get_diffuse_texture_result() {
            TextureResult::Texture(texture) => {
                Some(texture)
            },
            TextureResult::Index(idx) => {
                if idx < 0 {
                    None
                } else {
                    texture_lut.get(idx as usize)
                }
            }
        }
    }

    pub fn get_sphere_map_texture_result(&self) -> TextureResult {
        let sphere_map_texture_index = self.sphere_map_texture_index;
        if sphere_map_texture_index > -1 {
            TextureResult::Index(sphere_map_texture_index)
        } else {
            if let Some(texture) = self.sphere_map_texture_spa.as_ref() {
                TextureResult::Texture(texture)
            } else if let Some(texture) = self.sphere_map_texture_sph.as_ref() {
                TextureResult::Texture(texture)
            } else {
                TextureResult::Index(-1)
            }
        }
    }

    pub fn get_sphere_map_texture_object<'a, 'b: 'a, 'c: 'a>(&'c self, texture_lut: &'b Vec<ModelTexture>) -> Option<&'a ModelTexture> {
        match self.get_sphere_map_texture_result() {
            TextureResult::Texture(texture) => {
                Some(texture)
            },
            TextureResult::Index(idx) => {
                if idx < 0 {
                    None
                } else {
                    texture_lut.get(idx as usize)
                }
            }
        }
    }

    pub fn get_toon_texture_result(&self) -> TextureResult {
        TextureResult::Index(self.toon_texture_index)
    }

    pub fn get_toon_texture_object<'a, 'b: 'a, 'c: 'a>(&'c self, texture_lut: &'b Vec<ModelTexture>) -> Option<&'a ModelTexture> {
        match self.get_toon_texture_result() {
            TextureResult::Texture(texture) => {
                Some(texture)
            },
            TextureResult::Index(idx) => {
                if idx < 0 {
                    None
                } else {
                    texture_lut.get(idx as usize)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ModelBoneType {
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

#[derive(Debug, Default, Clone, Copy)]
pub struct ModelBoneFlags {
    pub has_destination_bone_index: bool,
    pub is_rotatable: bool,
    pub is_movable: bool,
    pub is_visible: bool,
    pub is_user_handleable: bool,
    pub has_constraint: bool,
    pub has_local_inherent: bool,
    pub has_inherent_orientation: bool,
    pub has_inherent_translation: bool,
    pub has_fixed_axis: bool,
    pub has_local_axes: bool,
    pub is_affected_by_physics_simulation: bool,
    pub has_external_parent_bone: bool,
}

impl ModelBoneFlags {
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
            | (v.has_inherent_translation as u16) << 9
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

#[derive(Debug, Clone)]
pub struct ModelBone {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub constraint: Option<ModelConstraint>,
    pub origin: F128,
    pub destination_origin: F128,
    pub fixed_axis: F128,
    pub local_x_axis: F128,
    pub local_z_axis: F128,
    pub inherent_coefficient: f32,
    pub parent_bone_index: i32,
    pub parent_inherent_bone_index: i32,
    pub effector_bone_index: i32,
    pub target_bone_index: i32,
    pub global_bone_index: i32,
    pub stage_index: i32,
    pub typ: ModelBoneType,
    pub flags: ModelBoneFlags,
}

impl Default for ModelBone {
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

impl ModelBone {
    fn compare_pmx(&self, other: &Self) -> i32 {
        if self.flags.is_affected_by_physics_simulation
            == other.flags.is_affected_by_physics_simulation
        {
            if self.stage_index == other.stage_index {
                return self.base.index as i32 - other.base.index as i32;
            }
            return self.stage_index - other.stage_index;
        }
        return if other.flags.is_affected_by_physics_simulation {
            -1
        } else {
            1
        };
    }

    fn parse_pmx(buffer: &mut Buffer, info: &ModelInfo, index: usize) -> Result<ModelBone, Status> {
        let bone_index_size = info.bone_index_size;
        let mut bone = ModelBone {
            base: ModelObject { index },
            name_ja: info.codec_type.get_string(buffer)?,
            name_en: info.codec_type.get_string(buffer)?,
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
            bone.constraint = Some(ModelConstraint::parse_pmx(buffer, info, usize::MAX, index)?);
        }
        Ok(bone)
    }

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let encoding = parent_model.get_codec_type().get_encoding();
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_f32_3_little_endian(self.origin)?;
            let size = info.bone_index_size as usize;
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
                    constraint.save_to_buffer(buffer, parent_model, info)?;
                }
            }
        } else {
            buffer.write_string(&self.name_ja, encoding_rs::SHIFT_JIS)?;
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

    pub fn get_constraint_object(&self) -> Option<&ModelConstraint> {
        self.constraint.as_ref()
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

    pub fn get_index(&self) -> usize {
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

#[derive(Debug, Clone)]
pub struct ModelConstraintJoint {
    pub base: ModelObject,
    pub bone_index: i32,
    pub has_angle_limit: bool,
    pub lower_limit: F128,
    pub upper_limit: F128,
}

impl ModelConstraintJoint {
    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let bone_index = parent_model
                .bones
                .get(self.bone_index as usize)
                .map(|rc| rc.base.index as i32)
                .unwrap_or(-1);
            buffer.write_integer(bone_index, info.bone_index_size as usize)?;
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

#[derive(Debug, Clone)]
pub struct ModelConstraint {
    pub base: ModelObject,
    pub effector_bone_index: i32,
    pub target_bone_index: i32,
    pub num_iterations: i32,
    pub angle_limit: f32,
    pub joints: Vec<ModelConstraintJoint>,
}

impl ModelConstraint {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
        target_bone_index: usize,
    ) -> Result<ModelConstraint, Status> {
        let bone_index_size = info.bone_index_size as usize;
        let mut constraint = ModelConstraint {
            base: ModelObject { index },
            effector_bone_index: buffer.read_integer_nullable(bone_index_size)?,
            target_bone_index: target_bone_index as i32,
            num_iterations: buffer.read_i32_little_endian()?,
            angle_limit: buffer.read_f32_little_endian()?,
            joints: vec![],
        };
        let num_joints = buffer.read_len()?;
        for i in 0..num_joints {
            let mut joint = ModelConstraintJoint {
                base: ModelObject { index: i },
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            buffer.write_integer(self.effector_bone_index, info.bone_index_size as usize)?;
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
            joint.save_to_buffer(buffer, parent_model, info)?;
        }
        Ok(())
    }

    pub fn get_index(&self) -> usize {
        self.base.index
    }

    pub fn get_effector_bone_index(&self) -> i32 {
        self.effector_bone_index
    }

    pub fn get_target_bone_index(&self) -> i32 {
        self.target_bone_index
    }
}

#[derive(Debug, Clone)]
pub struct ModelMorphBone {
    pub base: ModelObject,
    pub bone_index: i32,
    pub translation: F128,
    pub orientation: F128,
}

impl ModelMorphBone {
    fn parse_pmx(
        buffer: &mut Buffer,
        bone_index_size: usize,
    ) -> Result<Vec<ModelMorphBone>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphBone {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphGroup {
    pub base: ModelObject,
    pub morph_index: i32,
    pub weight: f32,
}

impl ModelMorphGroup {
    fn parse_pmx(
        buffer: &mut Buffer,
        morph_index_size: usize,
    ) -> Result<Vec<ModelMorphGroup>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphGroup {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphFlip {
    pub base: ModelObject,
    pub morph_index: i32,
    pub weight: f32,
}

impl ModelMorphFlip {
    fn parse_pmx(
        buffer: &mut Buffer,
        morph_index_size: usize,
    ) -> Result<Vec<ModelMorphFlip>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphFlip {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphImpulse {
    pub base: ModelObject,
    pub rigid_body_index: i32,
    pub is_local: bool,
    pub velocity: F128,
    pub torque: F128,
}

impl ModelMorphImpulse {
    fn parse_pmx(
        buffer: &mut Buffer,
        rigid_body_index_size: usize,
    ) -> Result<Vec<ModelMorphImpulse>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphImpulse {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphMaterial {
    pub base: ModelObject,
    pub material_index: i32,
    pub operation: ModelMorphMaterialOperationType,
    pub diffuse_color: F128,
    pub diffuse_opacity: f32,
    pub specular_color: F128,
    pub specular_power: f32,
    pub ambient_color: F128,
    pub edge_color: F128,
    pub edge_opacity: f32,
    pub edge_size: f32,
    pub diffuse_texture_blend: F128,
    pub sphere_map_texture_blend: F128,
    pub toon_texture_blend: F128,
}

impl ModelMorphMaterial {
    fn parse_pmx(
        buffer: &mut Buffer,
        material_index_size: usize,
    ) -> Result<Vec<ModelMorphMaterial>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphMaterial {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphUv {
    pub base: ModelObject,
    pub vertex_index: i32,
    pub position: F128,
}

impl ModelMorphUv {
    fn parse_pmx(
        buffer: &mut Buffer,
        vertex_index_size: usize,
    ) -> Result<Vec<ModelMorphUv>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphUv {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
pub struct ModelMorphVertex {
    pub base: ModelObject,
    pub vertex_index: i32,
    pub relative_index: i32,
    pub position: F128,
}

impl ModelMorphVertex {
    fn parse_pmx(
        buffer: &mut Buffer,
        vertex_index_size: usize,
    ) -> Result<Vec<ModelMorphVertex>, Status> {
        let num_objects = buffer.read_len()?;
        let mut vec = vec![];
        for index in 0..num_objects {
            let item = ModelMorphVertex {
                base: ModelObject { index },
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

#[derive(Debug, Clone)]
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

// TODO: 整合type和u
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

#[derive(Debug, Clone)]
pub struct ModelMorph {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub typ: ModelMorphType,
    pub category: ModelMorphCategory,
    pub morphs: ModelMorphU,
}

impl ModelMorph {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelMorph, Status> {
        let codec_type = info.codec_type;
        let mut morph = ModelMorph {
            base: ModelObject { index },
            name_ja: codec_type.get_string(buffer)?,
            name_en: codec_type.get_string(buffer)?,
            category: ModelMorphCategory::from(buffer.read_byte()?),
            typ: ModelMorphType::from(buffer.read_byte()?),
            morphs: ModelMorphU::BONES(vec![]),
        };
        match morph.typ {
            ModelMorphType::Bone => {
                morph.morphs = ModelMorphU::BONES(ModelMorphBone::parse_pmx(
                    buffer,
                    info.bone_index_size as usize,
                )?)
            }
            ModelMorphType::Flip => {
                morph.morphs = ModelMorphU::FLIPS(ModelMorphFlip::parse_pmx(
                    buffer,
                    info.morph_index_size as usize,
                )?)
            }
            ModelMorphType::Group => {
                morph.morphs = ModelMorphU::GROUPS(ModelMorphGroup::parse_pmx(
                    buffer,
                    info.morph_index_size as usize,
                )?)
            }
            ModelMorphType::Impulse => {
                morph.morphs = ModelMorphU::IMPULSES(ModelMorphImpulse::parse_pmx(
                    buffer,
                    info.rigid_body_index_size as usize,
                )?)
            }
            ModelMorphType::Material => {
                morph.morphs = ModelMorphU::MATERIALS(ModelMorphMaterial::parse_pmx(
                    buffer,
                    info.material_index_size as usize,
                )?)
            }
            ModelMorphType::Texture
            | ModelMorphType::Uva1
            | ModelMorphType::Uva2
            | ModelMorphType::Uva3
            | ModelMorphType::Uva4 => {
                morph.morphs = ModelMorphU::UVS(ModelMorphUv::parse_pmx(
                    buffer,
                    info.vertex_index_size as usize,
                )?)
            }
            ModelMorphType::Vertex => {
                morph.morphs = ModelMorphU::VERTICES(ModelMorphVertex::parse_pmx(
                    buffer,
                    info.vertex_index_size as usize,
                )?)
            }
            ModelMorphType::Unknown => return Err(Status::ErrorModelMorphCorrupted),
        }
        Ok(morph)
    }

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if is_pmx {
            let encoding = info.codec_type.get_encoding();
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_byte(self.category.into())?;
            buffer.write_byte(self.typ.into())?;
            buffer.write_i32_little_endian(self.morphs.len() as i32)?;
            match self.typ {
                ModelMorphType::Unknown => return Err(Status::ErrorModelMorphCorrupted),
                ModelMorphType::Group => {
                    if let ModelMorphU::GROUPS(groups) = &self.morphs {
                        for group in groups {
                            group.save_to_buffer(buffer, info.morph_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Vertex => {
                    if let ModelMorphU::VERTICES(vertices) = &self.morphs {
                        for vertex in vertices {
                            vertex.save_to_buffer(buffer, info.vertex_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Bone => {
                    if let ModelMorphU::BONES(bones) = &self.morphs {
                        for bone in bones {
                            bone.save_to_buffer(buffer, info.bone_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Texture
                | ModelMorphType::Uva1
                | ModelMorphType::Uva2
                | ModelMorphType::Uva3
                | ModelMorphType::Uva4 => {
                    if let ModelMorphU::UVS(uvs) = &self.morphs {
                        for uv in uvs {
                            uv.save_to_buffer(buffer, info.vertex_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Material => {
                    if let ModelMorphU::MATERIALS(materials) = &self.morphs {
                        for material in materials {
                            material.save_to_buffer(buffer, info.material_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Flip => {
                    if let ModelMorphU::FLIPS(flips) = &self.morphs {
                        for flip in flips {
                            flip.save_to_buffer(buffer, info.morph_index_size as usize)?;
                        }
                    }
                }
                ModelMorphType::Impulse => {
                    if let ModelMorphU::IMPULSES(impulses) = &self.morphs {
                        for impulse in impulses {
                            impulse.save_to_buffer(buffer, info.rigid_body_index_size as usize)?;
                        }
                    }
                }
            }
        } else {
            buffer.write_string(&self.name_ja, encoding_rs::SHIFT_JIS)?;
            buffer.write_i32_little_endian(self.morphs.len() as i32)?;
            buffer.write_byte(self.category.into())?;
            match self.category {
                ModelMorphCategory::Base => {
                    if let ModelMorphU::VERTICES(vertices) = &self.morphs {
                        for vertex in vertices {
                            buffer.write_i32_little_endian(vertex.vertex_index)?;
                            buffer.write_f32_3_little_endian(vertex.position)?;
                        }
                    }
                }
                _ => {
                    if let ModelMorphU::VERTICES(vertices) = &self.morphs {
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

    pub fn get_index(&self) -> usize {
        self.base.index
    }

    pub fn get_type(&self) -> ModelMorphType {
        self.typ
    }

    pub fn get_category(&self) -> &ModelMorphCategory {
        &self.category
    }

    pub fn get_u(&self) -> &ModelMorphU {
        &self.morphs
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelLabelItemType {
    Unknown = -1,
    Bone,
    Morph,
}

impl Default for ModelLabelItemType {
    fn default() -> Self {
        Self::Unknown
    }
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

#[derive(Debug, Default, Clone, Copy)]
pub struct ModelLabelItem {
    base: ModelObject,
    typ: ModelLabelItemType,
    item_idx: i32,
}

impl ModelLabelItem {
    pub fn create_from_bone_object(bone: &ModelBone) -> Self {
        Self {
            base: ModelObject::default(),
            typ: ModelLabelItemType::Bone,
            item_idx: bone.base.index as i32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelLabel {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub is_special: bool,
    pub items: Vec<ModelLabelItem>,
}

impl ModelLabel {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelLabel, Status> {
        let codec_type = info.codec_type;
        let bone_index_size = info.bone_index_size as usize;
        let morph_index_size = info.morph_index_size as usize;
        let mut label = ModelLabel {
            base: ModelObject { index },
            name_ja: codec_type.get_string(buffer)?,
            name_en: codec_type.get_string(buffer)?,
            is_special: buffer.read_byte()? != 0,
            items: vec![],
        };
        let num_items = buffer.read_len()?;
        for index in 0..num_items {
            let item_type = ModelLabelItemType::from(buffer.read_byte()?);
            match item_type {
                // TODO: need verify idx valid
                ModelLabelItemType::Bone => label.items.push(ModelLabelItem {
                    base: ModelObject { index },
                    typ: ModelLabelItemType::Bone,
                    item_idx: buffer.read_integer_nullable(bone_index_size)?,
                }),
                ModelLabelItemType::Morph => label.items.push(ModelLabelItem {
                    base: ModelObject { index },
                    typ: ModelLabelItemType::Morph,
                    item_idx: buffer.read_integer_nullable(morph_index_size)?,
                }),
                ModelLabelItemType::Unknown => return Err(Status::ErrorModelLabelCorrupted),
            }
        }
        Ok(label)
    }

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        parent_model: &Model,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if parent_model.is_pmx() {
            let encoding = parent_model.get_codec_type().get_encoding();
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_byte(self.is_special as u8)?;
            buffer.write_i32_little_endian(self.items.len() as i32)?;
            for item in &self.items {
                match item.typ {
                    ModelLabelItemType::Unknown => return Err(Status::ErrorModelLabelCorrupted),
                    ModelLabelItemType::Bone => {
                        if let Some(bone) = parent_model.get_one_bone_object(item.item_idx) {
                            buffer.write_byte(ModelLabelItemType::Bone.into())?;
                            buffer.write_integer(
                                bone.base.index as i32,
                                info.bone_index_size as usize,
                            )?;
                        } else {
                            return Err(Status::ErrorModelLabelCorrupted);
                        }
                    }
                    ModelLabelItemType::Morph => {
                        if let Some(morph) = parent_model.get_one_morph_object(item.item_idx) {
                            buffer.write_byte(ModelLabelItemType::Morph.into())?;
                            buffer.write_integer(
                                morph.base.index as i32,
                                info.morph_index_size as usize,
                            )?;
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

    pub fn get_index(&self) -> usize {
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

    pub fn insert_item_object(&mut self, mut item: ModelLabelItem, mut index: i32) -> () {
        if index >= 0 && (index as usize) < self.items.len() {
            item.base.index = index as usize;
            self.items.insert(index as usize, item);
            for item in &mut self.items[(index as usize) + 1..] {
                item.base.index += 1;
            }
        } else {
            item.base.index = self.items.len();
            self.items.push(item.clone());
        }
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

#[derive(Debug, Clone)]
pub struct ModelRigidBody {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub bone_index: i32,
    pub collision_group_id: i32,
    pub collision_mask: i32,
    pub shape_type: ModelRigidBodyShapeType,
    pub size: F128,
    pub origin: F128,
    pub orientation: F128,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub restitution: f32,
    pub friction: f32,
    pub transform_type: ModelRigidBodyTransformType,
    pub is_bone_relative: bool,
}

impl ModelRigidBody {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelRigidBody, Status> {
        // TODO: not process Unknown for shpe_type and transform_type
        let mut rigid_body = ModelRigidBody {
            base: ModelObject { index },
            name_ja: info.codec_type.get_string(buffer)?,
            name_en: info.codec_type.get_string(buffer)?,
            bone_index: buffer.read_integer_nullable(info.bone_index_size as usize)?,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if is_pmx {
            let encoding = info.codec_type.get_encoding();
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_integer(self.bone_index, info.bone_index_size as usize)?;
        } else {
            buffer.write_string(&self.name_ja, encoding_rs::SHIFT_JIS)?;
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

    pub fn get_index(&self) -> usize {
        self.base.index
    }

    pub fn get_bone_index(&self) -> i32 {
        self.bone_index
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

#[derive(Debug, Clone)]
pub struct ModelJoint {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub rigid_body_a_index: i32,
    pub rigid_body_b_index: i32,
    pub typ: ModelJointType,
    pub origin: F128,
    pub orientation: F128,
    pub linear_lower_limit: F128,
    pub linear_upper_limit: F128,
    pub angular_lower_limit: F128,
    pub angular_upper_limit: F128,
    pub linear_stiffness: F128,
    pub angular_stiffness: F128,
}

impl ModelJoint {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelJoint, Status> {
        let rigid_body_index_size = info.rigid_body_index_size as usize;
        let mut joint = ModelJoint {
            base: ModelObject { index },
            name_ja: info.codec_type.get_string(buffer)?,
            name_en: info.codec_type.get_string(buffer)?,
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        if is_pmx {
            let encoding = info.codec_type.get_encoding();
            let size = info.rigid_body_index_size as usize;
            buffer.write_string(&self.name_ja, encoding)?;
            buffer.write_string(&self.name_en, encoding)?;
            buffer.write_byte(self.typ.into())?;
            buffer.write_integer(self.rigid_body_a_index, size)?;
            buffer.write_integer(self.rigid_body_b_index, size)?;
        } else {
            buffer.write_string(&self.name_ja, encoding_rs::SHIFT_JIS)?;
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

    pub fn get_index(&self) -> usize {
        self.base.index
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

#[derive(Debug, Clone)]
pub struct ModelSoftBodyAnchor {
    pub base: ModelObject,
    pub rigid_body_index: i32,
    pub vertex_index: i32,
    pub is_near_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ModelSoftBody {
    pub base: ModelObject,
    pub name_ja: String,
    pub name_en: String,
    pub shape_type: ModelSoftBodyShapeType,
    pub material_index: i32,
    pub collision_group_id: u8,
    pub collision_mask: u16,
    pub flags: u8,
    pub bending_constraints_distance: i32,
    pub cluster_count: i32,
    pub total_mass: f32,
    pub collision_margin: f32,
    pub aero_model: ModelSoftBodyAeroModelType,
    pub velocity_correction_factor: f32,
    pub damping_coefficient: f32,
    pub drag_coefficient: f32,
    pub lift_coefficient: f32,
    pub pressure_coefficient: f32,
    pub volume_conversation_coefficient: f32,
    pub dynamic_friction_coefficient: f32,
    pub pose_matching_coefficient: f32,
    pub rigid_contact_hardness: f32,
    pub kinetic_contact_hardness: f32,
    pub soft_contact_hardness: f32,
    pub anchor_hardness: f32,
    pub soft_vs_rigid_hardness: f32,
    pub soft_vs_kinetic_hardness: f32,
    pub soft_vs_soft_hardness: f32,
    pub soft_vs_rigid_impulse_split: f32,
    pub soft_vs_kinetic_impulse_split: f32,
    pub soft_vs_soft_impulse_split: f32,
    pub velocity_solver_iterations: i32,
    pub positions_solver_iterations: i32,
    pub drift_solver_iterations: i32,
    pub cluster_solver_iterations: i32,
    pub linear_stiffness_coefficient: f32,
    pub angular_stiffness_coefficient: f32,
    pub volume_stiffness_coefficient: f32,
    pub anchors: Vec<ModelSoftBodyAnchor>,
    pub pinned_vertex_indices: Vec<u32>,
}

impl ModelSoftBody {
    fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelSoftBody, Status> {
        let material_index_size = info.material_index_size as usize;
        let rigid_body_index_size = info.rigid_body_index_size as usize;
        let vertex_index_size = info.vertex_index_size as usize;
        let mut soft_body = ModelSoftBody {
            base: ModelObject { index },
            name_ja: info.codec_type.get_string(buffer)?,
            name_en: info.codec_type.get_string(buffer)?,
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
        for index in 0..num_anchors {
            soft_body.anchors.push(ModelSoftBodyAnchor {
                base: ModelObject { index },
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

    fn save_to_buffer(
        &self,
        buffer: &mut MutableBuffer,
        is_pmx: bool,
        info: &ModelInfo,
    ) -> Result<(), Status> {
        let material_index_size = info.material_index_size as usize;
        let rigid_body_index_size = info.rigid_body_index_size as usize;
        let vertex_index_size = info.vertex_index_size as usize;
        let encoding = info.codec_type.get_encoding();
        buffer.write_string(&self.name_ja, encoding)?;
        buffer.write_string(&self.name_en, encoding)?;
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
}

#[derive(Debug, Clone)]
pub struct ModelTexture {
    pub base: ModelObject,
    pub path: String,
}

impl ModelTexture {
    pub fn parse_pmx(
        buffer: &mut Buffer,
        info: &ModelInfo,
        index: usize,
    ) -> Result<ModelTexture, Status> {
        Ok(ModelTexture {
            base: ModelObject { index },
            path: info.codec_type.get_string(buffer)?,
        })
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer, info: &ModelInfo) -> Result<(), Status> {
        let encoding = info.codec_type.get_encoding();
        buffer.write_string(&self.path, encoding)
    }

    pub fn get_path(&self) -> &str {
        self.path.as_str()
    }
}

#[test]
fn test_read_pmx_resource() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model_data = std::fs::read("test/example/Alicia/MMD/Alicia_solid.pmx")?;
    let mut buffer = Buffer::create(&model_data);
    match Model::load_from_buffer(&mut buffer) {
        Ok(_) => println!("Parse PMX Success"),
        Err(e) => println!("Parse PMX with {:?}", &e),
    }
    Ok(())
}

#[test]
fn test_save_pmx_resource() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model_data = std::fs::read("test/example/Alicia/MMD/Alicia_solid.pmx")?;
    let mut buffer = Buffer::create(&model_data);
    match Model::load_from_buffer(&mut buffer) {
        Ok(model) => {
            let mut mutable_buffer = MutableBuffer::create().unwrap();
            match model.save_to_buffer(&mut mutable_buffer) {
                Ok(()) => {
                    if let Ok(mut buffer) = mutable_buffer.create_buffer_object() {
                        match Model::load_from_buffer(&mut buffer) {
                            Ok(model) => {
                                println!("Parse Recomposed PMX successfully");
                            }
                            Err(e) => println!("Parse Recomposed PMX with {:?}", &e),
                        }
                    }
                }
                Err(e) => println!("Save PMX with {:?}", &e),
            }
        }
        Err(e) => println!("Parse PMX with {:?}", &e),
    }
    Ok(())
}

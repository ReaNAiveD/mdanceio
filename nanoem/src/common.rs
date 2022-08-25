use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NanoemError {
    MallocFailed,                                 //< Failed to allocate memory
    ReallocFailed,                                //< Failed to allocate memory
    NullObject,                                   //< Null object is referred
    BufferEnd,                                    //< Buffer is end
    DecodeUnicodeStringFailed,                    //< Failed to decode unicode string
    EncodeUnicodeStringFailed,                    //< Failed to encode unicode string
    DecodeJisStringFailed,                        //< Costum, Failed to decode jis string
    BufferNotEnd,              //< Costum, Finish Loading but Buffer is not End
    InvalidSignature = 100,    //< Invalid signature
    ModelVertexCorrupted,      //< Vertex data is corrupted
    ModelFaceCorrupted,        //< Face (Indices) data is corrupted
    ModelMaterialCorrupted,    //< Material data is corrupted
    ModelBoneCorrupted,        //< Bone data is corrupted
    ModelConstraintCorrupted,  //< IK Constraint data is corrupted
    ModelTextureCorrupted,     //< Texture reference data is corrupted
    ModelMorphCorrupted,       //< Morph data is corrupted
    ModelLabelCorrupted,       //< Label data is corrupted
    ModelRigidBodyCorrupted,   //< Rigid body data is corrupted
    ModelJointCorrupted,       //< Joint data is corrupted
    PmdEnglishCorrupted,       //< PMD English data is corrupted
    PmxInfoCorrupted,          //< PMX Metadata is corrupted
    MotionTargetNameCorrupted, //< Vertex data is corrupted
    MotionBoneKeyframeCorrupted, //< The bone keyframe is corrupted
    MotionCameraKeyframeCorrupted, //< The camera keyframe data is corrupted
    MotionLightKeyframeCorrupted, //< The light keyframe data is corrupted
    MotionModelKeyframeCorrupted, //< The model keyframe data is corrupted
    MotionMorphKeyframeCorrupted, //< The morph keyframe data is corrupted
    MotionSelfShadowKeyframeCorrupted, //< Self Shadow keyframe data is corrupted
    ModelSoftBodyCorrupted,    //< Vertex data is corrupted
    MotionBoneKeyframeReference = 200, //< (unused)
    MotionBoneKeyframeAlreadyExists, //< The bone keyframe already exists
    MotionBoneKeyframeNotFound, //< The bone keyframe is not found
    MotionCameraKeyframeReference, //< (unused)
    MotionCameraKeyframeAlreadyExists, //< The camera keyframe already exists
    MotionCameraKeyframeNotFound, //< The camera keyframe is not found
    MotionLightKeyframeReference, //< (unused)
    MotionLightKeyframeAlreadyExists, //< The light keyframe already exists
    MotionLightKeyframeNotFound, //< The light keyframe is not found
    MotionModelKeyframeReference, //< (unused)
    MotionModelKeyframeAlreadyExists, //< The model keyframe already exists
    MotionModelKeyframeNotFound, //< The model keyframe is not found
    MotionMorphKeyframeReference, //< (unused)
    MotionMorphKeyframeAlreadyExists, //< The morph keyframe already exists
    MotionMorphKeyframeNotFound, //< The morph keyframe is not found
    MotionSelfShadowKeyframeReference, //< (unused)
    MotionSelfShadowKeyframeAlreadyExists, //< The self shadow keyframe already exists
    MotionSelfShadowKeyframeNotFound, //< The self shadow keyframe is not found
    MotionAccessoryKeyframeReference, //< (unused)
    MotionAccessoryKeyframeAlreadyExists, //< The accessory keyframe already exists
    MotionAccessoryKeyframeNotFound, //< The accessory keyframe is not found
    EffectParameterReference,  //< (unused)
    EffectParameterAlreadyExists, //< The effect parameter keyframe already exists
    EffectParameterNotFound,   //< The effect parameter keyframe is not found
    ModelConstraintStateReference, //< (unused)
    ModelConstraintStateAlreadyExists, //< IK keyframe already exists
    ModelConstraintStateNotFound, //< IK keyframe not found
    ModelBindingReference,     //< (unused)
    ModelBindingAlreadyExists, //< Outside parent model keyframe already exists
    ModelBindingNotFound,      //< Outside parent model keyframe is not found
    ModelVertexReference = 300, //< (unused)
    ModelVertexAlreadyExists,  //< Vertex data already exists
    ModelVertexNotFound,       //< Vertex data is not found
    ModelMaterialReference,    //< (unused)
    ModelMaterialAlreadyExists, //< Material data already exists
    ModelMaterialNotFound,     //< Material data is not found
    ModelBoneReference,        //< (unused)
    ModelBoneAlreadyExists,    //< Bone data already exists
    ModelBoneNotFound,         //< Bone data is not found
    ModelConstraintReference,  //< (unused)
    ModelConstraintAlreadyExists, //< IK constraint data already exists
    ModelConstraintNotFound,   //< IK constraint data is not found
    ModelConstraintJointNotFound, //< IK constraint joint bone data is not found
    ModelTextureReference,     //< (unused)
    ModelTextureAlreadyExists, //< Texture reference data already exists
    ModelTextureNotFound,      //< Texture reference data is not found
    ModelMorphReference,       //< (unused)
    ModelMorphAlreadyExists,   //< Morph data already exists
    ModelMorphNotFound,        //< Morph data is not found
    ModelMorphTypeMismatch,    //< Morph type is not matched
    ModelMorphBoneNotFound,    //< Morph bone data is not found
    ModelMorphFlipNotFound,    //< Morph flip data is not found
    ModelMorphGroupNotFound,   //< Morph group data is not found
    ModelMorphImpulseNotFound, //< Morph impulse data is not found
    ModelMorphMaterialNotFound, //< Morph material data is not found
    ModelMorphUvNotFound,      //< Morph UV data is not found
    ModelMorphVertexNotFound,  //< Morph vertex data is not found
    ModelLabelReference,       //< (unused)
    ModelLabelAlreadyExists,   //< Label data already exists
    ModelLabelNotFound,        //< Label data is not found
    ModelLabelItemNotFound,    //< Label item data is not found
    ModelRigidBodyReference,   //< (unused)
    ModelRigidBodyAlreadyExists, //< Rigid body data already exists
    ModelRigidBodyNotFound,    //< Rigid body data is not found
    ModelJointReference,       //< (unused)
    ModelJointAlreadyExists,   //< Joint data already exists
    ModelJointNotFound,        //< Joint data is not found
    ModelSoftBodyReference,    //< (unused)
    ModelSoftBodyAlreadyExists, //< Soft body data already exists
    ModelSoftBodyNotFound,     //< Soft body data is not found
    ModelSoftBodyAnchorAlreadyExists, //< Soft body anchor data already exists
    ModelSoftBodyAnchorNotFound, //< Soft body anchor data is not found
    ModelVersionIncompatible = 400, //< Moel version is incompatible
    DocumentAccessoryAlreadyExists = 1000, //< Accessory data already exists
    DocumentAccessoryNotFound, //< Accessory data is not foun
    DocumentAccessoryKeyframeAlreadyExists, //< The accessory keyframe already exists
    DocumentAccessoryKeyframeNotFound, //< The accessory keyframe is not found
    DocumentCameraKeyframeAlreadyExists, //< The camera keyframe already exists
    DocumentCameraKeyframeNotFound, //< The camera keyframe is not found
    DocumentGravityKeyframeAlreadyExists, //< The physics simulation keyframe already exists
    DocumentGravityKeyframeNotFound, //< The physics simulation keyframe is not found
    DocumentLightKeyframeAlreadyExists, //< The light keyframe already exists
    DocumentLightKeyframeNotFound, //< The light keyframe is not found
    DocumentModelAlreadyExists, //< Model data already exists
    DocumentModelNotFound,     //< Model data is not found
    DocumentModelBoneKeyframeAlreadyExists, //< The bone keyframe already exists
    DocumentModelBoneKeyframeNotFound, //< The bone keyframe is not found
    DocumentModelBoneStateAlreadyExists, //< Bone state already exists
    DocumentModelBoneStateNotFound, //< Bone state is not found
    DocumentModelConstraintStateAlreadyExists, //< IK constraint state already exists
    DocumentModelConstraintStateNotFound, //< IK constraint state is not found
    DocumentModelModelKeyframeAlreadyExists, //< The model keyframe already exists
    DocumentModelModelKeyframeNotFound, //< The model keyframe is not found
    DocumentModelMorphKeyframeAlreadyExists, //< The morph keyframe already exists
    DocumentModelMorphKeyframeNotFound, //< The morph keyframe is not found
    DocumentModelMorphStateAlreadyExists, //< Morph state already exists
    DocumentModelMorphStateNotFound, //< Morph state is not found
    DocumentModelOutsideParentAlreadyExists, //< Model outside parent already exists
    DocumentModelOutsideParentNotFound, //< Model outside parent is not found
    DocumentModelOutsideParentStateAlreadyExists, //< Model outside parent state already exists
    DocumentModelOutsideParentStateNotFound, //< Model outside parent state is not found
    DocumentSelfShadowKeyframeAlreadyExists, //< The self shadow keyframe already exists
    DocumentSelfShadowKeyframeNotFound, //< The self shadow keyframe is not found
    DocumentAccessoryCorrupted, //< Accessory data is corrupted
    DocumentAccessoryKeyframeCorrupted, //< The accessory keyframe is corrupted
    DocumentAccessoryOutsideParentCorrupted, //< The accessory outside parent is corrupted
    DocumentCameraCorrupted,   //< Camera data is corrupted
    DocumentCameraKeyframeCorrupted, //< The camera keyframe is corrupted
    DocumentGravityCorrupted,  //< Physics simulation data is corrupted
    DocumentGravityKeyframeCorrupted, //< The physics simulation keyframe is corrupted
    DocumentLightCorrupted,    //< Light data is corrupted
    DocumentLightKeyframeCorrupted, //< The light keyframe is corrupted
    DocumentModelCorrupted,    //< Model data is corrupted
    DocumentModelKeyframeCorrupted, //< The model keyframe is corrupted
    DocumentModelBoneKeyframeCorrupted, //< The bone keyframe is corrupted
    DocumentModelBoneStateCorrupted, //< The bone state is corrupted
    DocumentModelConstraintStateCorrupted, //< The IK constraint state is corrupted
    DocumentModelMorphKeyframeCorrupted, //< The morph keyframe is corrupted
    DocumentModelMorphStateCorrupted, //< The morph state is corrupted
    DocumentModelOutsideParentCorrupted, //< The model outside parent is corrupted
    DocumentSelfShadowCorrupted, //< Self shadow data is corrupted
    DocumentSelfShadowKeyframeCorrupted, //< The self shadow keyframe is corrupted
    NoSupportForPMD = 2000,    //< Not Supported PMD file
}

impl std::fmt::Display for NanoemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for NanoemError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageType {
    Unknown = -1,
    Japanese,
    English,
}

impl Default for LanguageType {
    fn default() -> Self {
        Self::Japanese
    }
}

impl LanguageType {
    pub fn all() -> &'static [Self] {
        &[Self::Japanese, Self::English]
    }
}

#[macro_export]
macro_rules! read_primitive {
    ($typ: ty, $read_typ:ident) => {
        pub fn $read_typ(&mut self) -> Result<$typ, NanoemError> {
            let typ_len = size_of::<$typ>();
            if self.can_read_len(typ_len) {
                let result = <$typ>::from_le_bytes(
                    self.data[self.offset..self.offset + typ_len]
                        .try_into()
                        .expect("Slice From Buffer(passed can_read_len) with incorrect length! "),
                );
                self.offset += typ_len;
                Ok(result)
            } else {
                Err(NanoemError::BufferEnd)
            }
        }
    };
}

pub struct Buffer<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Buffer<'a> {
    pub fn create(data: &[u8]) -> Buffer {
        Buffer {
            data,
            offset: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    fn can_read_len_internal(&self, len: usize) -> bool {
        self.len() >= self.offset && self.len() - self.offset >= len
    }

    pub fn can_read_len(&self, len: usize) -> bool {
        self.can_read_len_internal(len)
    }

    pub fn is_end(&self) -> bool {
        self.len() <= self.offset
    }

    pub fn skip(&mut self, skip: usize) -> Result<(), NanoemError> {
        if self.can_read_len(skip) {
            self.offset += skip;
            Ok(())
        } else {
            Err(NanoemError::BufferEnd)
        }
    }

    pub fn seek(&mut self, position: usize) -> Result<(), NanoemError> {
        if position < self.len() {
            self.offset = position;
            Ok(())
        } else {
            Err(NanoemError::BufferEnd)
        }
    }

    pub fn read_byte(&mut self) -> Result<u8, NanoemError> {
        if self.can_read_len(1) {
            let result = self.data[self.offset];
            self.offset += 1;
            Ok(result)
        } else {
            Err(NanoemError::BufferEnd)
        }
    }

    pub fn read_len(&mut self) -> Result<usize, NanoemError> {
        let len = self.read_u32_little_endian()? as usize;
        if self.can_read_len_internal(len) {
            Ok(len)
        } else {
            Ok(0)
        }
    }

    read_primitive!(u16, read_u16_little_endian);
    read_primitive!(i16, read_i16_little_endian);
    read_primitive!(u32, read_u32_little_endian);
    read_primitive!(i32, read_i32_little_endian);
    read_primitive!(f32, read_f32_little_endian);

    pub fn read_clamped_little_endian(&mut self) -> Result<f32, NanoemError> {
        let v = self.read_f32_little_endian()?;
        Ok(v.clamp(0.0f32, 1.0f32))
    }

    pub fn read_f32_3_little_endian(&mut self) -> Result<[f32; 4], NanoemError> {
        Ok([
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            0.0f32,
        ])
    }

    pub fn read_f32_4_little_endian(&mut self) -> Result<[f32; 4], NanoemError> {
        Ok([
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
        ])
    }

    pub fn read_integer(&mut self, size: usize) -> Result<i32, NanoemError> {
        Ok(match size {
            1 => self.read_byte()? as i32,
            2 => self.read_u16_little_endian()? as i32,
            4 => self.read_i32_little_endian()?,
            _ => Err(NanoemError::BufferEnd)?,
        })
    }

    pub fn read_integer_nullable(&mut self, size: usize) -> Result<i32, NanoemError> {
        let value = self.read_integer(size)?;
        if (size == 2 && value == 0xffff) || (size == 1 && value == 0xff) {
            return Ok(-1);
        }
        Ok(value)
    }

    pub fn read_buffer(&mut self, len: usize) -> Result<&[u8], NanoemError> {
        if self.can_read_len(len) {
            let result = &self.data[self.offset..self.offset + len];
            self.offset += len;
            Ok(result)
        } else {
            Err(NanoemError::BufferEnd)
        }
    }

    pub fn try_get_string_with_byte_len(&self, len: usize) -> &[u8] {
        let remaining_len = self.len() - self.offset;
        let read_len = usize::min(len, remaining_len);
        &self.data[self.offset..self.offset + read_len]
    }

    pub fn read_string_from_cp932(&mut self, max_capacity: usize) -> Result<String, NanoemError> {
        if self.can_read_len(max_capacity) {
            let mut src = self.read_buffer(max_capacity)?;
            if let Some(pos) = src.iter().position(|c| *c == 0u8) {
                src = src.split_at(pos).0;
            }
            let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(src);
            if had_errors {
                Err(NanoemError::DecodeJisStringFailed)
            } else {
                Ok(cow.into())
            }
        } else {
            Err(NanoemError::BufferEnd)
        }
    }
}

#[macro_export]
macro_rules! write_primitive {
    ($typ: ty, $write_typ:ident) => {
        pub fn $write_typ(&mut self, value: $typ) -> Result<(), NanoemError> {
            self.write_byte_array(&value.to_le_bytes())
        }
    };
}

pub struct MutableBuffer {
    offset: usize,
    data: Vec<u8>,
}

impl MutableBuffer {
    pub fn create() -> Result<MutableBuffer, NanoemError> {
        Self::create_with_reserved_size(2 << 12)
    }

    pub fn create_with_reserved_size(capacity: usize) -> Result<MutableBuffer, NanoemError> {
        let mut buffer = MutableBuffer {
            offset: 0usize,
            data: Vec::new(),
        };
        buffer.ensure_size(capacity)?;
        Ok(buffer)
    }

    pub fn ensure_size(&mut self, required: usize) -> Result<(), NanoemError> {
        if self.data.try_reserve(required).is_err() {
            self.offset = 0;
            Err(NanoemError::ReallocFailed)
        } else {
            Ok(())
        }
    }

    pub fn write_byte_array(&mut self, data: &[u8]) -> Result<(), NanoemError> {
        self.ensure_size(data.len())?;
        self.data.extend_from_slice(data);
        self.offset += data.len();
        Ok(())
    }

    pub fn write_byte(&mut self, value: u8) -> Result<(), NanoemError> {
        self.write_byte_array(&[value])
    }

    write_primitive!(u16, write_u16_little_endian);
    write_primitive!(i16, write_i16_little_endian);
    write_primitive!(u32, write_u32_little_endian);
    write_primitive!(i32, write_i32_little_endian);
    write_primitive!(f32, write_f32_little_endian);

    pub fn write_string(
        &mut self,
        value: &str,
        encoding: &'static encoding_rs::Encoding,
    ) -> Result<(), NanoemError> {
        if encoding == encoding_rs::UTF_16LE {
            let bytes = value.encode_utf16().collect::<Vec<_>>();
            self.write_u32_little_endian((bytes.len() * 2) as u32)?;
            for c in bytes {
                self.write_u16_little_endian(c)?;
            }
            Ok(())
        } else {
            let (bytes, _, success) = encoding.encode(value);
            if !success {
                self.write_u32_little_endian(0u32)?;
                Err(NanoemError::EncodeUnicodeStringFailed)
            } else {
                self.write_u32_little_endian(bytes.len() as u32)?;
                self.write_byte_array(&bytes)?;
                Ok(())
            }
        }
    }

    pub fn write_integer(&mut self, value: i32, size: usize) -> Result<(), NanoemError> {
        match size {
            1 => self.write_byte(value as u8),
            2 => self.write_u16_little_endian(value as u16),
            _ => self.write_i32_little_endian(value),
        }
    }

    pub fn write_f32_2_little_endian(&mut self, value: [f32; 4]) -> Result<(), NanoemError> {
        self.write_f32_little_endian(value[0])?;
        self.write_f32_little_endian(value[1])
    }

    pub fn write_f32_3_little_endian(&mut self, value: [f32; 4]) -> Result<(), NanoemError> {
        self.write_f32_2_little_endian(value)?;
        self.write_f32_little_endian(value[2])
    }

    pub fn write_f32_4_little_endian(&mut self, value: [f32; 4]) -> Result<(), NanoemError> {
        self.write_f32_3_little_endian(value)?;
        self.write_f32_little_endian(value[3])
    }

    // TODO: now clone the total data vec, change to some pointer copy
    pub fn create_buffer_object(&self) -> Result<Buffer, NanoemError> {
        Ok(Buffer::create(&self.data[..]))
    }

    pub fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }
}

#[derive(Debug, Clone, Copy)]
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

impl CodecType {
    pub fn get_encoding_object(&self) -> &'static encoding_rs::Encoding {
        match self {
            CodecType::Unknown => encoding_rs::UTF_8,
            CodecType::Sjis => encoding_rs::SHIFT_JIS,
            CodecType::Utf8 => encoding_rs::UTF_8,
            CodecType::Utf16 => encoding_rs::UTF_16LE,
        }
    }
}

#[test]
fn test_from_le_to_u16() {
    let data: [u8; 2] = [20, 16];
    assert_eq!(2, size_of::<u16>());
    assert_eq!(
        ((data[1] as u16) << 8) | data[0] as u16,
        u16::from_le_bytes(data)
    );
}

#[test]
fn test_buffer_read_primitive() {
    let mut buffer = Buffer::create(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    assert_eq!(Ok(1), buffer.read_byte());
    assert_eq!(Ok((3 << 8) | 2), buffer.read_u16_little_endian());
    println!("{}", buffer.read_i32_little_endian().expect("Expect Error"));
}

#[test]
fn test_u8_to_string_too_short() {
    let v = vec![b'a', 0u8, 0u8, b'b'];
    let (cow, _, had_errors) = encoding_rs::UTF_8.decode(&v[0..4]);
    assert!(!had_errors);
    println!("{}", &cow);
}

#[test]
fn test_u8_to_string_len0() {
    let v = vec![];
    let (cow, _, had_errors) = encoding_rs::UTF_8.decode(&v[0..0]);
    assert!(!had_errors);
    println!("{}", &cow);
}

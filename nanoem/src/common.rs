use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Unknown = -1, //< Unknown
    Success,
    ErrorMallocFailed,                                 //< Failed to allocate memory
    ErrorReallocFailed,                                //< Failed to allocate memory
    ErrorNullObject,                                   //< Null object is referred
    ErrorBufferEnd,                                    //< Buffer is end
    ErrorDecodeUnicodeStringFailed,                    //< Failed to decode unicode string
    ErrorEncodeUnicodeStringFailed,                    //< Failed to encode unicode string
    ErrorDecodeJisStringFailed,                        //< Costum, Failed to decode jis string
    ErrorBufferNotEnd,              //< Costum, Finish Loading but Buffer is not End
    ErrorIllegalBufferWriteOffset,  //< Costum, Illegal pos to write byte to mutable buffer
    ErrorInvalidSignature = 100,    //< Invalid signature
    ErrorModelVertexCorrupted,      //< Vertex data is corrupted
    ErrorModelFaceCorrupted,        //< Face (Indices) data is corrupted
    ErrorModelMaterialCorrupted,    //< Material data is corrupted
    ErrorModelBoneCorrupted,        //< Bone data is corrupted
    ErrorModelConstraintCorrupted,  //< IK Constraint data is corrupted
    ErrorModelTextureCorrupted,     //< Texture reference data is corrupted
    ErrorModelMorphCorrupted,       //< Morph data is corrupted
    ErrorModelLabelCorrupted,       //< Label data is corrupted
    ErrorModelRigidBodyCorrupted,   //< Rigid body data is corrupted
    ErrorModelJointCorrupted,       //< Joint data is corrupted
    ErrorPmdEnglishCorrupted,       //< PMD English data is corrupted
    ErrorPmxInfoCorruputed,         //< PMX Metadata is corrupted
    ErrorMotionTargetNameCorrupted, //< Vertex data is corrupted
    ErrorMotionBoneKeyframeCorrupted, //< The bone keyframe is corrupted
    ErrorMotionCameraKeyframeCorrupted, //< The camera keyframe data is corrupted
    ErrorMotionLightKeyframeCorrupted, //< The light keyframe data is corrupted
    ErrorMotionModelKeyframeCorrupted, //< The model keyframe data is corrupted
    ErrorMotionMorphKeyframeCorrupted, //< The morph keyframe data is corrupted
    ErrorMotionSelfShadowKeyframeCorrupted, //< Self Shadow keyframe data is corrupted
    ErrorModelSoftBodyCorrupted,    //< Vertex data is corrupted
    ErrorMotionBoneKeyframeReference = 200, //< (unused)
    ErrorMotionBoneKeyframeAlreadyExists, //< The bone keyframe already exists
    ErrorMotionBoneKeyframeNotFound, //< The bone keyframe is not found
    ErrorMotionCameraKeyframeReference, //< (unused)
    ErrorMotionCameraKeyframeAlreadyExists, //< The camera keyframe already exists
    ErrorMotionCameraKeyframeNotFound, //< The camera keyframe is not found
    ErrorMotionLightKeyframeReference, //< (unused)
    ErrorMotionLightKeyframeAlreadyExists, //< The light keyframe already exists
    ErrorMotionLightKeyframeNotFound, //< The light keyframe is not found
    ErrorMotionModelKeyframeReference, //< (unused)
    ErrorMotionModelKeyframeAlreadyExists, //< The model keyframe already exists
    ErrorMotionModelKeyframeNotFound, //< The model keyframe is not found
    ErrorMotionMorphKeyframeReference, //< (unused)
    ErrorMotionMorphKeyframeAlreadyExists, //< The morph keyframe already exists
    ErrorMotionMorphKeyframeNotFound, //< The morph keyframe is not found
    ErrorMotionSelfShadowKeyframeReference, //< (unused)
    ErrorMotionSelfShadowKeyframeAlreadyExists, //< The self shadow keyframe already exists
    ErrorMotionSelfShadowKeyframeNotFound, //< The self shadow keyframe is not found
    ErrorMotionAccessoryKeyframeReference, //< (unused)
    ErrorMotionAccessoryKeyframeAlreadyExists, //< The accessory keyframe already exists
    ErrorMotionAccessoryKeyframeNotFound, //< The accessory keyframe is not found
    ErrorEffectParameterReference,  //< (unused)
    ErrorEffectParameterAlreadyExists, //< The effect parameter keyframe already exists
    ErrorEffectParameterNotFound,   //< The effect parameter keyframe is not found
    ErrorModelConstraintStateReference, //< (unused)
    ErrorModelConstraintStateAlreadyExists, //< IK keyframe already exists
    ErrorModelConstraintStateNotFound, //< IK keyframe not found
    ErrorModelBindingReference,     //< (unused)
    ErrorModelBindingAlreadyExists, //< Outside parent model keyframe already exists
    ErrorModelBindingNotFound,      //< Outside parent model keyframe is not found
    ErrorModelVertexReference = 300, //< (unused)
    ErrorModelVertexAlreadyExists,  //< Vertex data already exists
    ErrorModelVertexNotFound,       //< Vertex data is not found
    ErrorModelMaterialReference,    //< (unused)
    ErrorModelMaterialAlreadyExists, //< Material data already exists
    ErrorModelMaterialNotFound,     //< Material data is not found
    ErrorModelBoneReference,        //< (unused)
    ErrorModelBoneAlreadyExists,    //< Bone data already exists
    ErrorModelBoneNotFound,         //< Bone data is not found
    ErrorModelConstraintReference,  //< (unused)
    ErrorModelConstraintAlreadyExists, //< IK constraint data already exists
    ErrorModelConstraintNotFound,   //< IK constraint data is not found
    ErrorModelConstraintJointNotFound, //< IK constraint joint bone data is not found
    ErrorModelTextureReference,     //< (unused)
    ErrorModelTextureAlreadyExists, //< Texture reference data already exists
    ErrorModelTextureNotFound,      //< Texture reference data is not found
    ErrorModelMorphReference,       //< (unused)
    ErrorModelMorphAlreadyExists,   //< Morph data already exists
    ErrorModelMorphNotFound,        //< Morph data is not found
    ErrorModelMorphTypeMismatch,    //< Morph type is not matched
    ErrorModelMorphBoneNotFound,    //< Morph bone data is not found
    ErrorModelMorphFlipNotFound,    //< Morph flip data is not found
    ErrorModelMorphGroupNotFound,   //< Morph group data is not found
    ErrorModelMorphImpulseNotFound, //< Morph impulse data is not found
    ErrorModelMorphMaterialNotFound, //< Morph material data is not found
    ErrorModelMorphUvNotFound,      //< Morph UV data is not found
    ErrorModelMorphVertexNotFound,  //< Morph vertex data is not found
    ErrorModelLabelReference,       //< (unused)
    ErrorModelLabelAlreadyExists,   //< Label data already exists
    ErrorModelLabelNotFound,        //< Label data is not found
    ErrorModelLabelItemNotFound,    //< Label item data is not found
    ErrorModelRigidBodyReference,   //< (unused)
    ErrorModelRigidBodyAlreadyExists, //< Rigid body data already exists
    ErrorModelRigidBodyNotFound,    //< Rigid body data is not found
    ErrorModelJointReference,       //< (unused)
    ErrorModelJointAlreadyExists,   //< Joint data already exists
    ErrorModelJointNotFound,        //< Joint data is not found
    ErrorModelSoftBodyReference,    //< (unused)
    ErrorModelSoftBodyAlreadyExists, //< Soft body data already exists
    ErrorModelSoftBodyNotFound,     //< Soft body data is not found
    ErrorModelSoftBodyAnchorAlreadyExists, //< Soft body anchor data already exists
    ErrorModelSoftBodyAnchorNotFound, //< Soft body anchor data is not found
    ErrorModelVersionIncompatible = 400, //< Moel version is incompatible
    ErrorDocumentAccessoryAlreadyExists = 1000, //< Accessory data already exists
    ErrorDocumentAccessoryNotFound, //< Accessory data is not foun
    ErrorDocumentAccessoryKeyframeAlreadyExists, //< The accessory keyframe already exists
    ErrorDocumentAccessoryKeyframeNotFound, //< The accessory keyframe is not found
    ErrorDocumentCameraKeyframeAlreadyExists, //< The camera keyframe already exists
    ErrorDocumentCameraKeyframeNotFound, //< The camera keyframe is not found
    ErrorDocumentGravityKeyframeAlreadyExists, //< The physics simulation keyframe already exists
    ErrorDocumentGravityKeyframeNotFound, //< The physics simulation keyframe is not found
    ErrorDocumentLightKeyframeAlreadyExists, //< The light keyframe already exists
    ErrorDocumentLightKeyframeNotFound, //< The light keyframe is not found
    ErrorDocumentModelAlreadyExists, //< Model data already exists
    ErrorDocumentModelNotFound,     //< Model data is not found
    ErrorDocumentModelBoneKeyframeAlreadyExists, //< The bone keyframe already exists
    ErrorDocumentModelBoneKeyframeNotFound, //< The bone keyframe is not found
    ErrorDocumentModelBoneStateAlreadyExists, //< Bone state already exists
    ErrorDocumentModelBoneStateNotFound, //< Bone state is not found
    ErrorDocumentModelConstraintStateAlreadyExists, //< IK constraint state already exists
    ErrorDocumentModelConstraintStateNotFound, //< IK constraint state is not found
    ErrorDocumentModelModelKeyframeAlreadyExists, //< The model keyframe already exists
    ErrorDocumentModelModelKeyframeNotFound, //< The model keyframe is not found
    ErrorDocumentModelMorphKeyframeAlreadyExists, //< The morph keyframe already exists
    ErrorDocumentModelMorphKeyframeNotFound, //< The morph keyframe is not found
    ErrorDocumentModelMorphStateAlreadyExists, //< Morph state already exists
    ErrorDocumentModelMorphStateNotFound, //< Morph state is not found
    ErrorDocumentModelOutsideParentAlreadyExists, //< Model outside parent already exists
    ErrorDocumentModelOutsideParentNotFound, //< Model outside parent is not found
    ErrorDocumentModelOutsideParentStateAlreadyExists, //< Model outside parent state already exists
    ErrorDocumentModelOutsideParentStateNotFound, //< Model outside parent state is not found
    ErrorDocumentSelfShadowKeyframeAlreadyExists, //< The self shadow keyframe already exists
    ErrorDocumentSelfShadowKeyframeNotFound, //< The self shadow keyframe is not found
    ErrorDocumentAccessoryCorrupted, //< Accessory data is corrupted
    ErrorDocumentAccessoryKeyframeCorrupted, //< The accessory keyframe is corrupted
    ErrorDocumentAccessoryOutsideParentCorrupted, //< The accessory outside parent is corrupted
    ErrorDocumentCameraCorrupted,   //< Camera data is corrupted
    ErrorDocumentCameraKeyframeCorrupted, //< The camera keyframe is corrupted
    ErrorDocumentGravityCorrupted,  //< Physics simulation data is corrupted
    ErrorDocumentGravityKeyframeCorrupted, //< The physics simulation keyframe is corrupted
    ErrorDocumentLightCorrupted,    //< Light data is corrupted
    ErrorDocumentLightKeyframeCorrupted, //< The light keyframe is corrupted
    ErrorDocumentModelCorrupted,    //< Model data is corrupted
    ErrorDocumentModelKeyframeCorrupted, //< The model keyframe is corrupted
    ErrorDocumentModelBoneKeyframeCorrupted, //< The bone keyframe is corrupted
    ErrorDocumentModelBoneStateCorrupted, //< The bone state is corrupted
    ErrorDocumentModelConstraintStateCorrupted, //< The IK constraint state is corrupted
    ErrorDocumentModelMorphKeyframeCorrupted, //< The morph keyframe is corrupted
    ErrorDocumentModelMorphStateCorrupted, //< The morph state is corrupted
    ErrorDocumentModelOutsideParentCorrupted, //< The model outside parent is corrupted
    ErrorDocumentSelfShadowCorrupted, //< Self shadow data is corrupted
    ErrorDocumentSelfShadowKeyframeCorrupted, //< The self shadow keyframe is corrupted
    ErrorNoSupportForPMD = 2000,    //< Not Supported PMD file
}

pub enum LanguageType {
    Unknown = -1,
    Japanese,
    English,
}

enum UserDataDestroyCallback {}

pub struct UserData {
    destroy: UserDataDestroyCallback,
}

struct F128Components {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Default, Clone, Copy)]
#[repr(align(16))]
pub struct F128(pub [f32; 4]);

#[macro_export]
macro_rules! read_primitive {
    ($typ: ty, $read_typ:ident) => {
        pub fn $read_typ(&mut self) -> Result<$typ, Status> {
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
                Err(Status::ErrorBufferEnd)
            }
        }
    };
}

pub struct Buffer {
    data: Vec<u8>,
    idx: usize,
    offset: usize,
}

impl Buffer {
    pub fn create(data: Vec<u8>) -> Buffer {
        Buffer {
            data,
            idx: 0,
            offset: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
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

    pub fn skip(&mut self, skip: usize) -> Result<(), Status> {
        if self.can_read_len(skip) {
            self.offset += skip;
            Ok(())
        } else {
            Err(Status::ErrorBufferEnd)
        }
    }

    pub fn seek(&mut self, position: usize) -> Result<(), Status> {
        if position < self.len() {
            self.offset = position;
            Ok(())
        } else {
            Err(Status::ErrorBufferEnd)
        }
    }

    pub fn read_byte(&mut self) -> Result<u8, Status> {
        if self.can_read_len(1) {
            let result = self.data[self.offset];
            self.offset += 1;
            Ok(result)
        } else {
            Err(Status::ErrorBufferEnd)
        }
    }

    pub fn read_len(&mut self) -> Result<usize, Status> {
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

    pub fn read_clamped_little_endian(&mut self) -> Result<f32, Status> {
        let v = self.read_f32_little_endian()?;
        Ok(v.clamp(0.0f32, 1.0f32))
    }

    pub fn read_f32_3_little_endian(&mut self) -> Result<F128, Status> {
        return Ok(F128([
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            0.0f32,
        ]));
    }

    pub fn read_f32_4_little_endian(&mut self) -> Result<F128, Status> {
        return Ok(F128([
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
            self.read_f32_little_endian()?,
        ]));
    }

    pub fn read_integer(&mut self, size: usize) -> Result<i32, Status> {
        Ok(match size {
            1 => self.read_byte()? as i32,
            2 => self.read_u16_little_endian()? as i32,
            4 => self.read_i32_little_endian()?,
            _ => Err(Status::ErrorBufferEnd)?,
        })
    }

    pub fn read_integer_nullable(&mut self, size: usize) -> Result<i32, Status> {
        let value = self.read_integer(size)?;
        if (size == 2 && value == 0xffff) || (size == 1 && value == 0xff) {
            return Ok(-1);
        }
        Ok(value)
    }

    pub fn read_buffer(&mut self, len: usize) -> Result<&[u8], Status> {
        if self.can_read_len(len) {
            let result = &self.data[self.offset..self.offset + len];
            self.offset += len;
            Ok(result)
        } else {
            Err(Status::ErrorBufferEnd)
        }
    }

    pub fn try_get_string_with_byte_len(&self, len: usize) -> (&[u8], usize) {
        let remaining_len = self.len() - self.offset;
        let read_len = usize::min(len, remaining_len);
        let str_raw = &self.data[self.offset..self.offset + read_len];
        (str_raw, read_len)
        // let (cow, _, had_errors)  = encoding_rs::UTF_8.decode(str_raw);
        // if had_errors {
        //     return Err(Status::ErrorDecodeUnicodeStringFailed);
        // }
        // Ok(cow.into())
    }

    pub fn read_string_from_cp932(&mut self, max_capacity: usize) -> Result<String, Status> {
        if self.can_read_len(max_capacity) {
            let mut src = self.read_buffer(max_capacity)?;
            if let Some(pos) = src.iter().position(|c| *c == 0u8) {
                src = src.split_at(pos).0;
            }
            let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(src);
            if had_errors {
                Err(Status::ErrorDecodeJisStringFailed)
            } else {
                Ok(cow.into())
            }
        } else {
            Err(Status::ErrorBufferEnd)
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
    let mut buffer = Buffer::create(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    assert_eq!(Ok(1), buffer.read_byte());
    assert_eq!(Ok((3 << 8) | 2), buffer.read_u16_little_endian());
    println!("{}", buffer.read_i32_little_endian().expect("Expect Error"));
}

#[test]
fn test_u8_to_string_too_short() {
    let mut v = vec!['a' as u8, 0u8, 0u8, 'b' as u8];
    let (cow, encoding_used, had_errors) = encoding_rs::UTF_8.decode(&v[0..4]);
    assert_eq!(false, had_errors);
    println!("{}", &cow);
}

#[test]
fn test_u8_to_string_len0() {
    let mut v = vec![];
    let (cow, encoding_used, had_errors) = encoding_rs::UTF_8.decode(&v[0..0]);
    assert_eq!(false, had_errors);
    println!("{}", &cow);
}

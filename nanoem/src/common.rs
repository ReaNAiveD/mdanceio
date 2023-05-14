use std::mem::size_of;

use crate::utils::u8_slice_get_string;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NanoemError {
    ReallocFailed,
    BufferEnd,
    BufferNotEnd,
    DecodeStringFailed {
        data: Vec<u8>,
        parsed: String,
        encoding: &'static str,
    },
    EncodeStringFailed(String),
    InvalidSignature,
    ModelVertexCorrupted,
    ModelFaceCorrupted,
    ModelMaterialCorrupted,
    ModelBoneCorrupted,
    ModelConstraintCorrupted,
    ModelTextureCorrupted,
    ModelMorphCorrupted,
    ModelLabelCorrupted,
    ModelRigidBodyCorrupted,
    ModelJointCorrupted,
    ModelSoftBodyCorrupted,
    PmxInfoCorrupted,
    KeyframeAlreadyExists {
        track_name: String,
        frame_index: u32,
    },
    NoSupportForPMD,
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
        Buffer { data, offset: 0 }
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

    pub fn read_string_from_cp932(
        &mut self,
        max_capacity: usize,
        errors: &mut Vec<NanoemError>,
    ) -> Result<String, NanoemError> {
        if self.can_read_len(max_capacity) {
            let mut src = self.read_buffer(max_capacity)?;
            if let Some(pos) = src.iter().position(|c| *c == 0u8) {
                src = src.split_at(pos).0;
            }
            Ok(u8_slice_get_string(src, encoding_rs::SHIFT_JIS, errors))
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
            let (bytes, _, has_errors) = encoding.encode(value);
            if has_errors {
                self.write_u32_little_endian(0u32)?;
                Err(NanoemError::EncodeStringFailed(value.to_owned()))
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

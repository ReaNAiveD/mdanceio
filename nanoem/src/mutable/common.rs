use crate::{
    common::{Buffer, Status, F128},
    utils::CodecType,
};

#[macro_export]
macro_rules! write_primitive {
    ($typ: ty, $write_typ:ident) => {
        pub fn $write_typ(&mut self, value: $typ) -> Result<(), Status> {
            self.write_byte_array(&value.to_le_bytes())
        }
    };
}

struct MutableBuffer {
    offset: usize,
    data: Vec<u8>,
}

impl MutableBuffer {
    fn create() -> Result<MutableBuffer, Status> {
        return Self::create_with_reserved_size(2 << 12);
    }

    fn create_with_reserved_size(capacity: usize) -> Result<MutableBuffer, Status> {
        let mut buffer = MutableBuffer {
            offset: 0usize,
            data: Vec::new(),
        };
        buffer.ensure_size(capacity)?;
        Ok(buffer)
    }

    fn ensure_size(&mut self, required: usize) -> Result<(), Status> {
        if let Err(e) = self.data.try_reserve(required) {
            self.offset = 0;
            Err(Status::ErrorReallocFailed)
        } else {
            Ok(())
        }
    }

    // TODO: Not Process with callback
    fn write_byte_array(&mut self, data: &[u8]) -> Result<(), Status> {
        self.ensure_size(data.len())?;
        if self.offset == data.len() {
            self.data.extend_from_slice(data);
            Ok(())
        } else {
            Err(Status::ErrorIllegalBufferWriteOffset)
        }
    }

    fn write_byte(&mut self, value: u8) -> Result<(), Status> {
        self.write_byte_array(&[value])
    }

    write_primitive!(u16, write_u16_little_endian);
    write_primitive!(i16, write_i16_little_endian);
    write_primitive!(u32, write_u32_little_endian);
    write_primitive!(i32, write_i32_little_endian);
    write_primitive!(f32, write_f32_little_endian);

    fn write_string(&mut self, value: String, codec_type: CodecType) -> Result<(), Status> {
        let (bytes, _, success) = codec_type.get_encoding_object().encode(&value[..]);
        if !success {
            self.write_u32_little_endian(0u32)?;
            Err(Status::ErrorEncodeUnicodeStringFailed)
        } else {
            self.write_u32_little_endian(bytes.len() as u32)?;
            self.write_byte_array(&bytes)?;
            Ok(())
        }
    }

    fn write_integer(&mut self, value: i32, size: usize) -> Result<(), Status> {
        match size {
            1 => self.write_byte(value as u8),
            2 => self.write_u16_little_endian(value as u16),
            4 | _ => self.write_i32_little_endian(value),
        }
    }

    fn write_f32_2_little_endian(&mut self, value: F128) -> Result<(), Status> {
        self.write_f32_little_endian(value.0[0])?;
        self.write_f32_little_endian(value.0[1])
    }

    fn write_f32_3_little_endian(&mut self, value: F128) -> Result<(), Status> {
        self.write_f32_2_little_endian(value)?;
        self.write_f32_little_endian(value.0[2])
    }

    fn write_f32_4_little_endian(&mut self, value: F128) -> Result<(), Status> {
        self.write_f32_3_little_endian(value)?;
        self.write_f32_little_endian(value.0[3])
    }

    // TODO: now clone the total data vec, change to some pointer copy
    fn create_buffer_object(&self) -> Result<Buffer, Status> {
        Ok(Buffer::create(self.data.clone()))
    }
}

use std::cmp;

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

pub fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

pub fn u8_slice_get_string(slice: &[u8], codec: CodecType) -> Option<String> {
    let mut src = slice;
    if let Some(pos) = src.iter().position(|c| *c == 0u8) {
        src = src.split_at(pos).0;
    }
    let (cow, _, had_errors) = codec.get_encoding_object().decode(src);
    if had_errors {
        None
    } else {
        Some(cow.into())
    }
}

pub fn compare(a: &[u8], b: &[u8]) -> cmp::Ordering {
    for (ai, bi) in a.iter().zip(b.iter()) {
        match ai.cmp(&bi) {
            cmp::Ordering::Equal => continue,
            ord => return ord
        }
    }

    /* if every single element was equal, compare length */
    a.len().cmp(&b.len())
}

#[test]
fn test_fourcc() {
    assert_eq!(1u32, fourcc(1u8, 0u8, 0u8, 0u8));
}

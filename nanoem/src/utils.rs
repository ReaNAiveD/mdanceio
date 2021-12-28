pub fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

pub fn u8_slice_get_string(slice: &[u8]) -> Option<String> {
    let mut src = slice;
    if let Some(pos) = src.iter().position(|c| *c == 0u8) {
        src = src.split_at(pos).0;
    }
    let (cow, _, had_errors) = encoding_rs::UTF_8.decode(src);
    if had_errors {
        None
    } else {
        Some(cow.into())
    }
}

#[test]
fn test_fourcc() {
    assert_eq!(1u32, fourcc(1u8, 0u8, 0u8, 0u8));
}

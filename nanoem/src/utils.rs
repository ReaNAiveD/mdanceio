use std::cmp;

pub fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

pub fn u8_slice_get_string(
    slice: &[u8],
    encoding: &'static encoding_rs::Encoding,
) -> Option<String> {
    let mut src = slice;
    if let Some(pos) = src.iter().position(|c| *c == 0u8) {
        src = src.split_at(pos).0;
    }
    let (cow, _, had_errors) = encoding.decode(src);
    if had_errors {
        None
    } else {
        Some(cow.into())
    }
}

pub fn compare(a: &[u8], b: &[u8]) -> cmp::Ordering {
    for (ai, bi) in a.iter().zip(b.iter()) {
        match ai.cmp(bi) {
            cmp::Ordering::Equal => continue,
            ord => return ord,
        }
    }

    /* if every single element was equal, compare length */
    a.len().cmp(&b.len())
}

#[test]
fn test_fourcc() {
    assert_eq!(1u32, fourcc(1u8, 0u8, 0u8, 0u8));
}

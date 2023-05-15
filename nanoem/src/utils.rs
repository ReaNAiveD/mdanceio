use std::cmp;

use crate::common::NanoemError;

pub fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

pub fn u8_slice_get_string(
    slice: &[u8],
    encoding: &'static encoding_rs::Encoding,
    errors: &mut Vec<NanoemError>,
) -> String {
    let (cow, _, had_errors) = encoding.decode(slice);
    let result = cow.split('\0').next().unwrap_or("").to_owned();
    if had_errors {
        errors.push(NanoemError::DecodeStringFailed {
            data: slice.into(),
            parsed: result.clone(),
            encoding: encoding.name(),
        })
    }
    result
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

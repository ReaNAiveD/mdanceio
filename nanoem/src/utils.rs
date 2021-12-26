pub fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_le_bytes([a, b, c, d])
}

#[test]
fn test_fourcc() {
    assert_eq!(1u32, fourcc(1u8, 0u8, 0u8, 0u8));
}

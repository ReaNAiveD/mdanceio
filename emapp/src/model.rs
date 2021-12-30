use std::collections::HashMap;

struct Bone {}

impl Bone {
    const DEFAULT_BAZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];
    const DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT: [u8; 4] = [64, 0, 64, 127];
    const NAME_ROOT_PARENT_IN_JAPANESE: &'static [u8] = &[
        0xe5, 0x85, 0xa8, 0xe3, 0x81, 0xa6, 0xe3, 0x81, 0xae, 0xe8, 0xa6, 0xaa, 0x0,
    ];
}
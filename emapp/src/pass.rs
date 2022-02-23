pub struct Buffer {
    pub num_indices: usize,
    pub num_offset: usize,
    pub depth_enabled: bool,
}

impl Buffer {
    pub fn new(num_indices: usize, num_offset: usize, depth_enabled: bool) -> Self {
        Self {
            num_indices: usize::min(num_indices, 0x7fffffffusize),
            num_offset: usize::min(num_offset, 0x7fffffffusize),
            depth_enabled,
        }
    }
}
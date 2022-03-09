pub struct Buffer<'a> {
    pub num_indices: usize,
    pub num_offset: usize,
    pub vertex_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
    pub depth_enabled: bool,
}

impl<'a> Buffer<'a> {
    pub fn new(num_indices: usize, num_offset: usize, vertex_buffer: &'a wgpu::Buffer, index_buffer: &'a wgpu::Buffer, depth_enabled: bool) -> Self {
        Self {
            num_indices: usize::min(num_indices, 0x7fffffffusize),
            num_offset: usize::min(num_offset, 0x7fffffffusize),
            vertex_buffer,
            index_buffer,
            depth_enabled,
        }
    }

    pub fn is_depth_enabled(&self) -> bool {
        self.depth_enabled
    }
}
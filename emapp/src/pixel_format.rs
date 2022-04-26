pub struct PixelFormat {
    pub color_texture_formats: Vec<wgpu::TextureFormat>,
    pub depth_texture_format: wgpu::TextureFormat,
    pub num_sample: u32,
}

impl PixelFormat {
    pub fn new(num_sample: u32) -> Self {
        Self {
            color_texture_formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            // color_texture_formats: vec![wgpu::TextureFormat::Rgba8UnormSrgb],
            depth_texture_format: wgpu::TextureFormat::Depth24PlusStencil8,
            num_sample,
        }
    }
}
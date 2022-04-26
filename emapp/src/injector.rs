#[derive(Debug, Clone, Copy)]
pub struct Injector {
    // TODO
    pub pixel_format: wgpu::TextureFormat,
}

impl Injector {
    pub fn texture_format(&self) -> wgpu::TextureFormat {
        self.pixel_format
    }
}

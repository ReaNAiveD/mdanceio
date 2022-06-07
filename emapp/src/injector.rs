#[derive(Debug, Clone, Copy)]
pub struct Injector {
    // TODO
    pub pixel_format: wgpu::TextureFormat,
    pub window_device_pixel_ratio: f32,
    pub viewport_device_pixel_ratio: f32,
    pub window_size: [u16; 2],
    
}

impl Injector {
    pub fn texture_format(&self) -> wgpu::TextureFormat {
        self.pixel_format
    }
}

pub trait ImageView {
    // TODO
    fn handle(&self) -> &wgpu::Texture;
}
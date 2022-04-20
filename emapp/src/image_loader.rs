use crate::image_view::ImageView;

pub struct ImageLoader {
    // TODO
}

pub struct Image {
    handler: wgpu::Texture,
}

impl Image {
    pub fn empty() -> Self {
        todo!()
    }
}

impl ImageView for Image {
    fn handle(&self) -> &wgpu::Texture {
        &self.handler
    }
}
use image::GenericImageView;

use crate::{injector::Injector, project::Project, state_controller::StateController};
use std::io::Cursor;

pub struct BaseApplicationService {
    state_controller: StateController,
    injector: Injector,
}

impl BaseApplicationService {
    pub fn new(
        sc_desc: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        injector: Injector,
    ) -> Self {
        Self {
            state_controller: StateController::new(sc_desc, adapter, device, queue, injector),
            injector,
        }
    }

    pub fn draw_default_pass(
        &mut self,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let project = self.state_controller.current_mut_project();
        project.draw_viewport(view, adapter, device, queue);
    }

    pub fn load_model(&mut self, data: &[u8], device: &wgpu::Device) {
        self.state_controller.load_model(data, device);
    }

    pub fn load_texture(
        &mut self,
        key: &str,
        data: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let img = image::io::Reader::with_format(
            Cursor::new(data),
            image::ImageFormat::from_extension(key.split('.').rev().next().unwrap()).unwrap(),
        )
        .decode()
        .unwrap();
        self.load_decoded_texture(key, &img.to_rgba8(), img.dimensions(), device, queue);
    }

    pub fn load_decoded_texture(
        &mut self,
        key: &str,
        data: &[u8],
        dimensions: (u32, u32),
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.state_controller
            .load_texture(key, data, dimensions, device, queue);
    }
}

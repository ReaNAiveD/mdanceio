use crate::{injector::Injector, project::Project};

pub struct StateController {
    project: Project,
}

impl StateController {
    pub fn new(
        sc_desc: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        injector: Injector,
    ) -> Self {
        Self {
            project: Project::new(
                sc_desc,
                adapter,
                device,
                queue,
                injector,
            ),
        }
    }

    pub fn current_mut_project(&mut self) -> &mut Project {
        &mut self.project
    }

    pub fn load_model(&mut self, data: &[u8], device: &wgpu::Device) {
        self.project.load_model(data, device);
    }
    
    pub fn load_texture(
        &mut self,
        key: &str,
        data: &[u8],
        dimensions: (u32, u32),
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.project.load_texture(key, data, dimensions, device, queue);
    }

    pub fn update_bind_texture(&mut self) {
        self.project.update_bind_texture();
    }
}

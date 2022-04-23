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
    ) -> Self {
        Self {
            project: Project::new(
                sc_desc,
                adapter,
                device,
                queue,
                Injector {
                    pixel_format: wgpu::TextureFormat::Rgba8UnormSrgb,
                },
            ),
        }
    }

    pub fn current_mut_project(&mut self) -> &mut Project {
        &mut self.project
    }

    pub fn load_model(&mut self, data: &[u8], device: &wgpu::Device) {
        self.project.load_tmp_model(data, device);
    }
}

use crate::{project::Project, state_controller::StateController, injector::Injector};

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
}

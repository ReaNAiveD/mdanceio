use crate::{project::Project, state_controller::StateController};

pub struct BaseApplicationService {
    state_controller: StateController,
}

impl BaseApplicationService {
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
}

use crate::{state_controller::StateController, project::Project};

pub struct BaseApplicationService {
    state_controller: StateController,
}

impl BaseApplicationService {
    pub fn draw_default_pass(&mut self) {
        let project = self.state_controller.current_mut_project();
    }

    fn draw_project(&self, project: &mut Project) {
        project.draw_viewport();
    }
}
use crate::project::Project;

pub struct StateController {
    project: Project,
}

impl StateController {
    pub fn current_mut_project(&mut self) -> &mut Project {
        &mut self.project
    }
}
use std::{rc::Rc, cell::RefCell};

use crate::{project::Project, motion::Motion};

pub trait Undo {
    fn undo(&self);
    fn redo(&self);
    fn current_project(&self) -> &Rc<RefCell<Project>>;
    fn name(&self) -> &String;
}

struct BaseUndoCommand<'a> {
    project: Rc<RefCell<Project<'a>>>,
}

struct BaseKeyframeCommand<'a> {
    base: BaseUndoCommand<'a>,
    motion: Motion,
}

impl<'a> BaseKeyframeCommand<'a> {
    fn reset_transform_performed_at(&self) {
        self.base.project.borrow_mut().reset_transform_performed_at()
    }

    fn commit(&self, motion: &nanoem::motion::Motion) {
        let project = self.base.project.borrow_mut();
        let last_duration = project.project_duration();
        // motion.sort
    }
}
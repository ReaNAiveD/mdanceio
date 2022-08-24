use crate::project::Project;

#[derive(Debug, Default)]
pub struct TimeLineSegment {
    pub from: u32,
    pub to: u32,
    pub enable_from: bool,
    pub enable_to: bool,
}

impl TimeLineSegment {
    pub fn from_project(project: &Project) -> Self {
        Self {
            from: 0,
            to: project.project_duration(),
            enable_from: false,
            enable_to: false,
        }
    }

    pub fn normalized(&self, duration: u32) -> Self {
        Self {
            from: self.from.min(self.to).min(duration),
            to: self.from.max(self.to).min(duration),
            enable_from: self.enable_from,
            enable_to: self.enable_to,
        }
    }

    pub fn frame_index_from(&self) -> u32 {
        if self.enable_from {
            self.from
        } else {
            0
        }
    }

    pub fn frame_index_to(&self, duration: u32)  -> u32 {
        if self.enable_to {
            self.to
        } else {
            duration
        }
    }
}

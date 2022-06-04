use crate::model_program_bundle::CommonPass;

pub trait Technique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass>;

    fn reset_script_command_state(&self);

    fn reset_script_external_color(&self);

    fn has_next_script_command(&self) -> bool;
}
use crate::model_program_bundle::CommonPass;

pub trait Technique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        fallback_texture: &'b wgpu::TextureView,
        device: &wgpu::Device,
    ) -> Option<CommonPass<'a>>;

    fn reset_script_command_state(&self);

    fn reset_script_external_color(&self);

    fn has_next_script_command(&self) -> bool;
}

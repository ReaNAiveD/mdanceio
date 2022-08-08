use std::rc::Rc;

use crate::{project::Project, camera::PerspectiveCamera, shadow_camera::ShadowCamera, light::DirectionalLight, model::Model, model_program_bundle::ModelProgramBundle};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawType {
    // TODO
    Color = 0,
    Edge,
    GroundShadow,
    ShadowMap,
    ScriptExternalColor,
}

pub struct DrawContext<'a> {
    pub effect: &'a mut ModelProgramBundle,
    pub camera: &'a PerspectiveCamera,
    pub shadow: &'a ShadowCamera,
    pub light: &'a DirectionalLight,
    pub shared_fallback_texture: &'a wgpu::TextureView,
    pub viewport_texture_format: wgpu::TextureFormat,
    pub is_render_pass_viewport: bool,
    pub all_models: &'a (dyn Iterator<Item = &'a Model>),
    pub texture_bind_layout: &'a wgpu::BindGroupLayout,
    pub shadow_bind_layout: &'a wgpu::BindGroupLayout,
    pub texture_fallback_bind: &'a wgpu::BindGroup,
    pub shadow_fallback_bind: &'a wgpu::BindGroup,
}

pub trait Drawable {
    // TODO
    fn draw(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        typ: DrawType,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );

    fn is_visible(&self) -> bool;
}

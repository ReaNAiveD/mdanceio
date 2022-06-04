use crate::project::Project;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawType {
    // TODO
    Color = 0,
    Edge,
    GroundShadow,
    ShadowMap,
    ScriptExternalColor,
}

pub trait Drawable {
    // TODO
    fn draw(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        typ: DrawType,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    );

    fn is_visible(&self) -> bool;
}

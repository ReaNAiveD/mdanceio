pub enum ScriptClass {
    Object,
    Scene,
    SceneObject,
}

pub enum ScriptOrder {
    DependsOnScriptExternal,
    PreProcess,
    Standard,
    PostProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFormat {
    pub color: wgpu::TextureFormat,
    pub depth: Option<wgpu::TextureFormat>,
}

mod effect;
mod layout;
pub mod render_target;
pub mod technique;
mod uniform;

pub use effect::Effect;

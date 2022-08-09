pub mod global_uniform;
pub mod common;

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

pub trait IEffect {
    fn script_class(&self) -> ScriptClass;
    fn script_order(&self) -> ScriptOrder;
    fn has_script_external(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct Effect {
    
}
use std::{collections::HashMap, rc::Rc};

use super::{
    layout::RendererLayout,
    technique::{Technique, TechniqueType},
};

#[derive(Debug, Clone, Copy)]
pub struct EffectConfig {
    pub depth_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
}

pub struct Effect {
    config: EffectConfig,
    layout: Rc<RendererLayout>,
    pub technique: HashMap<TechniqueType, Technique>,
}

impl Effect {
    pub fn new(
        shaders: HashMap<TechniqueType, &str>,
        depth_enabled: bool,
        device: &wgpu::Device,
    ) -> Self {
        let layout = Rc::new(RendererLayout::new(device));
        let depth_compare = if depth_enabled {
            wgpu::CompareFunction::Less
        } else {
            wgpu::CompareFunction::Always
        };
        let config = EffectConfig {
            depth_enabled: true,
            depth_compare,
        };
        let technique = shaders
            .iter()
            .map(|(typ, shader)| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Effect/Object/Shader"),
                    source: wgpu::ShaderSource::Wgsl((*shader).into()),
                });
                (*typ, { Technique::new(*typ, config, shader, &layout) })
            })
            .collect();
        Self {
            config,
            layout,
            technique,
        }
    }

}
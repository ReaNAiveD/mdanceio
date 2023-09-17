use std::{collections::HashMap, sync::Arc};

use super::{
    layout::RendererLayout,
    render_target::DrawType,
    technique::{Technique, TechniqueType},
};

#[derive(Debug, Clone, Copy)]
pub struct EffectConfig {
    pub depth_enabled: bool,
}

pub struct Effect {
    layout: Arc<RendererLayout>,
    pub technique: HashMap<TechniqueType, Technique>,
}

impl Effect {
    pub fn new(
        shaders: HashMap<TechniqueType, &str>,
        depth_enabled: bool,
        fallback_shadow_bind: &Arc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) -> Self {
        let layout = Arc::new(RendererLayout::new(device));
        let config = EffectConfig {
            depth_enabled,
        };
        let technique = shaders
            .iter()
            .map(|(typ, shader)| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Effect/Object/Shader"),
                    source: wgpu::ShaderSource::Wgsl((*shader).into()),
                });
                (*typ, {
                    Technique::new(*typ, config, shader, &layout, fallback_shadow_bind)
                })
            })
            .collect();
        Self {
            layout,
            technique,
        }
    }

    pub fn find_technique(&self, draw_type: DrawType) -> Option<&Technique> {
        match draw_type {
            DrawType::Color(true) => {
                if self.technique.get(&TechniqueType::Zplot).is_some() {
                    self.technique.get(&TechniqueType::Object)
                } else {
                    self.technique
                        .get(&TechniqueType::ObjectSs)
                        .or_else(|| self.technique.get(&TechniqueType::Object))
                }
            }
            DrawType::Color(false) => self
                .technique
                .get(&TechniqueType::ObjectSs)
                .or_else(|| self.technique.get(&TechniqueType::Object)),
            DrawType::Edge => self.technique.get(&TechniqueType::Edge),
            DrawType::GroundShadow => self.technique.get(&TechniqueType::Shadow),
            DrawType::ShadowMap => self.technique.get(&TechniqueType::Zplot),
        }
    }
}
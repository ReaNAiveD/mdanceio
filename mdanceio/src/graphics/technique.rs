use std::{collections::HashMap, rc::Rc};

use super::common_pass::{CPass, CPassDescription, CPassLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TechniqueType {
    Color,
    Edge,
    Shadow,
    Zplot,
}

impl From<TechniqueType> for String {
    fn from(v: TechniqueType) -> Self {
        match v {
            TechniqueType::Color => "object".to_owned(),
            TechniqueType::Edge => "edge".to_owned(),
            TechniqueType::Shadow => "shadow".to_owned(),
            TechniqueType::Zplot => "zplot".to_owned(),
        }
    }
}

impl TechniqueType {
    pub fn from_str(v: &str) -> Option<Self> {
        match v {
            "object" => Some(Self::Color),
            "edge" => Some(Self::Edge),
            "shadow" => Some(Self::Shadow),
            "zplot" => Some(Self::Zplot),
            _ => None,
        }
    }
}

pub struct ObjectPassKey {
    pub color_format: wgpu::TextureFormat,
    pub is_add_blend: bool,
    pub depth_enabled: bool,
    pub line_draw_enabled: bool,
    pub point_draw_enabled: bool,
    pub culling_disabled: bool,
}

impl From<ObjectPassKey> for CPassDescription {
    fn from(key: ObjectPassKey) -> Self {
        let primitive_type = if key.line_draw_enabled {
            wgpu::PrimitiveTopology::LineList
        } else if key.point_draw_enabled {
            wgpu::PrimitiveTopology::PointList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let color_blend = if key.is_add_blend {
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };
        let depth_compare = if key.depth_enabled {
            wgpu::CompareFunction::LessEqual
        } else {
            wgpu::CompareFunction::Always
        };
        CPassDescription {
            color_texture_format: key.color_format,
            depth_texture_format: Some(wgpu::TextureFormat::Depth16Unorm),
            cull_mode: if key.culling_disabled {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            primitive_type,
            color_blend: Some(color_blend),
            depth_enabled: key.depth_enabled,
            depth_compare,
        }
    }
}

pub struct ObjectTechnique {
    technique_type: TechniqueType,
    shader: wgpu::ShaderModule,
    pipeline_bundle: HashMap<CPassDescription, Rc<CPass>>,
}

impl ObjectTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelColor"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/model_color.wgsl").into(),
            ),
        });
        Self {
            technique_type: TechniqueType::Color,
            shader,
            pipeline_bundle: HashMap::new(),
        }
    }

    pub fn ensure(
        &mut self,
        layout: &CPassLayout,
        key: ObjectPassKey,
        device: &wgpu::Device,
    ) -> &Rc<CPass> {
        let desc = key.into();
        self.pipeline_bundle
            .entry(desc)
            .or_insert_with(|| Rc::new(CPass::new(&self.shader, &desc, layout, device)))
    }
}

pub struct EdgePassKey {
    pub color_format: wgpu::TextureFormat,
    pub is_add_blend: bool,
    pub depth_enabled: bool,
    pub line_draw_enabled: bool,
    pub point_draw_enabled: bool,
}

impl From<EdgePassKey> for CPassDescription {
    fn from(key: EdgePassKey) -> Self {
        let primitive_type = if key.line_draw_enabled {
            wgpu::PrimitiveTopology::LineList
        } else if key.point_draw_enabled {
            wgpu::PrimitiveTopology::PointList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let color_blend = if key.is_add_blend {
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };
        let depth_compare = if key.depth_enabled {
            wgpu::CompareFunction::LessEqual
        } else {
            wgpu::CompareFunction::Always
        };
        CPassDescription {
            color_texture_format: key.color_format,
            depth_texture_format: Some(wgpu::TextureFormat::Depth16Unorm),
            cull_mode: Some(wgpu::Face::Front),
            primitive_type,
            color_blend: Some(color_blend),
            depth_enabled: key.depth_enabled,
            depth_compare,
        }
    }
}

pub struct EdgeTechnique {
    technique_type: TechniqueType,
    shader: wgpu::ShaderModule,
    pipeline_bundle: HashMap<CPassDescription, Rc<CPass>>,
}

impl EdgeTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelEdge"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/model_edge.wgsl").into(),
            ),
        });
        Self {
            technique_type: TechniqueType::Edge,
            shader,
            pipeline_bundle: HashMap::new(),
        }
    }

    pub fn ensure(
        &mut self,
        layout: &CPassLayout,
        key: EdgePassKey,
        device: &wgpu::Device,
    ) -> &Rc<CPass> {
        let desc = key.into();
        self.pipeline_bundle
            .entry(desc)
            .or_insert_with(|| Rc::new(CPass::new(&self.shader, &desc, layout, device)))
    }
}

pub struct ShadowPassKey {
    pub color_format: wgpu::TextureFormat,
    pub is_add_blend: bool,
    pub depth_enabled: bool,
    pub line_draw_enabled: bool,
    pub point_draw_enabled: bool,
}

impl From<ShadowPassKey> for CPassDescription {
    fn from(key: ShadowPassKey) -> Self {
        let primitive_type = if key.line_draw_enabled {
            wgpu::PrimitiveTopology::LineList
        } else if key.point_draw_enabled {
            wgpu::PrimitiveTopology::PointList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let color_blend = if key.is_add_blend {
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };
        let depth_compare = if key.depth_enabled {
            wgpu::CompareFunction::Less
        } else {
            wgpu::CompareFunction::Always
        };
        CPassDescription {
            color_texture_format: key.color_format,
            depth_texture_format: Some(wgpu::TextureFormat::Depth16Unorm),
            cull_mode: None,
            primitive_type,
            color_blend: Some(color_blend),
            depth_enabled: key.depth_enabled,
            depth_compare,
        }
    }
}

pub struct GroundShadowTechnique {
    technique_type: TechniqueType,
    shader: wgpu::ShaderModule,
    pipeline_bundle: HashMap<CPassDescription, Rc<CPass>>,
}

impl GroundShadowTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelGroundShadow"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/model_ground_shadow.wgsl").into(),
            ),
        });
        Self {
            technique_type: TechniqueType::Shadow,
            shader,
            pipeline_bundle: HashMap::new(),
        }
    }

    pub fn ensure(
        &mut self,
        layout: &CPassLayout,
        key: ShadowPassKey,
        device: &wgpu::Device,
    ) -> &Rc<CPass> {
        let desc = key.into();
        self.pipeline_bundle
            .entry(desc)
            .or_insert_with(|| Rc::new(CPass::new(&self.shader, &desc, layout, device)))
    }
}

pub struct ZplotPassKey {
    pub depth_enabled: bool,
    pub line_draw_enabled: bool,
    pub point_draw_enabled: bool,
    pub culling_disabled: bool,
}

impl From<ZplotPassKey> for CPassDescription {
    fn from(key: ZplotPassKey) -> Self {
        let primitive_type = if key.line_draw_enabled {
            wgpu::PrimitiveTopology::LineList
        } else if key.point_draw_enabled {
            wgpu::PrimitiveTopology::PointList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let depth_compare = if key.depth_enabled {
            wgpu::CompareFunction::LessEqual
        } else {
            wgpu::CompareFunction::Always
        };
        CPassDescription {
            color_texture_format: wgpu::TextureFormat::R32Float,
            depth_texture_format: Some(wgpu::TextureFormat::Depth16Unorm),
            cull_mode: if key.culling_disabled {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            primitive_type,
            color_blend: None,
            depth_enabled: key.depth_enabled,
            depth_compare,
        }
    }
}

pub struct ZplotTechnique {
    technique_type: TechniqueType,
    shader: wgpu::ShaderModule,
    pipeline_bundle: HashMap<CPassDescription, Rc<CPass>>,
}

impl ZplotTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelZplot"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/model_zplot.wgsl").into(),
            ),
        });
        Self {
            technique_type: TechniqueType::Zplot,
            shader,
            pipeline_bundle: HashMap::new(),
        }
    }

    pub fn ensure(
        &mut self,
        layout: &CPassLayout,
        key: ZplotPassKey,
        device: &wgpu::Device,
    ) -> &Rc<CPass> {
        let desc = key.into();
        self.pipeline_bundle
            .entry(desc)
            .or_insert_with(|| Rc::new(CPass::new(&self.shader, &desc, layout, device)))
    }
}

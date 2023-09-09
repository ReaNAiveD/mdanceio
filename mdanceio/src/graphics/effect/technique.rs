use std::{
    collections::HashMap,
    rc::{Rc, Weak},
    sync::RwLock,
};

use crate::{model::Material, project::ModelHandle};

use super::{
    effect::EffectConfig,
    layout::RendererLayout,
    render_target::RendererConfig,
    uniform::{MaterialUniform, UniformBind, UniformBindData},
    RenderFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TechniqueType {
    Object,
    ObjectSs,
    Edge,
    Shadow,
    Zplot,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct PipelineKey {
    pub format: RenderFormat,
    pub cull_mode: Option<wgpu::Face>,
    pub primitive_type: wgpu::PrimitiveTopology,
    pub color_blend: Option<wgpu::BlendState>,
}

#[derive(Debug, Clone)]
pub struct DrawPassModelContext {
    pub handle: ModelHandle,
    pub material_size: usize,
    pub add_blend: bool,
    pub buffer: Rc<wgpu::Buffer>,
    pub index_buffer: Rc<wgpu::Buffer>,
}

impl DrawPassModelContext {
    pub fn to_pass_vertex(&self, offset: u32, num: u32) -> DrawPassVertex {
        DrawPassVertex {
            buffer: self.buffer.clone(),
            index_buffer: self.index_buffer.clone(),
            offset,
            num,
        }
    }
}

#[derive(Debug)]
pub struct Technique {
    pub typ: TechniqueType,
    config: EffectConfig,
    shader: wgpu::ShaderModule,
    layout: Rc<RendererLayout>,
    fallback_shadow_bind: Rc<wgpu::BindGroup>,
    pipelines: RwLock<HashMap<PipelineKey, Weak<wgpu::RenderPipeline>>>,
    uniforms: RwLock<HashMap<ModelHandle, UniformBind>>,
}

impl Technique {
    pub fn new(
        typ: TechniqueType,
        config: EffectConfig,
        shader: wgpu::ShaderModule,
        layout: &Rc<RendererLayout>,
        fallback_shadow_bind: &Rc<wgpu::BindGroup>,
    ) -> Self {
        Self {
            typ,
            config,
            shader,
            layout: layout.clone(),
            fallback_shadow_bind: fallback_shadow_bind.clone(),
            pipelines: RwLock::new(HashMap::new()),
            uniforms: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_pipeline(
        &self,
        render_format: RenderFormat,
        color_blend: wgpu::BlendState,
        material: &Material,
        device: &wgpu::Device,
    ) -> Rc<wgpu::RenderPipeline> {
        let cull_mode = match self.typ {
            TechniqueType::Edge => Some(wgpu::Face::Front),
            TechniqueType::Shadow => None,
            _ => {
                if material.is_culling_disabled() {
                    None
                } else {
                    Some(wgpu::Face::Back)
                }
            }
        };
        let color_blend = if matches!(self.typ, TechniqueType::Zplot) {
            None
        } else {
            Some(color_blend)
        };
        let primitive_type = if material.is_line_draw_enabled() {
            wgpu::PrimitiveTopology::LineList
        } else if material.is_point_draw_enabled() {
            wgpu::PrimitiveTopology::PointList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let key = PipelineKey {
            format: render_format,
            cull_mode,
            primitive_type,
            color_blend,
        };
        self.get_sharing_pipeline(&key).unwrap_or_else(|| {
            let pipeline = Rc::new(self.build_pipeline(&key, device));
            self.pipelines
                .write()
                .unwrap()
                .insert(key, Rc::downgrade(&pipeline));
            pipeline
        })
    }

    pub fn get_sharing_pipeline(&self, key: &PipelineKey) -> Option<Rc<wgpu::RenderPipeline>> {
        self.pipelines
            .read()
            .ok()
            .and_then(|map| map.get(key).and_then(|weak| weak.upgrade()))
    }

    fn build_pipeline(&self, key: &PipelineKey, device: &wgpu::Device) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(""),
            layout: Some(&self.layout.pipeline_layout),
            primitive: wgpu::PrimitiveState {
                topology: key.primitive_type,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: key.cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: "vs_main",
                buffers: &[self.layout.vertex_buffer_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: key.format.color,
                    blend: key.color_blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            depth_stencil: key.format.depth.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: self.config.depth_enabled,
                depth_compare: self.config.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    pub fn get_uniform(
        &self,
        handle: ModelHandle,
        material_size: usize,
        device: &wgpu::Device,
    ) -> Rc<wgpu::BindGroup> {
        self.get_sharing_uniform(handle).unwrap_or_else(|| {
            let bind = UniformBind::new(&self.layout.uniform_bind_layout, material_size, device);
            let bg = bind.bind_group().clone();
            self.uniforms.write().unwrap().insert(handle, bind);
            bg
        })
    }

    pub fn get_sharing_uniform(&self, handle: ModelHandle) -> Option<Rc<wgpu::BindGroup>> {
        self.uniforms
            .read()
            .ok()
            .and_then(|map| map.get(&handle).map(|bind| bind.bind_group()))
    }

    pub fn update_uniform(
        &self,
        handle: ModelHandle,
        updater: &dyn Fn(&mut UniformBindData),
        queue: &wgpu::Queue,
    ) {
        match self.uniforms.read() {
            Ok(map) => {
                if let Some(bind) = map.get(&handle) {
                    let mut data = bind.get_empty_uniform_data();
                    updater(&mut data);
                    bind.update(&data, queue);
                }
            }
            Err(e) => {
                log::error!("Failed to read technique uniform map: {:?}", e);
            }
        }
    }

    pub fn get_draw_pass(
        &self,
        config: &RendererConfig,
        model_ctx: &DrawPassModelContext,
        material_idx: usize,
        material: &Material,
        shadow_bind: &Rc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) -> DrawPass {
        let color_blend = if model_ctx.add_blend {
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };
        let color_bind = material.bind_group();
        let format = if self.typ == TechniqueType::Zplot {
            RenderFormat {
                color: wgpu::TextureFormat::R32Float,
                depth: config.format.depth,
            }
        } else {
            config.format
        };
        let shadow_bind = if self.typ == TechniqueType::Zplot {
            &self.fallback_shadow_bind
        } else {
            shadow_bind
        };
        let pipeline = self.get_pipeline(format, color_blend, material, device);
        DrawPass::new(
            pipeline,
            format,
            DrawPassBind {
                color_bind,
                shadow_bind: shadow_bind.clone(),
                uniform_bind: self.get_uniform(model_ctx.handle, model_ctx.material_size, device),
                material_idx,
            },
            model_ctx.to_pass_vertex(material.index_offset, material.num_indices),
            device,
        )
    }
}

#[derive(Debug, Clone)]
pub struct DrawPassBind {
    pub color_bind: Rc<wgpu::BindGroup>,
    pub shadow_bind: Rc<wgpu::BindGroup>,
    pub uniform_bind: Rc<wgpu::BindGroup>,
    pub material_idx: usize,
}

#[derive(Debug, Clone)]
pub struct DrawPassVertex {
    pub buffer: Rc<wgpu::Buffer>,
    pub index_buffer: Rc<wgpu::Buffer>,
    pub offset: u32,
    pub num: u32,
}

#[derive(Debug)]
pub struct DrawPass {
    pub pipeline: Rc<wgpu::RenderPipeline>,
    pub format: RenderFormat,
    pub bind: DrawPassBind,
    pub vertex: DrawPassVertex,
    pub render_bundle: wgpu::RenderBundle,
}

impl DrawPass {
    fn new(
        pipeline: Rc<wgpu::RenderPipeline>,
        format: RenderFormat,
        bind: DrawPassBind,
        vertex: DrawPassVertex,
        device: &wgpu::Device,
    ) -> Self {
        let render_bundle = Self::build_bundle(&pipeline, format, &bind, &vertex, device);
        Self {
            pipeline,
            format,
            bind: bind.clone(),
            vertex: vertex.clone(),
            render_bundle,
        }
    }

    pub fn update_color_bind(&mut self, color_bind: Rc<wgpu::BindGroup>, device: &wgpu::Device) {
        self.bind.color_bind = color_bind;
        self.render_bundle = self.rebuild_bundle(device);
    }

    pub fn rebuild_bundle(&self, device: &wgpu::Device) -> wgpu::RenderBundle {
        Self::build_bundle(
            &self.pipeline,
            self.format,
            &self.bind,
            &self.vertex,
            device,
        )
    }
}

impl DrawPass {
    fn build_bundle(
        pipeline: &wgpu::RenderPipeline,
        format: RenderFormat,
        bind: &DrawPassBind,
        vertex: &DrawPassVertex,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("ModelProgramBundle/RenderBundleEncoder"),
                color_formats: &[Some(format.color)],
                depth_stencil: format.depth.map(|format| wgpu::RenderBundleDepthStencil {
                    format,
                    depth_read_only: false,
                    stencil_read_only: true,
                }),
                sample_count: 1,
                multiview: None,
            });
        encoder.set_pipeline(pipeline);
        encoder.set_bind_group(0, &bind.color_bind, &[]);
        encoder.set_bind_group(
            1,
            &bind.uniform_bind,
            &[(bind.material_idx * std::mem::size_of::<MaterialUniform>()) as u32],
        );
        encoder.set_bind_group(2, &bind.shadow_bind, &[]);
        encoder.set_vertex_buffer(0, vertex.buffer.slice(..));
        encoder.set_index_buffer(vertex.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        let vertex_indices = vertex.offset..(vertex.offset + vertex.num);
        encoder.draw_indexed(vertex_indices, 0, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("DrawPass/RenderBundle"),
        })
    }
}

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use cgmath::Vector4;

use crate::{
    model::{Material, Model},
    project::ModelHandle,
};

use super::{
    effect::Effect,
    technique::{DrawPass, DrawPassModelContext, TechniqueType},
    uniform::UniformBindData,
    RenderFormat,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ObjectKey {
    Model(ModelHandle),
    Material((ModelHandle, usize)),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawType {
    Color(bool), // (shadow_map_enabled)
    Edge,
    GroundShadow,
    ShadowMap,
}

#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub format: RenderFormat,
    pub size: wgpu::Extent3d,
    pub draw_types: HashSet<DrawType>,
}

pub struct OffscreenRenderTarget {
    name: String,
    desc: String,
    clear_color: Vector4<f32>,
    clear_depth: f32,
    color: wgpu::Texture,
    depth: Option<wgpu::Texture>,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    objects: HashMap<ObjectKey, Arc<Effect>>,
}

#[derive(Debug, Clone)]
pub struct RenderTargetBuilder {
    pub clear_color: Vector4<f32>,
    pub clear_depth: f32,
    pub config: RendererConfig,
}

pub struct ScreenRenderTarget {
    pub clear_color: Vector4<f32>,
    pub clear_depth: f32,
    pub config: RendererConfig,
    renderers: HashMap<ModelHandle, ModelRenderer>,
    fallback_effect: Arc<Effect>,
}

impl ScreenRenderTarget {
    pub fn new(
        builder: RenderTargetBuilder,
        models: &HashMap<ModelHandle, Model>,
        shadow_bind: &Arc<wgpu::BindGroup>,
        fallback_effect: &Arc<Effect>,
        device: &wgpu::Device,
    ) -> Self {
        let depth = builder.config.format.depth.map(|depth_format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ScreenRenderTarget/Depth"),
                view_formats: &[depth_format],
                size: builder.config.size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            })
        });
        let mut rt = Self {
            clear_color: builder.clear_color,
            clear_depth: builder.clear_depth,
            config: builder.config,
            renderers: HashMap::new(),
            fallback_effect: fallback_effect.clone(),
        };
        for (model_handle, model) in models {
            rt.add_model(*model_handle, model, None, shadow_bind, device);
        }
        rt
    }

    pub fn add_model(
        &mut self,
        model_handle: ModelHandle,
        model: &Model,
        effect: Option<&Arc<Effect>>,
        shadow_bind: &Arc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) {
        let renderer = ModelRenderer::new(
            model_handle,
            model,
            effect.unwrap_or(&self.fallback_effect),
            shadow_bind,
            &self.config,
            device,
        );
        self.renderers.insert(model_handle, renderer);
    }

    pub fn remove_model(&mut self, model_handle: ModelHandle) {
        self.renderers.remove(&model_handle);
    }

    pub fn set_model_effect(
        &mut self,
        model_handle: ModelHandle,
        model: &Model,
        effect: &Arc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(renderer) = self.renderers.get_mut(&model_handle) {
            renderer.set_effect(model, effect, device)
        }
    }

    pub fn set_material_effect(
        &mut self,
        model_handle: ModelHandle,
        model: &Model,
        material_idx: usize,
        effect: &Arc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(renderer) = self.renderers.get_mut(&model_handle) {
            let material = model
                .materials
                .get(material_idx)
                .expect("material idx out of range");
            renderer.set_material_effect(material_idx, material, effect, device);
        }
    }

    pub fn remove_effect(&mut self, effect: Arc<Effect>, fallback_effect: Arc<Effect>) {
        todo!("remove_effect")
    }

    pub fn update_format(
        &mut self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) {
        todo!("update_format")
    }

    pub fn update_bind(
        &mut self,
        model: ModelHandle,
        material_idx: usize,
        bind: Arc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) {
        if let Some(renderer) = self
            .renderers
            .get_mut(&model)
            .and_then(|model| model.renderers.get_mut(material_idx))
        {
            renderer.update_color_bind(bind, device);
        }
    }

    pub fn render_bundles(&self, draw_type: DrawType) -> impl Iterator<Item = &wgpu::RenderBundle> {
        self.renderers
            .values()
            .flat_map(|renderer| &renderer.renderers)
            .filter_map(move |renderer| renderer.find_pass(draw_type))
            .map(|pass| &pass.render_bundle)
    }

    pub fn draw(
        &self,
        draw_type: DrawType,
        updater: &dyn Fn(ModelHandle, &mut UniformBindData),
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        for (model_handle, renderer) in &self.renderers {
            for renderer in &renderer.renderers {
                if let Some(technique) = renderer.effect.find_technique(draw_type) {
                    technique.update_uniform(
                        *model_handle,
                        &|data| updater(*model_handle, data),
                        queue,
                    );
                }
            }
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(format!("ScreenRenderTarget/{:?}/CommandEncoder", draw_type).as_str()),
        });
        let color_attachments = wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: true,
            },
        };
        {
            let mut _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(format!("ScreenRenderTarget/{:?}/RenderPass", draw_type).as_str()),
                color_attachments: &[Some(color_attachments)],
                depth_stencil_attachment: depth_view.map(|tv| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: tv,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                    }
                }),
            });
            _rpass.execute_bundles(self.render_bundles(draw_type));
        }
        queue.submit(Some(encoder.finish()));
    }
}

pub struct ModelRenderer {
    pub effect: Option<Arc<Effect>>,
    pub model_ctx: DrawPassModelContext,
    pub shadow_bind: Arc<wgpu::BindGroup>,
    pub config: RendererConfig,
    pub renderers: Vec<MaterialRenderer>,
}

impl ModelRenderer {
    pub fn new(
        model_handle: ModelHandle,
        model: &Model,
        effect: &Arc<Effect>,
        shadow_bind: &Arc<wgpu::BindGroup>,
        config: &RendererConfig,
        device: &wgpu::Device,
    ) -> Self {
        let mut renderers = vec![];
        let model_ctx = DrawPassModelContext {
            handle: model_handle,
            material_size: model.materials.len(),
            add_blend: model.states.enable_add_blend,
            buffer: model.vertex_buffer.clone(),
            index_buffer: model.index_buffer.clone(),
        };
        for (idx, material) in model.materials.iter().enumerate() {
            renderers.push(MaterialRenderer::new(
                effect,
                config,
                &model_ctx,
                idx,
                material,
                shadow_bind,
                device,
            ));
        }
        Self {
            effect: None,
            model_ctx,
            shadow_bind: shadow_bind.clone(),
            config: config.clone(),
            renderers,
        }
    }

    pub fn set_effect(&mut self, model: &Model, effect: &Arc<Effect>, device: &wgpu::Device) {
        self.effect = Some(effect.clone());
        for (idx, renderer) in self.renderers.iter_mut().enumerate() {
            let material = model.materials.get(idx).expect("material idx out of range");
            renderer.set_effect(material, effect, device);
        }
    }

    pub fn remove_effect(
        &mut self,
        model: &Model,
        effect: &Arc<Effect>,
        fallback_effect: &Arc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(origin_effect) = &self.effect {
            if Arc::ptr_eq(origin_effect, effect) {
                self.effect = Some(fallback_effect.clone());
                self.set_effect(model, fallback_effect, device);
                return;
            }
        }
        for (idx, renderer) in self.renderers.iter_mut().enumerate() {
            if Arc::ptr_eq(&renderer.effect, effect) {
                let material = model.materials.get(idx).expect("material idx out of range");
                renderer.set_effect(material, effect, device);
            }
        }
        todo!("remove_effect")
    }

    pub fn set_material_effect(
        &mut self,
        material_idx: usize,
        material: &Material,
        effect: &Arc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(origin_effect) = &self.effect {
            if Arc::ptr_eq(origin_effect, effect) {
                return;
            }
        }
        if let Some(renderer) = self.renderers.get_mut(material_idx) {
            renderer.set_effect(material, effect, device);
            self.effect = None;
        }
    }
}

pub struct MaterialRenderer {
    pub effect: Arc<Effect>,
    config: RendererConfig,
    model_ctx: DrawPassModelContext,
    material_idx: usize,
    shadow_bind: Arc<wgpu::BindGroup>,
    pub passes: HashMap<DrawType, DrawPass>,
}

impl MaterialRenderer {
    pub fn new(
        effect: &Arc<Effect>,
        config: &RendererConfig,
        model_ctx: &DrawPassModelContext,
        material_idx: usize,
        material: &Material,
        shadow_bind: &Arc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) -> Self {
        let mut passes = HashMap::new();
        for draw_type in &config.draw_types {
            if let Some(technique) = effect.find_technique(*draw_type) {
                let draw_pass = technique.get_draw_pass(
                    config,
                    model_ctx,
                    material_idx,
                    material,
                    shadow_bind,
                    device,
                );
                passes.insert(*draw_type, draw_pass);
            }
        }
        Self {
            effect: effect.clone(),
            config: config.clone(),
            model_ctx: model_ctx.clone(),
            material_idx,
            shadow_bind: shadow_bind.clone(),
            passes,
        }
    }

    pub fn set_effect(&mut self, material: &Material, effect: &Arc<Effect>, device: &wgpu::Device) {
        self.effect = effect.clone();
        for (draw_type, draw_pass) in self.passes.iter_mut() {
            if let Some(technique) = effect.find_technique(*draw_type) {
                *draw_pass = technique.get_draw_pass(
                    &self.config,
                    &self.model_ctx,
                    self.material_idx,
                    material,
                    &self.shadow_bind,
                    device,
                );
            }
        }
    }

    pub fn update_color_bind(&mut self, color_bind: Arc<wgpu::BindGroup>, device: &wgpu::Device) {
        for draw_pass in self.passes.values_mut() {
            draw_pass.update_color_bind(color_bind.clone(), device);
        }
    }

    pub fn update_uniform(
        &self,
        model_handle: ModelHandle,
        draw_type: DrawType,
        updater: &dyn Fn(&mut UniformBindData),
        queue: &wgpu::Queue,
    ) {
        self.effect
            .find_technique(draw_type)
            .unwrap()
            .update_uniform(model_handle, updater, queue);
    }

    fn find_pass(&self, draw_type: DrawType) -> Option<&DrawPass> {
        self.passes.get(&draw_type)
    }
}

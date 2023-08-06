use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cgmath::Vector4;

use crate::{
    model::{Material, Model},
    project::ModelHandle,
};

use super::{
    effect::Effect,
    technique::{DrawPass, DrawPassModelContext, Technique, TechniqueType},
    RenderFormat,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ObjectKey {
    Model(ModelHandle),
    Material((ModelHandle, usize)),
}

#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub format: RenderFormat,
    pub size: wgpu::Extent3d,
    pub techniques: HashSet<TechniqueType>,
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
    objects: HashMap<ObjectKey, Rc<Effect>>,
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
    pub depth: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub config: RendererConfig,
    renderers: HashMap<ModelHandle, ModelRenderer>,
    fallback_effect: Rc<Effect>,
}

impl ScreenRenderTarget {
    pub fn new(
        builder: RenderTargetBuilder,
        models: &HashMap<ModelHandle, Model>,
        shadow_bind: &Rc<wgpu::BindGroup>,
        fallback_effect: &Rc<Effect>,
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
        let depth_view = depth
            .as_ref()
            .map(|depth| depth.create_view(&wgpu::TextureViewDescriptor::default()));
        let mut rt = Self {
            clear_color: builder.clear_color,
            clear_depth: builder.clear_depth,
            depth,
            depth_view,
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
        effect: Option<&Rc<Effect>>,
        shadow_bind: &Rc<wgpu::BindGroup>,
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
        effect: &Rc<Effect>,
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
        effect: &Rc<Effect>,
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

    pub fn remove_effect(&mut self, effect: Rc<Effect>, fallback_effect: Rc<Effect>) {
        todo!("remove_effect")
    }

    pub fn update_format(
        &mut self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) {
        todo!("update_format")
    }

    pub fn depth_view(&self) -> Option<wgpu::TextureView> {
        todo!("depth")
    }

    pub fn render_bundles(
        &self,
        technique_type: TechniqueType,
    ) -> impl Iterator<Item = &wgpu::RenderBundle> {
        self.renderers
            .values()
            .flat_map(|renderer| &renderer.renderers)
            .filter_map(move |renderer| renderer.passes.get(&technique_type))
            .map(|pass| &pass.render_bundle)
    }

    pub fn draw(
        &self,
        technique_type: TechniqueType,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ScreenRenderTarget/CommandEncoder"),
        });
        let color_attachments = wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: true,
            },
        };
        {
            let mut _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ScreenRenderTarget/RenderPass"),
                color_attachments: &[Some(color_attachments)],
                depth_stencil_attachment: self.depth_view.as_ref().map(|tv| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: tv,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1f32),
                            store: true,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: true,
                        }),
                    }
                }),
            });
            _rpass.execute_bundles(self.render_bundles(technique_type));
        }
        queue.submit(Some(encoder.finish()));
    }
}

// impl ScreenRenderTarget {
//     pub fn draw_color(
//         &self,
//         models: &HashMap<u32, Model>,
//         shadow_map_enabled: bool,
//         set_uniform: impl Fn(&mut UniformBindData),
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//     ) {
//         // Clear
//         // Initialize Fallback Uniform
//         // let mut bundles = vec![];
//         for (handle, model) in models {
//             let technique_type = if shadow_map_enabled {
//                 TechniqueType::Object
//             } else {
//                 TechniqueType::ObjectSs
//             };
//             if let Some(effect) = self.objects.get(&ObjectKey::Model(*handle)) {
//                 if let Some(technique) = effect.technique.get(&technique_type) {
//                     if let Some(pass) = technique.passes.get(handle) {
//                         let mut uniform = pass.uniform_bind.get_empty_uniform_data();
//                         set_uniform(&mut uniform);
//                         pass.uniform_bind.update(&uniform, queue);
//                     }
//                 }
//             } else {
//                 let mut initialized_effect = vec![];
//                 for idx in 0..model.materials.len() {
//                     if let Some(effect) = self.objects.get(&ObjectKey::Material((*handle, idx))) {
//                         if let Some(technique) = effect.technique.get(&technique_type) {
//                             if !initialized_effect.iter().any(|e| Rc::ptr_eq(effect, e)) {
//                                 if let Some(pass) = technique.passes.get(handle) {
//                                     let mut uniform = pass.uniform_bind.get_empty_uniform_data();
//                                     set_uniform(&mut uniform);
//                                     pass.uniform_bind.update(&uniform, queue);
//                                 }
//                                 initialized_effect.push(effect.clone());
//                             }
//                         }
//                     }
//                 }
//                 drop(initialized_effect);
//             }
//         }
//     }
// }

pub struct ModelRenderer {
    pub effect: Option<Rc<Effect>>,
    pub model_ctx: DrawPassModelContext,
    pub shadow_bind: Rc<wgpu::BindGroup>,
    pub config: RendererConfig,
    pub renderers: Vec<MaterialRenderer>,
}

impl ModelRenderer {
    pub fn new(
        model_handle: ModelHandle,
        model: &Model,
        effect: &Rc<Effect>,
        shadow_bind: &Rc<wgpu::BindGroup>,
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

    pub fn set_effect(&mut self, model: &Model, effect: &Rc<Effect>, device: &wgpu::Device) {
        self.effect = Some(effect.clone());
        for (idx, renderer) in self.renderers.iter_mut().enumerate() {
            let material = model.materials.get(idx).expect("material idx out of range");
            renderer.set_effect(material, effect, device);
        }
    }

    pub fn remove_effect(
        &mut self,
        model: &Model,
        effect: &Rc<Effect>,
        fallback_effect: &Rc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(origin_effect) = &self.effect {
            if Rc::ptr_eq(origin_effect, effect) {
                self.effect = Some(fallback_effect.clone());
                self.set_effect(model, fallback_effect, device);
                return;
            }
        }
        for (idx, renderer) in self.renderers.iter_mut().enumerate() {
            if Rc::ptr_eq(&renderer.effect, effect) {
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
        effect: &Rc<Effect>,
        device: &wgpu::Device,
    ) {
        if let Some(origin_effect) = &self.effect {
            if Rc::ptr_eq(origin_effect, effect) {
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
    pub effect: Rc<Effect>,
    config: RendererConfig,
    model_ctx: DrawPassModelContext,
    material_idx: usize,
    shadow_bind: Rc<wgpu::BindGroup>,
    pub passes: HashMap<TechniqueType, DrawPass>,
}

impl MaterialRenderer {
    pub fn new(
        effect: &Rc<Effect>,
        config: &RendererConfig,
        model_ctx: &DrawPassModelContext,
        material_idx: usize,
        material: &Material,
        shadow_bind: &Rc<wgpu::BindGroup>,
        device: &wgpu::Device,
    ) -> Self {
        let mut passes = HashMap::new();
        for technique_type in &config.techniques {
            if let Some(technique) = effect.technique.get(technique_type) {
                let draw_pass = technique.get_draw_pass(
                    config,
                    model_ctx,
                    material_idx,
                    material,
                    shadow_bind,
                    device,
                );
                passes.insert(*technique_type, draw_pass);
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

    pub fn set_effect(&mut self, material: &Material, effect: &Rc<Effect>, device: &wgpu::Device) {
        self.effect = effect.clone();
        for (technique_type, draw_pass) in self.passes.iter_mut() {
            if let Some(technique) = effect.technique.get(technique_type) {
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
}

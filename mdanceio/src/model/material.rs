use std::{collections::HashMap, sync::Arc};

use cgmath::{ElementWise, Vector3, Vector4, VectorSpace};

use crate::utils::{f128_to_vec3, f128_to_vec4, lerp_f32};

use super::{MaterialIndex, NanoemMaterial, NanoemTexture};

pub struct MaterialContext<'a> {
    pub fallback_texture: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub bind_group_layout: &'a wgpu::BindGroupLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialColor {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub diffuse_opacity: f32,
    pub specular_power: f32,
    pub diffuse_texture_blend_factor: Vector4<f32>,
    pub sphere_texture_blend_factor: Vector4<f32>,
    pub toon_texture_blend_factor: Vector4<f32>,
}

impl MaterialColor {
    pub fn new_reset(v: f32) -> Self {
        Self {
            ambient: Vector3::new(v, v, v),
            diffuse: Vector3::new(v, v, v),
            specular: Vector3::new(v, v, v),
            diffuse_opacity: v,
            specular_power: v.max(Material::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: Vector4::new(v, v, v, v),
            sphere_texture_blend_factor: Vector4::new(v, v, v, v),
            toon_texture_blend_factor: Vector4::new(v, v, v, v),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MaterialBlendColor {
    base: MaterialColor,
    add: MaterialColor,
    mul: MaterialColor,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialEdge {
    pub color: Vector3<f32>,
    pub opacity: f32,
    pub size: f32,
}

impl MaterialEdge {
    pub fn new_reset(v: f32) -> Self {
        Self {
            color: Vector3::new(v, v, v),
            opacity: v,
            size: v,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MaterialBlendEdge {
    base: MaterialEdge,
    add: MaterialEdge,
    mul: MaterialEdge,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialStates {
    pub visible: bool,
    pub display_diffuse_texture_uv_mesh_enabled: bool,
    pub display_sphere_map_texture_uv_mesh_enabled: bool,
}

pub struct Material {
    // TODO
    color: MaterialBlendColor,
    edge: MaterialBlendEdge,
    diffuse_image: Option<wgpu::TextureView>,
    sphere_map_image: Option<wgpu::TextureView>,
    toon_image: Option<wgpu::TextureView>,
    texture_bind: Arc<wgpu::BindGroup>,
    pub name: String,
    pub canonical_name: String,
    pub index_offset: u32,
    pub num_indices: u32,
    index_hash: HashMap<u32, u32>,
    toon_color: Vector4<f32>,
    states: MaterialStates,
    on_bind_updated: Option<Box<dyn Fn()>>,
    pub origin: NanoemMaterial,
}

impl Material {
    pub const MINIUM_SPECULAR_POWER: f32 = 0.1f32;

    pub fn from_nanoem(
        material: &NanoemMaterial,
        language_type: nanoem::common::LanguageType,
        ctx: &mut MaterialContext,
        index_offset: u32,
        num_indices: u32,
        device: &wgpu::Device,
    ) -> Self {
        let mut name = material.get_name(language_type).to_owned();
        let mut canonical_name = material
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Material{}", material.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("Model/TextureBindGroup/Material {}", &canonical_name).as_str()),
            layout: ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
            ],
        });
        let flags = material.flags;
        Self {
            color: MaterialBlendColor {
                base: MaterialColor {
                    ambient: Vector4::from(material.get_ambient_color()).truncate(),
                    diffuse: Vector4::from(material.get_diffuse_color()).truncate(),
                    specular: Vector4::from(material.get_specular_color()).truncate(),
                    diffuse_opacity: material.get_diffuse_opacity(),
                    specular_power: material.get_specular_power(),
                    diffuse_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    sphere_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    toon_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                },
                add: MaterialColor::new_reset(0f32),
                mul: MaterialColor::new_reset(1f32),
            },
            edge: MaterialBlendEdge {
                base: MaterialEdge {
                    color: Vector4::from(material.get_edge_color()).truncate(),
                    opacity: material.get_edge_opacity(),
                    size: material.get_edge_size(),
                },
                add: MaterialEdge::new_reset(0f32),
                mul: MaterialEdge::new_reset(1f32),
            },
            diffuse_image: None,
            sphere_map_image: None,
            toon_image: None,
            texture_bind: Arc::new(bind_group),
            name,
            canonical_name,
            num_indices,
            index_offset,
            index_hash: HashMap::new(),
            toon_color: Vector4::new(1f32, 1f32, 1f32, 1f32),
            states: MaterialStates {
                visible: true,
                display_diffuse_texture_uv_mesh_enabled: true,
                ..Default::default()
            },
            on_bind_updated: None,
            origin: material.clone(),
        }
    }

    pub fn set_on_bind_updated(&mut self, f: Box<dyn Fn()>) {
        self.on_bind_updated = Some(f);
    }

    pub fn reset(&mut self) {
        let material = &self.origin;
        self.color.base.ambient = Vector4::from(material.get_ambient_color()).truncate();
        self.color.base.diffuse = Vector4::from(material.get_diffuse_color()).truncate();
        self.color.base.specular = Vector4::from(material.get_specular_color()).truncate();
        self.color.base.diffuse_opacity = material.get_diffuse_opacity();
        self.color.base.specular_power = material
            .get_specular_power()
            .max(Self::MINIUM_SPECULAR_POWER);
        self.color.base.diffuse_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.sphere_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.toon_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.edge.base.color = Vector4::from(material.get_edge_color()).truncate();
        self.edge.base.opacity = material.get_edge_opacity();
        self.edge.base.size = material.get_edge_size();
    }

    pub fn reset_deform(&mut self) {
        self.color.mul = MaterialColor::new_reset(1f32);
        self.color.add = MaterialColor::new_reset(0f32);
        self.edge.mul = MaterialEdge::new_reset(1f32);
        self.edge.add = MaterialEdge::new_reset(0f32);
    }

    pub fn deform(&mut self, morph: &nanoem::model::ModelMorphMaterial, weight: f32) {
        const ONE_V4: Vector4<f32> = Vector4 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
            w: 1f32,
        };
        const ONE_V3: Vector3<f32> = Vector3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        };
        let diffuse_texture_blend_factor = f128_to_vec4(morph.diffuse_texture_blend);
        let sphere_texture_blend_factor = f128_to_vec4(morph.sphere_map_texture_blend);
        // TODO: nanoem use sphere_map_texture_blend, it may be a mistake
        let toon_texture_blend_factor = f128_to_vec4(morph.toon_texture_blend);
        match morph.operation {
            nanoem::model::ModelMorphMaterialOperationType::Multiply => {
                self.color.mul.ambient.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.mul.diffuse.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.mul.specular.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.mul.diffuse_opacity = lerp_f32(
                    self.color.mul.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.mul.specular_power =
                    lerp_f32(self.color.mul.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .mul
                    .diffuse_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .mul
                    .sphere_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .mul
                    .toon_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .mul
                    .color
                    .mul_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.mul.opacity = lerp_f32(self.edge.mul.opacity, morph.edge_opacity, weight);
                self.edge.mul.size = lerp_f32(self.edge.mul.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Add => {
                self.color.add.ambient.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.add.diffuse.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.add.specular.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.add.diffuse_opacity = lerp_f32(
                    self.color.add.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.add.specular_power =
                    lerp_f32(self.color.add.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .add
                    .diffuse_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .add
                    .sphere_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .add
                    .toon_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .add
                    .color
                    .add_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.add.opacity = lerp_f32(self.edge.add.opacity, morph.edge_opacity, weight);
                self.edge.add.size = lerp_f32(self.edge.add.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Unknown => {}
        }
    }

    pub fn is_visible(&self) -> bool {
        self.states.visible
    }

    pub fn color(&self) -> MaterialColor {
        MaterialColor {
            ambient: self
                .color
                .base
                .ambient
                .mul_element_wise(self.color.mul.ambient)
                + self.color.add.ambient,
            diffuse: self
                .color
                .base
                .diffuse
                .mul_element_wise(self.color.mul.diffuse)
                + self.color.add.diffuse,
            specular: self
                .color
                .base
                .specular
                .mul_element_wise(self.color.mul.specular)
                + self.color.add.specular,
            diffuse_opacity: self.color.base.diffuse_opacity * self.color.mul.diffuse_opacity
                + self.color.add.diffuse_opacity,
            specular_power: (self.color.base.specular_power * self.color.mul.specular_power
                + self.color.add.specular_power)
                .min(Self::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: self
                .color
                .base
                .diffuse_texture_blend_factor
                .mul_element_wise(self.color.mul.diffuse_texture_blend_factor)
                + self.color.add.diffuse_texture_blend_factor,
            sphere_texture_blend_factor: self
                .color
                .base
                .sphere_texture_blend_factor
                .mul_element_wise(self.color.mul.sphere_texture_blend_factor)
                + self.color.add.sphere_texture_blend_factor,
            toon_texture_blend_factor: self
                .color
                .base
                .toon_texture_blend_factor
                .mul_element_wise(self.color.mul.toon_texture_blend_factor)
                + self.color.add.toon_texture_blend_factor,
        }
    }

    pub fn edge(&self) -> MaterialEdge {
        MaterialEdge {
            color: self.edge.base.color.mul_element_wise(self.edge.mul.color) + self.edge.add.color,
            opacity: self.edge.base.opacity * self.edge.mul.opacity + self.edge.add.opacity,
            size: self.edge.base.size * self.edge.mul.size + self.edge.add.size,
        }
    }

    pub fn diffuse_view(&self) -> Option<&wgpu::TextureView> {
        self.diffuse_image.as_ref()
    }

    pub fn sphere_map_view(&self) -> Option<&wgpu::TextureView> {
        self.sphere_map_image.as_ref()
    }

    pub fn toon_view(&self) -> Option<&wgpu::TextureView> {
        self.toon_image.as_ref()
    }

    pub fn update_bind(&mut self, ctx: &mut MaterialContext, device: &wgpu::Device) {
        self.texture_bind = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("Model/TextureBindGroup/Material").as_str()),
            layout: ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.diffuse_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.sphere_map_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        self.toon_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
            ],
        }));
    }

    pub fn bind_group(&self) -> Arc<wgpu::BindGroup> {
        self.texture_bind.clone()
    }

    pub fn sphere_map_texture_type(&self) -> nanoem::model::ModelMaterialSphereMapTextureType {
        self.origin.sphere_map_texture_type
    }

    pub fn is_culling_disabled(&self) -> bool {
        self.origin.flags.is_culling_disabled
    }

    pub fn is_casting_shadow_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_enabled
    }
    pub fn is_casting_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_map_enabled
    }
    pub fn is_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_shadow_map_enabled
    }
    pub fn is_edge_enabled(&self) -> bool {
        self.origin.flags.is_edge_enabled
    }
    pub fn is_vertex_color_enabled(&self) -> bool {
        self.origin.flags.is_vertex_color_enabled
    }
    pub fn is_point_draw_enabled(&self) -> bool {
        self.origin.flags.is_point_draw_enabled
    }
    pub fn is_line_draw_enabled(&self) -> bool {
        self.origin.flags.is_line_draw_enabled
    }
}

pub struct MaterialSet {
    materials: Vec<Material>,
    textures: Vec<NanoemTexture>,
}

impl MaterialSet {
    pub fn new(
        nanoem_materials: &[NanoemMaterial],
        textures: &[NanoemTexture],
        language_type: nanoem::common::LanguageType,
        draw_ctx: &mut MaterialContext,
        device: &wgpu::Device,
    ) -> Self {
        let mut materials = vec![];
        let mut index_offset = 0usize;
        for (_, material) in nanoem_materials.iter().enumerate() {
            let num_indices = material.num_vertex_indices;
            materials.push(Material::from_nanoem(
                material,
                language_type,
                draw_ctx,
                index_offset as u32,
                num_indices as u32,
                device,
            ));
            index_offset += num_indices;
        }
        Self {
            materials,
            textures: textures.to_vec(),
        }
    }

    pub fn get(&self, idx: MaterialIndex) -> Option<&Material> {
        self.materials.get(idx)
    }

    pub fn get_mut(&mut self, idx: MaterialIndex) -> Option<&mut Material> {
        self.materials.get_mut(idx)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Material> {
        self.materials.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Material> {
        self.materials.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn create_all_images(
        &mut self,
        texture_lut: &HashMap<String, wgpu::Texture>,
        draw_ctx: &mut MaterialContext,
        device: &wgpu::Device,
    ) {
        for material in &mut self.materials {
            material.diffuse_image = material
                .origin
                .get_diffuse_texture_object(&self.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.sphere_map_image = material
                .origin
                .get_sphere_map_texture_object(&self.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.toon_image = material
                .origin
                .get_toon_texture_object(&self.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.update_bind(draw_ctx, device);
        }
    }

    pub fn update_image(
        &mut self,
        texture_key: &str,
        texture: &wgpu::Texture,
        draw_ctx: &mut MaterialContext,
        device: &wgpu::Device,
    ) -> Vec<MaterialIndex> {
        let mut updated_indices = vec![];
        for (idx, material) in self.materials.iter_mut().enumerate() {
            let mut updated = false;
            if let Some(texture) = material
                .origin
                .get_diffuse_texture_object(&self.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
            {
                material.diffuse_image =
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                updated = true;
            }
            if let Some(texture) = material
                .origin
                .get_sphere_map_texture_object(&self.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
            {
                material.sphere_map_image =
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                updated = true;
            }
            if let Some(texture) = material
                .origin
                .get_toon_texture_object(&self.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
            {
                material.toon_image =
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                updated = true;
            }
            if updated {
                material.update_bind(draw_ctx, device);
                updated_indices.push(idx);
            }
        }
        updated_indices
    }
}

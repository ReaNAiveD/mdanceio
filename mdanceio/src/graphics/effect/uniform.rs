use std::{num::NonZeroU64, rc::Rc};

use bytemuck::Zeroable;
use cgmath::Matrix4;
use wgpu::util::DeviceExt;

use crate::{
    camera::{Camera, PerspectiveCamera},
    light::{DirectionalLight, Light},
    model::{Material, Model},
    shadow_camera::ShadowCamera,
};

use super::technique::TechniqueType;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniform {
    model_matrix: [[f32; 4]; 4],                 // camera, shadow
    model_view_matrix: [[f32; 4]; 4],            // camera, shadow
    model_view_projection_matrix: [[f32; 4]; 4], // camera, shadow, zplot
    light_view_projection_matrix: [[f32; 4]; 4], // zplot
    light_color: [f32; 4],                       // light, shadow
    light_direction: [f32; 4],                   // light
    camera_position: [f32; 4],                   // camera
    shadow_map_size: [f32; 4],                   // zplot
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    ambient: [f32; 4],              // material
    diffuse: [f32; 4],              // material
    specular: [f32; 4],             // material
    edge_color: [f32; 4],           // edge
    enable_vertex_color: [f32; 4],  // material
    diffuse_blend_factor: [f32; 4], // material
    sphere_blend_factor: [f32; 4],  // material
    toon_blend_factor: [f32; 4],    // material
    use_texture_sampler: [f32; 4],  // material(0-2), zplot(3)
    sphere_texture_type: [f32; 4],  // material
    edge_size: f32,                 // edge
    padding: [f32; 23], // DynamicOffset must be aligned to `min_uniform_buffer_offset_alignment`, which is 256 by default
}

pub struct UniformBindData {
    model: ModelUniform,
    material: Vec<MaterialUniform>,
    opacity: f32,
}

impl UniformBindData {
    pub fn new(material_size: usize) -> Self {
        Self {
            model: ModelUniform::zeroed(),
            material: vec![MaterialUniform::zeroed(); material_size],
            opacity: 1f32,
        }
    }

    pub fn set_camera_parameters(
        &mut self,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
        model: &Model,
    ) {
        let (v, p) = camera.get_view_transform();
        let w = model.world_transform(world);
        self.model.model_matrix = w.into();
        self.model.model_view_matrix = (v * w).into();
        self.model.model_view_projection_matrix = (p * v * w).into();
        self.model.camera_position = camera.position().extend(0f32).into();
    }

    pub fn set_light_parameters(&mut self, light: &dyn Light) {
        self.model.light_color = light.color().extend(1f32).into();
        self.model.light_direction = light.direction().extend(0f32).into();
    }

    pub fn set_all_model_parameters(
        &mut self,
        model: &Model,
        models: &dyn Iterator<Item = &Model>,
    ) {
        self.opacity = model.opacity();
    }

    pub fn set_material_parameters(&mut self, material_idx: usize, material: &Material) {
        let uniform = &mut self.material[material_idx];
        let color = material.color();
        uniform.ambient = color.ambient.extend(1.0f32).into();
        uniform.diffuse = color
            .diffuse
            .extend(color.diffuse_opacity * self.opacity)
            .into();
        uniform.specular = color.specular.extend(color.specular_power).into();
        uniform.diffuse_blend_factor = color.diffuse_texture_blend_factor.into();
        uniform.sphere_blend_factor = color.sphere_texture_blend_factor.into();
        uniform.toon_blend_factor = color.toon_texture_blend_factor.into();
        let texture_type = if material.sphere_map_view().is_some() {
            material.sphere_map_texture_type()
        } else {
            nanoem::model::ModelMaterialSphereMapTextureType::TypeNone
        };
        let sphere_texture_type = match texture_type {
            nanoem::model::ModelMaterialSphereMapTextureType::TypeMultiply => {
                [1f32, 0f32, 0f32, 0f32]
            }
            nanoem::model::ModelMaterialSphereMapTextureType::TypeSubTexture => {
                [0f32, 1f32, 0f32, 0f32]
            }
            nanoem::model::ModelMaterialSphereMapTextureType::TypeAdd => [0f32, 0f32, 1f32, 0f32],
            _ => [0f32, 0f32, 0f32, 0f32],
        };
        uniform.sphere_texture_type = sphere_texture_type;
        uniform.enable_vertex_color = if material.is_vertex_color_enabled() {
            [1f32, 1f32, 1f32, 1f32]
        } else {
            [0f32, 0f32, 0f32, 0f32]
        };
        uniform.use_texture_sampler[0] = if material.diffuse_view().is_some() {
            1f32
        } else {
            0f32
        };
        uniform.use_texture_sampler[1] = if material.sphere_map_view().is_some() {
            1f32
        } else {
            0f32
        };
        uniform.use_texture_sampler[2] = if material.toon_view().is_some() {
            1f32
        } else {
            0f32
        };
    }

    pub fn set_edge_parameters(
        &mut self,
        material_idx: usize,
        material: &Material,
        edge_size: f32,
    ) {
        let edge = material.edge();
        let edge_color = edge.color.extend(edge.opacity);
        let uniform = &mut self.material[material_idx];
        uniform.edge_color = edge_color.into();
        uniform.edge_size = edge_size * edge.size;
    }

    pub fn set_ground_shadow_parameters(
        &mut self,
        light: &dyn Light,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
    ) {
        let (view_matrix, projection_matrix) = camera.get_view_transform();
        let origin_shadow_matrix = light.get_shadow_transform();
        let shadow_matrix = origin_shadow_matrix * world;
        let shadow_view_matrix = view_matrix * shadow_matrix;
        let shadow_view_projection_matrix = projection_matrix * shadow_view_matrix;
        self.model.model_matrix = shadow_matrix.into();
        self.model.model_view_matrix = shadow_view_matrix.into();
        self.model.model_view_projection_matrix = shadow_view_projection_matrix.into();
        self.model.light_color = light
            .ground_shadow_color()
            .extend(if light.is_translucent_ground_shadow_enabled() {
                0.5f32
            } else {
                1.0f32
            })
            .into();
    }

    pub fn set_shadow_map_parameters(
        &mut self,
        shadow_camera: &ShadowCamera,
        world: &Matrix4<f32>,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
    ) {
        let (view, projection) = shadow_camera.get_view_projection(camera, light);
        let crop = shadow_camera.get_crop_matrix();
        let shadow_map_matrix = projection * view * world;
        self.model.light_view_projection_matrix = (crop * shadow_map_matrix).into();
        self.model.shadow_map_size = shadow_camera
            .image_size()
            .map(|x| x as f32)
            .extend(0.005f32)
            .extend(u32::from(shadow_camera.coverage_mode()) as f32)
            .into();
        for uniform in &mut self.material {
            uniform.use_texture_sampler[3] = if shadow_camera.is_enabled() {
                1f32
            } else {
                0f32
            };
        }
    }
}

#[derive(Debug)]
pub struct UniformBind {
    material_size: usize,
    model_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    bind_group: Rc<wgpu::BindGroup>,
}

impl UniformBind {
    pub fn new(
        bind_layout: &wgpu::BindGroupLayout,
        material_size: usize,
        device: &wgpu::Device,
    ) -> Self {
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ModelProgramBundle/BindGroupBuffer/ModelUniform"),
            contents: bytemuck::cast_slice(&[ModelUniform::zeroed()]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ModelProgramBundle/BindGroupBuffer/MaterialUniform"),
            contents: bytemuck::cast_slice(&vec![MaterialUniform::zeroed(); material_size]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ModelProgramBundle/BindGroup/Uniform"),
            layout: bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &model_buffer,
                        offset: 0,
                        size: Some(
                            NonZeroU64::new(std::mem::size_of::<ModelUniform>() as u64).unwrap(),
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &material_buffer,
                        offset: 0,
                        size: Some(
                            NonZeroU64::new(std::mem::size_of::<MaterialUniform>() as u64).unwrap(),
                        ),
                    }),
                },
            ],
        });
        Self {
            material_size,
            model_buffer,
            material_buffer,
            bind_group: Rc::new(uniform_bind_group),
        }
    }

    pub fn get_empty_uniform_data(&self) -> UniformBindData {
        UniformBindData::new(self.material_size)
    }

    pub fn update(&self, uniform_data: &UniformBindData, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.model_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data.model]),
        );
        queue.write_buffer(
            &self.material_buffer,
            0,
            bytemuck::cast_slice(&uniform_data.material),
        );
    }

    pub fn bind_group(&self) -> Rc<wgpu::BindGroup> {
        self.bind_group.clone()
    }
}

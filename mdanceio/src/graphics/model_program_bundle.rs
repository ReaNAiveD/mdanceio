use std::num::NonZeroU64;

use super::{
    common_pass::{CPassBindGroup, CPassLayout, CPassVertexBuffer},
    technique::{
        EdgeTechnique, GroundShadowTechnique, ObjectPassKey, ObjectTechnique, TechniqueType,
        ZplotTechnique, EdgePassKey, ZplotPassKey, ShadowPassKey,
    },
};
use crate::{
    camera::PerspectiveCamera,
    light::DirectionalLight,
    model::{Material, VertexUnit},
};
use bytemuck::Zeroable;
use cgmath::{Matrix4, Vector4};
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera, light::Light, model::Model, shadow_camera::ShadowCamera,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelParametersUniform {
    model_matrix: [[f32; 4]; 4],                 // camera, shadow
    model_view_matrix: [[f32; 4]; 4],            // camera, shadow
    model_view_projection_matrix: [[f32; 4]; 4], // camera, shadow, zplot
    light_view_projection_matrix: [[f32; 4]; 4], // zplot
    light_color: [f32; 4],                       // light, edge, shadow
    light_direction: [f32; 4],                   //light, edge
    camera_position: [f32; 4],                   // camera
    material_ambient: [f32; 4],                  // material
    material_diffuse: [f32; 4],                  // material
    material_specular: [f32; 4],                 // material
    enable_vertex_color: [f32; 4],               // material
    diffuse_texture_blend_factor: [f32; 4],      // material
    sphere_texture_blend_factor: [f32; 4],       // material
    toon_texture_blend_factor: [f32; 4],         // material
    use_texture_sampler: [f32; 4],               // material(0-2), zplot(3)
    sphere_texture_type: [f32; 4],               // material
    shadow_map_size: [f32; 4],                   // zplot
    padding: [f32; 12],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CommonPassCacheKey {
    cull_mode: Option<wgpu::Face>,
    primitive_type: wgpu::PrimitiveTopology,
    technique_type: TechniqueType,
    is_add_blend: bool,
    is_depth_enabled: bool,
    is_offscreen_render_pass_active: bool,
}

#[derive(Debug, Clone)]
pub struct PassExecuteConfiguration<'a> {
    pub technique_type: TechniqueType,
    pub viewport_texture_format: wgpu::TextureFormat,
    pub is_render_pass_viewport: bool,
    pub texture_bind_layout: &'a wgpu::BindGroupLayout,
    pub shadow_bind_layout: &'a wgpu::BindGroupLayout,
}

pub struct UniformBindData {
    data: Vec<ModelParametersUniform>,
    opacity: f32,
}

impl UniformBindData {
    pub fn new(material_size: usize) -> Self {
        Self {
            data: vec![ModelParametersUniform::zeroed(); material_size],
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
        for uniform in &mut self.data {
            uniform.model_matrix = w.into();
            uniform.model_view_matrix = (v * w).into();
            uniform.model_view_projection_matrix = (p * v * w).into();
            uniform.camera_position = camera.position().extend(0f32).into();
        }
    }

    pub fn set_light_parameters(&mut self, light: &dyn Light) {
        for uniform in &mut self.data {
            uniform.light_color = light.color().extend(1f32).into();
            uniform.light_direction = light.direction().extend(0f32).into();
        }
    }

    pub fn set_all_model_parameters(
        &mut self,
        model: &Model,
        models: &dyn Iterator<Item = &Model>,
    ) {
        self.opacity = model.opacity();
    }

    pub fn set_material_parameters(&mut self, material_idx: usize, material: &Material) {
        let uniform = &mut self.data[material_idx];
        let color = material.color();
        uniform.material_ambient = color.ambient.extend(1.0f32).into();
        uniform.material_diffuse = color
            .diffuse
            .extend(color.diffuse_opacity * self.opacity)
            .into();
        uniform.material_specular = color.specular.extend(color.specular_power).into();
        uniform.diffuse_texture_blend_factor = color.diffuse_texture_blend_factor.into();
        uniform.sphere_texture_blend_factor = color.sphere_texture_blend_factor.into();
        uniform.toon_texture_blend_factor = color.toon_texture_blend_factor.into();
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
        let uniform = &mut self.data[material_idx];
        uniform.light_color = edge_color.into();
        uniform.light_direction =
            (Vector4::new(1f32, 1f32, 1f32, 1f32) * edge.size * edge_size).into();
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
        for uniform in &mut self.data {
            uniform.model_matrix = shadow_matrix.into();
            uniform.model_view_matrix = shadow_view_matrix.into();
            uniform.model_view_projection_matrix = shadow_view_projection_matrix.into();
            uniform.light_color = light
                .ground_shadow_color()
                .extend(
                    1.0f32
                        + if light.is_translucent_ground_shadow_enabled() {
                            -0.5f32
                        } else {
                            0f32
                        },
                )
                .into();
        }
    }

    pub fn set_shadow_map_parameters(
        &mut self,
        shadow_camera: &ShadowCamera,
        world: &Matrix4<f32>,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
        technique_type: TechniqueType,
    ) {
        let (view, projection) = shadow_camera.get_view_projection(camera, light);
        let crop = shadow_camera.get_crop_matrix();
        let shadow_map_matrix = projection * view * world;
        for uniform in &mut self.data {
            uniform.light_view_projection_matrix = (crop * shadow_map_matrix).into();
            uniform.shadow_map_size = shadow_camera
                .image_size()
                .map(|x| x as f32)
                .extend(0.005f32)
                .extend(u32::from(shadow_camera.coverage_mode()) as f32)
                .into();
            uniform.use_texture_sampler[3] = if shadow_camera.is_enabled() {
                1f32
            } else {
                0f32
            };
            if let TechniqueType::Zplot = technique_type {
                uniform.model_view_projection_matrix = shadow_map_matrix.into();
            }
        }
    }
}

#[derive(Debug)]
pub struct UniformBind {
    material_size: usize,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_layout: wgpu::BindGroupLayout,
}

impl UniformBind {
    pub fn new(material_size: usize, device: &wgpu::Device) -> Self {
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ModelProgramBundle/BindGroupLayout/Uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<ModelParametersUniform>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                }],
            });
        let uniform_buffer_data = vec![ModelParametersUniform::zeroed(); material_size];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ModelProgramBundle/BindGroupBuffer/Uniform"),
            contents: bytemuck::cast_slice(&uniform_buffer_data[..]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ModelProgramBundle/BindGroup/Uniform"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: Some(
                        NonZeroU64::new(std::mem::size_of::<ModelParametersUniform>() as u64)
                            .unwrap(),
                    ),
                }),
            }],
        });
        Self {
            material_size,
            buffer: uniform_buffer,
            bind_group: uniform_bind_group,
            bind_layout: uniform_bind_group_layout,
        }
    }

    pub fn get_empty_uniform_data(&self) -> UniformBindData {
        UniformBindData::new(self.material_size)
    }

    pub fn update(&self, uniform_data: &UniformBindData, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&uniform_data.data[..]),
        );
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_layout
    }
}

pub struct ModelRenderLayout {
    pub color_texture_format: wgpu::TextureFormat,
    pub depth_texture_format: wgpu::TextureFormat,
    pub color_bind_layout: wgpu::BindGroupLayout,
    pub uniform_bind_layout: wgpu::BindGroupLayout,
    pub shadow_bind_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub vertex_buffer_layout: wgpu::VertexBufferLayout<'static>,
}

impl ModelRenderLayout {
    pub fn new(
        color_texture_format: wgpu::TextureFormat,
        depth_texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Self {
        let color_bind_layout = Self::build_color_bind_layout(device);
        let uniform_bind_layout = Self::build_uniform_bind_layout(device);
        let shadow_bind_layout = Self::build_shadow_bind_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ModelProgramBundle/PipelineLayout"),
            bind_group_layouts: &[
                &color_bind_layout,
                &uniform_bind_layout,
                &shadow_bind_layout,
            ],
            push_constant_ranges: &[],
        });
        let vertex_buffer_layout = Self::build_vertex_buffer_layout(device);
        Self {
            color_texture_format,
            depth_texture_format,
            color_bind_layout,
            uniform_bind_layout,
            shadow_bind_layout,
            pipeline_layout,
            vertex_buffer_layout,
        }
    }
}

impl ModelRenderLayout {
    fn build_color_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ModelProgramBundle/BindGroupLayout/Color"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn build_shadow_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ModelProgramBundle/BindGroupLayout/Shadow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
    }

    fn build_uniform_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ModelProgramBundle/BindGroupLayout/Uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(
                        NonZeroU64::new(std::mem::size_of::<ModelParametersUniform>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        })
    }

    fn build_vertex_buffer_layout(device: &wgpu::Device) -> wgpu::VertexBufferLayout<'static> {
        let vertex_size = std::mem::size_of::<VertexUnit>();
        wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 2,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 3,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 4,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 5,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 6,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress * 7,
                    shader_location: 7,
                },
            ],
        }
    }
}

pub struct ModelProgramBundle {
    layout: ModelRenderLayout,
    object_technique: ObjectTechnique,
    edge_technique: EdgeTechnique,
    ground_shadow_technique: GroundShadowTechnique,
    zplot_technique: ZplotTechnique,
}

impl ModelProgramBundle {
    pub fn new(
        color_texture_format: wgpu::TextureFormat,
        depth_texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Self {
        let layout = ModelRenderLayout::new(color_texture_format, depth_texture_format, device);
        Self {
            layout,
            object_technique: ObjectTechnique::new(device),
            edge_technique: EdgeTechnique::new(device),
            ground_shadow_technique: GroundShadowTechnique::new(device),
            zplot_technique: ZplotTechnique::new(device),
        }
    }

    pub fn ensure_get_object_render_bundle(
        &mut self,
        render_config: ObjectPassKey,
        material_idx: usize,
        bind: CPassBindGroup,
        vertex: CPassVertexBuffer,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let cpass = self.object_technique.ensure(
            &CPassLayout {
                pipeline_layout: &self.layout.pipeline_layout,
                vertex_buffer_layout: self.layout.vertex_buffer_layout.clone(),
            },
            render_config,
            device,
        );
        cpass.build_render_bundle(material_idx, bind, vertex, device)
    }

    pub fn ensure_get_edge_render_bundle(
        &mut self,
        render_config: EdgePassKey,
        material_idx: usize,
        bind: CPassBindGroup,
        vertex: CPassVertexBuffer,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let cpass = self.edge_technique.ensure(
            &CPassLayout {
                pipeline_layout: &self.layout.pipeline_layout,
                vertex_buffer_layout: self.layout.vertex_buffer_layout.clone(),
            },
            render_config,
            device,
        );
        cpass.build_render_bundle(material_idx, bind, vertex, device)
    }

    pub fn ensure_get_shadow_render_bundle(
        &mut self,
        render_config: ShadowPassKey,
        material_idx: usize,
        bind: CPassBindGroup,
        vertex: CPassVertexBuffer,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let cpass = self.ground_shadow_technique.ensure(
            &CPassLayout {
                pipeline_layout: &self.layout.pipeline_layout,
                vertex_buffer_layout: self.layout.vertex_buffer_layout.clone(),
            },
            render_config,
            device,
        );
        cpass.build_render_bundle(material_idx, bind, vertex, device)
    }

    pub fn ensure_get_zplot_render_bundle(
        &mut self,
        render_config: ZplotPassKey,
        material_idx: usize,
        bind: CPassBindGroup,
        vertex: CPassVertexBuffer,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let cpass = self.zplot_technique.ensure(
            &CPassLayout {
                pipeline_layout: &self.layout.pipeline_layout,
                vertex_buffer_layout: self.layout.vertex_buffer_layout.clone(),
            },
            render_config,
            device,
        );
        cpass.build_render_bundle(material_idx, bind, vertex, device)
    }
}
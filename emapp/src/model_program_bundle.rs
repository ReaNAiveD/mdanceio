use std::{cell::RefCell, collections::HashMap, iter, mem, ops::Deref, rc::Rc};

use crate::{
    camera::PerspectiveCamera, light::DirectionalLight, model::Material, technique::Technique,
};
use bytemuck::Zeroable;
use cgmath::{Matrix4, Vector4};
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    drawable::Drawable,
    image_view::ImageView,
    light::Light,
    model::{Model, NanoemMaterial},
    pass,
    project::Project,
    shadow_camera::ShadowCamera,
};

// enum UniformBuffer {
//     ModelMatrix = 0,
//     ModelViewMatrix = 4,
//     ModelViewProjectionMatrix = 8,
//     LightViewProjectionMatrix = 12,
//     LightColor = 16,
//     LightDirection,
//     CameraPosition,
//     MaterialAmbient,
//     MaterialDiffuse,
//     MaterialSpecular,
//     EnableVertexColor,
//     DiffuseTextureBlendFactor,
//     SphereTextureBlendFactor,
//     ToonTextureBlendFactor,
//     UseTextureSampler,
//     SphereTextureType,
//     ShadowMapSize,
// }

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelParametersUniform {
    model_matrix: [[f32; 4]; 4],
    model_view_matrix: [[f32; 4]; 4],
    model_view_projection_matrix: [[f32; 4]; 4],
    light_view_projection_matrix: [[f32; 4]; 4],
    light_color: [f32; 4],
    light_direction: [f32; 4],
    camera_position: [f32; 4],
    material_ambient: [f32; 4],
    material_diffuse: [f32; 4],
    material_specular: [f32; 4],
    enable_vertex_color: [f32; 4],
    diffuse_texture_blend_factor: [f32; 4],
    sphere_texture_blend_factor: [f32; 4],
    toon_texture_blend_factor: [f32; 4],
    use_texture_sampler: [f32; 4],
    sphere_texture_type: [f32; 4],
    shadow_map_size: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSamplerStage {
    ShadowMapTextureSamplerStage0,
    DiffuseTextureSamplerStage,
    SphereTextureSamplerStage,
    ToonTextureSamplerStage,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassExecuteConfiguration {
    pub technique_type: TechniqueType,
    pub viewport_texture_format: wgpu::TextureFormat,
    pub is_render_pass_viewport: bool,
}

pub struct CommonPass {
    // TODO: uncompleted
    shader: wgpu::ShaderModule,
    uniform_buffer: ModelParametersUniform,
    bindings: HashMap<TextureSamplerStage, wgpu::TextureView>,
    cull_mode: Option<wgpu::Face>,
    primitive_type: wgpu::PrimitiveTopology,
    opacity: f32,
    pipeline_cache: RefCell<HashMap<CommonPassCacheKey, wgpu::RenderPipeline>>,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
}

impl CommonPass {
    pub fn new(device: &wgpu::Device, shader: wgpu::ShaderModule) -> Self {
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ModelProgramBundle/BindGroupLayout/Texture"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ModelProgramBundle/BindGroupLayout/Uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self {
            shader,
            uniform_buffer: ModelParametersUniform::zeroed(),
            bindings: HashMap::new(),
            cull_mode: None,
            primitive_type: wgpu::PrimitiveTopology::TriangleList,
            opacity: 1.0f32,
            pipeline_cache: RefCell::new(HashMap::new()),
            texture_bind_group_layout,
            uniform_bind_group_layout,
        }
    }
}

impl CommonPass {
    pub fn set_global_parameters(&mut self, _drawable: &impl Drawable) {}

    pub fn set_camera_parameters(
        &mut self,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
        model: &Model,
    ) {
        let (v, p) = camera.get_view_transform();
        let w = model.world_transform(world);
        self.uniform_buffer.model_matrix = w.into();
        self.uniform_buffer.model_view_matrix = (v * w).into();
        self.uniform_buffer.model_view_projection_matrix = (p * v * w).into();
        self.uniform_buffer.camera_position = camera.position().extend(0f32).into();
    }

    pub fn set_light_parameters(&mut self, light: &dyn Light, _adjustment: bool) {
        self.uniform_buffer.light_color = light.color().extend(1f32).into();
        self.uniform_buffer.light_direction = light.direction().extend(0f32).into();
    }

    pub fn set_all_model_parameters(
        &mut self,
        model: &Model,
        models: &dyn Iterator<Item = &Model>,
    ) {
        self.opacity = model.opacity();
    }

    pub fn set_material_parameters(
        &mut self,
        material: &Material,
        technique_type: TechniqueType,
        fallback: &wgpu::Texture,
    ) {
        let color = material.color();
        self.uniform_buffer.material_ambient = color.ambient.extend(1.0f32).into();
        self.uniform_buffer.material_diffuse = color
            .diffuse
            .extend(color.diffuse_opacity * self.opacity)
            .into();
        self.uniform_buffer.material_specular = color.specular.extend(color.specular_power).into();
        self.uniform_buffer.diffuse_texture_blend_factor =
            color.diffuse_texture_blend_factor.into();
        self.uniform_buffer.sphere_texture_blend_factor = color.sphere_texture_blend_factor.into();
        self.uniform_buffer.toon_texture_blend_factor = color.toon_texture_blend_factor.into();
        let texture_type = if material.sphere_map_image().is_some() {
            material.spheremap_texture_type()
        } else {
            nanoem::model::ModelMaterialSphereMapTextureType::TypeNone
        };
        let sphere_texture_type = [
            if texture_type == nanoem::model::ModelMaterialSphereMapTextureType::TypeMultiply {
                1.0f32
            } else {
                0.0f32
            },
            if texture_type == nanoem::model::ModelMaterialSphereMapTextureType::TypeSubTexture {
                1.0f32
            } else {
                0.0f32
            },
            if texture_type == nanoem::model::ModelMaterialSphereMapTextureType::TypeAdd {
                1.0f32
            } else {
                0.0f32
            },
            0f32,
        ];
        self.uniform_buffer.sphere_texture_type = sphere_texture_type;
        let enable_vertex_color = if material.is_vertex_color_enabled() {
            1.0f32
        } else {
            0.0f32
        };
        self.uniform_buffer.enable_vertex_color = Vector4::new(
            enable_vertex_color,
            enable_vertex_color,
            enable_vertex_color,
            enable_vertex_color,
        )
        .into();
        self.uniform_buffer.use_texture_sampler[0] = if self.set_image(
            material.diffuse_image(),
            TextureSamplerStage::DiffuseTextureSamplerStage,
            fallback,
        ) {
            1.0f32
        } else {
            0.0f32
        };
        self.uniform_buffer.use_texture_sampler[1] = if self.set_image(
            material.sphere_map_image(),
            TextureSamplerStage::SphereTextureSamplerStage,
            fallback,
        ) {
            1.0f32
        } else {
            0.0f32
        };
        self.uniform_buffer.use_texture_sampler[2] = if self.set_image(
            material.toon_image(),
            TextureSamplerStage::ToonTextureSamplerStage,
            fallback,
        ) {
            1.0f32
        } else {
            0.0f32
        };
        if material.is_line_draw_enabled() {
            self.primitive_type = wgpu::PrimitiveTopology::LineList;
        } else if material.is_point_draw_enabled() {
            self.primitive_type = wgpu::PrimitiveTopology::PointList;
        } else {
            self.primitive_type = wgpu::PrimitiveTopology::TriangleList;
        }
        self.cull_mode = match technique_type {
            TechniqueType::Color | TechniqueType::Zplot => {
                if material.is_culling_disabled() {
                    None
                } else {
                    Some(wgpu::Face::Back)
                }
            }
            TechniqueType::Edge => Some(wgpu::Face::Front),
            TechniqueType::Shadow => None,
        }
    }

    pub fn set_edge_parameters(
        &mut self,
        material: &Material,
        edge_size: f32,
        fallback: &wgpu::Texture,
    ) {
        let material = material;
        let edge = material.edge();
        let edge_color = edge.color.extend(edge.opacity);
        self.uniform_buffer.light_color = edge_color.into();
        self.uniform_buffer.light_direction =
            (Vector4::new(1f32, 1f32, 1f32, 1f32) * edge.size * edge_size).into();
        self.bindings.insert(
            TextureSamplerStage::ShadowMapTextureSamplerStage0,
            fallback.create_view(&wgpu::TextureViewDescriptor::default()),
        );
    }

    pub fn set_ground_shadow_parameters(
        &mut self,
        light: &dyn Light,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
        fallback: &wgpu::Texture,
    ) {
        let (view_matrix, projection_matrix) = camera.get_view_transform();
        let origin_shadow_matrix = light.get_shadow_transform();
        let shadow_matrix = origin_shadow_matrix * world;
        self.uniform_buffer.model_matrix = shadow_matrix.into();
        let shadow_view_matrix = view_matrix * shadow_matrix;
        self.uniform_buffer.model_view_matrix = shadow_view_matrix.into();
        let shadow_view_projection_matrix = projection_matrix * shadow_view_matrix;
        self.uniform_buffer.model_view_projection_matrix = shadow_view_projection_matrix.into();
        self.uniform_buffer.light_color = light
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
        self.bindings.insert(
            TextureSamplerStage::ShadowMapTextureSamplerStage0,
            fallback.create_view(&wgpu::TextureViewDescriptor::default()),
        );
    }

    pub fn set_shadow_map_parameters(
        &mut self,
        shadow_camera: &ShadowCamera,
        world: &Matrix4<f32>,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
        technique_type: TechniqueType,
        fallback: &wgpu::Texture,
    ) {
        let (view, projection) = shadow_camera.get_view_projection(camera, light);
        let crop = shadow_camera.get_crop_matrix();
        let shadow_map_matrix = projection * view * world;
        self.uniform_buffer.light_view_projection_matrix = (crop * shadow_map_matrix).into();
        self.uniform_buffer.shadow_map_size = shadow_camera
            .image_size()
            .map(|x| x as f32)
            .extend(0.005f32)
            .extend(u32::from(shadow_camera.coverage_mode()) as f32)
            .into();
        self.uniform_buffer.use_texture_sampler[3] = if shadow_camera.is_enabled() {
            1.0f32
        } else {
            0f32
        };
        let color_image = match technique_type {
            TechniqueType::Zplot => {
                self.bindings.insert(
                    TextureSamplerStage::DiffuseTextureSamplerStage,
                    fallback.create_view(&wgpu::TextureViewDescriptor::default()),
                );
                self.bindings.insert(
                    TextureSamplerStage::SphereTextureSamplerStage,
                    fallback.create_view(&wgpu::TextureViewDescriptor::default()),
                );
                self.bindings.insert(
                    TextureSamplerStage::ToonTextureSamplerStage,
                    fallback.create_view(&wgpu::TextureViewDescriptor::default()),
                );
                self.uniform_buffer.model_view_projection_matrix = shadow_map_matrix.into();
                fallback.create_view(&wgpu::TextureViewDescriptor::default())
            }
            _ => shadow_camera
                .color_image()
                .create_view(&wgpu::TextureViewDescriptor::default()),
        };
        self.bindings.insert(
            TextureSamplerStage::ShadowMapTextureSamplerStage0,
            color_image,
        );
    }

    // TODO: process with feature
    // #[cfg(target_feature = "enable_blendop_minmax")]
    fn get_add_blend_state(&self) -> (wgpu::BlendState, wgpu::ColorWrites) {
        (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add, // default
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Max,
                },
            },
            wgpu::ColorWrites::ALL,
        )
    }

    // TODO: process with feature
    // #[cfg(target_feature = "enable_blendop_minmax")]
    fn get_alpha_blend_state(&self) -> (wgpu::BlendState, wgpu::ColorWrites) {
        (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add, // default
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Max,
                },
            },
            wgpu::ColorWrites::ALL,
        )
    }

    fn get_texture_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ModelProgramBundle/BindGroup/Texture"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .bindings
                            .get(&TextureSamplerStage::ShadowMapTextureSamplerStage0)
                            .unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .bindings
                            .get(&TextureSamplerStage::DiffuseTextureSamplerStage)
                            .unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .bindings
                            .get(&TextureSamplerStage::SphereTextureSamplerStage)
                            .unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .bindings
                            .get(&TextureSamplerStage::ToonTextureSamplerStage)
                            .unwrap(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        return texture_bind_group;
    }

    pub fn execute(
        &mut self,
        buffer: &pass::Buffer,
        color_attachment_view: &wgpu::TextureView,
        depth_stencil_attachment_view: Option<&wgpu::TextureView>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        config: &PassExecuteConfiguration,
    ) {
        let is_add_blend = model.is_add_blend_enabled();
        let is_depth_enabled = buffer.is_depth_enabled();
        let key = CommonPassCacheKey {
            cull_mode: self.cull_mode,
            primitive_type: self.primitive_type,
            technique_type: config.technique_type,
            is_add_blend,
            is_depth_enabled,
            is_offscreen_render_pass_active: false,
        };
        let mut cache = self.pipeline_cache.borrow_mut();
        let pipeline = cache.entry(key).or_insert_with(|| {
            let vertex_size = mem::size_of::<crate::model::VertexUnit>();
            let texture_format = if config.technique_type == TechniqueType::Zplot {
                wgpu::TextureFormat::R32Float
            } else {
                config.viewport_texture_format
            };

            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ModelProgramBundle/PipelineLayout"),
                    bind_group_layouts: &[
                        &self.texture_bind_group_layout,
                        &self.uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
            // No Difference between technique type edge and other.
            let vertex_buffer_layout = wgpu::VertexBufferLayout {
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
            };
            // Project::setStandardDepthStencilState(desc.depth, desc.stencil);
            // in origin project
            let (blend_state, mut write_mask) = if is_add_blend {
                self.get_add_blend_state()
            } else {
                self.get_alpha_blend_state()
            };
            if config.is_render_pass_viewport {
                write_mask = wgpu::ColorWrites::ALL
            };
            let color_target_state = wgpu::ColorTargetState {
                format: texture_format,
                blend: if config.technique_type == TechniqueType::Zplot {
                    None
                } else {
                    Some(blend_state)
                },
                write_mask,
            };
            let depth_state = if config.technique_type == TechniqueType::Shadow && is_depth_enabled
            {
                wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    // stencil: wgpu::StencilState {
                    //     front: wgpu::StencilFaceState {
                    //         compare: wgpu::CompareFunction::Greater,
                    //         fail_op: wgpu::StencilOperation::default(),
                    //         depth_fail_op: wgpu::StencilOperation::default(),
                    //         pass_op: wgpu::StencilOperation::Replace,
                    //     },
                    //     back: wgpu::StencilFaceState {
                    //         compare: wgpu::CompareFunction::Greater,
                    //         fail_op: wgpu::StencilOperation::default(),
                    //         depth_fail_op: wgpu::StencilOperation::default(),
                    //         pass_op: wgpu::StencilOperation::Replace,
                    //     },
                    //     read_mask: 0,
                    //     write_mask: 0, // TODO: there was a ref=2 in original stencil state
                    // },
                    stencil: wgpu::StencilState::default(), // TODO
                    bias: wgpu::DepthBiasState::default(),
                }
            } else {
                wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8, // TODO: set to depth pixel format
                    depth_write_enabled: is_depth_enabled,
                    depth_compare: if is_depth_enabled {
                        wgpu::CompareFunction::LessEqual
                    } else {
                        wgpu::CompareFunction::Always
                    },
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ModelProgramBundle/Pipelines"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_buffer_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: "fs_main",
                    targets: &[Some(color_target_state)],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: self.primitive_type,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: self.cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(depth_state),
                multisample: wgpu::MultisampleState {
                    count: 1, // TODO: be configured by pixel format
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        });
        let texture_bind_group =
            self.get_texture_bind_group(&device, &self.texture_bind_group_layout);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ModelProgramBundle/BindGroupBuffer/Uniform"),
            contents: bytemuck::bytes_of(&[self.uniform_buffer]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ModelProgramBundle/BindGroup/Uniform"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model Pass Executor Encoder"),
        });
        encoder.push_debug_group("ModelProgramBundle::execute");
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Model Pass Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_attachment_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: depth_stencil_attachment_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            // m_lastDrawnRenderPass = handle;
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &texture_bind_group, &[]);
            rpass.set_bind_group(1, &uniform_bind_group, &[]);
            rpass.set_vertex_buffer(0, buffer.vertex_buffer.slice(..));
            rpass.set_index_buffer(buffer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(
                buffer.num_offset as u32..(buffer.num_offset + buffer.num_indices) as u32,
                0,
                0..1,
            );
        }
        encoder.pop_debug_group();
        queue.submit(iter::once(encoder.finish()));
    }

    pub fn set_image(
        &mut self,
        value: Option<&wgpu::Texture>,
        stage: TextureSamplerStage,
        fallback: &wgpu::Texture,
    ) -> bool {
        self.bindings.insert(
            stage,
            value.clone().map_or(
                fallback.create_view(&wgpu::TextureViewDescriptor::default()),
                |rc| rc.create_view(&wgpu::TextureViewDescriptor::default()),
            ),
        );
        value.is_some()
    }
}

pub struct ModelProgramBundle {
    object_technique: Box<ObjectTechnique>,
    object_technique_point_draw: Box<ObjectTechnique>,
    edge_technique: Box<EdgeTechnique>,
    ground_shadow_technique: Box<GroundShadowTechnique>,
    zplot_technique: Box<ZplotTechnique>,
}

impl ModelProgramBundle {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            object_technique: Box::new(ObjectTechnique::new(device, false)),
            object_technique_point_draw: Box::new(ObjectTechnique::new(device, true)),
            edge_technique: Box::new(EdgeTechnique::new(device)),
            ground_shadow_technique: Box::new(GroundShadowTechnique::new(device)),
            zplot_technique: Box::new(ZplotTechnique::new(device)),
        }
    }

    pub fn find_technique(
        &mut self,
        technique_type: &str,
        material: &Material,
        material_index: usize,
        num_material: usize,
        model_name: &str,
    ) -> Option<&mut (dyn Technique)> {
        TechniqueType::from_str(technique_type).map(|typ| match typ {
            TechniqueType::Color => {
                if material.is_point_draw_enabled() {
                    self.object_technique_point_draw.base.executed = false;
                    self.object_technique_point_draw.as_mut() as &mut (dyn Technique)
                } else {
                    self.object_technique.base.executed = false;
                    self.object_technique.as_mut() as &mut (dyn Technique)
                }
            }
            TechniqueType::Edge => {
                self.edge_technique.base.executed = false;
                self.edge_technique.as_mut() as &mut (dyn Technique)
            }
            TechniqueType::Shadow => {
                self.ground_shadow_technique.base.executed = false;
                self.ground_shadow_technique.as_mut() as &mut (dyn Technique)
            }
            TechniqueType::Zplot => {
                self.zplot_technique.base.executed = false;
                self.zplot_technique.as_mut() as &mut (dyn Technique)
            }
        })
    }
}

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

pub struct BaseTechnique {
    technique_type: TechniqueType,
    executed: bool,
    pass: CommonPass,
}

impl Technique for BaseTechnique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass> {
        None
    }

    fn reset_script_command_state(&self) {}

    fn reset_script_external_color(&self) {}

    fn has_next_script_command(&self) -> bool {
        false
    }
}

pub struct ObjectTechnique {
    base: BaseTechnique,
    is_point_draw_enabled: bool,
}

impl Deref for ObjectTechnique {
    type Target = BaseTechnique;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ObjectTechnique {
    pub fn new(device: &wgpu::Device, is_point_draw_enabled: bool) -> Self {
        log::trace!("Load model_color.wgsl");
        let sd = wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelColor"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("resources/shaders/model_color.wgsl").into(),
            ),
        };
        let shader = device.create_shader_module(sd);
        log::trace!("Finish Load model_color.wgsl");
        Self {
            is_point_draw_enabled,
            base: BaseTechnique {
                technique_type: TechniqueType::Color,
                executed: false,
                pass: CommonPass::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for ObjectTechnique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass> {
        if !self.base.executed {
            self.base.executed = true;
            Some(&mut self.base.pass)
        } else {
            None
        }
    }

    fn reset_script_command_state(&self) {
        self.base.reset_script_command_state()
    }

    fn reset_script_external_color(&self) {
        self.base.reset_script_external_color()
    }

    fn has_next_script_command(&self) -> bool {
        self.base.has_next_script_command()
    }
}

pub struct EdgeTechnique {
    base: BaseTechnique,
}

impl Deref for EdgeTechnique {
    type Target = BaseTechnique;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl EdgeTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        log::trace!("Load model_edge.wgsl");
        let sd = wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelEdge"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("resources/shaders/model_edge.wgsl").into(),
            ),
        };
        let shader = device.create_shader_module(sd);
        log::trace!("Finish Load model_edge.wgsl");
        Self {
            base: BaseTechnique {
                technique_type: TechniqueType::Edge,
                executed: false,
                pass: CommonPass::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for EdgeTechnique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass> {
        if !self.base.executed {
            self.base.executed = true;
            Some(&mut self.base.pass)
        } else {
            None
        }
    }

    fn reset_script_command_state(&self) {
        self.base.reset_script_command_state()
    }

    fn reset_script_external_color(&self) {
        self.base.reset_script_external_color()
    }

    fn has_next_script_command(&self) -> bool {
        self.base.has_next_script_command()
    }
}

pub struct GroundShadowTechnique {
    base: BaseTechnique,
}

impl Deref for GroundShadowTechnique {
    type Target = BaseTechnique;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl GroundShadowTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        log::trace!("Load model_zplot.wgsl");
        let sd = wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelGroundShadow"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("resources/shaders/model_ground_shadow.wgsl").into(),
            ),
        };
        let shader = device.create_shader_module(sd);
        log::trace!("Finish Load model_ground_shader.wgsl");
        Self {
            base: BaseTechnique {
                technique_type: TechniqueType::Shadow,
                executed: false,
                pass: CommonPass::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for GroundShadowTechnique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass> {
        if !self.base.executed {
            self.base.executed = true;
            Some(&mut self.base.pass)
        } else {
            None
        }
    }

    fn reset_script_command_state(&self) {
        self.base.reset_script_command_state()
    }

    fn reset_script_external_color(&self) {
        self.base.reset_script_external_color()
    }

    fn has_next_script_command(&self) -> bool {
        self.base.has_next_script_command()
    }
}

pub struct ZplotTechnique {
    base: BaseTechnique,
}

impl Deref for ZplotTechnique {
    type Target = BaseTechnique;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ZplotTechnique {
    pub fn new(device: &wgpu::Device) -> Self {
        log::trace!("Load model_zplot.wgsl");
        let sd = wgpu::ShaderModuleDescriptor {
            label: Some("ModelProgramBundle/ObjectTechnique/ModelZplot"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("resources/shaders/model_zplot.wgsl").into(),
            ),
        };
        let shader = device.create_shader_module(sd);
        log::trace!("Finish Load model_zplot.wgsl");
        Self {
            base: BaseTechnique {
                technique_type: TechniqueType::Zplot,
                executed: false,
                pass: CommonPass::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for ZplotTechnique {
    fn execute(&mut self, device: &wgpu::Device) -> Option<&mut CommonPass> {
        if !self.base.executed {
            self.base.executed = true;
            Some(&mut self.base.pass)
        } else {
            None
        }
    }

    fn reset_script_command_state(&self) {
        self.base.reset_script_command_state()
    }

    fn reset_script_external_color(&self) {
        self.base.reset_script_external_color()
    }

    fn has_next_script_command(&self) -> bool {
        self.base.has_next_script_command()
    }
}

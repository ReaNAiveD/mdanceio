use std::{
    cell::RefCell, collections::HashMap, iter, mem, num::NonZeroU64, ops::Deref, rc::Rc,
    time::Instant,
};

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
    padding: [f32; 12],
}

impl ModelParametersUniform {}

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

#[derive(Debug, Clone)]
pub struct PassExecuteConfiguration<'a> {
    pub technique_type: TechniqueType,
    pub viewport_texture_format: wgpu::TextureFormat,
    pub is_render_pass_viewport: bool,
    pub texture_bind_layout: &'a wgpu::BindGroupLayout,
    pub shadow_bind_layout: &'a wgpu::BindGroupLayout,
}

#[derive(Debug)]
pub struct UniformBindCache {
    data: Vec<ModelParametersUniform>,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_layout: wgpu::BindGroupLayout,
    /// If dirty, data is not synchronized with buffer.
    dirty: bool,
}

impl UniformBindCache {
    pub fn new(device: &wgpu::Device) -> Self {
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
        let uniform_buffer_data = vec![ModelParametersUniform::zeroed(); 25];
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
            data: uniform_buffer_data,
            buffer: uniform_buffer,
            bind_group: uniform_bind_group,
            bind_layout: uniform_bind_group_layout,
            dirty: false,
        }
    }

    pub fn resize(&mut self, new_size: usize, device: &wgpu::Device) {
        if new_size <= self.data.len() {
            return;
        }
        let mut extended_size = self.data.len() * 3 / 2 + 1;
        if new_size > extended_size {
            extended_size = new_size;
        }
        self.data
            .extend(vec![ModelParametersUniform::zeroed(); extended_size - self.data.len()].iter());
        self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ModelProgramBundle/BindGroupBuffer/Uniform"),
            contents: bytemuck::cast_slice(&self.data[..]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ModelProgramBundle/BindGroup/Uniform"),
            layout: &self.bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buffer.as_entire_binding(),
            }],
        });
        self.dirty = false;
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.data[..]));
        self.dirty = false;
    }

    pub fn get_uniform_mut(
        &mut self,
        index: usize,
        device: &wgpu::Device,
    ) -> &mut ModelParametersUniform {
        if index >= self.data.len() {
            self.resize(index + 1, device);
        }
        self.dirty = true;
        self.data.get_mut(index).unwrap()
    }

    pub fn bind_group(&mut self, queue: &wgpu::Queue) -> &wgpu::BindGroup {
        if self.dirty {
            self.update(queue);
        }
        &self.bind_group
    }

    pub fn bind_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_layout
    }
}

pub struct CommonPassCache {
    // TODO: uncompleted
    shader: wgpu::ShaderModule,
    uniform_cache: RefCell<UniformBindCache>,
    // shadow_sampler: wgpu::Sampler,
    // sampler: wgpu::Sampler,
    pipeline_cache: RefCell<HashMap<CommonPassCacheKey, wgpu::RenderPipeline>>,
    // uniform_bind_group_layout: wgpu::BindGroupLayout,
}

impl CommonPassCache {
    pub fn new(device: &wgpu::Device, shader: wgpu::ShaderModule) -> Self {
        Self {
            shader,
            uniform_cache: RefCell::new(UniformBindCache::new(device)),
            pipeline_cache: RefCell::new(HashMap::new()),
            // texture_bind_group_layout,
            // uniform_bind_group_layout,
            // shadow_sampler,
            // sampler,
        }
    }
}

pub struct CommonPass<'a> {
    cache: &'a CommonPassCache,
    material_index: usize,
    cull_mode: Option<wgpu::Face>,
    primitive_type: wgpu::PrimitiveTopology,
    opacity: f32,
    shadow_bind: &'a wgpu::BindGroup,
    texture_bind: &'a wgpu::BindGroup,
    fallback_shadow_bind: &'a wgpu::BindGroup,
    fallback_texture_bind: &'a wgpu::BindGroup,
}

impl<'a> CommonPass<'a> {
    pub fn new<'b: 'a, 'c: 'a>(
        cache: &'b CommonPassCache,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Self {
        Self {
            cache,
            material_index,
            cull_mode: None,
            primitive_type: wgpu::PrimitiveTopology::TriangleList,
            opacity: 1.0f32,
            shadow_bind: fallback_shadow_bind,
            texture_bind: fallback_texture_bind,
            fallback_shadow_bind,
            fallback_texture_bind,
        }
    }

    pub fn set_global_parameters(&mut self, _drawable: &impl Drawable) {}

    pub fn set_camera_parameters(
        &mut self,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
        model: &Model,
        device: &wgpu::Device,
    ) {
        let (v, p) = camera.get_view_transform();
        let w = model.world_transform(world);
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        uniform_buffer_data.model_matrix = w.into();
        uniform_buffer_data.model_view_matrix = (v * w).into();
        uniform_buffer_data.model_view_projection_matrix = (p * v * w).into();
        uniform_buffer_data.camera_position = camera.position().extend(0f32).into();
    }

    pub fn set_light_parameters(
        &mut self,
        light: &dyn Light,
        _adjustment: bool,
        device: &wgpu::Device,
    ) {
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        uniform_buffer_data.light_color = light.color().extend(1f32).into();
        uniform_buffer_data.light_direction = light.direction().extend(0f32).into();
    }

    pub fn set_all_model_parameters(
        &mut self,
        model: &Model,
        models: &dyn Iterator<Item = &Model>,
    ) {
        self.opacity = model.opacity();
    }

    pub fn set_material_parameters<'b: 'a>(
        &mut self,
        material: &'b Material,
        technique_type: TechniqueType,
        device: &wgpu::Device,
    ) {
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        let color = material.color();
        uniform_buffer_data.material_ambient = color.ambient.extend(1.0f32).into();
        uniform_buffer_data.material_diffuse = color
            .diffuse
            .extend(color.diffuse_opacity * self.opacity)
            .into();
        uniform_buffer_data.material_specular = color.specular.extend(color.specular_power).into();
        uniform_buffer_data.diffuse_texture_blend_factor =
            color.diffuse_texture_blend_factor.into();
        uniform_buffer_data.sphere_texture_blend_factor = color.sphere_texture_blend_factor.into();
        uniform_buffer_data.toon_texture_blend_factor = color.toon_texture_blend_factor.into();
        let texture_type = if material.sphere_map_view().is_some() {
            material.sphere_map_texture_type()
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
        uniform_buffer_data.sphere_texture_type = sphere_texture_type;
        let enable_vertex_color = if material.is_vertex_color_enabled() {
            1.0f32
        } else {
            0.0f32
        };
        uniform_buffer_data.enable_vertex_color = Vector4::new(
            enable_vertex_color,
            enable_vertex_color,
            enable_vertex_color,
            enable_vertex_color,
        )
        .into();
        uniform_buffer_data.use_texture_sampler[0] = if let Some(_) = material.diffuse_view() {
            1f32
        } else {
            0f32
        };
        uniform_buffer_data.use_texture_sampler[1] = if let Some(_) = material.sphere_map_view() {
            1f32
        } else {
            0f32
        };
        uniform_buffer_data.use_texture_sampler[2] = if let Some(_) = material.toon_view() {
            1f32
        } else {
            0f32
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

    pub fn set_material_texture<'b: 'a>(&mut self, material: &'b Material) {
        self.texture_bind = material.bind_group();
    }

    pub fn set_edge_parameters<'b: 'a>(
        &mut self,
        material: &'b Material,
        edge_size: f32,
        device: &wgpu::Device,
    ) {
        let material = material;
        let edge = material.edge();
        let edge_color = edge.color.extend(edge.opacity);
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        uniform_buffer_data.light_color = edge_color.into();
        uniform_buffer_data.light_direction =
            (Vector4::new(1f32, 1f32, 1f32, 1f32) * edge.size * edge_size).into();
    }

    pub fn set_ground_shadow_parameters<'b: 'a>(
        &mut self,
        light: &dyn Light,
        camera: &dyn Camera,
        world: &Matrix4<f32>,
        device: &wgpu::Device,
    ) {
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        let (view_matrix, projection_matrix) = camera.get_view_transform();
        let origin_shadow_matrix = light.get_shadow_transform();
        let shadow_matrix = origin_shadow_matrix * world;
        uniform_buffer_data.model_matrix = shadow_matrix.into();
        let shadow_view_matrix = view_matrix * shadow_matrix;
        uniform_buffer_data.model_view_matrix = shadow_view_matrix.into();
        let shadow_view_projection_matrix = projection_matrix * shadow_view_matrix;
        uniform_buffer_data.model_view_projection_matrix = shadow_view_projection_matrix.into();
        uniform_buffer_data.light_color = light
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

    pub fn set_shadow_map_parameters<'b: 'a>(
        &mut self,
        shadow_camera: &'b ShadowCamera,
        world: &Matrix4<f32>,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
        technique_type: TechniqueType,
        device: &wgpu::Device,
    ) {
        let (view, projection) = shadow_camera.get_view_projection(camera, light);
        let crop = shadow_camera.get_crop_matrix();
        let shadow_map_matrix = projection * view * world;
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_buffer_data = uniform_bind.get_uniform_mut(self.material_index, device);
        uniform_buffer_data.light_view_projection_matrix = (crop * shadow_map_matrix).into();
        uniform_buffer_data.shadow_map_size = shadow_camera
            .image_size()
            .map(|x| x as f32)
            .extend(0.005f32)
            .extend(u32::from(shadow_camera.coverage_mode()) as f32)
            .into();
        uniform_buffer_data.use_texture_sampler[3] = if shadow_camera.is_enabled() {
            1f32
        } else {
            0f32
        };
        match technique_type {
            TechniqueType::Zplot => {
                uniform_buffer_data.model_view_projection_matrix = shadow_map_matrix.into();
            }
            _ => {}
        };
    }

    pub fn set_shadow_map_texture<'b: 'a>(
        &mut self,
        shadow_camera: &'b ShadowCamera,
        technique_type: TechniqueType,
    ) {
        let color_image = match technique_type {
            TechniqueType::Zplot => {
                self.texture_bind = self.fallback_texture_bind;
                self.fallback_shadow_bind
            }
            _ => shadow_camera.bind_group(),
        };
        self.shadow_bind = color_image;
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

    pub fn execute(
        &mut self,
        material_idx: usize,
        buffer: &pass::Buffer,
        color_attachment_view: &wgpu::TextureView,
        depth_stencil_attachment_view: Option<&wgpu::TextureView>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        model: &Model,
        config: &PassExecuteConfiguration,
    ) {
        log::info!("Start Executing");
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
        log::info!("Getting Pipeline");
        let mut cache = self.cache.pipeline_cache.borrow_mut();
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
                        config.texture_bind_layout,
                        &self.cache.uniform_cache.borrow().bind_layout(),
                        config.shadow_bind_layout,
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
                    module: &self.cache.shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_buffer_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.cache.shader,
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

        // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //     label: Some("Model Pass Executor Encoder"),
        // });
        encoder.push_debug_group("ModelProgramBundle::execute");
        let mut uniform_bind = self.cache.uniform_cache.borrow_mut();
        let uniform_bind_group = uniform_bind.bind_group(queue);
        {
            log::info!("Begin Render Pass");
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
            log::info!("Setting Pipeline");
            rpass.set_pipeline(&pipeline);
            log::info!("Setting Bind Group");
            rpass.set_bind_group(0, self.texture_bind, &[]);

            rpass.set_bind_group(
                1,
                uniform_bind_group,
                &[(material_idx * std::mem::size_of::<ModelParametersUniform>()) as u32],
            );
            rpass.set_bind_group(2, self.shadow_bind, &[]);
            log::info!("Setting Vertex Buffer");
            rpass.set_vertex_buffer(0, buffer.vertex_buffer.slice(..));
            log::info!("Setting Index Buffer");
            rpass.set_index_buffer(buffer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            log::info!("Draw Indexed");
            rpass.draw_indexed(
                buffer.num_offset as u32..(buffer.num_offset + buffer.num_indices) as u32,
                0,
                0..1,
            );
        }
        encoder.pop_debug_group();
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
    pass_cache: CommonPassCache,
}

impl Technique for BaseTechnique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Option<CommonPass<'a>> {
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
                pass_cache: CommonPassCache::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for ObjectTechnique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Option<CommonPass<'a>> {
        if !self.base.executed {
            self.base.executed = true;
            Some(CommonPass::new(
                &self.pass_cache,
                material_index,
                fallback_shadow_bind,
                fallback_texture_bind,
            ))
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
                pass_cache: CommonPassCache::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for EdgeTechnique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Option<CommonPass<'a>> {
        if !self.base.executed {
            self.base.executed = true;
            Some(CommonPass::new(
                &self.pass_cache,
                material_index,
                fallback_shadow_bind,
                fallback_texture_bind,
            ))
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
                pass_cache: CommonPassCache::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for GroundShadowTechnique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Option<CommonPass<'a>> {
        if !self.base.executed {
            self.base.executed = true;
            Some(CommonPass::new(
                &self.pass_cache,
                material_index,
                fallback_shadow_bind,
                fallback_texture_bind,
            ))
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
                pass_cache: CommonPassCache::new(device, shader),
            },
        }
    }

    pub fn technique_type(&self) -> TechniqueType {
        self.base.technique_type
    }
}

impl Technique for ZplotTechnique {
    fn execute<'a, 'b: 'a>(
        &'b mut self,
        material_index: usize,
        fallback_shadow_bind: &'a wgpu::BindGroup,
        fallback_texture_bind: &'a wgpu::BindGroup,
    ) -> Option<CommonPass<'a>> {
        if !self.base.executed {
            self.base.executed = true;
            Some(CommonPass::new(
                &self.pass_cache,
                material_index,
                fallback_shadow_bind,
                fallback_texture_bind,
            ))
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

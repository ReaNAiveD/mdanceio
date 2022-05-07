use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cgmath::{Vector2, Vector4};

use crate::{
    accessory::Accessory,
    accessory_program_bundle::AccessoryProgramBundle,
    audio_player::AudioPlayer,
    background_video_renderer::BackgroundVideoRenderer,
    camera::{Camera, PerspectiveCamera},
    debug_capture::DebugCapture,
    drawable::{DrawType, Drawable},
    effect::{
        self, common::RenderPassScope, global_uniform::GlobalUniform, Effect, ScriptOrderType,
    },
    error::Error,
    event_publisher::EventPublisher,
    file_manager::FileManager,
    file_utils,
    forward::QuadVertexUnit,
    grid::Grid,
    image_loader::ImageLoader,
    image_view::ImageView,
    injector::Injector,
    internal::{BlitPass, ClearPass, DebugDrawer},
    light::{DirectionalLight, Light},
    model::{BindPose, Model, SkinDeformer, VisualizationClause},
    model_program_bundle::ModelProgramBundle,
    motion::Motion,
    physics_engine::PhysicsEngine,
    pixel_format::PixelFormat,
    primitive_2d::Primitive2d,
    progress::CancelPublisher,
    shadow_camera::ShadowCamera,
    time_line_segment::TimeLineSegment,
    track::Track,
    translator::{LanguageType, Translator},
    undo::UndoStack,
    uri::Uri,
};

pub trait Confirmor {
    fn seek(&mut self, frame_index: u32, project: &Project);
    fn play(&mut self, project: &Project);
    fn resume(&mut self, project: &Project);
}

pub trait RendererCapability {
    fn suggested_sample_level(&self) -> u32;
    fn supports_sample_level(&self, value: u32) -> bool;
}

pub trait SharedCancelPublisherFactory {
    fn cancel_publisher(&self) -> Rc<RefCell<dyn CancelPublisher>>;
}

pub trait SharedDebugCaptureFactory {
    fn debug_factory(&self) -> Rc<RefCell<dyn DebugCapture>>;
}

pub trait SharedResourceFactory {
    fn accessory_program_bundle(&self) -> &AccessoryProgramBundle;
    fn model_program_bundle(&self) -> &ModelProgramBundle;
    fn effect_global_uniform(&self) -> &GlobalUniform;
    fn toon_image(&self, value: i32) -> &dyn ImageView;
    fn toon_color(&self, value: i32) -> Vector4<f32>;
}

pub trait SkinDeformerFactory {
    fn create(&self, model: Rc<Ref<Model>>) -> Rc<dyn SkinDeformer>;
    fn begin(&self);
    fn end(&self);
}

pub struct SaveState {}

pub struct DrawQueue {}

pub struct BatchDrawQueue {}

pub struct SerialDrawQueue {}

pub enum EditingMode {}

pub enum FilePathMode {}

pub enum CursorType {}

pub struct RenderPassBundle {}

pub struct SharedRenderTargetImageContainer {}

struct Pass {
    name: String,
    color_texture: wgpu::Texture,
    depth_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
}

impl Pass {
    pub fn new(
        name: &str,
        size: Vector2<u16>,
        color_texture_format: wgpu::TextureFormat,
        sample_count: u32,
        device: &wgpu::Device,
    ) -> Self {
        let (color_texture, depth_texture, sampler) =
            Self::_update(name, size, color_texture_format, sample_count, device);
        Self {
            name: name.to_owned(),
            color_texture,
            depth_texture,
            sampler,
        }
    }

    pub fn update(&mut self, size: Vector2<u16>, device: &wgpu::Device, project: &Project) {
        let (color_texture, depth_texture, sampler) = Self::_update(
            self.name.as_str(),
            size,
            project.viewport_texture_format(),
            project.sample_count(),
            device,
        );
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.sampler = sampler;
    }

    fn _update(
        name: &str,
        size: Vector2<u16>,
        color_texture_format: wgpu::TextureFormat,
        sample_count: u32,
        device: &wgpu::Device,
    ) -> (wgpu::Texture, wgpu::Texture, wgpu::Sampler) {
        // TODO: Feature Query For msaa?
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(format!("{}/ColorTexture", name).as_str()),
            size: wgpu::Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: color_texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(format!("{}/DepthTexture", name).as_str()),
            size: wgpu::Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        let common_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(format!("{}/Sampler", name).as_str()),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::default(),
            ..Default::default()
        });
        (color_texture, depth_texture, common_sampler)
    }
}

struct FpsUnit {}

struct OffscreenRenderTargetCondition {}

pub struct Project {
    // background_video_renderer: Box<dyn BackgroundVideoRenderer<Error>>,
    // confirmor: Box<dyn Confirmor>,
    // file_manager: Box<dyn FileManager>,
    // event_publisher: Box<dyn EventPublisher>,
    // primitive_2d: Box<dyn Primitive2d>,
    // renderer_capability: Box<dyn RendererCapability>,
    // shared_cancel_publisher_factory: Box<dyn SharedCancelPublisherFactory>,
    // shared_resource_factory: Box<dyn SharedResourceFactory>,
    // translator: Box<dyn Translator>,
    // shared_image_loader: Option<Rc<ImageLoader>>,
    // transform_model_order_list: Vec<Rc<RefCell<Model>>>,
    // active_model_pair: (Option<Rc<RefCell<Model>>>, Option<Rc<RefCell<Model>>>),
    // active_accessory: Option<Rc<RefCell<Accessory>>>,
    // audio_player: Option<Box<dyn AudioPlayer>>,
    // physics_engine: Option<Rc<PhysicsEngine>>,
    camera: PerspectiveCamera,
    light: DirectionalLight,
    // grid: Option<Rc<Grid>>,
    // camera_motion: Rc<RefCell<Motion>>,
    // light_motion: Rc<RefCell<Motion>>,
    // self_shadow_motion: Rc<RefCell<Motion>>,
    shadow_camera: ShadowCamera,
    // undo_stack: Rc<RefCell<UndoStack>>,
    // all_models: Vec<Rc<RefCell<Model>>>,
    // all_accessories: Vec<Rc<RefCell<Accessory>>>,
    // all_motions: Vec<Rc<RefCell<Motion>>>,
    // drawable_to_motion_ptrs: HashMap<Rc<RefCell<dyn Drawable>>, Rc<RefCell<Motion>>>,
    // all_traces: Vec<Rc<RefCell<dyn Track>>>,
    // selected_track: Option<Rc<RefCell<dyn Track>>>,
    // last_save_state: Option<SaveState>,
    // draw_queue: Box<DrawQueue>,
    // batch_draw_queue: Box<BatchDrawQueue>,
    // serial_draw_queue: Box<SerialDrawQueue>,
    // offscreen_render_pass_scope: Box<RenderPassScope>,
    // viewport_pass_blitter: Box<BlitPass>,
    // render_pass_blitter: Box<BlitPass>,
    // shared_image_blitter: Box<BlitPass>,
    // render_pass_clear: Box<ClearPass>,
    // shared_debug_drawer: Rc<RefCell<DebugDrawer>>,
    viewport_texture_format: (wgpu::TextureFormat, wgpu::TextureFormat),
    // last_bind_pose: BindPose,
    // rigid_body_visualization_clause: VisualizationClause,
    draw_type: DrawType,
    // file_uri: (Uri, file_utils::TransientPath),
    // redo_file_uri: Uri,
    // drawables_to_attach_offscreen_render_target_effect: (String, HashSet<Rc<RefCell<dyn Drawable>>>),
    // current_render_pass: Option<Rc<RenderPass<'a>>>,
    // last_drawn_render_pass: Option<Rc<RenderPass<'a>>>,
    // current_offscreen_render_pass: Option<Rc<RenderPass<'a>>>,
    // origin_offscreen_render_pass: Option<Rc<RenderPass<'a>>>,
    // script_external_render_pass: Option<Rc<RenderPass<'a>>>,
    // shared_render_target_image_containers: HashMap<String, SharedRenderTargetImageContainer>,
    // editing_mode: EditingMode,
    // file_path_mode: FilePathMode,
    // playing_segment: TimeLineSegment,
    // selection_segment: TimeLineSegment,
    // base_duration: u32,
    language: LanguageType,
    // uniform_viewport_layout_rect: (Vector4<u16>, Vector4<u16>),
    // uniform_viewport_image_size: (Vector2<u16>, Vector2<u16>),
    // background_video_rect: Vector4<i32>,
    // bone_selection_rect: Vector4<i32>,
    // logical_scale_cursor_positions: HashMap<CursorType, Vector4<i32>>,
    // logical_scale_moving_cursor_position: Vector2<i32>,
    // scroll_delta: Vector2<i32>,
    // window_size: Vector2<u16>,
    // viewport_image_size: Vector2<u16>,
    // viewport_padding: Vector2<u16>,
    viewport_background_color: Vector4<f32>,
    // all_offscreen_render_targets: HashMap<Rc<RefCell<Effect>>, HashMap<String, Vec<OffscreenRenderTargetCondition>>>,
    fallback_texture: wgpu::Texture,
    // // TODO: bx::HandleAlloc *m_objectHandleAllocator;
    // accessory_handle_map: HashMap<u16, Rc<RefCell<Accessory>>>,
    // model_handle_map: HashMap<u16, Rc<RefCell<Model>>>,
    // motion_handle_map: HashMap<u16, Rc<RefCell<Motion>>>,
    // render_pass_bundle_map: HashMap<u32, RenderPassBundle>,
    // hashed_render_pass_bundle_map: HashMap<u32, Rc<RefCell<RenderPassBundle>>>,
    // redo_object_handles: HashMap<u16, u32>,
    // render_pass_string_map: HashMap<u32, String>,
    // render_pipeline_string_map: HashMap<u32, String>,
    viewport_primary_pass: Pass,
    viewport_secondary_pass: Pass,
    // context_2d_pass: Pass,
    // background_image: (Texture, Vector2<u16>),
    // preferred_motion_fps: FpsUnit,
    // editing_fps: u32,
    // bone_interpolation_type: i32,
    // camera_interpolation_type: i32,
    // model_clipboard: Vec<u8>,
    // motion_clipboard: Vec<u8>,
    // effect_order_set: HashMap<effect::ScriptOrderType, HashSet<Rc<RefCell<dyn Drawable>>>>,
    // effect_references: HashMap<String, (Rc<RefCell<Effect>>, i32)>,
    // loaded_effect_set: HashSet<Rc<RefCell<Effect>>>,
    depends_on_script_external: Vec<Box<dyn Drawable>>,
    // transform_performed_at: (u32, i32),
    // indices_of_material_to_attach_effect: (u16, HashSet<usize>),
    // window_device_pixel_ratio: (f32, f32),
    // viewport_device_pixel_ratio: (f32, f32),
    // uptime: (f64, f64),
    local_frame_index: (u32, u32),
    // time_step_factor: f32,
    // background_video_scale_factor: f32,
    // circle_radius: f32,
    sample_level: (u32, u32),
    // state_flags: u64,
    // confirm_seek_flags: u64,
    // last_physics_debug_flags: u32,
    // coordination_system: u32,
    // cursor_modifiers: u32,
    // actual_fps: u32,
    // actual_sequence: u32,
    // active: bool,
    tmp_model: Option<Box<Model>>,
    tmp_texture_map: HashMap<String, Rc<wgpu::Texture>>,
}

impl Project {
    pub const MINIMUM_BASE_DURATION: u32 = 300;
    pub const MAXIMUM_BASE_DURATION: u32 = i32::MAX as u32;
    pub const DEFAULT_CIRCLE_RADIUS_SIZE: f32 = 7.5f32;

    pub const REDO_LOG_FILE_EXTENSION: &'static str = "redo";
    pub const ARCHIVED_NATIVE_FORMAT_FILE_EXTENSION: &'static str = "nma";
    pub const FILE_SYSTEM_BASED_NATIVE_FORMAT_FILE_EXTENSION: &'static str = "nmm";
    pub const POLYGON_MOVIE_MAKER_FILE_EXTENSION: &'static str = "pmm";
    pub const VIEWPORT_PRIMARY_NAME: &'static str = "@mdanceio/Viewport/Primary";
    pub const VIEWPORT_SECONDARY_NAME: &'static str = "@mdanceio/Viewport/Secondary";

    pub fn new(
        sc_desc: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        injector: Injector,
    ) -> Self {
        log::trace!("Start Creating new Project");
        let viewport_primary_pass = Pass::new(
            Self::VIEWPORT_PRIMARY_NAME,
            Vector2::new(sc_desc.width as u16, sc_desc.height as u16),
            injector.texture_format(),
            1,
            &device,
        );

        let viewport_secondary_pass = Pass::new(
            Self::VIEWPORT_SECONDARY_NAME,
            Vector2::new(sc_desc.width as u16, sc_desc.height as u16),
            injector.texture_format(),
            1,
            &device,
        );
        log::trace!("Finish Primary and Secondary Pass");

        let fallback_texture = Self::create_fallback_image(&device, &queue);

        log::trace!("Finish Fallback texture");

        let camera = PerspectiveCamera::new();
        let shadow_camera = ShadowCamera::new(&device);
        let directional_light = DirectionalLight::new();

        Self {
            language: LanguageType::English,
            draw_type: DrawType::Color,
            depends_on_script_external: vec![],
            viewport_texture_format: (injector.texture_format(), injector.texture_format()),
            viewport_background_color: Vector4::new(0f32, 0f32, 0f32, 1f32),
            local_frame_index: (0, 0),
            sample_level: (0u32, 0u32),
            camera,
            shadow_camera,
            light: directional_light,
            fallback_texture,
            viewport_primary_pass,
            viewport_secondary_pass,
            tmp_model: None,
            tmp_texture_map: HashMap::new(),
        }
    }

    pub fn parse_language(&self) -> nanoem::common::LanguageType {
        match self.language {
            LanguageType::Japanese
            | LanguageType::ChineseSimplified
            | LanguageType::ChineseTraditional
            | LanguageType::Korean => nanoem::common::LanguageType::Japanese,
            LanguageType::English => nanoem::common::LanguageType::English,
        }
    }

    pub fn sample_count(&self) -> u32 {
        1 << self.sample_level.0
    }

    pub fn sample_level(&self) -> u32 {
        self.sample_level.0
    }

    pub fn global_camera(&self) -> &dyn Camera {
        &self.camera
    }

    pub fn active_camera(&self) -> &dyn Camera {
        // TODO: may use model camera
        &self.camera
    }

    pub fn shadow_camera(&self) -> &ShadowCamera {
        &self.shadow_camera
    }

    pub fn global_light(&self) -> &dyn Light {
        &self.light
    }

    pub fn shared_fallback_image(&self) -> &wgpu::Texture {
        &self.fallback_texture
    }

    pub fn viewport_texture_format(&self) -> wgpu::TextureFormat {
        self.viewport_texture_format.0
    }

    pub fn viewport_primary_depth_view(&self) -> wgpu::TextureView {
        self.viewport_primary_pass
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn is_render_pass_viewport(&self) -> bool {
        // TODO
        true
    }

    pub fn current_color_attachment_texture(&self) -> Option<&wgpu::TextureView> {
        None
    }

    // TODO: the origin is found from render_pass_bundle
    pub fn find_render_pass_pixel_format(&self, sample_count: u32) -> PixelFormat {
        PixelFormat::new(sample_count)
    }

    // pub fn set_transform_performed_at(&mut self, value: (u32, i32)) {
    //     self.transform_performed_at = value
    // }

    // pub fn reset_transform_performed_at(&mut self) {
    //     self.set_transform_performed_at((Motion::MAX_KEYFRAME_INDEX, 0))
    // }

    // pub fn duration(&self, base_duration: u32) -> u32 {
    //     let mut duration = base_duration.clamp(Self::MINIMUM_BASE_DURATION, Self::MAXIMUM_BASE_DURATION);
    //     if let Ok(motion) = self.camera_motion.try_borrow() {
    //         duration = duration.max(motion.duration())
    //     }
    //     if let Ok(motion) = self.light_motion.try_borrow() {
    //         duration = duration.max(motion.duration())
    //     }
    //     for motion in self.drawable_to_motion_ptrs.values() {
    //         if let Ok(motion) = motion.try_borrow() {
    //             duration = duration.max(motion.duration())
    //         }
    //     }
    //     duration
    // }

    // pub fn project_duration(&self) -> u32 {
    //     self.duration(self.base_duration)
    // }

    // pub fn find_model_by_name(&self, name: &String) -> Option<Rc<RefCell<Model>>> {
    //     self.transform_model_order_list.iter().find(|rc| rc.borrow().get_name() == name || rc.borrow().get_canonical_name() == name).map(|rc| rc.clone())
    // }

    // pub fn resolve_bone(&self, value: &(String, String)) -> Option<Rc<RefCell<nanoem::model::ModelBone>>> {
    //     if let Some(model) = self.find_model_by_name(&value.0) {
    //         return  model.borrow().find_bone(&value.1)
    //     }
    //     None
    // }

    // pub fn create_camera()
}

impl Project {
    pub fn load_tmp_model(&mut self, model_data: &[u8], device: &wgpu::Device) {
        if let Ok(model) = Model::new_from_bytes(model_data, self, 0, device) {
            self.tmp_model = Some(Box::new(model));
        }
    }

    pub fn load_texture(
        &mut self,
        key: &str,
        data: &[u8],
        dimensions: (u32, u32),
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = self
            .tmp_texture_map
            .entry(key.to_owned())
            .or_insert(Rc::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some(format!("Texture/{}", key).as_str()),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            })));
        // TODO: may have different size when different image with same name
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn update_bind_texture(&mut self) {
        self.tmp_model.as_mut().unwrap().create_all_images(&self.tmp_texture_map);
    }

    fn create_fallback_image(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let texture_size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("@mdanceio/FallbackImage"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0x00u8, 0x00u8, 0x00u8, 0xffu8],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4),
                rows_per_image: std::num::NonZeroU32::new(1),
            },
            texture_size,
        );
        fallback_texture
    }

    fn has_any_depends_on_script_external_effect(&self) -> bool {
        self.depends_on_script_external
            .iter()
            .any(|drawable| drawable.is_visible())
    }

    fn draw_all_effects_depends_on_script_external(
        &self,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // TODO: now use device from arg
        if self.has_any_depends_on_script_external_effect() {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.push_debug_group(
                format!(
                    "Project::drawDependsOnScriptExternal(size={})",
                    self.depends_on_script_external.len()
                )
                .as_str(),
            );
            for drawable in &self.depends_on_script_external {
                drawable.draw(
                    view,
                    DrawType::ScriptExternalColor,
                    self,
                    device,
                    queue,
                    adapter.get_info(),
                );
            }
            encoder.pop_debug_group();
        }
    }

    fn clear_view_port_primary_pass(
        &self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.push_debug_group("Project::clearViewportPass");
        let format = self.find_render_pass_pixel_format(self.sample_count());
        let depth_stencil_attachment_view = &self.viewport_primary_depth_view();
        {
            let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("@mdanceio/ClearPass/Vertices"),
                    contents: bytemuck::cast_slice(&QuadVertexUnit::generate_quad_tri_strip()),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            );
            let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("@mdanceio/ClearPass/Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("resources/shaders/clear.wgsl").into(),
                ),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("@mdanceio/ClearPass/PipelineLayout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
            let vertex_buffer_layout = wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertexUnit>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4],
            };
            let color_target_state = format
                .color_texture_formats
                .iter()
                .map(|format| wgpu::ColorTargetState {
                    format: format.clone(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })
                .collect::<Vec<_>>();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("@mdanceio/ClearPass/Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_buffer_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &color_target_state[..],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(wgpu::IndexFormat::Uint32),
                    cull_mode: Some(wgpu::Face::Back),
                    front_face: wgpu::FrontFace::Ccw,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("@mdanceio/ClearRenderPass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.viewport_background_color[0] as f64,
                            g: self.viewport_background_color[1] as f64,
                            b: self.viewport_background_color[2] as f64,
                            a: self.viewport_background_color[3] as f64,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_stencil_attachment_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    // stencil_ops: Some(wgpu::Operations {
                    //     load: wgpu::LoadOp::Clear(1),
                    //     store: false,
                    // }),
                    stencil_ops: None,
                }),
            });
            _render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            _render_pass.set_pipeline(&pipeline);
            _render_pass.draw(0..4, 0..1)
        }
        encoder.pop_debug_group();
        queue.submit(Some(encoder.finish()));
    }

    pub fn draw_viewport(
        &mut self,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.push_debug_group("Project::draw_viewport");
        let is_drawing_color_type = self.draw_type == DrawType::Color;
        let (view_matrix, projection_matrix) = self.active_camera().get_view_transform();
        self.draw_all_effects_depends_on_script_external(view, adapter, device, queue);
        self.clear_view_port_primary_pass(view, device, queue);
        if is_drawing_color_type {
            // TODO: 渲染后边的部分
        }
        self._draw_viewport(
            ScriptOrderType::Standard,
            self.draw_type,
            view,
            adapter,
            device,
            queue,
        );
        if is_drawing_color_type {
            // TODO: 渲染前边部分
        }
        self.local_frame_index.1 = 0;
        encoder.pop_debug_group();
        queue.submit(Some(encoder.finish()));
    }

    fn _draw_viewport(
        &self,
        order: ScriptOrderType,
        typ: DrawType,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if let Some(drawable) = &self.tmp_model {
            drawable.draw(view, typ, self, device, queue, adapter.get_info());
        }
    }
}

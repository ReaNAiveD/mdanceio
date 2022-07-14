use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cgmath::{ElementWise, Vector2, Vector3, Vector4, VectorSpace};

use crate::{
    accessory::Accessory,
    accessory_program_bundle::AccessoryProgramBundle,
    audio_player::AudioPlayer,
    background_video_renderer::BackgroundVideoRenderer,
    camera::{Camera, PerspectiveCamera},
    clear_pass::ClearPass,
    debug_capture::DebugCapture,
    drawable::{DrawType, Drawable},
    effect::{self, common::RenderPassScope, global_uniform::GlobalUniform, Effect, ScriptOrder},
    error::Error,
    event_publisher::EventPublisher,
    file_manager::FileManager,
    file_utils,
    forward::QuadVertexUnit,
    grid::Grid,
    image_loader::ImageLoader,
    image_view::ImageView,
    injector::Injector,
    internal::{BlitPass, DebugDrawer},
    light::{DirectionalLight, Light},
    model::{BindPose, Bone, Model, SkinDeformer, VisualizationClause},
    model_program_bundle::ModelProgramBundle,
    motion::Motion,
    physics_engine::{PhysicsEngine, RigidBodyFollowBone, SimulationMode, SimulationTiming},
    pixel_format::PixelFormat,
    primitive_2d::Primitive2d,
    progress::CancelPublisher,
    shadow_camera::ShadowCamera,
    time_line_segment::TimeLineSegment,
    track::Track,
    translator::{LanguageType, Translator},
    undo::UndoStack,
    uri::Uri,
    utils::lerp_f32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditingMode {
    None,
    Select,
    Move,
    Rotate,
}

pub enum FilePathMode {}

pub enum CursorType {}

#[derive(Debug, Clone, Copy, Default)]
struct HandleAllocator(u32);

impl HandleAllocator {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn next(&mut self) -> u32 {
        self.0 += 1;
        self.0
    }

    pub fn clear(&mut self) {
        self.0 = 0u32;
    }
}

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

#[derive(Debug, Clone, Copy)]
struct FpsUnit {
    value: f32,
    scale_factor: f32,
    inverted_value: f32,
    inverted_scale_factor: f32,
}

impl FpsUnit {
    pub const HALF_BASE_FPS: u32 = 30u32;
    pub const HALF_BASE_FPS_F32: f32 = Self::HALF_BASE_FPS as f32;

    pub fn new(value: f32) -> Self {
        Self {
            value: value.max(Self::HALF_BASE_FPS_F32),
            scale_factor: value / Self::HALF_BASE_FPS_F32,
            inverted_value: 1f32 / value,
            inverted_scale_factor: Self::HALF_BASE_FPS_F32 / value,
        }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

struct OffscreenRenderTargetCondition {}

struct ViewLayout {
    window_device_pixel_ratio: (f32, f32),
    viewport_device_pixel_ratio: (f32, f32),
    window_size: Vector2<u16>,
    viewport_image_size: Vector2<u16>,
    viewport_padding: Vector2<u16>,
    uniform_viewport_layout_rect: (Vector4<u16>, Vector4<u16>),
    uniform_viewport_image_size: (Vector2<u16>, Vector2<u16>),
}

impl ViewLayout {
    pub fn viewport_device_pixel_ratio(&self) -> f32 {
        self.viewport_device_pixel_ratio.0
    }

    pub fn window_device_pixel_ratio(&self) -> f32 {
        self.window_device_pixel_ratio.0
    }

    pub fn logical_scale_uniformed_viewport_layout_rect(&self) -> Vector4<u16> {
        self.uniform_viewport_layout_rect.0
    }

    pub fn logical_scale_uniformed_viewport_image_size(&self) -> Vector2<u16> {
        self.uniform_viewport_image_size.0
    }

    pub fn device_scale_uniformed_viewport_layout_rect(&self) -> Vector4<u16> {
        // TODO: Origin Likes below. s * dpr has any meaning?
        // const nanoem_f32_t dpr = viewportDevicePixelRatio(), s = windowDevicePixelRatio() / dpr;
        // return Vector4(logicalScaleUniformedViewportLayoutRect()) * dpr * s;
        let window_device_pixel_ratio = self.window_device_pixel_ratio();
        self.logical_scale_uniformed_viewport_layout_rect()
            .map(|v| ((v as f32) * window_device_pixel_ratio) as u16)
    }

    pub fn device_scale_uniformed_viewport_image_size(&self) -> Vector2<u16> {
        let window_device_pixel_ratio = self.window_device_pixel_ratio();
        self.logical_scale_uniformed_viewport_image_size()
            .map(|v| ((v as f32) * window_device_pixel_ratio) as u16)
    }

    fn adjust_viewport_image_rect(
        viewport_image_rect: &mut Vector4<f32>,
        viewport_layout_rect: Vector4<f32>,
        viewport_image_size: Vector2<f32>,
    ) {
        if viewport_layout_rect.z > viewport_image_size.x {
            viewport_image_rect.x += (viewport_layout_rect.z - viewport_image_size.x) * 0.5f32;
            viewport_image_rect.z = viewport_image_size.x;
        }
        if viewport_layout_rect.w > viewport_image_size.y {
            viewport_image_rect.y += (viewport_layout_rect.w - viewport_image_size.y) * 0.5f32;
            viewport_image_rect.w = viewport_image_size.y;
        }
    }

    pub fn logical_scale_uniformed_viewport_image_rect(&self) -> Vector4<f32> {
        let viewport_layout_rect = self
            .logical_scale_uniformed_viewport_layout_rect()
            .map(|v| v as f32);
        let mut viewport_image_rect = viewport_layout_rect;
        Self::adjust_viewport_image_rect(
            &mut viewport_image_rect,
            viewport_layout_rect,
            self.logical_scale_uniformed_viewport_image_size()
                .map(|v| v as f32),
        );
        viewport_image_rect.x -= self.viewport_padding.x as f32;
        viewport_image_rect.y -= self.viewport_padding.y as f32;
        viewport_image_rect
    }

    pub fn device_scale_uniformed_viewport_image_rect(&self) -> Vector4<f32> {
        let viewport_layout_rect = self
            .device_scale_uniformed_viewport_layout_rect()
            .map(|v| v as f32);
        let mut viewport_image_rect = viewport_layout_rect;
        Self::adjust_viewport_image_rect(
            &mut viewport_image_rect,
            viewport_layout_rect,
            self.device_scale_uniformed_viewport_image_size()
                .map(|v| v as f32),
        );
        viewport_image_rect
    }

    pub fn resolve_logical_cursor_position_in_viewport(
        &self,
        value: &Vector2<i32>,
    ) -> Vector2<i32> {
        let size = self.uniform_viewport_image_size.0;
        let offset =
            self.uniform_viewport_layout_rect.0.truncate().truncate() + self.viewport_padding;
        Vector2::new(
            value.x - offset.x as i32,
            size.y as i32 - (value.y - offset.y as i32),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectStates {
    pub disable_hidden_bone_bounds_rigid_body: bool,
    pub display_user_interface: bool,
    pub display_transform_handle: bool,
    pub enable_loop: bool,
    pub enable_shared_camera: bool,
    pub enable_ground_shadow: bool,
    pub enable_multiple_bone_selection: bool,
    pub enable_bezier_curve_adjustment: bool,
    pub enable_motion_merge: bool,
    pub enable_effect_plugin: bool,
    pub enable_viewport_locked: bool,
    pub disable_display_sync: bool,
    pub primary_cursor_type_left: bool,
    pub loading_redo_file: bool,
    pub enable_playing_audio_part: bool,
    pub enable_viewport_with_transparent: bool,
    pub enable_compiled_effect_cache: bool,
    pub reset_all_passes: bool,
    pub cancel_requested: bool,
    pub enable_uniformed_viewport_image_size: bool,
    pub viewport_hovered: bool,
    pub enable_fps_counter: bool,
    pub enable_performance_monitor: bool,
    pub input_text_focus: bool,
    pub viewport_image_size_changed: bool,
    pub enable_physics_simulation_for_bone_keyframe: bool,
    pub enable_image_anisotropy: bool,
    pub enable_image_mipmap: bool,
    pub enable_power_saving: bool,
    pub enable_model_editing: bool,
    pub viewport_window_detached: bool,
}

pub type ModelHandle = u32;

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
    transform_model_order_list: Vec<ModelHandle>,
    active_model_pair: (Option<ModelHandle>, Option<ModelHandle>),
    // active_accessory: Option<Rc<RefCell<Accessory>>>,
    // audio_player: Option<Box<dyn AudioPlayer>>,
    physics_engine: Box<PhysicsEngine>,
    camera: PerspectiveCamera,
    light: DirectionalLight,
    shadow_camera: ShadowCamera,
    grid: Box<Grid>,
    camera_motion: Motion,
    light_motion: Motion,
    self_shadow_motion: Motion,
    // undo_stack: Rc<RefCell<UndoStack>>,
    // all_models: Vec<Rc<RefCell<Model>>>,
    // all_accessories: Vec<Rc<RefCell<Accessory>>>,
    // all_motions: Vec<Rc<RefCell<Motion>>>,
    // drawable_to_motion_ptrs: HashMap<Rc<RefCell<dyn Drawable>>, Rc<RefCell<Motion>>>,
    model_to_motion: HashMap<ModelHandle, Motion>,
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
    clear_pass: Box<ClearPass>,
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
    editing_mode: EditingMode,
    // file_path_mode: FilePathMode,
    // playing_segment: TimeLineSegment,
    // selection_segment: TimeLineSegment,
    base_duration: u32,
    language: LanguageType,
    layout: ViewLayout,
    // background_video_rect: Vector4<i32>,
    // bone_selection_rect: Vector4<i32>,
    // logical_scale_cursor_positions: HashMap<CursorType, Vector4<i32>>,
    // logical_scale_moving_cursor_position: Vector2<i32>,
    // scroll_delta: Vector2<i32>,
    viewport_background_color: Vector4<f32>,
    // all_offscreen_render_targets: HashMap<Rc<RefCell<Effect>>, HashMap<String, Vec<OffscreenRenderTargetCondition>>>,
    fallback_texture: wgpu::Texture,
    // // TODO: bx::HandleAlloc *m_objectHandleAllocator;
    object_handler_allocator: HandleAllocator,
    // accessory_handle_map: HashMap<u16, Rc<RefCell<Accessory>>>,
    model_handle_map: HashMap<ModelHandle, Model>,
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
    preferred_motion_fps: FpsUnit,
    // editing_fps: u32,
    // bone_interpolation_type: i32,
    // camera_interpolation_type: i32,
    // model_clipboard: Vec<u8>,
    // motion_clipboard: Vec<u8>,
    // effect_order_set: HashMap<effect::ScriptOrderType, HashSet<Rc<RefCell<dyn Drawable>>>>,
    // effect_references: HashMap<String, (Rc<RefCell<Effect>>, i32)>,
    // loaded_effect_set: HashSet<Rc<RefCell<Effect>>>,
    depends_on_script_external: Vec<Box<dyn Drawable>>,
    transform_performed_at: (u32, i32),
    // indices_of_material_to_attach_effect: (u16, HashSet<usize>),
    // uptime: (f64, f64),
    local_frame_index: (u32, u32),
    time_step_factor: f32,
    // background_video_scale_factor: f32,
    // circle_radius: f32,
    sample_level: (u32, u32),
    state_flags: ProjectStates,
    // confirm_seek_flags: u64,
    // last_physics_debug_flags: u32,
    // coordination_system: u32,
    // cursor_modifiers: u32,
    // actual_fps: u32,
    // actual_sequence: u32,
    // active: bool,
    tmp_texture_map: HashMap<String, Rc<wgpu::Texture>>,
}

impl Project {
    pub const MINIMUM_BASE_DURATION: u32 = 300;
    pub const MAXIMUM_BASE_DURATION: u32 = i32::MAX as u32;
    pub const DEFAULT_CIRCLE_RADIUS_SIZE: f32 = 7.5f32;

    pub const DEFAULT_VIEWPORT_IMAGE_SIZE: [u16; 2] = [640, 360];
    pub const TIME_BASED_AUDIO_SOURCE_DEFAULT_SAMPLE_RATE: u32 = 1440;

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

        let layout = ViewLayout {
            window_device_pixel_ratio: (
                injector.window_device_pixel_ratio,
                injector.window_device_pixel_ratio,
            ),
            viewport_device_pixel_ratio: (
                injector.viewport_device_pixel_ratio,
                injector.viewport_device_pixel_ratio,
            ),
            window_size: injector.window_size.into(),
            viewport_image_size: Self::DEFAULT_VIEWPORT_IMAGE_SIZE.into(),
            viewport_padding: Vector2::new(0, 0),
            uniform_viewport_layout_rect: (Vector4::new(0, 0, 0, 0), Vector4::new(0, 0, 0, 0)),
            uniform_viewport_image_size: (
                Self::DEFAULT_VIEWPORT_IMAGE_SIZE.into(),
                Self::DEFAULT_VIEWPORT_IMAGE_SIZE.into(),
            ),
        };

        let mut camera = PerspectiveCamera::new();
        camera.update(
            layout.viewport_image_size,
            &layout.logical_scale_uniformed_viewport_image_rect(),
            camera.look_at(None),
        );
        camera.set_dirty(false);
        let mut shadow_camera = ShadowCamera::new(&device);
        if adapter.get_info().backend == wgpu::Backend::Gl {
            // TODO: disable shadow map when gles3
            shadow_camera.set_enabled(false);
        }
        shadow_camera.set_dirty(false);
        let directional_light = DirectionalLight::new();

        let mut physics_engine = Box::new(PhysicsEngine::new());
        physics_engine.simulation_mode = SimulationMode::EnableTracing;

        let mut object_handler_allocator = HandleAllocator::new();

        let mut camera_motion = Motion::empty();
        camera_motion.initialize_camera_frame_0(&camera, None);
        let mut light_motion = Motion::empty();
        light_motion.initialize_light_frame_0(&directional_light);
        let mut self_shadow_motion = Motion::empty();
        self_shadow_motion.initialize_self_shadow_frame_0(&shadow_camera);

        // TODO: build tracks

        Self {
            editing_mode: EditingMode::None,
            language: LanguageType::English,
            base_duration: Self::MINIMUM_BASE_DURATION,
            preferred_motion_fps: FpsUnit::new(60f32),
            time_step_factor: 1f32,
            layout,
            active_model_pair: (None, None),
            grid: Box::new(Grid::new(device)),
            camera_motion,
            light_motion,
            self_shadow_motion,
            model_to_motion: HashMap::new(),
            draw_type: DrawType::Color,
            depends_on_script_external: vec![],
            clear_pass: Box::new(ClearPass::new(device)),
            viewport_texture_format: (injector.texture_format(), injector.texture_format()),
            viewport_background_color: Vector4::new(0f32, 0f32, 0f32, 1f32),
            local_frame_index: (0, 0),
            transform_performed_at: (Motion::MAX_KEYFRAME_INDEX, 0),
            sample_level: (0u32, 0u32),
            camera,
            shadow_camera,
            light: directional_light,
            fallback_texture,
            object_handler_allocator,
            model_handle_map: HashMap::new(),
            transform_model_order_list: vec![],
            viewport_primary_pass,
            viewport_secondary_pass,
            physics_engine,
            tmp_texture_map: HashMap::new(),
            state_flags: ProjectStates {
                display_transform_handle: true,
                display_user_interface: true,
                enable_motion_merge: true,
                enable_uniformed_viewport_image_size: true,
                enable_fps_counter: true,
                enable_performance_monitor: true,
                enable_physics_simulation_for_bone_keyframe: true,
                enable_image_anisotropy: true,
                enable_ground_shadow: true,
                reset_all_passes: adapter.get_info().backend != wgpu::Backend::Gl,
                ..Default::default()
            },
        }
        // TODO: may need to publish set fps event
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

    pub fn current_frame_index(&self) -> u32 {
        self.local_frame_index.0
    }

    pub fn global_camera(&self) -> &PerspectiveCamera {
        &self.camera
    }

    pub fn global_camera_mut(&mut self) -> &mut PerspectiveCamera {
        &mut self.camera
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

    pub fn global_light_mut(&mut self) -> &mut DirectionalLight {
        &mut self.light
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

    pub fn set_transform_performed_at(&mut self, value: (u32, i32)) {
        self.transform_performed_at = value
    }

    pub fn reset_transform_performed_at(&mut self) {
        self.set_transform_performed_at((Motion::MAX_KEYFRAME_INDEX, 0))
    }

    pub fn duration(&self, base_duration: u32) -> u32 {
        let mut duration =
            base_duration.clamp(Self::MINIMUM_BASE_DURATION, Self::MAXIMUM_BASE_DURATION);
        duration = duration.max(self.camera_motion.duration());
        duration = duration.max(self.light_motion.duration());
        for motion in self.model_to_motion.values() {
            duration = duration.max(motion.duration());
        }
        duration
    }

    pub fn project_duration(&self) -> u32 {
        self.duration(self.base_duration)
    }

    pub fn base_duration(&self) -> u32 {
        self.base_duration
    }

    pub fn set_base_duration(&mut self, value: u32) {
        self.base_duration = value
            .clamp(Self::MINIMUM_BASE_DURATION, Self::MAXIMUM_BASE_DURATION)
            .max(self.base_duration);
        let new_duration = self.duration(value);
        // TODO: Update playing segment and selection segment
    }

    pub fn active_model(&self) -> Option<&Model> {
        self.active_model_pair
            .0
            .and_then(|idx| self.model_handle_map.get(&idx))
    }

    pub fn active_model_mut(&mut self) -> Option<&mut Model> {
        self.active_model_pair
            .0
            .and_then(|idx| self.model_handle_map.get_mut(&idx))
    }

    pub fn set_active_model(&mut self, model: Option<ModelHandle>) {
        let last_active_model = self.active_model_pair.0;
        if last_active_model != model && !self.state_flags.enable_model_editing {
            self.active_model_pair = (model, last_active_model);
            if self.editing_mode == EditingMode::None {
                self.editing_mode = EditingMode::Select;
            } else if model.is_none() {
                self.editing_mode = EditingMode::None;
            }
            self.camera.update(
                self.layout.viewport_image_size,
                &self.layout.logical_scale_uniformed_viewport_image_rect(),
                self.camera.bound_look_at(self),
            );
            // TODO: publish event
            // TODO: rebuild tracks
            self.internal_seek(self.local_frame_index.0);
        }
    }

    pub fn model_mut(&mut self, handle: ModelHandle) -> Option<&mut Model> {
        self.model_handle_map.get_mut(&handle)
    }

    pub fn find_model_by_name(&self, name: &str) -> Option<&Model> {
        for (idx, model) in &self.model_handle_map {
            if model.get_name() == name || model.get_canonical_name() == name {
                return Some(model);
            }
        }
        return None;
    }

    pub fn resolve_bone(&self, value: (&str, &str)) -> Option<&Bone> {
        self.find_model_by_name(value.0)
            .and_then(|model| model.find_bone(value.1))
    }

    pub fn resolve_model_motion(&self, model: ModelHandle) -> Option<&Motion> {
        self.model_to_motion.get(&model)
    }

    pub fn resolve_logical_cursor_position_in_viewport(
        &self,
        value: &Vector2<i32>,
    ) -> Vector2<i32> {
        self.layout
            .resolve_logical_cursor_position_in_viewport(value)
    }

    pub fn logical_scale_uniformed_viewport_image_size(&self) -> Vector2<u16> {
        self.layout.logical_scale_uniformed_viewport_image_size()
    }

    pub fn physics_simulation_time_step(&self) -> f32 {
        self.inverted_preferred_motion_fps()
            * self.preferred_motion_fps.scale_factor
            * self.time_step_factor
    }

    pub fn inverted_preferred_motion_fps(&self) -> f32 {
        self.preferred_motion_fps.inverted_value
    }

    pub fn is_physics_simulation_enabled(&self) -> bool {
        match self.physics_engine.simulation_mode {
            crate::physics_engine::SimulationMode::Disable => false,
            crate::physics_engine::SimulationMode::EnableAnytime
            | crate::physics_engine::SimulationMode::EnableTracing => true,
            crate::physics_engine::SimulationMode::EnablePlaying => self.is_playing(),
        }
    }

    pub fn is_playing(&self) -> bool {
        // TODO: project playing state
        false
    }
}

impl Project {
    pub fn update_global_camera(&mut self) {
        let bound_look_at = self.global_camera().bound_look_at(self);
        let viewport_image_size = self.layout.viewport_image_size;
        let logical_scale_uniformed_viewport_image_rect =
            self.layout.logical_scale_uniformed_viewport_image_rect();
        self.global_camera_mut().update(
            viewport_image_size,
            &logical_scale_uniformed_viewport_image_rect,
            bound_look_at,
        );
    }

    pub fn internal_seek(&mut self, frame_index: u32) {
        self.internal_seek_precisely(frame_index, 0f32, 0f32);
    }

    pub fn internal_seek_precisely(&mut self, frame_index: u32, amount: f32, delta: f32) {
        if self.transform_performed_at.0 != Motion::MAX_KEYFRAME_INDEX
            && frame_index != self.transform_performed_at.0
        {
            // TODO: active model undo stack set offset
            self.reset_transform_performed_at();
        }
        if frame_index < self.local_frame_index.0 {
            self.restart(frame_index);
        }
        self.synchronize_all_motions(frame_index, amount, SimulationTiming::Before);
        self.internal_perform_physics_simulation(delta);
        self.synchronize_all_motions(frame_index, amount, SimulationTiming::After);
        self.mark_all_models_dirty();
        self.camera.update(
            self.layout.viewport_image_size,
            &self.layout.logical_scale_uniformed_viewport_image_rect(),
            self.camera.bound_look_at(self),
        );
        self.light.set_dirty(false);
        self.camera.set_dirty(false);
        // TODO: seek background
        // FIXME?: there's nothing to ensure local_frame <= frame_index
        self.local_frame_index.1 = frame_index - self.local_frame_index.0;
        self.local_frame_index.0 = frame_index;
    }

    pub fn reset_all_model_edges(
        &mut self,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        let physics_simulation_time_step = self.physics_simulation_time_step();
        for (handle, model) in &mut self.model_handle_map {
            if model.edge_size_scale_factor() > 0f32 && !model.is_staging_vertex_buffer_dirty() {
                model.reset_all_morph_deform_states(
                    self.model_to_motion.get(handle).unwrap(),
                    self.local_frame_index.0,
                    &mut self.physics_engine,
                );
                model.deform_all_morphs(false);
                model.perform_all_bones_transform(
                    &mut self.physics_engine,
                    physics_simulation_time_step,
                    outside_parent_bone_map,
                );
                model.mark_staging_vertex_buffer_dirty();
            }
        }
    }

    pub fn synchronize_all_motions(
        &mut self,
        frame_index: u32,
        amount: f32,
        timing: SimulationTiming,
    ) {
        // TODO: for model motions
        for handle in &self.transform_model_order_list {
            if let Some(model) = self.model_handle_map.get_mut(handle) {
                let outside_parent_bone_map = HashMap::new();
                if let Some(motion) = self.model_to_motion.get(handle) {
                    model.synchronize_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        &mut self.physics_engine,
                        &outside_parent_bone_map,
                    );
                }
            }
        }
        if timing == SimulationTiming::After {
            // TODO: for accessory motions
            self.synchronize_camera(frame_index, amount);
            self.synchronize_light(frame_index, amount);
            self.synchronize_self_shadow(frame_index, amount);
        }
    }

    pub fn synchronize_camera(&mut self, frame_index: u32, amount: f32) {
        const CAMERA_DIRECTION: Vector3<f32> = Vector3::new(-1f32, 1f32, 1f32);
        let mut camera0 = self.camera.clone();
        camera0.synchronize_parameters(&self.camera_motion, frame_index, self);
        let look_at0 = camera0.look_at(self.active_model());
        if amount > 0f32
            && self
                .camera_motion
                .find_camera_keyframe(frame_index + 1)
                .is_none()
        {
            let mut camera1 = self.camera.clone();
            camera1.synchronize_parameters(&self.camera_motion, frame_index, self);
            let look_at1 = camera1.look_at(self.active_model());
            self.camera.set_angle(
                camera0
                    .angle()
                    .lerp(camera1.angle(), amount)
                    .mul_element_wise(CAMERA_DIRECTION),
            );
            self.camera
                .set_distance(lerp_f32(camera0.distance(), camera1.distance(), amount));
            self.camera.set_fov_radians(lerp_f32(
                camera0.fov_radians(),
                camera1.fov_radians(),
                amount,
            ));
            self.camera.set_look_at(look_at0.lerp(look_at1, amount));
        } else {
            self.camera
                .set_angle(camera0.angle().mul_element_wise(CAMERA_DIRECTION));
            self.camera.set_distance(camera0.distance());
            self.camera.set_fov_radians(camera0.fov_radians());
            self.camera.set_look_at(look_at0);
        }
        self.camera.set_perspective(camera0.is_perspective());
        self.camera.interpolation = camera0.interpolation;
        let bound_look_at = self.camera.bound_look_at(self);
        self.camera.update(
            self.layout.viewport_image_size,
            &self.layout.logical_scale_uniformed_viewport_image_rect(),
            bound_look_at,
        );
        self.camera.set_dirty(false);
    }

    pub fn synchronize_light(&mut self, frame_index: u32, amount: f32) {
        let mut light0 = self.light.clone();
        light0.synchronize_parameters(&self.light_motion, frame_index);
        if amount > 0f32 {
            let mut light1 = self.light.clone();
            light1.synchronize_parameters(&self.light_motion, frame_index + 1);
            self.light
                .set_color(light0.color().lerp(light1.color(), amount));
            self.light
                .set_direction(light0.direction().lerp(light1.direction(), amount));
        } else {
            self.light.set_color(light0.color());
            self.light.set_direction(light0.direction());
        }
        self.light.set_dirty(false);
    }

    pub fn synchronize_self_shadow(&mut self, frame_index: u32, amount: f32) {
        if let Some(keyframe) = self
            .self_shadow_motion
            .find_self_shadow_keyframe(frame_index)
        {
            self.shadow_camera.set_distance(keyframe.distance);
            self.shadow_camera
                .set_coverage_mode((keyframe.mode as u32).into());
            self.shadow_camera.set_dirty(false);
        }
    }

    pub fn mark_all_models_dirty(&mut self) {
        for (_, model) in &mut self.model_handle_map {
            model.mark_staging_vertex_buffer_dirty();
        }
        // TODO: mark blitter dirty
    }
}

impl Project {
    pub fn perform_model_bones_transform(&mut self, model: Option<ModelHandle>) {
        let physics_simulation_time_step = self.physics_simulation_time_step();
        let outside_parent_bone_map = HashMap::new();
        if let Some(model) = model
            .or_else(|| self.active_model_pair.0)
            .and_then(|handle| self.model_handle_map.get_mut(&handle))
        {
            model.perform_all_bones_transform(
                &mut self.physics_engine,
                physics_simulation_time_step,
                &outside_parent_bone_map,
            );
        }
    }

    pub fn register_bone_keyframes(
        &mut self,
        model: Option<ModelHandle>,
        bones: &HashMap<String, Vec<u32>>,
    ) {
        self.reset_transform_performed_at();
        if let Some((handle, model)) =
            model
                .or_else(|| self.active_model_pair.0)
                .and_then(|handle| {
                    self.model_handle_map
                        .get_mut(&handle)
                        .map(|model| (handle, model))
                })
        {
            if let Some(motion) = self.model_to_motion.get_mut(&handle) {
                let mut updaters = motion.build_add_bone_keyframes_updaters(
                    model,
                    bones,
                    self.state_flags.enable_bezier_curve_adjustment,
                    self.state_flags.enable_physics_simulation_for_bone_keyframe,
                );
                motion.apply_add_bone_keyframes_updaters(model, &mut updaters);
            }
        }
    }
}

impl Project {
    pub fn load_model(&mut self, model_data: &[u8], device: &wgpu::Device) {
        if let Ok(model) = Model::new_from_bytes(
            model_data,
            self.parse_language(),
            &self.fallback_texture,
            &mut self.physics_engine,
            &self.camera,
            device,
        ) {
            let handle = self.add_model(model);
            self.set_active_model(Some(handle));
        }
    }

    pub fn add_model(&mut self, mut model: Model) -> ModelHandle {
        model.clear_all_bone_bounds_rigid_bodies();
        if !self.state_flags.disable_hidden_bone_bounds_rigid_body {
            model.create_all_bone_bounds_rigid_bodies();
        }
        let model_handle = self.object_handler_allocator.next();
        self.model_handle_map.insert(model_handle, model);
        self.transform_model_order_list.push(model_handle);
        // TODO: add effect to kScriptOrderTypeStandard
        // TODO: publish event
        let motion = Motion::empty();
        // TODO: clear model undo stack
        self.add_model_motion(motion, model_handle);
        // TODO: applyAllOffscreenRenderTargetEffectsToDrawable
        model_handle
    }

    pub fn load_model_motion(&mut self, motion_data: &[u8]) -> Result<(), Error> {
        if self.active_model().is_some() {
            Motion::new_from_bytes(motion_data, self.local_frame_index.0).and_then(|motion| {
                if motion.opaque.target_model_name == Motion::CAMERA_AND_LIGHT_TARGET_MODEL_NAME {
                    return Err(Error::new(
                        "読み込まれたモーションはモデル用ではありません",
                        "",
                        crate::error::DomainType::DomainTypeApplication,
                    ));
                }
                // TODO: record history in motion redo
                let (missing_bones, missing_morphs) =
                    motion.test_all_missing_model_objects(self.active_model().unwrap());
                if missing_bones.len() > 0 && missing_morphs.len() > 0 {
                    // TODO: Dialog hint motion missing
                }
                // TODO: add all to motion selection
                let _ = self.add_model_motion(motion, self.active_model_pair.0.unwrap());
                self.restart_from_current();
                Ok(())
            })
        } else {
            Err(Error::new(
                "モデルモーションを読み込むためのモデルが選択されていません",
                "モデルを選択してください",
                crate::error::DomainType::DomainTypeApplication,
            ))
        }
    }

    pub fn load_camera_motion(&mut self, motion_data: &[u8]) -> Result<(), Error> {
        Motion::new_from_bytes(motion_data, self.local_frame_index.0).and_then(|motion| {
            if motion.opaque.target_model_name != Motion::CAMERA_AND_LIGHT_TARGET_MODEL_NAME {
                return Err(Error::new(
                    "読み込まれたモーションはカメラ及び照明用ではありません",
                    "",
                    crate::error::DomainType::DomainTypeApplication,
                ));
            }
            // TODO: record history in motion redo
            let _ = self.set_camera_motion(motion);
            Ok(())
        })
    }

    pub fn load_light_motion(&mut self, motion_data: &[u8]) -> Result<(), Error> {
        Motion::new_from_bytes(motion_data, self.local_frame_index.0).and_then(|motion| {
            if motion.opaque.target_model_name != Motion::CAMERA_AND_LIGHT_TARGET_MODEL_NAME {
                return Err(Error::new(
                    "読み込まれたモーションはカメラ及び照明用ではありません",
                    "",
                    crate::error::DomainType::DomainTypeApplication,
                ));
            }
            // TODO: record history in motion redo
            let _ = self.set_light_motion(motion);
            Ok(())
        })
    }

    pub fn add_model_motion(&mut self, mut motion: Motion, model: ModelHandle) -> Option<Motion> {
        let last_model_motion = self.model_to_motion.get(&model);
        if let Some(last_model_motion) = last_model_motion.map(|motion| motion.clone()) {
            if self.state_flags.enable_motion_merge {
                motion.merge_all_keyframes(&last_model_motion);
            }
            if let Some(model_object) = self.model_handle_map.get(&model) {
                motion.initialize_model_frame_0(model_object);
                // TODO: clear model undo stack
                self.model_to_motion.insert(model, motion);
                self.set_base_duration(self.project_duration());
                // TODO: publish add motion event
                return Some(last_model_motion);
            }
        }
        return None;
    }

    pub fn set_camera_motion(&mut self, mut motion: Motion) -> Motion {
        let last_motion = self.camera_motion.clone();
        self.camera_motion = motion;
        self.set_base_duration(self.project_duration());
        let active_model = self
            .active_model_pair
            .0
            .and_then(|idx| self.model_handle_map.get(&idx));
        self.camera_motion
            .initialize_camera_frame_0(&self.camera, active_model);
        self.synchronize_camera(self.local_frame_index.0, 0f32);
        // TODO: publish motion event
        last_motion
    }

    pub fn set_light_motion(&mut self, mut motion: Motion) -> Motion {
        let last_motion = self.light_motion.clone();
        self.light_motion = motion;
        self.set_base_duration(self.project_duration());
        self.light_motion.initialize_light_frame_0(&self.light);
        self.synchronize_light(self.local_frame_index.0, 0f32);
        // TODO: publish motion event
        last_motion
    }

    pub fn restart_from_current(&mut self) {
        self.restart(self.local_frame_index.0);
    }

    fn restart(&mut self, frame_index: u32) {
        self.synchronize_all_motions(frame_index, 0f32, SimulationTiming::Before);
        for (handle, model) in &mut self.model_handle_map {
            model.initialize_all_rigid_bodies_transform_feedback(&mut self.physics_engine);
            // TODO: soft_bodies
        }
        self.internal_perform_physics_simulation(self.physics_simulation_time_step());
        self.synchronize_all_motions(frame_index, 0f32, SimulationTiming::After);
        self.mark_all_models_dirty();
    }

    fn internal_perform_physics_simulation(&mut self, delta: f32) {
        if self.is_physics_simulation_enabled() {
            self.physics_engine.step(delta);
            for (_, model) in &mut self.model_handle_map {
                if model.is_physics_simulation_enabled() {
                    model.synchronize_all_rigid_bodies_transform_feedback_from_simulation(
                        RigidBodyFollowBone::Perform,
                        &mut self.physics_engine,
                    );
                }
            }
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
        for (handle, model) in &mut self.model_handle_map {
            model.create_all_images(&self.tmp_texture_map);
        }
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
                    Some(&self.viewport_primary_depth_view()),
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
            let pipeline = self.clear_pass.get_pipeline(
                &format.color_texture_formats,
                format.depth_texture_format,
                device,
            );
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("@mdanceio/ClearRenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
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
                })],
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
            _render_pass.set_vertex_buffer(0, self.clear_pass.vertex_buffer.slice(..));
            _render_pass.set_pipeline(&pipeline);
            _render_pass.draw(0..4, 0..1)
        }
        encoder.pop_debug_group();
        queue.submit(Some(encoder.finish()));
    }

    pub fn draw_grid(
        &mut self,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.grid.draw(
            view,
            Some(&self.viewport_primary_depth_view()),
            self,
            device,
            queue,
            adapter.get_info(),
        );
    }

    pub fn draw_shadow_map(
        &mut self,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.shadow_camera.is_enabled() {
            let (light_view, light_projection) = self.shadow_camera.get_view_projection(self);
            self.shadow_camera.clear(&self.clear_pass, device, queue);
            if self.editing_mode != EditingMode::Select {
                // scope(m_currentOffscreenRenderPass, pass), scope2(m_originOffscreenRenderPass, pass)
                for (handle, drawable) in &self.model_handle_map {
                    // TODO: judge effect script class
                    let color_view = self.shadow_camera.color_texture_view();
                    let depth_view = self.shadow_camera.depth_texture_view();
                    drawable.draw(
                        &color_view,
                        Some(&depth_view),
                        DrawType::ShadowMap,
                        self,
                        device,
                        queue,
                        adapter.get_info(),
                    )
                }
            }
        }
    }

    pub fn draw_viewport(
        &self,
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
            ScriptOrder::Standard,
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
        order: ScriptOrder,
        typ: DrawType,
        view: &wgpu::TextureView,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        for (handle, drawable) in &self.model_handle_map {
            drawable.draw(
                view,
                Some(&self.viewport_primary_depth_view()),
                typ,
                self,
                device,
                queue,
                adapter.get_info(),
            );
        }
    }
}

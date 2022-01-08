use std::{
    cell::{Ref, RefCell},
    rc::Rc, collections::{HashMap, HashSet},
};

use cgmath::{Vector4, Vector2};
use wgpu::{RenderPass, Texture};

use crate::{
    accessory::Accessory,
    accessory_program_bundle::AccessoryProgramBundle,
    audio_player::AudioPlayer,
    background_video_renderer::BackgroundVideoRenderer,
    camera::PerspectiveCamera,
    debug_capture::DebugCapture,
    effect::{common::RenderPassScope, global_uniform::GlobalUniform},
    error::Error,
    event_publisher::EventPublisher,
    file_manager::FileManager,
    grid::Grid,
    image_loader::ImageLoader,
    image_view::ImageView,
    internal::{BlitPass, ClearPass, DebugDrawer},
    light::Light,
    model::{BindPose, Model, SkinDeformer, VisualizationClause},
    model_program_bundle::ModelProgramBundle,
    motion::Motion,
    physics_engine::PhysicsEngine,
    primitive_2d::Primitive2d,
    progress::CancelPublisher,
    shadow_camera::ShadowCamera,
    track::Track,
    translator::{Translator, LanguageType},
    undo::UndoStack, drawable::{DrawType, Drawable}, uri::Uri, file_utils, time_line_segment::TimeLineSegment,
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

pub enum EditingMode {

}

pub enum FilePathMode {
    
}

pub enum CursorType {

}

pub struct RenderPassBundle {

}

struct Pass {

}

struct FpsUnit {

}

pub struct Project<'a> {
    background_video_renderer: Box<dyn BackgroundVideoRenderer<Error>>,
    confirmor: Box<dyn Confirmor>,
    file_manager: Box<dyn FileManager>,
    event_publisher: Box<dyn EventPublisher>,
    primitive_2d: Box<dyn Primitive2d>,
    renderer_capability: Box<dyn RendererCapability>,
    shared_cancel_publisher_factory: Box<dyn SharedCancelPublisherFactory>,
    shared_resource_factory: Box<dyn SharedResourceFactory>,
    translator: Box<dyn Translator>,
    image_loader: Rc<ImageLoader>,
    transform_model_order_list: Vec<Rc<RefCell<Model>>>,
    active_model_pair: (Rc<RefCell<Model>>, Rc<RefCell<Model>>),
    active_accessory: Rc<RefCell<Accessory>>,
    audio_player: Box<dyn AudioPlayer>,
    physics_engine: Rc<PhysicsEngine>,
    camera: Rc<PerspectiveCamera>,
    light: Rc<Light>,
    grid: Rc<Grid>,
    camera_motion: Rc<RefCell<Motion>>,
    light_motion: Rc<RefCell<Motion>>,
    self_shadow_motion: Rc<RefCell<Motion>>,
    shadow_camera: Rc<ShadowCamera>,
    undo_stack: Rc<RefCell<UndoStack>>,
    all_models: Vec<Rc<RefCell<Model>>>,
    all_accessories: Vec<Rc<RefCell<Accessory>>>,
    all_motions: Vec<Rc<RefCell<Motion>>>,
    // TODO: MotionHashMap m_drawable2MotionPtrs
    all_traces: Vec<Rc<RefCell<dyn Track>>>,
    selected_track: Option<Rc<RefCell<dyn Track>>>,
    last_save_state: Option<SaveState>,
    draw_queue: Box<DrawQueue>,
    batch_draw_queue: Box<BatchDrawQueue>,
    serial_draw_queue: Box<SerialDrawQueue>,
    offscreen_render_pass_scope: Box<RenderPassScope>,
    viewport_pass_blitter: Box<BlitPass>,
    render_pass_blitter: Box<BlitPass>,
    shared_image_blitter: Box<BlitPass>,
    render_pass_clear: Box<ClearPass>,
    shared_debug_drawer: Rc<RefCell<DebugDrawer>>,
    viewport_pixel_format: (wgpu::TextureFormat, wgpu::TextureFormat),
    last_bind_pose: BindPose,
    rigid_body_visualization_clause: VisualizationClause,
    draw_type: DrawType,
    file_uri: (Uri, file_utils::TransientPath),
    redo_file_uri: Uri,
    // TODO: OffscreenRenderTargetDrawableSet m_drawablesToAttachOffscreenRenderTargetEffect;
    current_render_pass: Option<Rc<RenderPass<'a>>>,
    last_drawn_render_pass: Option<Rc<RenderPass<'a>>>,
    current_offscreen_render_pass: Option<Rc<RenderPass<'a>>>,
    origin_offscreen_render_pass: Option<Rc<RenderPass<'a>>>,
    script_external_render_pass: Option<Rc<RenderPass<'a>>>,
    // TODO: SharedRenderTargetImageContainerMap m_sharedRenderTargetImageContainers;
    editing_mode: EditingMode,
    file_path_mode: FilePathMode,
    playing_segment: TimeLineSegment,
    selection_segment: TimeLineSegment,
    base_duration: u32,
    language: LanguageType,
    uniform_viewport_layout_rect: (Vector4<u16>, Vector4<u16>),
    uniform_viewport_image_size: (Vector2<u16>, Vector2<u16>),
    background_video_rect: Vector4<i32>,
    bone_selection_rect: Vector4<i32>,
    logical_scale_cursor_positions: HashMap<CursorType, Vector4<i32>>,
    logical_scale_moving_cursor_position: Vector2<i32>,
    scroll_delta: Vector2<i32>,
    window_size: Vector2<u16>,
    viewport_image_size: Vector2<u16>,
    viewport_padding: Vector2<u16>,
    viewport_background_color: Vector4<u8>,
    // TODO: OffscreenRenderTargetConditionListMap m_allOffscreenRenderTargets;
    // TODO: OffscreenRenderTargetEffectSetMap m_allOffscreenRenderTargetEffectSets;
    fallback_image: Texture,
    // TODO: bx::HandleAlloc *m_objectHandleAllocator;
    accessory_handle_map: HashMap<u16, Rc<RefCell<Accessory>>>,
    model_handle_map: HashMap<u16, Rc<RefCell<Model>>>,
    motion_handle_map: HashMap<u16, Rc<RefCell<Motion>>>,
    render_pass_bundle_map: HashMap<u32, RenderPassBundle>,
    hashed_render_pass_bundle_map: HashMap<u32, Rc<RefCell<RenderPassBundle>>>,
    redo_object_handles: HashMap<u16, u32>,
    render_pass_string_map: HashMap<u32, String>,
    render_pipeline_string_map: HashMap<u32, String>,
    viewport_primary_pass: Pass,
    viewport_secondary_pass: Pass,
    context_2d_pass: Pass,
    background_image: (Texture, Vector2<u16>),
    preferred_motion_fps: FpsUnit,
    editing_fps: u32,
    bone_interpolation_type: i32,
    camera_interpolation_type: i32,
    model_clipboard: Vec<u8>,
    motion_clipboard: Vec<u8>,
    // EffectOrderSet m_effectOrderSet;
    // EffectReferenceMap m_effectReferences;
    // LoadedEffectSet m_loadedEffectSet;
    depends_on_script_external: Vec<Rc<RefCell<dyn Drawable>>>,
    transform_performed_at: (u32, i32),
    indices_of_material_to_attach_effect: (u16, HashSet<usize>),
    window_device_pixel_ratio: (f32, f32),
    viewport_device_pixel_ratio: (f32, f32),
    uptime: (f64, f64),
    local_frame_index: (u32, u32),
    time_step_factor: f32,
    background_video_scale_factor: f32,
    circle_radius: f32,
    sample_level: (u32, u32),
    state_flags: u64,
    confirm_seek_flags: u64,
    last_physics_debug_flags: u32,
    coordination_system: u32,
    cursor_modifiers: u32,
    actual_fps: u32,
    actual_sequence: u32,
    active: bool,
}

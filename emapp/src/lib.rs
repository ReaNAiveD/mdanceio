mod accessory;
mod accessory_program_bundle;
mod audio_player;
mod background_video_renderer;
pub mod base_application_service;
mod bezier_curve;
mod bounding_box;
mod camera;
mod command;
mod debug_capture;
mod drawable;
mod effect;
pub mod error;
mod event_publisher;
pub mod file_manager;
mod file_utils;
mod forward;
mod grid;
mod image_loader;
mod image_view;
mod injector;
mod internal;
mod light;
pub mod model;
mod model_object_selection;
mod model_program_bundle;
pub mod motion;
pub mod motion_keyframe_selection;
mod pass;
mod physics_engine;
mod pixel_format;
mod primitive_2d;
mod progress;
pub mod project;
mod shadow_camera;
mod state_controller;
mod time_line_segment;
mod track;
mod translator;
mod undo;
mod uri;
#[cfg(target_arch = "wasm32")]
pub mod wasm_init;
mod utils;
mod ray;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

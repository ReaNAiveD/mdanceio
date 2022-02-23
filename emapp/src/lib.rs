mod background_video_renderer;
mod bezier_curve;
mod command;
pub mod error;
mod event_publisher;
pub mod file_manager;
pub mod model;
pub mod motion;
pub mod motion_keyframe_selection;
pub mod project;
mod uri;
mod primitive_2d;
mod progress;
mod debug_capture;
mod accessory_program_bundle;
mod model_program_bundle;
mod effect;
mod image_view;
mod translator;
mod image_loader;
mod accessory;
mod audio_player;
mod physics_engine;
mod camera;
mod light;
mod grid;
mod shadow_camera;
mod undo;
mod track;
mod internal;
mod drawable;
mod file_utils;
mod time_line_segment;
mod forward;
mod model_object_selection;
mod bounding_box;
mod pass;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

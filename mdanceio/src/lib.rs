#![allow(unknown_lints)]

pub mod android_proxy;
mod audio_player;
pub mod base_application_service;
mod bezier_curve;
mod bounding_box;
mod camera;
mod clear_pass;
mod deformer;
mod drawable;
mod effect;
mod error;
mod event_publisher;
mod forward;
mod grid;
pub mod injector;
mod light;
mod line_drawer;
mod model;
mod model_object_selection;
mod model_program_bundle;
mod motion;
mod motion_keyframe_selection;
pub mod offscreen_proxy;
mod pass;
mod physics_engine;
pub mod project;
mod ray;
mod shadow_camera;
mod technique;
mod time_line_segment;
mod translator;
mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm_proxy;

use android_proxy::AndroidProxy;

// #[cfg(target_os = "android")]
uniffi_macros::include_scaffolding!("mdanceio");

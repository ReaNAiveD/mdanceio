use cgmath::{Quaternion, Vector3};
use image::GenericImageView;

use crate::{
    error::MdanceioError,
    injector::Injector,
    project::{ModelHandle, Project},
};
use std::{collections::HashMap, io::Cursor};

pub struct BaseApplicationService {
    project: Project,
}

impl BaseApplicationService {
    pub fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        injector: Injector,
    ) -> Self {
        Self {
            project: Project::new(adapter, device, queue, injector),
        }
    }

    pub fn draw_default_pass(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.project.draw_shadow_map(device, queue);
        self.project.draw_viewport(view, device, queue);
        self.project.update(device, queue);
    }

    pub fn draw_from(
        &mut self,
        model_world: [f32; 16],
        camera_view: [f32; 16],
        camera_projection: [f32; 16],
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.project.draw_shadow_map_from(
            model_world,
            camera_view,
            camera_projection,
            device,
            queue,
        );
        self.project.draw_viewport_from(
            model_world,
            camera_view,
            camera_projection,
            view,
            device,
            queue,
        );
        self.project.update(device, queue);
    }

    pub fn update_current_project(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.project.update(device, queue);
    }

    pub fn load_model(
        &mut self,
        data: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<ModelHandle, MdanceioError> {
        self.project.load_model(data, device, queue)
    }

    pub fn load_model_motion(&mut self, data: &[u8]) -> Result<(), MdanceioError> {
        self.project.load_model_motion(data)
    }

    pub fn load_camera_motion(&mut self, data: &[u8]) -> Result<(), MdanceioError> {
        self.project.load_camera_motion(data)
    }

    pub fn load_light_motion(&mut self, data: &[u8]) -> Result<(), MdanceioError> {
        self.project.load_light_motion(data)
    }

    pub fn seek(&mut self, frame_index: u32) {
        self.project.seek(frame_index, true);
    }

    pub fn play(&mut self) {
        self.project.play();
    }

    pub fn enable_shadow_map(
        &mut self,
        handle: ModelHandle,
        value: bool,
    ) -> Result<(), MdanceioError> {
        if let Some(model) = self.project.model_mut(handle) {
            model.set_shadow_map_enabled(value);
            Ok(())
        } else {
            Err(MdanceioError::model_not_found())
        }
    }

    // pub fn disable_physics_simulation(&mut self) {
    //     self.project
    //         .set_physics_simulation_mode(SimulationMode::Disable)
    // }

    pub fn get_model_texture_paths(&self, model_handle: ModelHandle) -> Vec<String> {
        if let Some(model) = self.project.model(model_handle) {
            model
                .textures()
                .iter()
                .map(|texture| texture.path.clone())
                .collect()
        } else {
            vec![]
        }
    }

    pub fn set_camera_angle(&mut self, value: Vector3<f32>) {
        self.project.global_camera_mut().set_angle(value);
        self.project.update_global_camera();
        self.project.reset_all_model_edges(&HashMap::new());
        // TODO: use undo
    }

    pub fn set_camera_distance(&mut self, value: f32) {
        self.project.global_camera_mut().set_distance(value);
        self.project.update_global_camera();
        self.project.reset_all_model_edges(&HashMap::new());
    }

    pub fn set_camera_fov(&mut self, value: i32) {
        self.project.global_camera_mut().set_fov(value);
        self.project.update_global_camera();
        self.project.reset_all_model_edges(&HashMap::new());
    }

    pub fn set_camera_look_at(&mut self, value: Vector3<f32>) {
        self.project.global_camera_mut().set_look_at(value);
        self.project.update_global_camera();
        self.project.reset_all_model_edges(&HashMap::new());
    }

    pub fn set_light_color(&mut self, value: Vector3<f32>) {
        self.project.global_light_mut().set_color(value);
    }

    pub fn set_light_direction(&mut self, value: Vector3<f32>) {
        self.project.global_light_mut().set_direction(value);
    }

    pub fn set_model_bone_orientation(
        &mut self,
        model_handle: Option<ModelHandle>,
        bone_name: &str,
        value: Quaternion<f32>,
    ) {
        if let Some(model) = match model_handle {
            Some(handle) => self.project.model_mut(handle),
            None => self.project.active_model_mut(),
        } {
            if let Some(bone) = model.find_bone_mut(bone_name) {
                bone.local_user_orientation = value;
                self.project.perform_model_bones_transform(model_handle);
            }
        }
    }

    pub fn set_model_bone_translation(
        &mut self,
        model_handle: Option<ModelHandle>,
        bone_name: &str,
        value: Vector3<f32>,
    ) {
        if let Some(model) = match model_handle {
            Some(handle) => self.project.model_mut(handle),
            None => self.project.active_model_mut(),
        } {
            if let Some(bone) = model.find_bone_mut(bone_name) {
                bone.local_user_translation = value;
                self.project.perform_model_bones_transform(model_handle);
            }
        }
    }

    pub fn set_model_morph_weight(
        &mut self,
        model_handle: Option<ModelHandle>,
        morph_name: &str,
        value: f32,
    ) {
        if let Some(model) = match model_handle {
            Some(handle) => self.project.model_mut(handle),
            None => self.project.active_model_mut(),
        } {
            if let Some(morph) = model.find_morph_mut(morph_name) {
                morph.set_weight(value);
            }
        }
    }

    // pub fn register_all_selected_bone_keyframes(
    //     &mut self,
    //     model_handle: Option<ModelHandle>,
    //     bone_names: &[&str],
    // ) {
    //     let bones = bone_names
    //         .iter()
    //         .map(|bone_name| {
    //             (
    //                 (*bone_name).to_owned(),
    //                 vec![self.project.current_frame_index()],
    //             )
    //         })
    //         .collect::<HashMap<_, _>>();
    //     self.project.register_bone_keyframes(model_handle, &bones);
    // }

    pub fn load_texture(
        &mut self,
        key: &str,
        data: &[u8],
        update_bind: bool,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if let Some(format) =
            image::ImageFormat::from_extension(key.split('.').rev().next().unwrap())
        {
            let img = image::io::Reader::with_format(Cursor::new(data), format)
                .decode()
                .unwrap();
            self.load_decoded_texture(
                key,
                &img.to_rgba8(),
                img.dimensions(),
                update_bind,
                device,
                queue,
            );
        } else {
            log::warn!("Texture File {} Not supported", key);
        }
    }

    pub fn load_decoded_texture(
        &mut self,
        key: &str,
        data: &[u8],
        dimensions: (u32, u32),
        update_bind: bool,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.project
            .load_texture(key, data, dimensions, update_bind, device, queue);
    }

    pub fn update_bind_texture(&mut self, device: &wgpu::Device) {
        self.project.update_bind_texture(device);
    }
}

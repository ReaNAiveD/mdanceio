use std::sync::Arc;

use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Vector2, Vector3};

use crate::{
    camera::{Camera, PerspectiveCamera},
    light::{DirectionalLight, Light},
    utils::lerp_f32, graphics::ClearPass,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageMode {
    None,
    Type1,
    Type2,
}

impl From<u32> for CoverageMode {
    fn from(v: u32) -> Self {
        match v {
            1 => Self::Type1,
            2 => Self::Type2,
            _ => Self::None,
        }
    }
}

impl From<CoverageMode> for u32 {
    fn from(mode: CoverageMode) -> Self {
        match mode {
            CoverageMode::None => 0,
            CoverageMode::Type1 => 1,
            CoverageMode::Type2 => 2,
        }
    }
}

pub struct ShadowCamera {
    shadow_color_texture: wgpu::TextureView,
    bind_group: Arc<wgpu::BindGroup>,
    // fallback_color_texture: wgpu::Texture,
    shadow_depth_texture: wgpu::TextureView,
    // fallback_depth_texture: wgpu::Texture,
    clear_pass: ClearPass,
    texture_size: Vector2<u32>,
    coverage_mode: CoverageMode,
    distance: f32,
    enabled: bool,
    dirty: bool,
}

impl ShadowCamera {
    pub const PASS_NAME: &'static str = "@mdanceio/ShadowCamera/Pass";
    pub const COLOR_IMAGE_NAME: &'static str = "@mdanceio/ShadowCamera/ColorImage";
    pub const DEPTH_IMAGE_NAME: &'static str = "@mdanceio/ShadowCamera/DepthImage";
    pub const MAXIMUM_DISTANCE: f32 = 10000f32;
    pub const MINIMUM_DISTANCE: f32 = 0f32;
    pub const INITIAL_DISTANCE: f32 = 8875f32;
    pub const INITIAL_TEXTURE_SIZE: u32 = 2048;

    pub fn new(
        bind_group_layout: &wgpu::BindGroupLayout,
        shadow_sampler: &wgpu::Sampler,
        device: &wgpu::Device,
    ) -> Self {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowCamera/Color"),
            size: wgpu::Extent3d {
                width: Self::INITIAL_TEXTURE_SIZE,
                height: Self::INITIAL_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowCamera/Depth"),
            size: wgpu::Extent3d {
                width: Self::INITIAL_TEXTURE_SIZE,
                height: Self::INITIAL_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth16Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowCamera/BindGroup/Texture"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
            ],
        });
        let clear_pass = ClearPass::new(
            &[Some(wgpu::TextureFormat::R32Float)],
            Some(wgpu::TextureFormat::Depth16Unorm),
            device,
        );
        Self {
            shadow_color_texture: color_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            bind_group: Arc::new(bind_group),
            // fallback_color_texture,
            shadow_depth_texture: depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            // fallback_depth_texture,
            clear_pass,
            texture_size: Vector2::new(Self::INITIAL_TEXTURE_SIZE, Self::INITIAL_TEXTURE_SIZE),
            coverage_mode: CoverageMode::Type1,
            distance: Self::INITIAL_DISTANCE,
            enabled: true,
            dirty: false,
        }
    }

    pub fn clear(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.clear_pass.draw(
            &[Some(&self.shadow_color_texture)],
            Some(&self.shadow_depth_texture),
            device,
            queue,
        );
    }

    fn get_view_matrix(
        &self,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
    ) -> Matrix4<f32> {
        let camera_direction = camera.direction();
        let light_direction = light.direction().normalize();
        let raw_light_view_x = camera_direction.cross(light_direction);
        let mut view_length = raw_light_view_x.magnitude();
        if view_length.abs() < f32::EPSILON {
            view_length = 1f32;
        }
        let light_view_x = (raw_light_view_x / view_length).normalize();
        let light_view_y = light_direction.cross(light_view_x);
        let light_view_matrix3 = Matrix3 {
            x: light_view_x,
            y: light_view_y,
            z: light_direction,
        };
        let light_view_matrix3_t = light_view_matrix3.transpose();
        let light_view_matrix4: Matrix4<f32> = light_view_matrix3_t.into();
        let light_view_origin = camera.position() - light.direction() * 50f32;
        light_view_matrix4 * Matrix4::from_translation(-light_view_origin)
    }

    fn get_projection_matrix(
        &self,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
    ) -> Matrix4<f32> {
        let camera_direction = camera.direction();
        let light_direction = light.direction().normalize();
        let distance = (10000f32 - self.distance) / 100000f32;
        let angle = camera_direction.dot(light_direction).abs();
        let half_distance = distance * 0.5f32;
        let distance_x0_15 = distance * 0.15f32;
        let (c0, c1, c2, mut projection) = match self.coverage_mode() {
            CoverageMode::Type2 => {
                let distance_x3 = distance * 3f32;
                (
                    3.0f32,
                    -4.7f32,
                    1.8f32,
                    Matrix4::new(
                        distance_x3,
                        0f32,
                        0f32,
                        0f32,
                        0f32,
                        distance_x3,
                        distance * 1.5f32,
                        distance_x3,
                        0f32,
                        0f32,
                        distance_x0_15,
                        0f32,
                        0f32,
                        -1f32,
                        0f32,
                        1f32,
                    ),
                )
            }
            _ => {
                let distance_x2 = distance * 2f32;
                (
                    1.0f32,
                    -1.3f32,
                    0.4f32,
                    Matrix4::new(
                        distance_x2,
                        0f32,
                        0f32,
                        0f32,
                        0f32,
                        distance_x2,
                        half_distance,
                        distance,
                        0f32,
                        0f32,
                        distance_x0_15,
                        0f32,
                        0f32,
                        -1f32,
                        0f32,
                        1f32,
                    ),
                )
            }
        };
        if angle > 0.9f32 {
            let one_minus_angle = 1f32 - angle;
            projection = Matrix4::new(
                distance,
                0f32,
                0f32,
                0f32,
                0f32,
                distance,
                half_distance * one_minus_angle,
                distance * one_minus_angle,
                0f32,
                0f32,
                distance_x0_15,
                0f32,
                0f32,
                angle - 1f32,
                0f32,
                1f32,
            );
        } else if angle > 0.8f32 {
            let t = 10f32 * (angle - 0.8f32);
            projection[0][0] = Self::lerp(projection[0][0], distance, t);
            projection[1][1] = Self::lerp(projection[1][1], distance, t);
            projection[1][3] = distance * (c0 + c1 * t + c2 * t * t);
            projection[1][2] = 0.5f32 * projection[1][3];
            projection[3][1] = Self::lerp(-1.0f32, -0.1f32, t);
        }
        projection
    }

    pub fn get_view_projection(
        &self,
        camera: &PerspectiveCamera,
        light: &DirectionalLight,
    ) -> (Matrix4<f32>, Matrix4<f32>) {
        (
            self.get_view_matrix(camera, light),
            self.get_projection_matrix(camera, light),
        )
    }

    pub fn get_crop_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_translation(Vector3::new(0.5f32, 0.5f32, 0.5f32))
            * Matrix4::from_nonuniform_scale(0.5f32, -0.5f32, 0.5f32)
    }

    pub fn image_size(&self) -> Vector2<u32> {
        self.texture_size
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn coverage_mode(&self) -> CoverageMode {
        self.coverage_mode
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn bind_group(&self) -> &Arc<wgpu::BindGroup> {
        &self.bind_group
    }

    pub fn color_image(&self) -> &wgpu::TextureView {
        &self.shadow_color_texture
    }

    pub fn depth_image(&self) -> &wgpu::TextureView {
        &self.shadow_depth_texture
    }

    pub fn set_distance(&mut self, value: f32) {
        if value != self.distance {
            self.distance = value.clamp(Self::MAXIMUM_DISTANCE, Self::MAXIMUM_DISTANCE);
            self.dirty = true;
            // TODO: publish event
        }
    }

    pub fn set_coverage_mode(&mut self, value: CoverageMode) {
        if value != self.coverage_mode {
            self.coverage_mode = value;
            self.dirty = true;
            // TODO: publish coverage mode
        }
    }

    pub fn set_enabled(&mut self, value: bool) {
        if value != self.enabled {
            self.enabled = value;
            // invalidate textures when disable
            self.dirty = true;
            // TODO: publish event
        }
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }
}

impl ShadowCamera {
    fn lerp(start: f32, end: f32, t: f32) -> f32 {
        // TODO: alter to a lerp with high effic
        lerp_f32(start, end, t)
    }
}

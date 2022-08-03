use cgmath::{
    InnerSpace, Matrix, Matrix3, Matrix4, SquareMatrix, Vector1, Vector2, Vector3, Vector4,
    VectorSpace,
};

use crate::{camera::{Camera, PerspectiveCamera}, clear_pass::ClearPass, project::Project, utils::lerp_f32, light::{DirectionalLight, Light}};

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
    // Project *m_project;
    // sg_pass m_shadowPass;
    // sg_pass_desc m_shadowPassDesc;
    // sg_pass m_fallbackPass;
    // sg_pass_desc m_fallbackPassDesc;
    shadow_color_texture: wgpu::Texture,
    fallback_color_texture: wgpu::Texture,
    shadow_depth_texture: wgpu::Texture,
    fallback_depth_texture: wgpu::Texture,
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

    pub fn new(device: &wgpu::Device) -> Self {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowCamera/color"),
            size: wgpu::Extent3d {
                width: Self::INITIAL_TEXTURE_SIZE as u32,
                height: Self::INITIAL_TEXTURE_SIZE as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowCamera/depth"),
            size: wgpu::Extent3d {
                width: Self::INITIAL_TEXTURE_SIZE as u32,
                height: Self::INITIAL_TEXTURE_SIZE as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let fallback_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("@mdanceio/ShadowCamera/FallbackColorImage"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let fallback_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("@mdanceio/ShadowCamera/FallbackDepthImage"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        Self {
            shadow_color_texture: color_texture,
            fallback_color_texture,
            shadow_depth_texture: depth_texture,
            fallback_depth_texture,
            texture_size: Vector2::new(Self::INITIAL_TEXTURE_SIZE, Self::INITIAL_TEXTURE_SIZE),
            coverage_mode: CoverageMode::Type1,
            distance: Self::INITIAL_DISTANCE,
            enabled: true,
            dirty: false,
        }
    }

    pub fn clear(&mut self, clear_pass: &ClearPass, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let pipeline = clear_pass.get_pipeline(
            &[wgpu::TextureFormat::R32Float],
            wgpu::TextureFormat::Depth24PlusStencil8,
            device,
        );
        encoder.push_debug_group("ShadowCamera::clear");
        {
            let color_texture_view = self
                .shadow_color_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let depth_texture_view = self
                .shadow_depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ShadowCamera/Clear/Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1f32),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }),
            });
            _render_pass.set_vertex_buffer(0, clear_pass.vertex_buffer.slice(..));
            _render_pass.set_pipeline(&pipeline);
            _render_pass.draw(0..4, 0..1);
        }
        encoder.pop_debug_group();
        queue.submit(Some(encoder.finish()));
    }

    pub fn resize(&mut self, size: Vector2<u32>, device: &wgpu::Device) {
        if size != self.texture_size {
            self.texture_size = size.map(|s| s.max(256u32));
            self.update(device)
        }
    }

    pub fn update(&mut self, device: &wgpu::Device) {
        if self.enabled {
            let color_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ShadowCamera/color"),
                size: wgpu::Extent3d {
                    width: self.texture_size.x as u32,
                    height: self.texture_size.y as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ShadowCamera/color"),
                size: wgpu::Extent3d {
                    width: self.texture_size.x as u32,
                    height: self.texture_size.y as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            // TODO: register render pass
        }
    }

    fn get_view_matrix(&self, camera: &PerspectiveCamera, light: &DirectionalLight) -> Matrix4<f32> {
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

    fn get_projection_matrix(&self, camera: &PerspectiveCamera, light: &DirectionalLight) -> Matrix4<f32> {
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

    pub fn get_view_projection(&self, camera: &PerspectiveCamera, light: &DirectionalLight) -> (Matrix4<f32>, Matrix4<f32>) {
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

    pub fn color_image(&self) -> &wgpu::Texture {
        &self.shadow_color_texture
    }

    pub fn color_texture_view(&self) -> wgpu::TextureView {
        self.shadow_color_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn depth_image(&self) -> &wgpu::Texture {
        &self.shadow_depth_texture
    }

    pub fn depth_texture_view(&self) -> wgpu::TextureView {
        self.shadow_depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
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

use crate::{base_application_service::BaseApplicationService, injector::Injector};

pub struct AndroidGlProxy {
    size: wgpu::Extent3d,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,

    application: BaseApplicationService,
}

impl AndroidGlProxy {
    pub async fn init(width: u32, height: u32) -> Self {
        let texture_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let instance = wgpu::Instance::new(wgpu::Backends::GL);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("AndroidGlRendererDevice"),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let service = BaseApplicationService::new(
            &adapter,
            &device,
            &queue,
            Injector {
                pixel_format: texture_format,
                viewport_size: [width, height],
            },
        );
        Self {
            size,
            instance,
            adapter,
            adapter_info,
            device,
            queue,
            application: service,
        }
    }

    pub fn draw_with_world_and_camera(
        &mut self,
        world_matrix: &[f32],
        camera_view_matrix: &[f32],
        camera_projection_matrix: &[f32],
    ) {
        let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let color_view = {
            let hal_texture_color =
                <wgpu_hal::api::Gles as wgpu_hal::Api>::Texture::default_framebuffer(color_format);
            let color_texture = unsafe {
                self.device.create_texture_from_hal::<wgpu_hal::api::Gles>(
                    hal_texture_color,
                    &wgpu::TextureDescriptor {
                        label: Some("AndroidDefaultColorTexture"),
                        size: self.size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: color_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    },
                )
            };
            color_texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let depth_view = {
            let hal_texture_depth =
                <wgpu_hal::api::Gles as wgpu_hal::Api>::Texture::default_framebuffer(depth_format);
            let depth_texture = unsafe {
                self.device.create_texture_from_hal::<wgpu_hal::api::Gles>(
                    hal_texture_depth,
                    &wgpu::TextureDescriptor {
                        label: Some("AndroidDefaultColorTexture"),
                        size: self.size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: depth_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    },
                )
            };
            depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        self.application.draw_default_pass_with_depth(
            &color_view,
            &depth_view,
            &self.device,
            &self.queue,
        );
    }
}

// include!(concat!(env!("OUT_DIR"), "/android_gl_proxy.uniffi.rs"));
use std::sync::{Arc, Mutex};

use crate::{
    base_application_service::BaseApplicationService, error::MdanceioError, injector::Injector,
};

#[cfg(target_os = "android")]
fn create_main_views(
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
    color_format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let mut texture_desc = wgpu::TextureDescriptor {
        label: Some("AndroidGlTextureDescriptor"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Uint, // dummy
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    };
    let color_view = {
        let color_hal_texture =
            <wgpu_hal::api::Gles as wgpu_hal::Api>::Texture::default_framebuffer(color_format);
        texture_desc.format = color_format;
        let color_texture = unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Gles>(color_hal_texture, &texture_desc)
        };
        color_texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    let depth_view = {
        let depth_format = wgpu::TextureFormat::Depth16Unorm;
        let depth_hal_texture =
            <wgpu_hal::api::Gles as wgpu_hal::Api>::Texture::default_framebuffer(depth_format);
        texture_desc.format = depth_format;
        let depth_texture = unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Gles>(depth_hal_texture, &texture_desc)
        };
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    (color_view, depth_view)
}

#[derive(Debug, thiserror::Error)]
pub enum MdanceioAndroidError {
    #[error("{0:?}")]
    CommonError(MdanceioError),
}

impl From<MdanceioError> for MdanceioAndroidError {
    fn from(e: MdanceioError) -> Self {
        Self::CommonError(e)
    }
}

#[cfg(not(target_os = "android"))]
pub struct AndroidProxy {}

#[cfg(not(target_os = "android"))]
impl AndroidProxy {
    pub fn new(width: u32, height: u32) -> Self {
        Self {}
    }

    pub fn redraw(&self) {}

    pub fn play(&self) {}

    pub fn load_model(&self, data: &Vec<u8>) -> Result<(), MdanceioAndroidError> {
        Ok(())
    }

    pub fn load_model_motion(&self, data: &Vec<u8>) -> Result<(), MdanceioAndroidError> {
        Ok(())
    }

    pub fn load_texture(&self, key: String, data: &Vec<u8>, update_bind: bool) {}

    pub fn update_bind_texture(&self) {}
}

#[cfg(target_os = "android")]
pub struct AndroidProxy {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,

    color_format: wgpu::TextureFormat,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    service: Arc<Mutex<BaseApplicationService>>,
}

#[cfg(target_os = "android")]
impl AndroidProxy {
    pub fn new(width: u32, height: u32) -> Self {
        android_logger::init_once(
            android_logger::Config::default()
                .with_min_level(log::Level::Debug) // limit log level
                .with_tag("MDanceIO"),
        );

        let mut task_pool = futures::executor::LocalPool::new();
        let any_egl = unsafe { egl::DynamicInstance::load().expect("unable to load libEGL") };

        log::info!("EGL version is {}", any_egl.version());
        let egl = match any_egl.upcast::<egl::EGL1_5>() {
            Some(egl) => egl,
            None => panic!("EGL 1.5 or greater is required"),
        };
        log::info!("Hooking up to wgpu-hal");
        let exposed = unsafe {
            <wgpu_hal::api::Gles as wgpu_hal::Api>::Adapter::new_external(|name| {
                egl.get_proc_address(name)
                    .map_or(std::ptr::null(), |p| p as *const _)
            })
        }
        .expect("GL adapter can't be initialized");
        let instance = wgpu::Instance::new(wgpu::Backends::empty());
        let adapter = unsafe { instance.create_adapter_from_hal(exposed) };
        let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let limits = {
            let adapter_limits = adapter.limits();
            let desired_height = 16 << 10;
            wgpu::Limits {
                max_texture_dimension_2d: if adapter_limits.max_texture_dimension_2d
                    < desired_height
                {
                    println!(
                        "Adapter only supports {} texutre size, main levels are not compatible",
                        adapter_limits.max_texture_dimension_2d
                    );
                    adapter_limits.max_texture_dimension_2d
                } else {
                    desired_height
                },
                ..wgpu::Limits::downlevel_webgl2_defaults()
            }
        };

        let (device, queue) = task_pool
            .run_until(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("AndroidGlDevice"),
                    features: wgpu::Features::empty(),
                    limits,
                },
                None,
            ))
            .expect("fail to get renderer device");
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let (color_view, depth_view) = create_main_views(&device, extent, color_format);
        let service = BaseApplicationService::new(
            // &config,
            &adapter,
            &device,
            &queue,
            Injector {
                pixel_format: color_format,
                viewport_size: [width, height],
            },
        );
        Self {
            instance,
            device,
            queue,
            color_format,
            color_view,
            depth_view,
            service: Arc::new(Mutex::new(service)),
        }
    }

    pub fn redraw(&self) {
        self.service
            .lock()
            .expect("unable to get service draw lock")
            .draw_default_pass(&self.color_view, &self.device, &self.queue);
    }

    pub fn play(&self) {
        self.service
            .lock()
            .expect("unable to get service draw lock")
            .play();
    }

    pub fn load_model(&self, data: &Vec<u8>) -> Result<(), MdanceioAndroidError> {
        let model_result = self
            .service
            .lock()
            .expect("unable to get service draw lock")
            .load_model(data, &self.device, &self.queue);
        match model_result {
            Ok(handle) => {
                let _ = self
                    .service
                    .lock()
                    .expect("unable to get service draw lock")
                    .enable_shadow_map(handle, true);
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn load_model_motion(&self, data: &Vec<u8>) -> Result<(), MdanceioAndroidError> {
        self.service
            .lock()
            .expect("unable to get service draw lock")
            .load_model_motion(data)
            .map(|_| ())
            .map_err(|e| e.into())
    }

    pub fn load_texture(&self, key: String, data: &Vec<u8>, update_bind: bool) {
        self.service
            .lock()
            .expect("unable to get service draw lock")
            .load_texture(&key, data, update_bind, &self.device, &self.queue);
    }

    pub fn update_bind_texture(&self) {
        self.service
            .lock()
            .expect("unable to get service draw lock")
            .update_bind_texture(&self.device)
    }
}

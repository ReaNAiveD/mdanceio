use console_error_panic_hook;
use console_log;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures;

use crate::base_application_service::BaseApplicationService;
use crate::injector::Injector;

pub struct CanvasSize<T> {
    pub width: T,
    pub height: T,
}

impl<T> CanvasSize<T> {
    pub fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

#[wasm_bindgen]
pub struct WasmClient {
    instance: wgpu::Instance,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,

    service: BaseApplicationService,
}

/// 我希望通过WasmClient为JS层调用提供接口。JS层应自行处理渲染请求频率，通知视口resize，点击，拖动等事件，并处理好可能有的回调。
#[wasm_bindgen]
impl WasmClient {
    pub fn new(canvas: web_sys::HtmlCanvasElement) -> js_sys::Promise {
        let level: log::Level = log::Level::Trace;
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        log::info!("Initializing the surface...");

        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let size = CanvasSize::new(canvas.width(), canvas.height());
        let surface = instance
            .create_surface_from_canvas(canvas)
            .expect("Failed when creating surface!");
        wasm_bindgen_futures::future_to_promise(async move {
            let adapter = wgpu::util::initialize_adapter_from_env_or_default(
                &instance,
                backends,
                Some(&surface),
            )
            .await
            .expect("No suitable GPU adapters found on the system!");

            let adapter_info = adapter.get_info();
            log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

            let trace_dir = std::env::var("WGPU_TRACE");
            log::info!("Getting Render Device and Queue..");
            let features = adapter.features();
            let limits = adapter.limits();
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features,
                        limits,
                    },
                    trace_dir.ok().as_ref().map(std::path::Path::new),
                )
                .await
                .expect("Unable to find a suitable GPU adapter!");

            log::info!("Configuring Surface...");
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(surface_caps.formats[0]);
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };
            surface.configure(&device, &config);

            log::info!("Building MDanceIO Service...");
            let service = BaseApplicationService::new(
                // &config,
                &adapter,
                &device,
                &queue,
                Injector {
                    pixel_format: surface_format,
                    viewport_size: [size.width, size.height],
                },
            );

            Ok(WasmClient {
                instance,
                surface,
                config,
                adapter,
                adapter_info,
                device,
                queue,
                service,
            }
            .into())
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn redraw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.service
            .draw_default_pass(&view, &self.device, &self.queue);
        frame.present();
    }

    pub fn load_model(&mut self, data: &[u8]) -> Result<(), JsValue> {
        match self.service.load_model(data, &self.device, &self.queue) {
            Ok(handle) => {
                let _ = self.service.enable_shadow_map(handle, true);
                Ok(())
            }
            Err(e) => Err(e.to_string().into()),
        }
    }

    pub fn load_model_motion(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.service
            .load_model_motion(data)
            .map_err(|e| e.to_string().into())
    }

    pub fn load_camera_motion(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.service
            .load_camera_motion(data)
            .map_err(|e| e.to_string().into())
    }

    pub fn load_light_motion(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.service
            .load_light_motion(data)
            .map_err(|e| e.to_string().into())
    }

    pub fn get_texture_names(&self) -> Box<[JsValue]> {
        self.service
            .get_model_texture_paths(1)
            .iter()
            .map(|path| path.into())
            .collect()
    }

    pub fn load_texture(&mut self, key: &str, data: &[u8], update_bind: bool) {
        self.service
            .load_texture(key, data, update_bind, &self.device, &self.queue);
    }

    pub fn load_decoded_texture(
        &mut self,
        key: &str,
        data: &[u8],
        update_bind: bool,
        width: u32,
        height: u32,
    ) {
        self.service.load_decoded_texture(
            key,
            data,
            (width, height),
            update_bind,
            &self.device,
            &self.queue,
        );
    }

    pub fn update_bind_texture(&mut self) {
        self.service.update_bind_texture(&self.device);
    }

    pub fn seek(&mut self, frame_index: u32) {
        self.service.seek(frame_index);
    }

    pub fn update(&mut self) {
        self.service
            .update_current_project(&self.device, &self.queue);
    }

    pub fn play(&mut self) {
        self.service.play()
    }
}

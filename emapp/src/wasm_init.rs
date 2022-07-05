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
    pub fn new(canvas: &web_sys::HtmlCanvasElement) -> js_sys::Promise {
        // let query_string = web_sys::window().unwrap().location().search().unwrap();
        // let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
        //     .map(|x| x.parse().ok())
        //     .flatten()
        //     .unwrap_or(log::Level::Error);
        let level: log::Level = log::Level::Trace;
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        log::info!("Initializing the surface...");

        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());

        let instance = wgpu::Instance::new(wgpu::Backends::BROWSER_WEBGPU);
        let (size, surface) = unsafe {
            let size = CanvasSize::new(canvas.width(), canvas.height());
            let surface = instance.create_surface_from_canvas(&canvas);
            (size, surface)
        };
        wasm_bindgen_futures::future_to_promise(async move {
            let adapter = wgpu::util::initialize_adapter_from_env_or_default(
                &instance,
                backend,
                Some(&surface),
            )
            .await
            .expect("No suitable GPU adapters found on the system!");

            let adapter_info = adapter.get_info();
            log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

            let trace_dir = std::env::var("WGPU_TRACE");
            // TODO: wo may need to set feature and limit, ref to https://github.com/gfx-rs/wgpu/blob/master/wgpu/examples/framework.rs
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
                    },
                    trace_dir.ok().as_ref().map(std::path::Path::new),
                )
                .await
                .expect("Unable to find a suitable GPU adapter!");
            log::info!("Got Render Device and Queue");

            let surface_format = surface.get_supported_formats(&adapter)[0];
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Mailbox,
            };
            surface.configure(&device, &config);
            log::info!("Finish Configure Surface");

            let service = BaseApplicationService::new(
                &config,
                &adapter,
                &device,
                &queue,
                Injector {
                    pixel_format: surface_format,
                    window_device_pixel_ratio: 1.0f32,
                    viewport_device_pixel_ratio: 1.0f32,
                    window_size: [1, 1],
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
            .draw_default_pass(&view, &self.adapter, &self.device, &self.queue);
        frame.present();
    }

    pub fn load_model(&mut self, data: &[u8]) {
        self.service.load_model(data, &self.device);
    }

    pub fn load_texture(&mut self, key: &str, data: &[u8]) {
        self.service
            .load_texture(key, data, &self.device, &self.queue);
    }

    pub fn load_decoded_texture(&mut self, key: &str, data: &[u8], width: u32, height: u32) {
        self.service
            .load_decoded_texture(key, data, (width, height), &self.device, &self.queue);
    }

    pub fn update_bind_texture(&mut self) {
        self.service.update_bind_texture();
    }
}

/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}

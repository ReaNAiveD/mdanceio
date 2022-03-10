use std::future::Future;

use wasm_bindgen::prelude::*;

pub struct CanvasSize<T> {
    pub width: T,
    pub height: T,
}

impl<T> CanvasSize<T> {
    pub fn new(width: T, height: T) -> Self {
        Self {
            width,
            height,
        }
    }
}

#[wasm_bindgen]
pub struct WasmClient {
    instance: wgpu::Instance,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    adapt_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// 我希望通过WasmClient为JS层调用提供接口。JS层应自行处理渲染请求频率，通知视口resize，点击，拖动等事件，并处理好可能有的回调。
#[wasm_bindgen]
impl WasmClient {
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: &web_sys::HtmlCanvasElement) -> Result<Self, ()> {
        let query_string = web_sys::window().unwrap().location().search().unwrap();
        let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(log::Level::Error);
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        log::info!("Initializing the surface...");

        let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backend::all);

        let instance = wgpu::Instance::new(wgpu::Backends::BROWSER_WEBGPU);
        let (size, surface) = unsafe {
            let size = CanvasSize::new(canvas.width(), canvas.height());
            let surface = instance.create_surface_from_canvas(&canvas);
            (size, surface)
        };
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, Some(&surface))
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        surface.configure(&device, &config);
    

        Ok(Self {
            instance,
            surface,
            config,
            adapter,
            adapt_info,
            device,
            queue,
        })
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    #[wasm_bindgen]
    pub fn redraw(&self) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &config);
                self.surface.get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            },
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        // TODO: render project

        frame.present();
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

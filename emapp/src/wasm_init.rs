use std::future::Future;

pub struct WasmClient {
    window: winit::window::Window,
    event_loop: winit::event_loop::EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WasmClient {
    async fn new(canvas_element: &web_sys::HtmlCanvasElement) -> Self {
        let event_loop = winit::event_loop::EventLoop::new();
        let mut window_builder = winit::window::WindowBuilder::new();

        let window = window_builder.build(&event_loop).unwrap();

        {
            use winit::platform::web::WindowExtWebSys;
            let query_string = web_sys::window().unwrap().location().search().unwrap();
            let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
                .map(|x| x.parse().ok())
                .flatten()
                .unwrap_or(log::Level::Error);
            console_log::init_with_level(level).expect("could not initialize logger");
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| {
                    body.append_child(&web_sys::Element::from(window.canvas()))
                        .ok()
                })
                .expect("couldn't append canvas to document body");
        }

        log::info!("Initializing the surface...");

        let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backend::all);

        let instance = wgpu::Instance::new(backend);
        let (size, surface) = unsafe {
            let size = window.inner_size();
            let surface = instance.create_surface(&window);
            (size, surface)
        };
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, Some(&surface))
                .await
                .expect("No suitable GPU adapters found on the system!");

        {
            let adapter_info = adapter.get_info();
            log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
        }

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

        Self {
            window,
            event_loop,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
        }
    }

    pub fn start(&self) {
        let spawner = Spawner::new();
        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface.get_preferred_format(&self.adapter),
            width: self.size.width,
            height: self.size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        self.surface.configure(&self.device, &config);

        // Then init main Project

        self.event_loop.run(move |event, _, control_flow| {
            let _ = (&self.instance, &self.adapter); // force ownership by the closure
            *control_flow = if cfg!(feature = "metal-auto-capture") {
                winit::event_loop::ControlFlow::Exit
            } else {
                winit::event_loop::ControlFlow::Poll
            };
            match event {
                winit::event::Event::RedrawEventsCleared => {
                    // 重绘事件均已处理，请求下一次重绘。
                    self.window.request_redraw();
                }
                winit::event::Event::WindowEvent {
                    event:
                        winit::event::WindowEvent::Resized(size)
                        | winit::event::WindowEvent::ScaleFactorChanged {
                            new_inner_size: &mut size,
                            ..
                        },
                    ..
                } => {
                    log::info!("Resize to {:?}", size);
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    // TODO: Resize project
                    self.surface.configure(&self.device, &config);
                }
                winit::event::Event::RedrawRequested(_) => {
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
                },
                winit::event::Event::WindowEvent { window_id, event } => match event {
                    winit::event::WindowEvent::Resized(_) => todo!(),
                    winit::event::WindowEvent::Moved(_) => todo!(),
                    winit::event::WindowEvent::CloseRequested => todo!(),
                    winit::event::WindowEvent::Destroyed => todo!(),
                    winit::event::WindowEvent::DroppedFile(_) => todo!(),
                    winit::event::WindowEvent::HoveredFile(_) => todo!(),
                    winit::event::WindowEvent::HoveredFileCancelled => todo!(),
                    winit::event::WindowEvent::ReceivedCharacter(_) => todo!(),
                    winit::event::WindowEvent::Focused(_) => todo!(),
                    winit::event::WindowEvent::KeyboardInput {
                        device_id,
                        input,
                        is_synthetic,
                    } => todo!(),
                    winit::event::WindowEvent::ModifiersChanged(_) => todo!(),
                    winit::event::WindowEvent::CursorMoved {
                        device_id,
                        position,
                        modifiers,
                    } => todo!(),
                    winit::event::WindowEvent::CursorEntered { device_id } => todo!(),
                    winit::event::WindowEvent::CursorLeft { device_id } => todo!(),
                    winit::event::WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        modifiers,
                    } => todo!(),
                    winit::event::WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                        modifiers,
                    } => todo!(),
                    winit::event::WindowEvent::TouchpadPressure {
                        device_id,
                        pressure,
                        stage,
                    } => todo!(),
                    winit::event::WindowEvent::AxisMotion {
                        device_id,
                        axis,
                        value,
                    } => todo!(),
                    winit::event::WindowEvent::Touch(_) => todo!(),
                    winit::event::WindowEvent::ScaleFactorChanged {
                        scale_factor,
                        new_inner_size,
                    } => todo!(),
                    winit::event::WindowEvent::ThemeChanged(_) => todo!(),
                },
                winit::event::Event::NewEvents(_) => todo!(),
                winit::event::Event::DeviceEvent { device_id, event } => todo!(),
                winit::event::Event::UserEvent(_) => todo!(),
                winit::event::Event::Suspended => todo!(),
                winit::event::Event::Resumed => todo!(),
                winit::event::Event::MainEventsCleared => todo!(),
                winit::event::Event::LoopDestroyed => todo!(),
            }
        })
    }
}

pub fn run(title: &str) {
    use wasm_bindgen::{prelude::*, JsCast};

    let title = title.to_owned();
    wasm_bindgen_futures::spawn_local(async move {
        let setup = setup(&title).await;
        let start_closure = Closure::once_into_js(move || start(setup));

        // make sure to handle JS exceptions thrown inside start.
        // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
        // This is required, because winit uses JS exception for control flow to escape from `run`.
        if let Err(error) = call_catch(&start_closure) {
            let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                e.message().includes("Using exceptions for control flow", 0)
            });

            if !is_control_flow_exception {
                web_sys::console::error_1(&error);
            }
        }

        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
            fn call_catch(this: &JsValue) -> Result<(), JsValue>;
        }
    });
}
#[cfg(target_arch = "wasm32")]
pub struct Spawner {}

#[cfg(target_arch = "wasm32")]
impl Spawner {
    fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}

#[cfg(target_arch = "wasm32")]
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

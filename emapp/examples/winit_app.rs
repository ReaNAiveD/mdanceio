use emapp::base_application_service::BaseApplicationService;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct State {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    application: BaseApplicationService,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                Some(std::path::Path::new("target/wgpu-trace/winit-app")),
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let application = BaseApplicationService::new(
            &adapter,
            &device,
            &queue,
            emapp::injector::Injector {
                pixel_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                window_device_pixel_ratio: 1f32,
                viewport_device_pixel_ratio: 1f32,
                window_size: [size.width as u16, size.height as u16],
                viewport_size: [size.width as u16, size.height as u16],
            },
        );
        Self {
            surface,
            adapter,
            device,
            queue,
            config,
            size,
            application,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn load_sample_data(&mut self) -> Result<(), Box<dyn std::error::Error + 'static>> {
        let model_data = std::fs::read("emapp/tests/example/Alicia/MMD/Alicia_solid.pmx")?;
        self.application.load_model(&model_data, &self.device, &self.queue);
        drop(model_data);
        self.application.enable_model_shadow_map(true);
        let texture_dir = std::fs::read_dir("emapp/tests/example/Alicia/FBX/").unwrap();
        for texture_file in texture_dir {
            let texture_file = texture_file.unwrap();
            let texture_data = std::fs::read(texture_file.path())?;
            self.application.load_texture(
                texture_file.file_name().to_str().unwrap(),
                &texture_data,
                &self.device,
                &self.queue,
            );
        }
        self.application.update_bind_texture();
        let motion_data = std::fs::read("emapp/tests/example/Alicia/MMD Motion/2 for test 1.vmd")?;
        self.application.load_model_motion(&motion_data);
        self.application.seek(20);
        self.application.update_current_project(&self.device, &self.queue);
        drop(motion_data);
        Ok(())
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.application
            .draw_default_pass(&view, &self.device, &self.queue);
        output.present();
        Ok(())
    }
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    // let mut state = rt.block_on(async {State::new(&window).await});
    let mut state = State::new(&window).await;

    state.load_sample_data()?;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                // UPDATED!
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            // window.request_redraw();
        }
        _ => {}
    });
}

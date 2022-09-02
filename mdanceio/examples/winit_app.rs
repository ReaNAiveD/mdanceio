use std::path::PathBuf;

use log4rs::append::file::FileAppender;
use log4rs::encode::pattern::PatternEncoder;
use mdanceio::base_application_service::BaseApplicationService;
use winit::dpi::PhysicalSize;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct State {
    surface: wgpu::Surface,
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
            mdanceio::injector::Injector {
                pixel_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                viewport_size: [size.width, size.height],
            },
        );
        Self {
            surface,
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

    fn load_sample_data(
        &mut self,
        model_path: &PathBuf,
        texture_dir: &PathBuf,
        motion_path: Option<&PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error + 'static>> {
        log::info!("Loading Model...");
        let model_data = std::fs::read(model_path)?;
        let model_handle = self
            .application
            .load_model(&model_data, &self.device, &self.queue)?;
        drop(model_data);
        self.application.enable_shadow_map(model_handle, true)?;
        log::info!("Loading Texture...");
        let texture_dir = std::fs::read_dir(texture_dir).unwrap();
        for texture_file in texture_dir {
            let texture_first_entry = texture_file.unwrap();
            if texture_first_entry.metadata().unwrap().is_file() {
                let texture_data = std::fs::read(texture_first_entry.path())?;
                self.application.load_texture(
                    texture_first_entry.file_name().to_str().unwrap(),
                    &texture_data,
                    false,
                    &self.device,
                    &self.queue,
                );
            } else if texture_first_entry.metadata().unwrap().is_dir() {
                for texture_file in std::fs::read_dir(texture_first_entry.path()).unwrap() {
                    let texture_file = texture_file.unwrap();
                    if texture_file.metadata().unwrap().is_file() {
                        let texture_data = std::fs::read(texture_file.path())?;
                        self.application.load_texture(
                            format!(
                                "{}/{}",
                                texture_first_entry.file_name().to_str().unwrap(),
                                texture_file.file_name().to_str().unwrap()
                            )
                            .as_str(),
                            &texture_data,
                            false,
                            &self.device,
                            &self.queue,
                        );
                    }
                }
            }
        }
        self.application.update_bind_texture(&self.device);
        log::info!("Loading Motion...");
        if let Some(motion_path) = motion_path {
            let motion_data = std::fs::read(motion_path)?;
            self.application.load_model_motion(&motion_data)?;
        }
        self.application.play();
        Ok(())
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        log::debug!("Start rendering");
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
    let app = clap::App::new("mdrs")
        .version("0.1.0")
        .author("NAiveD <nice-die@live.com>")
        .setting(clap::AppSettings::DeriveDisplayOrder)
        .arg(
            clap::Arg::new("width")
                .long("width")
                .short('w')
                .default_value("800")
                .value_parser(clap::value_parser!(u32).range(1..))
                .help("Width of render target texture"),
        )
        .arg(
            clap::Arg::new("height")
                .long("height")
                .short('h')
                .default_value("600")
                .value_parser(clap::value_parser!(u32).range(1..))
                .help("Height of render target texture"),
        )
        .arg(
            clap::arg!(
                --model <FILE> "Path to the model to load"
            )
            .required(true)
            .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            clap::arg!(
                --texture <DIR> "Path to the texture directory"
            )
            .required(false)
            .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            clap::arg!(
                --motion <FILE> "Path to the motion to load"
            )
            .required(false)
            .value_parser(clap::value_parser!(PathBuf)),
        );

    let matches = app.clone().get_matches();
    let width: u32 = *matches
        .get_one("width")
        .expect("Parameter `width` is required. ");
    let height: u32 = *matches
        .get_one("height")
        .expect("Parameter `height` is required. ");
    let model_path = matches
        .get_one::<PathBuf>("model")
        .expect("Parameter `model` is required. ");
    let texture_dir = matches
        .get_one::<PathBuf>("texture")
        .cloned()
        .unwrap_or_else(|| {
            let mut dir = model_path.clone();
            dir.pop();
            dir
        });
    let motion_path = matches.get_one::<PathBuf>("motion");

    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S.%6f)} [{level}] - {m} [{file}:{line}]{n}",
        )))
        .build("target/log/output.log")?;
    let config = log4rs::config::Config::builder()
        .appender(log4rs::config::Appender::builder().build("logfile", Box::new(logfile)))
        .build(
            log4rs::config::Root::builder()
                .appender("logfile")
                .build(log::LevelFilter::Info),
        )?;
    log4rs::init_config(config)?;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize { width, height })
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    state.load_sample_data(model_path, &texture_dir, motion_path)?;

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
            window.request_redraw();
        }
        _ => {}
    });
}

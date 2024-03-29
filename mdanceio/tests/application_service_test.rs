use std::fs::File;
use std::io::Write;

use mdanceio::base_application_service::BaseApplicationService;
use mdanceio::injector::Injector;

struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

async fn create_png(
    png_output_path: &str,
    device: &wgpu::Device,
    output_buffer: &wgpu::Buffer,
    buffer_dimensions: &BufferDimensions,
    submission_index: wgpu::SubmissionIndex,
) {
    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    //
    // We pass our submission index so we don't need to wait for any other possible submissions.
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
    // If a file system is available, write the buffer as a PNG
    let has_file_system_available = cfg!(not(target_arch = "wasm32"));
    if !has_file_system_available {
        return;
    }

    if let Ok(Ok(())) = receiver.await {
        let padded_buffer = buffer_slice.get_mapped_range();

        let mut png_encoder = png::Encoder::new(
            File::create(png_output_path).unwrap(),
            buffer_dimensions.width as u32,
            buffer_dimensions.height as u32,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::Rgba);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(buffer_dimensions.unpadded_bytes_per_row)
            .unwrap();

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(buffer_dimensions.padded_bytes_per_row) {
            png_writer
                .write_all(&chunk[..buffer_dimensions.unpadded_bytes_per_row])
                .unwrap();
        }
        png_writer.finish().unwrap();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);

        output_buffer.unmap();
    }
}

#[tokio::test]
async fn render_frame_0() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let stdout = log4rs::append::console::ConsoleAppender::builder()
        .encoder(Box::new(log4rs::encode::pattern::PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S.%6f)} [{level}] - {m} [{file}:{line}]{n}",
        )))
        .build();
    let config = log4rs::config::Config::builder()
        .appender(log4rs::config::Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            log4rs::config::Root::builder()
                .appender("stdout")
                .build(log::LevelFilter::Info),
        )?;

    log4rs::init_config(config)?;

    let injector = Injector {
        pixel_format: wgpu::TextureFormat::Rgba8UnormSrgb,
        viewport_size: [1920, 1080],
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &Default::default(),
            Some(std::path::Path::new(
                "target/wgpu-trace/application-service-test",
            )),
        )
        .await
        .unwrap();

    let texture_desc = wgpu::TextureDescriptor {
        label: Some("TargetTexture"),
        size: wgpu::Extent3d {
            width: injector.viewport_size[0] as u32,
            height: injector.viewport_size[1] as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: injector.pixel_format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let texture = device.create_texture(&texture_desc);
    let texture_view = texture.create_view(&Default::default());

    let buffer_dimensions = BufferDimensions::new(
        injector.viewport_size[0] as usize,
        injector.viewport_size[1] as usize,
    );
    let output_buffer_size =
        (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height) as wgpu::BufferAddress;
    let output_buffer_desc = wgpu::BufferDescriptor {
        label: Some("OutputBuffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST
            // this tells wpgu that we want to read this buffer from the cpu
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    };
    let output_buffer = device.create_buffer(&output_buffer_desc);

    let mut application = BaseApplicationService::new(&adapter, &device, &queue, injector);

    let model_data = std::fs::read("tests/example/砂糖/砂糖.pmx")?;
    let model_handle = application.load_model(&model_data, &device, &queue)?;
    application.enable_shadow_map(model_handle, true)?;
    drop(model_data);
    let texture_dir = std::fs::read_dir("tests/example/砂糖").unwrap();
    for texture_file in texture_dir {
        let texture_first_entry = texture_file.unwrap();
        if texture_first_entry.metadata().unwrap().is_file() {
            let texture_data = std::fs::read(texture_first_entry.path())?;
            application.load_texture(
                texture_first_entry.file_name().to_str().unwrap(),
                &texture_data,
                false,
                &device,
                &queue,
            );
        } else if texture_first_entry.metadata().unwrap().is_dir() {
            for texture_file in std::fs::read_dir(texture_first_entry.path()).unwrap() {
                let texture_file = texture_file.unwrap();
                if texture_file.metadata().unwrap().is_file() {
                    let texture_data = std::fs::read(texture_file.path())?;
                    application.load_texture(
                        format!(
                            "{}/{}",
                            texture_first_entry.file_name().to_str().unwrap(),
                            texture_file.file_name().to_str().unwrap()
                        )
                        .as_str(),
                        &texture_data,
                        false,
                        &device,
                        &queue,
                    );
                }
            }
        }
    }
    application.update_bind_texture(&device);
    let motion_data = std::fs::read("tests/example/神里绫华_传说任务.vmd")?;
    application.load_model_motion(&motion_data);
    drop(motion_data);
    application.seek(609);
    application.update_current_project(&device, &queue);
    application.draw_default_pass(&texture_view, &device, &queue);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ReadBufferEncoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(buffer_dimensions.padded_bytes_per_row as u32),
                rows_per_image: None,
            },
        },
        texture_desc.size,
    );
    let submission_index = queue.submit(Some(encoder.finish()));

    create_png(
        "../target/image/1.png",
        &device,
        &output_buffer,
        &buffer_dimensions,
        submission_index,
    )
    .await;
    Ok(())
}

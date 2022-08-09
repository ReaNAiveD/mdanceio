use crate::{base_application_service::BaseApplicationService, injector::Injector};

pub struct OffscreenProxy {
    texture: wgpu::Texture,
    target: wgpu::TextureView,
    buffer_dimensions: BufferDimensions,
    target_buffer: wgpu::Buffer,

    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,

    service: BaseApplicationService,
}

impl OffscreenProxy {
    pub async fn init(width: u32, height: u32) -> Self {
        let texture_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
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
                    label: Some("OffscreenRendererDevice"),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let texture_desc = wgpu::TextureDescriptor {
            label: Some("OffscreenTarget"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let buffer_dimensions = BufferDimensions::new(width, height);
        let target_buffer_desc = wgpu::BufferDescriptor {
            label: Some("OffscreenTargetBuffer"),
            size: (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height)
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        };
        let target_buffer = device.create_buffer(&target_buffer_desc);

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
            texture,
            target: view,
            buffer_dimensions,
            target_buffer,
            instance,
            adapter,
            adapter_info,
            device,
            queue,
            service,
        }
    }

    pub fn redraw(&mut self) -> Vec<u8> {
        self.service
            .draw_default_pass(&self.target, &self.device, &self.queue);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("OffscreenBufferEncoder"),
            });
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &self.target_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(
                            self.buffer_dimensions.padded_bytes_per_row as u32,
                        )
                        .unwrap(),
                    ),
                    rows_per_image: std::num::NonZeroU32::new(self.buffer_dimensions.height),
                },
            },
            wgpu::Extent3d {
                width: self.buffer_dimensions.width,
                height: self.buffer_dimensions.height,
                depth_or_array_layers: 1,
            },
        );
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.target_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        let mut result = vec![];
        let mapped_buffer = buffer_slice.get_mapped_range();
        for chunk in mapped_buffer.chunks(self.buffer_dimensions.padded_bytes_per_row as usize) {
            result.extend(&chunk[..self.buffer_dimensions.unpadded_bytes_per_row as usize]);
        }
        return result;
    }
}

struct BufferDimensions {
    width: u32,
    height: u32,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
}

impl BufferDimensions {
    fn new(width: u32, height: u32) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>() as u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
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

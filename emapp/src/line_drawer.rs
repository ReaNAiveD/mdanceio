use std::{cell::RefCell, collections::HashMap, mem, iter};

use wgpu::util::DeviceExt;

use crate::{forward::LineVertexUnit, camera::Camera};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Uniform {
    view_project: [[f32; 4]; 4],
    color: [f32; 4],
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct LineDrawerCacheKey {
    primitive_type: wgpu::PrimitiveTopology,
    index_type: Option<wgpu::IndexFormat>,
}

pub struct LineDrawer {
    shader: wgpu::ShaderModule,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    primitive_type: wgpu::PrimitiveTopology,
    index_type: Option<wgpu::IndexFormat>, // None to disable indexed render
    pipelines: RefCell<HashMap<LineDrawerCacheKey, wgpu::RenderPipeline>>,
}

impl LineDrawer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("LineDrawer/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("resources/shaders/grid.wgsl").into()),
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LineDrawer/UniformBindGroup"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LineDrawer/PipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            shader,
            uniform_bind_group_layout,
            pipeline_layout,
            primitive_type: wgpu::PrimitiveTopology::LineList,
            index_type: None,
            pipelines: RefCell::new(HashMap::new()),
        }
    }

    pub fn draw(
        &self,
        color_attachment_view: &wgpu::TextureView,
        color_format: wgpu::TextureFormat,
        vertex_buffer: &wgpu::Buffer,
        num_vertices: u32,
        camera: &dyn Camera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let key = LineDrawerCacheKey {
            primitive_type: self.primitive_type,
            index_type: self.index_type,
        };
        let mut cache = self.pipelines.borrow_mut();
        let pipeline = cache.entry(key).or_insert_with(|| {
            let vertex_size = mem::size_of::<LineVertexUnit>();
            let vertex_buffer_layout = wgpu::VertexBufferLayout {
                array_stride: vertex_size as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 5,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 6,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm8x4,
                        offset: mem::size_of::<[f32; 3]>() as u64,
                        shader_location: 7,
                    },
                ],
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("LineDrawer/RenderPipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_buffer_layout],
                },
                primitive: wgpu::PrimitiveState {
                    topology: self.primitive_type,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: "ps_main",
                    targets: &[wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }],
                }),
                multiview: None,
            })
        });
        let (view, projection) = camera.get_view_transform();
        let uniform = Uniform {
            view_project: (projection * view).into(),
            color: [1.0f32, 1.0f32, 1.0f32, 1.0f32],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LineDrawer/UniformBuffer"),
            contents: bytemuck::bytes_of(&[uniform]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LineDrawer/UniformBindGroup"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LineDrawer/Encoder"),
        });
        {
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("LineDrawer/Pass"),
                color_attachments: &[
                    wgpu::RenderPassColorAttachment {
                        view: color_attachment_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    }
                ],
                depth_stencil_attachment: None,
            });
            _render_pass.set_pipeline(pipeline);
            _render_pass.set_bind_group(0, &uniform_bind_group, &[]);
            _render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            _render_pass.draw(0..num_vertices, 0..1);
        }
        queue.submit(iter::once(encoder.finish()));
    }
}

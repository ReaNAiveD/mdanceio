use std::{iter, mem};

use cgmath::{Matrix4, Vector4};
use wgpu::util::DeviceExt;

use crate::forward::LineVertexUnit;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Uniform {
    view_project: [[f32; 4]; 4],
    color: [f32; 4],
}

pub struct LineDrawer {
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    render_bundle: wgpu::RenderBundle,
    primitive_type: wgpu::PrimitiveTopology,
    texture_format: wgpu::TextureFormat,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
}

impl LineDrawer {
    pub fn new(
        vertices: &[LineVertexUnit],
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Self {
        let primitive_type = wgpu::PrimitiveTopology::LineList;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
        let render_pipeline = Self::build_pipeline(
            &shader,
            &pipeline_layout,
            primitive_type,
            texture_format,
            device,
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LineDrawer/UniformBuffer"),
            contents: bytemuck::bytes_of(&[Uniform::default()]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let num_vertices = vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LineDrawer/VertexBuffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let render_bundle = Self::build_render_bundle(
            &render_pipeline,
            &uniform_bind_group_layout,
            &uniform_buffer,
            &vertex_buffer,
            num_vertices,
            texture_format,
            device,
        );
        Self {
            shader,
            uniform_bind_group_layout,
            uniform_buffer,
            pipeline_layout,
            render_pipeline,
            primitive_type,
            texture_format,
            vertex_buffer,
            num_vertices,
            render_bundle,
        }
    }

    pub fn update_uniform(
        &self,
        view_project: Matrix4<f32>,
        color: Vector4<f32>,
        queue: &wgpu::Queue,
    ) {
        let uniform = Uniform {
            view_project: view_project.into(),
            color: color.into(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&[uniform]));
    }

    pub fn update_vertex_buffer(&self, vertices: &[LineVertexUnit], queue: &wgpu::Queue) {
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    pub fn replace_vertices(&mut self, vertices: &[LineVertexUnit], device: &wgpu::Device) {
        self.num_vertices = vertices.len() as u32;
        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LineDrawer/VertexBuffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.rebuild_render_bundle(device);
    }

    pub fn update_texture_format(
        &mut self,
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) {
        self.texture_format = texture_format;
        self.rebuild_pipeline(device);
    }

    pub fn update_primitive_topology(
        &mut self,
        primitive_topology: wgpu::PrimitiveTopology,
        device: &wgpu::Device,
    ) {
        self.primitive_type = primitive_topology;
        self.rebuild_pipeline(device);
    }

    pub fn draw(&self, color_view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LineDrawer/Encoder"),
        });
        {
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("LineDrawer/Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            _render_pass.execute_bundles(iter::once(&self.render_bundle));
        }
        queue.submit(iter::once(encoder.finish()));
    }
}

impl LineDrawer {
    fn build_vertex_buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        let vertex_size = mem::size_of::<LineVertexUnit>();
        wgpu::VertexBufferLayout {
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
        }
    }

    fn build_pipeline(
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        primitive_type: wgpu::PrimitiveTopology,
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LineDrawer/RenderPipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[Self::build_vertex_buffer_layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: primitive_type,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview: None,
        })
    }

    fn rebuild_pipeline(&mut self, device: &wgpu::Device) {
        self.render_pipeline = Self::build_pipeline(
            &self.shader,
            &self.pipeline_layout,
            self.primitive_type,
            self.texture_format,
            device,
        );
        self.rebuild_render_bundle(device);
    }

    fn build_render_bundle(
        render_pipeline: &wgpu::RenderPipeline,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        num_vertices: u32,
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LineDrawer/UniformBindGroup"),
            layout: uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("LineDrawer/RenderBundleEncoder"),
                color_formats: &[Some(texture_format)],
                depth_stencil: None,
                sample_count: 1,
                ..Default::default()
            });
        encoder.set_pipeline(render_pipeline);
        encoder.set_bind_group(0, &uniform_bind_group, &[]);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.draw(0..num_vertices, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("LineDrawer/RenderBundle"),
        })
    }

    fn rebuild_render_bundle(&mut self, device: &wgpu::Device) {
        self.render_bundle = Self::build_render_bundle(
            &self.render_pipeline,
            &self.uniform_bind_group_layout,
            &self.uniform_buffer,
            &self.vertex_buffer,
            self.num_vertices,
            self.texture_format,
            device,
        );
    }
}

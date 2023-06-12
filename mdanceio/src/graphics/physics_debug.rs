use std::mem;

use cgmath::Matrix4;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct VertexInput {
    position: [f32; 3],
    padding: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PhysicsDebugUniform {
    view_projection_matrix: [[f32; 4]; 4],
    color: [f32; 4],
}

pub struct PhysicsDrawerBuilder {
    shader: wgpu::ShaderModule,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bundle: wgpu::RenderBundle,
    color_format: wgpu::TextureFormat,
}

impl PhysicsDrawerBuilder {
    pub fn new(color_format: wgpu::TextureFormat, device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PhysicsDrawer/Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/physics_debug.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PhysicsDrawer/BindGroupLayout"),
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
            label: Some("PhysicsDrawer/PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PhysicsDrawer/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    attributes: &wgpu::vertex_attr_array![0=>Float32x3],
                    array_stride: mem::size_of::<[f32; 3]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::all(),
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhysicsDrawer/UniformBuffer"),
            contents: bytemuck::bytes_of(&[PhysicsDebugUniform::default()]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PhysicsDrawer/UniformBindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhysicsDrawer/VertexBuffer"),
            contents: bytemuck::cast_slice(&[VertexInput::default(), VertexInput::default()]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        });
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("physicsDrawer/RenderBundleEncoder"),
                color_formats: &[Some(color_format)],
                depth_stencil: None,
                sample_count: 1,
                multiview: None,
            });
        encoder.set_pipeline(&pipeline);
        encoder.set_bind_group(0, &uniform_bind_group, &[]);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.draw(0..2, 0..1);
        let bundle = encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("PhysicsDrawer/RenderBundle"),
        });
        Self {
            shader,
            uniform_buffer,
            vertex_buffer,
            pipeline,
            bundle,
            color_format,
        }
    }

    pub fn build<'a, 'b: 'a>(
        &'b self,
        view_projection: Matrix4<f32>,
        view: &'b wgpu::TextureView,
        device: &'b wgpu::Device,
        queue: &'b wgpu::Queue,
    ) -> PhysicsDrawer<'a> {
        PhysicsDrawer {
            device,
            queue,
            shader: &self.shader,
            bundle: &self.bundle,
            uniform_buffer: &self.uniform_buffer,
            vertex_buffer: &self.vertex_buffer,
            view_projection,
            view,
        }
    }
}

pub struct PhysicsDrawer<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub shader: &'a wgpu::ShaderModule,
    pub bundle: &'a wgpu::RenderBundle,
    pub uniform_buffer: &'a wgpu::Buffer,
    pub vertex_buffer: &'a wgpu::Buffer,
    pub view_projection: Matrix4<f32>,
    pub view: &'a wgpu::TextureView,
}

impl<'a> rapier3d::pipeline::DebugRenderBackend for PhysicsDrawer<'a> {
    fn draw_line(
        &mut self,
        _object: rapier3d::prelude::DebugRenderObject,
        a: rapier3d::prelude::Point<rapier3d::prelude::Real>,
        b: rapier3d::prelude::Point<rapier3d::prelude::Real>,
        color: [f32; 4],
    ) {
        let uniform = PhysicsDebugUniform {
            view_projection_matrix: self.view_projection.into(),
            color,
        };
        self.queue
            .write_buffer(self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
        let a = VertexInput {
            position: a.into(),
            padding: 0.,
        };
        let b = VertexInput {
            position: b.into(),
            padding: 0.,
        };
        self.queue
            .write_buffer(self.vertex_buffer, 0, bytemuck::cast_slice(&[a, b]));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PhysicsDrawer/CommandEncoder"),
            });{
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PhysicsDrawer/RenderPass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        render_pass.execute_bundles(std::iter::once(self.bundle));}
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

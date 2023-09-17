use std::iter;

use crate::forward::QuadVertexUnit;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ClearPassCacheKey(Vec<wgpu::TextureFormat>, wgpu::TextureFormat);

pub struct ClearPass {
    vertex_buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    render_bundle: wgpu::RenderBundle,
    color_formats: Vec<Option<wgpu::TextureFormat>>,
    depth_format: Option<wgpu::TextureFormat>,
}

impl ClearPass {
    pub fn new(
        color_formats: &[Option<wgpu::TextureFormat>],
        depth_format: Option<wgpu::TextureFormat>,
        device: &wgpu::Device,
    ) -> Self {
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("ClearPass/Vertices"),
                contents: bytemuck::cast_slice(&QuadVertexUnit::generate_quad_tri_strip()),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );
        let pipeline = Self::build_pipeline(color_formats, depth_format, device);
        let render_bundle = Self::build_render_bundle(
            &vertex_buffer,
            &pipeline,
            color_formats,
            depth_format,
            device,
        );
        Self {
            vertex_buffer,
            pipeline,
            render_bundle,
            color_formats: color_formats.to_vec(),
            depth_format,
        }
    }

    pub fn draw(
        &self,
        color_textures: &[Option<&wgpu::TextureView>],
        depth_texture: Option<&wgpu::TextureView>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ClearPass/CommandEncoder"),
        });
        let color_attachments = color_textures
            .iter()
            .map(|tv| {
                tv.as_ref().map(|tv| wgpu::RenderPassColorAttachment {
                    view: tv,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                })
            })
            .collect::<Vec<_>>();
        {
            let mut _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ClearPass/RenderPass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_texture.as_ref().map(|tv| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: tv,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1f32),
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            _rpass.execute_bundles(iter::once(&self.render_bundle));
        }
        queue.submit(Some(encoder.finish()));
    }

    fn build_pipeline(
        color_formats: &[Option<wgpu::TextureFormat>],
        depth_format: Option<wgpu::TextureFormat>,
        device: &wgpu::Device,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ClearPass/Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../resources/shaders/clear.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ClearPass/PipelineLayout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertexUnit>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4],
        };
        let color_target_state = color_formats
            .iter()
            .map(|format| {
                format.map(|format| wgpu::ColorTargetState {
                    format,
                    blend: if format == wgpu::TextureFormat::R32Float {
                        None
                    } else {
                        Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        })
                    },
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect::<Vec<_>>();
        let depth_stencil_state = depth_format.map(|depth_format| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ClearPass/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &color_target_state,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_stencil_state,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn build_render_bundle(
        vertex_buffer: &wgpu::Buffer,
        pipeline: &wgpu::RenderPipeline,
        color_formats: &[Option<wgpu::TextureFormat>],
        depth_format: Option<wgpu::TextureFormat>,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("ClearPass/RenderBundleEncoder"),
                color_formats,
                depth_stencil: depth_format.map(|format| wgpu::RenderBundleDepthStencil {
                    format,
                    depth_read_only: false,
                    stencil_read_only: true,
                }),
                sample_count: 1,
                ..Default::default()
            });
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.set_pipeline(pipeline);
        encoder.draw(0..4, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("ClearPass/RenderBundle"),
        })
    }
}

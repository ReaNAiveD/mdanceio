use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::forward::QuadVertexUnit;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ClearPassCacheKey(Vec<wgpu::TextureFormat>, wgpu::TextureFormat);

pub struct ClearPass {
    pipelines: RefCell<HashMap<ClearPassCacheKey, Rc<wgpu::RenderPipeline>>>,
    pub vertex_buffer: wgpu::Buffer,
}

impl ClearPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("@mdanceio/ClearPass/Vertices"),
                contents: bytemuck::cast_slice(&QuadVertexUnit::generate_quad_tri_strip()),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );
        Self {
            pipelines: RefCell::new(HashMap::new()),
            vertex_buffer,
        }
    }

    fn build_pipeline(
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("@mdanceio/ClearPass/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("resources/shaders/clear.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("@mdanceio/ClearPass/PipelineLayout"),
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
                Some(wgpu::ColorTargetState {
                    format: format.clone(),
                    blend: if *format == wgpu::TextureFormat::R32Float {
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
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("@mdanceio/ClearPass/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &color_target_state[..],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    pub fn get_pipeline(
        &self,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Rc<wgpu::RenderPipeline> {
        let key = ClearPassCacheKey(color_formats.to_vec(), depth_format);
        self.pipelines
            .borrow_mut()
            .entry(key.clone())
            .or_insert(Rc::new(Self::build_pipeline(
                color_formats,
                depth_format,
                device,
            )));
        self.pipelines.borrow().get(&key).unwrap().clone()
    }
}

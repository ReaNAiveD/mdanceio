use crate::graphics::model_program_bundle::ModelParametersUniform;

// pub struct Buffer<'a> {
//     pub num_indices: usize,
//     pub num_offset: usize,
//     pub vertex_buffer: &'a wgpu::Buffer,
//     pub index_buffer: &'a wgpu::Buffer,
//     pub depth_enabled: bool,
// }

// impl<'a> Buffer<'a> {
//     pub fn new(
//         num_indices: usize,
//         num_offset: usize,
//         vertex_buffer: &'a wgpu::Buffer,
//         index_buffer: &'a wgpu::Buffer,
//         depth_enabled: bool,
//     ) -> Self {
//         Self {
//             num_indices: usize::min(num_indices, 0x7fffffffusize),
//             num_offset: usize::min(num_offset, 0x7fffffffusize),
//             vertex_buffer,
//             index_buffer,
//             depth_enabled,
//         }
//     }

//     pub fn is_depth_enabled(&self) -> bool {
//         self.depth_enabled
//     }
// }

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct CPassDescription {
    pub color_texture_format: wgpu::TextureFormat,
    pub depth_texture_format: Option<wgpu::TextureFormat>,
    pub cull_mode: Option<wgpu::Face>,
    pub primitive_type: wgpu::PrimitiveTopology,
    pub color_blend: Option<wgpu::BlendState>,
    pub depth_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
}

pub struct CPassLayout<'a> {
    pub pipeline_layout: &'a wgpu::PipelineLayout,
    pub vertex_buffer_layout: wgpu::VertexBufferLayout<'a>,
}

pub struct CPassVertexBuffer<'a> {
    pub vertex_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
    pub num_offset: u32,
    pub num_indices: u32,
}

pub struct CPassBindGroup<'a> {
    pub color_bind: &'a wgpu::BindGroup,
    pub uniform_bind: &'a wgpu::BindGroup,
    pub shadow_bind: &'a wgpu::BindGroup,
}

pub struct CPass {
    pipeline: wgpu::RenderPipeline,
    color_texture_format: wgpu::TextureFormat,
    depth_texture_format: Option<wgpu::TextureFormat>,
    cull_mode: Option<wgpu::Face>,
    primitive_type: wgpu::PrimitiveTopology,
    color_blend: Option<wgpu::BlendState>,
    is_depth_enabled: bool,
}

impl CPass {
    pub fn new(
        shader: &wgpu::ShaderModule,
        desc: &CPassDescription,
        layout: &CPassLayout,
        device: &wgpu::Device,
    ) -> Self {
        let pipeline = Self::build_pipeline(shader, desc, layout, device);
        Self {
            pipeline,
            color_texture_format: desc.color_texture_format,
            depth_texture_format: desc.depth_texture_format,
            cull_mode: desc.cull_mode,
            primitive_type: desc.primitive_type,
            color_blend: desc.color_blend,
            is_depth_enabled: desc.depth_enabled,
        }
    }

    pub fn build_render_bundle(
        &self,
        material_idx: usize,
        bind: CPassBindGroup,
        vertex: CPassVertexBuffer,
        device: &wgpu::Device,
    ) -> wgpu::RenderBundle {
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("ModelProgramBundle/RenderBundleEncoder"),
                color_formats: &[Some(self.color_texture_format)],
                depth_stencil: self.depth_texture_format.map(|format| {
                    wgpu::RenderBundleDepthStencil {
                        format,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }
                }),
                sample_count: 1,
                multiview: None,
            });
        encoder.set_pipeline(&self.pipeline);
        encoder.set_bind_group(0, bind.color_bind, &[]);
        encoder.set_bind_group(
            1,
            bind.uniform_bind,
            &[(material_idx * std::mem::size_of::<ModelParametersUniform>()) as u32],
        );
        encoder.set_bind_group(2, bind.shadow_bind, &[]);
        encoder.set_vertex_buffer(0, vertex.vertex_buffer.slice(..));
        encoder.set_index_buffer(vertex.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        let vertex_indices = vertex.num_offset..(vertex.num_offset + vertex.num_indices);
        encoder.draw_indexed(vertex_indices, 0, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("ModelProgramBundle/RenderBundle"),
        })
    }
}

impl CPass {
    fn build_pipeline(
        shader: &wgpu::ShaderModule,
        desc: &CPassDescription,
        layout: &CPassLayout,
        device: &wgpu::Device,
    ) -> wgpu::RenderPipeline {
        let vertex_size = std::mem::size_of::<crate::model::VertexUnit>();
        // let texture_format = if technique.technique_type() == TechniqueType::Zplot {
        //     wgpu::TextureFormat::R32Float
        // } else {
        //     desc.color_texture_format
        // };
        let color_target_state = wgpu::ColorTargetState {
            format: desc.color_texture_format,
            blend: desc.color_blend,
            write_mask: wgpu::ColorWrites::ALL,
        };
        let depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth16Unorm,
            depth_write_enabled: desc.depth_enabled,
            depth_compare: desc.depth_compare,
            // depth_compare: if desc.depth_enabled {
            //     if technique.technique_type() == TechniqueType::Shadow {
            //         wgpu::CompareFunction::Less
            //     } else {
            //         wgpu::CompareFunction::LessEqual
            //     }
            // } else {
            //     wgpu::CompareFunction::Always
            // },
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ModelProgramBundle/Pipelines"),
            layout: Some(&layout.pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[layout.vertex_buffer_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(color_target_state)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: desc.primitive_type,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: desc.cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
}

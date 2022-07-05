use std::{collections::HashMap, mem};

use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    model::{Bone, Model, Morph, Vertex, VertexUnit},
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Argument {
    pub num_vertices: u32,
    pub num_max_morph_items: u32,
    pub edge_scale_factor: f32,
    pub padding: u32,
}

pub struct Deformer {
    shader: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    input_buffer: wgpu::Buffer,
    matrix_buffer: wgpu::Buffer,
    morph_weight_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    sdef_buffer: wgpu::Buffer,
    argument_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    num_groups: u32,
    num_vertices: usize,
    num_max_morph_items: usize,
}

impl Deformer {
    pub fn new(
        vertex_buffer_data: &[VertexUnit],
        vertices: &[Vertex],
        bones: &[Bone],
        fallback_bone: &Bone,
        morphs: &[Morph],
        edge_size: f32,
        device: &wgpu::Device,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader/Deformer"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("resources/shaders/model_skinning.wgsl").into(),
            ),
        });
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/InputBuffer"),
            contents: bytemuck::cast_slice(vertex_buffer_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let num_vertices = vertices.len();
        let mut vertex2morphs = vec![vec![]; num_vertices];
        for morph in morphs {
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Vertex(items) => {
                    for item in items {
                        if item.vertex_index >= 0 && (item.vertex_index as usize) < num_vertices {
                            vertex2morphs[item.vertex_index as usize].push((morph, item));
                        }
                    }
                }
                nanoem::model::ModelMorphType::Group(_)
                | nanoem::model::ModelMorphType::Bone(_)
                | nanoem::model::ModelMorphType::Texture(_)
                | nanoem::model::ModelMorphType::Uva1(_)
                | nanoem::model::ModelMorphType::Uva2(_)
                | nanoem::model::ModelMorphType::Uva3(_)
                | nanoem::model::ModelMorphType::Uva4(_)
                | nanoem::model::ModelMorphType::Material(_)
                | nanoem::model::ModelMorphType::Flip(_)
                | nanoem::model::ModelMorphType::Impulse(_) => {}
            }
        }
        let num_max_morph_items = vertex2morphs
            .iter()
            .map(|morphs| morphs.len())
            .max()
            .unwrap_or(0);
        let buffer_size = (num_max_morph_items * num_vertices).max(1);
        let mut vertex_deltas_buffer_data = vec![[0f32; 4]; num_max_morph_items * buffer_size];
        for (vertex_idx, morphs) in vertex2morphs.iter().enumerate() {
            for (idx, morph) in morphs.iter().enumerate() {
                let position = morph.1.position.0;
                let morph_idx = morph.0.origin.base.index + 1;
                vertex_deltas_buffer_data[vertex_idx * num_max_morph_items + idx] =
                    [position[0], position[1], position[2], morph_idx as f32];
            }
        }
        let vertex_deltas_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/VertexBuffer"),
            contents: bytemuck::cast_slice(&vertex_deltas_buffer_data[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut sdef_buffer_data = vec![[0f32; 4]; num_vertices * 3];
        for (idx, vertex) in vertices.iter().enumerate() {
            sdef_buffer_data[idx * 3 + 0] = vertex.origin.sdef_c.0;
            sdef_buffer_data[idx * 3 + 1] = vertex.origin.sdef_r0.0;
            sdef_buffer_data[idx * 3 + 2] = vertex.origin.sdef_r1.0;
        }
        let sdef_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/SdefBuffer"),
            contents: bytemuck::cast_slice(&sdef_buffer_data[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Deformer/BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Argument>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (bones.len().max(1) * mem::size_of::<[[f32; 4]; 4]>()) as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            ((morphs.len() + 1) * mem::size_of::<f32>()) as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (vertex_buffer_data.len() * mem::size_of::<VertexUnit>()) as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (sdef_buffer_data.len() * mem::size_of::<[f32; 4]>()) as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (vertex_deltas_buffer_data.len() * mem::size_of::<[f32; 4]>()) as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (vertex_buffer_data.len() * mem::size_of::<VertexUnit>()) as _,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Deformer/ComputePipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deformer/ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let argument = Argument {
            num_vertices: num_vertices as u32,
            num_max_morph_items: num_max_morph_items as u32,
            edge_scale_factor: edge_size,
            padding: 0,
        };
        let argument_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/ArgumentBuffer"),
            contents: bytemuck::bytes_of(&argument),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/MatrixBuffer"),
            contents: bytemuck::cast_slice(
                &Self::build_matrix_buffer_data(bones, fallback_bone, device)[..],
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let morph_weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/MorphBuffer"),
            contents: bytemuck::cast_slice(
                &Self::build_morph_weight_buffer_data(morphs, device)[..],
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let num_groups = ((num_vertices as f32) / 256f32).ceil() as u32 + 1;
        Self {
            shader,
            bind_group_layout,
            input_buffer,
            vertex_buffer: vertex_deltas_buffer,
            sdef_buffer,
            matrix_buffer,
            morph_weight_buffer,
            argument_buffer,
            pipeline,
            num_groups,
            num_vertices,
            num_max_morph_items,
        }
    }

    fn build_matrix_buffer_data(
        bones: &[Bone],
        fallback_bone: &Bone,
        device: &wgpu::Device,
    ) -> Vec<[[f32; 4]; 4]> {
        let mut matrix_buffer_data = vec![[[0f32; 4]; 4]; bones.len().max(1)];
        for (idx, bone) in bones.iter().enumerate() {
            matrix_buffer_data[idx] = bone.matrices.skinning_transform.into();
        }
        if bones.len() == 0 {
            matrix_buffer_data[0] = fallback_bone.matrices.skinning_transform.into();
        }
        matrix_buffer_data
    }

    fn build_morph_weight_buffer_data(morphs: &[Morph], device: &wgpu::Device) -> Vec<f32> {
        let mut morph_weight_buffer_data = vec![0f32; morphs.len() + 1];
        for (idx, morph) in morphs.iter().enumerate() {
            morph_weight_buffer_data[idx + 1] = morph.weight();
        }
        morph_weight_buffer_data
    }

    pub fn update_buffer(
        &self,
        model: &Model,
        camera: &dyn Camera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let argument = Argument {
            num_vertices: self.num_vertices as u32,
            num_max_morph_items: self.num_max_morph_items as u32,
            edge_scale_factor: model.edge_size(camera),
            padding: 0,
        };
        queue.write_buffer(&self.argument_buffer, 0, bytemuck::bytes_of(&argument));
        queue.write_buffer(
            &self.matrix_buffer,
            0,
            bytemuck::cast_slice(
                &Self::build_matrix_buffer_data(model.bones(), &model.shared_fallback_bone, device)
                    [..],
            ),
        );
        queue.write_buffer(
            &self.morph_weight_buffer,
            0,
            bytemuck::cast_slice(&Self::build_morph_weight_buffer_data(model.morphs(), device)[..]),
        );
    }

    pub fn execute(&self, output_buffer: &wgpu::Buffer, device: &wgpu::Device) {
        // let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Deformer/OutputBuffer"),
        //     size: (vertex_buffer.len() as u64) * (std::mem::size_of::<VertexUnit>() as u64),
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: true,
        // });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deformer/BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.argument_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.morph_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.sdef_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        command_encoder.push_debug_group("compute deform");
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Deformer/ComputePass"),
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(self.num_groups, 1, 1);
        }
        command_encoder.pop_debug_group();
    }
}

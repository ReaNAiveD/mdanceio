use std::mem;

use cgmath::{Matrix4, Quaternion, Vector3, VectorSpace, Zero};
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    model::{Bone, Model, VertexUnit, bone::BoneSet, vertex::VertexSet, morph::MorphSet},
    utils::{f128_to_vec3, mat4_truncate},
};

pub enum Deformer {
    Wgpu(Box<WgpuDeformer>),
    Software(CommonDeformer),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Argument {
    pub num_vertices: u32,
    pub num_max_morph_items: u32,
    pub edge_scale_factor: f32,
    pub padding: u32,
}

#[derive(Debug)]
pub struct WgpuDeformer {
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

impl WgpuDeformer {
    pub fn new(
        vertices: &VertexSet,
        bones: &BoneSet,
        fallback_bone: &Bone,
        morphs: &MorphSet,
        edge_size: f32,
        device: &wgpu::Device,
    ) -> Self {
        let vertex_buffer_data: Vec<VertexUnit> = vertices.iter().map(|v| v.simd.into()).collect();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader/Deformer"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../resources/shaders/model_skinning.wgsl").into(),
            ),
        });
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/InputBuffer"),
            contents: bytemuck::cast_slice(&vertex_buffer_data[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let num_vertices = vertices.len();
        let mut vertex2morphs = vec![vec![]; num_vertices];
        for morph in morphs.iter() {
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
        let mut vertex_deltas_buffer_data = vec![[0f32; 4]; buffer_size];
        for (vertex_idx, morphs) in vertex2morphs.iter().enumerate() {
            for (idx, morph) in morphs.iter().enumerate() {
                let position = morph.1.position;
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
            sdef_buffer_data[idx * 3] = vertex.origin.sdef_c;
            sdef_buffer_data[idx * 3 + 1] = vertex.origin.sdef_r0;
            sdef_buffer_data[idx * 3 + 2] = vertex.origin.sdef_r1;
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/MatrixBuffer"),
            contents: bytemuck::cast_slice(
                &Self::build_matrix_buffer_data(bones, fallback_bone)[..],
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let morph_weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deformer/MorphBuffer"),
            contents: bytemuck::cast_slice(&Self::build_morph_weight_buffer_data(morphs)[..]),
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

    fn build_matrix_buffer_data(bones: &BoneSet, fallback_bone: &Bone) -> Vec<[[f32; 4]; 4]> {
        let mut matrix_buffer_data = vec![[[0f32; 4]; 4]; bones.len().max(1)];
        for (idx, bone) in bones.iter().enumerate() {
            matrix_buffer_data[idx] = bone.matrices.skinning_transform.into();
        }
        if bones.len() == 0 {
            matrix_buffer_data[0] = fallback_bone.matrices.skinning_transform.into();
        }
        matrix_buffer_data
    }

    fn build_morph_weight_buffer_data(morphs: &MorphSet) -> Vec<f32> {
        let mut morph_weight_buffer_data = vec![0f32; morphs.len() + 1];
        for (idx, morph) in morphs.iter().enumerate() {
            morph_weight_buffer_data[idx + 1] = morph.weight();
        }
        morph_weight_buffer_data
    }

    pub fn update_buffer(&self, model: &Model, camera: &dyn Camera, queue: &wgpu::Queue) {
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
                &Self::build_matrix_buffer_data(model.bones(), &model.shared_fallback_bone)[..],
            ),
        );
        queue.write_buffer(
            &self.morph_weight_buffer,
            0,
            bytemuck::cast_slice(&Self::build_morph_weight_buffer_data(model.morphs())[..]),
        );
    }

    pub fn execute(
        &self,
        output_buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        log::debug!("Executing Skin Deformer...");
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
        queue.submit(Some(command_encoder.finish()));
    }
}

#[derive(Debug)]
pub struct CommonDeformer {
    vertex_buffer_data: Vec<VertexUnit>,
}

impl CommonDeformer {
    pub fn new(vertices: &VertexSet) -> Self {
        let vertex_buffer_data: Vec<VertexUnit> = vertices.iter().map(|v| v.simd.into()).collect();
        Self { vertex_buffer_data }
    }

    pub fn execute_model(
        &self,
        model: &Model,
        camera: &dyn Camera,
        output_buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.execute(
            model.vertices(),
            model.bones(),
            model.morphs(),
            model.edge_size(camera),
            output_buffer,
            device,
            queue,
        )
    }

    pub fn execute(
        &self,
        vertices: &VertexSet,
        bones: &BoneSet,
        morphs: &MorphSet,
        edge_size: f32,
        output_buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut output = self.vertex_buffer_data.clone();
        let mut vertex_position_deltas = vec![Vector3::zero(); vertices.len()];
        for morph in morphs.iter() {
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Vertex(items) => {
                    for item in items {
                        if let Some((vertex_idx, _)) = usize::try_from(item.vertex_index)
                            .ok()
                            .and_then(|idx| vertices.get(idx).map(|v| (idx, v)))
                        {
                            let position = f128_to_vec3(item.position);
                            vertex_position_deltas[vertex_idx] += position * morph.weight();
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
        for (idx, vertex) in vertices.iter().enumerate() {
            match vertex.origin.typ {
                nanoem::model::ModelVertexType::UNKNOWN => {
                    output[idx].position = vertex.origin.origin;
                    output[idx].normal = vertex.origin.normal;
                }
                nanoem::model::ModelVertexType::BDEF1 => {
                    let m = bones.try_get(vertex.origin.bone_indices[0]).unwrap()
                        .matrices
                        .skinning_transform;
                    let pos = m
                        * (f128_to_vec3(vertex.origin.origin) + vertex_position_deltas[idx])
                            .extend(1f32);
                    output[idx].position = (pos / pos.w).into();
                    output[idx].normal = (mat4_truncate(m) * f128_to_vec3(vertex.origin.normal))
                        .extend(0f32)
                        .into();
                }
                nanoem::model::ModelVertexType::BDEF2 => {
                    let weight = vertex.origin.bone_weights[0];
                    let m0 = bones.try_get(vertex.origin.bone_indices[0]).unwrap()
                        .matrices
                        .skinning_transform;
                    let m1 = bones.try_get(vertex.origin.bone_indices[1]).unwrap()
                        .matrices
                        .skinning_transform;
                    let pos = (f128_to_vec3(vertex.origin.origin) + vertex_position_deltas[idx])
                        .extend(1f32);
                    let normal = f128_to_vec3(vertex.origin.normal);
                    let pos = (m1 * pos).lerp(m0 * pos, weight);
                    output[idx].position = (pos / pos.w).into();
                    output[idx].normal = (mat4_truncate(m1) * normal)
                        .lerp(mat4_truncate(m0) * normal, weight)
                        .extend(0f32)
                        .into();
                }
                nanoem::model::ModelVertexType::BDEF4 | nanoem::model::ModelVertexType::QDEF => {
                    let weights = vertex.origin.bone_weights;
                    let m0 = bones.try_get(vertex.origin.bone_indices[0]).unwrap()
                        .matrices
                        .skinning_transform;
                    let m1 = bones.try_get(vertex.origin.bone_indices[1]).unwrap()
                        .matrices
                        .skinning_transform;
                    let m2 = bones.try_get(vertex.origin.bone_indices[2]).unwrap()
                        .matrices
                        .skinning_transform;
                    let m3 = bones.try_get(vertex.origin.bone_indices[3]).unwrap()
                        .matrices
                        .skinning_transform;
                    let pos = (f128_to_vec3(vertex.origin.origin) + vertex_position_deltas[idx])
                        .extend(1f32);
                    let normal = f128_to_vec3(vertex.origin.normal);
                    let pos = m0 * pos * weights[0]
                        + m1 * pos * weights[1]
                        + m2 * pos * weights[2]
                        + m3 * pos * weights[3];
                    output[idx].position = (pos / pos.w).into();
                    output[idx].normal = (mat4_truncate(m0) * normal * weights[0]
                        + mat4_truncate(m1) * normal * weights[1]
                        + mat4_truncate(m2) * normal * weights[2]
                        + mat4_truncate(m3) * normal * weights[3])
                        .extend(0f32)
                        .into();
                }
                nanoem::model::ModelVertexType::SDEF => {
                    let weights = vertex.origin.bone_weights;
                    let m0 = bones.try_get(vertex.origin.bone_indices[0]).unwrap()
                        .matrices
                        .skinning_transform;
                    let m1 = bones.try_get(vertex.origin.bone_indices[1]).unwrap()
                        .matrices
                        .skinning_transform;
                    let sdef_c = f128_to_vec3(vertex.origin.sdef_c);
                    let sdef_r0 = f128_to_vec3(vertex.origin.sdef_r0);
                    let sdef_r1 = f128_to_vec3(vertex.origin.sdef_r1);
                    let sdef_i = sdef_r0 * weights[0] + sdef_r1 * weights[1];
                    let sdef_r0_n = sdef_c + sdef_r0 - sdef_i;
                    let sdef_r1_n = sdef_c + sdef_r1 - sdef_i;
                    let r0 = (m0 * sdef_r0_n.extend(1f32)).truncate();
                    let r1 = (m1 * sdef_r1_n.extend(1f32)).truncate();
                    let c0 = (m0 * sdef_c.extend(1f32)).truncate();
                    let c1 = (m1 * sdef_c.extend(1f32)).truncate();
                    let delta = (r0 + c0 - sdef_c) * weights[0] + (r1 + c1 - sdef_c) * weights[1];
                    let t = (sdef_c + delta) * 0.5f32;
                    let p = (f128_to_vec3(vertex.origin.origin) + vertex_position_deltas[idx]
                        - sdef_c)
                        .extend(1f32);
                    let q0 = Quaternion::from(mat4_truncate(m0));
                    let q1 = Quaternion::from(mat4_truncate(m1));
                    let m = Matrix4::from(q1.slerp(q0, weights[0]));
                    output[idx].position = ((m * p).truncate() + t).extend(1f32).into();
                    output[idx].normal = (mat4_truncate(m) * f128_to_vec3(vertex.origin.normal))
                        .extend(0f32)
                        .into();
                }
            }
            output[idx].edge = ((f128_to_vec3(output[idx].position)
                + f128_to_vec3(output[idx].normal) * vertex.origin.edge_size)
                * edge_size)
                .extend(output[idx].edge[3])
                .into();
        }
        queue.write_buffer(output_buffer, 0u64, bytemuck::cast_slice(&output[..]));
    }
}

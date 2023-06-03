use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    iter,
};

use cgmath::{
    AbsDiffEq, ElementWise, InnerSpace, Matrix3, Matrix4, One, Quaternion, Rad, Rotation3, Vector3,
    Vector4, VectorSpace, Zero,
};
use nanoem::{
    model::{ModelMorphCategory, ModelRigidBodyTransformType},
    motion::{MotionBoneKeyframe, MotionModelKeyframe, MotionTrackBundle},
};

use crate::{
    bounding_box::BoundingBox,
    camera::{Camera, PerspectiveCamera},
    deformer::{CommonDeformer, Deformer, WgpuDeformer},
    drawable::{DrawContext, DrawType, Drawable},
    error::MdanceioError,
    graphics::{
        common_pass::{CPassBindGroup, CPassVertexBuffer},
        model_program_bundle::{ModelProgramBundle, UniformBind},
        technique::{EdgePassKey, ObjectPassKey, ShadowPassKey, TechniqueType, ZplotPassKey},
    },
    light::Light,
    motion::Motion,
    physics_engine::{PhysicsEngine, RigidBodyFollowBone, SimulationMode, SimulationTiming},
    utils::{
        f128_to_isometry, f128_to_vec3, f128_to_vec4, from_isometry, lerp_f32, mat4_truncate,
        to_isometry, to_na_vec3, Invert,
    },
};

use super::{
    bone::BoneSet, Bone, BoneIndex, MaterialIndex, MorphIndex, NanoemBone,
    NanoemConstraint, NanoemJoint, NanoemLabel, NanoemMaterial, NanoemModel, NanoemMorph,
    NanoemRigidBody, NanoemSoftBody, NanoemTexture, NanoemVertex, RigidBodyIndex, SoftBodyIndex,
    VertexIndex,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexUnit {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub texcoord: [f32; 4],
    pub edge: [f32; 4],
    pub uva: [[f32; 4]; 4],
    pub weights: [f32; 4],
    pub indices: [f32; 4],
    pub info: [f32; 4], /* type,vertexIndex,edgeSize,padding */
}

impl From<VertexSimd> for VertexUnit {
    fn from(simd: VertexSimd) -> Self {
        Self {
            position: simd.origin.into(),
            normal: simd.normal.into(),
            texcoord: simd.texcoord.into(),
            edge: simd.origin.into(),
            uva: [
                simd.origin_uva[0].into(),
                simd.origin_uva[1].into(),
                simd.origin_uva[2].into(),
                simd.origin_uva[3].into(),
            ],
            weights: simd.weights.into(),
            indices: simd.indices.into(),
            info: simd.info.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelMorphUsage {
    pub eyebrow: Option<MorphIndex>,
    pub eye: Option<MorphIndex>,
    pub lip: Option<MorphIndex>,
    pub other: Option<MorphIndex>,
}

#[derive(Debug, Clone, Copy, Default, Hash)]
struct ModelStates {
    visible: bool,
    uploaded: bool,
    vertex_shader_skinning: bool,
    show_all_bones: bool,
    show_all_vertex_points: bool,
    show_all_vertex_faces: bool,
    show_all_rigid_body_shapes: bool,
    show_all_rigid_body_shapes_color_by_shape: bool,
    compute_shader_skinning: bool,
    morph_weight_focus: bool,
    enable_shadow: bool,
    enable_shadow_map: bool,
    enable_add_blend: bool,
    show_all_joint_shapes: bool,
    dirty: bool,
    dirty_staging_buffer: bool,
    dirty_morph: bool,
    physics_simulation: bool,
    enable_ground_shadow: bool,
    show_all_material_shapes: bool,
    show_all_vertex_weights: bool,
    blending_vertex_weights_enabled: bool,
    show_all_vertex_normals: bool,
}

pub struct Model {
    camera: Box<PerspectiveCamera>,
    skin_deformer: Deformer,
    opaque: Box<NanoemModel>,
    vertices: Vec<Vertex>,
    vertex_indices: Vec<u32>,
    materials: Vec<Material>,
    bones: BoneSet,
    morphs: Vec<Morph>,
    labels: Vec<Label>,
    rigid_bodies: Vec<RigidBody>,
    joints: Vec<Joint>,
    soft_bodies: Vec<SoftBody>,
    active_morph: ModelMorphUsage,
    active_bone_pair: (Option<BoneIndex>, Option<BoneIndex>),
    bone_index_hash_map: HashMap<MaterialIndex, HashMap<BoneIndex, usize>>,
    morphs_by_name: HashMap<String, MorphIndex>,
    /// Map from target bone to constraint containing it
    outside_parents: HashMap<BoneIndex, (String, String)>,
    bone_bound_rigid_bodies: HashMap<BoneIndex, RigidBodyIndex>,
    pub shared_fallback_bone: Bone,
    bounding_box: BoundingBox,
    uniform_bind: UniformBind,
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffer: wgpu::Buffer,
    edge_color: Vector4<f32>,
    name: String,
    comment: String,
    canonical_name: String,
    states: ModelStates,
    edge_size_scale_factor: f32,
    opacity: f32,
    count_vertex_skinning_needed: i32,
    stage_vertex_buffer_index: usize,
}

impl Model {
    pub const INITIAL_WORLD_MATRIX: Matrix4<f32> = Matrix4::new(
        1f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32,
        1f32,
    );
    pub const DEFAULT_CM_SCALE_FACTOR: f32 = 0.1259496f32;
    pub const DEFAULT_MODEL_CORRECTION_HEIGHT: f32 = -2f32;
    pub const PMX_FORMAT_EXTENSION: &'static str = "pmx";
    pub const PMD_FORMAT_EXTENSION: &'static str = "pmd";

    pub const DRAW_BONE_CONNECTION_THICKNESS: f32 = 1.0f32;
    pub const DRAW_VERTEX_NORMAL_SCALE_FACTOR: f32 = 0.0f32;
    pub const MAX_BONE_UNIFORMS: usize = 55;

    pub fn new_from_bytes(
        bytes: &[u8],
        language_type: nanoem::common::LanguageType,
        physics_engine: &mut PhysicsEngine,
        global_camera: &PerspectiveCamera,
        effect: &mut ModelProgramBundle,
        fallback_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        shadow_bind: &wgpu::BindGroup,
        fallback_texture_bind: &wgpu::BindGroup,
        fallback_shadow_bind: &wgpu::BindGroup,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, MdanceioError> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        let initial_states = ModelStates {
            physics_simulation: true,
            enable_ground_shadow: true,
            uploaded: true,
            dirty: true,
            visible: true,
            ..Default::default()
        };
        match NanoemModel::load_from_buffer(&mut buffer) {
            Ok(nanoem_model) => {
                let opaque = Box::new(nanoem_model);
                let mut name = opaque.get_name(language_type).to_owned();
                let comment = opaque.get_comment(language_type).to_owned();
                let canonical_name = opaque
                    .get_name(nanoem::common::LanguageType::default())
                    .to_owned();
                if name.is_empty() {
                    name = canonical_name.clone();
                }

                // TODO: 共享fallback骨骼
                // let shared_fallback_bone = Rc::new(RefCell::new(Bone::new(
                //     "SharedFallbackBone",
                //     "SharedFallbackBone",
                // )));

                let mut vertices = opaque
                    .vertices
                    .iter()
                    .map(|vertex| Vertex::from_nanoem(vertex))
                    .collect::<Vec<_>>();
                let indices = opaque.vertex_indices.clone();
                let mut bones = BoneSet::new(&opaque.bones, &opaque.constraints, language_type);
                let mut bone_set: HashSet<BoneIndex> = HashSet::new();
                let mut morphs_by_name = HashMap::new();
                let morphs = opaque
                    .morphs
                    .iter()
                    .map(|morph| Morph::from_nanoem(&morph, language_type))
                    .collect::<Vec<_>>();
                for (index, morph) in opaque.morphs.iter().enumerate() {
                    if let nanoem::model::ModelMorphType::Vertex(morph_vertices) = morph.get_type()
                    {
                        for morph_vertex in morph_vertices {
                            if let Some(vertex) =
                                opaque.get_one_vertex_object(morph_vertex.vertex_index)
                            {
                                for bone_index in vertex.get_bone_indices() {
                                    if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                        bone_set.insert(bone.base.index);
                                    }
                                }
                            }
                        }
                    }
                    let category = morph.category;
                    for language in nanoem::common::LanguageType::all() {
                        morphs_by_name
                            .insert(morph.get_name(*language).to_owned(), morph.base.index);
                    }
                }
                let get_active_morph = |category: ModelMorphCategory| {
                    opaque
                        .morphs
                        .iter()
                        .enumerate()
                        .find(|(index, morph)| morph.category == category)
                        .map(|(idx, _)| idx)
                };
                let active_morph = ModelMorphUsage {
                    eyebrow: get_active_morph(ModelMorphCategory::Eyebrow),
                    eye: get_active_morph(ModelMorphCategory::Eye),
                    lip: get_active_morph(ModelMorphCategory::Lip),
                    other: get_active_morph(ModelMorphCategory::Other),
                };
                let labels = opaque
                    .labels
                    .iter()
                    .map(|label| Label::from_nanoem(label, language_type))
                    .collect();
                let rigid_bodies: Vec<RigidBody> = opaque
                    .rigid_bodies
                    .iter()
                    .map(|rigid_body| {
                        let is_dynamic = !matches!(
                            rigid_body.get_transform_type(),
                            nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation
                        );
                        let is_morph = if let Some(bone) =
                            opaque.get_one_bone_object(rigid_body.get_bone_index())
                        {
                            is_dynamic && bone_set.contains(&bone.base.index)
                        } else {
                            false
                        };
                        // TODO: initializeTransformFeedback
                        RigidBody::from_nanoem(
                            rigid_body,
                            language_type,
                            is_morph,
                            &bones,
                            physics_engine,
                        )
                    })
                    .collect();
                let joints = opaque
                    .joints
                    .iter()
                    .map(|joint| {
                        Joint::from_nanoem(joint, language_type, &rigid_bodies, physics_engine)
                    })
                    .collect();
                let soft_bodies = opaque
                    .soft_bodies
                    .iter()
                    .map(|soft_body| SoftBody::from_nanoem(soft_body, language_type))
                    .collect();

                let shared_fallback_bone = Bone::empty(usize::MAX);
                // split_bones_per_material();

                let edge_size_scale_factor = 1.0f32;

                let mut offset: usize = 0;
                let mut unique_bone_index_per_material = 0usize;
                let mut references: HashMap<usize, HashSet<VertexIndex>> = HashMap::new();
                let mut index_hash = HashMap::new();
                let mut bone_index_hash_map = HashMap::new();
                let mut count_vertex_skinning_needed = 0;
                for material in &opaque.materials {
                    let num_indices = material.num_vertex_indices;
                    for j in offset..offset + num_indices {
                        let vertex_index = &opaque.vertex_indices[j];
                        let vertex = &opaque.vertices[*vertex_index as usize];
                        for bone_index in vertex.bone_indices {
                            if let Some(bone) = opaque.get_one_bone_object(bone_index) {
                                let bone_index = bone.base.index;
                                if !index_hash.contains_key(&bone_index) {
                                    index_hash.insert(bone_index, unique_bone_index_per_material);
                                    unique_bone_index_per_material += 1;
                                }
                                references
                                    .entry(bone_index)
                                    .or_insert_with(|| HashSet::new())
                                    .insert(vertex.base.index);
                            }
                        }
                    }
                    if !index_hash.is_empty() {
                        if references.len() > Self::MAX_BONE_UNIFORMS {
                            let mut vertex_list: Vec<VertexIndex> = vec![];
                            let mut bone_vertex_list: Vec<(usize, Vec<VertexIndex>)> = vec![];
                            for vertex_reference in &references {
                                vertex_list.clear();
                                for vertex in vertex_reference.1 {
                                    vertex_list.push(*vertex)
                                }
                                bone_vertex_list.push((*vertex_reference.0, vertex_list.clone()));
                            }
                            bone_vertex_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                            index_hash.clear();
                            for j in 0..Self::MAX_BONE_UNIFORMS {
                                let pair = &bone_vertex_list[j];
                                index_hash.insert(pair.0, j);
                            }
                            for pair in &mut bone_vertex_list {
                                let all_vertices = &mut pair.1;
                                for vertex_index in all_vertices {
                                    if let Some(vertex) = vertices.get_mut(*vertex_index) {
                                        vertex.set_skinning_enabled(true)
                                    }
                                    count_vertex_skinning_needed += 1;
                                }
                            }
                        }
                        bone_index_hash_map.insert(material.base.index, index_hash.clone());
                    }
                    for it in &mut references {
                        it.1.clear()
                    }
                    offset += num_indices;
                    unique_bone_index_per_material = 0;
                    index_hash.clear();
                    references.clear();
                }

                log::trace!("Len(vertices): {}", vertices.len());
                let edge_size =
                    Self::internal_edge_size(&bones, global_camera, edge_size_scale_factor);

                log::info!("{:?}", device.limits());
                let skin_deformer = if device.limits().max_storage_buffers_per_shader_stage < 6 {
                    Deformer::Software(CommonDeformer::new(&vertices))
                } else {
                    Deformer::Wgpu(Box::new(WgpuDeformer::new(
                        &vertices,
                        &bones,
                        &shared_fallback_bone,
                        &morphs,
                        edge_size,
                        device,
                    )))
                };
                let bytes_per_vertex = std::mem::size_of::<VertexUnit>();
                let unpadded_size = vertices.len() * bytes_per_vertex;
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
                let padding = (align - unpadded_size % align) % align;
                let vertex_buffer_even = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(format!("Model/{}/VertexBuffer/Even", canonical_name).as_str()),
                    size: (unpadded_size + padding) as u64,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let vertex_buffer_odd = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(format!("Model/{}/VertexBuffer/Odd", canonical_name).as_str()),
                    size: (unpadded_size + padding) as u64,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let vertex_buffers = [vertex_buffer_even, vertex_buffer_odd];
                log::trace!("Len(index_buffer): {}", &opaque.vertex_indices.len());
                let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(format!("Model/{}/IndexBuffer", canonical_name).as_str()),
                        contents: bytemuck::cast_slice(&opaque.vertex_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                );
                let mut stage_vertex_buffer_index = 0;
                match &skin_deformer {
                    Deformer::Wgpu(deformer) => {
                        deformer.execute(&vertex_buffers[stage_vertex_buffer_index], device, queue);
                    }
                    Deformer::Software(deformer) => deformer.execute(
                        &vertices,
                        &bones,
                        &morphs,
                        edge_size,
                        &vertex_buffers[stage_vertex_buffer_index],
                        device,
                        queue,
                    ),
                }
                stage_vertex_buffer_index = 1 - stage_vertex_buffer_index;

                let uniform_bind = effect.get_uniform_bind(opaque.materials.len(), device);

                let mut materials = vec![];
                let mut index_offset = 0usize;
                for (idx, material) in opaque.materials.iter().enumerate() {
                    let num_indices = material.num_vertex_indices;
                    materials.push(Material::from_nanoem(
                        material,
                        language_type,
                        &mut MaterialDrawContext {
                            effect,
                            fallback_texture,
                            sampler,
                            bind_group_layout,
                            color_format,
                            is_add_blend: initial_states.enable_add_blend,
                            uniform_bind: uniform_bind.bind_group(),
                            shadow_bind,
                            fallback_texture_bind,
                            fallback_shadow_bind,
                            vertex_buffers: &vertex_buffers,
                            index_buffer: &index_buffer,
                        },
                        index_offset as u32,
                        num_indices as u32,
                        device,
                    ));
                }
                let mut index_offset = 0;
                for nanoem_material in &opaque.materials {
                    let num_indices = nanoem_material.get_num_vertex_indices();
                    for i in index_offset..(index_offset + num_indices) {
                        let vertex_index = indices[i];
                        if let Some(vertex) = vertices.get_mut(vertex_index as usize) {
                            vertex.set_material(nanoem_material.get_index());
                        }
                    }
                    index_offset += num_indices;
                }

                let mut camera = Box::new(PerspectiveCamera::new());
                camera.set_angle(global_camera.angle());
                camera.set_distance(global_camera.distance());
                camera.set_fov(global_camera.fov());
                camera.set_look_at(global_camera.look_at(None));
                // camera.update(viewport_image_size, bound_look_at)

                Ok(Self {
                    camera,
                    opaque,
                    skin_deformer,
                    bone_index_hash_map,
                    bones,
                    morphs,
                    vertices,
                    vertex_indices: indices,
                    materials,
                    labels,
                    rigid_bodies,
                    joints,
                    soft_bodies,
                    morphs_by_name,
                    active_bone_pair: (None, None),
                    active_morph,
                    outside_parents: HashMap::new(),
                    bone_bound_rigid_bodies: HashMap::new(),
                    // shared_fallback_bone,
                    vertex_buffers,
                    index_buffer,
                    uniform_bind,
                    shared_fallback_bone,
                    name,
                    comment,
                    canonical_name,
                    opacity: 1.0f32,
                    count_vertex_skinning_needed,
                    stage_vertex_buffer_index,
                    edge_color: Vector4::new(0f32, 0f32, 0f32, 1f32),
                    edge_size_scale_factor,
                    bounding_box: BoundingBox::new(),
                    states: initial_states,
                })
            }
            Err(status) => Err(MdanceioError::from_nanoem(
                "Cannot load the model: ",
                status,
            )),
        }
    }

    pub fn loadable_extensions() -> Vec<&'static str> {
        vec![Self::PMD_FORMAT_EXTENSION, Self::PMX_FORMAT_EXTENSION]
    }

    pub fn is_loadable_extension(extension: &str) -> bool {
        Self::loadable_extensions()
            .iter()
            .any(|ext| ext.to_lowercase().eq(extension))
    }

    pub fn create_all_images(
        &mut self,
        texture_lut: &HashMap<String, wgpu::Texture>,
        effect: &mut ModelProgramBundle,
        fallback_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        shadow_bind: &wgpu::BindGroup,
        fallback_texture_bind: &wgpu::BindGroup,
        fallback_shadow_bind: &wgpu::BindGroup,
        device: &wgpu::Device,
    ) {
        // TODO: 创建所有材质贴图并绑定到Material上
        let is_add_blend = self.is_add_blend_enabled();
        let mut index_offset = 0;
        for material in &mut self.materials {
            let num_indices = material.origin.num_vertex_indices as u32;
            material.diffuse_image = material
                .origin
                .get_diffuse_texture_object(&self.opaque.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.sphere_map_image = material
                .origin
                .get_sphere_map_texture_object(&self.opaque.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.toon_image = material
                .origin
                .get_toon_texture_object(&self.opaque.textures)
                .and_then(|texture_object| texture_lut.get(&texture_object.path))
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            material.update_bind(
                &mut MaterialDrawContext {
                    effect,
                    fallback_texture,
                    sampler,
                    bind_group_layout,
                    color_format,
                    is_add_blend,
                    uniform_bind: self.uniform_bind.bind_group(),
                    shadow_bind,
                    fallback_texture_bind,
                    fallback_shadow_bind,
                    vertex_buffers: &self.vertex_buffers,
                    index_buffer: &self.index_buffer,
                },
                index_offset,
                num_indices,
                device,
            );
            index_offset += num_indices;
        }
    }

    pub fn update_image(
        &mut self,
        texture_key: &str,
        texture: &wgpu::Texture,
        effect: &mut ModelProgramBundle,
        fallback_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        shadow_bind: &wgpu::BindGroup,
        fallback_texture_bind: &wgpu::BindGroup,
        fallback_shadow_bind: &wgpu::BindGroup,
        device: &wgpu::Device,
    ) {
        let is_add_blend = self.is_add_blend_enabled();
        let mut index_offset = 0;
        for material in &mut self.materials {
            let num_indices = material.origin.num_vertex_indices as u32;
            let mut updated = false;
            material
                .origin
                .get_diffuse_texture_object(&self.opaque.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
                .map(|texture| {
                    material.diffuse_image =
                        Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                    updated = true;
                });
            material
                .origin
                .get_sphere_map_texture_object(&self.opaque.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
                .map(|texture| {
                    material.sphere_map_image =
                        Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                    updated = true;
                });
            material
                .origin
                .get_toon_texture_object(&self.opaque.textures)
                .and_then(|texture_object| {
                    if texture_object.path == texture_key {
                        Some(texture)
                    } else {
                        None
                    }
                })
                .map(|texture| {
                    material.toon_image =
                        Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                    updated = true;
                });
            if updated {
                material.update_bind(
                    &mut MaterialDrawContext {
                        effect,
                        fallback_texture,
                        sampler,
                        bind_group_layout,
                        color_format,
                        is_add_blend,
                        uniform_bind: self.uniform_bind.bind_group(),
                        shadow_bind,
                        fallback_texture_bind,
                        fallback_shadow_bind,
                        vertex_buffers: &self.vertex_buffers,
                        index_buffer: &self.index_buffer,
                    },
                    index_offset,
                    num_indices,
                    device,
                );
            }
            index_offset += num_indices;
        }
    }

    fn create_image() {}

    pub fn create_all_bone_bounds_rigid_bodies(&mut self) {
        for (handle, rigid_body) in self.rigid_bodies.iter().enumerate() {
            if rigid_body.origin.transform_type != ModelRigidBodyTransformType::FromBoneToSimulation
            {
                if let Ok(bone) = usize::try_from(rigid_body.origin.bone_index) {
                    self.bone_bound_rigid_bodies.insert(bone, handle);
                }
            }
        }
    }

    pub fn clear_all_bone_bounds_rigid_bodies(&mut self) {
        self.bone_bound_rigid_bodies.clear();
    }

    pub fn initialize_all_rigid_bodies_transform_feedback(
        &mut self,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                rigid_body.initialize_transform_feedback(bone, physics_engine);
            }
        }
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        amount: f32,
        timing: SimulationTiming,
        physics_engine: &mut PhysicsEngine,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        let mut visible = true;
        if timing == SimulationTiming::Before {
            if let Some(keyframe) = motion.find_model_keyframe(frame_index) {
                self.edge_color = f128_to_vec4(keyframe.edge_color);
                self.edge_size_scale_factor = keyframe.edge_scale_factor;
                // TODO: do something if having active effect
                visible = keyframe.visible;
                self.set_physics_simulation_enabled(keyframe.is_physics_simulation_enabled);
                self.set_visible(visible);
                self.synchronize_all_constraint_states(
                    keyframe,
                    &motion.opaque.local_bone_motion_track_bundle,
                );
                self.synchronize_all_outside_parents(
                    keyframe,
                    &motion.opaque.global_motion_track_bundle,
                );
            } else {
                if let (Some(prev_frame), Some(next_frame)) =
                    motion.opaque.search_closest_model_keyframes(frame_index)
                {
                    let coef = Motion::coefficient(
                        prev_frame.base.frame_index,
                        next_frame.base.frame_index,
                        frame_index,
                    );
                    self.edge_color = f128_to_vec4(prev_frame.edge_color)
                        .lerp(f128_to_vec4(next_frame.edge_color), coef);
                    self.edge_size_scale_factor = lerp_f32(
                        prev_frame.edge_scale_factor,
                        next_frame.edge_scale_factor,
                        coef,
                    );
                    // TODO: do something if having active effect
                }
                visible = self.states.visible
            }
        }
        if visible {
            match timing {
                SimulationTiming::Before => {
                    self.bounding_box.reset();
                    self.reset_all_materials();
                    self.reset_all_bone_local_transform();
                    self.synchronize_morph_motion(motion, frame_index, amount);
                    self.synchronize_bone_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        physics_engine,
                        outside_parent_bone_map,
                    );
                    self.synchronize_all_rigid_body_kinematics(motion, frame_index, physics_engine);
                    self.synchronize_all_rigid_bodies_transform_feedback_to_simulation(
                        physics_engine,
                    );
                }
                SimulationTiming::After => {
                    self.synchronize_bone_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        physics_engine,
                        outside_parent_bone_map,
                    );
                }
            }
        }
    }

    fn synchronize_all_constraint_states(
        &mut self,
        keyframe: &MotionModelKeyframe,
        local_bone_motion_track_bundle: &MotionTrackBundle<MotionBoneKeyframe>,
    ) {
        for state in &keyframe.constraint_states {
            if let Some(constraint) = local_bone_motion_track_bundle
                .resolve_id(state.bone_id)
                .and_then(|name| self.bones.find_mut_constraint(name))
            {
                constraint.states.enabled = state.enabled;
            }
        }
    }

    fn synchronize_all_outside_parents(
        &mut self,
        keyframe: &MotionModelKeyframe,
        global_motion_track_bundle: &MotionTrackBundle<()>,
    ) {
        self.outside_parents.clear();
        for op in &keyframe.outside_parents {
            if let Some(bound_bone) = global_motion_track_bundle
                .resolve_id(op.local_bone_track_index)
                .and_then(|subject_bone_name| self.bones.find(subject_bone_name))
            {
                if let Some(target_object_name) =
                    global_motion_track_bundle.resolve_id(op.global_model_track_index)
                {
                    if let Some(target_bone_name) =
                        global_motion_track_bundle.resolve_id(op.global_bone_track_index)
                    {
                        // TODO: verify model and bone exists
                        self.outside_parents.insert(
                            bound_bone.origin.base.index,
                            (target_object_name.clone(), target_bone_name.clone()),
                        );
                    }
                }
            }
        }
    }

    fn synchronize_all_rigid_body_kinematics(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                let bone_name = bone.canonical_name.clone();
                if let (Some(prev_frame), Some(next_frame)) = (
                    motion.find_bone_keyframe(&bone_name, frame_index),
                    motion.find_bone_keyframe(&bone_name, frame_index + 1),
                ) {
                    if prev_frame.is_physics_simulation_enabled
                        && !next_frame.is_physics_simulation_enabled
                    {
                        rigid_body.enable_kinematic(physics_engine);
                    }
                } else {
                    if let (Some(prev_frame), Some(next_frame)) = motion
                        .opaque
                        .search_closest_bone_keyframes(&bone_name, frame_index)
                    {
                        if prev_frame.is_physics_simulation_enabled
                            && !next_frame.is_physics_simulation_enabled
                        {
                            rigid_body.enable_kinematic(physics_engine);
                        }
                    }
                }
            }
        }
    }

    fn synchronize_all_rigid_bodies_transform_feedback_to_simulation(
        &mut self,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            let bone = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx));
            rigid_body.apply_all_forces(bone, physics_engine);
            if let Some(bone) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx))
            {
                rigid_body.synchronize_transform_feedback_to_simulation(bone, physics_engine);
            }
        }
    }

    pub fn synchronize_all_rigid_bodies_transform_feedback_from_simulation(
        &mut self,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in &mut self.rigid_bodies {
            if let Some((bone_idx, bone)) = usize::try_from(rigid_body.origin.bone_index)
                .ok()
                .and_then(|idx| self.bones.get(idx).map(|bone| (idx, bone)))
            {
                let parent_bone = usize::try_from(bone.origin.parent_bone_index)
                    .ok()
                    .and_then(|idx| self.bones.get(idx))
                    .cloned();
                rigid_body.synchronize_transform_feedback_from_simulation(
                    self.bones.get_mut(bone_idx).unwrap(),
                    (&parent_bone).as_ref(),
                    follow_type,
                    physics_engine,
                );
            }
        }
    }

    fn synchronize_bone_motion(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        amount: f32,
        timing: SimulationTiming,
        physics_engine: &mut PhysicsEngine,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        if let SimulationTiming::Before = timing {
            for bone in self.bones.iter_mut() {
                let rigid_body = self
                    .bone_bound_rigid_bodies
                    .get(&bone.origin.base.index)
                    .and_then(|idx| self.rigid_bodies.get_mut(*idx));
                bone.synchronize_motion(motion, rigid_body, frame_index, amount, physics_engine);
            }
        }
        self.apply_all_bones_transform(timing, outside_parent_bone_map);
    }

    fn synchronize_morph_motion(&mut self, motion: &Motion, frame_index: u32, amount: f32) {
        if !self.states.dirty_morph {
            self.reset_all_morphs();
            for morph in &mut self.morphs {
                let name = morph.canonical_name.to_string();
                morph.synchronize_motion(motion, &name, frame_index, amount);
            }
            self.deform_all_morphs(true);
            for morph in &mut self.morphs {
                morph.dirty = false;
            }
            self.states.dirty_morph = true;
        }
    }

    pub fn reset_all_morph_deform_states(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        physics_engine: &mut PhysicsEngine,
    ) {
        let mut active_morphs = HashSet::new();
        active_morphs.insert(self.active_morph.eyebrow);
        active_morphs.insert(self.active_morph.eye);
        active_morphs.insert(self.active_morph.lip);
        active_morphs.insert(self.active_morph.other);
        for morph_idx in 0..self.morphs.len() {
            let morph = self.morphs.get(morph_idx).unwrap();
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Bone(children) => {
                    for child in children {
                        if let Some((idx, target_bone)) = usize::try_from(child.bone_index)
                            .ok()
                            .and_then(|idx| self.bones.get_mut(idx).map(|bone| (idx, bone)))
                        {
                            let rigid_body = self
                                .bone_bound_rigid_bodies
                                .get(&idx)
                                .and_then(|idx| self.rigid_bodies.get_mut(*idx));
                            target_bone.reset_morph_transform();
                            target_bone.synchronize_motion(
                                motion,
                                rigid_body,
                                frame_index,
                                0f32,
                                physics_engine,
                            );
                        }
                    }
                }
                nanoem::model::ModelMorphType::Flip(children) => {
                    for target_morph_index in children
                        .iter()
                        .map(|child| usize::try_from(child.morph_index).ok())
                        .collect::<Vec<_>>()
                    {
                        if let Some((idx, target_morph)) = target_morph_index
                            .and_then(|idx| self.morphs.get_mut(idx).map(|morph| (idx, morph)))
                        {
                            if !active_morphs.contains(&Some(idx)) {
                                let name = target_morph.canonical_name.clone();
                                target_morph.synchronize_motion(motion, &name, frame_index, 0f32);
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Group(children) => {
                    for target_morph_index in children
                        .iter()
                        .map(|child| usize::try_from(child.morph_index).ok())
                        .collect::<Vec<_>>()
                    {
                        if let Some((idx, target_morph)) = target_morph_index
                            .and_then(|idx| self.morphs.get_mut(idx).map(|morph| (idx, morph)))
                        {
                            if !active_morphs.contains(&Some(idx)) {
                                let name = target_morph.canonical_name.clone();
                                target_morph.synchronize_motion(motion, &name, frame_index, 0f32);
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Material(children) => {
                    for child in children {
                        if let Some(target_material) = usize::try_from(child.material_index)
                            .ok()
                            .and_then(|idx| self.materials.get_mut(idx))
                        {
                            target_material.reset();
                        }
                    }
                }
                nanoem::model::ModelMorphType::Vertex(_)
                | nanoem::model::ModelMorphType::Texture(_)
                | nanoem::model::ModelMorphType::Uva1(_)
                | nanoem::model::ModelMorphType::Uva2(_)
                | nanoem::model::ModelMorphType::Uva3(_)
                | nanoem::model::ModelMorphType::Uva4(_)
                | nanoem::model::ModelMorphType::Impulse(_) => {}
            }
        }
    }

    fn apply_all_bones_transform(
        &mut self,
        timing: SimulationTiming,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        // TODO: Here nanoem use a ordered bone. If any sort to bone happened, I will change logic here.
        for idx in self.bones.iter_idx() {
            let bone = self.bones.get(idx).unwrap();
            if (bone.origin.flags.is_affected_by_physics_simulation
                && timing == SimulationTiming::After)
                || (!bone.origin.flags.is_affected_by_physics_simulation
                    && timing == SimulationTiming::Before)
            {
                self.bones.apply_local_transform(idx);
                let outside_parent_bone = self
                    .outside_parents
                    .get(&idx)
                    .and_then(|op_path| outside_parent_bone_map.get(op_path));
                let bone = self.bones.get_mut(idx).unwrap();
                if let Some(outside_parent_bone) = outside_parent_bone {
                    bone.apply_outside_parent_transform(outside_parent_bone);
                }
                self.bounding_box.set(bone.world_translation());
            }
        }
    }

    fn reset_all_bone_transforms(&mut self) {
        for bone in self.bones.iter_mut() {
            bone.reset_local_transform();
            bone.reset_morph_transform();
            bone.reset_user_transform();
        }
    }

    fn reset_all_bone_local_transform(&mut self) {
        for bone in self.bones.iter_mut() {
            bone.reset_local_transform();
        }
    }

    fn reset_all_bone_morph_transform(&mut self) {
        for bone in self.bones.iter_mut() {
            bone.reset_local_transform();
            bone.reset_morph_transform();
        }
    }

    fn reset_all_materials(&mut self) {
        for material in &mut self.materials {
            material.reset();
        }
    }

    fn reset_all_morphs(&mut self) {
        for morph in &mut self.morphs {
            morph.reset();
        }
    }

    fn reset_all_vertices(&mut self) {
        for vertex in &mut self.vertices {
            vertex.reset();
        }
    }

    pub fn set_physics_simulation_enabled(&mut self, value: bool) {
        if self.states.physics_simulation != value {
            self.set_all_physics_objects_enabled(value && self.states.visible);
            self.states.physics_simulation = value;
        }
    }

    pub fn set_shadow_map_enabled(&mut self, value: bool) {
        if self.states.enable_shadow_map != value {
            self.states.enable_shadow_map = value;
            // TODO: publish set shadow map event
        }
    }

    pub fn set_visible(&mut self, value: bool) {
        if self.states.visible != value {
            self.set_all_physics_objects_enabled(value & self.states.physics_simulation);
            // TODO: enable effect
            self.states.visible = value;
            // TODO: publish set visible event
        }
    }

    pub fn set_all_physics_objects_enabled(&mut self, value: bool) {
        if value {
            for soft_body in &mut self.soft_bodies {
                soft_body.enable();
            }
            for rigid_body in &mut self.rigid_bodies {
                rigid_body.enable();
            }
            for joint in &mut self.joints {
                joint.enable();
            }
        } else {
            for soft_body in &mut self.soft_bodies {
                soft_body.disable();
            }
            for rigid_body in &mut self.rigid_bodies {
                rigid_body.disable();
            }
            for joint in &mut self.joints {
                joint.disable();
            }
        }
    }

    pub fn deform_all_morphs(&mut self, check_dirty: bool) {
        for morph_idx in 0..self.morphs.len() {
            self.pre_deform_morph(morph_idx);
        }
        for morph_idx in 0..self.morphs.len() {
            self.deform_morph(morph_idx, check_dirty);
        }
    }

    fn pre_deform_morph(&mut self, morph_idx: MorphIndex) {
        if let Some(morph) = self.morphs.get(morph_idx) {
            let weight = morph.weight;
            match &morph.origin.typ {
                nanoem::model::ModelMorphType::Group(children) => {
                    for (target_morph_idx, child_weight) in children
                        .iter()
                        .map(|child| (usize::try_from(child.morph_index).ok(), child.weight))
                        .collect::<Vec<_>>()
                    {
                        if let Some(target_morph) =
                            target_morph_idx.and_then(|idx| self.morphs.get_mut(idx))
                        {
                            if let nanoem::model::ModelMorphType::Flip(_) = target_morph.origin.typ
                            {
                                target_morph.set_forced_weight(weight * child_weight);
                                self.pre_deform_morph(target_morph_idx.unwrap());
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Flip(children) => {
                    if weight > 0f32 && children.len() > 0 {
                        let target_idx = (((children.len() + 1) as f32 * weight) as usize - 1)
                            .clamp(0, children.len() - 1);
                        let child = &children[target_idx];
                        let child_weight = child.weight;
                        usize::try_from(child.morph_index)
                            .ok()
                            .and_then(|idx| self.morphs.get_mut(idx))
                            .map(|morph| morph.set_weight(weight));
                    }
                }
                nanoem::model::ModelMorphType::Material(children) => {
                    for child in children {
                        if let Some(material) = usize::try_from(child.material_index)
                            .ok()
                            .and_then(|idx| self.materials.get_mut(idx))
                        {
                            material.reset_deform();
                        } else {
                            for material in &mut self.materials {
                                material.reset_deform();
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Vertex(_)
                | nanoem::model::ModelMorphType::Bone(_)
                | nanoem::model::ModelMorphType::Texture(_)
                | nanoem::model::ModelMorphType::Uva1(_)
                | nanoem::model::ModelMorphType::Uva2(_)
                | nanoem::model::ModelMorphType::Uva3(_)
                | nanoem::model::ModelMorphType::Uva4(_)
                | nanoem::model::ModelMorphType::Impulse(_) => {}
            }
        }
    }

    fn deform_morph(&mut self, morph_idx: MorphIndex, check_dirty: bool) {
        if let Some(morph) = self.morphs.get_mut(morph_idx) {
            if !check_dirty || (check_dirty && morph.dirty) {
                let weight = morph.weight;
                match &morph.origin.typ {
                    nanoem::model::ModelMorphType::Group(children) => {
                        for (child_weight, target_morph_idx) in children
                            .iter()
                            .map(|child| (child.weight, usize::try_from(child.morph_index).ok()))
                            .collect::<Vec<_>>()
                        {
                            if let Some(target_morph) =
                                target_morph_idx.and_then(|idx| self.morphs.get_mut(idx))
                            {
                                if let nanoem::model::ModelMorphType::Flip(_) =
                                    target_morph.origin.typ
                                {
                                    target_morph.set_forced_weight(weight * child_weight);
                                    self.deform_morph(target_morph_idx.unwrap(), false);
                                }
                            }
                        }
                    }

                    nanoem::model::ModelMorphType::Flip(children) => {}
                    nanoem::model::ModelMorphType::Impulse(children) => {
                        for child in children {
                            if let Some(rigid_body) = usize::try_from(child.rigid_body_index)
                                .ok()
                                .and_then(|idx| self.rigid_bodies.get_mut(idx))
                            {
                                let torque = f128_to_vec3(child.torque);
                                let velocity = f128_to_vec3(child.velocity);
                                if torque.abs_diff_eq(
                                    &Vector3::zero(),
                                    Vector3::<f32>::default_epsilon(),
                                ) && velocity.abs_diff_eq(
                                    &Vector3::zero(),
                                    Vector3::<f32>::default_epsilon(),
                                ) {
                                    rigid_body.mark_all_forces_reset();
                                } else if child.is_local {
                                    rigid_body.add_local_torque_force(torque, weight);
                                    rigid_body.add_local_velocity_force(velocity, weight);
                                } else {
                                    rigid_body.add_global_torque_force(torque, weight);
                                    rigid_body.add_global_velocity_force(velocity, weight);
                                }
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Material(children) => {
                        for child in children {
                            if let Some(material) = usize::try_from(child.material_index)
                                .ok()
                                .and_then(|idx| self.materials.get_mut(idx))
                            {
                                material.deform(child, weight);
                            } else {
                                for material in &mut self.materials {
                                    material.deform(child, weight);
                                }
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Bone(children) => {
                        for child in children {
                            if let Some(bone) = usize::try_from(child.bone_index)
                                .ok()
                                .and_then(|idx| self.bones.get_mut(idx))
                            {
                                bone.update_local_morph_transform(child, weight);
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Vertex(children) => {
                        for child in children {
                            if let Some(vertex) = usize::try_from(child.vertex_index)
                                .ok()
                                .and_then(|idx| self.vertices.get_mut(idx))
                            {
                                vertex.deform(child, weight);
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Texture(children)
                    | nanoem::model::ModelMorphType::Uva1(children)
                    | nanoem::model::ModelMorphType::Uva2(children)
                    | nanoem::model::ModelMorphType::Uva3(children)
                    | nanoem::model::ModelMorphType::Uva4(children) => {
                        for child in children {
                            if let Some(vertex) = usize::try_from(child.vertex_index)
                                .ok()
                                .and_then(|idx| self.vertices.get_mut(idx))
                            {
                                vertex.deform_uv(
                                    child,
                                    morph.origin.typ.uv_index().unwrap(),
                                    weight,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn perform_all_bones_transform(
        &mut self,
        physics_engine: &mut PhysicsEngine,
        physics_simulation_time_step: f32,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
        self.bounding_box.reset();
        self.apply_all_bones_transform(SimulationTiming::Before, outside_parent_bone_map);
        if physics_engine.simulation_mode == SimulationMode::EnableAnytime {
            self.synchronize_all_rigid_bodies_transform_feedback_to_simulation(physics_engine);
            physics_engine.step(physics_simulation_time_step);
            self.synchronize_all_rigid_bodies_transform_feedback_from_simulation(
                RigidBodyFollowBone::Skip,
                physics_engine,
            );
        }
        self.apply_all_bones_transform(SimulationTiming::After, outside_parent_bone_map);
        self.mark_staging_vertex_buffer_dirty();
        // TODO: handle owned camera
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn vertices_len(&self) -> usize {
        self.vertices.len()
    }

    pub fn bones(&self) -> &BoneSet {
        &self.bones
    }

    pub fn active_bone(&self) -> Option<&Bone> {
        self.active_bone_pair.0.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active_bone(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.0 != bone_idx {
            // TODO: publish set event
            self.active_bone_pair.0 = bone_idx;
        }
    }

    pub fn morphs(&self) -> &[Morph] {
        &self.morphs
    }

    pub fn textures(&self) -> &[NanoemTexture] {
        &self.opaque.textures
    }

    pub fn has_any_dirty_bone(&self) -> bool {
        self.bones
            .iter()
            .map(|bone| bone.states.dirty)
            .fold(false, |a, b| a | b)
    }

    pub fn has_any_dirty_morph(&self) -> bool {
        self.morphs
            .iter()
            .map(|morph| morph.dirty)
            .fold(false, |a, b| a | b)
    }

    pub fn active_outside_parent_subject_bone(&self) -> Option<&Bone> {
        self.active_bone_pair.1.and_then(|idx| self.bones.get(idx))
    }

    pub fn set_active_outside_parent_subject_bone(&mut self, bone_idx: Option<BoneIndex>) {
        if self.active_bone_pair.1 != bone_idx {
            self.active_bone_pair.1 = bone_idx;
        }
    }

    pub fn find_bone(&self, name: &str) -> Option<&Bone> {
        self.bones.find(name)
    }

    pub fn find_bone_mut(&mut self, name: &str) -> Option<&mut Bone> {
        self.bones.find_mut(name)
    }

    pub fn find_morph(&self, name: &str) -> Option<&Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|index| self.morphs.get(*index))
    }

    pub fn find_morph_mut(&mut self, name: &str) -> Option<&mut Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|index| self.morphs.get_mut(*index))
    }

    pub fn parent_bone(&self, bone: &Bone) -> Option<&Bone> {
        usize::try_from(bone.origin.parent_bone_index)
            .ok()
            .and_then(|idx| self.bones.get(idx))
    }

    fn internal_edge_size(
        bones: &BoneSet,
        camera: &dyn Camera,
        edge_size_scale_factor: f32,
    ) -> f32 {
        if let Some(second_bone) = &bones.get(1) {
            let bone_position = second_bone.world_translation();
            (bone_position - camera.position()).magnitude()
                * (camera.fov() as f32 / 30f32).clamp(0f32, 1f32)
                * 0.001f32
                * edge_size_scale_factor
        } else {
            0f32
        }
    }

    pub fn edge_size(&self, camera: &dyn Camera) -> f32 {
        Self::internal_edge_size(&self.bones, camera, self.edge_size_scale_factor)
    }

    pub fn edge_size_scale_factor(&self) -> f32 {
        self.edge_size_scale_factor
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_canonical_name(&self) -> &str {
        &self.canonical_name
    }

    pub fn is_add_blend_enabled(&self) -> bool {
        self.states.enable_add_blend
    }

    pub fn is_physics_simulation_enabled(&self) -> bool {
        self.states.physics_simulation
    }

    pub fn is_staging_vertex_buffer_dirty(&self) -> bool {
        self.states.dirty_staging_buffer
    }

    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    pub fn world_transform(&self, initial: &Matrix4<f32>) -> Matrix4<f32> {
        initial.clone()
    }

    pub fn contains_bone(&self, name: &str) -> bool {
        self.bones.find(name).is_some()
    }

    pub fn contains_morph(&self, name: &str) -> bool {
        self.morphs_by_name.contains_key(name)
    }
}

impl Drawable for Model {
    fn draw(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        typ: DrawType,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.is_visible() {
            match typ {
                DrawType::Color | DrawType::ScriptExternalColor => self.draw_color(
                    typ == DrawType::ScriptExternalColor,
                    color_view,
                    depth_view,
                    context,
                    device,
                    queue,
                ),
                DrawType::Edge => {
                    let edge_size_scale_factor = self.edge_size(context.camera);
                    if edge_size_scale_factor > 0f32 {
                        self.draw_edge(
                            edge_size_scale_factor,
                            color_view,
                            depth_view,
                            context,
                            device,
                            queue,
                        );
                    }
                }
                DrawType::GroundShadow => {
                    if self.states.enable_ground_shadow {
                        self.draw_ground_shadow(color_view, depth_view, context, device, queue)
                    }
                }
                DrawType::ShadowMap => {
                    if self.states.enable_shadow_map {
                        self.draw_shadow_map(color_view, depth_view, context, device, queue)
                    }
                }
            }
        }
    }

    fn is_visible(&self) -> bool {
        self.states.visible
    }
}

impl Model {
    pub fn mark_staging_vertex_buffer_dirty(&mut self) {
        self.states.dirty_staging_buffer = true;
    }

    pub fn update_staging_vertex_buffer(
        &mut self,
        camera: &dyn Camera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.states.dirty_staging_buffer {
            match &self.skin_deformer {
                Deformer::Wgpu(deformer) => {
                    deformer.update_buffer(self, camera, queue);
                    deformer.execute(
                        &self.vertex_buffers[self.stage_vertex_buffer_index],
                        device,
                        queue,
                    );
                }
                Deformer::Software(deformer) => deformer.execute_model(
                    self,
                    camera,
                    &self.vertex_buffers[self.stage_vertex_buffer_index],
                    device,
                    queue,
                ),
            }
            self.stage_vertex_buffer_index = 1 - self.stage_vertex_buffer_index;
            self.states.dirty_morph = false;
            self.states.dirty_staging_buffer = false;
        }
    }

    fn draw_color(
        &self,
        script_external_color: bool,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut uniform_data = self.uniform_bind.get_empty_uniform_data();
        uniform_data.set_camera_parameters(
            context.camera,
            &context.world.unwrap_or(Self::INITIAL_WORLD_MATRIX),
            self,
        );
        uniform_data.set_light_parameters(context.light);
        uniform_data.set_all_model_parameters(self, context.all_models);
        for (idx, material) in self.materials.iter().enumerate() {
            uniform_data.set_material_parameters(idx, material);
        }
        uniform_data.set_shadow_map_parameters(
            context.shadow,
            &context.world.unwrap_or(Self::INITIAL_WORLD_MATRIX),
            context.camera,
            context.light,
            TechniqueType::Color,
        );
        self.uniform_bind.update(&uniform_data, queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model/CommandEncoder/Color"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Model/RenderPass/Color"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            let bundles = self
                .materials
                .iter()
                .map(|material| &material.object_bundle);
            rpass.execute_bundles(bundles);
        }
        queue.submit(iter::once(encoder.finish()));
    }

    fn draw_edge(
        &self,
        edge_size_scale_factor: f32,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mut uniform_data = self.uniform_bind.get_empty_uniform_data();
        uniform_data.set_camera_parameters(
            context.camera,
            &context.world.unwrap_or(Self::INITIAL_WORLD_MATRIX),
            self,
        );
        uniform_data.set_light_parameters(context.light);
        uniform_data.set_all_model_parameters(self, context.all_models);
        for (idx, material) in self.materials.iter().enumerate() {
            uniform_data.set_material_parameters(idx, material);
            uniform_data.set_edge_parameters(idx, material, edge_size_scale_factor);
        }
        self.uniform_bind.update(&uniform_data, queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model/CommandEncoder/Edge"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Model/RenderPass/Edge"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            let bundles = self.materials.iter().map(|material| &material.edge_bundle);
            rpass.execute_bundles(bundles);
        }
        queue.submit(iter::once(encoder.finish()));
    }

    fn draw_ground_shadow(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let world = context.light.get_shadow_transform();
        let mut uniform_data = self.uniform_bind.get_empty_uniform_data();
        uniform_data.set_camera_parameters(context.camera, &world, self);
        uniform_data.set_light_parameters(context.light);
        uniform_data.set_all_model_parameters(self, context.all_models);
        for (idx, material) in self.materials.iter().enumerate() {
            uniform_data.set_material_parameters(idx, &material);
        }
        uniform_data.set_ground_shadow_parameters(context.light, context.camera, &world);
        self.uniform_bind.update(&uniform_data, queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model/CommandEncoder/GroundShadow"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Model/RenderPass/GroundShadow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            let bundles = self
                .materials
                .iter()
                .map(|material| &material.shadow_bundle);
            rpass.execute_bundles(bundles);
        }
        queue.submit(iter::once(encoder.finish()));
    }

    fn draw_shadow_map(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        context: DrawContext,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let world: Matrix4<f32> = context.light.get_shadow_transform();
        let mut uniform_data = self.uniform_bind.get_empty_uniform_data();
        uniform_data.set_camera_parameters(
            context.camera,
            &context.world.unwrap_or(Self::INITIAL_WORLD_MATRIX),
            self,
        );
        uniform_data.set_light_parameters(context.light);
        uniform_data.set_all_model_parameters(self, context.all_models);
        uniform_data.set_shadow_map_parameters(
            context.shadow,
            &context.world.unwrap_or(Self::INITIAL_WORLD_MATRIX),
            context.camera,
            context.light,
            TechniqueType::Zplot,
        );
        self.uniform_bind.update(&uniform_data, queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model/CommandEncoder/ShadowMap"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Model/RenderPass/ShadowMap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: None,
                    }
                }),
            });
            let bundles = self.materials.iter().map(|material| &material.zplot_bundle);
            rpass.execute_bundles(bundles);
        }
        queue.submit(iter::once(encoder.finish()));
    }

    // pub fn draw_bones(&self, device: &wgpu::Device) {

    // }
}

// TODO: Optimize duplicated vertex structure
#[derive(Debug, Clone, Copy)]
pub struct VertexSimd {
    // may use simd128
    pub origin: Vector4<f32>,
    pub normal: Vector4<f32>,
    pub texcoord: Vector4<f32>,
    pub info: Vector4<f32>,
    pub indices: Vector4<f32>,
    pub delta: Vector4<f32>,
    pub weights: Vector4<f32>,
    pub origin_uva: [Vector4<f32>; 4],
    pub delta_uva: [Vector4<f32>; 5],
}

#[derive(Clone)]
pub struct Vertex {
    material: Option<MaterialIndex>,
    soft_body: Option<SoftBodyIndex>,
    bones: [Option<BoneIndex>; 4],
    states: u32,
    pub simd: VertexSimd,
    pub origin: NanoemVertex,
}

impl Vertex {
    const PRIVATE_STATE_SKINNING_ENABLED: u32 = 1 << 1;
    const PRIVATE_STATE_EDITING_MASKED: u32 = 1 << 2;
    const PRIVATE_STATE_INITIAL_VALUE: u32 = 0;

    fn from_nanoem(vertex: &NanoemVertex) -> Self {
        let direction = Vector4::new(1f32, 1f32, 1f32, 1f32);
        let texcoord = vertex.get_tex_coord();
        let bone_indices: [i32; 4] = vertex.get_bone_indices();
        let mut bones = [None; 4];
        match vertex.typ {
            nanoem::model::ModelVertexType::UNKNOWN => {}
            nanoem::model::ModelVertexType::BDEF1 => {
                bones[0] = vertex.bone_indices.get(0).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
            }
            nanoem::model::ModelVertexType::BDEF2 | nanoem::model::ModelVertexType::SDEF => {
                bones[0] = vertex.bone_indices.get(0).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
                bones[1] = vertex.bone_indices.get(1).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
            }
            nanoem::model::ModelVertexType::BDEF4 | nanoem::model::ModelVertexType::QDEF => {
                bones[0] = vertex.bone_indices.get(0).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
                bones[1] = vertex.bone_indices.get(1).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
                bones[2] = vertex.bone_indices.get(2).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
                bones[3] = vertex.bone_indices.get(3).and_then(|idx| {
                    if *idx >= 0 {
                        Some(*idx as usize)
                    } else {
                        None
                    }
                });
            }
        }
        let simd = VertexSimd {
            origin: vertex.get_origin().into(),
            normal: vertex.get_normal().into(),
            texcoord: Vector4::new(
                texcoord[0].fract(),
                texcoord[1].fract(),
                texcoord[2],
                texcoord[3],
            ),
            info: Vector4::new(
                vertex.edge_size,
                i32::from(vertex.typ) as f32,
                vertex.get_index() as f32,
                1f32,
            ),
            indices: bones
                .map(|bone_idx| bone_idx.map(|idx| idx as f32).unwrap_or(-1f32))
                .into(),
            delta: Vector4::zero(),
            weights: vertex.get_bone_weights().into(),
            origin_uva: vertex.get_additional_uv().map(|uv| uv.into()),
            delta_uva: [
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
                Vector4::zero(),
            ],
        };
        Self {
            material: None,
            soft_body: None,
            bones,
            states: Self::PRIVATE_STATE_INITIAL_VALUE,
            simd,
            origin: vertex.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.simd.delta = Vector4::zero();
        self.simd.delta_uva = [Vector4::zero(); 5];
    }

    pub fn deform(&mut self, morph: &nanoem::model::ModelMorphVertex, weight: f32) {
        self.simd.delta = self.simd.delta + f128_to_vec4(morph.position) * weight;
    }

    pub fn deform_uv(&mut self, morph: &nanoem::model::ModelMorphUv, uv_idx: usize, weight: f32) {
        self.simd.delta_uva[uv_idx].add_assign_element_wise(f128_to_vec4(morph.position) * weight);
    }

    pub fn set_material(&mut self, material_idx: MaterialIndex) {
        self.material = Some(material_idx)
    }

    pub fn set_skinning_enabled(&mut self, value: bool) {
        self.states = if value {
            self.states | Self::PRIVATE_STATE_SKINNING_ENABLED
        } else {
            self.states & !Self::PRIVATE_STATE_SKINNING_ENABLED
        }
    }
}

pub struct MaterialDrawContext<'a> {
    pub effect: &'a mut ModelProgramBundle,
    pub fallback_texture: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub bind_group_layout: &'a wgpu::BindGroupLayout,
    pub color_format: wgpu::TextureFormat,
    pub is_add_blend: bool,
    pub uniform_bind: &'a wgpu::BindGroup,
    pub shadow_bind: &'a wgpu::BindGroup,
    pub fallback_texture_bind: &'a wgpu::BindGroup,
    pub fallback_shadow_bind: &'a wgpu::BindGroup,
    pub vertex_buffers: &'a [wgpu::Buffer; 2],
    pub index_buffer: &'a wgpu::Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialColor {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub diffuse_opacity: f32,
    pub specular_power: f32,
    pub diffuse_texture_blend_factor: Vector4<f32>,
    pub sphere_texture_blend_factor: Vector4<f32>,
    pub toon_texture_blend_factor: Vector4<f32>,
}

impl MaterialColor {
    pub fn new_reset(v: f32) -> Self {
        Self {
            ambient: Vector3::new(v, v, v),
            diffuse: Vector3::new(v, v, v),
            specular: Vector3::new(v, v, v),
            diffuse_opacity: v,
            specular_power: v.max(Material::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: Vector4::new(v, v, v, v),
            sphere_texture_blend_factor: Vector4::new(v, v, v, v),
            toon_texture_blend_factor: Vector4::new(v, v, v, v),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MaterialBlendColor {
    base: MaterialColor,
    add: MaterialColor,
    mul: MaterialColor,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialEdge {
    pub color: Vector3<f32>,
    pub opacity: f32,
    pub size: f32,
}

impl MaterialEdge {
    pub fn new_reset(v: f32) -> Self {
        Self {
            color: Vector3::new(v, v, v),
            opacity: v,
            size: v,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MaterialBlendEdge {
    base: MaterialEdge,
    add: MaterialEdge,
    mul: MaterialEdge,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialStates {
    pub visible: bool,
    pub display_diffuse_texture_uv_mesh_enabled: bool,
    pub display_sphere_map_texture_uv_mesh_enabled: bool,
}

#[derive(Debug)]
pub struct Material {
    // TODO
    color: MaterialBlendColor,
    edge: MaterialBlendEdge,
    diffuse_image: Option<wgpu::TextureView>,
    sphere_map_image: Option<wgpu::TextureView>,
    toon_image: Option<wgpu::TextureView>,
    texture_bind: wgpu::BindGroup,
    object_bundle: wgpu::RenderBundle,
    edge_bundle: wgpu::RenderBundle,
    shadow_bundle: wgpu::RenderBundle,
    zplot_bundle: wgpu::RenderBundle,
    name: String,
    canonical_name: String,
    index_hash: HashMap<u32, u32>,
    toon_color: Vector4<f32>,
    states: MaterialStates,
    origin: NanoemMaterial,
}

impl Material {
    pub const MINIUM_SPECULAR_POWER: f32 = 0.1f32;

    pub fn from_nanoem(
        material: &NanoemMaterial,
        language_type: nanoem::common::LanguageType,
        ctx: &mut MaterialDrawContext,
        num_offset: u32,
        num_indices: u32,
        device: &wgpu::Device,
    ) -> Self {
        let mut name = material.get_name(language_type).to_owned();
        let mut canonical_name = material
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Material{}", material.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("Model/TextureBindGroup/Material {}", &canonical_name).as_str()),
            layout: ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ctx.fallback_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
            ],
        });
        let flags = material.flags;
        let (object_bundle, edge_bundle, shadow_bundle, zplot_bundle) = Self::build_bundles(
            material.get_index(),
            ctx.effect,
            ctx.color_format,
            ctx.is_add_blend,
            flags.is_line_draw_enabled,
            flags.is_point_draw_enabled,
            flags.is_culling_disabled,
            &bind_group,
            ctx.uniform_bind,
            ctx.shadow_bind,
            ctx.fallback_texture_bind,
            ctx.fallback_shadow_bind,
            &ctx.vertex_buffers[0],
            ctx.index_buffer,
            num_offset,
            num_indices,
            device,
        );
        Self {
            color: MaterialBlendColor {
                base: MaterialColor {
                    ambient: Vector4::from(material.get_ambient_color()).truncate(),
                    diffuse: Vector4::from(material.get_diffuse_color()).truncate(),
                    specular: Vector4::from(material.get_specular_color()).truncate(),
                    diffuse_opacity: material.get_diffuse_opacity(),
                    specular_power: material.get_specular_power(),
                    diffuse_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    sphere_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                    toon_texture_blend_factor: Vector4::new(1f32, 1f32, 1f32, 1f32),
                },
                add: MaterialColor::new_reset(0f32),
                mul: MaterialColor::new_reset(1f32),
            },
            edge: MaterialBlendEdge {
                base: MaterialEdge {
                    color: Vector4::from(material.get_edge_color()).truncate(),
                    opacity: material.get_edge_opacity(),
                    size: material.get_edge_size(),
                },
                add: MaterialEdge::new_reset(0f32),
                mul: MaterialEdge::new_reset(1f32),
            },
            diffuse_image: None,
            sphere_map_image: None,
            toon_image: None,
            texture_bind: bind_group,
            object_bundle,
            edge_bundle,
            shadow_bundle,
            zplot_bundle,
            name,
            canonical_name,
            index_hash: HashMap::new(),
            toon_color: Vector4::new(1f32, 1f32, 1f32, 1f32),
            states: MaterialStates {
                visible: true,
                display_diffuse_texture_uv_mesh_enabled: true,
                ..Default::default()
            },
            origin: material.clone(),
        }
    }

    pub fn reset(&mut self) {
        let material = &self.origin;
        self.color.base.ambient = Vector4::from(material.get_ambient_color()).truncate();
        self.color.base.diffuse = Vector4::from(material.get_diffuse_color()).truncate();
        self.color.base.specular = Vector4::from(material.get_specular_color()).truncate();
        self.color.base.diffuse_opacity = material.get_diffuse_opacity();
        self.color.base.specular_power = material
            .get_specular_power()
            .max(Self::MINIUM_SPECULAR_POWER);
        self.color.base.diffuse_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.sphere_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.color.base.toon_texture_blend_factor = Vector4::new(1f32, 1f32, 1f32, 1f32);
        self.edge.base.color = Vector4::from(material.get_edge_color()).truncate();
        self.edge.base.opacity = material.get_edge_opacity();
        self.edge.base.size = material.get_edge_size();
    }

    pub fn reset_deform(&mut self) {
        self.color.mul = MaterialColor::new_reset(1f32);
        self.color.add = MaterialColor::new_reset(0f32);
        self.edge.mul = MaterialEdge::new_reset(1f32);
        self.edge.add = MaterialEdge::new_reset(0f32);
    }

    pub fn deform(&mut self, morph: &nanoem::model::ModelMorphMaterial, weight: f32) {
        const ONE_V4: Vector4<f32> = Vector4 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
            w: 1f32,
        };
        const ONE_V3: Vector3<f32> = Vector3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        };
        let diffuse_texture_blend_factor = f128_to_vec4(morph.diffuse_texture_blend);
        let sphere_texture_blend_factor = f128_to_vec4(morph.sphere_map_texture_blend);
        // TODO: nanoem use sphere_map_texture_blend, it may be a mistake
        let toon_texture_blend_factor = f128_to_vec4(morph.toon_texture_blend);
        match morph.operation {
            nanoem::model::ModelMorphMaterialOperationType::Multiply => {
                self.color.mul.ambient.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.mul.diffuse.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.mul.specular.mul_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.mul.diffuse_opacity = lerp_f32(
                    self.color.mul.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.mul.specular_power =
                    lerp_f32(self.color.mul.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .mul
                    .diffuse_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .mul
                    .sphere_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .mul
                    .toon_texture_blend_factor
                    .mul_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .mul
                    .color
                    .mul_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.mul.opacity = lerp_f32(self.edge.mul.opacity, morph.edge_opacity, weight);
                self.edge.mul.size = lerp_f32(self.edge.mul.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Add => {
                self.color.add.ambient.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.ambient_color), weight),
                );
                self.color.add.diffuse.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.diffuse_color), weight),
                );
                self.color.add.specular.add_assign_element_wise(
                    ONE_V3.lerp(f128_to_vec3(morph.specular_color), weight),
                );
                self.color.add.diffuse_opacity = lerp_f32(
                    self.color.add.diffuse_opacity,
                    morph.diffuse_opacity,
                    weight,
                );
                self.color.add.specular_power =
                    lerp_f32(self.color.add.specular_power, morph.specular_power, weight)
                        .max(Self::MINIUM_SPECULAR_POWER);
                self.color
                    .add
                    .diffuse_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.diffuse_texture_blend), weight),
                    );
                self.color
                    .add
                    .sphere_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.sphere_map_texture_blend), weight),
                    );
                self.color
                    .add
                    .toon_texture_blend_factor
                    .add_assign_element_wise(
                        ONE_V4.lerp(f128_to_vec4(morph.toon_texture_blend), weight),
                    );
                self.edge
                    .add
                    .color
                    .add_assign_element_wise(ONE_V3.lerp(f128_to_vec3(morph.edge_color), weight));
                self.edge.add.opacity = lerp_f32(self.edge.add.opacity, morph.edge_opacity, weight);
                self.edge.add.size = lerp_f32(self.edge.add.size, morph.edge_size, weight);
            }
            nanoem::model::ModelMorphMaterialOperationType::Unknown => {}
        }
    }

    pub fn is_visible(&self) -> bool {
        self.states.visible
    }

    pub fn color(&self) -> MaterialColor {
        MaterialColor {
            ambient: self
                .color
                .base
                .ambient
                .mul_element_wise(self.color.mul.ambient)
                + self.color.add.ambient,
            diffuse: self
                .color
                .base
                .diffuse
                .mul_element_wise(self.color.mul.diffuse)
                + self.color.add.diffuse,
            specular: self
                .color
                .base
                .specular
                .mul_element_wise(self.color.mul.specular)
                + self.color.add.specular,
            diffuse_opacity: self.color.base.diffuse_opacity * self.color.mul.diffuse_opacity
                + self.color.add.diffuse_opacity,
            specular_power: (self.color.base.specular_power * self.color.mul.specular_power
                + self.color.add.specular_power)
                .min(Self::MINIUM_SPECULAR_POWER),
            diffuse_texture_blend_factor: self
                .color
                .base
                .diffuse_texture_blend_factor
                .mul_element_wise(self.color.mul.diffuse_texture_blend_factor)
                + self.color.add.diffuse_texture_blend_factor,
            sphere_texture_blend_factor: self
                .color
                .base
                .sphere_texture_blend_factor
                .mul_element_wise(self.color.mul.sphere_texture_blend_factor)
                + self.color.add.sphere_texture_blend_factor,
            toon_texture_blend_factor: self
                .color
                .base
                .toon_texture_blend_factor
                .mul_element_wise(self.color.mul.toon_texture_blend_factor)
                + self.color.add.toon_texture_blend_factor,
        }
    }

    pub fn edge(&self) -> MaterialEdge {
        MaterialEdge {
            color: self.edge.base.color.mul_element_wise(self.edge.mul.color) + self.edge.add.color,
            opacity: self.edge.base.opacity * self.edge.mul.opacity + self.edge.add.opacity,
            size: self.edge.base.size * self.edge.mul.size + self.edge.add.size,
        }
    }

    pub fn diffuse_view(&self) -> Option<&wgpu::TextureView> {
        self.diffuse_image.as_ref()
    }

    pub fn sphere_map_view(&self) -> Option<&wgpu::TextureView> {
        self.sphere_map_image.as_ref()
    }

    pub fn toon_view(&self) -> Option<&wgpu::TextureView> {
        self.toon_image.as_ref()
    }

    pub fn update_bind(
        &mut self,
        ctx: &mut MaterialDrawContext,
        num_offset: u32,
        num_indices: u32,
        device: &wgpu::Device,
    ) {
        self.texture_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("Model/TextureBindGroup/Material").as_str()),
            layout: ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.diffuse_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.sphere_map_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        self.toon_view().unwrap_or(ctx.fallback_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
            ],
        });
        self.rebuild_bundles(ctx, num_offset, num_indices, device);
    }

    pub fn rebuild_bundles(
        &mut self,
        ctx: &mut MaterialDrawContext,
        num_offset: u32,
        num_indices: u32,
        device: &wgpu::Device,
    ) {
        let material_idx = self.origin.get_index();
        let flags = self.origin.flags;
        let (object_bundle, edge_bundle, shadow_bundle, zplot_bundle) = Self::build_bundles(
            material_idx,
            ctx.effect,
            ctx.color_format,
            ctx.is_add_blend,
            flags.is_line_draw_enabled,
            flags.is_point_draw_enabled,
            flags.is_culling_disabled,
            &self.texture_bind,
            ctx.uniform_bind,
            ctx.shadow_bind,
            ctx.fallback_texture_bind,
            ctx.fallback_shadow_bind,
            &ctx.vertex_buffers[0],
            ctx.index_buffer,
            num_offset,
            num_indices,
            device,
        );
        self.object_bundle = object_bundle;
        self.edge_bundle = edge_bundle;
        self.shadow_bundle = shadow_bundle;
        self.zplot_bundle = zplot_bundle;
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.texture_bind
    }

    pub fn sphere_map_texture_type(&self) -> nanoem::model::ModelMaterialSphereMapTextureType {
        self.origin.sphere_map_texture_type
    }

    pub fn is_culling_disabled(&self) -> bool {
        self.origin.flags.is_culling_disabled
    }

    pub fn is_casting_shadow_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_enabled
    }
    pub fn is_casting_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_casting_shadow_map_enabled
    }
    pub fn is_shadow_map_enabled(&self) -> bool {
        self.origin.flags.is_shadow_map_enabled
    }
    pub fn is_edge_enabled(&self) -> bool {
        self.origin.flags.is_edge_enabled
    }
    pub fn is_vertex_color_enabled(&self) -> bool {
        self.origin.flags.is_vertex_color_enabled
    }
    pub fn is_point_draw_enabled(&self) -> bool {
        self.origin.flags.is_point_draw_enabled
    }
    pub fn is_line_draw_enabled(&self) -> bool {
        self.origin.flags.is_line_draw_enabled
    }
}

impl Material {
    fn build_bundles(
        material_idx: usize,
        effect: &mut ModelProgramBundle,
        color_format: wgpu::TextureFormat,
        is_add_blend: bool,
        line_draw_enabled: bool,
        point_draw_enabled: bool,
        culling_disabled: bool,
        color_bind: &wgpu::BindGroup,
        uniform_bind: &wgpu::BindGroup,
        shadow_bind: &wgpu::BindGroup,
        fallback_texture_bind: &wgpu::BindGroup,
        fallback_shadow_bind: &wgpu::BindGroup,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        num_offset: u32,
        num_indices: u32,
        device: &wgpu::Device,
    ) -> (
        wgpu::RenderBundle,
        wgpu::RenderBundle,
        wgpu::RenderBundle,
        wgpu::RenderBundle,
    ) {
        let object_bundle = effect.ensure_get_object_render_bundle(
            ObjectPassKey {
                color_format,
                is_add_blend,
                depth_enabled: true,
                line_draw_enabled,
                point_draw_enabled,
                culling_disabled,
            },
            material_idx,
            CPassBindGroup {
                color_bind,
                uniform_bind,
                shadow_bind,
            },
            CPassVertexBuffer {
                vertex_buffer,
                index_buffer,
                num_offset,
                num_indices,
            },
            device,
        );
        let edge_bundle = effect.ensure_get_edge_render_bundle(
            EdgePassKey {
                color_format,
                is_add_blend,
                depth_enabled: true,
                line_draw_enabled,
                point_draw_enabled,
            },
            material_idx,
            CPassBindGroup {
                color_bind,
                uniform_bind,
                shadow_bind,
            },
            CPassVertexBuffer {
                vertex_buffer,
                index_buffer,
                num_offset,
                num_indices,
            },
            device,
        );
        let shadow_bundle = effect.ensure_get_shadow_render_bundle(
            ShadowPassKey {
                color_format,
                is_add_blend,
                depth_enabled: true,
                line_draw_enabled,
                point_draw_enabled,
            },
            material_idx,
            CPassBindGroup {
                color_bind,
                uniform_bind,
                shadow_bind,
            },
            CPassVertexBuffer {
                vertex_buffer,
                index_buffer,
                num_offset,
                num_indices,
            },
            device,
        );
        let zplot_bundle = effect.ensure_get_zplot_render_bundle(
            ZplotPassKey {
                depth_enabled: true,
                line_draw_enabled,
                point_draw_enabled,
                culling_disabled,
            },
            material_idx,
            CPassBindGroup {
                color_bind: fallback_texture_bind,
                uniform_bind,
                shadow_bind: fallback_shadow_bind,
            },
            CPassVertexBuffer {
                vertex_buffer,
                index_buffer,
                num_offset,
                num_indices,
            },
            device,
        );
        (object_bundle, edge_bundle, shadow_bundle, zplot_bundle)
    }
}

pub struct Morph {
    pub name: String,
    pub canonical_name: String,
    weight: f32,
    pub dirty: bool,
    pub origin: NanoemMorph,
}

impl Morph {
    pub fn from_nanoem(morph: &NanoemMorph, language: nanoem::common::LanguageType) -> Self {
        let mut name = morph.get_name(language).to_owned();
        let mut canonical_name = morph
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Morph{}", morph.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        Self {
            name,
            canonical_name,
            weight: 0f32,
            dirty: false,
            origin: morph.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.weight = 0f32;
        self.dirty = false;
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn set_weight(&mut self, value: f32) {
        self.dirty = self.weight.abs() > f32::EPSILON || value.abs() > f32::EPSILON;
        self.weight = value;
    }

    pub fn set_forced_weight(&mut self, value: f32) {
        self.dirty = false;
        self.weight = value;
    }

    pub fn synchronize_motion(
        &mut self,
        motion: &Motion,
        name: &str,
        frame_index: u32,
        amount: f32,
    ) {
        let w0 = self.synchronize_weight(motion, name, frame_index);
        if amount > 0f32 {
            let w1 = self.synchronize_weight(motion, name, frame_index + 1);
            self.set_weight(lerp_f32(w0, w1, amount));
        } else {
            self.set_weight(w0);
        }
    }

    fn synchronize_weight(&mut self, motion: &Motion, name: &str, frame_index: u32) -> f32 {
        if let Some(keyframe) = motion.find_morph_keyframe(name, frame_index) {
            keyframe.weight
        } else if let (Some(prev_frame), Some(next_frame)) = motion
            .opaque
            .search_closest_morph_keyframes(name, frame_index)
        {
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            lerp_f32(prev_frame.weight, next_frame.weight, coef)
        } else {
            0f32
        }
    }
}

pub struct Label {
    name: String,
    canonical_name: String,
    origin: NanoemLabel,
}

impl Label {
    const NAME_EXPRESSION_IN_JAPANESE_UTF8: &'static [u8] =
        &[0xe8, 0xa1, 0xa8, 0xe6, 0x83, 0x85, 0x0];
    const NAME_EXPRESSION_IN_JAPANESE: &'static str = "表情";

    pub fn from_nanoem(label: &NanoemLabel, language: nanoem::common::LanguageType) -> Self {
        let mut name = label.get_name(language).to_owned();
        let mut canonical_name = label
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", label.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        Self {
            name,
            canonical_name,
            origin: label.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RigidBodyStates {
    pub enabled: bool,
    pub all_forces_should_reset: bool,
    pub editing_masked: bool,
}

#[derive(Debug)]
pub struct RigidBody {
    // TODO: physics engine and shape mesh and engine rigid_body
    physics_rigid_body: rapier3d::dynamics::RigidBodyHandle,
    physics_collider: Option<rapier3d::geometry::ColliderHandle>,
    initial_world_transform: nalgebra::Isometry3<f32>,
    global_torque_force: (Vector3<f32>, bool),
    global_velocity_force: (Vector3<f32>, bool),
    local_torque_force: (Vector3<f32>, bool),
    local_velocity_force: (Vector3<f32>, bool),
    name: String,
    canonical_name: String,
    states: RigidBodyStates,
    origin: NanoemRigidBody,
}

impl RigidBody {
    pub const PRIVATE_STATE_ENABLED: u32 = 1u32 << 1;
    pub const PRIVATE_STATE_ALL_FORCES_SHOULD_RESET: u32 = 1u32 << 2;
    pub const PRIVATE_STATE_EDITING_MASKED: u32 = 1u32 << 3;
    pub const PRIVATE_STATE_RESERVED: u32 = 1u32 << 31;

    pub const PRIVATE_STATE_INITIAL_VALUE: u32 = 0u32;

    pub fn from_nanoem(
        rigid_body: &NanoemRigidBody,
        language: nanoem::common::LanguageType,
        is_morph: bool,
        bones: &BoneSet,
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        // TODO: set physics engine and bind to engine rigid_body
        let mut name = rigid_body.get_name(language).to_owned();
        let mut canonical_name = rigid_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", rigid_body.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let orientation = rigid_body.orientation;
        let origin = rigid_body.origin;
        let mut world_transform = f128_to_isometry(origin, orientation);
        let mut initial_world_transform = world_transform;
        if rigid_body.is_bone_relative {
            if let Some(bone) = usize::try_from(rigid_body.bone_index)
                .ok()
                .and_then(|idx| bones.get(idx))
            {
                let bone_origin = bone.origin.origin;
                let offset = nalgebra::Isometry3::translation(
                    bone_origin[0],
                    bone_origin[1],
                    bone_origin[2],
                );
                world_transform = offset * world_transform;
                initial_world_transform = world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                world_transform *= skinning_transform;
            }
        }
        let rigid_body_builder = rapier3d::dynamics::RigidBodyBuilder::new(
            rapier3d::dynamics::RigidBodyType::KinematicPositionBased,
        )
        .position(world_transform)
        .angular_damping(rigid_body.angular_damping)
        .linear_damping(rigid_body.linear_damping);
        let size = rigid_body.size;
        let mut collider_builder = match rigid_body.shape_type {
            nanoem::model::ModelRigidBodyShapeType::Unknown => None,
            nanoem::model::ModelRigidBodyShapeType::Sphere => {
                Some(rapier3d::geometry::ColliderBuilder::ball(size[0]))
            }

            nanoem::model::ModelRigidBodyShapeType::Box => Some(
                rapier3d::geometry::ColliderBuilder::cuboid(size[0], size[1], size[2]),
            ),

            nanoem::model::ModelRigidBodyShapeType::Capsule => Some(
                rapier3d::geometry::ColliderBuilder::capsule_y(size[1] / 2f32, size[0]),
            ),
        };
        let mut mass = rigid_body.mass;
        if let nanoem::model::ModelRigidBodyTransformType::FromBoneToSimulation =
            rigid_body.transform_type
        {
            mass = 0f32;
        }
        collider_builder = collider_builder.map(|builder| {
            builder
                .mass(mass)
                .friction(rigid_body.friction)
                .restitution(rigid_body.restitution)
                .collision_groups(rapier3d::geometry::InteractionGroups::new(
                    rapier3d::geometry::Group::from_bits_truncate(
                        0x1u32 << rigid_body.collision_group_id.clamp(0, 15),
                    ),
                    rapier3d::geometry::Group::from_bits_truncate(
                        (rigid_body.collision_mask & 0xffff) as u32,
                    ),
                ))
        });
        let rigid_body_handle = physics_engine
            .rigid_body_set
            .insert(rigid_body_builder.build());
        let physics_collider = collider_builder.map(|collider_builder| {
            physics_engine.collider_set.insert_with_parent(
                collider_builder.build(),
                rigid_body_handle,
                &mut physics_engine.rigid_body_set,
            )
        });
        Self {
            global_torque_force: (Vector3::zero(), false),
            global_velocity_force: (Vector3::zero(), false),
            local_torque_force: (Vector3::zero(), false),
            local_velocity_force: (Vector3::zero(), false),
            name,
            canonical_name,
            states: RigidBodyStates::default(),
            origin: rigid_body.clone(),
            physics_rigid_body: rigid_body_handle,
            physics_collider,
            initial_world_transform,
        }
    }

    pub fn enable(&mut self) {
        if !self.states.enabled {
            // TODO: add to physics engine
            self.states.enabled = true;
        }
    }

    pub fn disable(&mut self) {
        if self.states.enabled {
            // TODO: remove from physics engine
            self.states.enabled = false;
        }
    }

    pub fn mark_all_forces_reset(&mut self) {
        self.states.all_forces_should_reset = true;
    }

    pub fn add_global_torque_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.global_torque_force.0 += value * weight;
        self.global_torque_force.1 = true;
    }

    pub fn add_global_velocity_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.global_velocity_force.0 += value * weight;
        self.global_velocity_force.1 = true;
    }

    pub fn add_local_torque_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.local_torque_force.0 += value * weight;
        self.local_torque_force.1 = true;
    }

    pub fn add_local_velocity_force(&mut self, value: Vector3<f32>, weight: f32) {
        self.local_velocity_force.0 += value * weight;
        self.local_velocity_force.1 = true;
    }

    pub fn initialize_transform_feedback(
        &mut self,
        bone: &Bone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            let skinning_transform = to_isometry(bone.matrices.skinning_transform);
            physics_rigid_body
                .set_position(skinning_transform * self.initial_world_transform, true);
            physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
            physics_rigid_body.reset_forces(true);
        }
    }

    pub fn enable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if !physics_rigid_body.is_kinematic() {
                physics_rigid_body.set_body_type(
                    rapier3d::dynamics::RigidBodyType::KinematicVelocityBased,
                    true,
                );
            }
        }
    }

    pub fn disable_kinematic(&mut self, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if physics_rigid_body.is_kinematic() {
                physics_rigid_body.set_body_type(rapier3d::dynamics::RigidBodyType::Dynamic, true);
            }
        }
    }

    pub fn apply_all_forces(&mut self, bone: Option<&Bone>, physics_engine: &mut PhysicsEngine) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if self.states.all_forces_should_reset {
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            } else {
                if self.global_torque_force.1 {
                    physics_rigid_body
                        .apply_torque_impulse(to_na_vec3(self.global_torque_force.0), true);
                    self.global_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_torque_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rigid_body.position())
                        .rotation;
                        physics_rigid_body.apply_torque_impulse(
                            local_orientation * to_na_vec3(self.local_torque_force.0),
                            true,
                        );
                        self.local_torque_force = (Vector3::new(0f32, 0f32, 0f32), false);
                    }
                }
                if self.global_velocity_force.1 {
                    physics_rigid_body
                        .apply_impulse(to_na_vec3(self.global_velocity_force.0), true);
                    self.global_velocity_force = (Vector3::new(0f32, 0f32, 0f32), false);
                }
                if self.local_velocity_force.1 {
                    if let Some(bone) = bone {
                        let local_orientation = (to_isometry(bone.matrices.world_transform)
                            * physics_rigid_body.position())
                        .rotation;
                        physics_rigid_body.apply_torque_impulse(
                            local_orientation * to_na_vec3(self.local_velocity_force.0),
                            true,
                        );
                        self.local_velocity_force = (Vector3::new(0f32, 0f32, 0f32), false);
                    }
                }
            }
            self.states.all_forces_should_reset = false;
        }
    }

    pub fn synchronize_transform_feedback_to_simulation(
        &mut self,
        bone: &Bone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if self.origin.transform_type == ModelRigidBodyTransformType::FromBoneToSimulation
                || physics_rigid_body.is_kinematic()
            {
                let initial_transform = self.initial_world_transform;
                let skinning_transform = to_isometry(bone.matrices.skinning_transform);
                let world_transform = skinning_transform * initial_transform;
                physics_rigid_body.set_position(world_transform, true);
                physics_rigid_body.set_linvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.set_angvel(rapier3d::na::vector![0f32, 0f32, 0f32], true);
                physics_rigid_body.reset_forces(true);
            }
        }
    }

    pub fn synchronize_transform_feedback_from_simulation(
        &mut self,
        bone: &mut Bone,
        parent_bone: Option<&Bone>,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if let Some(physics_rigid_body) = physics_engine
            .rigid_body_set
            .get_mut(self.physics_rigid_body)
        {
            if (self.origin.transform_type == ModelRigidBodyTransformType::FromSimulationToBone
                || self.origin.transform_type
                    == ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone)
                && !physics_rigid_body.is_kinematic()
            {
                let initial_transform = self.initial_world_transform;
                let mut world_transform = *physics_rigid_body.position();
                if follow_type == RigidBodyFollowBone::Perform
                    && self.origin.transform_type
                        == ModelRigidBodyTransformType::FromBoneOrientationAndSimulationToBone
                {
                    let local_transform = to_isometry(bone.matrices.local_transform);
                    world_transform =
                        nalgebra::Isometry3::from(local_transform.translation.inverse())
                            * world_transform;
                    physics_rigid_body.set_position(world_transform, true);
                }
                let skinning_transform =
                    from_isometry(world_transform * initial_transform.inverse());
                bone.update_matrices_by_skinning(skinning_transform);
                if let Some(parent_bone) = parent_bone {
                    let offset =
                        f128_to_vec3(self.origin.origin) - f128_to_vec3(parent_bone.origin.origin);
                    let local_transform = parent_bone
                        .matrices
                        .world_transform
                        .affine_invert()
                        .unwrap()
                        * bone.matrices.world_transform;
                    bone.local_user_translation = local_transform[3].truncate() - offset;
                    bone.local_user_orientation = mat4_truncate(local_transform).into();
                } else {
                    let local_transform = bone.matrices.world_transform;
                    bone.local_user_translation =
                        local_transform[3].truncate() - f128_to_vec3(bone.origin.origin);
                    bone.local_user_orientation = mat4_truncate(local_transform).into();
                }
                physics_rigid_body.wake_up(false);
            }
        }
    }

    pub fn is_kinematic(&self, physics_engine: &PhysicsEngine) -> bool {
        physics_engine
            .rigid_body_set
            .get(self.physics_rigid_body)
            .unwrap()
            .is_kinematic()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JointStates {
    pub enabled: bool,
    pub editing_masked: bool,
}

pub struct Joint {
    // TODO: physics engine and shape mesh and engine rigid_body
    name: String,
    canonical_name: String,
    physics_joint: Option<rapier3d::dynamics::ImpulseJointHandle>,
    states: JointStates,
    origin: NanoemJoint,
}

impl Joint {
    pub fn from_nanoem(
        joint: &NanoemJoint,
        language: nanoem::common::LanguageType,
        rigid_bodies: &[RigidBody],
        physics_engine: &mut PhysicsEngine,
    ) -> Self {
        let mut name = joint.get_name(language).to_owned();
        let mut canonical_name = joint
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", joint.get_index());
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        let orientation = joint.orientation;
        let origin = joint.origin;
        let world_transform = f128_to_isometry(origin, orientation);
        let rigid_body_a = usize::try_from(joint.rigid_body_a_index)
            .ok()
            .and_then(|idx| rigid_bodies.get(idx));
        let rigid_body_b = usize::try_from(joint.rigid_body_b_index)
            .ok()
            .and_then(|idx| rigid_bodies.get(idx));
        let physics_joint = if let (Some(rigid_body_a), Some(rigid_body_b)) =
            (rigid_body_a, rigid_body_b)
        {
            let mut physics_joint = rapier3d::dynamics::GenericJoint::default();
            let local_frame_a = rigid_body_a.initial_world_transform.inverse() * world_transform;
            let local_frame_b = rigid_body_b.initial_world_transform.inverse() * world_transform;
            physics_joint.set_local_frame1(local_frame_a);
            physics_joint.set_local_frame2(local_frame_b);

            fn limit(
                joint: &mut rapier3d::dynamics::GenericJoint,
                axis: rapier3d::dynamics::JointAxis,
                min: f32,
                max: f32,
                max_limit: f32,
                stiffness: f32,
            ) {
                if max - min < max_limit && max - min > 0f32 {
                    joint.set_limits(axis, [min, max]);
                    if stiffness > 0f32 {
                        joint.set_motor(axis, 0f32, 0f32, stiffness, 1f32);
                    }
                } else if min == max {
                    joint.lock_axes(axis.into());
                }
            }

            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::X,
                joint.linear_lower_limit[0],
                joint.linear_upper_limit[0],
                100.,
                joint.linear_stiffness[0],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::Y,
                joint.linear_lower_limit[1],
                joint.linear_upper_limit[1],
                100.,
                joint.linear_stiffness[1],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::Z,
                joint.linear_lower_limit[2],
                joint.linear_upper_limit[2],
                100.,
                joint.linear_stiffness[2],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngX,
                joint.angular_lower_limit[0],
                joint.angular_upper_limit[0],
                PI * 2.,
                joint.angular_stiffness[0],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngY,
                joint.angular_lower_limit[1],
                joint.angular_upper_limit[1],
                PI * 2.,
                joint.angular_stiffness[1],
            );
            limit(
                &mut physics_joint,
                rapier3d::dynamics::JointAxis::AngZ,
                joint.angular_lower_limit[2],
                joint.angular_upper_limit[2],
                PI * 2.,
                joint.angular_stiffness[2],
            );
            Some(physics_engine.impulse_joint_set.insert(
                rigid_body_a.physics_rigid_body,
                rigid_body_b.physics_rigid_body,
                physics_joint,
                true,
            ))
        } else {
            None
        };

        Self {
            name,
            canonical_name,
            physics_joint,
            states: JointStates::default(),
            origin: joint.clone(),
        }
    }

    pub fn enable(&mut self) {
        if !self.states.enabled {
            // TODO: add to physics engine
            self.states.enabled = true;
        }
    }

    pub fn disable(&mut self) {
        if self.states.enabled {
            // TODO: remove from physics engine
            self.states.enabled = false;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SoftBodyStates {
    pub enabled: bool,
    pub editing_masked: bool,
}

pub struct SoftBody {
    // TODO: physics engine and shape mesh and engine soft_body
    name: String,
    canonical_name: String,
    pub states: SoftBodyStates,
    origin: NanoemSoftBody,
}

impl SoftBody {
    pub fn from_nanoem(soft_body: &NanoemSoftBody, language: nanoem::common::LanguageType) -> Self {
        // TODO: should resolve physic engine
        let mut name = soft_body.get_name(language).to_owned();
        let mut canonical_name = soft_body
            .get_name(nanoem::common::LanguageType::default())
            .to_owned();
        if canonical_name.is_empty() {
            canonical_name = format!("Label{}", soft_body.base.index);
        }
        if name.is_empty() {
            name = canonical_name.clone();
        }
        Self {
            name,
            canonical_name,
            states: SoftBodyStates::default(),
            origin: soft_body.clone(),
        }
    }

    pub fn enable(&mut self) {
        if !self.states.enabled {
            // TODO: add to physics engine
            self.states.enabled = true;
        }
    }

    pub fn disable(&mut self) {
        if self.states.enabled {
            // TODO: remove from physics engine
            self.states.enabled = false;
        }
    }
}

pub struct VisualizationClause {
    // TODO
}

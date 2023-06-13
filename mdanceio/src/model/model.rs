use std::{
    collections::{HashMap, HashSet},
    iter,
};

use cgmath::{AbsDiffEq, ElementWise, InnerSpace, Matrix4, Vector3, Vector4, VectorSpace, Zero};
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
    model::{material::MaterialDrawContext, VertexUnit},
    motion::Motion,
    physics_engine::{PhysicsEngine, RigidBodyFollowBone, SimulationMode, SimulationTiming},
    utils::{f128_to_vec3, f128_to_vec4, lerp_f32},
};

use super::{
    bone::BoneSet,
    material::MaterialSet,
    morph::MorphSet,
    rigid_body::{Joint, RigidBodySet},
    vertex::VertexSet,
    Bone, BoneIndex, MaterialIndex, Morph, MorphIndex, NanoemLabel, NanoemMaterial, NanoemModel,
    NanoemMorph, NanoemSoftBody, NanoemTexture, NanoemVertex, RigidBodyIndex, SoftBodyIndex,
    VertexIndex,
};

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
    materials: MaterialSet,
    vertices: VertexSet,
    bones: BoneSet,
    morphs: MorphSet,
    labels: Vec<Label>,
    rigid_bodies: RigidBodySet,
    joints: Vec<Joint>,
    soft_bodies: Vec<SoftBody>,
    active_morph: ModelMorphUsage,
    active_bone_pair: (Option<BoneIndex>, Option<BoneIndex>),
    bone_index_hash_map: HashMap<MaterialIndex, HashMap<BoneIndex, usize>>,
    /// Map from target bone to constraint containing it
    outside_parents: HashMap<BoneIndex, (String, String)>,
    pub shared_fallback_bone: Bone,
    bounding_box: BoundingBox,
    uniform_bind: UniformBind,
    vertex_buffer: wgpu::Buffer,
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

                let bytes_per_vertex = std::mem::size_of::<VertexUnit>();
                let unpadded_size = opaque.vertices.len() * bytes_per_vertex;
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
                let padding = (align - unpadded_size % align) % align;
                let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(format!("Model/{}/VertexBuffer", canonical_name).as_str()),
                    size: (unpadded_size + padding) as u64,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                log::trace!("Len(index_buffer): {}", &opaque.vertex_indices.len());
                let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(format!("Model/{}/IndexBuffer", canonical_name).as_str()),
                        contents: bytemuck::cast_slice(&opaque.vertex_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                );
                let uniform_bind = effect.get_uniform_bind(opaque.materials.len(), device);
                let materials = MaterialSet::new(
                    &opaque.materials,
                    &opaque.textures,
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
                        vertex_buffer: &vertex_buffer,
                        index_buffer: &index_buffer,
                    },
                    device,
                );
                let bones = BoneSet::new(&opaque.bones, &opaque.constraints, language_type);
                let mut vertices = VertexSet::new(
                    &opaque.vertices,
                    &opaque.vertex_indices,
                    &materials,
                );
                let morphs = MorphSet::new(&opaque.morphs, &vertices, &bones, language_type);
                let get_active_morph = |category: ModelMorphCategory| {
                    morphs
                        .iter()
                        .enumerate()
                        .find(|(_, morph)| morph.origin.category == category)
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
                let rigid_bodies = RigidBodySet::new(
                    &opaque.rigid_bodies,
                    &bones,
                    morphs.affected_bones(),
                    language_type,
                    physics_engine,
                );
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
                                if let std::collections::hash_map::Entry::Vacant(e) =
                                    index_hash.entry(bone_index)
                                {
                                    e.insert(unique_bone_index_per_material);
                                    unique_bone_index_per_material += 1;
                                }
                                references
                                    .entry(bone_index)
                                    .or_insert_with(HashSet::new)
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
                let mut stage_vertex_buffer_index = 0;
                match &skin_deformer {
                    Deformer::Wgpu(deformer) => {
                        deformer.execute(&vertex_buffer, device, queue);
                    }
                    Deformer::Software(deformer) => deformer.execute(
                        &vertices,
                        &bones,
                        &morphs,
                        edge_size,
                        &vertex_buffer,
                        device,
                        queue,
                    ),
                }
                stage_vertex_buffer_index = 1 - stage_vertex_buffer_index;

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
                    materials,
                    labels,
                    rigid_bodies,
                    joints,
                    soft_bodies,
                    active_bone_pair: (None, None),
                    active_morph,
                    outside_parents: HashMap::new(),
                    // shared_fallback_bone,
                    vertex_buffer,
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
        let is_add_blend = self.is_add_blend_enabled();
        self.materials.create_all_images(
            texture_lut,
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
                vertex_buffer: &self.vertex_buffer,
                index_buffer: &self.index_buffer,
            },
            device,
        );
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
        self.materials.update_image(
            texture_key,
            texture,
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
                vertex_buffer: &self.vertex_buffer,
                index_buffer: &self.index_buffer,
            },
            device,
        );
    }

    pub fn reset_physics_simulation(&mut self, physics_engine: &mut PhysicsEngine) {
        self.initialize_all_rigid_bodies_transform_feedback(physics_engine);
        self.rigid_bodies.synchronize_from_simulation(
            &mut self.bones,
            RigidBodyFollowBone::Perform,
            physics_engine,
        );
        self.mark_staging_vertex_buffer_dirty();
    }

    pub fn initialize_all_rigid_bodies_transform_feedback(
        &mut self,
        physics_engine: &mut PhysicsEngine,
    ) {
        for rigid_body in self.rigid_bodies.iter_mut() {
            rigid_body.initialize_transform_feedback(
                self.bones.try_get(rigid_body.origin.bone_index),
                physics_engine,
            );
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
                self.set_physics_simulation_enabled(
                    keyframe.is_physics_simulation_enabled,
                    physics_engine,
                );
                self.set_visible(visible, physics_engine);
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
                    self.reset_materials();
                    self.reset_bone_local_transform();
                    self.synchronize_morph_motion(motion, frame_index, amount);
                    self.synchronize_bone_motion(
                        motion,
                        frame_index,
                        amount,
                        timing,
                        physics_engine,
                        outside_parent_bone_map,
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
                let rigid_body = self.rigid_bodies.find_mut_by_bone(bone.handle);
                bone.synchronize_motion(motion, rigid_body, frame_index, amount, physics_engine);
            }
        }
        self.apply_bones_transform(timing, outside_parent_bone_map);
    }

    fn synchronize_morph_motion(&mut self, motion: &Motion, frame_index: u32, amount: f32) {
        if !self.states.dirty_morph {
            self.reset_all_morphs();
            for morph in self.morphs.iter_mut() {
                let name = morph.canonical_name.to_string();
                morph.synchronize_motion(motion, &name, frame_index, amount);
            }
            self.deform_all_morphs(true);
            for morph in self.morphs.iter_mut() {
                morph.dirty = false;
            }
            self.states.dirty_morph = true;
        }
    }

    pub fn synchronize_to_simulation_by_lerp(
        &self,
        physics_engine: &mut PhysicsEngine,
        amount: f32,
    ) {
        if self.is_physics_simulation_enabled() {
            self.rigid_bodies.synchronize_to_simulation_by_lerp(
                &self.bones,
                physics_engine,
                amount,
            );
        }
    }

    pub fn synchronize_from_simulation(
        &mut self,
        follow_type: RigidBodyFollowBone,
        physics_engine: &mut PhysicsEngine,
    ) {
        if self.is_physics_simulation_enabled() {
            self.rigid_bodies.synchronize_from_simulation(
                &mut self.bones,
                follow_type,
                physics_engine,
            );
        }
    }

    pub fn reset_morphs_deform_state(
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
                        if let Some(bone) = self.bones.try_get_mut(child.bone_index) {
                            let rigid_body = self.rigid_bodies.find_mut_by_bone_bound(bone.handle);
                            bone.reset_morph_transform();
                            bone.synchronize_motion(
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

    fn apply_bones_transform(
        &mut self,
        timing: SimulationTiming,
        outside_parent_bone_map: &HashMap<(String, String), Bone>,
    ) {
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

    pub fn apply_forces(&mut self, physics_engine: &mut PhysicsEngine) {
        self.rigid_bodies.apply_forces(&self.bones, physics_engine);
    }

    fn reset_bones_transform(&mut self) {
        for bone in self.bones.iter_mut() {
            bone.reset_local_transform();
            bone.reset_morph_transform();
            bone.reset_user_transform();
        }
    }

    fn reset_bone_local_transform(&mut self) {
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

    fn reset_materials(&mut self) {
        for material in self.materials.iter_mut() {
            material.reset();
        }
    }

    fn reset_all_morphs(&mut self) {
        for morph in  self.morphs.iter_mut() {
            morph.reset();
        }
    }

    fn reset_all_vertices(&mut self) {
        for vertex in self.vertices.iter_mut() {
            vertex.reset();
        }
    }

    pub fn set_physics_simulation_enabled(
        &mut self,
        value: bool,
        physics_engine: &mut PhysicsEngine,
    ) {
        if self.states.physics_simulation != value {
            self.set_all_physics_objects_enabled(value && self.states.visible, physics_engine);
            self.states.physics_simulation = value;
        }
    }

    pub fn set_shadow_map_enabled(&mut self, value: bool) {
        if self.states.enable_shadow_map != value {
            self.states.enable_shadow_map = value;
            // TODO: publish set shadow map event
        }
    }

    pub fn set_visible(&mut self, value: bool, physics_engine: &mut PhysicsEngine) {
        if self.states.visible != value {
            self.set_all_physics_objects_enabled(
                value & self.states.physics_simulation,
                physics_engine,
            );
            // TODO: enable effect
            self.states.visible = value;
            // TODO: publish set visible event
        }
    }

    pub fn set_all_physics_objects_enabled(
        &mut self,
        value: bool,
        physics_engine: &mut PhysicsEngine,
    ) {
        if value {
            for soft_body in &mut self.soft_bodies {
                soft_body.enable();
            }
            for rigid_body in self.rigid_bodies.iter_mut() {
                rigid_body.enable(physics_engine);
            }
            for joint in &mut self.joints {
                joint.enable();
            }
        } else {
            for soft_body in &mut self.soft_bodies {
                soft_body.disable();
            }
            for rigid_body in self.rigid_bodies.iter_mut() {
                rigid_body.disable(physics_engine);
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
                    if weight > 0f32 && !children.is_empty() {
                        let target_idx = (((children.len() + 1) as f32 * weight) as usize - 1)
                            .clamp(0, children.len() - 1);
                        let child = &children[target_idx];
                        let child_weight = child.weight;
                        if let Some(morph) = usize::try_from(child.morph_index)
                            .ok()
                            .and_then(|idx| self.morphs.get_mut(idx))
                        {
                            morph.set_weight(child_weight);
                        }
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
                            for material in self.materials.iter_mut() {
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
            if !check_dirty || morph.dirty {
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

                    nanoem::model::ModelMorphType::Flip(_children) => {}
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
                                for material in self.materials.iter_mut() {
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
        self.apply_bones_transform(SimulationTiming::Before, outside_parent_bone_map);
        if physics_engine.simulation_mode == SimulationMode::EnableAnytime {
            self.rigid_bodies.apply_forces(&self.bones, physics_engine);
            physics_engine.step(physics_simulation_time_step, |physics_engine, amount| {
                self.rigid_bodies.synchronize_to_simulation_by_lerp(
                    &self.bones,
                    physics_engine,
                    amount,
                )
            });
            self.rigid_bodies.synchronize_from_simulation(
                &mut self.bones,
                RigidBodyFollowBone::Skip,
                physics_engine,
            );
        }
        self.apply_bones_transform(SimulationTiming::After, outside_parent_bone_map);
        self.mark_staging_vertex_buffer_dirty();
        // TODO: handle owned camera
    }

    pub fn vertices(&self) -> &VertexSet {
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

    pub fn morphs(&self) -> &MorphSet {
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
        self.morphs.find(name)
    }

    pub fn find_morph_mut(&mut self, name: &str) -> Option<&mut Morph> {
        self.morphs.find_mut(name)
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
        *initial
    }

    pub fn contains_bone(&self, name: &str) -> bool {
        self.bones.find(name).is_some()
    }

    pub fn contains_morph(&self, name: &str) -> bool {
        self.morphs.contains(name)
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
                    deformer.execute(&self.vertex_buffer, device, queue);
                }
                Deformer::Software(deformer) => {
                    deformer.execute_model(self, camera, &self.vertex_buffer, device, queue)
                }
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

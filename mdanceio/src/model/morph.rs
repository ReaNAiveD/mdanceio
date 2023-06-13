use std::collections::{HashMap, HashSet};

use cgmath::{AbsDiffEq, Vector3, Zero};
use nanoem::model::ModelMorphCategory;

use crate::{motion::Motion, physics_engine::PhysicsEngine, utils::f128_to_vec3};

use super::{
    bone::BoneSet, material::MaterialSet, rigid_body::RigidBodySet, vertex::VertexSet, BoneIndex,
    MorphIndex, NanoemMorph,
};

pub struct Morph {
    pub name: String,
    pub canonical_name: String,
    pub handle: MorphIndex,
    pub weight: f32,
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
            handle: morph.base.index,
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

    pub fn synchronize_motion(&mut self, motion: &Motion, frame_index: u32, amount: f32) {
        let weight = motion.find_morph_weight(&self.canonical_name, frame_index, amount);
        self.set_weight(weight);
    }
}

#[derive(Debug, Clone, Copy)]
struct MorphUsage {
    pub eyebrow: Option<MorphIndex>,
    pub eye: Option<MorphIndex>,
    pub lip: Option<MorphIndex>,
    pub other: Option<MorphIndex>,
}

pub struct MorphSet {
    morphs: Vec<Morph>,
    morphs_by_name: HashMap<String, MorphIndex>,
    pub affected_bones: HashSet<BoneIndex>,
    active_morph: MorphUsage,
}

impl MorphSet {
    pub fn new(
        morphs: &[NanoemMorph],
        vertices: &VertexSet,
        bones: &BoneSet,
        language_type: nanoem::common::LanguageType,
    ) -> Self {
        let morphs = morphs
            .iter()
            .map(|morph| Morph::from_nanoem(morph, language_type))
            .collect::<Vec<_>>();
        let mut morphs_by_name = HashMap::new();
        let mut affected_bones: HashSet<BoneIndex> = HashSet::new();
        for morph in &morphs {
            if let nanoem::model::ModelMorphType::Vertex(morph_vertices) = &morph.origin.typ {
                for morph_vertex in morph_vertices {
                    if let Some(vertex) = vertices.try_get(morph_vertex.vertex_index) {
                        for bone_index in vertex.origin.bone_indices {
                            if let Some(bone) = bones.try_get(bone_index) {
                                affected_bones.insert(bone.handle);
                            }
                        }
                    }
                }
            }
            for language in nanoem::common::LanguageType::all() {
                morphs_by_name.insert(
                    morph.origin.get_name(*language).to_owned(),
                    morph.origin.base.index,
                );
            }
        }
        let get_active_morph = |category: ModelMorphCategory| {
            morphs
                .iter()
                .enumerate()
                .find(|(_, morph)| morph.origin.category == category)
                .map(|(idx, _)| idx)
        };
        let active_morph = MorphUsage {
            eyebrow: get_active_morph(ModelMorphCategory::Eyebrow),
            eye: get_active_morph(ModelMorphCategory::Eye),
            lip: get_active_morph(ModelMorphCategory::Lip),
            other: get_active_morph(ModelMorphCategory::Other),
        };
        Self {
            morphs,
            morphs_by_name,
            affected_bones,
            active_morph,
        }
    }

    pub fn get(&self, idx: MorphIndex) -> Option<&Morph> {
        self.morphs.get(idx)
    }

    pub fn get_mut(&mut self, idx: MorphIndex) -> Option<&mut Morph> {
        self.morphs.get_mut(idx)
    }

    pub fn find(&self, name: &str) -> Option<&Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|idx| self.morphs.get(*idx))
    }

    pub fn find_mut(&mut self, name: &str) -> Option<&mut Morph> {
        self.morphs_by_name
            .get(name)
            .and_then(|idx| self.morphs.get_mut(*idx))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.morphs_by_name.contains_key(name)
    }

    pub fn try_get(&self, idx: i32) -> Option<&Morph> {
        usize::try_from(idx).ok().and_then(|idx| self.get(idx))
    }

    pub fn len(&self) -> usize {
        self.morphs.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Morph> {
        self.morphs.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Morph> {
        self.morphs.iter_mut()
    }
}

impl MorphSet {
    pub fn reset(&mut self) {
        for morph in self.morphs.iter_mut() {
            morph.reset();
        }
    }

    pub fn deform_all(
        &mut self,
        check_dirty: bool,
        materials: &mut MaterialSet,
        vertices: &mut VertexSet,
        bones: &mut BoneSet,
        rigid_bodies: &mut RigidBodySet,
    ) {
        for morph_idx in 0..self.morphs.len() {
            self.pre_deform(morph_idx, materials);
        }
        for morph_idx in 0..self.morphs.len() {
            self.deform(
                morph_idx,
                check_dirty,
                materials,
                vertices,
                bones,
                rigid_bodies,
            );
        }
    }

    fn pre_deform(&mut self, morph_idx: MorphIndex, materials: &mut MaterialSet) {
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
                                self.pre_deform(target_morph_idx.unwrap(), materials);
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
                            .and_then(|idx| materials.get_mut(idx))
                        {
                            material.reset_deform();
                        } else {
                            for material in materials.iter_mut() {
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

    fn deform(
        &mut self,
        morph_idx: MorphIndex,
        check_dirty: bool,
        materials: &mut MaterialSet,
        vertices: &mut VertexSet,
        bones: &mut BoneSet,
        rigid_bodies: &mut RigidBodySet,
    ) {
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
                                    self.deform(
                                        target_morph_idx.unwrap(),
                                        false,
                                        materials,
                                        vertices,
                                        bones,
                                        rigid_bodies,
                                    );
                                }
                            }
                        }
                    }

                    nanoem::model::ModelMorphType::Flip(_children) => {}
                    nanoem::model::ModelMorphType::Impulse(children) => {
                        for child in children {
                            if let Some(rigid_body) = usize::try_from(child.rigid_body_index)
                                .ok()
                                .and_then(|idx| rigid_bodies.get_mut(idx))
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
                                .and_then(|idx| materials.get_mut(idx))
                            {
                                material.deform(child, weight);
                            } else {
                                for material in materials.iter_mut() {
                                    material.deform(child, weight);
                                }
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Bone(children) => {
                        for child in children {
                            if let Some(bone) = usize::try_from(child.bone_index)
                                .ok()
                                .and_then(|idx| bones.get_mut(idx))
                            {
                                bone.update_local_morph_transform(child, weight);
                            }
                        }
                    }
                    nanoem::model::ModelMorphType::Vertex(children) => {
                        for child in children {
                            if let Some(vertex) = usize::try_from(child.vertex_index)
                                .ok()
                                .and_then(|idx| vertices.get_mut(idx))
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
                                .and_then(|idx| vertices.get_mut(idx))
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

    pub fn reset_deform_state(
        &mut self,
        motion: &Motion,
        frame_index: u32,
        materials: &mut MaterialSet,
        bones: &mut BoneSet,
        rigid_bodies: &mut RigidBodySet,
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
                        if let Some(bone) = bones.try_get_mut(child.bone_index) {
                            let rigid_body: Option<&mut super::RigidBody> = rigid_bodies.find_mut_by_bone_bound(bone.handle);
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
                                target_morph.synchronize_motion(motion, frame_index, 0f32);
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
                                target_morph.synchronize_motion(motion, frame_index, 0f32);
                            }
                        }
                    }
                }
                nanoem::model::ModelMorphType::Material(children) => {
                    for child in children {
                        if let Some(target_material) = usize::try_from(child.material_index)
                            .ok()
                            .and_then(|idx| materials.get_mut(idx))
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
}

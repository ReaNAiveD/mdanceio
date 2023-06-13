use std::collections::{HashMap, HashSet};

use crate::motion::Motion;

use super::{bone::BoneSet, vertex::VertexSet, BoneIndex, MorphIndex, NanoemMorph};

pub struct Morph {
    pub name: String,
    pub canonical_name: String,
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
        let weight = motion.find_morph_weight(name, frame_index, amount);
        self.set_weight(weight);
    }
}

pub struct MorphSet {
    morphs: Vec<Morph>,
    morphs_by_name: HashMap<String, MorphIndex>,
    affected_bones: HashSet<BoneIndex>,
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
            .map(|morph| Morph::from_nanoem(&morph, language_type))
            .collect::<Vec<_>>();
        let mut morphs_by_name = HashMap::new();
        let mut affected_bones: HashSet<BoneIndex> = HashSet::new();
        for (_, morph) in morphs.iter().enumerate() {
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
        Self {
            morphs,
            morphs_by_name,
            affected_bones,
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

    pub fn affected_bones(&self) -> &HashSet<BoneIndex> {
        &self.affected_bones
    }
}

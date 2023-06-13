use cgmath::{ElementWise, Vector4, Zero};

use crate::utils::f128_to_vec4;

use super::{bone::BoneSet, BoneIndex, MaterialIndex, NanoemVertex, SoftBodyIndex, VertexIndex, material::MaterialSet};

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

pub struct VertexSet {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl VertexSet {
    pub fn new(
        vertices: &[NanoemVertex],
        vertex_indices: &[u32],
        materials: &MaterialSet,
    ) -> Self {
        let mut vertices = vertices.iter().map(Vertex::from_nanoem).collect::<Vec<_>>();
        let indices = vertex_indices.to_vec();
        let mut index_offset = 0;
        for material in materials.iter() {
            let num_indices = material.origin.num_vertex_indices;
            for vertex_index in indices.iter().skip(index_offset).take(num_indices) {
                if let Some(vertex) = vertices.get_mut(*vertex_index as usize) {
                    vertex.set_material(material.origin.base.index);
                }
            }
            index_offset += num_indices;
        }

        Self { vertices, indices }
    }

    pub fn get(&self, idx: VertexIndex) -> Option<&Vertex> {
        self.vertices.get(idx)
    }

    pub fn get_mut(&mut self, idx: VertexIndex) -> Option<&mut Vertex> {
        self.vertices.get_mut(idx)
    }

    pub fn try_get(&self, idx: i32) -> Option<&Vertex> {
        usize::try_from(idx).ok().and_then(|idx| self.get(idx))
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vertex> {
        self.vertices.iter_mut()
    }
}

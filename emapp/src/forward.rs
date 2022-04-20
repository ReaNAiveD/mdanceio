use cgmath::{Vector3, Vector4};

pub const MAX_COLOR_ATTACHMENTS: usize = 4;

#[repr(C, align(16))]
pub struct LineVertexUnit {
    position: Vector3<f32>,
    color: Vector4<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertexUnit {
    position: [f32; 4],
    texcoord: [f32; 4],
}

impl QuadVertexUnit {
    pub fn generate_quad_tri_strip() -> [QuadVertexUnit; 4] {
        Self::_generate_quad_tri_strip(-1f32, 1f32, 1f32, -1f32)
    }

    fn _generate_quad_tri_strip(minx: f32, miny: f32, maxx: f32, maxy: f32) -> [QuadVertexUnit; 4] {
        let (minu, minv, maxu, maxv) = (0.0f32, 0.0f32, 1.0f32, 1.0f32);
        [
            QuadVertexUnit {
                position: [minx, miny, 0f32, 0f32],
                texcoord: [minu, minv, 0f32, 0f32],
            },
            QuadVertexUnit {
                position: [maxx, miny, 0f32, 0f32],
                texcoord: [maxu, minv, 0f32, 0f32],
            },
            QuadVertexUnit {
                position: [minx, maxy, 0f32, 0f32],
                texcoord: [minu, maxv, 0f32, 0f32],
            },
            QuadVertexUnit {
                position: [maxx, maxy, 0f32, 0f32],
                texcoord: [maxu, maxv, 0f32, 0f32],
            },
        ]
    }
}

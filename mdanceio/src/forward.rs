#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertexUnit {
    pub position: [f32; 3],
    pub color: [u8; 4],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertexUnit {
    pub position: [f32; 4],
    pub texcoord: [f32; 4],
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
                position: [minx, maxy, 0f32, 0f32],
                texcoord: [minu, maxv, 0f32, 0f32],
            },
            QuadVertexUnit {
                position: [maxx, miny, 0f32, 0f32],
                texcoord: [maxu, minv, 0f32, 0f32],
            },
            QuadVertexUnit {
                position: [maxx, maxy, 0f32, 0f32],
                texcoord: [maxu, maxv, 0f32, 0f32],
            },
        ]
    }
}

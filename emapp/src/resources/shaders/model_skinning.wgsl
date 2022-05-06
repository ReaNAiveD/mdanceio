struct VertexUnit {
    position: vec4<f32>,
    normal: vec4<f32>,
    texcoord: vec4<f32>,
    edge: vec4<f32>,
    uva: array<vec4<f32>, 4>,
    weights: vec4<f32>,
    indices: vec4<f32>,
    info: vec4<f32>,
}

struct SdefUnit {
    c: vec4<f32>,
    r0: vec4<f32>,
    r1: vec4<f32>,
}

var<uniform> c_arg: vec4<u32>;
var<storage, read> u_matrices_buffer: array<mat4x4<f32>,>;
var<storage, read> u_morph_weight_buffer: array<f32>;
var<storage, read> u_vertex_buffer: array<VertexUnit>;
var<storage, read> u_sdef_buffer: array<SdefUnit>;
var<storage, read> u_vertex_position_deltas_buffer: array<vec4<f32>,>;
var<storage, read_write> o_vertex_buffer: array<VertexUnit>;

fn perform_skinning(sdef: SdefUnit, vertex_position_delta: vec3<f32>, unit: VertexUnit) {
    
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= c_arg.x) {
        return;
    }
    let unit = u_vertex_buffer[index];
    let sdef = u_sdef_buffer[index];
    var vertex_position_delta = vec3<f32>(0.0, 0.0, 0.0);
    let num_morph_deltas = c_arg.y;

    var i : u32 = 0u;
    loop {
        if (i >= num_morph_deltas) {
            break;
        }
        let offset = index * num_morph_deltas + i;
        let value = u_vertex_position_deltas_buffer[offset];
        let morph_index = value.w;
        let weight = u_morph_weight_buffer[u32(morph_index)];
        if (weight != 0.0) {
            vertex_position_delta += value.xyz * weight;
        }

        continue {
            i = i + 1u;
        }
    }
}

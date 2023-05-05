fn shrink_matrix(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

fn to_quaternion(m: mat4x4<f32>) -> vec4<f32> {
    var q: vec4<f32>;
    let x = m[0][0] - m[1][1] - m[2][2];
    let y = m[1][1] - m[0][0] - m[2][2];
    let z = m[2][2] - m[0][0] - m[1][1];
    let w = m[0][0] + m[1][1] + m[2][2];
    var biggest_value = w;
    var biggest_index = 0;
    if (x > biggest_value) {
        biggest_value = x;
        biggest_index = 1;
    }
    if (y > biggest_value) {
        biggest_value = y;
        biggest_index = 2;
    }
    if (z > biggest_value) {
        biggest_value = z;
        biggest_index = 3;
    }
    let biggest = sqrt(biggest_value + 1.0) * 0.5;
    let mult = 0.25 / biggest;
    let bi = biggest_index;
    switch bi {
        case 0: {
            q.x = (m[1][2] - m[2][1]) * mult;
            q.y = (m[2][0] - m[0][2]) * mult;
            q.z = (m[0][1] - m[1][0]) * mult;
            q.w = biggest;
        }
        case 1: {
            q.x = biggest;
            q.y = (m[0][1] + m[1][0]) * mult;
            q.z = (m[2][0] + m[0][2]) * mult;
            q.w = (m[1][2] - m[2][1]) * mult;
        }
        case 2: {
            q.x = (m[0][1] + m[1][0]) * mult;
            q.y = biggest;
            q.z = (m[1][2] + m[2][1]) * mult;
            q.w = (m[2][0] - m[0][2]) * mult;
        }
        case 3: {
            q.x = (m[2][0] + m[0][2]) * mult;
            q.y = (m[1][2] + m[2][1]) * mult;
            q.z = biggest;
            q.w = (m[0][1] - m[1][0]) * mult;
        }
        default: {
            q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    return q;
}

fn to_matrix(q: vec4<f32>) -> mat4x4<f32> {
    var m: mat4x4<f32>;
    m[0] = vec4<f32>(
        1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
        2.0 * q.x * q.y + 2.0 * q.w * q.z,
        2.0 * q.x * q.z - 2.0 * q.w * q.y,
        0.0
    );
    m[1] = vec4<f32>(
        2.0 * q.x * q.y - 2.0 * q.w * q.z,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
        2.0 * q.y * q.z + 2.0 * q.w * q.x,
        0.0
    );
    m[2] = vec4<f32>(
        2.0 * q.x * q.z + 2.0 * q.w * q.y,
        2.0 * q.y * q.z - 2.0 * q.w * q.x,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y,
        0.0
    );
    m[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return m;
}

fn slerp(x: vec4<f32>, y: vec4<f32>, a: f32) -> vec4<f32> {
    var z = y;
    var theta = dot(x, y);
    if (theta < 0.0) {
        z = -y;
        theta = -theta;
    }
    var result: vec4<f32>;
    if (theta >= 1.0) {
        result = mix(x, z, a);
    } else {
        let angle = acos(theta);
        result = (sin((1.0 - a) * angle) * x + sin(a * angle) * z) / sin(angle);
    }
    return result;
}

struct VertexUnit {
    position: vec4<f32>,
    normal: vec4<f32>,
    texcoord: vec4<f32>,
    edge: vec4<f32>,
    uva: array<vec4<f32>, 4>,
    weights: vec4<f32>,
    indices: vec4<f32>,
    info: vec4<f32>,
};

struct SdefUnit {
    c: vec4<f32>,
    r0: vec4<f32>,
    r1: vec4<f32>,
};

struct SkinningResult {
    position: vec4<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> c_arg: vec4<u32>;
@group(0) @binding(1)
var<storage, read> u_matrices_buffer: array<mat4x4<f32>>;
@group(0) @binding(2)
var<storage, read> u_morph_weight_buffer: array<f32>;
@group(0) @binding(3)
var<storage, read> u_vertex_buffer: array<VertexUnit>;
@group(0) @binding(4)
var<storage, read> u_sdef_buffer: array<SdefUnit>;
@group(0) @binding(5)
var<storage, read> u_vertex_position_deltas_buffer: array<vec4<f32>>;
@group(0) @binding(6)
var<storage, read_write> o_vertex_buffer: array<VertexUnit>;

// model_skinning_tf use a different make_matrix
fn make_matrix(index: f32) -> mat4x4<f32> {
    return u_matrices_buffer[u32(index)];
}

fn perform_skinning_bdef1(unit: VertexUnit, vertex_position_delta: vec3<f32>) -> SkinningResult {
    let m0 = make_matrix(unit.indices.x);
    var out: SkinningResult;
    out.position = m0 * vec4<f32>(unit.position.xyz + vertex_position_delta, 1.0);
    out.normal = shrink_matrix(m0) * unit.normal.xyz;
    return out;
}

fn perform_skinning_bdef2(unit: VertexUnit, vertex_position_delta: vec3<f32>) -> SkinningResult {
    let weight = unit.weights.x;
    var out: SkinningResult;
    out.position = vec4<f32>(unit.position.xyz + vertex_position_delta, 1.0);
    out.normal = unit.normal.xyz;
    if (weight == 0.0) {
        let m1 = make_matrix(unit.indices.y);
        out.position = m1 * out.position;
        out.normal = shrink_matrix(m1) * out.normal;
    } else if (weight == 1.0) {
        let m0 = make_matrix(unit.indices.x);
        out.position = m0 * out.position;
        out.normal = shrink_matrix(m0) * out.normal;
    } else {
        let m0 = make_matrix(unit.indices.x);
        let m1 = make_matrix(unit.indices.y);
        out.position = mix(m1 * out.position, m0 * out.position, weight);
        out.normal = mix(shrink_matrix(m1) * out.normal, shrink_matrix(m0) * out.normal, weight);
    }
    return out;
}

fn perform_skinning_bdef4(unit: VertexUnit, vertex_position_delta: vec3<f32>) -> SkinningResult {
    let weights = unit.weights;
    let indices = unit.indices;
    let m0 = make_matrix(indices.x);
    let m1 = make_matrix(indices.y);
    let m2 = make_matrix(indices.z);
    let m3 = make_matrix(indices.w);
    var out: SkinningResult;
    out.position = vec4<f32>(unit.position.xyz + vertex_position_delta, 1.0);
    out.normal = unit.normal.xyz;
    out.position = m0 * out.position * weights.xxxx + m1 * out.position * weights.yyyy + m2 * out.position * weights.zzzz + m3 * out.position * weights.wwww;
    out.normal = shrink_matrix(m0) * out.normal * weights.xxx + shrink_matrix(m1) * out.normal * weights.yyy + shrink_matrix(m2) * out.normal * weights.zzz + shrink_matrix(m3) * out.normal * weights.www;
    return out;
}

fn perform_skinning_sdef(unit: VertexUnit, sdef: SdefUnit, vertex_position_delta: vec3<f32>) -> SkinningResult {
    let weights = unit.weights.xy;
    let indices = unit.indices.xy;
    let m0 = make_matrix(indices.x);
    let m1 = make_matrix(indices.y);
    let sdef_c = sdef.c.xyz;
    let sdef_r0 = sdef.r0.xyz;
    let sdef_r1 = sdef.r1.xyz;
    let sdef_i = sdef_r0 * weights.xxx + sdef_r1 * weights.yyy;
    let sdef_r0_n = sdef_c + sdef_r0 - sdef_i;
    let sdef_r1_n = sdef_c + sdef_r1 - sdef_i;
    let r0 = (m0 * vec4<f32>(sdef_r0_n, 1.0)).xyz;
    let r1 = (m1 * vec4<f32>(sdef_r1_n, 1.0)).xyz;
    let c0 = (m0 * vec4<f32>(sdef_c, 1.0)).xyz;
    let c1 = (m1 * vec4<f32>(sdef_c, 1.0)).xyz;
    let delta = (r0 + c0 - sdef_c) * weights.xxx + (r1 + c1 - sdef_c) * weights.yyy;
    let t = (sdef_c + delta) * 0.5;
    let q0 = to_quaternion(m0);
    let q1 = to_quaternion(m1);
    let p = vec4<f32>(unit.position.xyz + vertex_position_delta - sdef_c, 1.0);
    let m = to_matrix(slerp(q1, q0, weights.x));
    var out: SkinningResult;
    out.position = vec4<f32>((m * p).xyz + t, unit.position.w);
    out.normal = shrink_matrix(m) * unit.normal.xyz;
    return out;
}

fn perform_skinning(sdef: SdefUnit, vertex_position_delta: vec3<f32>, unit: ptr<function, VertexUnit>) {
    let typ = u32((*unit).info.y);
    var skinning_result: SkinningResult;
    switch typ {
        // BDEF1
        case 0u: {
            skinning_result = perform_skinning_bdef1(*unit, vertex_position_delta);
        }
        // BDEF2
        case 1u: {
            skinning_result = perform_skinning_bdef2(*unit, vertex_position_delta);
        }
        // SDEF
        case 3u: {
            skinning_result = perform_skinning_sdef(*unit, sdef, vertex_position_delta);
        }
        // BDEF4 | QDEF
        case 2u, 4u: {
            skinning_result = perform_skinning_bdef4(*unit, vertex_position_delta);
        }
        default: {
            skinning_result.position = (*unit).position;
            skinning_result.normal = (*unit).normal.xyz;
        }
    }
    (*unit).position = vec4<f32>(skinning_result.position.xyz, (*unit).position.w);
    (*unit).normal = vec4<f32>(skinning_result.normal, (*unit).normal.w);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= c_arg.x) {
        return;
    }
    var unit = u_vertex_buffer[index];
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
            vertex_position_delta = vertex_position_delta + value.xyz * weight;
        }
        continuing {
            i = i + 1u;
        }
    }
    perform_skinning(sdef, vertex_position_delta, &unit);
    unit.edge = vec4<f32>(unit.position.xyz + (unit.normal.xyz * unit.info.xxx) * f32(c_arg.z), unit.edge.w);
    o_vertex_buffer[index] = unit;
}

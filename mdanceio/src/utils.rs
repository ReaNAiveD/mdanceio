use cgmath::{
    perspective, BaseFloat, BaseNum, ElementWise, InnerSpace, Matrix3, Matrix4, Quaternion,
    Rad, SquareMatrix, Vector3, Vector4,
};

pub fn f128_to_vec3(v: [f32; 4]) -> Vector3<f32> {
    Vector3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

pub fn f128_to_vec4(v: [f32; 4]) -> Vector4<f32> {
    v.into()
}

pub fn f128_to_quat(v: [f32; 4]) -> Quaternion<f32> {
    v.into()
}

pub fn quat_to_f128(v: Quaternion<f32>) -> [f32; 4] {
    v.into()
}

pub fn f32_array_to_mat4_col_major_order(v: [f32; 16]) -> Matrix4<f32> {
    Matrix4::new(
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13],
        v[14], v[15],
    )
}

pub fn f128_to_isometry(origin: [f32; 4], orientation: [f32; 4]) -> nalgebra::Isometry3<f32> {
    nalgebra::Isometry::from_parts(
        nalgebra::Translation3::new(origin[0], origin[1], origin[2]),
        nalgebra::UnitQuaternion::from_euler_angles(orientation[0], orientation[1], orientation[2]),
    )
}

pub fn mat4_truncate<S>(v: Matrix4<S>) -> Matrix3<S>
where
    S: BaseNum,
{
    Matrix3 {
        x: v.x.truncate(),
        y: v.y.truncate(),
        z: v.z.truncate(),
    }
}

pub fn infinite_perspective<S: BaseFloat, A: Into<Rad<S>>>(
    fovy: A,
    aspect: S,
    near: S,
) -> Matrix4<S> {
    let two: S = cgmath::num_traits::cast(2).unwrap();
    let mut result = perspective(fovy, aspect, near, S::infinity());
    result[2][2] = S::one();
    result[2][3] = S::one();
    result[3][2] = -two * near;
    result
}

pub fn lerp_f32(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

pub fn lerp_rad(a: Rad<f32>, b: Rad<f32>, amount: f32) -> Rad<f32> {
    Rad(a.0 + (b.0 - a.0) * amount)
}

pub fn to_na_vec3(v: Vector3<f32>) -> nalgebra::Vector3<f32> {
    nalgebra::Vector3::new(v[0], v[1], v[2])
}

pub fn to_na_mat3(v: Matrix3<f32>) -> nalgebra::Matrix3<f32> {
    nalgebra::Matrix3::new(
        v[0][0], v[1][0], v[2][0], v[0][1], v[1][1], v[2][1], v[0][2], v[1][2], v[2][2],
    )
}

pub fn to_isometry(v: Matrix4<f32>) -> nalgebra::Isometry3<f32> {
    nalgebra::Isometry {
        rotation: nalgebra::UnitQuaternion::from_matrix(&to_na_mat3(mat4_truncate(v))),
        translation: nalgebra::Translation {
            vector: nalgebra::vector![v[3][0], v[3][1], v[3][2]],
        },
    }
}

pub fn from_isometry(iso: nalgebra::Isometry3<f32>) -> Matrix4<f32> {
    from_na_mat4(iso.to_homogeneous())
}

pub fn from_na_mat4(v: nalgebra::Matrix4<f32>) -> Matrix4<f32> {
    Matrix4::new(
        v.m11, v.m21, v.m31, v.m41, v.m12, v.m22, v.m32, v.m42, v.m13, v.m23, v.m33, v.m43, v.m14,
        v.m24, v.m34, v.m44,
    )
}

#[test]
fn test_na_mat_trans() {
    let o = (1f32, 2f32, 3f32);
    let it = nalgebra::Isometry3::translation(1f32, 2f32, 3f32);
    let ct = from_isometry(it);
    let iv = nalgebra::Point3::new(o.0, o.1, o.2);
    let cv = Vector4::new(o.0, o.1, o.2, 1f32);
    let it2 = to_isometry(ct);
    let ip = it * iv;
    let cp = ct * cv;
    let ip2 = it2 * iv;
    assert_eq!(ip.x, cp.x);
    assert_eq!(ip.y, cp.y);
    assert_eq!(ip.z, cp.z);
    assert_eq!(cp.w, 1f32);
    assert_eq!(ip.x, ip2.x);
    assert_eq!(ip.y, ip2.y);
    assert_eq!(ip.z, ip2.z);
    println!("I {:?}", it * iv);
    println!("C {:?}", ct * cv);
    println!("I2 {:?}", it2 * iv);
    println!("Quaternion {:?}", f128_to_isometry([0., 0., 0., 0.], [0., 1., 0., 0.]));
}

pub trait CompareElementWise<Rhs = Self> {
    fn min_element_wise(self, other: Rhs) -> Self;
    fn max_element_wise(self, other: Rhs) -> Self;
    fn clamp_element_wise(self, min: Self, max: Self) -> Self;
}

impl<S: BaseFloat> CompareElementWise for Vector3<S> {
    fn min_element_wise(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    fn max_element_wise(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    fn clamp_element_wise(self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.min(max.x).max(min.x),
            self.y.min(max.y).max(min.y),
            self.z.min(max.z).max(min.z),
        )
    }
}

pub fn lerp_element_wise<T: ElementWise + Copy>(a: T, b: T, amount: T) -> T {
    a.add_element_wise(b.sub_element_wise(a).mul_element_wise(amount))
}

#[test]
fn test_lerp_element_wise() {
    let result = lerp_element_wise(
        Vector3::new(0f32, 1f32, 0.8f32),
        Vector3::new(1f32, 0.5f32, 0.2f32),
        Vector3::new(0.75f32, 0.6f32, 0.5f32),
    );
    assert_eq!(result, Vector3::new(0.75f32, 0.7f32, 0.5f32));
}

pub trait Invert {
    fn affine_invert(&self) -> Option<Self>
    where
        Self: Sized;
}

impl<S: BaseFloat> Invert for Matrix4<S> {
    fn affine_invert(&self) -> Option<Self> {
        Matrix3::new(
            self[0][0], self[0][1], self[0][2], self[1][0], self[1][1], self[1][2], self[2][0],
            self[2][1], self[2][2],
        )
        .invert()
        .map(|inv| Matrix4 {
            x: inv[0].extend(S::zero()),
            y: inv[1].extend(S::zero()),
            z: inv[2].extend(S::zero()),
            w: (-(inv * Vector3::new(self[3][0], self[3][1], self[3][2]))).extend(S::one()),
        })
    }
}

pub fn project_no<S, U>(
    obj: &Vector3<S>,
    model: &Matrix4<S>,
    proj: &Matrix4<S>,
    viewport: &Vector4<U>,
) -> Vector3<S>
where
    S: BaseFloat + Copy,
    U: Into<S> + Copy,
{
    let mut tmp = obj.extend(S::one());
    tmp = model * tmp;
    tmp = proj * tmp;

    tmp = tmp / tmp.w;
    let p_5 = S::one() / (S::one() + S::one());
    tmp = tmp * p_5 + Vector4::new(p_5, p_5, p_5, p_5);
    tmp.x = tmp.x * viewport.z.into() + viewport.x.into();
    tmp.y = tmp.y * viewport.w.into() + viewport.y.into();

    tmp.truncate()
}

// TODO: if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT, do un_project_zo
pub fn project<S, U>(
    obj: &Vector3<S>,
    model: &Matrix4<S>,
    proj: &Matrix4<S>,
    viewport: &Vector4<U>,
) -> Vector3<S>
where
    S: BaseFloat + Copy,
    U: Into<S> + Copy,
{
    project_no(obj, model, proj, viewport)
}

pub fn un_project_no<S, U>(
    win: &Vector3<S>,
    model: &Matrix4<S>,
    proj: &Matrix4<S>,
    viewport: &Vector4<U>,
) -> Option<Vector3<S>>
where
    S: BaseFloat + Copy,
    U: Into<S> + Copy,
{
    (proj * model).invert().map(|inv| {
        let mut tmp = win.extend(S::one());
        tmp.x = (tmp.x - viewport[0].into()) / viewport[2].into();
        tmp.y = (tmp.y - viewport[1].into()) / viewport[3].into();
        tmp = tmp * (S::one() + S::one()) - Vector4::new(S::one(), S::one(), S::one(), S::one());
        let mut obj = inv * tmp;
        obj = obj / obj.w;
        obj.truncate()
    })
}

// TODO: if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT, do un_project_zo
pub fn un_project<S, U>(
    win: &Vector3<S>,
    model: &Matrix4<S>,
    proj: &Matrix4<S>,
    viewport: &Vector4<U>,
) -> Option<Vector3<S>>
where
    S: BaseFloat + Copy,
    U: Into<S> + Copy,
{
    un_project_no(win, model, proj, viewport)
}

pub fn intersect_ray_plane<S>(
    orig: &Vector3<S>,
    dir: &Vector3<S>,
    plane_orig: &Vector3<S>,
    plane_normal: &Vector3<S>,
) -> Option<S>
where
    S: BaseFloat + Copy,
{
    let d = dir.dot(*plane_normal);
    let epsilon = S::epsilon();
    if d.abs() > epsilon {
        let tmp_intersection_distance = (plane_orig - orig).dot(*plane_normal) / d;
        if tmp_intersection_distance > S::zero() {
            return Some(tmp_intersection_distance);
        }
    }
    None
}

#[test]
fn test_m4_affine_invert() {
    let a = Matrix4::new(
        2.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 3.0, 1.0,
    );
    let inv = a.affine_invert().unwrap();
    assert_eq!(inv * a, Matrix4::identity());
}

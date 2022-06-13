use cgmath::{
    BaseFloat, BaseNum, ElementWise, InnerSpace, Matrix3, Matrix4, SquareMatrix, Vector3, Vector4, Quaternion,
};
use nanoem::common::F128;

pub fn f128_to_vec3(v: F128) -> Vector3<f32> {
    Vector3 {
        x: v.0[0],
        y: v.0[1],
        z: v.0[2],
    }
}

pub fn f128_to_vec4(v: F128) -> Vector4<f32> {
    v.0.into()
}

pub fn f128_to_quat(v: F128) -> Quaternion<f32> {
    v.0.into()
}

pub fn mat4_truncate<S>(v: Matrix4<S>) -> Matrix3<S> where S: BaseNum {
    Matrix3 {
        x: v.x.truncate(),
        y: v.y.truncate(),
        z: v.z.truncate(),
    }
}

pub fn lerp_f32(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

pub struct EnumUtils {}

macro_rules! set_enabled {
    ($typ: ty, $fn_name: ident) => {
        pub fn $fn_name(flags: &mut $typ, mask: $typ, enabled: bool) {
            *flags = if enabled {
                *flags | mask
            } else {
                *flags & (!mask)
            }
        }
    };
}

macro_rules! is_enabled {
    ($typ: ty, $fn_name: ident) => {
        pub fn $fn_name(flags: &$typ, mask: $typ) -> bool {
            (*flags & mask) != (0 as $typ)
        }
    };
}

impl EnumUtils {
    set_enabled!(u32, set_enabled_u32);
    is_enabled!(u32, is_enabled_u32);
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

pub trait Invert {
    fn affine_invert(&self) -> Option<Self>
    where
        Self: Sized;
}

impl<S: BaseFloat> Invert for Matrix4<S> {
    fn affine_invert(&self) -> Option<Self> {
        if let Some(inv) = Matrix3::new(
            self[0][0], self[0][1], self[0][2], self[1][0], self[1][1], self[1][2], self[2][0],
            self[2][1], self[2][2],
        )
        .invert()
        {
            Some(Matrix4 {
                x: inv[0].extend(S::zero()),
                y: inv[1].extend(S::zero()),
                z: inv[2].extend(S::zero()),
                w: (-(inv * Vector3::new(self[3][0], self[3][1], self[3][2]))).extend(S::one()),
            })
        } else {
            None
        }
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
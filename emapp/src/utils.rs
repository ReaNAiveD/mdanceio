use cgmath::{BaseFloat, Vector3, Matrix3, Matrix4, ElementWise, BaseNum, SquareMatrix};

pub trait Invert {
    fn affine_invert(&self) -> Option<Self> where Self: Sized;
}

impl<S: BaseFloat> Invert for Matrix4<S> {
    fn affine_invert(&self) -> Option<Self> {
        if let Some(inv) = Matrix3::new(
            self[0][0], self[0][1], self[0][2], 
            self[1][0], self[1][1], self[1][2], 
            self[2][0], self[2][1], self[2][2],
        ).invert() {
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

#[test]
fn test_m4_affine_invert() {
    let a = Matrix4::new(
        2.0, 1.0, 4.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 3.0, 0.0,
        1.0, 2.0, 3.0, 1.0,
    );
    let inv = a.affine_invert().unwrap();
    assert_eq!(inv * a, Matrix4::identity());
}

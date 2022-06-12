use std::{cell::RefCell, collections::HashMap, rc::Rc};

use cgmath::{Vector1, Vector2, Vector4, Quaternion, BaseNum, VectorSpace};
use nanoem::{
    common::Status,
    motion::{
        MotionAccessoryKeyframe, MotionBoneKeyframe, MotionCameraKeyframe, MotionFormatType,
        MotionLightKeyframe, MotionModelKeyframe,
    },
};

use crate::{
    bezier_curve::BezierCurve, model::Model, motion_keyframe_selection::MotionKeyframeSelection,
    project::Project, uri::Uri,
};

pub type NanoemMotion = nanoem::motion::Motion;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CurveCacheKey {
    next: [u8; 4],
    interval: u32,
}

pub struct Motion {
    selection: Box<dyn MotionKeyframeSelection>,
    pub opaque: NanoemMotion,
    bezier_curves_data: RefCell<HashMap<CurveCacheKey, Box<BezierCurve>>>,
    keyframe_bezier_curves: RefCell<HashMap<Rc<RefCell<MotionBoneKeyframe>>, BezierCurve>>,
    annotations: HashMap<String, String>,
    file_uri: Uri,
    format_type: MotionFormatType,
    handle: u16,
    pub dirty: bool,
}

impl Motion {
    pub const NMD_FORMAT_EXTENSION: &'static str = "nmd";
    pub const VMD_FORMAT_EXTENSION: &'static str = "vmd";
    pub const CAMERA_AND_LIGHT_TARGET_MODEL_NAME: &'static [u8] = &[
        0xe3, 0x82, 0xab, 0xe3, 0x83, 0xa1, 0xe3, 0x83, 0xa9, 0xe3, 0x83, 0xbb, 0xe7, 0x85, 0xa7,
        0xe6, 0x98, 0x8e, 0,
    ];
    pub const MAX_KEYFRAME_INDEX: u32 = u32::MAX;

    pub fn loadable_extensions() -> Vec<&'static str> {
        vec![Self::NMD_FORMAT_EXTENSION, Self::VMD_FORMAT_EXTENSION]
    }

    pub fn is_loadable_extension(extension: &str) -> bool {
        Self::loadable_extensions()
            .iter()
            .any(|ext| ext.to_lowercase().eq(extension))
    }

    pub fn uri_has_loadable_extension(uri: &Uri) -> bool {
        if let Some(ext) = uri.absolute_path_extension() {
            Self::is_loadable_extension(ext)
        } else {
            false
        }
    }

    pub fn add_frame_index_delta(value: i32, frame_index: u32) -> Option<u32> {
        let mut result = false;
        if value > 0 {
            if frame_index <= Self::MAX_KEYFRAME_INDEX - value as u32 {
                return Some(frame_index + (value as u32));
            }
        } else if value < 0 {
            if frame_index >= value.abs() as u32 {
                return Some(frame_index - (value.abs() as u32));
            }
        }
        None
    }

    pub fn subtract_frame_index_delta(value: i32, frame_index: u32) -> Option<u32> {
        Self::add_frame_index_delta(-value, frame_index)
    }

    pub fn copy_all_accessory_keyframes(
        keyframes: &[MotionAccessoryKeyframe],
        target: &mut NanoemMotion,
        offset: i32,
    ) -> Result<(), Status> {
        for keyframe in keyframes {
            let frame_index = keyframe.frame_index_with_offset(offset);
            let mut n_keyframe = keyframe.clone();
            keyframe.copy_outside_parent(target, &mut n_keyframe);
            let _ = target.add_accessory_keyframe(n_keyframe, frame_index);
        }
        target.sort_all_keyframes();
        Ok(())
    }

    pub fn copy_all_accessory_keyframes_from_motion(
        source: &NanoemMotion,
        target: &mut NanoemMotion,
        offset: i32,
    ) -> Result<(), Status> {
        Self::copy_all_accessory_keyframes(
            source.get_all_accessory_keyframe_objects(),
            target,
            offset,
        )
    }

    pub fn copy_all_bone_keyframes(
        keyframes: &[MotionBoneKeyframe],
        parent_motion: &NanoemMotion,
        selection: &(dyn MotionKeyframeSelection),
        model: &Model,
        target: &mut NanoemMotion,
        offset: i32,
    ) -> Result<(), Status> {
        for keyframe in keyframes {
            let name = keyframe.get_name(parent_motion);
            // TODO: unfinished
        }
        Ok(())
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }

    pub fn duration(&self) -> u32 {
        self.opaque
            .get_max_frame_index()
            .min(Project::MAXIMUM_BASE_DURATION)
    }

    pub fn find_bone_keyframe(
        &self,
        name: &String,
        frame_index: u32,
    ) -> Option<&MotionBoneKeyframe> {
        self.opaque.find_bone_keyframe_object(name, frame_index)
    }

    pub fn find_model_keyframe(&self, frame_index: u32) -> Option<&MotionModelKeyframe> {
        self.opaque.find_model_keyframe_object(frame_index)
    }

    pub fn find_camera_keyframe(&self, frame_index: u32) -> Option<&MotionCameraKeyframe> {
        self.opaque.find_camera_keyframe_object(frame_index)
    }

    pub fn find_light_keyframe(&self, frame_index: u32) -> Option<&MotionLightKeyframe> {
        self.opaque.find_light_keyframe_object(frame_index)
    }

    pub fn coefficient(prev_frame_index: u32, next_frame_index: u32, frame_index: u32) -> f32 {
        let interval = next_frame_index - prev_frame_index;
        if prev_frame_index == next_frame_index {
            1f32
        } else {
            (frame_index - prev_frame_index) as f32 / (interval as f32)
        }
    }

    pub fn lerp_interpolation<T>(
        &self,
        next_interpolation: &[u8; 4],
        prev_value: &T,
        next_value: &T,
        interval: u32,
        coef: f32,
    ) -> T
    where
        T: VectorSpace<Scalar = f32>,
    {
        if KeyframeInterpolationPoint::is_linear_interpolation(next_interpolation) {
            prev_value.lerp(*next_value, coef)
        } else {
            let t2 = self.bezier_curve(next_interpolation, interval, coef);
            prev_value.lerp(*next_value, t2)
        }
    }

    pub fn slerp_interpolation(
        &self,
        next_interpolation: &[u8; 4],
        prev_value: &Quaternion<f32>,
        next_value: &Quaternion<f32>,
        interval: u32,
        coef: f32,) -> Quaternion<f32> {
            if KeyframeInterpolationPoint::is_linear_interpolation(next_interpolation) {
                prev_value.slerp(*next_value, coef)
            } else {
                let t2 = self.bezier_curve(next_interpolation, interval, coef);
                prev_value.slerp(*next_value, t2)
            }
        }

    pub fn lerp_value_interpolation(
        &self,
        next_interpolation: &[u8; 4],
        prev_value: f32,
        next_value: f32,
        interval: u32,
        coef: f32,
    ) -> f32 {
        self.lerp_interpolation(
            next_interpolation,
            &Vector1::new(prev_value),
            &Vector1::new(next_value),
            interval,
            coef,
        )
        .x
    }

    pub fn bezier_curve(&self, next: &[u8; 4], interval: u32, value: f32) -> f32 {
        let key = CurveCacheKey {
            next: next.clone(),
            interval,
        };
        let mut cache = self.bezier_curves_data.borrow_mut();
        if let Some(curve) = cache.get(&key) {
            curve.value(value)
        } else {
            let curve = Box::new(BezierCurve::create(
                &Vector2::new(next[0], next[1]),
                &Vector2::new(next[2], next[3]),
                interval,
            ));
            let r = curve.value(value);
            cache.insert(key, curve);
            r
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KeyframeInterpolationPoint {
    pub bezier_control_point: Vector4<u8>,
    pub is_linear_interpolation: bool,
}

impl Default for KeyframeInterpolationPoint {
    fn default() -> Self {
        Self {
            bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
            is_linear_interpolation: true,
        }
    }
}

impl KeyframeInterpolationPoint {
    const DEFAULT_BEZIER_CONTROL_POINT: [u8; 4] = [20, 20, 107, 107];

    pub fn is_linear_interpolation(interpolation: &[u8; 4]) -> bool {
        interpolation[0] == interpolation[1]
            && interpolation[2] == interpolation[3]
            && interpolation[0] + interpolation[2] == interpolation[1] + interpolation[3]
    }

    pub fn build(interpolation: [u8; 4]) -> Self {
        if Self::is_linear_interpolation(&interpolation) {
            Self {
                bezier_control_point: Self::DEFAULT_BEZIER_CONTROL_POINT.into(),
                is_linear_interpolation: true,
            }
        } else {
            Self {
                bezier_control_point: Vector4::from(interpolation),
                is_linear_interpolation: false,
            }
        }
    }
}

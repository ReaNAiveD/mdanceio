use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cgmath::{BaseNum, Quaternion, Vector1, Vector2, Vector4, VectorSpace};
use nanoem::{
    common::{Status, F128},
    motion::{
        MotionAccessoryKeyframe, MotionBoneKeyframe, MotionCameraKeyframe, MotionFormatType,
        MotionLightKeyframe, MotionModelKeyframe, MotionMorphKeyframe, MotionSelfShadowKeyframe,
        MotionTrackBundle,
    },
};

use crate::{
    bezier_curve::BezierCurve, error::Error, model::Model,
    motion_keyframe_selection::MotionKeyframeSelection, project::Project, uri::Uri,
};

pub type NanoemMotion = nanoem::motion::Motion;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CurveCacheKey {
    next: [u8; 4],
    interval: u32,
}

pub struct Motion {
    // selection: Box<dyn MotionKeyframeSelection>,
    pub opaque: NanoemMotion,
    bezier_curves_data: RefCell<HashMap<CurveCacheKey, Box<BezierCurve>>>,
    keyframe_bezier_curves: RefCell<HashMap<Rc<RefCell<MotionBoneKeyframe>>, BezierCurve>>,
    annotations: HashMap<String, String>,
    // file_uri: Uri,
    // format_type: MotionFormatType,
    handle: u32,
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

    pub fn new_from_bytes(bytes: &[u8], offset: u32, handle: u32) -> Result<Self, Error> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        match NanoemMotion::load_from_buffer(&mut buffer, offset) {
            Ok(motion) => Ok(Self {
                opaque: motion,
                bezier_curves_data: RefCell::new(HashMap::new()),
                keyframe_bezier_curves: RefCell::new(HashMap::new()),
                annotations: HashMap::new(),
                handle,
                dirty: false,
            }),
            Err(status) => Err(Error::from_nanoem("Cannot load the model: ", status)),
        }
    }

    pub fn empty(handle: u32) -> Self {
        Self {
            // selection: (),
            opaque: NanoemMotion::empty(),
            bezier_curves_data: RefCell::new(HashMap::new()),
            keyframe_bezier_curves: RefCell::new(HashMap::new()),
            annotations: HashMap::new(),
            // file_uri: (),
            // format_type: (),
            handle,
            dirty: false,
        }
    }

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

    pub fn find_bone_keyframe(&self, name: &str, frame_index: u32) -> Option<&MotionBoneKeyframe> {
        self.opaque.find_bone_keyframe_object(name, frame_index)
    }

    pub fn find_model_keyframe(&self, frame_index: u32) -> Option<&MotionModelKeyframe> {
        self.opaque.find_model_keyframe_object(frame_index)
    }

    pub fn find_morph_keyframe(
        &self,
        name: &str,
        frame_index: u32,
    ) -> Option<&MotionMorphKeyframe> {
        self.opaque.find_morph_keyframe_object(name, frame_index)
    }

    pub fn find_camera_keyframe(&self, frame_index: u32) -> Option<&MotionCameraKeyframe> {
        self.opaque.find_camera_keyframe_object(frame_index)
    }

    pub fn find_light_keyframe(&self, frame_index: u32) -> Option<&MotionLightKeyframe> {
        self.opaque.find_light_keyframe_object(frame_index)
    }

    pub fn find_self_shadow_keyframe(&self, frame_index: u32) -> Option<&MotionSelfShadowKeyframe> {
        self.opaque.find_self_shadow_keyframe_object(frame_index)
    }

    pub fn test_all_missing_model_objects(&self, model: &Model) -> (Vec<String>, Vec<String>) {
        let mut bones = vec![];
        let mut morphs = vec![];
        for bone_name in self.opaque.local_bone_motion_track_bundle.tracks.keys() {
            if !model.contains_bone(bone_name) {
                bones.push(bone_name.clone());
            }
        }
        for morph_name in self.opaque.local_morph_motion_track_bundle.tracks.keys() {
            if !model.contains_morph(morph_name) {
                morphs.push(morph_name.clone())
            }
        }
        (bones, morphs)
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
        coef: f32,
    ) -> Quaternion<f32> {
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

struct Merger<'a, 'b> {
    source: &'a NanoemMotion,
    overrid: bool,
    dest: &'b mut NanoemMotion,
}

impl Merger<'_, '_> {
    fn reverse_bone_keyframe(origin: &MotionBoneKeyframe) -> MotionBoneKeyframe {
        let mut result = origin.clone();
        result.translation = F128([
            -origin.translation.0[0],
            origin.translation.0[1],
            origin.translation.0[2],
            0f32,
        ]);
        result.orientation = F128([
            origin.orientation.0[0],
            -origin.orientation.0[1],
            -origin.orientation.0[2],
            origin.orientation.0[3],
        ]);
        result
    }

    fn add_bone_keyframe(
        &mut self,
        origin: &MotionBoneKeyframe,
        frame_index: u32,
        name: &String,
        reverse: bool,
        reversed_bone_name_set: &mut HashSet<String>,
    ) {
        const LEFT: &'static str = "左";
        const RIGHT: &'static str = "右";
        let (new_name, new_frame) = if reverse && name.starts_with(LEFT) {
            let new_name = name.replacen(LEFT, RIGHT, 1);
            reversed_bone_name_set.insert(new_name.clone());
            (new_name, Self::reverse_bone_keyframe(origin))
        } else if reverse && name.starts_with(RIGHT) {
            let new_name = name.replacen(RIGHT, LEFT, 1);
            reversed_bone_name_set.insert(new_name.clone());
            (new_name, Self::reverse_bone_keyframe(origin))
        } else {
            (name.clone(), origin.clone())
        };
        if self.overrid
            || self
                .dest
                .find_bone_keyframe_object(name, frame_index)
                .is_none()
        {
            let _ = self.dest.local_bone_motion_track_bundle.force_add_keyframe(
                new_frame,
                frame_index,
                &new_name,
            );
        }
    }

    pub fn merge_all_bone_keyframes(&mut self, reverse: bool) {
        let mut reversed_bone_name_set = HashSet::new();
        for (name, track) in &self.source.local_bone_motion_track_bundle.tracks {
            for (frame_idx, keyframe) in &track.keyframes {
                self.add_bone_keyframe(
                    keyframe,
                    *frame_idx,
                    name,
                    reverse,
                    &mut reversed_bone_name_set,
                );
            }
        }
    }

    pub fn merge_all_camera_keyframes(&mut self) {
        for keyframe in &self.source.camera_keyframes {
            if self.overrid {
                let _ = self
                    .dest
                    .remove_camera_keyframe_object(keyframe.base.frame_index);
            }
            let _ = self
                .dest
                .add_camera_keyframe(keyframe.clone(), keyframe.base.frame_index);
        }
    }

    pub fn merge_all_light_keyframes(&mut self) {
        for keyframe in &self.source.light_keyframes {
            if self.overrid {
                let _ = self
                    .dest
                    .remove_light_keyframe_object(keyframe.base.frame_index);
            }
            let _ = self
                .dest
                .add_light_keyframe(keyframe.clone(), keyframe.base.frame_index);
        }
    }

    pub fn merge_all_model_keyframes(&mut self) {
        for keyframe in &self.source.model_keyframes {
            if self.overrid {
                let _ = self
                    .dest
                    .remove_model_keyframe_object(keyframe.base.frame_index);
            }
            let _ = self
                .dest
                .add_model_keyframe(keyframe.clone(), keyframe.base.frame_index);
        }
    }

    pub fn merge_all_morph_keyframes(&mut self) {
        for (name, track) in &self.source.local_morph_motion_track_bundle.tracks {
            for (frame_index, keyframe) in &track.keyframes {
                if self.overrid
                    || self
                        .dest
                        .find_morph_keyframe_object(name, *frame_index)
                        .is_none()
                {
                    self.dest
                        .local_morph_motion_track_bundle
                        .force_add_keyframe(keyframe.clone(), *frame_index, name);
                }
            }
        }
    }

    pub fn merge_all_self_shadow_keyframes(&mut self) {
        for keyframe in &self.source.self_shadow_keyframes {
            if self.overrid {
                let _ = self
                    .dest
                    .remove_self_shadow_keyframe_object(keyframe.base.frame_index);
            }
            let _ = self
                .dest
                .add_self_shadow_keyframe(keyframe.clone(), keyframe.base.frame_index);
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

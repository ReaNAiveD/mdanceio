use std::{cell::RefCell, collections::HashMap, rc::Rc};

use nanoem::{motion::{MotionBoneKeyframe, MotionFormatType, MotionAccessoryKeyframe}, common::Status};

use crate::{
    bezier_curve::BezierCurve, motion_keyframe_selection::MotionKeyframeSelection, uri::Uri, project::Project, model::Model,
};

pub struct Motion {
    selection: Box<dyn MotionKeyframeSelection>,
    opaque: Rc<RefCell<nanoem::motion::Motion>>,
    bezier_curves_data: RefCell<HashMap<u64, BezierCurve>>,
    keyframe_bezier_curves: RefCell<HashMap<Rc<RefCell<MotionBoneKeyframe>>, BezierCurve>>,
    annotations: HashMap<String, String>,
    file_uri: Uri,
    format_type: MotionFormatType,
    handle: u16,
    dirty: bool,
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
        Self::loadable_extensions().iter().any(|ext| ext.to_lowercase().eq(extension))
    }

    pub fn uri_has_loadable_extension(uri: &Uri) -> bool {
        if let Some(ext) = uri.absolute_path_extension() {
            Self::is_loadable_extension(ext)
        } else {
            false
        }
    }

    pub fn add_frame_index_delta(value: i32, frame_index: u32, new_frame_index: &mut u32) -> bool {
        let mut result = false;
        if value > 0 {
            if frame_index <= Self::MAX_KEYFRAME_INDEX - value as u32 {
                *new_frame_index = frame_index + (value as u32);
                result = true;
            } 
        } else if value < 0 {
            if frame_index >= value.abs() as u32 {
                *new_frame_index = frame_index - (value.abs() as u32);
                result = true;
            }
        }
        result
    }

    pub fn subtract_frame_index_delta(value: i32, frame_index: u32, new_frame_index: &mut u32) -> bool {
        Self::add_frame_index_delta(-value, frame_index, new_frame_index)
    }

    pub fn copy_all_accessory_keyframes(keyframes: &[MotionAccessoryKeyframe], target: &mut nanoem::motion::Motion, offset: i32) -> Result<(), Status> {
        for keyframe in keyframes {
            let frame_index = keyframe.frame_index_with_offset(offset);
            let mut n_keyframe = keyframe.clone();
            keyframe.copy_outside_parent(target, &mut n_keyframe);
            let _ = target.add_accessory_keyframe(n_keyframe, frame_index);
        }
        target.sort_all_keyframes();
        Ok(())
    }

    pub fn copy_all_accessory_keyframes_from_motion(source: &nanoem::motion::Motion, target: &mut nanoem::motion::Motion, offset: i32) -> Result<(), Status> {
        Self::copy_all_accessory_keyframes(source.get_all_accessory_keyframe_objects(), target, offset)
    }

    pub fn copy_all_bone_keyframes(keyframes: &[MotionBoneKeyframe], parent_motion: &nanoem::motion::Motion, selection: &(dyn MotionKeyframeSelection), model: &Model, target: &mut nanoem::motion::Motion, offset: i32) -> Result<(), Status> {
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
        self.opaque.borrow().get_max_frame_index().min(Project::MAXIMUM_BASE_DURATION)
    }
}

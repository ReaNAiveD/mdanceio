use std::{cell::RefCell, collections::HashMap, rc::Rc};

use nanoem::{motion::{MotionBoneKeyframe, MotionFormatType, MotionAccessoryKeyframe}, common::Status};

use crate::{
    bezier_curve::BezierCurve, motion_keyframe_selection::MotionKeyframeSelection, uri::Uri,
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

    pub fn copy_all_accessory_keyframes(keyframes: &[MotionAccessoryKeyframe], motion: &nanoem::motion::Motion, offset: i32) -> Result<(), Status> {
        for keyframe in keyframes {
            keyframe.frame_index_with_offset(offset);
            // TODO: unfinished
        }
        Ok(())
    }
}

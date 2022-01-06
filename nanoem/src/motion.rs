use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashMap,
    rc::{Rc, Weak},
};

use crate::{
    common::{Buffer, MutableBuffer, Status, UserData, F128},
    utils::{compare, u8_slice_get_string, CodecType},
};

static NANOEM_MOTION_OBJECT_NOT_FOUND: i32 = -1;

#[derive(Clone, Default)]
struct StringCache(HashMap<String, i32>);

struct MotionTrack {
    id: i32,
    name: String,
    keyframes: HashMap<u32, MotionKeyframeObject>,
}

pub struct MotionTrackBundle {
    tracks_by_name: HashMap<String, MotionTrack>,
}

impl MotionTrackBundle {
    fn get_by_name(&self, name: &String) -> Option<&MotionTrack> {
        self.tracks_by_name.get(name)
    }

    fn get_mut_by_name(&mut self, name: &String) -> Option<&mut MotionTrack> {
        self.tracks_by_name.get_mut(name)
    }

    fn put_new(&mut self, id: i32, name: &String) -> Option<MotionTrack> {
        self.tracks_by_name.insert(
            name.clone(),
            MotionTrack {
                id,
                name: name.clone(),
                keyframes: HashMap::new(),
            },
        )
    }

    fn resolve_name(&self, id: i32) -> Option<&String> {
        for track in self.tracks_by_name.values() {
            if track.id == id {
                return Some(&track.name);
            }
        }
        None
    }

    fn resolve_id(&mut self, name: &String, allocated_id: &mut i32) -> (i32, Option<String>) {
        if let Some(track) = self.get_by_name(name) {
            (track.id, Some(track.name.clone()))
        } else {
            *allocated_id += 1;
            let id = *allocated_id;
            self.put_new(id, name);
            (id, None)
        }
    }

    fn add_keyframe(&mut self, keyframe: MotionKeyframeObject, frame_index: u32, name: &String) {
        if let Some(track) = self.get_mut_by_name(name) {
            track.keyframes.insert(frame_index, keyframe);
        }
    }

    fn remove_keyframe(&mut self, frame_index: u32, name: &String) {
        if let Some(track) = self.get_mut_by_name(name) {
            track.keyframes.remove(&frame_index);
        }
    }

    pub fn find_keyframes_map(&self, name: &String) -> Option<&HashMap<u32, MotionKeyframeObject>> {
        if let Some(track) = self.tracks_by_name.get(name) {
            Some(&track.keyframes)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MotionFormatType {
    Unknown = -1,
    VMD,
    NMD,
}

impl From<i32> for MotionFormatType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::VMD,
            1 => Self::NMD,
            _ => Self::Unknown,
        }
    }
}

impl Default for MotionFormatType {
    fn default() -> Self {
        Self::Unknown
    }
}

pub struct Motion {
    annotations: HashMap<String, String>,
    target_model_name: String,
    accessory_keyframes: Vec<MotionAccessoryKeyframe>,
    bone_keyframes: Vec<Rc<RefCell<MotionBoneKeyframe>>>,
    morph_keyframes: Vec<Rc<RefCell<MotionMorphKeyframe>>>,
    camera_keyframes: Vec<MotionCameraKeyframe>,
    light_keyframes: Vec<MotionLightKeyframe>,
    model_keyframes: Vec<MotionModelKeyframe>,
    self_shadow_keyframes: Vec<MotionSelfShadowKeyframe>,
    local_bone_motion_track_allocated_id: i32,
    local_bone_motion_track_bundle: MotionTrackBundle,
    local_morph_motion_track_allocated_id: i32,
    local_morph_motion_track_bundle: MotionTrackBundle,
    global_motion_track_allocated_id: i32,
    global_motion_track_bundle: MotionTrackBundle,
    typ: MotionFormatType,
    max_frame_index: u32,
    preferred_fps: f32,
    user_data: Option<Rc<RefCell<UserData>>>,
}

impl Motion {
    const VMD_SIGNATURE_SIZE: usize = 30;
    const VMD_SIGNATURE_TYPE2: &'static [u8] = b"Vocaloid Motion Data 0002";
    const VMD_SIGNATURE_TYPE1: &'static [u8] = b"Vocaloid Motion Data file";
    const VMD_TARGET_MODEL_NAME_LENGTH_V2: usize = 20;
    const VMD_TARGET_MODEL_NAME_LENGTH_V1: usize = 10;

    fn resolve_local_bone_track_id(&mut self, name: &String) -> (i32, Option<String>) {
        self.local_bone_motion_track_bundle
            .resolve_id(name, &mut self.local_bone_motion_track_allocated_id)
    }

    fn resolve_local_morph_track_id(&mut self, name: &String) -> (i32, Option<String>) {
        self.local_morph_motion_track_bundle
            .resolve_id(name, &mut self.local_morph_motion_track_allocated_id)
    }

    fn resolve_global_track_id(&mut self, name: &String) -> (i32, Option<String>) {
        self.global_motion_track_bundle
            .resolve_id(name, &mut self.global_motion_track_allocated_id)
    }

    fn set_max_frame_index(&mut self, base: &MotionKeyframeBase) {
        if base.frame_index > self.max_frame_index {
            self.max_frame_index = base.frame_index;
        }
    }

    fn parse_bone_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_bone_keyframes = buffer.read_len()?;
        if num_bone_keyframes > 0 {
            let mut cache = StringCache::default();
            self.bone_keyframes.clear();
            for i in 0..num_bone_keyframes {
                let keyframe = MotionBoneKeyframe::parse_vmd(self, buffer, &mut cache, offset)?;
                {
                    self.set_max_frame_index(&keyframe.borrow().base);
                }
                keyframe.borrow_mut().base.index = i as i32;
                self.bone_keyframes.push(keyframe);
            }
            self.bone_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.borrow().base, &b.borrow().base));
        }
        Ok(())
    }

    fn parse_morph_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_morph_keyframes = buffer.read_len()?;
        if num_morph_keyframes > 0 {
            let mut cache = StringCache::default();
            for i in 0..num_morph_keyframes {
                let mut keyframe =
                    MotionMorphKeyframe::parse_vmd(self, buffer, &mut cache, offset)?;
                {
                    self.set_max_frame_index(&keyframe.borrow().base);
                }
                keyframe.borrow_mut().base.index = i as i32;
                self.morph_keyframes.push(keyframe);
            }
            self.morph_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.borrow().base, &b.borrow().base));
        }
        Ok(())
    }

    fn parse_camera_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_camera_keyframes = buffer.read_len()?;
        if num_camera_keyframes > 0 {
            for i in 0..num_camera_keyframes {
                let mut keyframe = MotionCameraKeyframe::parse_vmd(self, buffer, offset)?;
                self.set_max_frame_index(&keyframe.base);
                keyframe.base.index = i as i32;
                self.camera_keyframes.push(keyframe);
            }
            self.camera_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.base, &b.base));
        }
        Ok(())
    }

    fn parse_light_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_light_keyframes = buffer.read_len()?;
        if num_light_keyframes > 0 {
            for i in 0..num_light_keyframes {
                let mut keyframe = MotionLightKeyframe::parse_vmd(buffer, offset)?;
                self.set_max_frame_index(&keyframe.base);
                keyframe.base.index = i as i32;
                self.light_keyframes.push(keyframe);
            }
            self.light_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.base, &b.base))
        }
        Ok(())
    }

    fn parse_self_shadow_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_self_shadow_keyframes = buffer.read_len()?;
        if num_self_shadow_keyframes > 0 {
            for i in 0..num_self_shadow_keyframes {
                let mut keyframe = MotionSelfShadowKeyframe::parse_vmd(buffer, offset)?;
                self.set_max_frame_index(&keyframe.base);
                keyframe.base.index = i as i32;
                self.self_shadow_keyframes.push(keyframe);
            }
            self.self_shadow_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.base, &b.base))
        }
        Ok(())
    }

    fn parse_model_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_model_keyframes = buffer.read_len()?;
        if num_model_keyframes > 0 {
            for i in 0..num_model_keyframes {
                let mut keyframe = MotionModelKeyframe::parse_vmd(self, buffer, offset)?;
                self.set_max_frame_index(&keyframe.base);
                keyframe.base.index = i as i32;
                self.model_keyframes.push(keyframe);
            }
            self.model_keyframes
                .sort_by(|a, b| MotionKeyframeBase::compare(&a.base, &b.base))
        }
        Ok(())
    }

    fn parse_vmd(&mut self, buffer: &mut Buffer, offset: u32) -> Result<(), Status> {
        self.parse_bone_keyframe_block_vmd(buffer, offset)?;
        self.parse_morph_keyframe_block_vmd(buffer, offset)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.parse_camera_keyframe_block_vmd(buffer, offset)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.parse_light_keyframe_block_vmd(buffer, offset)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.parse_self_shadow_keyframe_block_vmd(buffer, offset)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.parse_model_keyframe_block_vmd(buffer, offset)?;
        if buffer.is_end() {
            return Ok(());
        } else {
            return Err(Status::ErrorBufferNotEnd);
        }
    }

    fn load_from_buffer_vmd(&mut self, buffer: &mut Buffer, offset: u32) -> Result<(), Status> {
        let sig = buffer.read_buffer(Self::VMD_SIGNATURE_SIZE)?;
        if compare(
            &sig[0..Self::VMD_SIGNATURE_TYPE2.len()],
            Self::VMD_SIGNATURE_TYPE2,
        ) == Ordering::Equal
        {
            self.target_model_name =
                buffer.read_string_from_cp932(Self::VMD_TARGET_MODEL_NAME_LENGTH_V2)?;
            self.parse_vmd(buffer, offset)?;
        } else if compare(
            &sig[0..Self::VMD_SIGNATURE_TYPE1.len()],
            Self::VMD_SIGNATURE_TYPE1,
        ) == Ordering::Equal
        {
            self.target_model_name =
                buffer.read_string_from_cp932(Self::VMD_TARGET_MODEL_NAME_LENGTH_V1)?;
            self.parse_vmd(buffer, offset)?;
        } else {
            return Err(Status::ErrorInvalidSignature);
        }
        Ok(())
    }

    fn load_from_buffer(&mut self, buffer: &mut Buffer, offset: u32) -> Result<(), Status> {
        self.load_from_buffer_vmd(buffer, offset)
    }

    fn save_to_buffer_bone_keyframe_block_vmd(
        &mut self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.bone_keyframes.len() as u32)?;
        for keyframe in &self.bone_keyframes {
            keyframe
                .borrow()
                .save_to_buffer(&mut self.local_bone_motion_track_bundle, buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_morph_keyframe_block_vmd(
        &mut self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.morph_keyframes.len() as u32)?;
        for keyframe in &self.morph_keyframes {
            keyframe
                .borrow()
                .save_to_buffer(&mut self.local_morph_motion_track_bundle, buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_light_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.light_keyframes.len() as u32)?;
        for keyframe in &self.light_keyframes {
            keyframe.save_to_buffer(buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_self_shadow_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.self_shadow_keyframes.len() as u32)?;
        for keyframe in &self.self_shadow_keyframes {
            keyframe.save_to_buffer(buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_model_keyframe_block_vmd(
        &mut self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.model_keyframes.len() as u32)?;
        for keyframe in &self.model_keyframes {
            keyframe.save_to_buffer(&mut self.local_bone_motion_track_bundle, buffer)?;
        }
        Ok(())
    }

    fn assign_global_trace_id(&mut self, value: &String) -> Result<i32, Status> {
        let (output, found_string) = self.resolve_global_track_id(&value);
        return Ok(output);
    }
}

#[macro_export]
macro_rules! search_closest {
    ($fn_name: ident, $field_name: ident, $typ: ty) => {
        pub fn $fn_name(&self, base_index: u32) -> (Option<&$typ>, Option<&$typ>) {
            let mut prev_keyframe: Option<&$typ> = None;
            let mut next_keyframe: Option<&$typ> = None;
            let mut last_keyframe: Option<&$typ> = None;
            let mut prev_nearest: u32 = u32::MAX;
            let mut next_nearest: u32 = u32::MAX;
            for keyframe in &self.$field_name {
                let frame_index = keyframe.base.frame_index;
                if base_index > frame_index && base_index - prev_nearest > base_index - frame_index
                {
                    prev_nearest = frame_index;
                    prev_keyframe = Some(keyframe)
                } else if base_index < frame_index
                    && next_nearest - base_index > frame_index - base_index
                {
                    next_nearest = frame_index;
                    next_keyframe = Some(keyframe);
                } else if last_keyframe.is_none()
                    || frame_index > last_keyframe.unwrap().base.frame_index
                {
                    last_keyframe = Some(keyframe);
                }
            }
            (
                prev_keyframe,
                if next_keyframe.is_some() {
                    next_keyframe
                } else {
                    last_keyframe
                },
            )
        }
    };
}

#[macro_export]
macro_rules! search_closest_with_borrow {
    ($fn_name: ident, $field_name: ident, $typ: ty) => {
        pub fn $fn_name(&self, base_index: u32) -> (Option<&$typ>, Option<&$typ>) {
            let mut prev_keyframe: Option<&$typ> = None;
            let mut next_keyframe: Option<&$typ> = None;
            let mut last_keyframe: Option<&$typ> = None;
            let mut prev_nearest: u32 = u32::MAX;
            let mut next_nearest: u32 = u32::MAX;
            for keyframe in &self.$field_name {
                let frame_index = keyframe.borrow().base.frame_index;
                if base_index > frame_index && base_index - prev_nearest > base_index - frame_index
                {
                    prev_nearest = frame_index;
                    prev_keyframe = Some(keyframe)
                } else if base_index < frame_index
                    && next_nearest - base_index > frame_index - base_index
                {
                    next_nearest = frame_index;
                    next_keyframe = Some(keyframe);
                } else if last_keyframe.is_none()
                    || frame_index > last_keyframe.unwrap().borrow().base.frame_index
                {
                    last_keyframe = Some(keyframe);
                }
            }
            (
                prev_keyframe,
                if next_keyframe.is_some() {
                    next_keyframe
                } else {
                    last_keyframe
                },
            )
        }
    };
}

impl Motion {
    pub fn get_format_type(&self) -> MotionFormatType {
        self.typ
    }

    pub fn get_target_model_name(&self) -> &String {
        &self.target_model_name
    }

    pub fn get_max_frame_index(&self) -> usize {
        self.max_frame_index as usize
    }

    pub fn get_annotation(&self, key: &String) -> Option<&String> {
        self.annotations.get(key)
    }

    pub fn get_all_accessory_keyframe_objects(&self) -> &Vec<MotionAccessoryKeyframe> {
        &self.accessory_keyframes
    }

    pub fn get_all_bone_keyframe_objects(&self) -> &Vec<Rc<RefCell<MotionBoneKeyframe>>> {
        &self.bone_keyframes
    }

    pub fn get_all_camera_keyframe_objects(&self) -> &Vec<MotionCameraKeyframe> {
        &self.camera_keyframes
    }

    pub fn get_all_light_keyframe_objects(&self) -> &Vec<MotionLightKeyframe> {
        &self.light_keyframes
    }

    pub fn get_all_motion_keyframe_objects(&self) -> &Vec<MotionModelKeyframe> {
        &self.model_keyframes
    }

    pub fn get_all_morph_keyframe_objects(&self) -> &Vec<Rc<RefCell<MotionMorphKeyframe>>> {
        &self.morph_keyframes
    }

    pub fn get_all_self_shadow_keyframe_objects(&self) -> &Vec<MotionSelfShadowKeyframe> {
        &self.self_shadow_keyframes
    }

    pub fn extract_bone_track_keyframes(
        &self,
        name: &String,
    ) -> Option<Vec<&MotionKeyframeObject>> {
        if let Some(keyframe_map) = self.local_bone_motion_track_bundle.find_keyframes_map(name) {
            Some(keyframe_map.values().collect())
        } else {
            None
        }
    }

    pub fn extract_morph_track_keyframes(
        &self,
        name: &String,
    ) -> Option<Vec<&MotionKeyframeObject>> {
        if let Some(keyframe_map) = self
            .local_morph_motion_track_bundle
            .find_keyframes_map(name)
        {
            Some(keyframe_map.values().collect())
        } else {
            None
        }
    }

    pub fn find_accessory_keyframe_object(&self, index: u32) -> Option<&MotionAccessoryKeyframe> {
        self.accessory_keyframes
            .binary_search_by(|keyframe| keyframe.base.frame_index.cmp(&index))
            .ok()
            .map(|index| self.accessory_keyframes.get(index))
            .unwrap_or_default()
    }

    pub fn find_bone_keyframe_object(
        &self,
        name: &String,
        index: u32,
    ) -> Option<&Rc<RefCell<MotionBoneKeyframe>>> {
        if let Some(keyframes_map) = self.local_bone_motion_track_bundle.find_keyframes_map(name) {
            if let Some(keyframe) = keyframes_map.get(&index) {
                match keyframe {
                    MotionKeyframeObject::Bone(bone_keyframe) => Some(bone_keyframe),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn find_camera_keyframe_object(&self, index: u32) -> Option<&MotionCameraKeyframe> {
        self.camera_keyframes
            .binary_search_by(|keyframe| keyframe.base.frame_index.cmp(&index))
            .ok()
            .map(|index| self.camera_keyframes.get(index))
            .unwrap_or_default()
    }

    pub fn find_light_keyframe_object(&self, index: u32) -> Option<&MotionLightKeyframe> {
        self.light_keyframes
            .binary_search_by(|keyframe| keyframe.base.frame_index.cmp(&index))
            .ok()
            .map(|index| self.light_keyframes.get(index))
            .unwrap_or_default()
    }

    pub fn find_model_keyframe_object(&self, index: u32) -> Option<&MotionModelKeyframe> {
        self.model_keyframes
            .binary_search_by(|keyframe| keyframe.base.frame_index.cmp(&index))
            .ok()
            .map(|index| self.model_keyframes.get(index))
            .unwrap_or_default()
    }

    pub fn find_morph_keyframe_object(
        &self,
        name: &String,
        index: u32,
    ) -> Option<&Rc<RefCell<MotionMorphKeyframe>>> {
        if let Some(keyframes_map) = self
            .local_morph_motion_track_bundle
            .find_keyframes_map(name)
        {
            if let Some(keyframe) = keyframes_map.get(&index) {
                match keyframe {
                    MotionKeyframeObject::Morph(morph_keyframe) => Some(morph_keyframe),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn find_self_shadow_keyframe_object(
        &self,
        index: u32,
    ) -> Option<&MotionSelfShadowKeyframe> {
        self.self_shadow_keyframes
            .binary_search_by(|keyframe| keyframe.base.frame_index.cmp(&index))
            .ok()
            .map(|index| self.self_shadow_keyframes.get(index))
            .unwrap_or_default()
    }

    search_closest!(
        search_closest_accessory_keyframes,
        accessory_keyframes,
        MotionAccessoryKeyframe
    );
    search_closest!(
        search_closest_camera_keyframes,
        camera_keyframes,
        MotionCameraKeyframe
    );
    search_closest!(
        search_closest_light_keyframes,
        light_keyframes,
        MotionLightKeyframe
    );
    search_closest!(
        search_closest_model_keyframes,
        model_keyframes,
        MotionModelKeyframe
    );
    search_closest!(
        search_closest_self_shadow_model_keyframes,
        self_shadow_keyframes,
        MotionSelfShadowKeyframe
    );
    search_closest_with_borrow!(
        search_closest_bone_keyframes,
        bone_keyframes,
        Rc<RefCell<MotionBoneKeyframe>>
    );
    search_closest_with_borrow!(
        search_closest_morph_keyframes,
        morph_keyframes,
        Rc<RefCell<MotionMorphKeyframe>>
    );

    pub fn get_user_data(&self) -> Option<&Rc<RefCell<UserData>>> {
        self.user_data.as_ref()
    }

    pub fn set_user_data(&mut self, user_data: &Rc<RefCell<UserData>>) {
        self.user_data = Some(user_data.clone())
    }
}

pub enum MotionKeyframeObject {
    Accessory(Rc<RefCell<MotionAccessoryKeyframe>>),
    Bone(Rc<RefCell<MotionBoneKeyframe>>),
    Camera(Rc<RefCell<MotionCameraKeyframe>>),
    Light(Rc<RefCell<MotionLightKeyframe>>),
    Model(Rc<RefCell<MotionModelKeyframe>>),
    Morph(Rc<RefCell<MotionMorphKeyframe>>),
    SelfShadow(Rc<RefCell<MotionSelfShadowKeyframe>>),
}

#[derive(Debug, Clone, Copy)]
pub enum MotionEffectParameterValue {
    BOOL(bool),
    INT(i32),
    FLOAT(f32),
    VECTOR4(F128),
}

impl Default for MotionEffectParameterValue {
    fn default() -> Self {
        MotionEffectParameterValue::BOOL(false)
    }
}

enum MotionParentKeyframe {
    ACCESSORY(Weak<RefCell<MotionAccessoryKeyframe>>),
    CAMERA(Weak<RefCell<MotionCameraKeyframe>>),
    MODEL(Weak<RefCell<MotionModelKeyframe>>),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MotionEffectParameter {
    parameter_id: i32,
    // keyframe: MotionParentKeyframe,
    value: MotionEffectParameterValue,
}

impl MotionEffectParameter {
    fn create_from_accessory_keyframe(
        keyframe: Rc<RefCell<MotionAccessoryKeyframe>>,
    ) -> Result<MotionEffectParameter, Status> {
        Ok(MotionEffectParameter {
            parameter_id: 0,
            // keyframe: MotionParentKeyframe::ACCESSORY(Rc::downgrade(&keyframe)),
            value: MotionEffectParameterValue::default(),
        })
    }

    fn create_from_model_keyframe(
        keyframe: Rc<RefCell<MotionModelKeyframe>>,
    ) -> Result<MotionEffectParameter, Status> {
        Ok(MotionEffectParameter {
            parameter_id: 0,
            // keyframe: MotionParentKeyframe::MODEL(Rc::downgrade(&keyframe)),
            value: MotionEffectParameterValue::default(),
        })
    }

    fn get_name<'a: 'b, 'b>(&self, parent_motion: &'a mut Motion) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_name(self.parameter_id)
    }

    fn set_name(&mut self, parent_motion: &mut Motion, value: &String) -> Result<(), Status> {
        self.parameter_id = parent_motion.assign_global_trace_id(value)?;
        Ok(())
    }

    fn get_value(&self) -> MotionEffectParameterValue {
        self.value
    }

    fn set_value(&mut self, value: MotionEffectParameterValue) {
        self.value = value;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MotionOutsideParent {
    global_model_track_index: i32,
    global_bone_track_index: i32,
    local_bone_track_index: i32,
}

impl MotionOutsideParent {
    fn get_target_object_name<'a: 'b, 'b>(
        &self,
        parent_motion: &'a Motion,
    ) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_name(self.global_model_track_index)
    }

    fn set_target_object_name(
        &mut self,
        parent_motion: &mut Motion,
        value: &String,
    ) -> Result<(), Status> {
        self.global_model_track_index = parent_motion.assign_global_trace_id(value)?;
        Ok(())
    }

    fn get_target_bone_name<'a: 'b, 'b>(
        &self,
        parent_motion: &'a Motion,
    ) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_name(self.global_bone_track_index)
    }

    fn set_target_bone_name(
        &mut self,
        parent_motion: &mut Motion,
        value: &String,
    ) -> Result<(), Status> {
        self.global_bone_track_index = parent_motion.assign_global_trace_id(value)?;
        Ok(())
    }
}

struct MotionKeyframeBase {
    index: i32,
    frame_index: u32,
    is_selected: bool,
    user_data: Option<Rc<RefCell<UserData>>>,
    annotations: HashMap<String, String>,
}

impl MotionKeyframeBase {
    fn compare(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.frame_index.cmp(&b.frame_index)
    }
}

pub struct MotionAccessoryKeyframe {
    base: MotionKeyframeBase,
    translation: F128,
    orientation: F128,
    scale_factor: f32,
    opacity: f32,
    is_add_blending_enabled: bool,
    is_shadow_enabled: bool,
    visible: bool,
    effect_parameters: Vec<MotionEffectParameter>,
    accessory_id: i32,
    outside_parent: Option<MotionOutsideParent>,
}

impl Clone for MotionAccessoryKeyframe {
    fn clone(&self) -> Self {
        Self {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: 0,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            translation: self.translation.clone(),
            orientation: self.orientation.clone(),
            scale_factor: self.scale_factor.clone(),
            opacity: self.opacity.clone(),
            is_add_blending_enabled: self.is_add_blending_enabled.clone(),
            is_shadow_enabled: self.is_shadow_enabled.clone(),
            visible: self.visible.clone(),
            effect_parameters: self.effect_parameters.clone(),
            accessory_id: 0,
            outside_parent: self.outside_parent.clone(),
        }
    }
}

impl MotionAccessoryKeyframe {
    fn create() -> MotionAccessoryKeyframe {
        MotionAccessoryKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: 0,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            translation: F128::default(),
            orientation: F128::default(),
            scale_factor: f32::default(),
            opacity: f32::default(),
            is_add_blending_enabled: bool::default(),
            is_shadow_enabled: true,
            visible: true,
            effect_parameters: vec![],
            accessory_id: 0,
            outside_parent: None,
        }
    }

    pub fn frame_index(&self) -> u32 {
        self.base.frame_index
    }

    pub fn frame_index_with_offset(&self, offset: i32) -> u32 {
        let frame_index = self.frame_index();
        if offset > 0 && frame_index + (offset as u32) < frame_index {
            u32::MAX
        } else if offset < 0 && frame_index - (offset.abs() as u32) > frame_index {
            0
        } else if offset >= 0 {
            frame_index + offset as u32
        } else {
            frame_index - offset.abs() as u32
        }
    }

    pub fn get_all_effect_parameters(&self) -> &[MotionEffectParameter] {
        &self.effect_parameters[..]
    }

    pub fn add_effect_parameter(&mut self, parameter: &MotionEffectParameter) {
        self.effect_parameters.push(parameter.clone());
    }

    pub fn set_outside_parent(&mut self, value: &MotionOutsideParent) {
        self.outside_parent = Some(*value);
    }

    fn try_copy_outside_parent(
        &self,
        parent_motion: &mut Motion,
        target_keyframe: &mut MotionAccessoryKeyframe,
    ) -> Result<(), Status> {
        if let Some(outside_parent) = (&self.outside_parent).as_ref() {
            let mut n_outside_parent = MotionOutsideParent::default();
            let name = outside_parent
                .get_target_object_name(parent_motion)
                .map_or("".into(), |name| name.clone());
            n_outside_parent.set_target_object_name(parent_motion, &name)?;
            let name = outside_parent
                .get_target_bone_name(parent_motion)
                .map_or("".into(), |name| name.clone());
            n_outside_parent.set_target_bone_name(parent_motion, &name)?;
            target_keyframe.set_outside_parent(&n_outside_parent);
        }
        Ok(())
    }

    /// Try to copy outside parent to target keyframe if it exists.
    /// Old outside parent will be replaced from target keyframe.
    pub fn copy_outside_parent(
        &self,
        parent_motion: &mut Motion,
        target_keyframe: &mut MotionAccessoryKeyframe,
    ) {
        let _ = self.try_copy_outside_parent(parent_motion, target_keyframe);
    }

    pub fn copy_all_effect_parameters(
        &self,
        parent_motion: &mut Motion,
        target_keyframe: &mut MotionAccessoryKeyframe,
    ) -> Result<(), Status> {
        let parameters = self.get_all_effect_parameters();
        for parameter in parameters {
            let mut n_parameter = MotionEffectParameter::default();
            let name = parameter
                .get_name(parent_motion)
                .map_or("".into(), |name| name.clone());
            n_parameter.set_name(parent_motion, &name)?;
            n_parameter.set_value(n_parameter.get_value());
            target_keyframe.add_effect_parameter(&n_parameter);
        }
        Ok(())
    }
}

const MOTION_BONE_KEYFRAME_INTERPOLATION_TYPE_MAX_ENUM: usize = 4;
const DEFAULT_INTERPOLATION: [u8; 4] = [20u8, 20u8, 107u8, 107u8];

struct MotionBoneKeyframeInterpolation {
    translation_x: [u8; 4],
    translation_y: [u8; 4],
    translation_z: [u8; 4],
    orientation: [u8; 4],
}

impl Default for MotionBoneKeyframeInterpolation {
    fn default() -> Self {
        Self {
            translation_x: DEFAULT_INTERPOLATION,
            translation_y: DEFAULT_INTERPOLATION,
            translation_z: DEFAULT_INTERPOLATION,
            orientation: DEFAULT_INTERPOLATION,
        }
    }
}

pub struct MotionBoneKeyframe {
    base: MotionKeyframeBase,
    translation: F128,
    orientation: F128,
    interpolation: MotionBoneKeyframeInterpolation,
    bone_id: i32,
    stage_index: u32,
    is_physics_simulation_enabled: bool,
}

impl MotionBoneKeyframe {
    const VMD_BONE_KEYFRAME_NAME_LENGTH: usize = 15;

    fn parse_vmd(
        parent_motion: &mut Motion,
        buffer: &mut Buffer,
        cache: &mut StringCache,
        offset: u32,
    ) -> Result<Rc<RefCell<MotionBoneKeyframe>>, Status> {
        let mut bone_keyframe = MotionBoneKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: 0,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            translation: F128::default(),
            orientation: F128::default(),
            interpolation: MotionBoneKeyframeInterpolation::default(),
            bone_id: 0,
            stage_index: 0,
            is_physics_simulation_enabled: true,
        };
        // try read utf8 name here
        let (str, len) = buffer.try_get_string_with_byte_len(Self::VMD_BONE_KEYFRAME_NAME_LENGTH);
        let mut name: Option<String> = None;
        // if raw_name read not empty
        if len > 0 && str[0] != 0u8 {
            let key = u8_slice_get_string(str, CodecType::Sjis).unwrap_or_default();
            // try get bone_id from cache or create new motion track with name
            if let Some(bone_id) = cache.0.get(&key) {
                buffer.skip(len)?;
                name = parent_motion
                    .local_bone_motion_track_bundle
                    .resolve_name(bone_id.clone())
                    .map(|name| name.clone());
                bone_keyframe.bone_id = bone_id.clone();
            } else {
                let read_name =
                    buffer.read_string_from_cp932(Self::VMD_BONE_KEYFRAME_NAME_LENGTH)?;
                name = Some(read_name.clone());
                let (bone_id, found_name) = parent_motion.resolve_local_bone_track_id(&read_name);
                bone_keyframe.bone_id = bone_id;
                if let Some(found_name) = found_name {
                    name = Some(found_name.clone())
                }
                cache.0.insert(key, bone_keyframe.bone_id);
            }
        } else {
            buffer.skip(Self::VMD_BONE_KEYFRAME_NAME_LENGTH)?;
            name = None;
        }
        bone_keyframe.base.frame_index = buffer.read_u32_little_endian()? + offset;
        bone_keyframe.translation = buffer.read_f32_3_little_endian()?;
        bone_keyframe.orientation = buffer.read_f32_4_little_endian()?;
        for i in 0..4 {
            bone_keyframe.interpolation.translation_x[i] = buffer.read_byte()?;
            bone_keyframe.interpolation.translation_y[i] = buffer.read_byte()?;
            bone_keyframe.interpolation.translation_z[i] = buffer.read_byte()?;
            bone_keyframe.interpolation.orientation[i] = buffer.read_byte()?;
        }
        buffer.skip(48usize)?;
        let bone_keyframe_index = bone_keyframe.base.frame_index;
        let rc = Rc::new(RefCell::new(bone_keyframe));
        parent_motion.local_bone_motion_track_bundle.add_keyframe(
            MotionKeyframeObject::Bone(rc.clone()),
            bone_keyframe_index,
            &name.unwrap_or_default(),
        );
        Ok(rc)
    }

    fn save_to_buffer(
        &self,
        bone_motion_track_bundle: &mut MotionTrackBundle,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        let name = bone_motion_track_bundle.resolve_name(self.bone_id);
        let name_tmp_vec: Vec<u8>;
        let name_cp932: &[u8] = if let Some(name) = &name {
            let (c, _, success) = encoding_rs::SHIFT_JIS.encode(name);
            if success {
                name_tmp_vec = c.to_vec();
                &name_tmp_vec[..]
            } else {
                &[]
            }
        } else {
            &[]
        };
        if name_cp932.len() >= Self::VMD_BONE_KEYFRAME_NAME_LENGTH {
            buffer.write_byte_array(&name_cp932[0..Self::VMD_BONE_KEYFRAME_NAME_LENGTH])?;
        } else {
            buffer.write_byte_array(
                &[
                    name_cp932,
                    &vec![0u8; Self::VMD_BONE_KEYFRAME_NAME_LENGTH - name_cp932.len()][0..],
                ]
                .concat(),
            )?;
        }
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_3_little_endian(self.translation)?;
        buffer.write_f32_4_little_endian(self.orientation)?;
        for i in 0..4 {
            buffer.write_byte(self.interpolation.translation_x[i])?;
            buffer.write_byte(self.interpolation.translation_y[i])?;
            buffer.write_byte(self.interpolation.translation_z[i])?;
            buffer.write_byte(self.interpolation.orientation[i])?;
        }
        for _ in 0..48 {
            buffer.write_byte(0u8)?;
        }
        Ok(())
    }
}

struct MotionCameraKeyframeInterpolation {
    lookat_x: [u8; 4],
    lookat_y: [u8; 4],
    lookat_z: [u8; 4],
    angle: [u8; 4],
    fov: [u8; 4],
    distance: [u8; 4],
}

impl Default for MotionCameraKeyframeInterpolation {
    fn default() -> Self {
        Self {
            lookat_x: DEFAULT_INTERPOLATION,
            lookat_y: DEFAULT_INTERPOLATION,
            lookat_z: DEFAULT_INTERPOLATION,
            angle: DEFAULT_INTERPOLATION,
            fov: DEFAULT_INTERPOLATION,
            distance: DEFAULT_INTERPOLATION,
        }
    }
}

pub struct MotionCameraKeyframe {
    base: MotionKeyframeBase,
    look_at: F128,
    angle: F128,
    distance: f32,
    fov: i32,
    interpolation: MotionCameraKeyframeInterpolation,
    is_perspective_view: bool,
    stage_index: u32,
    outside_parent: Option<MotionOutsideParent>,
}

impl MotionCameraKeyframe {
    fn parse_vmd(
        parent_motion: &mut Motion,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<MotionCameraKeyframe, Status> {
        let mut camera_keyframe = MotionCameraKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: buffer.read_u32_little_endian()? + offset,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            distance: buffer.read_f32_little_endian()?,
            look_at: buffer.read_f32_3_little_endian()?,
            angle: buffer.read_f32_3_little_endian()?,
            fov: i32::default(),
            interpolation: MotionCameraKeyframeInterpolation::default(),
            is_perspective_view: false,
            stage_index: 0,
            outside_parent: None,
        };
        for i in 0..4 {
            camera_keyframe.interpolation.lookat_x[i] = buffer.read_byte()?;
            camera_keyframe.interpolation.lookat_y[i] = buffer.read_byte()?;
            camera_keyframe.interpolation.lookat_z[i] = buffer.read_byte()?;
            camera_keyframe.interpolation.angle[i] = buffer.read_byte()?;
            camera_keyframe.interpolation.fov[i] = buffer.read_byte()?;
            camera_keyframe.interpolation.distance[i] = buffer.read_byte()?;
        }
        camera_keyframe.fov = buffer.read_i32_little_endian()?;
        camera_keyframe.is_perspective_view = buffer.read_byte()? == 0u8;
        Ok(camera_keyframe)
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_little_endian(self.distance)?;
        buffer.write_f32_3_little_endian(self.look_at)?;
        buffer.write_f32_3_little_endian(self.angle)?;
        for i in 0..4 {
            buffer.write_byte(self.interpolation.lookat_x[i])?;
            buffer.write_byte(self.interpolation.lookat_y[i])?;
            buffer.write_byte(self.interpolation.lookat_z[i])?;
            buffer.write_byte(self.interpolation.angle[i])?;
            buffer.write_byte(self.interpolation.fov[i])?;
            buffer.write_byte(self.interpolation.distance[i])?;
        }
        buffer.write_i32_little_endian(self.fov)?;
        buffer.write_byte(if self.is_perspective_view { 1u8 } else { 0u8 })?;
        Ok(())
    }
}

pub struct MotionLightKeyframe {
    base: MotionKeyframeBase,
    color: F128,
    direction: F128,
}

impl MotionLightKeyframe {
    fn parse_vmd(buffer: &mut Buffer, offset: u32) -> Result<MotionLightKeyframe, Status> {
        let light_keyframe = MotionLightKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: buffer.read_u32_little_endian()? + offset,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            color: buffer.read_f32_3_little_endian()?,
            direction: buffer.read_f32_3_little_endian()?,
        };
        Ok(light_keyframe)
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_3_little_endian(self.color)?;
        buffer.write_f32_3_little_endian(self.direction)?;
        Ok(())
    }
}

const PMD_BONE_NAME_LENGTH: usize = 20;

pub struct MotionModelKeyframeConstraintState {
    bone_id: i32,
    enabled: bool,
}

impl MotionModelKeyframeConstraintState {
    fn save_to_buffer(
        &self,
        bone_motion_track_bundle: &mut MotionTrackBundle,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        let name = bone_motion_track_bundle.resolve_name(self.bone_id);
        let name_cp932: Vec<u8> = if let Some(name) = &name {
            let (c, _, success) = encoding_rs::SHIFT_JIS.encode(name);
            if success {
                c.to_vec()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        if name_cp932.len() >= MotionModelKeyframe::PMD_BONE_NAME_LENGTH {
            buffer.write_byte_array(&name_cp932[0..MotionModelKeyframe::PMD_BONE_NAME_LENGTH])?;
        } else {
            buffer.write_byte_array(
                &[
                    &name_cp932[..],
                    &vec![0u8; MotionModelKeyframe::PMD_BONE_NAME_LENGTH - name_cp932.len()][0..],
                ]
                .concat(),
            )?;
        }
        buffer.write_byte(self.enabled as u8)?;
        Ok(())
    }
}

pub struct MotionModelKeyframe {
    base: MotionKeyframeBase,
    visible: bool,
    constraint_states: Vec<MotionModelKeyframeConstraintState>,
    effect_parameters: Vec<MotionEffectParameter>,
    outside_parents: Vec<MotionOutsideParent>,
    has_edge_option: bool,
    edge_scale_factor: f32,
    edge_color: F128,
    is_add_blending_enabled: bool,
    is_physics_simulation_enabled: bool,
}

impl MotionModelKeyframe {
    const PMD_BONE_NAME_LENGTH: usize = 20;

    fn parse_vmd(
        parent_motion: &mut Motion,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<MotionModelKeyframe, Status> {
        let mut model_keyframe = MotionModelKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: buffer.read_u32_little_endian()? + offset,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            visible: buffer.read_byte()? == 0,
            constraint_states: vec![],
            effect_parameters: vec![],
            outside_parents: vec![],
            has_edge_option: false,
            edge_scale_factor: 0f32,
            edge_color: F128::default(),
            is_add_blending_enabled: false,
            is_physics_simulation_enabled: true,
        };
        let num_constraint_states = buffer.read_len()?;
        if num_constraint_states > 0 {
            model_keyframe.constraint_states.clear();
            for i in 0..num_constraint_states {
                let name = buffer.read_string_from_cp932(Self::PMD_BONE_NAME_LENGTH)?;
                let (bone_id, _) = parent_motion.resolve_local_bone_track_id(&name);
                model_keyframe
                    .constraint_states
                    .push(MotionModelKeyframeConstraintState {
                        bone_id,
                        enabled: buffer.read_byte()? != 0,
                    });
            }
        }
        Ok(model_keyframe)
    }

    pub fn save_to_buffer(
        &self,
        bone_motion_track_bundle: &mut MotionTrackBundle,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_byte(self.visible as u8)?;
        buffer.write_u32_little_endian(self.constraint_states.len() as u32)?;
        for constraint_state in &self.constraint_states {
            constraint_state.save_to_buffer(bone_motion_track_bundle, buffer)?;
        }
        Ok(())
    }
}

const VMD_MORPH_KEYFRAME_NAME_LENGTH: usize = 15;

pub struct MotionMorphKeyframe {
    base: MotionKeyframeBase,
    weight: f32,
    morph_id: i32,
}

impl MotionMorphKeyframe {
    const VMD_MORPH_KEYFRAME_NAME_LENGTH: usize = 15;

    fn parse_vmd(
        parent_motion: &mut Motion,
        buffer: &mut Buffer,
        cache: &mut StringCache,
        offset: u32,
    ) -> Result<Rc<RefCell<MotionMorphKeyframe>>, Status> {
        let mut keyframe_morph_id = 0;
        let (str, _) = buffer.try_get_string_with_byte_len(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH);
        let mut name: Option<String> = None;
        if str.len() > 0 && str[0] != 0u8 {
            let key = u8_slice_get_string(str, CodecType::Sjis).unwrap_or_default();
            if let Some(morph_id) = cache.0.get(&key) {
                buffer.skip(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH)?;
                name = parent_motion
                    .local_morph_motion_track_bundle
                    .resolve_name(*morph_id)
                    .map(|name| name.clone());
                keyframe_morph_id = *morph_id;
            } else {
                name = Some(buffer.read_string_from_cp932(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH)?);
                let (morph_id, found_name) =
                    parent_motion.resolve_local_morph_track_id(&name.clone().unwrap_or_default());
                keyframe_morph_id = morph_id;
                if let Some(found_name) = found_name {
                    name = Some(found_name.clone());
                }
                cache.0.insert(key, keyframe_morph_id);
            }
        } else {
            buffer.skip(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH)?;
        }
        let frame_index = buffer.read_u32_little_endian()? + offset;
        let motion_morph_keyframe = MotionMorphKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            weight: buffer.read_f32_little_endian()?,
            morph_id: keyframe_morph_id,
        };
        let rc = Rc::new(RefCell::new(motion_morph_keyframe));
        parent_motion.local_morph_motion_track_bundle.add_keyframe(
            MotionKeyframeObject::Morph(rc.clone()),
            frame_index,
            &name.unwrap_or_default(),
        );
        Ok(rc)
    }

    fn save_to_buffer(
        &self,
        morph_motion_track_bundle: &mut MotionTrackBundle,
        buffer: &mut MutableBuffer,
    ) -> Result<(), Status> {
        let name = morph_motion_track_bundle.resolve_name(self.morph_id);
        let name_cp932: Vec<u8> = if let Some(name) = &name {
            let (c, _, success) = encoding_rs::SHIFT_JIS.encode(name);
            if success {
                c.to_vec()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        if name_cp932.len() >= Self::VMD_MORPH_KEYFRAME_NAME_LENGTH {
            buffer.write_byte_array(&name_cp932[0..Self::VMD_MORPH_KEYFRAME_NAME_LENGTH])?;
        } else {
            buffer.write_byte_array(
                &[
                    &name_cp932[..],
                    &vec![0u8; Self::VMD_MORPH_KEYFRAME_NAME_LENGTH - name_cp932.len()][0..],
                ]
                .concat(),
            )?;
        }
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_little_endian(self.weight)?;
        Ok(())
    }

    fn set_name(&mut self, parent_motion: &mut Motion, value: &String) -> Result<(), Status> {
        let (morph_id, found_name) = parent_motion.resolve_local_morph_track_id(value);
        Ok(())
    }
}

pub struct MotionSelfShadowKeyframe {
    base: MotionKeyframeBase,
    distance: f32,
    mode: i32,
}

impl MotionSelfShadowKeyframe {
    fn parse_vmd(buffer: &mut Buffer, offset: u32) -> Result<MotionSelfShadowKeyframe, Status> {
        let self_shadow_keyframe = MotionSelfShadowKeyframe {
            base: MotionKeyframeBase {
                index: 0,
                frame_index: buffer.read_u32_little_endian()? + offset,
                is_selected: false,
                user_data: None,
                annotations: HashMap::new(),
            },
            mode: buffer.read_byte()? as i32,
            distance: buffer.read_f32_little_endian()?,
        };
        Ok(self_shadow_keyframe)
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), Status> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_byte(self.mode as u8)?;
        buffer.write_f32_little_endian(self.distance)
    }
}

#[test]
fn test_bool_to_u8() {
    assert_eq!(1u8, true as u8);
    assert_eq!(0u8, false as u8);
}

#[test]
fn test_load_from_buffer() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut motion = Motion {
        annotations: HashMap::new(),
        target_model_name: String::default(),
        accessory_keyframes: vec![],
        bone_keyframes: vec![],
        morph_keyframes: vec![],
        camera_keyframes: vec![],
        light_keyframes: vec![],
        model_keyframes: vec![],
        self_shadow_keyframes: vec![],
        local_bone_motion_track_allocated_id: 0,
        local_bone_motion_track_bundle: MotionTrackBundle {
            tracks_by_name: HashMap::new(),
        },
        local_morph_motion_track_allocated_id: 0,
        local_morph_motion_track_bundle: MotionTrackBundle {
            tracks_by_name: HashMap::new(),
        },
        global_motion_track_allocated_id: 0,
        global_motion_track_bundle: MotionTrackBundle {
            tracks_by_name: HashMap::new(),
        },
        typ: MotionFormatType::default(),
        max_frame_index: 0,
        preferred_fps: 30f32,
        user_data: None,
    };

    let mut buffer = Buffer::create(std::fs::read(
        "test/example/Alicia/MMD Motion/2 for test 1.vmd",
    )?);
    match motion.load_from_buffer(&mut buffer, 0) {
        Ok(_) => println!("Parse VMD Success"),
        Err(e) => println!("Parse VMD with {:?}", &e),
    }
    Ok(())
}

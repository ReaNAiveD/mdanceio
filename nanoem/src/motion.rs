use std::{cmp::Ordering, collections::HashMap};

use crate::{
    common::{Buffer, MutableBuffer, NanoemError},
    utils::{compare, u8_slice_get_string},
};

pub trait Keyframe {
    fn frame_index(&self) -> u32;
}

#[derive(Debug, Clone)]
pub struct MotionTrack<K: Sized> {
    id: i32,
    pub name: String,
    pub keyframes: HashMap<u32, K>,
    pub ordered_frame_index: Vec<u32>,
}

impl<K> MotionTrack<K>
where
    K: Keyframe,
{
    pub fn search_closest(&self, frame_index: u32) -> (Option<&K>, Option<&K>) {
        match self.ordered_frame_index.binary_search(&frame_index) {
            Ok(pos) => (
                pos.checked_sub(1)
                    .and_then(|pos| self.keyframes.get(&self.ordered_frame_index[pos])),
                pos.checked_add(1)
                    .and_then(|pos| self.ordered_frame_index.get(pos))
                    .or_else(|| self.ordered_frame_index.last())
                    .and_then(|pos| self.keyframes.get(pos)),
            ),
            Err(pos) => (
                pos.checked_sub(1)
                    .and_then(|pos| self.keyframes.get(&self.ordered_frame_index[pos])),
                self.ordered_frame_index
                    .get(pos)
                    .or_else(|| self.ordered_frame_index.last())
                    .and_then(|pos| self.keyframes.get(pos)),
            ),
        }
    }

    /// Insert a Keyframe into the track without keeping ordered_frame_index sort.
    ///
    /// # Argument
    ///
    /// * `keyframe` - The keyframe to be inserted. The `frame_index` of keyframe will be used as hash key.
    /// * `force` - If force, we will replace old keyframe when `frame_index` conflicts.
    ///
    /// # Return
    ///
    /// Return a `Err(Status::ErrorMotionCameraKeyframeAlreadyExists)` when conflicting and not force.
    /// Otherwise return a Ok with replaced keyframe.
    fn insert_keyframe_no_sort(
        &mut self,
        keyframe: K,
        force: bool,
    ) -> Result<Option<K>, NanoemError> {
        let frame_index = keyframe.frame_index();
        if !force && self.keyframes.contains_key(&frame_index) {
            return Err(NanoemError::KeyframeAlreadyExists {
                track_name: self.name.clone(),
                frame_index,
            });
        }
        let old = self.keyframes.insert(frame_index, keyframe);
        if old.is_none() {
            self.ordered_frame_index.push(frame_index);
        }
        Ok(old)
    }

    pub fn insert_keyframe(&mut self, keyframe: K) -> Option<K> {
        let frame_index = keyframe.frame_index();
        let old = self.keyframes.insert(frame_index, keyframe);
        if old.is_none() {
            let pos = self
                .ordered_frame_index
                .binary_search(&frame_index)
                .unwrap_or_else(|e| e);
            self.ordered_frame_index.insert(pos, frame_index);
        }
        old
    }

    pub fn remove_keyframe(&mut self, frame_index: u32) -> Option<K> {
        if let Ok(index) = self.ordered_frame_index.binary_search(&frame_index) {
            self.ordered_frame_index.remove(index);
        }
        self.keyframes.remove(&frame_index)
    }
}

impl<K> MotionTrack<K> {
    pub fn new(name: &str, id: i32) -> Self {
        Self {
            id,
            name: name.to_owned(),
            keyframes: HashMap::new(),
            ordered_frame_index: vec![],
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &K> + '_ {
        self.ordered_frame_index
            .iter()
            .filter_map(|frame_index| self.keyframes.get(frame_index))
    }

    fn sort(&mut self) {
        self.ordered_frame_index.sort();
    }

    pub fn max_frame_index(&self) -> Option<u32> {
        self.ordered_frame_index.last().copied()
    }

    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct IdAllocator(i32);

impl IdAllocator {
    pub fn next(&mut self) -> i32 {
        self.0 += 1;
        self.0
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

#[derive(Debug, Clone)]
pub struct MotionTrackBundle<K: Sized> {
    allocator: IdAllocator,
    pub tracks: HashMap<String, MotionTrack<K>>,
}

impl<K> Default for MotionTrackBundle<K> {
    fn default() -> Self {
        Self {
            allocator: IdAllocator::default(),
            tracks: HashMap::new(),
        }
    }
}

impl<K> MotionTrackBundle<K> {
    pub fn new() -> MotionTrackBundle<K> {
        Self::default()
    }

    pub fn keyframe_len(&self) -> usize {
        self.tracks
            .values()
            .map(|track| track.keyframes.len())
            .sum()
    }

    fn get_by_name(&self, name: &str) -> Option<&MotionTrack<K>> {
        self.tracks.get(name)
    }

    pub fn resolve_id(&self, id: i32) -> Option<&String> {
        for track in self.tracks.values() {
            if track.id == id {
                return Some(&track.name);
            }
        }
        None
    }

    pub fn resolve_name(&self, name: &str) -> Option<i32> {
        self.get_by_name(name).map(|track| track.id)
    }

    fn resolve_name_or_new(&mut self, name: &str) -> i32 {
        self.tracks
            .entry(name.to_owned())
            .or_insert_with(|| MotionTrack {
                id: self.allocator.next(),
                name: name.to_string(),
                keyframes: HashMap::new(),
                ordered_frame_index: vec![],
            })
            .id
    }

    fn sort(&mut self) {
        for track in &mut self.tracks.values_mut() {
            track.sort();
        }
    }

    fn clear(&mut self) {
        self.allocator.clear();
        self.tracks = HashMap::new();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, String)> + '_ {
        self.tracks.iter().flat_map(|(name, track)| {
            track
                .keyframes
                .values()
                .map(|keyframe| (keyframe, name.to_owned()))
        })
    }

    pub fn find_keyframes_map(&self, track_name: &str) -> Option<&HashMap<u32, K>> {
        if let Some(track) = self.tracks.get(track_name) {
            Some(&track.keyframes)
        } else {
            None
        }
    }
}

impl<K> MotionTrackBundle<K>
where
    K: Keyframe,
{
    pub fn max_frame_index(&self) -> Option<u32> {
        self.tracks
            .values()
            .filter_map(|track| track.ordered_frame_index.last())
            .max()
            .copied()
    }

    /// Insert a Keyframe into the track without keeping ordered_frame_index sort.
    ///
    /// # Argument
    ///
    /// * `keyframe` - The keyframe to be inserted. The `frame_index` of keyframe will be used as hash key.
    /// * `force` - If force, we will replace old keyframe when `frame_index` conflicts.
    ///
    /// # Return
    ///
    /// Return a `Err(Status::ErrorMotionCameraKeyframeAlreadyExists)` when conflicting and not force.
    /// Otherwise return a Ok with replaced keyframe.
    fn insert_keyframe_no_sort(
        &mut self,
        keyframe: K,
        track_name: &str,
        force: bool,
    ) -> Result<Option<K>, NanoemError> {
        let track = self
            .tracks
            .entry(track_name.to_owned())
            .or_insert_with(|| MotionTrack {
                id: self.allocator.next(),
                name: track_name.to_string(),
                keyframes: HashMap::new(),
                ordered_frame_index: vec![],
            });
        track.insert_keyframe_no_sort(keyframe, force)
    }

    pub fn insert_keyframe(&mut self, keyframe: K, track_name: &str) -> Option<K> {
        let track = self
            .tracks
            .entry(track_name.to_owned())
            .or_insert_with(|| MotionTrack {
                id: self.allocator.next(),
                name: track_name.to_string(),
                keyframes: HashMap::new(),
                ordered_frame_index: vec![],
            });
        track.insert_keyframe(keyframe)
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

#[derive(Debug, Clone)]
pub struct Motion {
    pub annotations: HashMap<String, String>,
    pub target_model_name: String,
    pub accessory_keyframes: MotionTrack<MotionAccessoryKeyframe>,
    pub camera_keyframes: MotionTrack<MotionCameraKeyframe>,
    pub light_keyframes: MotionTrack<MotionLightKeyframe>,
    pub model_keyframes: MotionTrack<MotionModelKeyframe>,
    pub self_shadow_keyframes: MotionTrack<MotionSelfShadowKeyframe>,
    pub local_bone_motion_track_bundle: MotionTrackBundle<MotionBoneKeyframe>,
    pub local_morph_motion_track_bundle: MotionTrackBundle<MotionMorphKeyframe>,
    pub global_motion_track_bundle: MotionTrackBundle<()>, // 这个是用于NMD的
    pub typ: MotionFormatType,
    pub preferred_fps: f32,
    pub errors: Vec<NanoemError>,
}

impl Motion {
    const VMD_SIGNATURE_SIZE: usize = 30;
    const VMD_SIGNATURE_TYPE2: &'static [u8] = b"Vocaloid Motion Data 0002\0";
    const VMD_SIGNATURE_TYPE1: &'static [u8] = b"Vocaloid Motion Data file\0";
    const VMD_TARGET_MODEL_NAME_LENGTH_V2: usize = 20;
    const VMD_TARGET_MODEL_NAME_LENGTH_V1: usize = 10;

    pub fn empty() -> Self {
        Self {
            annotations: HashMap::new(),
            target_model_name: "".to_owned(),
            accessory_keyframes: MotionTrack::new("accessory", 0),
            camera_keyframes: MotionTrack::new("camera", 0),
            light_keyframes: MotionTrack::new("light", 0),
            model_keyframes: MotionTrack::new("model", 0),
            self_shadow_keyframes: MotionTrack::new("self_shadow", 0),
            local_bone_motion_track_bundle: MotionTrackBundle::new(),
            local_morph_motion_track_bundle: MotionTrackBundle::new(),
            global_motion_track_bundle: MotionTrackBundle::new(),
            typ: MotionFormatType::Unknown,
            preferred_fps: 30f32,
            errors: vec![],
        }
    }

    fn resolve_global_track_name(&mut self, name: &str) -> i32 {
        self.global_motion_track_bundle.resolve_name_or_new(name)
    }

    pub fn load_from_buffer(buffer: &mut Buffer, offset: u32) -> Result<Self, NanoemError> {
        Self::load_from_buffer_vmd(buffer, offset)
    }

    fn load_from_buffer_vmd(buffer: &mut Buffer, offset: u32) -> Result<Self, NanoemError> {
        let mut motion = Self::empty();
        let sig = buffer.read_buffer(Self::VMD_SIGNATURE_SIZE)?;
        if compare(
            &sig[0..Self::VMD_SIGNATURE_TYPE2.len()],
            Self::VMD_SIGNATURE_TYPE2,
        ) == Ordering::Equal
        {
            motion.typ = MotionFormatType::VMD;
            motion.target_model_name = buffer.read_string_from_cp932(
                Self::VMD_TARGET_MODEL_NAME_LENGTH_V2,
                &mut motion.errors,
            )?;
            motion.parse_vmd(buffer, offset)?;
        } else if compare(
            &sig[0..Self::VMD_SIGNATURE_TYPE1.len()],
            Self::VMD_SIGNATURE_TYPE1,
        ) == Ordering::Equal
        {
            motion.typ = MotionFormatType::VMD;
            motion.target_model_name = buffer.read_string_from_cp932(
                Self::VMD_TARGET_MODEL_NAME_LENGTH_V1,
                &mut motion.errors,
            )?;
            motion.parse_vmd(buffer, offset)?;
        } else {
            motion.typ = MotionFormatType::Unknown;
            return Err(NanoemError::InvalidSignature);
        }
        Ok(motion)
    }

    fn parse_vmd(&mut self, buffer: &mut Buffer, offset: u32) -> Result<(), NanoemError> {
        self.local_bone_motion_track_bundle =
            Self::parse_bone_keyframe_block_vmd(buffer, offset, &mut self.errors)?;
        self.local_morph_motion_track_bundle =
            Self::parse_morph_keyframe_block_vmd(buffer, offset, &mut self.errors)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.camera_keyframes = Self::parse_camera_keyframe_block_vmd(buffer, offset, &mut self.errors)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.light_keyframes = Self::parse_light_keyframe_block_vmd(buffer, offset, &mut self.errors)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.self_shadow_keyframes = Self::parse_self_shadow_keyframe_block_vmd(buffer, offset, &mut self.errors)?;
        if buffer.is_end() {
            return Ok(());
        }
        self.model_keyframes = Self::parse_model_keyframe_block_vmd(
            buffer,
            offset,
            &mut self.local_bone_motion_track_bundle,
            &mut self.errors,
        )?;
        if buffer.is_end() {
            Ok(())
        } else {
            Err(NanoemError::BufferNotEnd)
        }
    }

    fn parse_bone_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrackBundle<MotionBoneKeyframe>, NanoemError> {
        let num_bone_keyframes = buffer.read_len()?;
        let mut local_bone_motion_track_bundle = MotionTrackBundle::new();
        if num_bone_keyframes > 0 {
            local_bone_motion_track_bundle.clear();
            for _ in 0..num_bone_keyframes {
                let (keyframe, bone_name) = MotionBoneKeyframe::parse_vmd(buffer, offset, errors)?;
                if let Err(e) = local_bone_motion_track_bundle
                    .insert_keyframe_no_sort(keyframe, &bone_name, false)
                {
                    errors.push(e)
                };
            }
            local_bone_motion_track_bundle.sort();
        }
        Ok(local_bone_motion_track_bundle)
    }

    fn parse_morph_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrackBundle<MotionMorphKeyframe>, NanoemError> {
        let num_morph_keyframes = buffer.read_len()?;
        let mut local_morph_motion_track_bundle = MotionTrackBundle::new();
        if num_morph_keyframes > 0 {
            for _ in 0..num_morph_keyframes {
                let (keyframe, morph_name) =
                    MotionMorphKeyframe::parse_vmd(buffer, offset, errors)?;
                if let Err(e) = local_morph_motion_track_bundle.insert_keyframe_no_sort(
                    keyframe,
                    &morph_name,
                    false,
                ) {
                    errors.push(e)
                };
            }
            local_morph_motion_track_bundle.sort();
        }
        Ok(local_morph_motion_track_bundle)
    }

    fn parse_camera_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrack<MotionCameraKeyframe>, NanoemError> {
        let num_camera_keyframes = buffer.read_len()?;
        let mut camera_keyframes = MotionTrack::new("camera", 0);
        if num_camera_keyframes > 0 {
            for _ in 0..num_camera_keyframes {
                let keyframe = MotionCameraKeyframe::parse_vmd(buffer, offset)?;
                if let Err(e) = camera_keyframes.insert_keyframe_no_sort(keyframe, false) {
                    errors.push(e)
                };
            }
            camera_keyframes.sort();
        }
        Ok(camera_keyframes)
    }

    fn parse_light_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrack<MotionLightKeyframe>, NanoemError> {
        let num_light_keyframes = buffer.read_len()?;
        let mut light_keyframes = MotionTrack::new("light", 0);
        if num_light_keyframes > 0 {
            for _ in 0..num_light_keyframes {
                let keyframe = MotionLightKeyframe::parse_vmd(buffer, offset)?;
                if let Err(e) = light_keyframes.insert_keyframe_no_sort(keyframe, false) {
                    errors.push(e)
                };
            }
            light_keyframes.sort();
        }
        Ok(light_keyframes)
    }

    fn parse_self_shadow_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrack<MotionSelfShadowKeyframe>, NanoemError> {
        let num_self_shadow_keyframes = buffer.read_len()?;
        let mut self_shadow_keyframes = MotionTrack::new("self_shadow", 0);
        if num_self_shadow_keyframes > 0 {
            for _ in 0..num_self_shadow_keyframes {
                let keyframe = MotionSelfShadowKeyframe::parse_vmd(buffer, offset)?;
                if let Err(e) = self_shadow_keyframes.insert_keyframe_no_sort(keyframe, false) {
                    errors.push(e)
                };
            }
            self_shadow_keyframes.sort();
        }
        Ok(self_shadow_keyframes)
    }

    fn parse_model_keyframe_block_vmd(
        buffer: &mut Buffer,
        offset: u32,
        local_bone_track: &mut MotionTrackBundle<MotionBoneKeyframe>,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionTrack<MotionModelKeyframe>, NanoemError> {
        let num_model_keyframes = buffer.read_len()?;
        let mut model_keyframes = MotionTrack::new("model", 0);
        if num_model_keyframes > 0 {
            for _ in 0..num_model_keyframes {
                let keyframe =
                    MotionModelKeyframe::parse_vmd(buffer, offset, local_bone_track, errors)?;
                if let Err(e) = model_keyframes.insert_keyframe_no_sort(keyframe, false) {
                    errors.push(e)
                };
            }
            model_keyframes.sort();
        }
        Ok(model_keyframes)
    }

    pub fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), NanoemError> {
        buffer.write_byte_array(Self::VMD_SIGNATURE_TYPE2)?;
        buffer.write_i32_little_endian(0)?;
        let (bytes, _, has_errors) = encoding_rs::SHIFT_JIS.encode(&self.target_model_name);
        if has_errors {
            return Err(NanoemError::EncodeStringFailed(
                self.target_model_name.to_owned(),
            ));
        }
        buffer
            .write_byte_array(&bytes[0..Self::VMD_TARGET_MODEL_NAME_LENGTH_V2.min(bytes.len())])?;
        if Self::VMD_TARGET_MODEL_NAME_LENGTH_V2 > bytes.len() {
            buffer.write_byte_array(&vec![
                0u8;
                Self::VMD_TARGET_MODEL_NAME_LENGTH_V2 - bytes.len()
            ])?;
        }
        self.save_to_buffer_bone_keyframe_block_vmd(buffer)?;
        self.save_to_buffer_morph_keyframe_block_vmd(buffer)?;
        self.save_to_buffer_camera_keyframe_block_vmd(buffer)?;
        self.save_to_buffer_light_keyframe_block_vmd(buffer)?;
        self.save_to_buffer_self_shadow_keyframe_block_vmd(buffer)?;
        self.save_to_buffer_model_keyframe_block_vmd(buffer)?;
        Ok(())
    }

    fn save_to_buffer_bone_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.local_bone_motion_track_bundle.keyframe_len() as u32)?;
        for (keyframe, bone_name) in self.local_bone_motion_track_bundle.iter() {
            keyframe.save_to_buffer(&bone_name, buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_morph_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer
            .write_u32_little_endian(self.local_morph_motion_track_bundle.keyframe_len() as u32)?;
        for (keyframe, morph_name) in self.local_morph_motion_track_bundle.iter() {
            keyframe.save_to_buffer(&morph_name, buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_camera_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.camera_keyframes.len() as u32)?;
        for keyframe in self.camera_keyframes.iter() {
            keyframe.save_to_buffer(buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_light_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.light_keyframes.len() as u32)?;
        for keyframe in self.light_keyframes.iter() {
            keyframe.save_to_buffer(buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_self_shadow_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.self_shadow_keyframes.len() as u32)?;
        for keyframe in self.self_shadow_keyframes.iter() {
            keyframe.save_to_buffer(buffer)?;
        }
        Ok(())
    }

    fn save_to_buffer_model_keyframe_block_vmd(
        &self,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.model_keyframes.len() as u32)?;
        for keyframe in self.model_keyframes.iter() {
            keyframe.save_to_buffer(&self.local_bone_motion_track_bundle, buffer)?;
        }
        Ok(())
    }

    fn assign_global_trace_id(&mut self, value: &str) -> Result<i32, NanoemError> {
        let id = self.resolve_global_track_name(value);
        Ok(id)
    }
}

#[macro_export]
macro_rules! search_closest {
    ($fn_name: ident, $field_name: ident, $typ: ty) => {
        pub fn $fn_name(&self, base_index: u32) -> (Option<&$typ>, Option<&$typ>) {
            self.$field_name.search_closest(base_index)
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

    pub fn max_frame_index(&self) -> u32 {
        self.local_bone_motion_track_bundle
            .max_frame_index()
            .unwrap_or(0)
            .max(
                self.local_morph_motion_track_bundle
                    .max_frame_index()
                    .unwrap_or(0),
            )
            .max(self.camera_keyframes.max_frame_index().unwrap_or(0))
            .max(self.light_keyframes.max_frame_index().unwrap_or(0))
            .max(self.self_shadow_keyframes.max_frame_index().unwrap_or(0))
            .max(self.model_keyframes.max_frame_index().unwrap_or(0))
    }

    pub fn get_annotation(&self, key: &str) -> Option<&String> {
        self.annotations.get(key)
    }

    pub fn get_all_accessory_keyframe_objects(
        &self,
    ) -> impl Iterator<Item = &MotionAccessoryKeyframe> {
        self.accessory_keyframes.iter()
    }

    pub fn get_all_bone_keyframe_objects(
        &self,
    ) -> impl Iterator<Item = (&MotionBoneKeyframe, String)> {
        self.local_bone_motion_track_bundle.iter()
    }

    pub fn get_all_camera_keyframe_objects(&self) -> impl Iterator<Item = &MotionCameraKeyframe> {
        self.camera_keyframes.iter()
    }

    pub fn get_all_light_keyframe_objects(&self) -> impl Iterator<Item = &MotionLightKeyframe> {
        self.light_keyframes.iter()
    }

    pub fn get_all_motion_keyframe_objects(&self) -> impl Iterator<Item = &MotionModelKeyframe> {
        self.model_keyframes.iter()
    }

    pub fn get_all_morph_keyframe_objects(
        &self,
    ) -> impl Iterator<Item = (&MotionMorphKeyframe, String)> {
        self.local_morph_motion_track_bundle.iter()
    }

    pub fn get_all_self_shadow_keyframe_objects(
        &self,
    ) -> impl Iterator<Item = &MotionSelfShadowKeyframe> {
        self.self_shadow_keyframes.iter()
    }

    pub fn extract_bone_track_keyframes(
        &self,
        name: &str,
    ) -> Option<impl Iterator<Item = &MotionBoneKeyframe>> {
        self.local_bone_motion_track_bundle
            .find_keyframes_map(name)
            .map(|keyframe_map| keyframe_map.values())
    }

    pub fn extract_morph_track_keyframes(
        &self,
        name: &str,
    ) -> Option<impl Iterator<Item = &MotionMorphKeyframe>> {
        self.local_morph_motion_track_bundle
            .find_keyframes_map(name)
            .map(|keyframe_map| keyframe_map.values())
    }

    pub fn find_accessory_keyframe_object(&self, index: u32) -> Option<&MotionAccessoryKeyframe> {
        self.accessory_keyframes.keyframes.get(&index)
    }

    pub fn find_bone_keyframe_object(&self, name: &str, index: u32) -> Option<&MotionBoneKeyframe> {
        if let Some(keyframes_map) = self.local_bone_motion_track_bundle.find_keyframes_map(name) {
            keyframes_map.get(&index)
        } else {
            None
        }
    }

    pub fn find_camera_keyframe_object(&self, frame_index: u32) -> Option<&MotionCameraKeyframe> {
        self.camera_keyframes.keyframes.get(&frame_index)
    }

    pub fn remove_camera_keyframe_object(
        &mut self,
        frame_index: u32,
    ) -> Option<MotionCameraKeyframe> {
        self.camera_keyframes.remove_keyframe(frame_index)
    }

    pub fn find_light_keyframe_object(&self, frame_index: u32) -> Option<&MotionLightKeyframe> {
        self.light_keyframes.keyframes.get(&frame_index)
    }

    pub fn remove_light_keyframe_object(
        &mut self,
        frame_index: u32,
    ) -> Option<MotionLightKeyframe> {
        self.light_keyframes.remove_keyframe(frame_index)
    }

    pub fn find_model_keyframe_object(&self, frame_index: u32) -> Option<&MotionModelKeyframe> {
        self.model_keyframes.keyframes.get(&frame_index)
    }

    pub fn remove_model_keyframe_object(
        &mut self,
        frame_index: u32,
    ) -> Option<MotionModelKeyframe> {
        self.model_keyframes.remove_keyframe(frame_index)
    }

    pub fn find_morph_keyframe_object(
        &self,
        name: &str,
        index: u32,
    ) -> Option<&MotionMorphKeyframe> {
        if let Some(keyframes_map) = self
            .local_morph_motion_track_bundle
            .find_keyframes_map(name)
        {
            keyframes_map.get(&index)
        } else {
            None
        }
    }

    pub fn find_self_shadow_keyframe_object(
        &self,
        frame_index: u32,
    ) -> Option<&MotionSelfShadowKeyframe> {
        self.self_shadow_keyframes.keyframes.get(&frame_index)
    }

    pub fn remove_self_shadow_keyframe_object(
        &mut self,
        frame_index: u32,
    ) -> Option<MotionSelfShadowKeyframe> {
        self.self_shadow_keyframes.remove_keyframe(frame_index)
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

    pub fn search_closest_bone_keyframes(
        &self,
        track_name: &str,
        frame_index: u32,
    ) -> (Option<&MotionBoneKeyframe>, Option<&MotionBoneKeyframe>) {
        if let Some(track) = self.local_bone_motion_track_bundle.get_by_name(track_name) {
            track.search_closest(frame_index)
        } else {
            (None, None)
        }
    }

    pub fn search_closest_morph_keyframes(
        &self,
        track_name: &str,
        frame_index: u32,
    ) -> (Option<&MotionMorphKeyframe>, Option<&MotionMorphKeyframe>) {
        if let Some(track) = self.local_morph_motion_track_bundle.get_by_name(track_name) {
            track.search_closest(frame_index)
        } else {
            (None, None)
        }
    }

    pub fn add_accessory_keyframe(
        &mut self,
        keyframe: MotionAccessoryKeyframe,
    ) -> Option<MotionAccessoryKeyframe> {
        self.accessory_keyframes.insert_keyframe(keyframe)
    }

    pub fn add_camera_keyframe(
        &mut self,
        keyframe: MotionCameraKeyframe,
    ) -> Option<MotionCameraKeyframe> {
        self.camera_keyframes.insert_keyframe(keyframe)
    }

    pub fn add_light_keyframe(
        &mut self,
        keyframe: MotionLightKeyframe,
    ) -> Option<MotionLightKeyframe> {
        self.light_keyframes.insert_keyframe(keyframe)
    }

    pub fn add_model_keyframe(
        &mut self,
        keyframe: MotionModelKeyframe,
    ) -> Option<MotionModelKeyframe> {
        self.model_keyframes.insert_keyframe(keyframe)
    }

    pub fn add_self_shadow_keyframe(
        &mut self,
        keyframe: MotionSelfShadowKeyframe,
    ) -> Option<MotionSelfShadowKeyframe> {
        self.self_shadow_keyframes.insert_keyframe(keyframe)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MotionEffectParameterValue {
    BOOL(bool),
    INT(i32),
    FLOAT(f32),
    VECTOR4([f32; 4]),
}

impl Default for MotionEffectParameterValue {
    fn default() -> Self {
        MotionEffectParameterValue::BOOL(false)
    }
}

impl MotionEffectParameterValue {
    fn lerp(a: f32, b: f32, amount: f32) -> f32 {
        (1f32 - amount) * a + amount * b
    }

    pub fn lerp_or_first(self, other: Self, amount: f32) -> Self {
        match (self, other) {
            (MotionEffectParameterValue::INT(a), MotionEffectParameterValue::INT(b)) => {
                MotionEffectParameterValue::INT(Self::lerp(a as f32, b as f32, amount) as i32)
            }
            (MotionEffectParameterValue::FLOAT(a), MotionEffectParameterValue::FLOAT(b)) => {
                MotionEffectParameterValue::FLOAT(Self::lerp(a, b, amount))
            }
            (MotionEffectParameterValue::VECTOR4(a), MotionEffectParameterValue::VECTOR4(b)) => {
                MotionEffectParameterValue::VECTOR4([
                    Self::lerp(a[0], b[0], amount),
                    Self::lerp(a[1], b[1], amount),
                    Self::lerp(a[2], b[2], amount),
                    Self::lerp(a[3], b[3], amount),
                ])
            }
            _ => self,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MotionEffectParameter {
    pub parameter_id: i32,
    // pub keyframe: MotionParentKeyframe,
    pub value: MotionEffectParameterValue,
}

impl MotionEffectParameter {
    fn get_name<'a: 'b, 'b>(&self, parent_motion: &'a Motion) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_id(self.parameter_id)
    }

    fn set_name(&mut self, parent_motion: &mut Motion, value: &str) -> Result<(), NanoemError> {
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
    pub global_model_track_index: i32, // TargetObject
    pub global_bone_track_index: i32,  // TargetBone
    pub local_bone_track_index: i32,   // SubjectBone
}

impl MotionOutsideParent {
    fn get_target_object_name<'a: 'b, 'b>(&self, parent_motion: &'a Motion) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_id(self.global_model_track_index)
    }

    fn set_target_object_name(
        &mut self,
        parent_motion: &mut Motion,
        value: &str,
    ) -> Result<(), NanoemError> {
        self.global_model_track_index = parent_motion.assign_global_trace_id(value)?;
        Ok(())
    }

    fn get_target_bone_name<'a: 'b, 'b>(&self, parent_motion: &'a Motion) -> Option<&'b String> {
        parent_motion
            .global_motion_track_bundle
            .resolve_id(self.global_bone_track_index)
    }

    fn set_target_bone_name(
        &mut self,
        parent_motion: &mut Motion,
        value: &str,
    ) -> Result<(), NanoemError> {
        self.global_bone_track_index = parent_motion.assign_global_trace_id(value)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MotionKeyframeBase {
    pub frame_index: u32,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug)]
pub struct MotionAccessoryKeyframe {
    pub base: MotionKeyframeBase,
    pub translation: [f32; 4],
    pub orientation: [f32; 4],
    pub scale_factor: f32,
    pub opacity: f32,
    pub is_add_blending_enabled: bool,
    pub is_shadow_enabled: bool,
    pub visible: bool,
    pub effect_parameters: Vec<MotionEffectParameter>,
    pub accessory_id: i32,
    pub outside_parent: Option<MotionOutsideParent>,
}

impl Clone for MotionAccessoryKeyframe {
    fn clone(&self) -> Self {
        Self {
            base: MotionKeyframeBase {
                frame_index: 0,
                annotations: HashMap::new(),
            },
            translation: self.translation,
            orientation: self.orientation,
            scale_factor: self.scale_factor,
            opacity: self.opacity,
            is_add_blending_enabled: self.is_add_blending_enabled,
            is_shadow_enabled: self.is_shadow_enabled,
            visible: self.visible,
            effect_parameters: self.effect_parameters.clone(),
            accessory_id: 0,
            outside_parent: self.outside_parent,
        }
    }
}

impl Keyframe for MotionAccessoryKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionAccessoryKeyframe {
    pub fn create() -> MotionAccessoryKeyframe {
        MotionAccessoryKeyframe {
            base: MotionKeyframeBase {
                frame_index: 0,
                annotations: HashMap::new(),
            },
            translation: <[f32; 4]>::default(),
            orientation: <[f32; 4]>::default(),
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
        } else if offset < 0 && frame_index - offset.unsigned_abs() > frame_index {
            0
        } else if offset >= 0 {
            frame_index + offset as u32
        } else {
            frame_index - offset.unsigned_abs()
        }
    }

    pub fn get_all_effect_parameters(&self) -> &[MotionEffectParameter] {
        &self.effect_parameters[..]
    }

    pub fn add_effect_parameter(&mut self, parameter: &MotionEffectParameter) {
        self.effect_parameters.push(*parameter);
    }

    pub fn set_outside_parent(&mut self, value: &MotionOutsideParent) {
        self.outside_parent = Some(*value);
    }

    fn try_copy_outside_parent(
        &self,
        parent_motion: &mut Motion,
        target_keyframe: &mut MotionAccessoryKeyframe,
    ) -> Result<(), NanoemError> {
        if let Some(outside_parent) = self.outside_parent.as_ref() {
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
    ) -> Result<(), NanoemError> {
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

const DEFAULT_INTERPOLATION: [u8; 4] = [20u8, 20u8, 107u8, 107u8];

#[derive(Debug, Clone, Copy, Hash)]
pub struct MotionBoneKeyframeInterpolation {
    pub translation_x: [u8; 4],
    pub translation_y: [u8; 4],
    pub translation_z: [u8; 4],
    pub orientation: [u8; 4],
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

#[derive(Debug, Clone)]
pub struct MotionBoneKeyframe {
    pub base: MotionKeyframeBase,
    pub translation: [f32; 4],
    pub orientation: [f32; 4],
    pub interpolation: MotionBoneKeyframeInterpolation,
    pub stage_index: u32,
    pub is_physics_simulation_enabled: bool,
}

impl Keyframe for MotionBoneKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionBoneKeyframe {
    const VMD_BONE_KEYFRAME_NAME_LENGTH: usize = 15;

    /// Parse bytes from a vmd file into `BoneKeyframe`
    ///
    /// # Argument
    ///
    /// `buffer` - bytes buffer read from
    /// `offset` - offset applied to `frame_index`
    ///
    /// # Return
    ///
    /// The bone keyframe and bone name if successful.
    fn parse_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<(MotionBoneKeyframe, String), NanoemError> {
        let mut bone_keyframe = MotionBoneKeyframe {
            base: MotionKeyframeBase {
                frame_index: 0,
                annotations: HashMap::new(),
            },
            translation: <[f32; 4]>::default(),
            orientation: <[f32; 4]>::default(),
            interpolation: MotionBoneKeyframeInterpolation::default(),
            stage_index: 0,
            is_physics_simulation_enabled: true,
        };
        // try read utf8 name here
        let str = buffer.try_get_string_with_byte_len(Self::VMD_BONE_KEYFRAME_NAME_LENGTH);
        let len = str.len();
        let name = if len > 0 && str[0] != 0u8 {
            let s = u8_slice_get_string(str, encoding_rs::SHIFT_JIS, errors);
            buffer.skip(len)?;
            s
        } else {
            buffer.skip(Self::VMD_BONE_KEYFRAME_NAME_LENGTH)?;
            "".to_owned()
        };
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
        Ok((bone_keyframe, name))
    }

    fn save_to_buffer(
        &self,
        bone_name: &str,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        let (c, _, had_errors) = encoding_rs::SHIFT_JIS.encode(bone_name);
        if !had_errors {
            buffer.write_byte_array(&c[0..Self::VMD_BONE_KEYFRAME_NAME_LENGTH.min(c.len())])?;
            if Self::VMD_BONE_KEYFRAME_NAME_LENGTH > c.len() {
                buffer
                    .write_byte_array(&vec![0u8; Self::VMD_BONE_KEYFRAME_NAME_LENGTH - c.len()])?;
            }
        } else {
            buffer.write_byte_array(&[0u8; Self::VMD_BONE_KEYFRAME_NAME_LENGTH])?;
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
        buffer.write_byte_array(&[0u8; 48])?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MotionCameraKeyframeInterpolation {
    pub lookat_x: [u8; 4],
    pub lookat_y: [u8; 4],
    pub lookat_z: [u8; 4],
    pub angle: [u8; 4],
    pub fov: [u8; 4],
    pub distance: [u8; 4],
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

#[derive(Debug, Clone)]
pub struct MotionCameraKeyframe {
    pub base: MotionKeyframeBase,
    pub look_at: [f32; 4],
    pub angle: [f32; 4],
    pub distance: f32,
    pub fov: i32,
    pub interpolation: MotionCameraKeyframeInterpolation,
    pub is_perspective_view: bool,
    pub stage_index: u32,
    pub outside_parent: Option<MotionOutsideParent>,
}

impl Keyframe for MotionCameraKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionCameraKeyframe {
    fn parse_vmd(buffer: &mut Buffer, offset: u32) -> Result<MotionCameraKeyframe, NanoemError> {
        let mut camera_keyframe = MotionCameraKeyframe {
            base: MotionKeyframeBase {
                frame_index: buffer.read_u32_little_endian()? + offset,
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

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), NanoemError> {
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

#[derive(Debug, Clone)]
pub struct MotionLightKeyframe {
    pub base: MotionKeyframeBase,
    pub color: [f32; 4],
    pub direction: [f32; 4],
}

impl Keyframe for MotionLightKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionLightKeyframe {
    fn parse_vmd(buffer: &mut Buffer, offset: u32) -> Result<MotionLightKeyframe, NanoemError> {
        let light_keyframe = MotionLightKeyframe {
            base: MotionKeyframeBase {
                frame_index: buffer.read_u32_little_endian()? + offset,
                annotations: HashMap::new(),
            },
            color: buffer.read_f32_3_little_endian()?,
            direction: buffer.read_f32_3_little_endian()?,
        };
        Ok(light_keyframe)
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_3_little_endian(self.color)?;
        buffer.write_f32_3_little_endian(self.direction)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct MotionModelKeyframeConstraintState {
    pub bone_id: i32,
    pub enabled: bool,
}

impl MotionModelKeyframeConstraintState {
    fn save_to_buffer(
        &self,
        bone_motion_track_bundle: &MotionTrackBundle<MotionBoneKeyframe>,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        let name = bone_motion_track_bundle.resolve_id(self.bone_id);
        let name_cp932: Vec<u8> = if let Some(name) = &name {
            let (c, _, has_errors) = encoding_rs::SHIFT_JIS.encode(name);
            if !has_errors {
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

#[derive(Debug, Clone)]
pub struct MotionModelKeyframe {
    pub base: MotionKeyframeBase,
    pub visible: bool,
    pub constraint_states: Vec<MotionModelKeyframeConstraintState>,
    pub effect_parameters: Vec<MotionEffectParameter>,
    pub outside_parents: Vec<MotionOutsideParent>,
    pub has_edge_option: bool,
    pub edge_scale_factor: f32,
    pub edge_color: [f32; 4],
    pub is_add_blending_enabled: bool,
    pub is_physics_simulation_enabled: bool,
}

impl Keyframe for MotionModelKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionModelKeyframe {
    const PMD_BONE_NAME_LENGTH: usize = 20;

    fn parse_vmd(
        buffer: &mut Buffer,
        offset: u32,
        local_bone_track: &mut MotionTrackBundle<MotionBoneKeyframe>,
        errors: &mut Vec<NanoemError>,
    ) -> Result<MotionModelKeyframe, NanoemError> {
        let mut model_keyframe = MotionModelKeyframe {
            base: MotionKeyframeBase {
                frame_index: buffer.read_u32_little_endian()? + offset,
                annotations: HashMap::new(),
            },
            visible: buffer.read_byte()? != 0,
            constraint_states: vec![],
            effect_parameters: vec![],
            outside_parents: vec![],
            has_edge_option: false,
            edge_scale_factor: 0f32,
            edge_color: <[f32; 4]>::default(),
            is_add_blending_enabled: false,
            is_physics_simulation_enabled: true,
        };
        let num_constraint_states = buffer.read_len()?;
        if num_constraint_states > 0 {
            model_keyframe.constraint_states.clear();
            for _ in 0..num_constraint_states {
                let name = buffer.read_string_from_cp932(Self::PMD_BONE_NAME_LENGTH, errors)?;
                let bone_id = local_bone_track.resolve_name_or_new(&name);
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
        bone_motion_track_bundle: &MotionTrackBundle<MotionBoneKeyframe>,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_byte(self.visible as u8)?;
        buffer.write_u32_little_endian(self.constraint_states.len() as u32)?;
        for constraint_state in &self.constraint_states {
            constraint_state.save_to_buffer(bone_motion_track_bundle, buffer)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MotionMorphKeyframe {
    pub base: MotionKeyframeBase,
    pub weight: f32,
}

impl Keyframe for MotionMorphKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionMorphKeyframe {
    const VMD_MORPH_KEYFRAME_NAME_LENGTH: usize = 15;

    fn parse_vmd(
        buffer: &mut Buffer,
        offset: u32,
        errors: &mut Vec<NanoemError>,
    ) -> Result<(MotionMorphKeyframe, String), NanoemError> {
        let str = buffer.try_get_string_with_byte_len(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH);
        let len = str.len();
        let name = if len > 0 && str[0] != 0u8 {
            let s = u8_slice_get_string(str, encoding_rs::SHIFT_JIS, errors);
            buffer.skip(len)?;
            s
        } else {
            buffer.skip(Self::VMD_MORPH_KEYFRAME_NAME_LENGTH)?;
            "".to_owned()
        };
        let frame_index = buffer.read_u32_little_endian()? + offset;
        let motion_morph_keyframe = MotionMorphKeyframe {
            base: MotionKeyframeBase {
                frame_index,
                annotations: HashMap::new(),
            },
            weight: buffer.read_f32_little_endian()?,
        };
        Ok((motion_morph_keyframe, name))
    }

    fn save_to_buffer(
        &self,
        morph_name: &str,
        buffer: &mut MutableBuffer,
    ) -> Result<(), NanoemError> {
        let (c, _, has_errors) = encoding_rs::SHIFT_JIS.encode(morph_name);
        if !has_errors {
            buffer.write_byte_array(&c[0..Self::VMD_MORPH_KEYFRAME_NAME_LENGTH.min(c.len())])?;
            if Self::VMD_MORPH_KEYFRAME_NAME_LENGTH > c.len() {
                buffer
                    .write_byte_array(&vec![0u8; Self::VMD_MORPH_KEYFRAME_NAME_LENGTH - c.len()])?;
            }
        } else {
            buffer.write_byte_array(&[0u8; Self::VMD_MORPH_KEYFRAME_NAME_LENGTH])?;
        }
        buffer.write_u32_little_endian(self.base.frame_index)?;
        buffer.write_f32_little_endian(self.weight)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MotionSelfShadowKeyframe {
    pub base: MotionKeyframeBase,
    pub distance: f32,
    pub mode: i32,
}

impl Keyframe for MotionSelfShadowKeyframe {
    fn frame_index(&self) -> u32 {
        self.base.frame_index
    }
}

impl MotionSelfShadowKeyframe {
    fn parse_vmd(
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<MotionSelfShadowKeyframe, NanoemError> {
        let self_shadow_keyframe = MotionSelfShadowKeyframe {
            base: MotionKeyframeBase {
                frame_index: buffer.read_u32_little_endian()? + offset,
                annotations: HashMap::new(),
            },
            mode: buffer.read_byte()? as i32,
            distance: buffer.read_f32_little_endian()?,
        };
        Ok(self_shadow_keyframe)
    }

    fn save_to_buffer(&self, buffer: &mut MutableBuffer) -> Result<(), NanoemError> {
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
fn test_sig_len() {
    assert_eq!(Motion::VMD_SIGNATURE_TYPE2.len(), 26usize);
    assert_eq!(Motion::VMD_SIGNATURE_TYPE1.len(), 26usize);
}

#[test]
fn test_load_from_buffer() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let motion_data = std::fs::read("test/example/Alicia/MMD Motion/2暘儖乕僾僗僥僢僾5.vmd")?;
    let mut buffer = Buffer::create(&motion_data);
    match Motion::load_from_buffer(&mut buffer, 0) {
        Ok(motion) => {
            for e in motion.errors {
                println!("{}", e);
            }
            println!("Parse VMD Success");
        }
        Err(e) => panic!("Parse VMD with {:?}", &e),
    }
    Ok(())
}

#[test]
fn test_save_into_buffer() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let motion_data = std::fs::read("test/example/Alicia/MMD Motion/2暘儖乕僾僗僥僢僾5.vmd")?;
    let mut buffer = Buffer::create(&motion_data);
    match Motion::load_from_buffer(&mut buffer, 0) {
        Ok(motion) => {
            println!("Parse VMD Success");
            let mut mut_buffer = MutableBuffer::create().unwrap();
            match motion.save_to_buffer(&mut mut_buffer) {
                Ok(_) => {
                    if let Ok(mut buffer) = mut_buffer.create_buffer_object() {
                        match Motion::load_from_buffer(&mut buffer, 0) {
                            Ok(motion) => {
                                assert!(motion.local_bone_motion_track_bundle.keyframe_len() > 0);
                                println!("Parse Saved VMD Success");
                            }
                            Err(e) => panic!("Parse Saved VMD with {:?}", &e),
                        }
                    }
                }
                Err(e) => panic!("Save VMD with {:?}", &e),
            }
        }
        Err(e) => panic!("Parse VMD with {:?}", &e),
    }
    Ok(())
}

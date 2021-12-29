use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};

use crate::{
    common::{Buffer, Status, UserData, F128},
    utils::u8_slice_get_string,
};

#[derive(Clone, Default)]
struct StringCache(HashMap<String, i32>);

struct MotionTrack {
    id: i32,
    name: String,
    keyframes: HashMap<u32, MotionKeyframeObject>,
}

struct MotionTrackBundle {
    tracks_by_name: HashMap<String, MotionTrack>,
}

impl MotionTrackBundle {
    fn get_mut_by_name(&mut self, name: String) -> Option<&mut MotionTrack> {
        self.tracks_by_name.get_mut(&name)
    }

    fn put_new(&mut self, id: i32, name: String) -> Option<MotionTrack> {
        self.tracks_by_name.insert(
            name.clone(),
            MotionTrack {
                id,
                name,
                keyframes: HashMap::new(),
            },
        )
    }

    fn resolve_name(&self, id: i32) -> Option<String> {
        for track in self.tracks_by_name.values() {
            if track.id == id {
                return Some(track.name.clone());
            }
        }
        None
    }

    fn resolve_id(&mut self, name: String, allocated_id: &mut i32) -> (i32, Option<String>) {
        if let Some(track) = self.get_mut_by_name(name.clone()) {
            (track.id, Some(track.name.clone()))
        } else {
            *allocated_id += 1;
            let id = *allocated_id;
            self.put_new(id, name);
            (id, None)
        }
    }

    fn add_keyframe(&mut self, keyframe: MotionKeyframeObject, frame_index: u32, name: String) {
        if let Some(track) = self.get_mut_by_name(name) {
            track.keyframes.insert(frame_index, keyframe);
        }
    }
}

pub struct Motion {
    annotations: HashMap<String, String>,
    target_model_name: String,
    accessory_keyframes: Vec<MotionAccessoryKeyframe>,
    bone_keyframes: Vec<MotionBoneKeyframe>,
    morph_keyframes: Vec<MotionMorphKeyframe>,
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
    typ: i32,
    max_frame_index: u32,
    preferred_fps: f32,
    user_data: Rc<RefCell<UserData>>,
}

impl Motion {
    fn resolve_local_bone_track_id(&mut self, name: String) -> (i32, Option<String>) {
        self.local_bone_motion_track_bundle
            .resolve_id(name, &mut self.local_bone_motion_track_allocated_id)
    }

    fn resolve_local_morph_track_id(&mut self, name: String) -> (i32, Option<String>) {
        self.local_morph_motion_track_bundle
            .resolve_id(name, &mut self.local_morph_motion_track_allocated_id)
    }

    fn parse_bone_keyframe_block_vmd(
        &mut self,
        buffer: &mut Buffer,
        offset: u32,
    ) -> Result<(), Status> {
        let num_bone_keyframes = buffer.read_len()?;
        if num_bone_keyframes > 0 {
            let cache = StringCache::default();
            self.bone_keyframes.clear();
        }
        Ok(())
    }
}

enum MotionKeyframeObject {
    Accessory(Rc<RefCell<MotionAccessoryKeyframe>>),
    Bone(Rc<RefCell<MotionBoneKeyframe>>),
    Camera(Rc<RefCell<MotionCameraKeyframe>>),
    Light(Rc<RefCell<MotionLightKeyframe>>),
    Model(Rc<RefCell<MotionModelKeyframe>>),
    Morph(Rc<RefCell<MotionMorphKeyframe>>),
    SelfShadow(Rc<RefCell<MotionSelfShadowKeyframe>>),
}

enum MotionEffectParameterValue {
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

pub struct MotionEffectParameter {
    parameter_id: i32,
    keyframe: MotionParentKeyframe,
    value: MotionEffectParameterValue,
}

impl MotionEffectParameter {
    fn create_from_accessory_keyframe(
        keyframe: Rc<RefCell<MotionAccessoryKeyframe>>,
    ) -> Result<MotionEffectParameter, Status> {
        Ok(MotionEffectParameter {
            parameter_id: 0,
            keyframe: MotionParentKeyframe::ACCESSORY(Rc::downgrade(&keyframe)),
            value: MotionEffectParameterValue::default(),
        })
    }

    fn create_from_model_keyframe(
        keyframe: Rc<RefCell<MotionModelKeyframe>>,
    ) -> Result<MotionEffectParameter, Status> {
        Ok(MotionEffectParameter {
            parameter_id: 0,
            keyframe: MotionParentKeyframe::MODEL(Rc::downgrade(&keyframe)),
            value: MotionEffectParameterValue::default(),
        })
    }
}
pub struct MotionOutsideParent {
    keyframe: MotionParentKeyframe,
    global_model_track_index: i32,
    global_bone_track_index: i32,
    local_bone_track_index: i32,
}

impl MotionOutsideParent {
    fn create_from_accessory_keyframe(
        keyframe: Rc<RefCell<MotionAccessoryKeyframe>>,
    ) -> Result<MotionOutsideParent, Status> {
        Ok(MotionOutsideParent {
            keyframe: MotionParentKeyframe::ACCESSORY(Rc::downgrade(&keyframe)),
            global_model_track_index: 0,
            global_bone_track_index: 0,
            local_bone_track_index: 0,
        })
    }
    fn create_from_camera_keyframe(
        keyframe: Rc<RefCell<MotionCameraKeyframe>>,
    ) -> Result<MotionOutsideParent, Status> {
        Ok(MotionOutsideParent {
            keyframe: MotionParentKeyframe::CAMERA(Rc::downgrade(&keyframe)),
            global_model_track_index: 0,
            global_bone_track_index: 0,
            local_bone_track_index: 0,
        })
    }
    fn create_from_model_keyframe(
        keyframe: Rc<RefCell<MotionModelKeyframe>>,
    ) -> Result<MotionOutsideParent, Status> {
        Ok(MotionOutsideParent {
            keyframe: MotionParentKeyframe::MODEL(Rc::downgrade(&keyframe)),
            global_model_track_index: 0,
            global_bone_track_index: 0,
            local_bone_track_index: 0,
        })
    }
}

struct MotionKeyframeBase {
    index: i32,
    frame_index: u32,
    is_selected: bool,
    user_data: Option<Rc<RefCell<UserData>>>,
    annotations: HashMap<String, String>,
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
    accessary_id: i32,
    outside_parent: Option<MotionOutsideParent>,
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
            accessary_id: 0,
            outside_parent: None,
        }
    }
}

const VMD_BONE_KEYFRAME_NAME_LENGTH: usize = 15;
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
    interplation: MotionBoneKeyframeInterpolation,
    bone_id: i32,
    stage_index: u32,
    is_physics_simulation_enabled: bool,
}

impl MotionBoneKeyframe {
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
            interplation: MotionBoneKeyframeInterpolation::default(),
            bone_id: 0,
            stage_index: 0,
            is_physics_simulation_enabled: true,
        };
        // try read utf8 name here
        let (str, len) = buffer.try_get_string_with_byte_len(VMD_BONE_KEYFRAME_NAME_LENGTH);
        let mut name: Option<String> = None;
        // if raw_name read not empty
        if len > 0 && str[0] != 0u8 {
            let key = u8_slice_get_string(str).unwrap_or_default();
            // try get bone_id from cache or create new motion track with name
            if let Some(bone_id) = cache.0.get(&key) {
                buffer.skip(len)?;
                name = parent_motion
                    .local_bone_motion_track_bundle
                    .resolve_name(bone_id.clone());
                bone_keyframe.bone_id = bone_id.clone();
            } else {
                let read_name = buffer.read_string_from_cp932(VMD_BONE_KEYFRAME_NAME_LENGTH)?;
                name = Some(read_name.clone());
                let (bone_id, found_name) = parent_motion.resolve_local_bone_track_id(read_name);
                bone_keyframe.bone_id = bone_id;
                if let Some(found_name) = found_name {
                    name = Some(found_name)
                }
                cache.0.insert(key, bone_keyframe.bone_id);
            }
        } else {
            buffer.skip(VMD_BONE_KEYFRAME_NAME_LENGTH)?;
            name = None;
        }
        bone_keyframe.base.frame_index = buffer.read_u32_little_endian()? + offset;
        bone_keyframe.translation = buffer.read_f32_3_little_endian()?;
        bone_keyframe.orientation = buffer.read_f32_4_little_endian()?;
        for i in 0..4 {
            bone_keyframe.interplation.translation_x[i] = buffer.read_byte()?;
            bone_keyframe.interplation.translation_y[i] = buffer.read_byte()?;
            bone_keyframe.interplation.translation_z[i] = buffer.read_byte()?;
            bone_keyframe.interplation.orientation[i] = buffer.read_byte()?;
        }
        buffer.skip(48usize)?;
        let bone_keyframe_index = bone_keyframe.base.frame_index;
        let rc = Rc::new(RefCell::new(bone_keyframe));
        parent_motion.local_bone_motion_track_bundle.add_keyframe(
            MotionKeyframeObject::Bone(rc.clone()),
            bone_keyframe_index,
            name.unwrap_or_default(),
        );
        Ok(rc)
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
    interplation: MotionCameraKeyframeInterpolation,
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
            interplation: MotionCameraKeyframeInterpolation::default(),
            is_perspective_view: false,
            stage_index: 0,
            outside_parent: None,
        };
        for i in 0..4 {
            camera_keyframe.interplation.lookat_x[i] = buffer.read_byte()?;
            camera_keyframe.interplation.lookat_y[i] = buffer.read_byte()?;
            camera_keyframe.interplation.lookat_z[i] = buffer.read_byte()?;
            camera_keyframe.interplation.angle[i] = buffer.read_byte()?;
            camera_keyframe.interplation.fov[i] = buffer.read_byte()?;
            camera_keyframe.interplation.distance[i] = buffer.read_byte()?;
        }
        camera_keyframe.fov = buffer.read_i32_little_endian()?;
        camera_keyframe.is_perspective_view = buffer.read_byte()? == 0u8;
        Ok(camera_keyframe)
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
}

const PMD_BONE_NAME_LENGTH: usize = 20;

pub struct MotionModelKeyframeConstraintState {
    bone_id: i32,
    enabled: bool,
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
                let name = buffer.read_string_from_cp932(PMD_BONE_NAME_LENGTH)?;
                let (bone_id, _) = parent_motion.resolve_local_bone_track_id(name);
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
}

const VMD_MORPH_KEYFRAME_NAME_LENGTH: usize = 15;

pub struct MotionMorphKeyframe {
    base: MotionKeyframeBase,
    weight: f32,
    morph_id: i32,
}

impl MotionMorphKeyframe {
    fn parse_vmd(
        parent_motion: &mut Motion,
        buffer: &mut Buffer,
        cache: &mut StringCache,
        offset: u32,
    ) -> Result<Rc<RefCell<MotionMorphKeyframe>>, Status> {
        let mut keyframe_morph_id = 0;
        let (str, _) = buffer.try_get_string_with_byte_len(VMD_MORPH_KEYFRAME_NAME_LENGTH);
        let mut name: Option<String> = None;
        if str.len() > 0 && str[0] != 0u8 {
            let key = u8_slice_get_string(str).unwrap_or_default();
            if let Some(morph_id) = cache.0.get(&key) {
                buffer.skip(VMD_MORPH_KEYFRAME_NAME_LENGTH)?;
                name = parent_motion
                    .local_morph_motion_track_bundle
                    .resolve_name(*morph_id);
                keyframe_morph_id = *morph_id;
            } else {
                name = Some(buffer.read_string_from_cp932(VMD_MORPH_KEYFRAME_NAME_LENGTH)?);
                let (morph_id, found_name) =
                    parent_motion.resolve_local_morph_track_id(name.clone().unwrap_or_default());
                keyframe_morph_id = morph_id;
                if let Some(found_name) = found_name {
                    name = Some(found_name);
                }
                cache.0.insert(key, keyframe_morph_id);
            }
        } else {
            buffer.skip(VMD_MORPH_KEYFRAME_NAME_LENGTH)?;
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
            name.unwrap_or_default(),
        );
        Ok(rc)
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
}

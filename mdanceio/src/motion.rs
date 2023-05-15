use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use cgmath::{Quaternion, Vector1, Vector2, Vector4, VectorSpace};
use nanoem::{
    common::NanoemError,
    motion::{
        MotionAccessoryKeyframe, MotionBoneKeyframe, MotionBoneKeyframeInterpolation,
        MotionCameraKeyframe, MotionKeyframeBase, MotionLightKeyframe, MotionModelKeyframe,
        MotionModelKeyframeConstraintState, MotionMorphKeyframe, MotionSelfShadowKeyframe,
    },
};

use crate::{
    bezier_curve::BezierCurve,
    camera::PerspectiveCamera,
    error::MdanceioError,
    light::{DirectionalLight, Light},
    model::{Bone, Model},
    motion_keyframe_selection::MotionKeyframeSelection,
    project::Project,
    shadow_camera::ShadowCamera,
    utils::{f128_to_quat, f128_to_vec4},
};

pub type NanoemMotion = nanoem::motion::Motion;

pub struct KeyframeBound {
    pub previous: Option<u32>,
    pub current: u32,
    pub next: Option<u32>,
}

pub struct BoneKeyframeBezierControlPointParameter {
    pub translation_x: Vector4<u8>,
    pub translation_y: Vector4<u8>,
    pub translation_z: Vector4<u8>,
    pub orientation: Vector4<u8>,
}

pub struct BoneKeyframeTranslationBezierControlPointParameter {
    pub translation_x: Vector4<u8>,
    pub translation_y: Vector4<u8>,
    pub translation_z: Vector4<u8>,
}

pub struct BoneKeyframeState {
    pub translation: Vector4<f32>,
    pub orientation: Quaternion<f32>,
    pub stage_index: u32,
    pub bezier_param: BoneKeyframeBezierControlPointParameter,
    pub enable_physics_simulation: bool,
}

impl BoneKeyframeState {
    fn from_bone(bone: &Bone, enable_physics_simulation: bool) -> Self {
        Self {
            translation: bone.local_user_translation.extend(1f32),
            orientation: bone.local_user_orientation,
            stage_index: 0,
            bezier_param: BoneKeyframeBezierControlPointParameter {
                translation_x: bone.interpolation.translation_x.bezier_control_point,
                translation_y: bone.interpolation.translation_y.bezier_control_point,
                translation_z: bone.interpolation.translation_z.bezier_control_point,
                orientation: bone.interpolation.orientation.bezier_control_point,
            },
            enable_physics_simulation,
        }
    }

    fn from_keyframe(keyframe: &MotionBoneKeyframe) -> Self {
        Self {
            translation: f128_to_vec4(keyframe.translation),
            orientation: f128_to_quat(keyframe.orientation),
            stage_index: keyframe.stage_index,
            bezier_param: BoneKeyframeBezierControlPointParameter {
                translation_x: keyframe.interpolation.translation_x.into(),
                translation_y: keyframe.interpolation.translation_y.into(),
                translation_z: keyframe.interpolation.translation_z.into(),
                orientation: keyframe.interpolation.orientation.into(),
            },
            enable_physics_simulation: keyframe.is_physics_simulation_enabled,
        }
    }
}

pub struct BoneKeyframeOverrideInterpolation {
    pub target_frame_index: u32,
    pub translation_params: (
        BoneKeyframeTranslationBezierControlPointParameter,
        BoneKeyframeTranslationBezierControlPointParameter,
    ),
}

pub struct BoneKeyframeUpdater {
    pub name: String,
    pub state: (BoneKeyframeState, Option<BoneKeyframeState>),
    pub bezier_curve_override: Option<BoneKeyframeOverrideInterpolation>,
    pub was_dirty: bool,
    pub frame_index: u32,
}

impl BoneKeyframeUpdater {
    pub fn updated(&self) -> bool {
        self.state.1.is_some()
    }

    pub fn selected(&self) -> bool {
        self.state.1.is_none()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CurveCacheKey {
    next: [u8; 4],
    interval: u32,
}

#[derive(Debug, Clone)]
pub struct Motion {
    pub opaque: NanoemMotion,
    bezier_curves_data: RefCell<HashMap<CurveCacheKey, Box<BezierCurve>>>,
    annotations: HashMap<String, String>,
    pub dirty: bool,
}

impl Motion {
    pub const NMD_FORMAT_EXTENSION: &'static str = "nmd";
    pub const VMD_FORMAT_EXTENSION: &'static str = "vmd";
    pub const CAMERA_AND_LIGHT_TARGET_MODEL_NAME: &'static str = "カァラ・照明";
    pub const CAMERA_AND_LIGHT_TARGET_MODEL_NAME_BYTES: &'static [u8] = &[
        0xe3, 0x82, 0xab, 0xe3, 0x83, 0xa1, 0xe3, 0x83, 0xa9, 0xe3, 0x83, 0xbb, 0xe7, 0x85, 0xa7,
        0xe6, 0x98, 0x8e, 0,
    ];
    pub const MAX_KEYFRAME_INDEX: u32 = u32::MAX;

    pub fn new_from_bytes(bytes: &[u8], offset: u32) -> Result<Self, MdanceioError> {
        let mut buffer = nanoem::common::Buffer::create(bytes);
        match NanoemMotion::load_from_buffer(&mut buffer, offset) {
            Ok(motion) => Ok(Self {
                opaque: motion,
                bezier_curves_data: RefCell::new(HashMap::new()),
                annotations: HashMap::new(),
                dirty: false,
            }),
            Err(status) => Err(MdanceioError::from_nanoem(
                "Cannot load the motion: ",
                status,
            )),
        }
    }

    pub fn empty() -> Self {
        Self {
            // selection: (),
            opaque: NanoemMotion::empty(),
            bezier_curves_data: RefCell::new(HashMap::new()),
            annotations: HashMap::new(),
            // file_uri: (),
            // format_type: (),
            dirty: false,
        }
    }

    pub fn initialize_model_frame_0(&mut self, model: &Model) {
        for bone in model.bones() {
            if self.find_bone_keyframe(&bone.canonical_name, 0).is_none() {
                let _ = self.opaque.local_bone_motion_track_bundle.insert_keyframe(
                    MotionBoneKeyframe {
                        base: MotionKeyframeBase {
                            frame_index: 0,
                            annotations: HashMap::new(),
                        },
                        translation: bone.local_user_translation.extend(1f32).into(),
                        orientation: bone.local_user_orientation.into(),
                        interpolation: nanoem::motion::MotionBoneKeyframeInterpolation {
                            translation_x: Bone::DEFAULT_BEZIER_CONTROL_POINT,
                            translation_y: Bone::DEFAULT_BEZIER_CONTROL_POINT,
                            translation_z: Bone::DEFAULT_BEZIER_CONTROL_POINT,
                            orientation: Bone::DEFAULT_BEZIER_CONTROL_POINT,
                        },
                        stage_index: 0,
                        is_physics_simulation_enabled: true,
                    },
                    &bone.canonical_name,
                );
            }
        }
        for morph in model.morphs() {
            if self.find_morph_keyframe(&morph.canonical_name, 0).is_none() {
                let _ = self.opaque.local_morph_motion_track_bundle.insert_keyframe(
                    MotionMorphKeyframe {
                        base: MotionKeyframeBase {
                            frame_index: 0,
                            annotations: HashMap::new(),
                        },
                        weight: morph.weight(),
                    },
                    &morph.canonical_name,
                );
            }
        }
        if self.find_model_keyframe(0).is_none() {
            let constraint_states = model
                .constraints()
                .iter()
                .filter_map(|constraint| {
                    usize::try_from(constraint.origin.target_bone_index)
                        .ok()
                        .and_then(|idx| model.bones().get(idx))
                        .map(|bone| &bone.canonical_name)
                        .and_then(|name| {
                            self.opaque
                                .local_bone_motion_track_bundle
                                .resolve_name(name)
                        })
                        .map(|track_id| MotionModelKeyframeConstraintState {
                            bone_id: track_id,
                            enabled: true,
                        })
                })
                .collect::<Vec<_>>();
            let _ = self.opaque.add_model_keyframe(MotionModelKeyframe {
                base: MotionKeyframeBase {
                    frame_index: 0,
                    annotations: HashMap::new(),
                },
                visible: true,
                constraint_states,
                effect_parameters: vec![],
                outside_parents: vec![],
                has_edge_option: false,
                edge_scale_factor: 1f32,
                edge_color: [0f32, 0f32, 0f32, 1f32],
                is_add_blending_enabled: false,
                is_physics_simulation_enabled: true,
            });
        }
    }

    pub fn initialize_camera_frame_0(
        &mut self,
        camera: &PerspectiveCamera,
        active_model: Option<&Model>,
    ) {
        if self.find_camera_keyframe(0).is_none() {
            let _ = self.opaque.add_camera_keyframe(MotionCameraKeyframe {
                base: MotionKeyframeBase {
                    frame_index: 0,
                    annotations: HashMap::new(),
                },
                look_at: camera.look_at(active_model).extend(0f32).into(),
                angle: camera.angle().extend(0f32).into(),
                distance: -camera.distance(),
                fov: camera.fov(),
                interpolation: nanoem::motion::MotionCameraKeyframeInterpolation::default(),
                is_perspective_view: camera.is_perspective(),
                stage_index: 0,
                outside_parent: None,
            });
        }
    }

    pub fn initialize_light_frame_0(&mut self, light: &DirectionalLight) {
        if self.find_light_keyframe(0).is_none() {
            let _ = self.opaque.add_light_keyframe(MotionLightKeyframe {
                base: MotionKeyframeBase {
                    frame_index: 0,
                    annotations: HashMap::new(),
                },
                color: light.color().extend(0f32).into(),
                direction: light.direction().extend(0f32).into(),
            });
        }
    }

    pub fn initialize_self_shadow_frame_0(&mut self, shadow: &ShadowCamera) {
        if self.find_self_shadow_keyframe(0).is_none() {
            let _ = self
                .opaque
                .add_self_shadow_keyframe(MotionSelfShadowKeyframe {
                    base: MotionKeyframeBase {
                        frame_index: 0,
                        annotations: HashMap::new(),
                    },
                    distance: shadow.distance(),
                    mode: u32::from(shadow.coverage_mode()) as i32,
                });
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

    pub fn add_frame_index_delta(value: i32, frame_index: u32) -> Option<u32> {
        if value > 0 {
            if frame_index <= Self::MAX_KEYFRAME_INDEX - value as u32 {
                return Some(frame_index + (value as u32));
            }
        } else if value < 0 && frame_index >= value.unsigned_abs() {
            return Some(frame_index - value.unsigned_abs());
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
    ) -> Result<(), NanoemError> {
        for keyframe in keyframes {
            let _frame_index = keyframe.frame_index_with_offset(offset);
            let mut n_keyframe = keyframe.clone();
            keyframe.copy_outside_parent(target, &mut n_keyframe);
            let _ = target.add_accessory_keyframe(n_keyframe);
        }
        Ok(())
    }

    pub fn copy_all_accessory_keyframes_from_motion(
        source: &NanoemMotion,
        target: &mut NanoemMotion,
        offset: i32,
    ) -> Result<(), NanoemError> {
        Self::copy_all_accessory_keyframes(
            &source
                .get_all_accessory_keyframe_objects()
                .cloned()
                .collect::<Vec<_>>(),
            target,
            offset,
        )
    }

    pub fn merge_all_keyframes(&mut self, source: &Motion) {
        self.internal_merge_all_keyframes(source, false, false);
    }

    fn internal_merge_all_keyframes(&mut self, source: &Motion, overrid: bool, reverse: bool) {
        let mut merger = Merger {
            source: &source.opaque,
            overrid,
            dest: &mut self.opaque,
        };
        // TODO: handle accessory keyframes
        merger.merge_all_bone_keyframes(reverse);
        merger.merge_all_camera_keyframes();
        merger.merge_all_light_keyframes();
        merger.merge_all_model_keyframes();
        merger.merge_all_morph_keyframes();
        merger.merge_all_self_shadow_keyframes();
        self.dirty = true;
    }

    pub fn build_bone_keyframe_updater(
        &self,
        bone: &Bone,
        bound: &KeyframeBound,
        enable_bezier_curve_adjustment: bool,
        enable_physics_simulation: bool,
    ) -> BoneKeyframeUpdater {
        let name = bone.name.clone();
        let mut new_state = BoneKeyframeState::from_bone(bone, enable_physics_simulation);
        let old_state;
        let bezier_curve_override;
        if let Some(keyframe) = self.find_bone_keyframe(&name, bound.current) {
            old_state = Some(BoneKeyframeState::from_keyframe(keyframe));
            bezier_curve_override = None;
        } else {
            old_state = None;
            if let Some(prev_frame_index) = bound.previous {
                let prev_keyframe = self.find_bone_keyframe(&name, prev_frame_index).unwrap();
                let movable = bone.origin.flags.is_movable;
                let prev_interpolation_translation_x: Vector4<u8> =
                    prev_keyframe.interpolation.translation_x.into();
                let prev_interpolation_translation_y: Vector4<u8> =
                    prev_keyframe.interpolation.translation_y.into();
                let prev_interpolation_translation_z: Vector4<u8> =
                    prev_keyframe.interpolation.translation_z.into();
                let mut get_new_interpolation = |prev_interpolation_value: Vector4<u8>| {
                    if enable_bezier_curve_adjustment && movable {
                        if KeyframeInterpolationPoint::is_linear_interpolation(
                            &prev_keyframe.interpolation.translation_x,
                        ) {
                            Bone::DEFAULT_AUTOMATIC_BAZIER_CONTROL_POINT.into()
                        } else if bound.next.is_some() && bound.next.unwrap() > prev_frame_index {
                            let next_frame_index = bound.next.unwrap();
                            let interval = next_frame_index - prev_frame_index;
                            let c0 = [prev_interpolation_value[0], prev_interpolation_value[1]];
                            let c1 = [prev_interpolation_value[2], prev_interpolation_value[3]];
                            let bezier_curve =
                                BezierCurve::create(&c0.into(), &c1.into(), interval);
                            let amount =
                                (bound.current - prev_frame_index) as f32 / (interval as f32);
                            let pair = bezier_curve.split(amount);
                            new_state.bezier_param.translation_x = pair.1.to_parameters();
                            pair.0.to_parameters()
                        } else {
                            prev_interpolation_value
                        }
                    } else {
                        prev_interpolation_value
                    }
                };
                let new_interpolation_translation_x =
                    get_new_interpolation(prev_interpolation_translation_x);
                let new_interpolation_translation_y =
                    get_new_interpolation(prev_interpolation_translation_y);
                let new_interpolation_translation_z =
                    get_new_interpolation(prev_interpolation_translation_z);
                let bezier_curve_override_target_frame_index = prev_frame_index;
                bezier_curve_override = Some(BoneKeyframeOverrideInterpolation {
                    target_frame_index: prev_frame_index,
                    translation_params: (
                        BoneKeyframeTranslationBezierControlPointParameter {
                            translation_x: new_interpolation_translation_x,
                            translation_y: new_interpolation_translation_y,
                            translation_z: new_interpolation_translation_z,
                        },
                        BoneKeyframeTranslationBezierControlPointParameter {
                            translation_x: prev_interpolation_translation_x,
                            translation_y: prev_interpolation_translation_y,
                            translation_z: prev_interpolation_translation_z,
                        },
                    ),
                })
            } else {
                bezier_curve_override = None
            }
        }
        BoneKeyframeUpdater {
            name,
            state: (new_state, old_state),
            bezier_curve_override,
            was_dirty: false,
            frame_index: bound.current,
        }
    }

    pub fn build_add_bone_keyframes_updaters(
        &self,
        model: &Model,
        bones: &HashMap<String, Vec<u32>>,
        enable_bezier_curve_adjustment: bool,
        enable_physics_simulation: bool,
    ) -> Vec<BoneKeyframeUpdater> {
        if bones.is_empty() {
            // (m_project->isPlaying() || m_project->isModelEditingEnabled())
            return vec![];
        }
        let mut updaters = vec![];
        for (name, frame_indices) in bones {
            if let Some(bone) = model.find_bone(name) {
                for frame_index in frame_indices {
                    let (prev, next) = self
                        .opaque
                        .search_closest_bone_keyframes(name, *frame_index);
                    updaters.push(self.build_bone_keyframe_updater(
                        bone,
                        &KeyframeBound {
                            previous: prev.map(|frame| frame.base.frame_index),
                            current: *frame_index,
                            next: next.map(|frame| frame.base.frame_index),
                        },
                        enable_bezier_curve_adjustment,
                        enable_physics_simulation,
                    ));
                }
            }
        }
        updaters
    }

    pub fn apply_add_bone_keyframes_updaters(
        &mut self,
        model: &mut Model,
        updaters: &mut [BoneKeyframeUpdater],
    ) {
        let last_duration = self.duration();
        for updater in updaters {
            let old = self.opaque.local_bone_motion_track_bundle.insert_keyframe(
                MotionBoneKeyframe {
                    base: MotionKeyframeBase {
                        frame_index: updater.frame_index,
                        annotations: HashMap::new(),
                    },
                    translation: updater.state.0.translation.into(),
                    orientation: updater.state.0.orientation.into(),
                    interpolation: MotionBoneKeyframeInterpolation {
                        translation_x: updater.state.0.bezier_param.translation_x.into(),
                        translation_y: updater.state.0.bezier_param.translation_y.into(),
                        translation_z: updater.state.0.bezier_param.translation_z.into(),
                        orientation: updater.state.0.bezier_param.orientation.into(),
                    },
                    stage_index: updater.state.0.stage_index,
                    is_physics_simulation_enabled: updater.state.0.enable_physics_simulation,
                },
                &updater.name,
            );
            if old.is_none() && updater.updated() {
                log::warn!("No existing keyframe when update bone keyframe")
            }
            if !updater.updated() && updater.selected() {
                // TODO: add keyframe to motion selection
            }
            if let Some(bone) = model.find_bone_mut(&updater.name) {
                if bone.states.dirty {
                    updater.was_dirty = true;
                    bone.states.dirty = false;
                }
            }
        }
        self.set_dirty(true);
        let current_duration = self.duration();
        if last_duration != current_duration {
            // TODO: publish duration updated event
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn set_dirty(&mut self, value: bool) {
        self.dirty = value;
    }

    pub fn duration(&self) -> u32 {
        self.opaque
            .max_frame_index()
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
        for (bone_name, track) in &self.opaque.local_bone_motion_track_bundle.tracks {
            if !model.contains_bone(bone_name) && track.len() > 1 {
                bones.push(bone_name.clone());
            }
        }
        for (morph_name, track) in &self.opaque.local_morph_motion_track_bundle.tracks {
            if !model.contains_morph(morph_name) && track.len() > 1 {
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
        result.translation = [
            -origin.translation[0],
            origin.translation[1],
            origin.translation[2],
            0f32,
        ];
        result.orientation = [
            origin.orientation[0],
            -origin.orientation[1],
            -origin.orientation[2],
            origin.orientation[3],
        ];
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
            let _ = self
                .dest
                .local_bone_motion_track_bundle
                .insert_keyframe(new_frame, &new_name);
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
        for (frame_index, keyframe) in &self.source.camera_keyframes.keyframes {
            if self.overrid
                || self
                    .dest
                    .find_camera_keyframe_object(*frame_index)
                    .is_none()
            {
                let _ = self.dest.add_camera_keyframe(keyframe.clone());
            }
        }
    }

    pub fn merge_all_light_keyframes(&mut self) {
        for (frame_index, keyframe) in &self.source.light_keyframes.keyframes {
            if self.overrid || self.dest.find_light_keyframe_object(*frame_index).is_none() {
                let _ = self.dest.add_light_keyframe(keyframe.clone());
            }
        }
    }

    pub fn merge_all_model_keyframes(&mut self) {
        for (frame_index, keyframe) in &self.source.model_keyframes.keyframes {
            if self.overrid || self.dest.find_model_keyframe_object(*frame_index).is_none() {
                let _ = self.dest.add_model_keyframe(keyframe.clone());
            }
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
                        .insert_keyframe(keyframe.clone(), name);
                }
            }
        }
    }

    pub fn merge_all_self_shadow_keyframes(&mut self) {
        for (frame_index, keyframe) in &self.source.self_shadow_keyframes.keyframes {
            if self.overrid
                || self
                    .dest
                    .find_self_shadow_keyframe_object(*frame_index)
                    .is_none()
            {
                let _ = self.dest.add_self_shadow_keyframe(keyframe.clone());
            }
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

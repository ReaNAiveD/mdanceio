use std::collections::{HashMap, HashSet};

use cgmath::{Deg, One, Quaternion, Rad, Vector1, Vector2, Vector3, Vector4, VectorSpace, Zero};
use nanoem::{
    common::NanoemError,
    motion::{
        MotionAccessoryKeyframe, MotionBoneKeyframe, MotionBoneKeyframeInterpolation,
        MotionCameraKeyframe, MotionCameraKeyframeInterpolation, MotionKeyframeBase,
        MotionLightKeyframe, MotionModelKeyframe, MotionModelKeyframeConstraintState,
        MotionMorphKeyframe, MotionSelfShadowKeyframe, MotionTrack,
    },
};

use crate::{
    bezier_curve::{BezierCurve, BezierCurveCache, BezierCurveFactory, Curve},
    camera::PerspectiveCamera,
    error::MdanceioError,
    light::{DirectionalLight, Light},
    model::{Bone, Model},
    project::Project,
    shadow_camera::{CoverageMode, ShadowCamera},
    utils::{f128_to_quat, f128_to_vec3, f128_to_vec4, lerp_element_wise, lerp_f32, lerp_rad},
};

pub type NanoemMotion = nanoem::motion::Motion;

trait Seek {
    type Frame;

    fn find(&self, frame_index: u32) -> Option<Self::Frame>;

    fn seek(&self, frame_index: u32, curve_factory: &dyn BezierCurveFactory) -> Self::Frame;

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame;
}

#[derive(Debug, Clone, Copy)]
pub struct BoneKeyframeInterpolation {
    pub translation: Vector3<KeyframeInterpolationPoint>,
    pub orientation: KeyframeInterpolationPoint,
}

impl Default for BoneKeyframeInterpolation {
    fn default() -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::default(),
                y: KeyframeInterpolationPoint::default(),
                z: KeyframeInterpolationPoint::default(),
            },
            orientation: KeyframeInterpolationPoint::default(),
        }
    }
}

impl BoneKeyframeInterpolation {
    pub fn zero() -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::zero(),
                y: KeyframeInterpolationPoint::zero(),
                z: KeyframeInterpolationPoint::zero(),
            },
            orientation: KeyframeInterpolationPoint::zero(),
        }
    }

    pub fn build(interpolation: MotionBoneKeyframeInterpolation) -> Self {
        Self {
            translation: Vector3 {
                x: KeyframeInterpolationPoint::new(&interpolation.translation_x),
                y: KeyframeInterpolationPoint::new(&interpolation.translation_y),
                z: KeyframeInterpolationPoint::new(&interpolation.translation_z),
            },
            orientation: KeyframeInterpolationPoint::new(&interpolation.orientation),
        }
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        BoneKeyframeInterpolation {
            translation: Vector3 {
                x: self.translation.x.lerp(other.translation.x, amount),
                y: self.translation.y.lerp(other.translation.y, amount),
                z: self.translation.z.lerp(other.translation.z, amount),
            },
            orientation: self.orientation.lerp(other.orientation, amount),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoneFrameTransform {
    pub translation: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub interpolation: BoneKeyframeInterpolation,
    pub local_transform_mix: Option<f32>,
    pub enable_physics: bool,
    pub disable_physics: bool,
}

impl BoneFrameTransform {
    pub fn mixed_translation(&self, local_user_translation: Vector3<f32>) -> Vector3<f32> {
        if let Some(coef) = self.local_transform_mix {
            local_user_translation.lerp(self.translation, coef)
        } else {
            self.translation
        }
    }

    pub fn mixed_orientation(&self, local_user_orientation: Quaternion<f32>) -> Quaternion<f32> {
        if let Some(coef) = self.local_transform_mix {
            local_user_orientation.slerp(self.orientation, coef)
        } else {
            self.orientation
        }
    }
}

impl Default for BoneFrameTransform {
    fn default() -> Self {
        Self {
            translation: Vector3::zero(),
            orientation: Quaternion::one(),
            interpolation: BoneKeyframeInterpolation::default(),
            local_transform_mix: None,
            enable_physics: false,
            disable_physics: false,
        }
    }
}

impl Seek for MotionTrack<MotionBoneKeyframe> {
    type Frame = BoneFrameTransform;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| BoneFrameTransform {
                translation: f128_to_vec3(keyframe.translation),
                orientation: f128_to_quat(keyframe.orientation),
                interpolation: BoneKeyframeInterpolation::build(keyframe.interpolation),
                local_transform_mix: None,
                enable_physics: keyframe.is_physics_simulation_enabled,
                disable_physics: false,
            })
    }

    fn seek(&self, frame_index: u32, bezier_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let next_translation = f128_to_vec3(next_frame.translation);
            let next_orientation = f128_to_quat(next_frame.orientation);
            let prev_enabled = prev_frame.is_physics_simulation_enabled;
            let next_enabled = next_frame.is_physics_simulation_enabled;
            if prev_enabled && !next_enabled {
                BoneFrameTransform {
                    translation: next_translation,
                    orientation: next_orientation,
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    local_transform_mix: Some(coef),
                    enable_physics: false,
                    disable_physics: true,
                }
            } else {
                let prev_translation = f128_to_vec3(prev_frame.translation);
                let prev_orientation = f128_to_quat(prev_frame.orientation);
                let translation_interpolation = Vector3::new(
                    &next_frame.interpolation.translation_x,
                    &next_frame.interpolation.translation_y,
                    &next_frame.interpolation.translation_z,
                )
                .map(KeyframeInterpolationPoint::new);
                let amounts = translation_interpolation
                    .map(|interpolation| interpolation.curve_value(interval, coef, bezier_factory));
                let translation = lerp_element_wise(prev_translation, next_translation, amounts);
                let orientation_interpolation =
                    KeyframeInterpolationPoint::new(&next_frame.interpolation.orientation);
                let amount = orientation_interpolation.curve_value(interval, coef, bezier_factory);
                let orientation = prev_orientation.slerp(next_orientation, amount);
                BoneFrameTransform {
                    translation,
                    orientation,
                    interpolation: BoneKeyframeInterpolation::build(next_frame.interpolation),
                    local_transform_mix: None,
                    enable_physics: prev_enabled && next_enabled,
                    disable_physics: false,
                }
            }
        } else {
            BoneFrameTransform::default()
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        let f0 = Self::seek(self, frame_index, curve_factory);
        if amount > 0f32 {
            let f1 = Self::seek(self, frame_index + 1, curve_factory);
            let local_transform_mix = match (f0.local_transform_mix, f1.local_transform_mix) {
                (Some(a0), Some(a1)) => Some(lerp_f32(a0, a1, amount)),
                (None, Some(a1)) => Some(amount * a1),
                (Some(a0), None) => Some((1f32 - amount) * a0),
                _ => None,
            };
            BoneFrameTransform {
                translation: f0.translation.lerp(f1.translation, amount),
                orientation: f0.orientation.slerp(f1.orientation, amount),
                interpolation: f0.interpolation.lerp(f1.interpolation, amount),
                local_transform_mix,
                enable_physics: f0.enable_physics && f1.enable_physics,
                disable_physics: f0.disable_physics || f1.disable_physics,
            }
        } else {
            f0
        }
    }
}

impl Seek for MotionTrack<MotionMorphKeyframe> {
    type Frame = f32;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| keyframe.weight)
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            lerp_f32(prev_frame.weight, next_frame.weight, coef)
        } else {
            0f32
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        let w0 = self.seek(frame_index, curve_factory);
        if amount > 0f32 {
            let w1 = self.seek(frame_index, curve_factory);
            lerp_f32(w0, w1, amount)
        } else {
            w0
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraKeyframeInterpolation {
    pub lookat: Vector3<KeyframeInterpolationPoint>,
    pub angle: KeyframeInterpolationPoint,
    pub fov: KeyframeInterpolationPoint,
    pub distance: KeyframeInterpolationPoint,
}

impl Default for CameraKeyframeInterpolation {
    fn default() -> Self {
        Self {
            lookat: [KeyframeInterpolationPoint::default(); 3].into(),
            angle: KeyframeInterpolationPoint::default(),
            fov: KeyframeInterpolationPoint::default(),
            distance: KeyframeInterpolationPoint::default(),
        }
    }
}

impl From<MotionCameraKeyframeInterpolation> for CameraKeyframeInterpolation {
    fn from(v: MotionCameraKeyframeInterpolation) -> Self {
        Self {
            lookat: Vector3::new(v.lookat_x.into(), v.lookat_y.into(), v.lookat_z.into()),
            angle: v.angle.into(),
            fov: v.fov.into(),
            distance: v.distance.into(),
        }
    }
}

impl CameraKeyframeInterpolation {
    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        Self {
            lookat: Vector3::new(
                self.lookat.x.lerp(other.lookat.x, amount),
                self.lookat.y.lerp(other.lookat.y, amount),
                self.lookat.z.lerp(other.lookat.z, amount),
            ),
            angle: self.angle.lerp(other.angle, amount),
            fov: self.fov.lerp(other.fov, amount),
            distance: self.distance.lerp(other.distance, amount),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraTransform {
    pub lookat: Vector3<f32>,
    pub angle: Vector3<f32>,
    pub fov: Rad<f32>,
    pub distance: f32,
    pub perspective: bool,
    pub outside_parent: Option<(i32, i32)>,
    pub interpolation: CameraKeyframeInterpolation,
}

impl From<&MotionCameraKeyframe> for CameraTransform {
    fn from(v: &MotionCameraKeyframe) -> Self {
        Self {
            lookat: f128_to_vec3(v.look_at),
            angle: f128_to_vec3(v.angle),
            fov: Deg(v.fov as f32).into(),
            distance: v.distance,
            perspective: v.is_perspective_view,
            outside_parent: v
                .outside_parent
                .map(|op| (op.global_model_track_index, op.global_bone_track_index)),
            interpolation: v.interpolation.into(),
        }
    }
}

impl Seek for MotionTrack<MotionCameraKeyframe> {
    type Frame = Option<CameraTransform>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, bezier_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let interval = next_frame.base.frame_index - prev_frame.base.frame_index;
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let prev_interpolation: CameraKeyframeInterpolation = prev_frame.interpolation.into();
            let interpolation: CameraKeyframeInterpolation = next_frame.interpolation.into();
            let lookat_amount = interpolation
                .lookat
                .map(|v| v.curve_value(interval, coef, bezier_factory));
            let lookat = lerp_element_wise(
                f128_to_vec3(prev_frame.look_at),
                f128_to_vec3(next_frame.look_at),
                lookat_amount,
            );
            let angle = f128_to_vec3(prev_frame.angle).lerp(
                f128_to_vec3(next_frame.angle),
                interpolation
                    .angle
                    .curve_value(interval, coef, bezier_factory),
            );
            let prev_fov: Rad<f32> = Deg(prev_frame.fov as f32).into();
            let next_fov: Rad<f32> = Deg(prev_frame.fov as f32).into();
            let fov = Rad(lerp_f32(prev_fov.0, next_fov.0, coef));
            let distance = lerp_f32(
                prev_frame.distance,
                next_frame.distance,
                interpolation
                    .distance
                    .curve_value(interval, coef, bezier_factory),
            );
            let perspective = prev_frame.is_perspective_view;
            let outside_parent = prev_frame
                .outside_parent
                .map(|op| (op.global_model_track_index, op.global_bone_track_index));
            Some(CameraTransform {
                lookat,
                angle,
                fov,
                distance,
                perspective,
                outside_parent,
                interpolation: prev_interpolation.lerp(interpolation, coef),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(CameraTransform {
                        lookat: frame0.lookat.lerp(frame1.lookat, amount),
                        angle: frame0.angle.lerp(frame1.angle, amount),
                        fov: lerp_rad(frame0.fov, frame1.fov, amount),
                        distance: lerp_f32(frame0.distance, frame1.distance, amount),
                        perspective: frame0.perspective,
                        outside_parent: frame0.outside_parent,
                        interpolation: frame0.interpolation,
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LightFrame {
    pub color: Vector3<f32>,
    pub direction: Vector3<f32>,
}

impl From<&MotionLightKeyframe> for LightFrame {
    fn from(v: &MotionLightKeyframe) -> Self {
        Self {
            color: f128_to_vec3(v.color),
            direction: f128_to_vec3(v.direction),
        }
    }
}

impl Seek for MotionTrack<MotionLightKeyframe> {
    type Frame = Option<LightFrame>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            let prev: LightFrame = prev_frame.into();
            let next: LightFrame = next_frame.into();
            Some(LightFrame {
                color: prev.color.lerp(next.color, coef),
                direction: prev.direction.lerp(next.direction, coef),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(LightFrame {
                        color: frame0.color.lerp(frame1.color, amount),
                        direction: frame0.direction.lerp(frame1.direction, amount),
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SelfShadowParam {
    pub distance: f32,
    pub coverage: CoverageMode,
}

impl From<&MotionSelfShadowKeyframe> for SelfShadowParam {
    fn from(v: &MotionSelfShadowKeyframe) -> Self {
        Self {
            distance: v.distance,
            coverage: (v.mode as u32).into(),
        }
    }
}

impl Seek for MotionTrack<MotionSelfShadowKeyframe> {
    type Frame = Option<SelfShadowParam>;

    fn find(&self, frame_index: u32) -> Option<Self::Frame> {
        self.keyframes
            .get(&frame_index)
            .map(|keyframe| Some(keyframe.into()))
    }

    fn seek(&self, frame_index: u32, _curve_factory: &dyn BezierCurveFactory) -> Self::Frame {
        if let Some(frame) = self.find(frame_index) {
            frame
        } else if let (Some(prev_frame), Some(next_frame)) = self.search_closest(frame_index) {
            let coef = Motion::coefficient(
                prev_frame.base.frame_index,
                next_frame.base.frame_index,
                frame_index,
            );
            Some(SelfShadowParam {
                distance: lerp_f32(prev_frame.distance, next_frame.distance, coef),
                coverage: (prev_frame.mode as u32).into(),
            })
        } else {
            None
        }
    }

    fn seek_precisely(
        &self,
        frame_index: u32,
        amount: f32,
        curve_factory: &dyn BezierCurveFactory,
    ) -> Self::Frame {
        if let Some(frame0) = self.seek(frame_index, curve_factory) {
            if amount > 0f32 {
                if let Some(frame1) = self.seek(frame_index, curve_factory) {
                    Some(SelfShadowParam {
                        distance: lerp_f32(frame0.distance, frame1.distance, amount),
                        coverage: frame0.coverage,
                    })
                } else {
                    Some(frame0)
                }
            } else {
                Some(frame0)
            }
        } else {
            None
        }
    }
}

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
    pub x: Vector4<u8>,
    pub y: Vector4<u8>,
    pub z: Vector4<u8>,
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
                translation_x: bone
                    .interpolation
                    .translation
                    .x
                    .bezier_control_point()
                    .into(),
                translation_y: bone
                    .interpolation
                    .translation
                    .y
                    .bezier_control_point()
                    .into(),
                translation_z: bone
                    .interpolation
                    .translation
                    .z
                    .bezier_control_point()
                    .into(),
                orientation: bone.interpolation.orientation.bezier_control_point().into(),
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
    // Will Get a new Empty bundle when clone
    bezier_cache: BezierCurveCache,
    pub dirty: bool,
}

impl Motion {
    pub const NMD_FORMAT_EXTENSION: &'static str = "nmd";
    pub const VMD_FORMAT_EXTENSION: &'static str = "vmd";
    pub const CAMERA_AND_LIGHT_TARGET_MODEL_NAME: &'static str = "カメラ・照明";
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
                bezier_cache: BezierCurveCache::new(),
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
            bezier_cache: BezierCurveCache::new(),
            // annotations: HashMap::new(),
            // file_uri: (),
            // format_type: (),
            dirty: false,
        }
    }

    pub fn initialize_model_frame_0(&mut self, model: &Model) {
        for bone in model.bones().iter() {
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
                .bones()
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
                            let bezier_curve =
                                BezierCurve::from_parameters(prev_interpolation_value, interval);
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
                            x: new_interpolation_translation_x,
                            y: new_interpolation_translation_y,
                            z: new_interpolation_translation_z,
                        },
                        BoneKeyframeTranslationBezierControlPointParameter {
                            x: prev_interpolation_translation_x,
                            y: prev_interpolation_translation_y,
                            z: prev_interpolation_translation_z,
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

    pub fn find_bone_transform(
        &self,
        name: &str,
        frame_index: u32,
        amount: f32,
    ) -> BoneFrameTransform {
        if let Some(track) = self.opaque.local_bone_motion_track_bundle.tracks.get(name) {
            track.seek_precisely(frame_index, amount, &self.bezier_cache)
        } else {
            BoneFrameTransform::default()
        }
    }

    pub fn find_bone_keyframe(&self, name: &str, frame_index: u32) -> Option<&MotionBoneKeyframe> {
        self.opaque.find_bone_keyframe_object(name, frame_index)
    }

    pub fn find_model_keyframe(&self, frame_index: u32) -> Option<&MotionModelKeyframe> {
        self.opaque.find_model_keyframe_object(frame_index)
    }

    pub fn find_morph_weight(&self, name: &str, frame_index: u32, amount: f32) -> f32 {
        if let Some(track) = self.opaque.local_morph_motion_track_bundle.tracks.get(name) {
            track.seek_precisely(frame_index, amount, &self.bezier_cache)
        } else {
            0f32
        }
    }

    pub fn find_morph_keyframe(
        &self,
        name: &str,
        frame_index: u32,
    ) -> Option<&MotionMorphKeyframe> {
        self.opaque.find_morph_keyframe_object(name, frame_index)
    }

    pub fn find_camera_transform(&self, frame_index: u32, amount: f32) -> Option<CameraTransform> {
        self.opaque
            .camera_keyframes
            .seek_precisely(frame_index, amount, &self.bezier_cache)
    }

    pub fn find_camera_keyframe(&self, frame_index: u32) -> Option<&MotionCameraKeyframe> {
        self.opaque.find_camera_keyframe_object(frame_index)
    }

    pub fn find_light_transform(&self, frame_index: u32, amount: f32) -> Option<LightFrame> {
        self.opaque
            .light_keyframes
            .seek_precisely(frame_index, amount, &self.bezier_cache)
    }

    pub fn find_light_keyframe(&self, frame_index: u32) -> Option<&MotionLightKeyframe> {
        self.opaque.find_light_keyframe_object(frame_index)
    }

    pub fn find_self_shadow_frame(&self, frame_index: u32, amount: f32) -> Option<SelfShadowParam> {
        self.opaque
            .self_shadow_keyframes
            .seek_precisely(frame_index, amount, &self.bezier_cache)
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
        interpolation: &[u8; 4],
        prev_value: &T,
        next_value: &T,
        interval: u32,
        coef: f32,
    ) -> T
    where
        T: VectorSpace<Scalar = f32>,
    {
        let interpolation = KeyframeInterpolationPoint::new(interpolation);
        let amount = interpolation.curve_value(interval, coef, &self.bezier_cache);
        prev_value.lerp(*next_value, amount)
    }

    pub fn slerp_interpolation(
        &self,
        interpolation: &[u8; 4],
        prev_value: &Quaternion<f32>,
        next_value: &Quaternion<f32>,
        interval: u32,
        coef: f32,
    ) -> Quaternion<f32> {
        let interpolation = KeyframeInterpolationPoint::new(interpolation);
        let amount = interpolation.curve_value(interval, coef, &self.bezier_cache);
        prev_value.slerp(*next_value, amount)
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
        name: &str,
        reverse: bool,
        reversed_bone_name_set: &mut HashSet<String>,
    ) {
        const LEFT: &str = "左";
        const RIGHT: &str = "右";
        let (new_name, new_frame) = if reverse && name.starts_with(LEFT) {
            let new_name = name.replacen(LEFT, RIGHT, 1);
            reversed_bone_name_set.insert(new_name.clone());
            (new_name, Self::reverse_bone_keyframe(origin))
        } else if reverse && name.starts_with(RIGHT) {
            let new_name = name.replacen(RIGHT, LEFT, 1);
            reversed_bone_name_set.insert(new_name.clone());
            (new_name, Self::reverse_bone_keyframe(origin))
        } else {
            (name.to_owned(), origin.clone())
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
    // pub bezier_control_point: Vector4<u8>,
    pub control_point1: Vector2<u8>,
    pub control_point2: Vector2<u8>,
    pub is_linear: bool,
}

impl Default for KeyframeInterpolationPoint {
    fn default() -> Self {
        Self {
            control_point1: Vector2::new(20, 20),
            control_point2: Vector2::new(107, 107),
            is_linear: true,
        }
    }
}

impl From<[u8; 4]> for KeyframeInterpolationPoint {
    fn from(v: [u8; 4]) -> Self {
        Self::new(&v)
    }
}

impl KeyframeInterpolationPoint {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn is_linear_interpolation(interpolation: &[u8; 4]) -> bool {
        interpolation[0] == interpolation[1]
            && interpolation[2] == interpolation[3]
            && interpolation[0] + interpolation[2] == interpolation[1] + interpolation[3]
    }

    pub fn new(interpolation: &[u8; 4]) -> Self {
        if Self::is_linear_interpolation(interpolation) {
            Self::default()
        } else {
            Self {
                control_point1: Vector2::new(interpolation[0], interpolation[1]),
                control_point2: Vector2::new(interpolation[2], interpolation[3]),
                is_linear: false,
            }
        }
    }

    pub fn bezier_control_point(&self) -> [u8; 4] {
        [
            self.control_point1[0],
            self.control_point1[1],
            self.control_point2[0],
            self.control_point2[1],
        ]
    }

    pub fn lerp(&self, other: Self, amount: f32) -> Self {
        Self {
            control_point1: self
                .control_point1
                .map(|v| v as f32)
                .lerp(other.control_point1.map(|v| v as f32), amount)
                .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
            control_point2: self
                .control_point2
                .map(|v| v as f32)
                .lerp(other.control_point2.map(|v| v as f32), amount)
                .map(|v| v.clamp(0f32, u8::MAX as f32) as u8),
            is_linear: self.is_linear,
        }
    }

    pub fn curve_value(
        &self,
        interval: u32,
        amount: f32,
        bezier_factory: &dyn BezierCurveFactory,
    ) -> f32 {
        if self.is_linear {
            amount
        } else {
            let curve =
                bezier_factory.get_or_new(self.control_point1, self.control_point2, interval);
            curve.value(amount)
        }
    }
}

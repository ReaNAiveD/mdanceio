use std::{rc::Rc, cell::RefCell, collections::HashMap};

use nanoem::motion::MotionBoneKeyframe;

use crate::{motion_keyframe_selection::MotionKeyframeSelection, bezier_curve::BezierCurve};

pub struct Motion {
    selection: Box<dyn MotionKeyframeSelection>,
    opaque: Rc<RefCell<nanoem::motion::Motion>>,
    bezier_curves_data: RefCell<HashMap<u64, BezierCurve>>,
    keyframe_bezier_curves: RefCell<HashMap<Rc<RefCell<MotionBoneKeyframe>>, BezierCurve>>,
    annotations: HashMap<String, String>,
}
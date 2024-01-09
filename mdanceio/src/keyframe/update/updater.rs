use crate::motion::KeyframeBound;

pub trait Updatable {
    type Object;
    type ObjectUpdater: KeyframeUpdater;

    fn apply_add(&mut self, updater: &mut Self::ObjectUpdater, object: Option<&mut Self::Object>);
    fn apply_remove(
        &mut self,
        updater: &mut Self::ObjectUpdater,
        object: Option<&mut Self::Object>,
    );
}

pub trait KeyframeUpdater {
    fn updated(&self) -> bool;
    fn selected(&self) -> bool;
}

pub trait AddKeyframe {
    type Object;
    type Args;
    type ObjectUpdater: KeyframeUpdater;
    fn build_updater_add(
        &self,
        object: &Self::Object,
        bound: &KeyframeBound,
        args: Self::Args,
    ) -> Self::ObjectUpdater;
}

pub trait RemoveKeyframe {
    type ObjectKeyframe;
    type ObjectUpdater: KeyframeUpdater;
    fn build_updater_remove(&self, keyframe: &Self::ObjectKeyframe) -> Self::ObjectUpdater;
}

mod bezier_curve;
mod command;
pub mod error;
mod event_publisher;
pub mod model;
pub mod motion;
pub mod motion_keyframe_selection;
mod uri;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

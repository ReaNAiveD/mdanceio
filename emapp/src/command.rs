pub trait Undo {
    fn undo();
    fn redo();
    fn current_project();
    fn name() -> String;
}
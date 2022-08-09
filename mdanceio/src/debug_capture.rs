pub trait DebugCapture {
    fn start(&mut self, label: &String);
    fn stop(&mut self);
}
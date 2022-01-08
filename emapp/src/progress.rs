use std::rc::Rc;

pub trait CancelSubscriber {
    fn on_cancelled(&self);
}

pub trait CancelPublisher {
    fn start(&mut self);
    fn add_subscriber(&mut self, subscriber: Rc<dyn CancelSubscriber>);
    fn remove_subscriber(&mut self, subscriber: Rc<dyn CancelSubscriber>);
    fn stop(&mut self);
}
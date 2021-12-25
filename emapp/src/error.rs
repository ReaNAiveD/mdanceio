use crate::event_publisher::EventPublisher;

pub enum DomainType {
    DomainTypeUnknown,
    DomainTypeOS,
    DomainTypeMinizip,
    DomainTypeNanoem,
    DomainTypeNanodxm,
    DomainTypeNanomqo,
    DomainTypeApplication,
    DomainTypePlugin,
    DomainTypeCancel = 0x7ffffffe,
}

pub trait Exception {
    fn has_reason() -> bool;
    fn has_recovery_suggestion() -> bool;
    fn reason() -> &'static str;
    fn recovery_suggestion() -> &'static str;
    fn code() -> i32;
    fn is_canceled() -> bool;
    fn domain() -> DomainType;
    fn notify(publisher: impl EventPublisher);
}
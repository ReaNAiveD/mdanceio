use crate::event_publisher::EventPublisher;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn has_reason(&self) -> bool;
    fn has_recovery_suggestion(&self) -> bool;
    fn reason(&self) -> &str;
    fn recovery_suggestion(&self) -> &str;
    fn code(&self) -> i32;
    fn is_canceled(&self) -> bool;
    fn domain(&self) -> DomainType;
    // fn notify(&self, publisher: impl EventPublisher);
}

pub struct Error {
    reason:  String,
    recovery_suggestion: String,
    code: i32,
    domain: DomainType,
}

impl Exception for Error {
    fn has_reason(&self) -> bool {
        !self.reason.is_empty()
    }

    fn has_recovery_suggestion(&self) -> bool {
        !self.recovery_suggestion.is_empty()
    }

    fn reason(&self) -> &str {
        self.reason.as_str()
    }

    fn recovery_suggestion(&self) -> &str {
        self.recovery_suggestion.as_str()
    }

    fn code(&self) -> i32 {
        self.code
    }

    fn is_canceled(&self) -> bool {
        false
    }

    fn domain(&self) -> DomainType {
        self.domain
    }
}

impl Error {
    pub fn shader_unloaded_error() -> Self {
        Self {
            reason: "Technique Pass executed without shader".to_owned(),
            recovery_suggestion: "Try Restart or Report to us".to_owned(),
            code: 3,
            domain: DomainType::DomainTypeApplication,
        }
    }
}
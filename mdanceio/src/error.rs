#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainType {
    OS,
    Minizip,
    Nanoem,
    Nanodxm,
    Nanomqo,
    Application,
    Plugin,
    Cancel,
}

#[derive(Debug)]
pub struct MdanceioError {
    reason: String,
    recovery_suggestion: String,
    code: i32,
    domain: DomainType,
}

impl std::fmt::Display for MdanceioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recovery_hint = if self.recovery_suggestion.is_empty() {
            "".to_owned()
        } else {
            format!("(Try \"{}\" to recover)", self.recovery_suggestion)
        };
        write!(f, "[{:?} - {}]{}{}",self.domain, self.code, self.reason, recovery_hint)
    }
}

impl std::error::Error for MdanceioError {}

impl MdanceioError {
    pub fn new(reason: &str, recovery_suggestion: &str, domain: DomainType) -> Self {
        Self {
            reason: reason.to_owned(),
            recovery_suggestion: recovery_suggestion.to_owned(),
            code: 0,
            domain,
        }
    }

    pub fn from_nanoem(message: &str, status: nanoem::common::NanoemError) -> Self {
        Self {
            reason: message.to_owned(),
            recovery_suggestion: "".to_owned(),
            code: 0,
            domain: DomainType::Nanoem,
        }
    }

    pub fn shader_unloaded_error() -> Self {
        Self {
            reason: "Technique Pass executed without shader".to_owned(),
            recovery_suggestion: "Try Restart or Report to us".to_owned(),
            code: 3,
            domain: DomainType::Application,
        }
    }
}

pub enum LanguageType {
    Japanese,
    English,
    ChineseSimplified,
    ChineseTraditional,
    Korean,
}

pub trait Translator {
    fn translate(&self, text: &str) -> String;
    fn is_translatable(&self, text: &str) -> bool;

    fn language(&self) -> LanguageType;
    fn set_language(&mut self, lang: LanguageType);
    fn is_supported_language(&self, lang: LanguageType) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uri {
    absolute_path: String,
    fragment: String,
}

impl Uri {
    pub fn absolute_path(&self) -> &String {
        &self.absolute_path
    }

    pub fn fragment(&self) -> &String {
        &self.fragment
    }

    pub fn string_by_deleting_trailing_path_separator(value: &String) -> String {
        if value.ends_with("/") {
            value[0..value.len() - 1].into()
        } else {
            value.clone()
        }
    }

    pub fn string_by_deleting_last_component(value: &String) -> String {
        if let Some((left, _)) = value.rsplit_once('/') {
            left.into()
        } else {
            value.clone()
        }
    }

    pub fn string_by_deleting_path_extension(value: &String) -> String {
        if let Some(pos) = value.rfind('/') {
            if let Some(pos_dot) = value[pos + 1..].rfind('.') {
                return value[..pos_dot].into()
            }
        } else if let Some(pos_dot) = value.rfind('.') {
            return value[..pos_dot].into()
        }
        return value.clone();
    }

    pub fn last_path_component(value: &String) -> &str {
        if let Some((_, right)) = value.rsplit_once('/') {
            right
        } else {
            &value[..]
        }
    }

    pub fn path_extension(value: &String) -> Option<&str> {
        let last_component = Self::last_path_component(value);
        if let Some(pos) = last_component.rfind('.') {
            Some(&last_component[pos + 1..])
        } else {
            None
        }
    }

    pub fn create_from_file_path(path: &String) -> Self {
        Self {
            absolute_path: path.clone(),
            fragment: "".into(),
        }
    }

    pub fn create_from_file_path_with_fragment(path: &String, fragment: &String) -> Self {
        Self {
            absolute_path: path.clone(),
            fragment: fragment.clone(),
        }
    }

    pub fn absolute_path_by_deleting_last_path_component(&self) -> String {
        Self::string_by_deleting_last_component(&self.absolute_path)
    }

    pub fn absolute_path_by_deleting_path_extension(&self) -> String {
        Self::string_by_deleting_path_extension(&self.absolute_path)
    }

    pub fn last_absolute_path_component(&self) -> &str {
        Self::last_path_component(&self.absolute_path)
    }

    pub fn absolute_path_extension(&self) -> Option<&str> {
        Self::path_extension(&self.absolute_path)
    }

    pub fn is_empty(&self) -> bool {
        self.absolute_path.is_empty() && self.fragment.is_empty()
    }

    pub fn has_fragment(&self) -> bool {
        !self.fragment.is_empty()
    }

    pub fn equal_to_absolute_path(&self, other: &str) -> bool {
        (&self.absolute_path[..]).eq(other)
    }

    pub fn equal_to_filename(&self, other: &str) -> bool {
        self.last_absolute_path_component().to_lowercase().eq(&other.to_lowercase())
    }
}
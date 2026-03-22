pub struct User {
    first_name: String,
    last_name: String,
    email: String,
}

impl User {
    pub fn new(first_name: &str, last_name: &str, email: &str) -> Self {
        Self {
            first_name: first_name.to_string(),
            last_name: last_name.to_string(),
            email: email.to_string(),
        }
    }

    pub fn email(&self) -> &str {
        &self.email
    }

    pub fn first_name(&self) -> &str {
        &self.first_name
    }

    pub fn last_name(&self) -> &str {
        &self.last_name
    }

    pub fn display_name(&self) -> String {
        format!("{} {}", self.first_name, self.last_name)
    }
}

mod models;
mod utils;

use models::user::User;

fn main() {
    let user = User::new("Alice", "Smith", "alice@example.com");
    println!("Created user: {}", user.email());
}

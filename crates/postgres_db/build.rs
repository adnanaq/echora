//! Triggers Cargo rebuilds when SQL migration files change.

fn main() {
    println!("cargo:rerun-if-changed=migrations");
}

//! Triggers Cargo rebuilds when SQL migration files change.

// Rust guideline compliant 2026-10-09

fn main() {
    println!("cargo:rerun-if-changed=migrations");
}

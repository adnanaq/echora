//! Shared configuration, error types, and OpenTelemetry setup for Echora Rust services.

// Rust guideline compliant 2026-10-09

pub mod config;
pub mod error;
pub mod telemetry;

#[doc(inline)]
pub use config::{CommonSettings, ObservabilitySettings};
#[doc(inline)]
pub use error::Error;
#[doc(inline)]
pub use telemetry::{TelemetryGuard, init_telemetry};

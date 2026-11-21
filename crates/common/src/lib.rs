//! Shared configuration, error types, and OpenTelemetry setup for Echora Rust services.

mod config;
mod error;
mod telemetry;

#[doc(inline)]
pub use config::{CommonSettings, ObservabilitySettings};
#[doc(inline)]
pub use error::Error;
#[doc(inline)]
pub use telemetry::{TelemetryGuard, init_telemetry};

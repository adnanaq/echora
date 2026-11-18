//! Error types for the shared `common` crate.

// Rust guideline compliant 2026-10-09

use std::fmt;
use std::num::ParseIntError;

/// Represents failures from loading configuration or initializing telemetry.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    InvalidEnv { name: &'static str, message: String },
    Dotenv(dotenvy::Error),
    TelemetryInit(String),
}

impl Error {
    /// Creates an invalid environment variable error for `name`.
    pub(crate) fn invalid_env(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidEnv {
                name,
                message: message.into(),
            },
        }
    }

    /// Creates a telemetry initialization error with `message`.
    pub(crate) fn telemetry_init(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::TelemetryInit(message.into()),
        }
    }

    /// Returns `true` when an environment variable failed parsing or validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use common::Error;
    ///
    /// let err = Error::from("bad".parse::<u16>().unwrap_err());
    /// assert!(err.is_invalid_env());
    /// ```
    #[must_use]
    pub fn is_invalid_env(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidEnv { .. })
    }

    /// Returns `true` when loading a `.env` file failed.
    #[must_use]
    pub fn is_dotenv(&self) -> bool {
        matches!(self.kind, ErrorKind::Dotenv(_))
    }

    /// Returns `true` when setting up logging or OpenTelemetry failed.
    #[must_use]
    pub fn is_telemetry_init(&self) -> bool {
        matches!(self.kind, ErrorKind::TelemetryInit(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::InvalidEnv { name, message } => {
                write!(formatter, "invalid environment variable {name}: {message}")
            }
            ErrorKind::Dotenv(err) => write!(formatter, "failed to load .env file: {err}"),
            ErrorKind::TelemetryInit(message) => {
                write!(formatter, "failed to initialize telemetry: {message}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Dotenv(err) => Some(err),
            ErrorKind::InvalidEnv { .. } | ErrorKind::TelemetryInit(_) => None,
        }
    }
}

impl From<dotenvy::Error> for Error {
    fn from(err: dotenvy::Error) -> Self {
        Self {
            kind: ErrorKind::Dotenv(err),
        }
    }
}

impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Self::invalid_env("INTEGER", err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[test]
    fn error_kind_predicates_match_variants() {
        let env_err = Error::invalid_env("POSTGRES_SERVICE_PORT", "out of range");
        assert!(env_err.is_invalid_env());
        assert!(!env_err.is_dotenv());
        assert!(!env_err.is_telemetry_init());
        assert!(env_err.to_string().contains("POSTGRES_SERVICE_PORT"));
        assert!(env_err.source().is_none());

        let tel_err = Error::telemetry_init("collector unreachable");
        assert!(tel_err.is_telemetry_init());
        assert!(!tel_err.is_invalid_env());
    }
}

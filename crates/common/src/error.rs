//! Error types for the shared `common` crate.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt;
use std::num::ParseIntError;

/// Represents failures from loading configuration or initializing telemetry.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    InvalidEnv { name: &'static str, message: String },
    Dotenv(dotenvy::Error),
    TelemetryInit(String),
}

impl Error {
    /// Creates an error of `kind`, capturing a backtrace when `RUST_BACKTRACE` asks for one.
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an invalid environment variable error for `name`.
    pub(crate) fn invalid_env(name: &'static str, message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidEnv {
            name,
            message: message.into(),
        })
    }

    /// Creates a telemetry initialization error with `message`.
    pub(crate) fn telemetry_init(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::TelemetryInit(message.into()))
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
    /// assert!(!err.is_dotenv());
    /// assert!(err.to_string().starts_with(
    ///     "invalid environment variable INTEGER: invalid digit found in string"
    /// ));
    /// ```
    #[must_use]
    pub fn is_invalid_env(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidEnv { .. })
    }

    /// Returns `true` when loading a `.env` file failed.
    ///
    /// # Examples
    ///
    /// ```
    /// use common::Error;
    ///
    /// let err = Error::from(dotenvy::Error::LineParse("BAD LINE".to_owned(), 1));
    /// assert!(err.is_dotenv());
    /// assert!(!err.is_invalid_env());
    /// assert!(err.to_string().starts_with("failed to load .env file: "));
    /// ```
    #[must_use]
    pub fn is_dotenv(&self) -> bool {
        matches!(self.kind, ErrorKind::Dotenv(_))
    }

    /// Returns `true` when setting up logging or OpenTelemetry failed.
    ///
    /// # Examples
    ///
    /// ```
    /// use common::Error;
    ///
    /// let bad_setting = Error::from("bad".parse::<u16>().unwrap_err());
    /// assert!(!bad_setting.is_telemetry_init());
    /// assert!(bad_setting.is_invalid_env(), "a bad setting is not a telemetry failure");
    /// ```
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
        .and_then(|()| {
            if self.backtrace.status() == BacktraceStatus::Captured {
                write!(formatter, "\n\nbacktrace:\n{}", self.backtrace)
            } else {
                Ok(())
            }
        })
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
        Self::from_kind(ErrorKind::Dotenv(err))
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
        assert!(tel_err.to_string().contains("collector unreachable"));
        assert!(tel_err.source().is_none());

        let dotenv_err = Error::from(dotenvy::Error::LineParse("BAD LINE".to_owned(), 3));
        assert!(dotenv_err.is_dotenv());
        assert!(!dotenv_err.is_telemetry_init());
        assert!(dotenv_err.to_string().contains("failed to load .env file"));
        assert!(dotenv_err.source().is_some());

        let port_err = Error::from("not-a-port".parse::<u16>().expect_err("not a number"));
        assert!(port_err.is_invalid_env());
        assert!(port_err.to_string().contains("INTEGER"));
    }

    #[test]
    fn display_captured_backtrace_follows_summary() {
        let traced_error = Error {
            kind: ErrorKind::TelemetryInit("collector unreachable".to_owned()),
            backtrace: Backtrace::force_capture(),
        };
        let untraced_error = Error {
            kind: ErrorKind::TelemetryInit("collector unreachable".to_owned()),
            backtrace: Backtrace::disabled(),
        };

        let traced_text = traced_error.to_string();

        assert!(
            traced_text.starts_with(
                "failed to initialize telemetry: collector unreachable\n\nbacktrace:\n"
            )
        );
        assert_eq!(
            untraced_error.to_string(),
            "failed to initialize telemetry: collector unreachable"
        );
    }
}

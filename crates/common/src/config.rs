//! Environment-backed settings shared across Echora Rust services.

// Rust guideline compliant 2026-10-09

use std::collections::HashMap;
use std::env;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;

use crate::Error;

/// Default gRPC port for the PostgreSQL Service.
///
/// Chosen right after `8001` (`vector_service`), `8002` (`enrichment_service`),
/// and `8003` (`review_page`) to avoid port collisions in local compose.
const DEFAULT_POSTGRES_SERVICE_PORT: u16 = 8004;

/// Holds deployment environment, logging, and telemetry configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonSettings {
    /// Deployment environment name (for example `development` or `production`).
    pub environment: String,
    /// Whether debug behavior is enabled.
    pub debug: bool,
    /// Minimum log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
    pub log_level: String,
    /// OpenTelemetry configuration matching Python `ObservabilityConfig`.
    pub observability: ObservabilitySettings,
    /// Network bind address for the PostgreSQL Service gRPC server.
    pub postgres_service_host: String,
    /// Network bind port for the PostgreSQL Service gRPC server.
    pub postgres_service_port: u16,
}

/// Holds OpenTelemetry toggle and exporter configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors Python ObservabilityConfig env toggles"
)]
pub struct ObservabilitySettings {
    /// Enables OpenTelemetry initialization.
    pub otel_enabled: bool,
    /// OTLP gRPC collector endpoint URL.
    pub otel_exporter_otlp_endpoint: String,
    /// Enables structured logging setup.
    pub otel_enable_logging: bool,
    /// Enables distributed tracing setup.
    pub otel_enable_tracing: bool,
    /// Enables metrics export setup.
    pub otel_enable_metrics: bool,
}

impl Default for ObservabilitySettings {
    fn default() -> Self {
        Self {
            otel_enabled: true,
            otel_exporter_otlp_endpoint: "http://localhost:4317".to_owned(),
            otel_enable_logging: true,
            otel_enable_tracing: true,
            otel_enable_metrics: true,
        }
    }
}

impl Default for CommonSettings {
    fn default() -> Self {
        Self {
            environment: "development".to_owned(),
            debug: true,
            log_level: "INFO".to_owned(),
            observability: ObservabilitySettings::default(),
            postgres_service_host: "0.0.0.0".to_owned(),
            postgres_service_port: DEFAULT_POSTGRES_SERVICE_PORT,
        }
    }
}

impl CommonSettings {
    /// Loads settings from process environment, `.env`, and built-in defaults.
    ///
    /// Process environment variables take precedence over `.env` entries, matching
    /// Python's `EnvSettings`.
    ///
    /// # Errors
    ///
    /// Returns an error when an environment variable cannot be parsed as a valid
    /// boolean, port, or log level, or when `.env` exists but cannot be read.
    pub fn from_env() -> Result<Self, Error> {
        let dotenv_values = load_optional_dotenv(Path::new(".env"))?;
        Self::from_lookup(|name| {
            env::var(name)
                .ok()
                .or_else(|| dotenv_values.get(name).cloned())
        })
    }

    /// Loads settings from a custom lookup function without reading process state.
    ///
    /// # Errors
    ///
    /// Returns an error when any looked-up value fails validation.
    pub fn from_lookup<F>(lookup: F) -> Result<Self, Error>
    where
        F: Fn(&str) -> Option<String>,
    {
        let defaults = Self::default();
        let environment = lookup("ENVIRONMENT").unwrap_or(defaults.environment);
        let debug = parse_bool(&lookup, "DEBUG", defaults.debug)?;
        let log_level = parse_log_level(&lookup, "LOG_LEVEL", &defaults.log_level)?;
        let postgres_service_host =
            lookup("POSTGRES_SERVICE_HOST").unwrap_or(defaults.postgres_service_host);
        let postgres_service_port = parse_port(
            &lookup,
            "POSTGRES_SERVICE_PORT",
            defaults.postgres_service_port,
        )?;

        let obs_defaults = defaults.observability;
        let observability = ObservabilitySettings {
            otel_enabled: parse_bool(&lookup, "OTEL_ENABLED", obs_defaults.otel_enabled)?,
            otel_exporter_otlp_endpoint: lookup("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or(obs_defaults.otel_exporter_otlp_endpoint),
            otel_enable_logging: parse_bool(
                &lookup,
                "OTEL_ENABLE_LOGGING",
                obs_defaults.otel_enable_logging,
            )?,
            otel_enable_tracing: parse_bool(
                &lookup,
                "OTEL_ENABLE_TRACING",
                obs_defaults.otel_enable_tracing,
            )?,
            otel_enable_metrics: parse_bool(
                &lookup,
                "OTEL_ENABLE_METRICS",
                obs_defaults.otel_enable_metrics,
            )?,
        };

        Ok(Self {
            environment,
            debug,
            log_level,
            observability,
            postgres_service_host,
            postgres_service_port,
        })
    }

    /// Parses the PostgreSQL Service listen address into a `SocketAddr`.
    ///
    /// # Errors
    ///
    /// Returns an error when `postgres_service_host` and `postgres_service_port`
    /// do not form a valid socket address.
    pub fn postgres_service_addr(&self) -> Result<SocketAddr, Error> {
        let host = self.postgres_service_host.as_str();
        let ip_text = match (host.strip_prefix('['), host.strip_suffix(']')) {
            (Some(stripped), Some(_)) => &stripped[..stripped.len() - 1],
            (None, None) => host,
            _ => {
                return Err(Error::invalid_env(
                    "POSTGRES_SERVICE_HOST",
                    "unbalanced brackets in IP address",
                ));
            }
        };
        let ip = ip_text
            .parse::<IpAddr>()
            .map_err(|err| Error::invalid_env("POSTGRES_SERVICE_HOST", err.to_string()))?;
        Ok(SocketAddr::new(ip, self.postgres_service_port))
    }
}

fn load_optional_dotenv(path: &Path) -> Result<HashMap<String, String>, Error> {
    match dotenvy::from_path_iter(path) {
        Ok(iter) => {
            let mut values = HashMap::new();
            for env_entry in iter {
                let (key, value) = env_entry?;
                values.insert(key, value);
            }
            Ok(values)
        }
        Err(err) if err.not_found() => Ok(HashMap::new()),
        Err(err) => Err(Error::from(err)),
    }
}

fn parse_bool<F>(lookup: &F, name: &'static str, default: bool) -> Result<bool, Error>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(env_value) = lookup(name) else {
        return Ok(default);
    };
    match env_value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(Error::invalid_env(
            name,
            format!("expected boolean, got {env_value:?}"),
        )),
    }
}

fn parse_port<F>(lookup: &F, name: &'static str, default: u16) -> Result<u16, Error>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(env_value) = lookup(name) else {
        return Ok(default);
    };
    let port: u16 = env_value
        .trim()
        .parse()
        .map_err(|err| Error::invalid_env(name, format!("invalid port {env_value:?}: {err}")))?;
    if port == 0 {
        return Err(Error::invalid_env(name, "port must be between 1 and 65535"));
    }
    Ok(port)
}

fn parse_log_level<F>(lookup: &F, name: &'static str, default: &str) -> Result<String, Error>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(env_value) = lookup(name) else {
        return Ok(default.to_owned());
    };
    let normalized_level = env_value.trim().to_ascii_uppercase();
    match normalized_level.as_str() {
        "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" => Ok(normalized_level),
        _ => Err(Error::invalid_env(
            name,
            format!("expected DEBUG, INFO, WARNING, ERROR, or CRITICAL, got {env_value:?}"),
        )),
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "Unit tests panic on unexpected Result variants"
)]
mod tests {
    use super::*;

    #[test]
    fn from_lookup_uses_defaults_when_unset() {
        let settings = CommonSettings::from_lookup(|_| None).expect("defaults should be valid");
        assert_eq!(settings.environment, "development");
        assert!(settings.debug);
        assert_eq!(settings.log_level, "INFO");
        assert_eq!(settings.postgres_service_host, "0.0.0.0");
        assert_eq!(settings.postgres_service_port, 8004);
        assert!(settings.observability.otel_enabled);
        assert_eq!(
            settings.observability.otel_exporter_otlp_endpoint,
            "http://localhost:4317"
        );
    }

    #[test]
    fn from_lookup_parses_overrides_and_socket_addr() {
        let overrides = HashMap::from([
            ("ENVIRONMENT", "production"),
            ("DEBUG", "false"),
            ("LOG_LEVEL", "warning"),
            ("POSTGRES_SERVICE_HOST", "127.0.0.1"),
            ("POSTGRES_SERVICE_PORT", "9004"),
            ("OTEL_ENABLED", "0"),
        ]);
        let settings = CommonSettings::from_lookup(|name| {
            overrides.get(name).map(|value| (*value).to_owned())
        })
        .expect("overrides should parse");

        assert_eq!(settings.environment, "production");
        assert!(!settings.debug);
        assert_eq!(settings.log_level, "WARNING");
        assert!(!settings.observability.otel_enabled);
        assert_eq!(
            settings
                .postgres_service_addr()
                .expect("valid socket address")
                .to_string(),
            "127.0.0.1:9004"
        );
    }

    #[test]
    fn from_lookup_rejects_invalid_port_and_bool() {
        let bad_port = CommonSettings::from_lookup(|name| {
            (name == "POSTGRES_SERVICE_PORT").then(|| "0".to_owned())
        })
        .expect_err("port 0 must fail");
        assert!(bad_port.is_invalid_env());

        let bad_bool = CommonSettings::from_lookup(|name| {
            (name == "OTEL_ENABLED").then(|| "maybe".to_owned())
        })
        .expect_err("non-boolean must fail");
        assert!(bad_bool.is_invalid_env());
    }

    #[test]
    fn postgres_service_addr_supports_ipv6_bare_and_bracketed() {
        for (host, expected) in [("::1", "[::1]:8004"), ("[::1]", "[::1]:8004")] {
            let settings = CommonSettings::from_lookup(|name| {
                (name == "POSTGRES_SERVICE_HOST").then(|| host.to_owned())
            })
            .expect("valid host");
            assert_eq!(
                settings
                    .postgres_service_addr()
                    .expect("valid ipv6 socket address")
                    .to_string(),
                expected
            );
        }

        let unbalanced = CommonSettings::from_lookup(|name| {
            (name == "POSTGRES_SERVICE_HOST").then(|| "[::1".to_owned())
        })
        .expect("settings parse");
        assert!(
            unbalanced
                .postgres_service_addr()
                .expect_err("unbalanced brackets must fail")
                .is_invalid_env()
        );
    }
}

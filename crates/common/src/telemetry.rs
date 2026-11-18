//! Structured logging and OpenTelemetry bootstrap for Echora Rust services.

// Rust guideline compliant 2026-10-09

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::Level;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

use crate::{CommonSettings, Error};

/// Flushes pending OpenTelemetry spans and metrics when dropped.
#[derive(Debug)]
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take()
            && let Err(err) = provider.shutdown()
        {
            tracing::event!(
                name: "telemetry.shutdown.failed",
                Level::WARN,
                error.message = %err,
                "failed to shut down OpenTelemetry tracer provider: {{error.message}}",
            );
        }
        if let Some(provider) = self.meter_provider.take()
            && let Err(err) = provider.shutdown()
        {
            tracing::event!(
                name: "telemetry.metrics_shutdown.failed",
                Level::WARN,
                error.message = %err,
                "failed to shut down OpenTelemetry meter provider: {{error.message}}",
            );
        }
    }
}

/// Initializes structured `tracing` logs and optional OpenTelemetry OTLP export.
///
/// Reads log level, structured logging toggle, and OTLP collector settings from
/// `settings` and installs the global `tracing` subscriber.
///
/// # Errors
///
/// Returns an error when an OTLP exporter cannot be built or a global
/// subscriber is already installed.
pub fn init_telemetry(
    service_name: &str,
    settings: &CommonSettings,
) -> Result<TelemetryGuard, Error> {
    let filter_directive = map_log_level(&settings.log_level);
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_directive));

    let resource = Resource::builder()
        .with_service_name(service_name.to_owned())
        .with_attribute(KeyValue::new(
            "deployment.environment.name",
            settings.environment.clone(),
        ))
        .build();

    let (tracer_provider, otel_layer) =
        if settings.observability.otel_enabled && settings.observability.otel_enable_tracing {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(settings.observability.otel_exporter_otlp_endpoint.clone())
                .build()
                .map_err(|err| Error::telemetry_init(err.to_string()))?;

            let provider = SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_resource(resource.clone())
                .build();

            let tracer = provider.tracer(service_name.to_owned());
            let layer = tracing_opentelemetry::layer().with_tracer(tracer);
            (Some(provider), Some(layer))
        } else {
            (None, None)
        };

    let meter_provider =
        if settings.observability.otel_enabled && settings.observability.otel_enable_metrics {
            let exporter = opentelemetry_otlp::MetricExporter::builder()
                .with_tonic()
                .with_endpoint(settings.observability.otel_exporter_otlp_endpoint.clone())
                .build()
                .map_err(|err| Error::telemetry_init(err.to_string()))?;

            let provider = SdkMeterProvider::builder()
                .with_periodic_exporter(exporter)
                .with_resource(resource)
                .build();
            global::set_meter_provider(provider.clone());
            Some(provider)
        } else {
            None
        };

    let fmt_layer = settings
        .observability
        .otel_enable_logging
        .then(|| tracing_subscriber::fmt::layer().json());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()
        .map_err(|err| Error::telemetry_init(err.to_string()))?;

    tracing::event!(
        name: "telemetry.initialized",
        Level::INFO,
        service.name = service_name,
        deployment.environment = %settings.environment,
        otel.enabled = settings.observability.otel_enabled,
        "telemetry initialized for {{service.name}}",
    );

    Ok(TelemetryGuard {
        tracer_provider,
        meter_provider,
    })
}

fn map_log_level(python_level: &str) -> &'static str {
    match python_level {
        "DEBUG" => "debug",
        "WARNING" => "warn",
        "ERROR" | "CRITICAL" => "error",
        _ => "info",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_log_level_translates_python_names() {
        assert_eq!(map_log_level("DEBUG"), "debug");
        assert_eq!(map_log_level("INFO"), "info");
        assert_eq!(map_log_level("WARNING"), "warn");
        assert_eq!(map_log_level("ERROR"), "error");
        assert_eq!(map_log_level("CRITICAL"), "error");
    }
}

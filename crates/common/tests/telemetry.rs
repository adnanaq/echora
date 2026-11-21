//! Integration tests for telemetry setup, which installs process-wide state.

use common::{CommonSettings, ObservabilitySettings, init_telemetry};

fn build_settings(
    collector_endpoint: &str,
    is_tracing_enabled: bool,
    is_metrics_enabled: bool,
) -> CommonSettings {
    CommonSettings {
        observability: ObservabilitySettings {
            otel_enabled: true,
            otel_exporter_otlp_endpoint: collector_endpoint.to_owned(),
            otel_enable_logging: true,
            otel_enable_tracing: is_tracing_enabled,
            otel_enable_metrics: is_metrics_enabled,
        },
        ..CommonSettings::default()
    }
}

#[tokio::test]
async fn init_telemetry_installs_once_and_refuses_invalid_collector() {
    let invalid_tracing =
        init_telemetry("telemetry-test", &build_settings("not a url", true, false))
            .expect_err("an invalid collector address must be refused for tracing");
    let invalid_metrics =
        init_telemetry("telemetry-test", &build_settings("not a url", false, true))
            .expect_err("an invalid collector address must be refused for metrics");

    let telemetry_guard = init_telemetry(
        "telemetry-test",
        &build_settings("http://127.0.0.1:4317", true, true),
    )
    .expect("the first valid setup installs the subscriber");
    let repeated_setup = init_telemetry(
        "telemetry-test",
        &build_settings("http://127.0.0.1:4317", false, false),
    )
    .expect_err("a second subscriber must be refused");
    drop(telemetry_guard);

    assert!(invalid_tracing.is_telemetry_init());
    assert!(invalid_metrics.is_telemetry_init());
    assert!(repeated_setup.is_telemetry_init());
}

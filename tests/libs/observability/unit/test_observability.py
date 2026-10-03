from __future__ import annotations

import inspect

import observability
from observability.metrics import _HISTOGRAM_VIEWS


def _reset_telemetry_state(monkeypatch) -> None:
    monkeypatch.setattr(observability, "_telemetry_initialized", False)
    monkeypatch.setattr(observability, "_telemetry_init_signature", None)


def test_setup_telemetry_initializes_all_signal_pipelines(monkeypatch) -> None:
    _reset_telemetry_state(monkeypatch)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        observability,
        "setup_logging",
        lambda *, level, service_name, environment, **_: calls.append(
            ("logging", (level, service_name, environment))
        ),
    )
    monkeypatch.setattr(
        observability,
        "setup_tracing",
        lambda *, service_name, endpoint, resource_attributes, **_: calls.append(
            ("tracing", (service_name, endpoint, resource_attributes))
        ),
    )
    monkeypatch.setattr(
        observability,
        "setup_metrics",
        lambda *, service_name, endpoint, resource_attributes, **_: calls.append(
            ("metrics", (service_name, endpoint, resource_attributes))
        ),
    )
    monkeypatch.setattr(
        observability,
        "instrument_grpc_client",
        lambda: calls.append(("grpc_client", True)),
    )
    monkeypatch.setattr(
        observability,
        "instrument_aiohttp_client",
        lambda: calls.append(("aiohttp_client", True)),
    )

    observability.setup_telemetry(
        service_name="echora-test-service",
        version="1.2.3",
        environment="development",
        endpoint="http://otel-collector:4317",
        log_level="DEBUG",
        enable_grpc_client_instrumentation=True,
        enable_aiohttp_client_instrumentation=True,
    )

    call_names = [name for name, _ in calls]
    assert call_names == [
        "logging",
        "tracing",
        "metrics",
        "grpc_client",
        "aiohttp_client",
    ]


def test_no_grpc_server_hook_is_exported() -> None:
    assert "instrument_grpc_server" not in observability.__all__
    assert not hasattr(observability, "instrument_grpc_server")


def test_no_qdrant_client_auto_instrumentation_is_offered() -> None:
    assert "instrument_qdrant_client" not in observability.__all__
    assert not hasattr(observability, "instrument_qdrant_client")
    assert (
        "enable_qdrant_client_instrumentation"
        not in inspect.signature(observability.setup_telemetry).parameters
    )


def test_batcher_histograms_have_explicit_buckets() -> None:
    names = {view._instrument_name for view in _HISTOGRAM_VIEWS}
    assert {
        "echora_batcher_queue_wait_seconds",
        "echora_batcher_batch_size",
    } <= names


def test_setup_telemetry_respects_signal_toggles(monkeypatch) -> None:
    _reset_telemetry_state(monkeypatch)
    calls: list[str] = []

    monkeypatch.setattr(
        observability, "setup_logging", lambda **_: calls.append("logging")
    )
    monkeypatch.setattr(
        observability, "setup_tracing", lambda **_: calls.append("tracing")
    )
    monkeypatch.setattr(
        observability, "setup_metrics", lambda **_: calls.append("metrics")
    )

    observability.setup_telemetry(
        service_name="echora-test-service",
        version="1.2.3",
        environment="development",
        endpoint="http://otel-collector:4317",
        enable_logging=False,
        enable_tracing=False,
        enable_metrics=True,
    )

    assert calls == ["metrics"]


def test_setup_telemetry_idempotent_repeated_calls(monkeypatch) -> None:
    _reset_telemetry_state(monkeypatch)
    calls: list[str] = []

    monkeypatch.setattr(
        observability, "setup_logging", lambda **_: calls.append("logging")
    )
    monkeypatch.setattr(
        observability, "setup_tracing", lambda **_: calls.append("tracing")
    )
    monkeypatch.setattr(
        observability, "setup_metrics", lambda **_: calls.append("metrics")
    )
    monkeypatch.setattr(
        observability, "instrument_grpc_client", lambda: calls.append("grpc_client")
    )

    observability.setup_telemetry(
        service_name="echora-test-service",
        version="1.2.3",
        environment="development",
        endpoint="http://otel-collector:4317",
        enable_grpc_client_instrumentation=True,
    )
    observability.setup_telemetry(
        service_name="echora-test-service",
        version="1.2.3",
        environment="development",
        endpoint="http://otel-collector:4317",
        enable_grpc_client_instrumentation=True,
    )

    assert calls == ["logging", "tracing", "metrics", "grpc_client"]

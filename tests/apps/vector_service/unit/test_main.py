from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from grpc_health.v1 import health_pb2
from vector_service import main


def test_setup_observability_calls_telemetry_bootstrap(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_setup_telemetry(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main, "setup_telemetry", _fake_setup_telemetry)

    settings = SimpleNamespace(
        service=SimpleNamespace(
            api_version="9.9.9",
            log_level="INFO",
        ),
        observability=SimpleNamespace(
            otel_enabled=True,
            otel_exporter_otlp_endpoint="http://localhost:4317",
            otel_enable_metrics=True,
            otel_enable_tracing=True,
            otel_enable_logging=True,
            otel_enable_grpc_server_instrumentation=True,
            otel_enable_grpc_client_instrumentation=True,
            otel_enable_aiohttp_client_instrumentation=False,
        ),
        environment=SimpleNamespace(value="development"),
    )

    main._setup_observability(settings)

    assert captured["service_name"] == "echora-vector-service"
    assert captured["version"] == "9.9.9"
    assert captured["endpoint"] == "http://localhost:4317"
    assert captured["enable_grpc_server_instrumentation"] is True
    assert captured["enable_aiohttp_client_instrumentation"] is False


@pytest.mark.asyncio
async def test_initial_readiness_serving_when_healthy() -> None:
    runtime = SimpleNamespace(
        qdrant_client=SimpleNamespace(health_check=AsyncMock(return_value=True))
    )
    health_servicer = AsyncMock()

    await main._publish_initial_readiness(runtime, health_servicer)

    expected_status = health_pb2.HealthCheckResponse.SERVING
    assert health_servicer.set.await_count == 3
    health_servicer.set.assert_any_await("", expected_status)
    health_servicer.set.assert_any_await(
        "vector_service.v1.VectorAdminService", expected_status
    )
    health_servicer.set.assert_any_await(
        "vector_service.v1.VectorSearchService", expected_status
    )


@pytest.mark.asyncio
async def test_initial_readiness_not_serving_when_unhealthy() -> None:
    runtime = SimpleNamespace(
        qdrant_client=SimpleNamespace(health_check=AsyncMock(return_value=False))
    )
    health_servicer = AsyncMock()

    await main._publish_initial_readiness(runtime, health_servicer)

    expected_status = health_pb2.HealthCheckResponse.NOT_SERVING
    health_servicer.set.assert_any_await("", expected_status)
    health_servicer.set.assert_any_await(
        "vector_service.v1.VectorAdminService", expected_status
    )
    health_servicer.set.assert_any_await(
        "vector_service.v1.VectorSearchService", expected_status
    )

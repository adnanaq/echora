from __future__ import annotations

from unittest.mock import create_autospec, patch

import pytest
from common.config.settings import Settings
from grpc_health.v1 import health, health_pb2
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from vector_processing import (
    MultiVectorEmbeddingManager,
    TextProcessor,
    VisionProcessor,
)
from vector_service import main
from vector_service.runtime import VectorRuntime

_HEALTH_SERVICES = (
    "",
    "vector_service.v1.VectorAdminService",
    "vector_service.v1.VectorSearchService",
)


def _runtime(*, qdrant_healthy: bool) -> VectorRuntime:
    qdrant_client = create_autospec(QdrantClient, instance=True)
    qdrant_client.health_check.return_value = qdrant_healthy
    return VectorRuntime(
        qdrant_client=qdrant_client,
        async_qdrant_client=create_autospec(AsyncQdrantClient, instance=True),
        text_processor=create_autospec(TextProcessor, instance=True),
        vision_processor=create_autospec(VisionProcessor, instance=True),
        embedding_manager=create_autospec(MultiVectorEmbeddingManager, instance=True),
        embedding_cache=None,
    )


async def _published_statuses(servicer: health.aio.HealthServicer) -> dict[str, int]:
    return {
        service: (
            await servicer.Check(health_pb2.HealthCheckRequest(service=service), None)
        ).status
        for service in _HEALTH_SERVICES
    }


def test_setup_observability_otel_enabled_passes_settings_to_setup_telemetry() -> None:
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "development",
            "OTEL_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
            "OTEL_ENABLE_AIOHTTP_CLIENT_INSTRUMENTATION": "false",
            "OTEL_ENABLE_REDIS_INSTRUMENTATION": "true",
        },
        clear=True,
    ):
        settings = Settings()

    with patch.object(main, "setup_telemetry", autospec=True) as setup_telemetry:
        main._setup_observability(settings)

    arguments = setup_telemetry.call_args.kwargs
    assert arguments["service_name"] == "echora-vector-service"
    assert arguments["version"] == settings.service.api_version
    assert arguments["environment"] == "development"
    assert arguments["endpoint"] == "http://collector:4317"
    assert arguments["log_level"] == settings.service.log_level
    assert arguments["enable_logging"] is True
    assert arguments["enable_tracing"] is True
    assert arguments["enable_metrics"] is True
    assert arguments["enable_grpc_client_instrumentation"] is True
    assert arguments["enable_aiohttp_client_instrumentation"] is False
    assert arguments["enable_redis_instrumentation"] is True


@pytest.mark.asyncio
async def test_publish_initial_readiness_healthy_qdrant_sets_serving() -> None:
    servicer = health.aio.HealthServicer()

    await main._publish_initial_readiness(_runtime(qdrant_healthy=True), servicer)

    assert await _published_statuses(servicer) == dict.fromkeys(
        _HEALTH_SERVICES, health_pb2.HealthCheckResponse.SERVING
    )


@pytest.mark.asyncio
async def test_publish_initial_readiness_unhealthy_qdrant_sets_not_serving() -> None:
    servicer = health.aio.HealthServicer()

    await main._publish_initial_readiness(_runtime(qdrant_healthy=False), servicer)

    assert await _published_statuses(servicer) == dict.fromkeys(
        _HEALTH_SERVICES, health_pb2.HealthCheckResponse.NOT_SERVING
    )

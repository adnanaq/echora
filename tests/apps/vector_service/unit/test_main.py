from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from grpc_health.v1 import health_pb2
from vector_service import main


@pytest.mark.asyncio
async def test_publish_initial_readiness_sets_serving_for_healthy_qdrant() -> None:
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
async def test_publish_initial_readiness_sets_not_serving_for_unhealthy_qdrant() -> (
    None
):
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

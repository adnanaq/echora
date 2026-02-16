from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vector_proto.v1 import vector_admin_pb2
from vector_service.routes import admin


def _runtime_with_stats(stats_payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_client=SimpleNamespace(
            get_stats=AsyncMock(return_value=stats_payload),
            collection_name="anime_database",
            vector_size=1024,
            image_vector_size=768,
            distance_metric="cosine",
        ),
        text_processor=SimpleNamespace(get_model_info=lambda: {"model": "text"}),
        vision_processor=SimpleNamespace(get_model_info=lambda: {"model": "vision"}),
    )


@pytest.mark.asyncio
async def test_get_stats_returns_error_when_qdrant_payload_contains_error() -> None:
    runtime = _runtime_with_stats({"error": "collection does not exist"})

    response = await admin.get_stats(
        runtime=runtime,
        request=vector_admin_pb2.GetStatsRequest(),
        context=None,
    )

    assert response.stats_json == ""
    assert response.error.code == "GET_STATS_FAILED"
    assert response.error.message == "collection does not exist"
    assert response.error.retryable is True

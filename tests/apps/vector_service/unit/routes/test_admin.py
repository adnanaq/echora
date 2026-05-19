from __future__ import annotations

import json
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


@pytest.mark.asyncio
async def test_get_stats_returns_payload_when_qdrant_stats_are_valid() -> None:
    runtime = _runtime_with_stats(
        {"collection_name": "anime_database", "total_documents": 42}
    )

    response = await admin.get_stats(
        runtime=runtime,
        request=vector_admin_pb2.GetStatsRequest(),
        context=None,
    )

    payload = json.loads(response.stats_json)
    assert payload["stats"]["collection_name"] == "anime_database"
    assert payload["stats"]["total_documents"] == 42
    assert payload["collection"]["collection_name"] == "anime_database"
    assert payload["processors"]["text_processor"]["model"] == "text"
    assert payload["processors"]["vision_processor"]["model"] == "vision"
    assert not response.HasField("error")


@pytest.mark.asyncio
async def test_get_stats_returns_retryable_error_when_qdrant_raises() -> None:
    runtime = _runtime_with_stats({})
    runtime.qdrant_client.get_stats = AsyncMock(
        side_effect=RuntimeError("qdrant unavailable")
    )

    response = await admin.get_stats(
        runtime=runtime,
        request=vector_admin_pb2.GetStatsRequest(),
        context=None,
    )

    assert response.stats_json == ""
    assert response.error.code == "GET_STATS_FAILED"
    assert response.error.message == "qdrant unavailable"
    assert response.error.retryable is True

"""Unit tests for the search route ValueError scoping fix."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.protobuf import struct_pb2
from vector_proto.v1 import vector_search_pb2
from vector_service.routes import search as search_route


def _runtime(
    *,
    text_embedding: list[float] | None = None,
    search_results: list[dict] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text_processor=SimpleNamespace(
            encode_text=AsyncMock(return_value=text_embedding or [0.1] * 4)
        ),
        vision_processor=SimpleNamespace(
            encode_image=AsyncMock(return_value=None)
        ),
        qdrant_client=SimpleNamespace(
            search=AsyncMock(return_value=search_results or [])
        ),
    )


@pytest.mark.asyncio
async def test_image_value_error_returns_invalid_image_input() -> None:
    """ValueError from _encode_image_bytes is labeled INVALID_IMAGE_INPUT."""
    runtime = _runtime()
    runtime.vision_processor.encode_image = AsyncMock(
        side_effect=ValueError("unsupported image format")
    )

    request = vector_search_pb2.SearchRequest(image=b"not-a-real-image")

    with patch(
        "vector_service.routes.search._encode_image_bytes",
        side_effect=ValueError("unsupported image format"),
    ):
        response = await search_route.search(runtime, request, context=None)

    assert response.error.code == "INVALID_IMAGE_INPUT"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_value_error_from_qdrant_search_is_not_labeled_invalid_image_input() -> None:
    """ValueError from downstream qdrant search is NOT misclassified as INVALID_IMAGE_INPUT."""
    runtime = _runtime()
    runtime.qdrant_client.search = AsyncMock(
        side_effect=ValueError("qdrant value error")
    )

    request = vector_search_pb2.SearchRequest(query_text="action anime")

    response = await search_route.search(runtime, request, context=None)

    # Must fall through to the generic SEARCH_FAILED handler, not INVALID_IMAGE_INPUT
    assert response.error.code == "SEARCH_FAILED"
    assert response.error.code != "INVALID_IMAGE_INPUT"


@pytest.mark.asyncio
async def test_successful_text_search_returns_data() -> None:
    hits = [{"id": "1", "score": 0.95, "payload": {"title": "Cowboy Bebop"}}]
    runtime = _runtime(search_results=hits)

    request = vector_search_pb2.SearchRequest(query_text="space western")

    response = await search_route.search(runtime, request, context=None)

    assert len(response.data) == 1
    assert response.data[0].id == "1"
    assert abs(response.data[0].similarity_score - 0.95) < 1e-6
    assert not response.HasField("error")

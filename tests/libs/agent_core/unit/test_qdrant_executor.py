"""Unit tests for ``QdrantExecutor`` dependency and image-query behavior."""

from unittest.mock import AsyncMock

import pytest

from agent_core.retrieval import QdrantExecutor
from agent_core.schemas import EntityType, SearchIntent


def test_qdrant_executor_requires_vision_processor() -> None:
    """Require explicit vision processor dependency in constructor."""
    qdrant = AsyncMock()
    text_processor = AsyncMock()

    with pytest.raises(TypeError):
        QdrantExecutor(
            qdrant=qdrant,
            text_processor=text_processor,
        )


@pytest.mark.asyncio
async def test_search_image_query_uses_vision_embedding() -> None:
    """Use vision processor for image-only query and call Qdrant with image embedding."""
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(
        return_value=[
            {
                "id": "019bce3b-d48e-3d81-61ba-518ea655b2de",
                "entity_type": "anime",
                "title": "!NVADE SHOW!",
                "similarity_score": 0.74,
            }
        ]
    )

    text_processor = AsyncMock()
    text_processor.encode_text = AsyncMock(return_value=None)

    vision_processor = AsyncMock()
    vision_processor.encode_image = AsyncMock(return_value=[0.1, 0.2, 0.3])

    executor = QdrantExecutor(
        qdrant=qdrant,
        text_processor=text_processor,
        vision_processor=vision_processor,
    )

    intent = SearchIntent(
        rationale="test image query",
        entity_type=EntityType.ANIME,
        query=None,
        image_query="data:image/png;base64,aW1hZ2UtYnl0ZXM=",
        filters={},
    )

    result = await executor.search(intent=intent, limit=5)

    assert result.count == 1
    assert result.raw_data[0]["id"] == "019bce3b-d48e-3d81-61ba-518ea655b2de"
    vision_processor.encode_image.assert_awaited_once()
    qdrant.search.assert_awaited_once_with(
        text_embedding=None,
        image_embedding=[0.1, 0.2, 0.3],
        entity_type="anime",
        limit=5,
        filters=None,
    )


@pytest.mark.asyncio
async def test_search_rejects_image_url_input() -> None:
    """Reject URL-based image queries and avoid calling Qdrant search."""
    qdrant = AsyncMock()
    text_processor = AsyncMock()
    text_processor.encode_text = AsyncMock(return_value=None)
    vision_processor = AsyncMock()

    executor = QdrantExecutor(
        qdrant=qdrant,
        text_processor=text_processor,
        vision_processor=vision_processor,
    )

    intent = SearchIntent(
        rationale="reject url",
        entity_type=EntityType.ANIME,
        query=None,
        image_query="https://example.com/image.png",
        filters={},
    )

    result = await executor.search(intent=intent, limit=5)

    assert result.count == 0
    assert result.raw_data == []
    assert result.summary == "Failed to embed image query."
    qdrant.search.assert_not_awaited()
    vision_processor.encode_image.assert_not_awaited()

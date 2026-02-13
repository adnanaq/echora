"""Unit tests for strict-contract QdrantClient."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from common.config import get_settings
from qdrant_client.models import OverwritePayloadOperation, SetPayloadOperation
from qdrant_db import QdrantClient
from qdrant_db.contracts import (
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    SearchFilterCondition,
    SearchRequest,
)
from qdrant_db.errors import DuplicateUpdateError, ValidationError


@pytest_asyncio.fixture
async def mock_client() -> QdrantClient:
    settings = get_settings()
    mock_async_client = AsyncMock()

    with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
        client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=mock_async_client,
        )

    return client


@pytest.mark.asyncio
async def test_search_text_only_uses_query_api(mock_client: QdrantClient) -> None:
    mock_client.client.query_points = AsyncMock(
        return_value=SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="anime-123",
                    payload={"title": "Naruto"},
                    score=0.91,
                )
            ]
        )
    )

    request = SearchRequest(text_embedding=[0.1] * 1024, limit=5)
    results = await mock_client.search(request)

    assert len(results) == 1
    assert results[0].id == "anime-123"
    assert results[0].payload["title"] == "Naruto"
    assert results[0].score == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_search_multivector_uses_prefetch_fusion(mock_client: QdrantClient) -> None:
    mock_client.client.query_points = AsyncMock(
        return_value=SimpleNamespace(points=[])
    )

    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        image_embedding=[0.2] * 768,
        limit=10,
    )
    await mock_client.search(request)

    call = mock_client.client.query_points.call_args.kwargs
    assert call["prefetch"]
    assert call["query"] is not None


@pytest.mark.asyncio
async def test_update_vectors_last_wins(mock_client: QdrantClient) -> None:
    mock_client.client.update_vectors = AsyncMock(return_value=None)

    result = await mock_client.update_vectors(
        updates=[
            BatchVectorUpdateItem(
                point_id="550e8400-e29b-41d4-a716-446655440000",
                vector_name="text_vector",
                vector_data=[0.1] * 1024,
            ),
            BatchVectorUpdateItem(
                point_id="550e8400-e29b-41d4-a716-446655440000",
                vector_name="text_vector",
                vector_data=[0.2] * 1024,
            ),
        ],
        dedup_policy="last-wins",
    )

    assert result.successful == 1
    assert result.failed == 0
    assert result.duplicates_removed == 1
    assert mock_client.client.update_vectors.call_count == 1


@pytest.mark.asyncio
async def test_update_vectors_duplicate_fail_raises(mock_client: QdrantClient) -> None:
    with pytest.raises(DuplicateUpdateError):
        await mock_client.update_vectors(
            updates=[
                BatchVectorUpdateItem(
                    point_id="550e8400-e29b-41d4-a716-446655440000",
                    vector_name="text_vector",
                    vector_data=[0.1] * 1024,
                ),
                BatchVectorUpdateItem(
                    point_id="550e8400-e29b-41d4-a716-446655440000",
                    vector_name="text_vector",
                    vector_data=[0.2] * 1024,
                ),
            ],
            dedup_policy="fail",
        )


@pytest.mark.asyncio
async def test_update_payload_merge_uses_set_payload_operation(
    mock_client: QdrantClient,
) -> None:
    mock_client.client.batch_update_points = AsyncMock(return_value=None)

    result = await mock_client.update_payload(
        updates=[
            BatchPayloadUpdateItem(
                point_id="550e8400-e29b-41d4-a716-446655440000",
                payload={"title": "A"},
            )
        ],
        mode="merge",
    )

    assert result.successful == 1
    call = mock_client.client.batch_update_points.call_args.kwargs
    operations = call["update_operations"]
    assert len(operations) == 1
    assert isinstance(operations[0], SetPayloadOperation)


@pytest.mark.asyncio
async def test_update_payload_overwrite_uses_overwrite_operation(
    mock_client: QdrantClient,
) -> None:
    mock_client.client.batch_update_points = AsyncMock(return_value=None)

    result = await mock_client.update_payload(
        updates=[
            BatchPayloadUpdateItem(
                point_id="550e8400-e29b-41d4-a716-446655440000",
                payload={"title": "A"},
            )
        ],
        mode="overwrite",
    )

    assert result.successful == 1
    call = mock_client.client.batch_update_points.call_args.kwargs
    operations = call["update_operations"]
    assert len(operations) == 1
    assert isinstance(operations[0], OverwritePayloadOperation)


@pytest.mark.asyncio
async def test_update_payload_empty_payload_raises(mock_client: QdrantClient) -> None:
    with pytest.raises(ValidationError):
        await mock_client.update_payload(
            updates=[
                BatchPayloadUpdateItem(
                    point_id="550e8400-e29b-41d4-a716-446655440000",
                    payload={},
                )
            ],
            mode="merge",
        )


def test_search_filter_condition_range_validation() -> None:
    with pytest.raises(ValueError):
        SearchFilterCondition(field="score", operator="range", value={})


def test_search_request_requires_embedding() -> None:
    with pytest.raises(ValueError):
        SearchRequest(limit=10)

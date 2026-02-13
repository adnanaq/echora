"""Unit tests for strict-contract QdrantClient."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from common.config import get_settings
from qdrant_client.models import OverwritePayloadOperation, SetPayloadOperation, SparseVector
from qdrant_db import QdrantClient
from qdrant_db.contracts import (
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    SearchFilterCondition,
    SearchRequest,
)
from qdrant_db.errors import DuplicateUpdateError, ValidationError
from vector_db_interface import VectorDocument


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


@pytest_asyncio.fixture
async def mock_sparse_client() -> QdrantClient:
    settings = get_settings()
    sparse_config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
        },
    )
    mock_async_client = AsyncMock()

    with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
        client = await QdrantClient.create(
            config=sparse_config,
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
async def test_search_sparse_only_uses_query_api(mock_sparse_client: QdrantClient) -> None:
    mock_sparse_client.client.query_points = AsyncMock(
        return_value=SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="anime-123",
                    payload={"title": "Naruto"},
                    score=0.83,
                )
            ]
        )
    )

    request = SearchRequest(
        sparse_embedding={"indices": [1, 5], "values": [0.7, 0.3]},
        limit=5,
    )
    results = await mock_sparse_client.search(request)

    assert len(results) == 1
    assert results[0].id == "anime-123"
    call = mock_sparse_client.client.query_points.call_args.kwargs
    assert call["using"] == "text_sparse_vector"
    assert isinstance(call["query"], SparseVector)


@pytest.mark.asyncio
async def test_search_text_sparse_uses_prefetch_fusion(
    mock_sparse_client: QdrantClient,
) -> None:
    mock_sparse_client.client.query_points = AsyncMock(
        return_value=SimpleNamespace(points=[])
    )

    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        sparse_embedding={"indices": [1, 4], "values": [0.9, 0.2]},
        limit=10,
    )
    await mock_sparse_client.search(request)

    call = mock_sparse_client.client.query_points.call_args.kwargs
    assert len(call["prefetch"]) == 2
    assert call["query"] is not None


@pytest.mark.asyncio
async def test_add_documents_sparse_payload_converts_to_sparse_vector(
    mock_sparse_client: QdrantClient,
) -> None:
    mock_sparse_client.client.upsert = AsyncMock(return_value=None)

    result = await mock_sparse_client.add_documents(
        documents=[
            VectorDocument(
                id="anime-123",
                vectors={
                    "text_vector": [0.1] * 1024,
                    "text_sparse_vector": {"indices": [1, 5], "values": [0.9, 0.4]},
                },
                payload={"title": "Naruto", "entity_type": "anime"},
            )
        ],
        batch_size=1,
    )

    assert result.successful == 1
    call = mock_sparse_client.client.upsert.call_args.kwargs
    points = call["points"]
    assert len(points) == 1
    assert isinstance(points[0].vector["text_sparse_vector"], SparseVector)


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


def test_search_request_accepts_sparse_embedding() -> None:
    request = SearchRequest(sparse_embedding={"indices": [1], "values": [0.2]}, limit=5)
    assert request.sparse_embedding is not None
    assert request.sparse_embedding.indices == [1]


def test_search_request_rejects_sparse_length_mismatch() -> None:
    with pytest.raises(ValueError):
        SearchRequest(
            sparse_embedding={"indices": [1, 2], "values": [0.2]},
            limit=5,
        )


@pytest.mark.asyncio
async def test_initialize_collection_includes_sparse_vectors_config() -> None:
    settings = get_settings()
    sparse_config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
        },
    )
    mock_async_client = AsyncMock()
    mock_async_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[])
    )
    mock_async_client.create_collection = AsyncMock(return_value=True)

    client = QdrantClient(config=sparse_config, async_qdrant_client=mock_async_client)
    await client.create_collection()

    call = mock_async_client.create_collection.call_args.kwargs
    assert call["sparse_vectors_config"] is not None
    assert "text_sparse_vector" in call["sparse_vectors_config"]

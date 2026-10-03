"""Unit tests for strict-contract QdrantClient."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
import pytest_asyncio
from common.config import get_settings
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    CollectionsResponse,
    QueryResponse,
    Record,
    ScoredPoint,
)
from qdrant_client.models import (
    OverwritePayloadOperation,
    QuantizationSearchParams,
    SearchParams,
    SetPayloadOperation,
    SparseVector,
)
from qdrant_db import QdrantClient
from qdrant_db.collection.manager import QdrantCollectionManager
from qdrant_db.contracts import (
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    SearchFilterCondition,
    SearchRequest,
    SparseVectorData,
)
from qdrant_db.errors import DuplicateUpdateError, PermanentQdrantError, ValidationError
from vector_db_interface import VectorDocument


@pytest_asyncio.fixture
async def mock_client() -> QdrantClient:
    settings = get_settings()
    async_client = create_autospec(AsyncQdrantClient, instance=True)

    with patch.object(QdrantCollectionManager, "initialize_collection", autospec=True):
        client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_client,
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
    async_client = create_autospec(AsyncQdrantClient, instance=True)

    with patch.object(QdrantCollectionManager, "initialize_collection", autospec=True):
        client = await QdrantClient.create(
            config=sparse_config,
            async_qdrant_client=async_client,
        )

    return client


@pytest.mark.asyncio
async def test_search_text_only_returns_hits_from_query_points(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(
        points=[
            ScoredPoint(
                version=0,
                id="anime-123",
                payload={"title": "Naruto"},
                score=0.91,
            )
        ]
    )

    request = SearchRequest(text_embedding=[0.1] * 1024, limit=5)
    results = await mock_client.search(request)

    assert len(results) == 1
    assert results[0].id == "anime-123"
    assert results[0].payload["title"] == "Naruto"
    assert results[0].score == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_search_text_and_image_sends_prefetch_fusion_query(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        image_embedding=[0.2] * 768,
        limit=10,
    )
    await mock_client.search(request)

    call = async_mock.query_points.call_args.kwargs
    assert call["prefetch"]
    assert call["query"] is not None


@pytest.mark.asyncio
async def test_search_sparse_only_returns_hits_from_query_points(
    mock_sparse_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.query_points.return_value = QueryResponse(
        points=[
            ScoredPoint(
                version=0,
                id="anime-123",
                payload={"title": "Naruto"},
                score=0.83,
            )
        ]
    )

    request = SearchRequest(
        sparse_embedding=SparseVectorData(indices=[1, 5], values=[0.7, 0.3]),
        limit=5,
    )
    results = await mock_sparse_client.search(request)

    assert len(results) == 1
    assert results[0].id == "anime-123"
    call = async_mock.query_points.call_args.kwargs
    assert call["using"] == "text_sparse_vector"
    assert isinstance(call["query"], SparseVector)


@pytest.mark.asyncio
async def test_search_text_and_sparse_sends_two_prefetch_branches(
    mock_sparse_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        sparse_embedding=SparseVectorData(indices=[1, 4], values=[0.9, 0.2]),
        limit=10,
    )
    await mock_sparse_client.search(request)

    call = async_mock.query_points.call_args.kwargs
    assert len(call["prefetch"]) == 2
    assert call["query"] is not None


@pytest.mark.asyncio
async def test_add_documents_sparse_payload_converts_to_sparse_vector(
    mock_sparse_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.upsert.return_value = None

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
    call = async_mock.upsert.call_args.kwargs
    points = call["points"]
    assert len(points) == 1
    assert isinstance(points[0].vector["text_sparse_vector"], SparseVector)


@pytest.mark.asyncio
async def test_update_vectors_duplicate_with_last_wins_keeps_last_update(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.update_vectors.return_value = None

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
    assert async_mock.update_vectors.call_count == 1


@pytest.mark.asyncio
async def test_update_vectors_duplicate_with_fail_policy_raises_duplicate_update_error(
    mock_client: QdrantClient,
) -> None:
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
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.batch_update_points.return_value = None

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
    call = async_mock.batch_update_points.call_args.kwargs
    operations = call["update_operations"]
    assert len(operations) == 1
    assert isinstance(operations[0], SetPayloadOperation)


@pytest.mark.asyncio
async def test_update_payload_overwrite_uses_overwrite_operation(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.batch_update_points.return_value = None

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
    call = async_mock.batch_update_points.call_args.kwargs
    operations = call["update_operations"]
    assert len(operations) == 1
    assert isinstance(operations[0], OverwritePayloadOperation)


def test_search_filter_condition_invalid_range_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SearchFilterCondition(field="score", operator="range", value={})


def test_search_request_without_embedding_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SearchRequest(limit=10)


def test_search_request_sparse_embedding_only_is_accepted() -> None:
    request = SearchRequest(
        sparse_embedding=SparseVectorData(indices=[1], values=[0.2]), limit=5
    )
    assert request.sparse_embedding is not None
    assert request.sparse_embedding.indices == [1]


def test_search_request_sparse_length_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SearchRequest(
            sparse_embedding=SparseVectorData(indices=[1, 2], values=[0.2]),
            limit=5,
        )


@pytest.mark.asyncio
async def test_initialize_collection_missing_collection_creates_it_with_sparse_vectors() -> (
    None
):
    settings = get_settings()
    sparse_config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
        },
    )
    async_client = create_autospec(AsyncQdrantClient, instance=True)
    async_client.get_collections.return_value = CollectionsResponse(collections=[])
    async_client.create_collection.return_value = True

    client = QdrantClient(config=sparse_config, async_qdrant_client=async_client)
    await client.create_collection()

    call = async_client.create_collection.call_args.kwargs
    assert call["sparse_vectors_config"] is not None
    assert "text_sparse_vector" in call["sparse_vectors_config"]


@pytest.mark.asyncio
async def test_search_telemetry_raises_still_returns_hits(
    mock_client: QdrantClient,
) -> None:
    class _ExplodingTelemetry:
        class DB_QUERY_DURATION:
            @staticmethod
            def record(*_a: object, **_kw: object) -> None:
                raise RuntimeError("telemetry boom")

        class DB_ERRORS:
            @staticmethod
            def add(*_a: object, **_kw: object) -> None:
                raise RuntimeError("telemetry boom")

    mock_client._telemetry = _ExplodingTelemetry()  # type: ignore[assignment]

    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(
        points=[
            ScoredPoint(version=0, id="anime-1", payload={"title": "Naruto"}, score=0.9)
        ]
    )

    from qdrant_db.contracts import SearchRequest

    results = await mock_client.search(
        SearchRequest(text_embedding=[0.1] * 1024, limit=1)
    )
    assert len(results) == 1
    assert results[0].id == "anime-1"


@pytest.mark.asyncio
async def test_add_documents_batch_size_zero_raises_validation_error(
    mock_client: QdrantClient,
) -> None:
    with pytest.raises(ValidationError, match="batch_size must be >= 1"):
        await mock_client.add_documents(
            documents=[
                VectorDocument(
                    id="anime-1",
                    vectors={"text_vector": [0.1] * 1024},
                    payload={},
                )
            ],
            batch_size=0,
        )


@pytest.mark.asyncio
async def test_add_documents_transient_failure_retries_and_succeeds(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    # First call raises a transient error; second succeeds.
    async_mock.upsert.side_effect = [ConnectionError("temporary"), None]

    result = await mock_client.add_documents(
        documents=[
            VectorDocument(
                id="anime-1",
                vectors={"text_vector": [0.1] * 1024},
                payload={},
            )
        ],
        max_retries=1,
        retry_delay=0.0,
    )

    assert result.successful == 1
    assert async_mock.upsert.call_count == 2


@pytest.mark.asyncio
async def test_add_documents_every_retry_fails_raises_permanent_error(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.upsert.side_effect = ConnectionError("always fails")

    with pytest.raises(PermanentQdrantError):
        await mock_client.add_documents(
            documents=[
                VectorDocument(
                    id="anime-1",
                    vectors={"text_vector": [0.1] * 1024},
                    payload={},
                )
            ],
            max_retries=1,
            retry_delay=0.0,
        )


@pytest.mark.asyncio
async def test_search_entity_type_adds_entity_type_filter(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await mock_client.search(
        SearchRequest(text_embedding=[0.1] * 1024, limit=5, entity_type="anime")
    )

    call = async_mock.query_points.call_args.kwargs
    qdrant_filter = call["query_filter"]
    assert qdrant_filter is not None
    assert any(getattr(c, "key", None) == "entity_type" for c in qdrant_filter.must)


@pytest.mark.asyncio
async def test_search_score_threshold_forwards_it_to_query_points(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await mock_client.search(
        SearchRequest(text_embedding=[0.1] * 1024, limit=5, score_threshold=0.75)
    )

    call = async_mock.query_points.call_args.kwargs
    assert call["score_threshold"] == pytest.approx(0.75)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_payload", [True, False])
async def test_search_single_vector_forwards_payload_choice(
    mock_client: QdrantClient, with_payload: bool
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await mock_client.search(
        SearchRequest(text_embedding=[0.1] * 1024, limit=5, with_payload=with_payload)
    )

    assert async_mock.query_points.call_args.kwargs["with_payload"] is with_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("with_payload", [True, False])
async def test_search_fusion_forwards_payload_choice(
    mock_sparse_client: QdrantClient, with_payload: bool
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await mock_sparse_client.search(
        SearchRequest(
            text_embedding=[0.1] * 1024,
            sparse_embedding=SparseVectorData(indices=[1, 4], values=[0.9, 0.2]),
            limit=10,
            with_payload=with_payload,
        )
    )

    assert async_mock.query_points.call_args.kwargs["with_payload"] is with_payload


def test_search_request_payload_choice_unset_defaults_to_true() -> None:
    assert SearchRequest(text_embedding=[0.1] * 1024).with_payload is True


@pytest.mark.asyncio
async def test_get_by_id_missing_point_returns_none(mock_client: QdrantClient) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.retrieve.return_value = []

    result = await mock_client.get_by_id("nonexistent-id")

    assert result is None


@pytest.mark.asyncio
async def test_get_by_id_existing_point_returns_id_and_payload(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    fake_point = Record(id="abc", payload={"title": "Bebop"}, vector=None)
    async_mock.retrieve.return_value = [fake_point]

    result = await mock_client.get_by_id("abc")

    assert result is not None
    assert result["id"] == "abc"
    assert result["payload"] == {"title": "Bebop"}
    assert "vector" not in result


@pytest.mark.asyncio
async def test_get_by_id_with_vectors_includes_vector(
    mock_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    fake_vector = {"text_vector": [0.1, 0.2]}
    fake_point = Record(id="abc", payload={"title": "Bebop"}, vector=fake_vector)
    async_mock.retrieve.return_value = [fake_point]

    result = await mock_client.get_by_id("abc", with_vectors=True)

    assert result is not None
    assert result["vector"] == fake_vector


@pytest.mark.asyncio
async def test_update_vectors_sparse_data_converts_to_sparse_vector(
    mock_sparse_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.update_vectors.return_value = None

    result = await mock_sparse_client.update_vectors(
        updates=[
            BatchVectorUpdateItem(
                point_id="550e8400-e29b-41d4-a716-446655440000",
                vector_name="text_sparse_vector",
                vector_data={"indices": [1, 3], "values": [0.8, 0.6]},
            )
        ]
    )

    assert result.successful == 1
    call = async_mock.update_vectors.call_args.kwargs
    points = call["points"]
    assert isinstance(points[0].vector["text_sparse_vector"], SparseVector)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_args",
    [
        {"text_embedding": [0.1] * 1024},
        {"text_embedding": [0.1] * 1024, "image_embedding": [0.2] * 768},
    ],
    ids=["single_vector", "fusion"],
)
async def test_search_unbatched_records_query_points_span(
    mock_client: QdrantClient, request_args: dict, span_exporter: InMemorySpanExporter
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])
    span_exporter.clear()

    await mock_client.search(SearchRequest(limit=5, **request_args))

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "qdrant.query_points"
    assert span.kind is trace.SpanKind.CLIENT
    assert span.attributes["db.system"] == "qdrant"
    assert span.attributes["db.collection.name"] == mock_client.collection_name


@pytest_asyncio.fixture
async def batching_client() -> QdrantClient:
    settings = get_settings()
    batching_config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "qdrant_query_batch_max_size": 16,
            "qdrant_query_batch_max_wait_ms": 5.0,
        },
    )
    with patch.object(QdrantCollectionManager, "initialize_collection", autospec=True):
        client = await QdrantClient.create(
            config=batching_config,
            async_qdrant_client=create_autospec(AsyncQdrantClient, instance=True),
        )
    yield client
    await client.close()


def _hybrid_request(point_id: int) -> SearchRequest:
    return SearchRequest(
        text_embedding=[0.1] * 1024,
        sparse_embedding=SparseVectorData(indices=[point_id], values=[1.0]),
        limit=10,
        with_payload=False,
    )


@pytest.mark.asyncio
async def test_search_concurrent_searches_share_one_query_batch_call(
    batching_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, batching_client._async_client)
    async_mock.query_batch_points.side_effect = lambda collection_name, requests: [
        QueryResponse(
            points=[
                ScoredPoint(
                    version=0,
                    id=request.prefetch[1].query.indices[0],
                    score=0.5,
                    payload=None,
                )
            ]
        )
        for request in requests
    ]

    results = await asyncio.gather(
        *(batching_client.search(_hybrid_request(point_id)) for point_id in (1, 2, 3))
    )

    assert async_mock.query_batch_points.call_count == 1
    assert len(async_mock.query_batch_points.call_args.kwargs["requests"]) == 3
    assert [[hit.id for hit in hits] for hits in results] == [["1"], ["2"], ["3"]]
    async_mock.query_points.assert_not_called()


@pytest.mark.asyncio
async def test_search_batched_searches_record_batch_size_on_span(
    batching_client: QdrantClient,
    span_exporter: InMemorySpanExporter,
) -> None:
    async_mock = cast(AsyncMock, batching_client._async_client)
    async_mock.query_batch_points.side_effect = lambda collection_name, requests: [
        QueryResponse(points=[]) for _ in requests
    ]
    span_exporter.clear()

    await asyncio.gather(
        *(batching_client.search(_hybrid_request(point_id)) for point_id in (1, 2))
    )

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    span = spans["qdrant.query_batch_points"]
    assert span.attributes["db.operation.batch.size"] == 2
    assert span.parent is not None
    assert span.parent.span_id == spans["batch.qdrant_query"].context.span_id


@pytest.mark.asyncio
async def test_search_lone_batched_search_records_no_batch_size(
    batching_client: QdrantClient,
    span_exporter: InMemorySpanExporter,
) -> None:
    async_mock = cast(AsyncMock, batching_client._async_client)
    async_mock.query_batch_points.side_effect = lambda collection_name, requests: [
        QueryResponse(points=[]) for _ in requests
    ]
    span_exporter.clear()

    await batching_client.search(_hybrid_request(1))

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert (
        "db.operation.batch.size" not in spans["qdrant.query_batch_points"].attributes
    )


def test_qdrant_config_query_batching_defaults_to_off() -> None:
    assert get_settings().qdrant.qdrant_query_batch_max_size == 1


@pytest.mark.asyncio
async def test_get_by_id_records_retrieve_span(
    mock_client: QdrantClient,
    span_exporter: InMemorySpanExporter,
) -> None:
    async_mock = cast(AsyncMock, mock_client._async_client)
    async_mock.retrieve.return_value = []
    span_exporter.clear()

    await mock_client.get_by_id("abc")

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "qdrant.retrieve"
    assert span.attributes["db.collection.name"] == mock_client.collection_name


@pytest_asyncio.fixture
async def tuned_search_client() -> QdrantClient:
    settings = get_settings()
    tuned_config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "qdrant_search_hnsw_ef": 512,
            "qdrant_search_rescore": True,
            "qdrant_search_oversampling": 4.0,
        },
    )
    with patch.object(QdrantCollectionManager, "initialize_collection", autospec=True):
        return await QdrantClient.create(
            config=tuned_config,
            async_qdrant_client=create_autospec(AsyncQdrantClient, instance=True),
        )


TUNED_SEARCH_PARAMS = SearchParams(
    hnsw_ef=512,
    quantization=QuantizationSearchParams(rescore=True, oversampling=4.0),
)


@pytest.mark.asyncio
async def test_search_hybrid_sends_search_params_on_dense_branch_only(
    tuned_search_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, tuned_search_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await tuned_search_client.search(
        SearchRequest(
            text_embedding=[0.1] * 1024,
            sparse_embedding=SparseVectorData(indices=[1], values=[0.9]),
            limit=10,
        )
    )

    prefetch = async_mock.query_points.call_args.kwargs["prefetch"]
    assert [(branch.using, branch.params) for branch in prefetch] == [
        ("text_vector", TUNED_SEARCH_PARAMS),
        ("text_sparse_vector", None),
    ]


@pytest.mark.asyncio
async def test_search_text_only_sends_search_params(
    tuned_search_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, tuned_search_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await tuned_search_client.search(
        SearchRequest(text_embedding=[0.1] * 1024, limit=5)
    )

    assert async_mock.query_points.call_args.kwargs["search_params"] == (
        TUNED_SEARCH_PARAMS
    )


@pytest.mark.asyncio
async def test_search_default_settings_sends_no_search_params(
    mock_sparse_client: QdrantClient,
) -> None:
    async_mock = cast(AsyncMock, mock_sparse_client._async_client)
    async_mock.query_points.return_value = QueryResponse(points=[])

    await mock_sparse_client.search(
        SearchRequest(
            text_embedding=[0.1] * 1024,
            sparse_embedding=SparseVectorData(indices=[1], values=[0.9]),
            limit=10,
        )
    )

    call = async_mock.query_points.call_args.kwargs
    assert call["search_params"] is None
    assert all(branch.params is None for branch in call["prefetch"])

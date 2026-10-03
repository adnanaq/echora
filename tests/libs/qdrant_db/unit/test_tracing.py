import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from common.config import get_settings
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from qdrant_db import QdrantClient
from qdrant_db.collection.manager import QdrantCollectionManager
from qdrant_db.contracts import BatchPayloadUpdateItem, BatchVectorUpdateItem
from qdrant_db.errors import PermanentQdrantError
from qdrant_db.tracing import qdrant_span
from vector_db_interface import VectorDocument

_SPANS = InMemorySpanExporter()
_TRACER_PROVIDER = TracerProvider()
_TRACER_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPANS))
trace.set_tracer_provider(_TRACER_PROVIDER)

COLLECTION = "span_test_collection"


def _client(async_client: AsyncMock) -> QdrantClient:
    return QdrantClient(
        config=get_settings().qdrant,
        async_qdrant_client=async_client,
        collection_name=COLLECTION,
    )


def _manager(async_client: AsyncMock) -> QdrantCollectionManager:
    return QdrantCollectionManager(
        config=get_settings().qdrant,
        async_client=async_client,
        collection_name=COLLECTION,
    )


def _async_client() -> AsyncMock:
    async_client = AsyncMock()
    async_client.get_collections.return_value = SimpleNamespace(collections=[])
    async_client.get_collection.return_value = SimpleNamespace(model_dump=lambda: {})
    async_client.count.return_value = SimpleNamespace(count=0)
    async_client.scroll.return_value = ([], None)
    return async_client


def _document() -> VectorDocument:
    return VectorDocument(
        id="point-1", vectors={"text_vector": [0.1] * 1024}, payload={}
    )


def test_qdrant_span_names_the_call_and_collection() -> None:
    _SPANS.clear()

    with qdrant_span("scroll", COLLECTION):
        pass

    (span,) = _SPANS.get_finished_spans()
    assert span.name == "qdrant.scroll"
    assert span.kind is trace.SpanKind.CLIENT
    assert span.attributes == {
        "db.system": "qdrant",
        "db.operation.name": "scroll",
        "db.collection.name": COLLECTION,
    }


def test_qdrant_span_without_collection_has_no_collection_attribute() -> None:
    _SPANS.clear()

    with qdrant_span("get_collections"):
        pass

    (span,) = _SPANS.get_finished_spans()
    assert "db.collection.name" not in span.attributes


def test_qdrant_span_marks_failed_calls_as_errors() -> None:
    _SPANS.clear()

    with pytest.raises(RuntimeError), qdrant_span("upsert", COLLECTION):
        raise RuntimeError("qdrant down")

    (span,) = _SPANS.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_qdrant_span_lasts_as_long_as_the_awaited_call() -> None:
    _SPANS.clear()

    with qdrant_span("query_points", COLLECTION):
        await asyncio.sleep(0.02)

    (span,) = _SPANS.get_finished_spans()
    assert (span.end_time - span.start_time) / 1e9 >= 0.02


CLIENT_CALLS: list[
    tuple[str, Callable[[QdrantClient], Awaitable[object]], list[str]]
] = [
    ("health_check", lambda client: client.health_check(), ["qdrant.get_collections"]),
    (
        "get_stats",
        lambda client: client.get_stats(),
        ["qdrant.get_collection", "qdrant.count"],
    ),
    ("scroll", lambda client: client.scroll(limit=5), ["qdrant.scroll"]),
    (
        "add_documents",
        lambda client: client.add_documents([_document()]),
        ["qdrant.upsert"],
    ),
    (
        "update_vectors",
        lambda client: client.update_vectors(
            [
                BatchVectorUpdateItem(
                    point_id="point-1",
                    vector_name="text_vector",
                    vector_data=[0.1] * 1024,
                )
            ]
        ),
        ["qdrant.update_vectors"],
    ),
    (
        "update_payload",
        lambda client: client.update_payload(
            [BatchPayloadUpdateItem(point_id="point-1", key="title", payload={"a": 1})]
        ),
        ["qdrant.batch_update_points"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "expected_spans"),
    [(call, spans) for _, call, spans in CLIENT_CALLS],
    ids=[name for name, _, _ in CLIENT_CALLS],
)
async def test_client_calls_record_one_span_per_qdrant_call(
    call: Callable[[QdrantClient], Awaitable[object]], expected_spans: list[str]
) -> None:
    client = _client(_async_client())
    _SPANS.clear()

    await call(client)

    spans = _SPANS.get_finished_spans()
    assert [span.name for span in spans] == expected_spans
    assert all(span.kind is trace.SpanKind.CLIENT for span in spans)


@pytest.mark.asyncio
async def test_each_upsert_retry_records_its_own_span() -> None:
    async_client = _async_client()
    async_client.upsert.side_effect = [ConnectionError("reset"), None]
    client = _client(async_client)
    _SPANS.clear()

    await client.add_documents([_document()], retry_delay=0)

    spans = _SPANS.get_finished_spans()
    assert [span.name for span in spans] == ["qdrant.upsert", "qdrant.upsert"]
    assert [span.status.status_code for span in spans] == [
        StatusCode.ERROR,
        StatusCode.UNSET,
    ]


@pytest.mark.asyncio
async def test_failed_stats_call_records_an_error_span() -> None:
    async_client = _async_client()
    async_client.get_collection.side_effect = RuntimeError("qdrant down")
    client = _client(async_client)
    _SPANS.clear()

    with pytest.raises(PermanentQdrantError):
        await client.get_stats()

    (span,) = _SPANS.get_finished_spans()
    assert span.name == "qdrant.get_collection"
    assert span.status.status_code is StatusCode.ERROR


MANAGER_CALLS: list[
    tuple[str, Callable[[QdrantCollectionManager], Awaitable[object]], list[str]]
] = [
    (
        "collection_exists",
        lambda manager: manager.collection_exists(),
        ["qdrant.get_collections"],
    ),
    (
        "delete_collection",
        lambda manager: manager.delete_collection(),
        ["qdrant.delete_collection"],
    ),
    (
        "initialize_collection",
        lambda manager: manager.initialize_collection(),
        ["qdrant.get_collections", "qdrant.create_collection"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "expected_spans"),
    [(call, spans) for _, call, spans in MANAGER_CALLS],
    ids=[name for name, _, _ in MANAGER_CALLS],
)
async def test_manager_calls_record_one_span_per_qdrant_call(
    call: Callable[[QdrantCollectionManager], Awaitable[object]],
    expected_spans: list[str],
) -> None:
    manager = _manager(_async_client())
    _SPANS.clear()

    with patch.object(manager, "setup_payload_indexes", new=AsyncMock()):
        await call(manager)

    spans = _SPANS.get_finished_spans()
    assert [span.name for span in spans] == expected_spans
    assert all(
        span.attributes["db.collection.name"] == COLLECTION
        for span in spans
        if span.name != "qdrant.get_collections"
    )


@pytest.mark.asyncio
async def test_payload_index_setup_records_a_span_per_index() -> None:
    manager = _manager(_async_client())
    _SPANS.clear()

    await manager.setup_payload_indexes()

    spans = _SPANS.get_finished_spans()
    indexed_fields = get_settings().qdrant.qdrant_indexed_payload_fields
    assert [span.name for span in spans] == ["qdrant.create_payload_index"] * len(
        indexed_fields
    )


@pytest.mark.asyncio
async def test_compatibility_check_records_a_get_collection_span() -> None:
    async_client = _async_client()
    async_client.get_collection.side_effect = RuntimeError("qdrant down")
    manager = _manager(async_client)
    _SPANS.clear()

    with pytest.raises(RuntimeError):
        await manager._validate_compatibility()

    (span,) = _SPANS.get_finished_spans()
    assert span.name == "qdrant.get_collection"
    assert span.status.status_code is StatusCode.ERROR

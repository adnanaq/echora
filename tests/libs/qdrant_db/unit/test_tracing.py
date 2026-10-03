import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from common.config import get_settings
from opentelemetry import trace
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


def test_qdrant_span_with_collection_records_call_and_collection(
    span_exporter: InMemorySpanExporter,
) -> None:
    with qdrant_span("scroll", COLLECTION):
        pass

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "qdrant.scroll"
    assert span.kind is trace.SpanKind.CLIENT
    assert span.attributes == {
        "db.system": "qdrant",
        "db.operation.name": "scroll",
        "db.collection.name": COLLECTION,
    }


def test_qdrant_span_without_collection_has_no_collection_attribute(
    span_exporter: InMemorySpanExporter,
) -> None:
    with qdrant_span("get_collections"):
        pass

    (span,) = span_exporter.get_finished_spans()
    assert "db.collection.name" not in span.attributes


def test_qdrant_span_failed_call_marks_span_as_error(
    span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError), qdrant_span("upsert", COLLECTION):
        raise RuntimeError("qdrant down")

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_qdrant_span_awaited_call_spans_full_duration(
    span_exporter: InMemorySpanExporter,
) -> None:
    with qdrant_span("query_points", COLLECTION):
        await asyncio.sleep(0.02)

    (span,) = span_exporter.get_finished_spans()
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
async def test_qdrant_client_each_method_records_one_span_per_qdrant_call(
    call: Callable[[QdrantClient], Awaitable[object]],
    expected_spans: list[str],
    span_exporter: InMemorySpanExporter,
) -> None:
    client = _client(_async_client())
    span_exporter.clear()

    await call(client)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == expected_spans
    assert all(span.kind is trace.SpanKind.CLIENT for span in spans)


@pytest.mark.asyncio
async def test_add_documents_retried_upsert_records_span_per_attempt(
    span_exporter: InMemorySpanExporter,
) -> None:
    async_client = _async_client()
    async_client.upsert.side_effect = [ConnectionError("reset"), None]
    client = _client(async_client)
    span_exporter.clear()

    await client.add_documents([_document()], retry_delay=0)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["qdrant.upsert", "qdrant.upsert"]
    assert [span.status.status_code for span in spans] == [
        StatusCode.ERROR,
        StatusCode.UNSET,
    ]


@pytest.mark.asyncio
async def test_get_stats_failed_qdrant_call_records_error_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    async_client = _async_client()
    async_client.get_collection.side_effect = RuntimeError("qdrant down")
    client = _client(async_client)
    span_exporter.clear()

    with pytest.raises(PermanentQdrantError):
        await client.get_stats()

    (span,) = span_exporter.get_finished_spans()
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
async def test_collection_manager_each_method_records_one_span_per_qdrant_call(
    call: Callable[[QdrantCollectionManager], Awaitable[object]],
    expected_spans: list[str],
    span_exporter: InMemorySpanExporter,
) -> None:
    manager = _manager(_async_client())
    span_exporter.clear()

    with patch.object(manager, "setup_payload_indexes", new=AsyncMock()):
        await call(manager)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == expected_spans
    assert all(
        span.attributes["db.collection.name"] == COLLECTION
        for span in spans
        if span.name != "qdrant.get_collections"
    )


@pytest.mark.asyncio
async def test_setup_payload_indexes_each_index_records_one_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    manager = _manager(_async_client())
    span_exporter.clear()

    await manager.setup_payload_indexes()

    spans = span_exporter.get_finished_spans()
    indexed_fields = get_settings().qdrant.qdrant_indexed_payload_fields
    assert [span.name for span in spans] == ["qdrant.create_payload_index"] * len(
        indexed_fields
    )


@pytest.mark.asyncio
async def test_validate_compatibility_failed_get_collection_records_error_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    async_client = _async_client()
    async_client.get_collection.side_effect = RuntimeError("qdrant down")
    manager = _manager(async_client)
    span_exporter.clear()

    with pytest.raises(RuntimeError):
        await manager._validate_compatibility()

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "qdrant.get_collection"
    assert span.status.status_code is StatusCode.ERROR

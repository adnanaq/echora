#!/usr/bin/env python3
"""Integration tests for strict-contract QdrantClient APIs.

This suite intentionally covers only the current public API:
- add_documents
- update_vectors
- update_payload
- search(SearchRequest)

Deprecated API surfaces are intentionally not tested:
- update_single_point_vector
- update_batch_point_vectors
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_db import QdrantClient
from qdrant_db.contracts import (
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    SearchFilterCondition,
    SearchRequest,
    SparseVectorData,
)
from qdrant_db.errors import DuplicateUpdateError, PermanentQdrantError, ValidationError
from vector_db_interface import VectorDocument

pytestmark = pytest.mark.integration


def _one_hot(index: int, size: int) -> list[float]:
    """Build a dense one-hot vector."""
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def _make_doc(
    *,
    doc_id: str,
    title: str,
    text_vector: list[float] | None = None,
    image_vector: list[list[float]] | None = None,
    sparse: SparseVectorData | None = None,
    year: int | None = None,
) -> VectorDocument:
    """Create a VectorDocument for integration tests."""
    vectors: dict[str, list[float] | list[list[float]] | dict[str, list[int] | list[float]]] = {}

    if text_vector is None:
        text_vector = [0.1] * 1024
    vectors["text_vector"] = text_vector

    if image_vector is not None:
        vectors["image_vector"] = image_vector

    if sparse is not None:
        vectors["text_sparse_vector"] = {
            "indices": sparse.indices,
            "values": sparse.values,
        }

    payload: dict[str, str | int] = {
        "title": title,
        "entity_type": "anime",
    }
    if year is not None:
        payload["year"] = year

    return VectorDocument(id=doc_id, vectors=vectors, payload=payload)


@pytest.mark.asyncio
async def test_add_documents_success(client: QdrantClient) -> None:
    """Insert documents and verify operation counts."""
    docs = [
        _make_doc(doc_id=str(uuid.uuid4()), title="Add Doc 1"),
        _make_doc(doc_id=str(uuid.uuid4()), title="Add Doc 2"),
    ]
    result = await client.add_documents(docs, batch_size=2)
    assert result.total == 2
    assert result.successful == 2
    assert result.failed == 0


@pytest.mark.asyncio
async def test_add_documents_empty_input(client: QdrantClient) -> None:
    """Empty inserts return empty operation result."""
    result = await client.add_documents([], batch_size=10)
    assert result.total == 0
    assert result.successful == 0
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_empty_input(client: QdrantClient) -> None:
    """Empty vector updates return empty operation result."""
    result = await client.update_vectors([])
    assert result.total == 0
    assert result.successful == 0
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_dense_success(client: QdrantClient) -> None:
    """Update dense vectors through strict typed request."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Dense Update")], batch_size=1)

    result = await client.update_vectors(
        [
            BatchVectorUpdateItem(
                point_id=doc_id,
                vector_name="text_vector",
                vector_data=_one_hot(3, 1024),
            )
        ]
    )

    assert result.total == 1
    assert result.successful == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_multivector_image_success(client: QdrantClient) -> None:
    """Update multivector image field through strict typed request."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Image Update")], batch_size=1)

    result = await client.update_vectors(
        [
            BatchVectorUpdateItem(
                point_id=doc_id,
                vector_name="image_vector",
                vector_data=[_one_hot(0, 768), _one_hot(5, 768)],
            )
        ]
    )

    assert result.successful == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_rejects_invalid_vector_name(client: QdrantClient) -> None:
    """Invalid vector names are rejected by validation."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Invalid Name")], batch_size=1)

    with pytest.raises(ValidationError, match="Invalid vector name"):
        await client.update_vectors(
            [
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="does_not_exist",
                    vector_data=[0.1] * 1024,
                )
            ]
        )


@pytest.mark.asyncio
async def test_update_vectors_rejects_dimension_mismatch(client: QdrantClient) -> None:
    """Invalid dimensions are rejected before issuing update."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Bad Dim")], batch_size=1)

    with pytest.raises(ValidationError, match="dimension mismatch"):
        await client.update_vectors(
            [
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="text_vector",
                    vector_data=[0.1] * 512,
                )
            ]
        )


@pytest.mark.asyncio
async def test_update_vectors_sparse_success(client: QdrantClient) -> None:
    """Sparse vectors can be updated through strict typed request."""
    doc_id = str(uuid.uuid4())
    sparse = SparseVectorData(indices=[1, 2, 3], values=[0.2, 0.4, 0.6])
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Sparse Update", sparse=sparse)],
        batch_size=1,
    )

    result = await client.update_vectors(
        [
            BatchVectorUpdateItem(
                point_id=doc_id,
                vector_name="text_sparse_vector",
                vector_data=SparseVectorData(indices=[10, 20], values=[0.8, 0.5]),
            )
        ]
    )
    assert result.successful == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_dedup_last_wins(client: QdrantClient) -> None:
    """last-wins dedup removes duplicate updates for same point/vector."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Dedup")], batch_size=1)

    result = await client.update_vectors(
        updates=[
            BatchVectorUpdateItem(
                point_id=doc_id,
                vector_name="text_vector",
                vector_data=_one_hot(1, 1024),
            ),
            BatchVectorUpdateItem(
                point_id=doc_id,
                vector_name="text_vector",
                vector_data=_one_hot(2, 1024),
            ),
        ],
        dedup_policy="last-wins",
    )
    assert result.successful == 1
    assert result.duplicates_removed == 1


@pytest.mark.asyncio
async def test_update_vectors_dedup_fail_raises(client: QdrantClient) -> None:
    """fail dedup policy raises when duplicates are present."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Dedup Fail")], batch_size=1)

    with pytest.raises(DuplicateUpdateError):
        await client.update_vectors(
            updates=[
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="text_vector",
                    vector_data=_one_hot(1, 1024),
                ),
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="text_vector",
                    vector_data=_one_hot(2, 1024),
                ),
            ],
            dedup_policy="fail",
        )


@pytest.mark.asyncio
async def test_update_vectors_retries_transient_errors(client: QdrantClient) -> None:
    """Transient failures are retried and can recover."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Retry OK")], batch_size=1)

    original_update = client.client.update_vectors
    attempts = 0

    async def flaky_update(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary network failure")
        return await original_update(*args, **kwargs)

    with patch.object(client.client, "update_vectors", new=AsyncMock(side_effect=flaky_update)):
        result = await client.update_vectors(
            [
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="text_vector",
                    vector_data=_one_hot(4, 1024),
                )
            ],
            max_retries=2,
            retry_delay=0.01,
        )

    assert attempts == 2
    assert result.successful == 1


@pytest.mark.asyncio
async def test_update_vectors_no_retry_on_non_transient(client: QdrantClient) -> None:
    """Non-transient failures should not retry and should surface as permanent."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="No Retry")], batch_size=1)

    attempts = 0

    async def hard_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request shape")

    with patch.object(client.client, "update_vectors", new=AsyncMock(side_effect=hard_fail)):
        with pytest.raises(PermanentQdrantError, match="Failed to batch update vectors"):
            await client.update_vectors(
                [
                    BatchVectorUpdateItem(
                        point_id=doc_id,
                        vector_name="text_vector",
                        vector_data=_one_hot(5, 1024),
                    )
                ],
                max_retries=3,
                retry_delay=0.01,
            )

    assert attempts == 1


@pytest.mark.asyncio
async def test_update_vectors_retry_exhausted_raises(client: QdrantClient) -> None:
    """Persistent transient failures raise permanent error after retries."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Retry Exhausted")], batch_size=1)

    attempts = 0

    async def always_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise ConnectionError("connection timeout")

    with patch.object(client.client, "update_vectors", new=AsyncMock(side_effect=always_fail)):
        with pytest.raises(PermanentQdrantError, match="Failed to batch update vectors"):
            await client.update_vectors(
                [
                    BatchVectorUpdateItem(
                        point_id=doc_id,
                        vector_name="text_vector",
                        vector_data=_one_hot(6, 1024),
                    )
                ],
                max_retries=2,
                retry_delay=0.01,
            )

    assert attempts == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_update_payload_merge_and_overwrite(client: QdrantClient) -> None:
    """Payload updates support merge and overwrite modes."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Payload Test", year=2020)], batch_size=1)

    merge_result = await client.update_payload(
        [BatchPayloadUpdateItem(point_id=doc_id, payload={"tag": "action"})],
        mode="merge",
    )
    assert merge_result.successful == 1

    overwrite_result = await client.update_payload(
        [BatchPayloadUpdateItem(point_id=doc_id, payload={"title": "Overwritten", "entity_type": "anime"})],
        mode="overwrite",
    )
    assert overwrite_result.successful == 1

    payload = await client.get_by_id(doc_id, with_vectors=False)
    assert payload is not None
    assert payload.get("title") == "Overwritten"


@pytest.mark.asyncio
async def test_update_payload_rejects_empty_payload(client: QdrantClient) -> None:
    """Empty payload updates are rejected."""
    doc_id = str(uuid.uuid4())
    await client.add_documents([_make_doc(doc_id=doc_id, title="Empty Payload")], batch_size=1)

    with pytest.raises(ValidationError, match="Payload must not be empty"):
        await client.update_payload(
            [BatchPayloadUpdateItem(point_id=doc_id, payload={})],
            mode="merge",
        )


@pytest.mark.asyncio
async def test_search_text_only(client: QdrantClient) -> None:
    """Dense-only search returns expected nearest point."""
    target_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    docs = [
        _make_doc(doc_id=target_id, title="Dense Target", text_vector=_one_hot(0, 1024)),
        _make_doc(doc_id=other_id, title="Dense Other", text_vector=_one_hot(1, 1024)),
    ]
    await client.add_documents(docs, batch_size=2)

    results = await client.search(SearchRequest(text_embedding=_one_hot(0, 1024), limit=5))
    assert len(results) > 0
    assert results[0].id == target_id


@pytest.mark.asyncio
async def test_sparse_vector_insertion_and_retrieval(client: QdrantClient):
    """Test end-to-end sparse vector insertion and retrieval."""
    test_id = str(uuid.uuid4())
    sparse_vector = SparseVectorData(indices=[10, 25, 103, 567], values=[0.7, 0.3, 0.5, 0.2])
    document = _make_doc(doc_id=test_id, title="Sparse Test Anime", sparse=sparse_vector)

    result = await client.add_documents([document], batch_size=1)
    assert result.successful == 1

    results = await client.search(SearchRequest(sparse_embedding=sparse_vector, limit=5))
    assert len(results) > 0
    assert any(hit.id == test_id for hit in results)


@pytest.mark.asyncio
async def test_sparse_vector_validation(client: QdrantClient):
    """Test sparse vector contract validation behavior."""
    test_id = str(uuid.uuid4())
    valid_sparse = SparseVectorData(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])
    result = await client.add_documents(
        [_make_doc(doc_id=test_id, title="Valid Sparse", sparse=valid_sparse)],
        batch_size=1,
    )
    assert result.successful == 1

    with pytest.raises(ValueError, match="indices and values must have the same length"):
        SparseVectorData(indices=[1, 2], values=[0.5])
    with pytest.raises(ValueError, match="indices must be non-negative"):
        SparseVectorData(indices=[1, -5, 10], values=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="indices must be unique"):
        SparseVectorData(indices=[1, 5, 1], values=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="indices must not exceed u32 maximum"):
        SparseVectorData(indices=[4294967296], values=[0.5])


@pytest.mark.asyncio
async def test_hybrid_search_dense_sparse_fusion(client: QdrantClient):
    """Test hybrid search with RRF fusion."""
    test_docs = []
    for i in range(5):
        sparse_vec = SparseVectorData(indices=[i * 10, i * 10 + 5, i * 10 + 8], values=[0.8, 0.5, 0.3])
        test_docs.append(
            _make_doc(
                doc_id=str(uuid.uuid4()),
                title=f"Hybrid Test {i}",
                text_vector=[0.1 + i * 0.01] * 1024,
                sparse=sparse_vec,
            )
        )
    result = await client.add_documents(test_docs, batch_size=5)
    assert result.successful == 5

    request = SearchRequest(
        text_embedding=[0.11] * 1024,
        sparse_embedding=SparseVectorData(indices=[10, 15, 18], values=[0.7, 0.4, 0.3]),
        fusion_method="rrf",
        limit=3,
    )
    results = await client.search(request)
    assert len(results) > 0
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_hybrid_search_dbsf_fusion(client: QdrantClient):
    """Test hybrid search with DBSF fusion."""
    test_id = str(uuid.uuid4())
    sparse_vec = SparseVectorData(indices=[5, 10, 15], values=[0.9, 0.6, 0.4])
    await client.add_documents(
        [_make_doc(doc_id=test_id, title="DBSF Test", text_vector=[0.2] * 1024, sparse=sparse_vec)],
        batch_size=1,
    )

    request = SearchRequest(
        text_embedding=[0.21] * 1024,
        sparse_embedding=SparseVectorData(indices=[5, 10], values=[0.8, 0.5]),
        fusion_method="dbsf",
        limit=5,
    )
    results = await client.search(request)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_sparse_vector_update(client: QdrantClient):
    """Test updating sparse vectors using strict update_vectors API."""
    test_id = str(uuid.uuid4())
    initial_sparse = SparseVectorData(indices=[1, 2, 3], values=[0.5, 0.5, 0.5])
    await client.add_documents(
        [_make_doc(doc_id=test_id, title="Sparse Update Test", sparse=initial_sparse)],
        batch_size=1,
    )

    updated_sparse = SparseVectorData(indices=[5, 10, 15, 20], values=[0.9, 0.7, 0.5, 0.3])
    result = await client.update_vectors(
        [
            BatchVectorUpdateItem(
                point_id=test_id,
                vector_name="text_sparse_vector",
                vector_data=updated_sparse,
            )
        ],
        dedup_policy="last-wins",
    )
    assert result.successful == 1

    results = await client.search(SearchRequest(sparse_embedding=updated_sparse, limit=5))
    assert any(hit.id == test_id for hit in results)


@pytest.mark.asyncio
async def test_sparse_only_keyword_search(client: QdrantClient):
    """Test sparse-only retrieval behavior for keyword-style vectors."""
    docs: list[tuple[str, VectorDocument, str]] = []
    keywords_map = {
        "action": [100, 101, 102],
        "romance": [200, 201, 202],
        "comedy": [300, 301, 302],
    }
    for genre, indices in keywords_map.items():
        doc_id = str(uuid.uuid4())
        sparse = SparseVectorData(indices=indices, values=[1.0, 0.8, 0.6])
        docs.append(
            (
                doc_id,
                _make_doc(doc_id=doc_id, title=f"{genre.title()} Anime", sparse=sparse),
                genre,
            )
        )
    await client.add_documents([d[1] for d in docs], batch_size=3)

    action_query = SparseVectorData(indices=[100, 101], values=[1.0, 0.9])
    results = await client.search(SearchRequest(sparse_embedding=action_query, limit=5))
    assert len(results) > 0


@pytest.mark.asyncio
async def test_sparse_vector_empty_results(client: QdrantClient):
    """Sparse query with non-existent indices returns valid result container."""
    request = SearchRequest(
        sparse_embedding=SparseVectorData(indices=[9999, 9998, 9997], values=[1.0, 1.0, 1.0]),
        limit=5,
    )
    results = await client.search(request)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_collection_compatibility_with_sparse_vectors(client: QdrantClient):
    """Collection exposes configured sparse vectors."""
    collection_info = await client.get_collection_info()
    sparse_vectors = getattr(collection_info.config.params, "sparse_vectors", None)
    assert sparse_vectors is not None
    assert isinstance(sparse_vectors, dict)
    assert "text_sparse_vector" in sparse_vectors


@pytest.mark.asyncio
async def test_sparse_vector_with_filters(client: QdrantClient):
    """Sparse retrieval combined with payload filtering."""
    docs = []
    for year in [2020, 2021, 2022]:
        sparse = SparseVectorData(indices=[50, 51, 52], values=[0.8, 0.6, 0.4])
        docs.append(
            _make_doc(
                doc_id=str(uuid.uuid4()),
                title=f"Anime {year}",
                sparse=sparse,
                year=year,
            )
        )
    await client.add_documents(docs, batch_size=3)

    request = SearchRequest(
        sparse_embedding=SparseVectorData(indices=[50, 51], values=[0.9, 0.7]),
        filters=[SearchFilterCondition(field="year", operator="eq", value=2021)],
        limit=5,
    )
    results = await client.search(request)
    assert len(results) > 0
    assert all(hit.payload.get("year") == 2021 for hit in results if "year" in hit.payload)


@pytest.mark.asyncio
async def test_sparse_vector_dict_format_compatibility(client: QdrantClient):
    """Dict-style sparse vectors are accepted and searchable."""
    test_id = str(uuid.uuid4())
    doc = VectorDocument(
        id=test_id,
        vectors={
            "text_vector": [0.1] * 1024,
            "text_sparse_vector": {
                "indices": [1, 2, 3, 4, 5],
                "values": [0.9, 0.8, 0.7, 0.6, 0.5],
            },
        },
        payload={"title": "Dict Format Test", "entity_type": "anime"},
    )
    result = await client.add_documents([doc], batch_size=1)
    assert result.successful == 1

    results = await client.search(
        SearchRequest(
            sparse_embedding=SparseVectorData(indices=[1, 2, 3], values=[1.0, 0.9, 0.8]),
            limit=5,
        )
    )
    assert any(hit.id == test_id for hit in results)


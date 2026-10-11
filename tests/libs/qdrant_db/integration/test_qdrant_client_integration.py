import uuid
from unittest.mock import patch

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
    vectors: dict[
        str, list[float] | list[list[float]] | dict[str, list[int] | list[float]]
    ] = {}

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
async def test_add_documents_two_documents_reports_both_successful(
    client: QdrantClient,
) -> None:
    docs = [
        _make_doc(doc_id=str(uuid.uuid4()), title="Add Doc 1"),
        _make_doc(doc_id=str(uuid.uuid4()), title="Add Doc 2"),
    ]
    result = await client.add_documents(docs, batch_size=2)
    assert result.total == 2
    assert result.successful == 2
    assert result.failed == 0


@pytest.mark.asyncio
async def test_add_documents_without_documents_reports_nothing(
    client: QdrantClient,
) -> None:
    result = await client.add_documents([], batch_size=10)
    assert result.total == 0
    assert result.successful == 0
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_without_updates_reports_nothing(
    client: QdrantClient,
) -> None:
    result = await client.update_vectors([])
    assert result.total == 0
    assert result.successful == 0
    assert result.failed == 0


@pytest.mark.asyncio
async def test_update_vectors_dense_vector_reports_success(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Dense Update")], batch_size=1
    )

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
async def test_update_vectors_image_multivector_reports_success(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Image Update")], batch_size=1
    )

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
async def test_update_vectors_unknown_vector_name_raises_validation_error(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Invalid Name")], batch_size=1
    )

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
@pytest.mark.parametrize(
    ("vector_data", "error"),
    [
        ([0.1] * 512, "dimension mismatch"),
    ],
)
async def test_update_vectors_wrong_dimension_raises_validation_error(
    client: QdrantClient, vector_data: list, error: str
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Bad Vector")], batch_size=1
    )

    with pytest.raises(ValidationError, match=error):
        await client.update_vectors(
            [
                BatchVectorUpdateItem(
                    point_id=doc_id,
                    vector_name="text_vector",
                    vector_data=vector_data,
                )
            ]
        )


@pytest.mark.asyncio
async def test_update_vectors_sparse_vector_reports_success(
    client: QdrantClient,
) -> None:
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
async def test_update_vectors_last_wins_duplicate_keeps_one_update(
    client: QdrantClient,
) -> None:
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
async def test_update_vectors_fail_policy_duplicate_raises(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Dedup Fail")], batch_size=1
    )

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
async def test_update_vectors_transient_error_retries_and_succeeds(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Retry OK")], batch_size=1
    )

    original_update = client._async_client.update_vectors
    attempts = 0

    async def flaky_update(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary network failure")
        return await original_update(*args, **kwargs)

    with patch.object(
        client._async_client, "update_vectors", autospec=True, side_effect=flaky_update
    ):
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
async def test_update_vectors_permanent_error_raises_without_retry(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="No Retry")], batch_size=1
    )

    attempts = 0

    async def hard_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request shape")

    with patch.object(
        client._async_client, "update_vectors", autospec=True, side_effect=hard_fail
    ):
        with pytest.raises(
            PermanentQdrantError, match="Failed to batch update vectors"
        ):
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
async def test_update_vectors_repeated_transient_error_raises_after_retries(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Retry Exhausted")], batch_size=1
    )

    attempts = 0

    async def always_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise ConnectionError("connection timeout")

    with patch.object(
        client._async_client, "update_vectors", autospec=True, side_effect=always_fail
    ):
        with pytest.raises(
            PermanentQdrantError, match="Failed to batch update vectors"
        ):
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

    assert attempts == 3


@pytest.mark.asyncio
async def test_update_payload_overwrite_after_merge_stores_overwritten_payload(
    client: QdrantClient,
) -> None:
    doc_id = str(uuid.uuid4())
    await client.add_documents(
        [_make_doc(doc_id=doc_id, title="Payload Test", year=2020)], batch_size=1
    )

    merge_result = await client.update_payload(
        [BatchPayloadUpdateItem(point_id=doc_id, payload={"tag": "action"})],
        mode="merge",
    )
    assert merge_result.successful == 1

    overwrite_result = await client.update_payload(
        [
            BatchPayloadUpdateItem(
                point_id=doc_id,
                payload={"title": "Overwritten", "entity_type": "anime"},
            )
        ],
        mode="overwrite",
    )
    assert overwrite_result.successful == 1

    result = await client.get_by_id(doc_id, with_vectors=False)
    assert result is not None
    assert result["payload"].get("title") == "Overwritten"


@pytest.mark.asyncio
async def test_search_dense_embedding_returns_nearest_point_first(
    client: QdrantClient,
) -> None:
    target_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    docs = [
        _make_doc(
            doc_id=target_id, title="Dense Target", text_vector=_one_hot(0, 1024)
        ),
        _make_doc(doc_id=other_id, title="Dense Other", text_vector=_one_hot(1, 1024)),
    ]
    await client.add_documents(docs, batch_size=2)

    results = await client.search(
        SearchRequest(text_embedding=_one_hot(0, 1024), limit=5)
    )
    assert len(results) > 0
    assert results[0].id == target_id


@pytest.mark.asyncio
async def test_search_sparse_embedding_returns_added_point(client: QdrantClient):
    test_id = str(uuid.uuid4())
    sparse_vector = SparseVectorData(
        indices=[10, 25, 103, 567], values=[0.7, 0.3, 0.5, 0.2]
    )
    document = _make_doc(
        doc_id=test_id, title="Sparse Test Anime", sparse=sparse_vector
    )

    result = await client.add_documents([document], batch_size=1)
    assert result.successful == 1

    results = await client.search(
        SearchRequest(sparse_embedding=sparse_vector, limit=5)
    )
    assert len(results) > 0
    assert any(hit.id == test_id for hit in results)


@pytest.mark.asyncio
async def test_add_documents_sparse_vector_reports_success(client: QdrantClient):
    sparse = SparseVectorData(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])

    result = await client.add_documents(
        [_make_doc(doc_id=str(uuid.uuid4()), title="Valid Sparse", sparse=sparse)],
        batch_size=1,
    )

    assert result.successful == 1


@pytest.mark.asyncio
async def test_search_rrf_fusion_returns_at_most_limit(client: QdrantClient):
    test_docs = []
    for i in range(5):
        sparse_vec = SparseVectorData(
            indices=[i * 10, i * 10 + 5, i * 10 + 8], values=[0.8, 0.5, 0.3]
        )
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
async def test_search_dbsf_fusion_returns_points(client: QdrantClient):
    test_id = str(uuid.uuid4())
    sparse_vec = SparseVectorData(indices=[5, 10, 15], values=[0.9, 0.6, 0.4])
    await client.add_documents(
        [
            _make_doc(
                doc_id=test_id,
                title="DBSF Test",
                text_vector=[0.2] * 1024,
                sparse=sparse_vec,
            )
        ],
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
async def test_update_vectors_new_sparse_vector_finds_point_by_it(client: QdrantClient):
    test_id = str(uuid.uuid4())
    initial_sparse = SparseVectorData(indices=[1, 2, 3], values=[0.5, 0.5, 0.5])
    await client.add_documents(
        [_make_doc(doc_id=test_id, title="Sparse Update Test", sparse=initial_sparse)],
        batch_size=1,
    )

    updated_sparse = SparseVectorData(
        indices=[5, 10, 15, 20], values=[0.9, 0.7, 0.5, 0.3]
    )
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

    results = await client.search(
        SearchRequest(sparse_embedding=updated_sparse, limit=5)
    )
    assert any(hit.id == test_id for hit in results)


@pytest.mark.asyncio
async def test_search_sparse_keyword_query_returns_points(client: QdrantClient):
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
async def test_search_unmatched_sparse_indices_returns_list(client: QdrantClient):
    request = SearchRequest(
        sparse_embedding=SparseVectorData(
            indices=[9999, 9998, 9997], values=[1.0, 1.0, 1.0]
        ),
        limit=5,
    )
    results = await client.search(request)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_get_stats_collection_lists_sparse_vector(client: QdrantClient):
    collection_info = await client.get_stats()
    sparse_vectors = (
        collection_info.get("config", {}).get("params", {}).get("sparse_vectors")
    )
    assert sparse_vectors is not None
    assert isinstance(sparse_vectors, dict)
    assert "text_sparse_vector" in sparse_vectors


@pytest.mark.asyncio
async def test_search_sparse_query_with_year_filter_returns_that_year(
    client: QdrantClient,
):
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
    assert all(
        hit.payload.get("year") == 2021 for hit in results if "year" in hit.payload
    )


@pytest.mark.asyncio
async def test_add_documents_sparse_vector_as_dict_is_searchable(client: QdrantClient):
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
            sparse_embedding=SparseVectorData(
                indices=[1, 2, 3], values=[1.0, 0.9, 0.8]
            ),
            limit=5,
        )
    )
    assert any(hit.id == test_id for hit in results)

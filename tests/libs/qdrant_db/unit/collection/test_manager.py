from unittest.mock import create_autospec

import pytest
from common.config import get_settings
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    CollectionConfig,
    CollectionInfo,
    CollectionParams,
    CollectionStatus,
    Distance,
    HnswConfig,
    OptimizersConfig,
    OptimizersStatusOneOf,
    SparseVectorParams,
    VectorParams,
    WalConfig,
)
from qdrant_db.collection.manager import QdrantCollectionManager
from qdrant_db.errors import CollectionCompatibilityError


def _make_async_client() -> AsyncQdrantClient:
    return create_autospec(AsyncQdrantClient, instance=True)


def _make_manager(
    async_client: AsyncQdrantClient,
    collection_name: str = "test_collection",
    **config_overrides: object,
) -> QdrantCollectionManager:
    settings = get_settings()
    config = settings.qdrant.model_copy(deep=True, update=config_overrides)
    return QdrantCollectionManager(
        config=config,
        async_client=async_client,
        collection_name=collection_name,
    )


def _make_vector_params(
    size: int = 1024,
    distance: Distance = Distance.COSINE,
) -> VectorParams:
    return VectorParams(size=size, distance=distance)


def _collection_info(
    vectors: object,
    sparse_vectors: dict[str, SparseVectorParams] | None = None,
) -> CollectionInfo:
    return CollectionInfo(
        status=CollectionStatus.GREEN,
        optimizer_status=OptimizersStatusOneOf.OK,
        indexed_vectors_count=0,
        points_count=0,
        segments_count=1,
        config=CollectionConfig(
            params=CollectionParams(
                vectors=vectors,  # type: ignore[arg-type]
                sparse_vectors=sparse_vectors,
                shard_number=1,
                replication_factor=1,
                write_consistency_factor=1,
                on_disk_payload=True,
            ),
            hnsw_config=HnswConfig(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
            optimizer_config=OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                flush_interval_sec=5,
            ),
            wal_config=WalConfig(wal_capacity_mb=32, wal_segments_ahead=0),
        ),
        payload_schema={},
    )


@pytest.mark.asyncio
async def test_collection_exists_present_returns_true() -> None:
    async_client = _make_async_client()
    async_client.collection_exists.return_value = True
    manager = _make_manager(async_client)

    assert await manager.collection_exists() is True


@pytest.mark.asyncio
async def test_collection_exists_missing_returns_false() -> None:
    async_client = _make_async_client()
    async_client.collection_exists.return_value = False
    manager = _make_manager(async_client)

    assert await manager.collection_exists() is False


@pytest.mark.asyncio
async def test_collection_exists_client_error_returns_false() -> None:
    async_client = _make_async_client()
    async_client.collection_exists.side_effect = RuntimeError("down")
    manager = _make_manager(async_client)

    assert await manager.collection_exists() is False


@pytest.mark.asyncio
async def test_delete_collection_success_returns_true() -> None:
    async_client = _make_async_client()
    async_client.delete_collection.return_value = True
    manager = _make_manager(async_client)

    assert await manager.delete_collection() is True


@pytest.mark.asyncio
async def test_delete_collection_client_error_returns_false() -> None:
    async_client = _make_async_client()
    async_client.delete_collection.side_effect = RuntimeError("fail")
    manager = _make_manager(async_client)

    assert await manager.delete_collection() is False


@pytest.mark.asyncio
async def test_create_collection_missing_collection_creates_and_returns_true() -> None:
    async_client = _make_async_client()
    async_client.collection_exists.return_value = False
    async_client.create_collection.return_value = True
    manager = _make_manager(async_client)

    created = await manager.create_collection()

    assert created is True
    async_client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_collection_existing_compatible_skips_create() -> None:
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "multivector_vectors": [],
        },
    )
    async_client = _make_async_client()
    async_client.collection_exists.return_value = True
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(
        vectors=vectors,
        sparse_vectors={"text_sparse_vector": SparseVectorParams()},
    )
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    await manager.initialize_collection()

    async_client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_collection_concurrent_already_exists_validates_schema() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "multivector_vectors": [],
        },
    )
    async_client = _make_async_client()
    async_client.collection_exists.return_value = False
    async_client.create_collection.side_effect = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b'{"status":{"error":"already exists"}}',
        headers={},
    )
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(
        vectors=vectors,
        sparse_vectors={"text_sparse_vector": SparseVectorParams()},
    )
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    await manager.initialize_collection()

    async_client.create_collection.assert_awaited_once()
    async_client.get_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_index_deletes_and_recreates_collection() -> None:
    async_client = _make_async_client()
    async_client.collection_exists.return_value = False
    async_client.delete_collection.return_value = True
    async_client.create_collection.return_value = True
    manager = _make_manager(async_client)

    cleared = await manager.clear_index()

    assert cleared is True
    async_client.delete_collection.assert_awaited_once()
    async_client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_payload_indexes_creates_configured_indexes() -> None:
    async_client = _make_async_client()
    manager = _make_manager(
        async_client, qdrant_indexed_payload_fields={"genre": "keyword"}
    )

    await manager.setup_payload_indexes()

    async_client.create_payload_index.assert_awaited_once()
    call = async_client.create_payload_index.call_args.kwargs
    assert call["field_name"] == "genre"


@pytest.mark.asyncio
async def test_validate_compatibility_single_vector_layout_raises_compatibility_error() -> (
    None
):
    async_client = _make_async_client()
    async_client.get_collection.return_value = _collection_info(
        vectors=VectorParams(size=1024, distance=Distance.COSINE)
    )
    manager = _make_manager(async_client)

    with pytest.raises(CollectionCompatibilityError, match="single-vector layout"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_missing_vector_raises_compatibility_error() -> (
    None
):
    async_client = _make_async_client()
    async_client.get_collection.return_value = _collection_info(vectors={})
    manager = _make_manager(async_client)

    with pytest.raises(CollectionCompatibilityError, match="missing required vector"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_size_mismatch_raises_compatibility_error() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant
    async_client = _make_async_client()
    vectors = {name: _make_vector_params(size=1) for name in config.vector_names}
    async_client.get_collection.return_value = _collection_info(vectors=vectors)
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    with pytest.raises(CollectionCompatibilityError, match="size mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_distance_mismatch_raises_compatibility_error() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant
    async_client = _make_async_client()
    vectors = {
        name: _make_vector_params(size=dim, distance=Distance.EUCLID)
        for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(vectors=vectors)
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    with pytest.raises(CollectionCompatibilityError, match="distance mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_multivector_mismatch_raises_compatibility_error() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True, update={"multivector_vectors": ["text_vector"]}
    )
    async_client = _make_async_client()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(vectors=vectors)
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    with pytest.raises(CollectionCompatibilityError, match="multivector mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_missing_sparse_config_raises_compatibility_error() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "multivector_vectors": [],
        },
    )
    async_client = _make_async_client()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(vectors=vectors)
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    with pytest.raises(
        CollectionCompatibilityError, match="missing sparse vector configuration"
    ):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_missing_sparse_vector_raises_compatibility_error() -> (
    None
):
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "multivector_vectors": [],
        },
    )
    async_client = _make_async_client()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    async_client.get_collection.return_value = _collection_info(
        vectors=vectors,
        sparse_vectors={"other_vec": SparseVectorParams()},
    )
    manager = QdrantCollectionManager(
        config=config, async_client=async_client, collection_name="test_collection"
    )

    with pytest.raises(
        CollectionCompatibilityError, match="missing required sparse vectors"
    ):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_initialize_collection_non_already_exists_unexpected_response_reraises() -> (
    None
):
    async_client = _make_async_client()
    async_client.collection_exists.return_value = False
    async_client.create_collection.side_effect = UnexpectedResponse(
        status_code=500,
        reason_phrase="Server Error",
        content=b"internal error",
        headers={},
    )
    manager = _make_manager(async_client)

    with pytest.raises(UnexpectedResponse):
        await manager.initialize_collection()


@pytest.mark.asyncio
async def test_clear_index_delete_failure_returns_false() -> None:
    async_client = _make_async_client()
    async_client.delete_collection.side_effect = RuntimeError("fail")
    manager = _make_manager(async_client)

    assert await manager.clear_index() is False
    async_client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_payload_indexes_empty_fields_skips_index_creation() -> None:
    async_client = _make_async_client()
    manager = _make_manager(async_client, qdrant_indexed_payload_fields={})

    await manager.setup_payload_indexes()

    async_client.create_payload_index.assert_not_awaited()

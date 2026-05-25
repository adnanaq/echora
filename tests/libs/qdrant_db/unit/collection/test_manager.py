"""Unit tests for QdrantCollectionManager."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from common.config import get_settings
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance
from qdrant_db.collection.manager import QdrantCollectionManager
from qdrant_db.errors import CollectionCompatibilityError


def _make_manager(
    async_client: AsyncMock,
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
    multivector_config: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        size=size, distance=distance, multivector_config=multivector_config
    )


def _collection_info(vectors: object, sparse_vectors: object = None) -> SimpleNamespace:
    params = SimpleNamespace(vectors=vectors)
    if sparse_vectors is not None:
        params.sparse_vectors = sparse_vectors
    return SimpleNamespace(config=SimpleNamespace(params=params))


# ---------------------------------------------------------------------------
# collection_exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collection_exists_true() -> None:
    mock = AsyncMock()
    mock.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="test_collection")]
        )
    )
    manager = _make_manager(mock)
    assert await manager.collection_exists() is True


@pytest.mark.asyncio
async def test_collection_exists_false() -> None:
    mock = AsyncMock()
    mock.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[SimpleNamespace(name="other")])
    )
    manager = _make_manager(mock)
    assert await manager.collection_exists() is False


@pytest.mark.asyncio
async def test_collection_exists_returns_false_on_exception() -> None:
    mock = AsyncMock()
    mock.get_collections = AsyncMock(side_effect=RuntimeError("down"))
    manager = _make_manager(mock)
    assert await manager.collection_exists() is False


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_collection_success() -> None:
    mock = AsyncMock()
    mock.delete_collection = AsyncMock(return_value=None)
    manager = _make_manager(mock)
    assert await manager.delete_collection() is True


@pytest.mark.asyncio
async def test_delete_collection_returns_false_on_exception() -> None:
    mock = AsyncMock()
    mock.delete_collection = AsyncMock(side_effect=RuntimeError("fail"))
    manager = _make_manager(mock)
    assert await manager.delete_collection() is False


# ---------------------------------------------------------------------------
# create_collection / initialize_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_collection_creates_when_missing() -> None:
    mock = AsyncMock()
    mock.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    mock.create_collection = AsyncMock(return_value=None)
    manager = _make_manager(mock)
    result = await manager.create_collection()
    assert result is True
    mock.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_collection_validates_when_exists() -> None:
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "multivector_vectors": [],
        },
    )
    mock = AsyncMock()
    mock.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="test_collection")]
        )
    )
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(
        return_value=_collection_info(
            vectors=vectors,
            sparse_vectors={"text_sparse_vector": {}},
        )
    )
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    await manager.initialize_collection()
    mock.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_collection_idempotent_on_concurrent_creation() -> None:
    """Existence check returns missing, create_collection races and raises already-exists,
    compatibility validation then succeeds — startup must not fail."""
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "primary_sparse_vector_name": "text_sparse_vector",
            "multivector_vectors": [],
        },
    )
    mock = AsyncMock()
    # First call: collection missing (triggers create path)
    mock.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    # create_collection raises "already exists" — simulates concurrent instance winning the race
    already_exists_exc = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b'{"status":{"error":"already exists"}}',
        headers={},
    )
    mock.create_collection = AsyncMock(side_effect=already_exists_exc)
    # get_collection returns a compatible schema for the compatibility check
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(
        return_value=_collection_info(
            vectors=vectors,
            sparse_vectors={"text_sparse_vector": {}},
        )
    )
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    # Must not raise — concurrent creation should be handled gracefully
    await manager.initialize_collection()
    mock.create_collection.assert_called_once()
    mock.get_collection.assert_called_once()


# ---------------------------------------------------------------------------
# clear_index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_index_deletes_then_creates() -> None:
    mock = AsyncMock()
    mock.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    mock.delete_collection = AsyncMock(return_value=None)
    mock.create_collection = AsyncMock(return_value=None)
    manager = _make_manager(mock)
    result = await manager.clear_index()
    assert result is True
    mock.delete_collection.assert_called_once()
    mock.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_clear_index_returns_false_if_delete_fails() -> None:
    mock = AsyncMock()
    mock.delete_collection = AsyncMock(side_effect=RuntimeError("fail"))
    manager = _make_manager(mock)
    assert await manager.clear_index() is False
    mock.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# setup_payload_indexes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_payload_indexes_skips_when_no_fields() -> None:
    mock = AsyncMock()
    manager = _make_manager(mock, qdrant_indexed_payload_fields={})
    await manager.setup_payload_indexes()
    mock.create_payload_index.assert_not_called()


@pytest.mark.asyncio
async def test_setup_payload_indexes_creates_indexes() -> None:
    mock = AsyncMock()
    mock.create_payload_index = AsyncMock(return_value=None)
    manager = _make_manager(mock, qdrant_indexed_payload_fields={"genre": "keyword"})
    await manager.setup_payload_indexes()
    mock.create_payload_index.assert_called_once()
    call = mock.create_payload_index.call_args.kwargs
    assert call["field_name"] == "genre"


# ---------------------------------------------------------------------------
# _validate_compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_single_vector_layout() -> None:
    mock = AsyncMock()
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors=None))
    manager = _make_manager(mock)
    with pytest.raises(CollectionCompatibilityError, match="single-vector layout"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_missing_vector() -> None:
    mock = AsyncMock()
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors={}))
    manager = _make_manager(mock)
    with pytest.raises(CollectionCompatibilityError, match="missing required vector"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_size_mismatch() -> None:
    settings = get_settings()
    config = settings.qdrant
    mock = AsyncMock()
    vectors = {name: _make_vector_params(size=1) for name in config.vector_names}
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors=vectors))
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    with pytest.raises(CollectionCompatibilityError, match="size mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_distance_mismatch() -> None:
    settings = get_settings()
    config = settings.qdrant
    mock = AsyncMock()
    vectors = {
        name: _make_vector_params(size=dim, distance=Distance.EUCLID)
        for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors=vectors))
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    with pytest.raises(CollectionCompatibilityError, match="distance mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_multivector_mismatch() -> None:
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True, update={"multivector_vectors": ["text_vector"]}
    )
    mock = AsyncMock()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors=vectors))
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    with pytest.raises(CollectionCompatibilityError, match="multivector mismatch"):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_missing_sparse_config() -> None:
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "multivector_vectors": [],
        },
    )
    mock = AsyncMock()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(return_value=_collection_info(vectors=vectors))
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    with pytest.raises(
        CollectionCompatibilityError, match="missing sparse vector configuration"
    ):
        await manager._validate_compatibility()


@pytest.mark.asyncio
async def test_validate_compatibility_raises_for_missing_sparse_vector() -> None:
    settings = get_settings()
    config = settings.qdrant.model_copy(
        deep=True,
        update={
            "sparse_vector_names": ["text_sparse_vector"],
            "multivector_vectors": [],
        },
    )
    mock = AsyncMock()
    vectors = {
        name: _make_vector_params(size=dim) for name, dim in config.vector_names.items()
    }
    mock.get_collection = AsyncMock(
        return_value=_collection_info(vectors=vectors, sparse_vectors={"other_vec": {}})
    )
    manager = QdrantCollectionManager(
        config=config, async_client=mock, collection_name="test_collection"
    )
    with pytest.raises(
        CollectionCompatibilityError, match="missing required sparse vectors"
    ):
        await manager._validate_compatibility()

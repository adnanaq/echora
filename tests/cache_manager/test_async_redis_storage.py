"""
Comprehensive tests for AsyncRedisStorage with 100% coverage.

Tests the async Redis storage backend for Hishel HTTP caching, including:
- Initialization and client ownership
- Key generation methods
- Entry creation, retrieval, updates, and deletion
- Stream operations (saving and retrieving)
- TTL handling and refresh on access
- Cleanup operations (soft/hard deletion)
- Edge cases and error scenarios
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hishel._core.models import Entry, EntryMeta, Headers, Request, Response

from src.cache_manager.async_redis_storage import AsyncRedisStorage

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """Create a mock async Redis client."""
    mock_client = AsyncMock()
    mock_client.from_url = AsyncMock(return_value=mock_client)
    mock_client.aclose = AsyncMock()
    mock_client.pipeline = MagicMock(return_value=AsyncMock())
    mock_client.hset = AsyncMock()
    mock_client.hgetall = AsyncMock(return_value={})
    mock_client.sadd = AsyncMock()
    mock_client.smembers = AsyncMock(return_value=set())
    mock_client.srem = AsyncMock()
    mock_client.expire = AsyncMock()
    mock_client.rpush = AsyncMock()
    mock_client.lrange = AsyncMock(return_value=[])
    mock_client.scan = AsyncMock(return_value=(0, []))
    mock_client.delete = AsyncMock()
    return mock_client


@pytest.fixture
def storage_with_mock_client(mock_redis_client: AsyncMock) -> AsyncRedisStorage:
    """Create AsyncRedisStorage with mocked Redis client."""
    return AsyncRedisStorage(
        client=mock_redis_client,
        default_ttl=3600.0,
        refresh_ttl_on_access=True,
        key_prefix="test_cache",
    )


@pytest.fixture
def mock_request() -> Request:
    """Create a mock HTTP request."""
    return Request(
        method="GET",
        url="https://api.example.com/anime/1",
        headers=Headers({}),
        metadata={},
    )


@pytest.fixture
def mock_response() -> Response:
    """Create a mock HTTP response with async stream."""

    async def mock_stream() -> AsyncIterator[bytes]:
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    return Response(
        status_code=200,
        headers=Headers({"Content-Type": "application/json"}),
        stream=mock_stream(),
        metadata={},
    )


# ============================================================================
# Test Class: Initialization and Client Ownership
# ============================================================================


class TestAsyncRedisStorageInit:
    """Test initialization and client ownership patterns."""

    @patch("src.cache_manager.async_redis_storage.Redis")
    def test_init_without_client_creates_new_client(
        self, mock_redis_class: MagicMock
    ) -> None:
        """Test initialization without client creates new Redis client."""
        mock_client = AsyncMock()
        mock_redis_class.from_url.return_value = mock_client

        storage = AsyncRedisStorage(
            redis_url="redis://localhost:6379/1",
            default_ttl=7200.0,
            refresh_ttl_on_access=False,
            key_prefix="custom_prefix",
        )

        # Should create client and own it with connection pool config
        assert storage._owns_client is True
        mock_redis_class.from_url.assert_called_once_with(
            "redis://localhost:6379/1",
            decode_responses=False,
            max_connections=100,
            socket_keepalive=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        assert storage.client == mock_client
        assert storage.default_ttl == 7200.0
        assert storage.refresh_ttl_on_access is False
        assert storage.key_prefix == "custom_prefix"

    def test_init_with_client_does_not_own_client(
        self, mock_redis_client: AsyncMock
    ) -> None:
        """Test initialization with provided client does not own it."""
        storage = AsyncRedisStorage(client=mock_redis_client, default_ttl=1800.0)

        # Should not own the client
        assert storage._owns_client is False
        assert storage.client == mock_redis_client
        assert storage.default_ttl == 1800.0
        assert storage.refresh_ttl_on_access is True  # default
        assert storage.key_prefix == "hishel_cache"  # default

    def test_init_defaults(self, mock_redis_client: AsyncMock) -> None:
        """Test initialization with default parameters."""
        storage = AsyncRedisStorage(client=mock_redis_client)

        assert storage.default_ttl is None
        assert storage.refresh_ttl_on_access is True
        assert storage.key_prefix == "hishel_cache"

    @patch("src.cache_manager.async_redis_storage.Redis")
    def test_init_without_client_uses_default_url(
        self, mock_redis_class: MagicMock
    ) -> None:
        """Test initialization without client uses default Redis URL."""
        mock_client = AsyncMock()
        mock_redis_class.from_url.return_value = mock_client

        storage = AsyncRedisStorage()

        mock_redis_class.from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=False,
            max_connections=100,
            socket_keepalive=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        assert storage._owns_client is True


# ============================================================================
# Test Class: Key Generation Methods
# ============================================================================


class TestKeyGeneration:
    """Test Redis key generation methods."""

    def test_make_key(self, storage_with_mock_client: AsyncRedisStorage) -> None:
        """Test _make_key generates correct format."""
        key = storage_with_mock_client._make_key("entry", "abc123")
        assert key == "test_cache:entry:abc123"

    def test_entry_key(self, storage_with_mock_client: AsyncRedisStorage) -> None:
        """Test _entry_key generates correct entry key."""
        entry_id = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        key = storage_with_mock_client._entry_key(entry_id)
        assert key == "test_cache:entry:12345678-1234-1234-1234-123456789abc"

    def test_stream_key(self, storage_with_mock_client: AsyncRedisStorage) -> None:
        """Test _stream_key generates correct stream key."""
        entry_id = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        key = storage_with_mock_client._stream_key(entry_id)
        assert key == "test_cache:stream:12345678-1234-1234-1234-123456789abc"

    def test_index_key_with_str_cache_key(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _index_key with string cache key."""
        cache_key = "api.example.com/anime/1"
        key = storage_with_mock_client._index_key(cache_key)
        expected_hex = cache_key.encode("utf-8").hex()
        assert key == f"test_cache:key_index:{expected_hex}"

    def test_index_key_with_bytes_cache_key(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _index_key with bytes cache key."""
        cache_key = b"api.example.com/anime/1"
        key = storage_with_mock_client._index_key(cache_key)
        expected_hex = cache_key.hex()
        assert key == f"test_cache:key_index:{expected_hex}"


# ============================================================================
# Test Class: Create Entry
# ============================================================================


class TestCreateEntry:
    """Test entry creation and storage."""

    @pytest.mark.asyncio
    async def test_create_entry_basic(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test basic entry creation."""
        cache_key = "test_key"
        entry_id = uuid.uuid4()

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Mock index key doesn't exist (will be created with TTL)
        mock_redis_client.ttl.return_value = -2

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            entry = await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
                id_=entry_id,
            )

            # Check entry structure
            assert entry.id == entry_id
            assert entry.request == mock_request
            assert entry.cache_key == cache_key.encode("utf-8")
            assert isinstance(entry.meta.created_at, float)

            # Check Redis operations
            mock_pipeline.hset.assert_called_once()
            mock_pipeline.sadd.assert_called_once()
            mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entry_generates_id_if_not_provided(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test entry creation generates UUID if not provided."""
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Mock index key doesn't exist (will be created with TTL)
        mock_redis_client.ttl.return_value = -2

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            entry = await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
                id_=None,  # No ID provided
            )

            # Check entry has generated UUID
            assert isinstance(entry.id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_create_entry_sets_ttl_from_request_metadata(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test entry creation uses TTL from request metadata."""
        request_with_ttl = Request(
            method="GET",
            url="https://api.example.com/anime/1",
            headers={},
            metadata={"hishel_ttl": 1200},  # Custom TTL
        )
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Mock index key doesn't exist (will be created with TTL)
        mock_redis_client.ttl.return_value = -2

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=request_with_ttl,
                response=mock_response,
                key=cache_key,
            )

            # Check expire was called with custom TTL (1200 seconds)
            assert mock_pipeline.expire.call_count == 2  # entry, index

    @pytest.mark.asyncio
    async def test_create_entry_uses_default_ttl(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test entry creation uses default TTL."""
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Mock index key doesn't exist (will be created with TTL)
        mock_redis_client.ttl.return_value = -2

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Check expire was called (storage has default_ttl=3600.0)
            assert mock_pipeline.expire.call_count == 2

    @pytest.mark.asyncio
    async def test_create_entry_no_ttl_when_none(
        self,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test entry creation doesn't set TTL when None."""
        storage = AsyncRedisStorage(
            client=mock_redis_client,
            default_ttl=None,  # No default TTL
        )
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Check expire was NOT called
            mock_pipeline.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_entry_preserves_persistent_index_ttl_minus_1(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test that persistent index keys (TTL=-1) are not given finite TTL.

        Bug fix test: When index key has TTL=-1 (persistent), we should NOT
        apply expire() to avoid converting persistent keys to temporary ones.

        Scenario:
        - Index key exists with TTL=-1 (no expiry)
        - New entry added with TTL=3600
        - Expected: expire() called ONLY for entry_key, NOT for index_key
        - Current (buggy): expire() called for BOTH (incorrect)
        """
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # CRITICAL: Index key has TTL=-1 (persistent, no expiry)
        mock_redis_client.ttl.return_value = -1

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Should call expire ONLY for entry_key (not index_key)
            expire_calls = mock_pipeline.expire.call_args_list
            assert len(expire_calls) == 1, (
                f"Expected 1 expire call (entry only), got {len(expire_calls)}. "
                "Persistent index key (TTL=-1) should NOT be given finite TTL."
            )

            # Verify it was the entry_key, not the index_key
            called_key = expire_calls[0][0][0]
            assert "entry:" in called_key, "Should only expire entry_key"
            assert (
                "key_index:" not in called_key
            ), "Should NOT expire persistent index_key"

    @pytest.mark.asyncio
    async def test_create_entry_extends_shorter_index_ttl(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test that index keys with shorter TTL are extended to match new entry TTL.

        Scenario:
        - Index key exists with TTL=1800 (30 min)
        - New entry added with TTL=3600 (60 min)
        - Expected: expire() called for BOTH entry_key AND index_key (extend)
        """
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Index key has shorter TTL (1800 < 3600)
        mock_redis_client.ttl.return_value = 1800

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Should call expire for BOTH entry_key and index_key
            expire_calls = mock_pipeline.expire.call_args_list
            assert len(expire_calls) == 2, (
                f"Expected 2 expire calls (entry + index), got {len(expire_calls)}. "
                "Index with shorter TTL should be extended."
            )

            # Verify both keys got expire called
            called_keys = [call[0][0] for call in expire_calls]
            assert any(
                "entry:" in key for key in called_keys
            ), "Should expire entry_key"
            assert any(
                "key_index:" in key for key in called_keys
            ), "Should extend index_key TTL"

    @pytest.mark.asyncio
    async def test_create_entry_preserves_longer_index_ttl(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test that index keys with longer TTL are not shrunk.

        Scenario:
        - Index key exists with TTL=7200 (2 hours)
        - New entry added with TTL=3600 (1 hour)
        - Expected: expire() called ONLY for entry_key, NOT for index_key (don't shrink)
        """
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Index key has longer TTL (7200 > 3600)
        mock_redis_client.ttl.return_value = 7200

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Should call expire ONLY for entry_key (not index_key)
            expire_calls = mock_pipeline.expire.call_args_list
            assert len(expire_calls) == 1, (
                f"Expected 1 expire call (entry only), got {len(expire_calls)}. "
                "Index with longer TTL should NOT be shrunk."
            )

            # Verify it was the entry_key, not the index_key
            called_key = expire_calls[0][0][0]
            assert "entry:" in called_key, "Should only expire entry_key"
            assert (
                "key_index:" not in called_key
            ), "Should NOT shrink index_key with longer TTL"

    @pytest.mark.asyncio
    async def test_create_entry_creates_missing_index_with_ttl(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_request: Request,
        mock_response: Response,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test that missing index keys (TTL=-2) get created with TTL.

        Scenario:
        - Index key doesn't exist (TTL=-2)
        - New entry added with TTL=3600
        - Expected: expire() called for BOTH entry_key AND index_key (create new)
        """
        cache_key = "test_key"

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock()

        # Index key doesn't exist (TTL=-2)
        mock_redis_client.ttl.return_value = -2

        with patch("src.cache_manager.async_redis_storage.pack") as mock_pack:
            mock_pack.return_value = b"serialized_entry"

            await storage_with_mock_client.create_entry(
                request=mock_request,
                response=mock_response,
                key=cache_key,
            )

            # Should call expire for BOTH entry_key and index_key
            expire_calls = mock_pipeline.expire.call_args_list
            assert len(expire_calls) == 2, (
                f"Expected 2 expire calls (entry + index), got {len(expire_calls)}. "
                "Missing index should be created with TTL."
            )

            # Verify both keys got expire called
            called_keys = [call[0][0] for call in expire_calls]
            assert any(
                "entry:" in key for key in called_keys
            ), "Should expire entry_key"
            assert any(
                "key_index:" in key for key in called_keys
            ), "Should create index_key with TTL"


# ============================================================================
# Test Class: Get Entries
# ============================================================================


class TestGetEntries:
    """Test entry retrieval."""

    @pytest.mark.asyncio
    async def test_get_entries_empty_index(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test get_entries returns empty list when index has no entries."""
        mock_redis_client.smembers.return_value = set()

        entries = await storage_with_mock_client.get_entries("nonexistent_key")

        assert entries == []
        mock_redis_client.smembers.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entries_retrieves_valid_entry(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test get_entries retrieves valid entry."""
        entry_id = uuid.uuid4()
        cache_key = "test_key"

        # Mock index lookup
        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}

        # Mock entry retrieval
        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = mock_entry
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}
            mock_redis_client.lrange.return_value = [
                b"chunk1",
                b"chunk2",
                AsyncRedisStorage._COMPLETE_CHUNK_MARKER,
            ]

            entries = await storage_with_mock_client.get_entries(cache_key)

            assert len(entries) == 1
            assert entries[0].id == entry_id
            mock_unpack.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entries_skips_missing_data(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test get_entries skips entries with missing data."""
        entry_id = uuid.uuid4()
        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}
        mock_redis_client.hgetall.return_value = {}  # No data field

        entries = await storage_with_mock_client.get_entries("test_key")

        assert entries == []

    @pytest.mark.asyncio
    async def test_get_entries_skips_invalid_entries(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test get_entries skips entries that aren't Entry objects."""
        entry_id = uuid.uuid4()
        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = "not_an_entry_object"  # Invalid type

            entries = await storage_with_mock_client.get_entries("test_key")

            assert entries == []

    @pytest.mark.asyncio
    async def test_get_entries_skips_entries_without_response(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test get_entries skips entries with None response."""
        entry_id = uuid.uuid4()
        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

        # Entry with None response
        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=None,  # No response
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = mock_entry

            entries = await storage_with_mock_client.get_entries("test_key")

            assert entries == []

    @pytest.mark.asyncio
    async def test_get_entries_skips_soft_deleted(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test get_entries skips soft deleted entries."""
        entry_id = uuid.uuid4()
        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

        # Soft deleted entry
        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(
                created_at=time.time(), deleted_at=time.time()
            ),  # Soft deleted
            cache_key=b"test_key",
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = mock_entry

            entries = await storage_with_mock_client.get_entries("test_key")

            assert entries == []

    @pytest.mark.asyncio
    async def test_get_entries_refreshes_ttl_on_access(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test get_entries refreshes TTL when refresh_ttl_on_access=True."""
        entry_id = uuid.uuid4()
        cache_key = "test_key"

        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}

        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = mock_entry
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}
            mock_redis_client.lrange.return_value = [
                AsyncRedisStorage._COMPLETE_CHUNK_MARKER
            ]

            await storage_with_mock_client.get_entries(cache_key)

            # Check expire was called (storage has default_ttl=3600.0 and refresh_ttl_on_access=True)
            assert mock_redis_client.expire.call_count == 2  # entry key + stream key

    @pytest.mark.asyncio
    async def test_get_entries_no_ttl_refresh_when_disabled(
        self,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test get_entries doesn't refresh TTL when refresh_ttl_on_access=False."""
        storage = AsyncRedisStorage(
            client=mock_redis_client,
            default_ttl=3600.0,
            refresh_ttl_on_access=False,  # Disabled
        )

        entry_id = uuid.uuid4()
        cache_key = "test_key"

        mock_redis_client.smembers.return_value = {str(entry_id).encode("utf-8")}

        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = mock_entry
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}
            mock_redis_client.lrange.return_value = [
                AsyncRedisStorage._COMPLETE_CHUNK_MARKER
            ]

            await storage.get_entries(cache_key)

            # Check expire was NOT called
            mock_redis_client.expire.assert_not_called()


# ============================================================================
# Test Class: Update Entry
# ============================================================================


class TestUpdateEntry:
    """Test entry updates."""

    @pytest.mark.asyncio
    async def test_update_entry_with_new_entry_object(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test update_entry with new Entry object."""
        entry_id = uuid.uuid4()
        cache_key = "test_key"

        # Mock current entry
        current_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        # Mock new entry
        new_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=304,
                headers={"ETag": "abc123"},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.execute = AsyncMock()

        with (
            patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack,
            patch("src.cache_manager.async_redis_storage.pack") as mock_pack,
        ):
            mock_unpack.return_value = current_entry
            mock_pack.return_value = b"serialized_new_entry"
            mock_redis_client.hgetall.return_value = {
                b"data": b"serialized_current_entry"
            }

            result = await storage_with_mock_client.update_entry(entry_id, new_entry)

            assert result == new_entry
            mock_pipeline.hset.assert_called_once()
            mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entry_with_callable(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test update_entry with callable transformer."""
        entry_id = uuid.uuid4()
        cache_key = "test_key"

        # Mock current entry
        current_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(
                status_code=200,
                headers={},
                stream=None,
                metadata={},
            ),
            meta=EntryMeta(created_at=time.time()),
            cache_key=cache_key.encode("utf-8"),
        )

        # Callable that transforms entry
        def transform_entry(entry: Entry) -> Entry:
            return Entry(
                id=entry.id,
                request=entry.request,
                response=Response(
                    status_code=304,
                    headers={"ETag": "transformed"},
                    stream=None,
                    metadata={},
                ),
                meta=entry.meta,
                cache_key=entry.cache_key,
            )

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.execute = AsyncMock()

        with (
            patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack,
            patch("src.cache_manager.async_redis_storage.pack") as mock_pack,
        ):
            mock_unpack.return_value = current_entry
            mock_pack.return_value = b"serialized_transformed_entry"
            mock_redis_client.hgetall.return_value = {
                b"data": b"serialized_current_entry"
            }

            result = await storage_with_mock_client.update_entry(
                entry_id, transform_entry
            )

            assert result is not None
            assert result.response.status_code == 304
            mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entry_not_found(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test update_entry returns None when entry not found."""
        entry_id = uuid.uuid4()
        mock_redis_client.hgetall.return_value = {}  # No data

        new_entry = Entry(
            id=entry_id,
            request=Request(
                method="GET", url="https://api.example.com", headers={}, metadata={}
            ),
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        result = await storage_with_mock_client.update_entry(entry_id, new_entry)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_entry_invalid_data(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test update_entry returns None when data is invalid."""
        entry_id = uuid.uuid4()
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = "not_an_entry"  # Invalid type

            new_entry = Entry(
                id=entry_id,
                request=Request(
                    method="GET", url="https://api.example.com", headers={}, metadata={}
                ),
                response=Response(
                    status_code=200, headers={}, stream=None, metadata={}
                ),
                meta=EntryMeta(created_at=time.time()),
                cache_key=b"test_key",
            )

            result = await storage_with_mock_client.update_entry(entry_id, new_entry)

            assert result is None

    @pytest.mark.asyncio
    async def test_update_entry_id_mismatch_raises_error(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test update_entry raises ValueError when entry ID mismatch."""
        entry_id = uuid.uuid4()
        different_id = uuid.uuid4()

        # Mock current entry
        current_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        # New entry with different ID
        new_entry = Entry(
            id=different_id,  # Different ID
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = current_entry
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

            with pytest.raises(ValueError, match="Entry ID mismatch"):
                await storage_with_mock_client.update_entry(entry_id, new_entry)

    @pytest.mark.asyncio
    async def test_update_entry_updates_cache_key_index(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test update_entry updates cache key index when cache key changes."""
        entry_id = uuid.uuid4()
        old_cache_key = "old_key"
        new_cache_key = "new_key"

        # Mock current entry
        current_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=old_cache_key.encode("utf-8"),
        )

        # New entry with different cache key
        new_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=new_cache_key.encode("utf-8"),  # Different cache key
        )

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.hset = MagicMock()
        mock_pipeline.srem = MagicMock()
        mock_pipeline.sadd = MagicMock()
        mock_pipeline.execute = AsyncMock()

        with (
            patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack,
            patch("src.cache_manager.async_redis_storage.pack") as mock_pack,
        ):
            mock_unpack.return_value = current_entry
            mock_pack.return_value = b"serialized_new_entry"
            mock_redis_client.hgetall.return_value = {
                b"data": b"serialized_current_entry"
            }

            result = await storage_with_mock_client.update_entry(entry_id, new_entry)

            assert result == new_entry
            # Check index was updated
            mock_pipeline.srem.assert_called_once()  # Remove from old index
            mock_pipeline.sadd.assert_called_once()  # Add to new index
            assert mock_pipeline.hset.call_count == 2  # data + cache_key


# ============================================================================
# Test Class: Remove Entry (Soft Delete)
# ============================================================================


class TestRemoveEntry:
    """Test soft deletion of entries."""

    @pytest.mark.asyncio
    async def test_remove_entry_soft_deletes(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test remove_entry marks entry as deleted."""
        entry_id = uuid.uuid4()

        # Mock current entry
        mock_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        with (
            patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack,
            patch("src.cache_manager.async_redis_storage.pack") as mock_pack,
        ):
            mock_unpack.return_value = mock_entry
            mock_pack.return_value = b"serialized_deleted_entry"
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

            with patch.object(
                storage_with_mock_client, "mark_pair_as_deleted"
            ) as mock_mark:
                deleted_entry = Entry(
                    id=entry_id,
                    request=mock_request,
                    response=Response(
                        status_code=200, headers={}, stream=None, metadata={}
                    ),
                    meta=EntryMeta(created_at=time.time(), deleted_at=time.time()),
                    cache_key=b"test_key",
                )
                mock_mark.return_value = deleted_entry

                await storage_with_mock_client.remove_entry(entry_id)

                mock_mark.assert_called_once_with(mock_entry)
                mock_redis_client.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_entry_not_found(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test remove_entry does nothing when entry not found."""
        entry_id = uuid.uuid4()
        mock_redis_client.hgetall.return_value = {}  # No data

        # Should not raise error
        await storage_with_mock_client.remove_entry(entry_id)

        mock_redis_client.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_entry_invalid_data(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test remove_entry does nothing when data is invalid."""
        entry_id = uuid.uuid4()
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = "not_an_entry"  # Invalid type

            await storage_with_mock_client.remove_entry(entry_id)

            mock_redis_client.hset.assert_not_called()


# ============================================================================
# Test Class: Stream Operations
# ============================================================================


class TestStreamOperations:
    """Test stream saving and retrieval."""

    @pytest.mark.asyncio
    async def test_save_stream_saves_chunks_to_redis(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _save_stream saves chunks to Redis."""

        async def mock_stream() -> AsyncIterator[bytes]:
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        entry_id = uuid.uuid4()
        stream_key = storage_with_mock_client._stream_key(entry_id)

        # Consume the wrapped stream
        chunks: List[bytes] = []
        async for chunk in storage_with_mock_client._save_stream(
            mock_stream(), entry_id, ttl=None
        ):
            chunks.append(chunk)

        # Check all chunks were yielded
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

        # Check Redis rpush was called for each chunk + completion marker
        assert mock_redis_client.rpush.call_count == 4  # 3 chunks + marker
        mock_redis_client.rpush.assert_any_call(stream_key, b"chunk1")
        mock_redis_client.rpush.assert_any_call(stream_key, b"chunk2")
        mock_redis_client.rpush.assert_any_call(stream_key, b"chunk3")
        mock_redis_client.rpush.assert_any_call(
            stream_key, AsyncRedisStorage._COMPLETE_CHUNK_MARKER
        )

    @pytest.mark.asyncio
    async def test_save_stream_sets_ttl_on_stream_key(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _save_stream sets TTL on the stream key."""

        async def mock_stream() -> AsyncIterator[bytes]:
            yield b"chunk1"

        entry_id = uuid.uuid4()
        stream_key = storage_with_mock_client._stream_key(entry_id)
        test_ttl = 60

        # Consume the wrapped stream
        async for _ in storage_with_mock_client._save_stream(
            mock_stream(), entry_id, ttl=test_ttl
        ):
            pass

        mock_redis_client.expire.assert_called_once_with(stream_key, test_ttl)

    @pytest.mark.asyncio
    async def test_stream_data_from_cache_yields_chunks(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _stream_data_from_cache yields chunks from Redis."""
        entry_id = uuid.uuid4()

        # Mock Redis lrange to return chunks
        mock_redis_client.lrange.return_value = [
            b"chunk1",
            b"chunk2",
            b"chunk3",
            AsyncRedisStorage._COMPLETE_CHUNK_MARKER,
        ]

        # Consume the stream
        chunks: List[bytes] = []
        async for chunk in storage_with_mock_client._stream_data_from_cache(entry_id):
            chunks.append(chunk)

        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
        mock_redis_client.lrange.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_data_from_cache_stops_at_marker(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _stream_data_from_cache stops at completion marker."""
        entry_id = uuid.uuid4()

        # Mock Redis lrange with marker in the middle
        mock_redis_client.lrange.return_value = [
            b"chunk1",
            AsyncRedisStorage._COMPLETE_CHUNK_MARKER,
            b"chunk2",  # Should not be yielded
        ]

        # Consume the stream
        chunks: List[bytes] = []
        async for chunk in storage_with_mock_client._stream_data_from_cache(entry_id):
            chunks.append(chunk)

        assert chunks == [b"chunk1"]  # Only first chunk before marker


# ============================================================================
# Test Class: TTL Handling
# ============================================================================


class TestTTLHandling:
    """Test TTL calculation and handling."""

    def test_get_entry_ttl_from_request_metadata(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _get_entry_ttl returns TTL from request metadata."""
        request = Request(
            method="GET",
            url="https://api.example.com",
            headers={},
            metadata={"hishel_ttl": 1800},
        )

        ttl = storage_with_mock_client._get_entry_ttl(request)

        assert ttl == 1800.0

    def test_get_entry_ttl_from_request_metadata_float(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _get_entry_ttl handles float TTL in metadata."""
        request = Request(
            method="GET",
            url="https://api.example.com",
            headers={},
            metadata={"hishel_ttl": 1800.5},
        )

        ttl = storage_with_mock_client._get_entry_ttl(request)

        assert ttl == 1800.5

    def test_get_entry_ttl_uses_default_when_no_metadata(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _get_entry_ttl returns default TTL when no metadata."""
        request = Request(
            method="GET",
            url="https://api.example.com",
            headers={},
            metadata={},
        )

        ttl = storage_with_mock_client._get_entry_ttl(request)

        assert ttl == 3600.0  # storage default_ttl

    def test_get_entry_ttl_ignores_non_numeric_metadata(
        self, storage_with_mock_client: AsyncRedisStorage
    ) -> None:
        """Test _get_entry_ttl ignores non-numeric TTL in metadata."""
        request = Request(
            method="GET",
            url="https://api.example.com",
            headers={},
            metadata={"hishel_ttl": "invalid"},
        )

        ttl = storage_with_mock_client._get_entry_ttl(request)

        assert ttl == 3600.0  # Falls back to default

    def test_get_entry_ttl_returns_none_when_no_default(
        self, mock_redis_client: AsyncMock
    ) -> None:
        """Test _get_entry_ttl returns None when no default TTL."""
        storage = AsyncRedisStorage(client=mock_redis_client, default_ttl=None)
        request = Request(
            method="GET",
            url="https://api.example.com",
            headers={},
            metadata={},
        )

        ttl = storage._get_entry_ttl(request)

        assert ttl is None


# ============================================================================
# Test Class: Cleanup Operations
# ============================================================================


class TestCleanupOperations:
    """Test cleanup of expired and soft-deleted entries."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_soft_deleted_entries(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test cleanup_expired removes soft-deleted entries."""
        entry_id = uuid.uuid4()
        entry_key = storage_with_mock_client._entry_key(entry_id)

        # Mock scan to return one entry key
        mock_redis_client.scan.side_effect = [
            (0, [entry_key.encode("utf-8")]),  # First call returns entry
        ]

        # Mock soft-deleted entry
        deleted_entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(
                created_at=time.time() - 10000,  # Old
                deleted_at=time.time() - 5000,  # Deleted long ago
            ),
            cache_key=b"test_key",
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = deleted_entry
            mock_redis_client.hgetall.return_value = {
                b"data": b"serialized_entry",
                b"cache_key": b"test_key",
            }

            with (
                patch.object(
                    storage_with_mock_client, "is_soft_deleted"
                ) as mock_is_deleted,
                patch.object(
                    storage_with_mock_client, "is_safe_to_hard_delete"
                ) as mock_is_safe,
            ):
                mock_is_deleted.return_value = True
                mock_is_safe.return_value = True

                # Mock pipeline for hard delete
                mock_pipeline = AsyncMock()
                mock_redis_client.pipeline.return_value = mock_pipeline
                mock_pipeline.delete = MagicMock()
                mock_pipeline.execute = AsyncMock()

                cleaned = await storage_with_mock_client.cleanup_expired()

                assert cleaned == 1
                mock_redis_client.srem.assert_called_once()  # Remove from index
                assert mock_pipeline.delete.call_count == 2  # entry + stream

    @pytest.mark.asyncio
    async def test_cleanup_expired_skips_non_deletable_entries(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
        mock_request: Request,
    ) -> None:
        """Test cleanup_expired skips entries that aren't safe to delete."""
        entry_id = uuid.uuid4()
        entry_key = storage_with_mock_client._entry_key(entry_id)

        # Mock scan
        mock_redis_client.scan.side_effect = [
            (0, [entry_key.encode("utf-8")]),
        ]

        # Mock entry (not soft deleted)
        entry = Entry(
            id=entry_id,
            request=mock_request,
            response=Response(status_code=200, headers={}, stream=None, metadata={}),
            meta=EntryMeta(created_at=time.time()),
            cache_key=b"test_key",
        )

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = entry
            mock_redis_client.hgetall.return_value = {b"data": b"serialized_entry"}

            with patch.object(
                storage_with_mock_client, "is_soft_deleted"
            ) as mock_is_deleted:
                mock_is_deleted.return_value = False  # Not deleted

                cleaned = await storage_with_mock_client.cleanup_expired()

                assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_handles_empty_scan(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test cleanup_expired handles empty scan results."""
        mock_redis_client.scan.return_value = (0, [])  # No keys

        cleaned = await storage_with_mock_client.cleanup_expired()

        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_handles_multiple_scan_iterations(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test cleanup_expired handles multiple scan iterations."""
        entry_key1 = b"test_cache:entry:id1"
        entry_key2 = b"test_cache:entry:id2"

        # Mock scan to return multiple batches
        mock_redis_client.scan.side_effect = [
            (1, [entry_key1]),  # First batch, cursor=1
            (0, [entry_key2]),  # Second batch, cursor=0 (done)
        ]

        # Mock hgetall to return invalid data (skip entries)
        mock_redis_client.hgetall.return_value = {}

        cleaned = await storage_with_mock_client.cleanup_expired()

        assert cleaned == 0
        assert mock_redis_client.scan.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired_skips_invalid_entry_type(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test cleanup_expired skips entries that aren't Entry type (line 412 coverage)."""
        entry_key = b"test_cache:entry:id1"

        # Mock scan
        mock_redis_client.scan.side_effect = [
            (0, [entry_key]),
        ]

        # Mock hgetall with data
        mock_redis_client.hgetall.return_value = {b"data": b"serialized_invalid_entry"}

        with patch("src.cache_manager.async_redis_storage.unpack") as mock_unpack:
            mock_unpack.return_value = "not_an_entry_object"  # Invalid type

            cleaned = await storage_with_mock_client.cleanup_expired()

            assert cleaned == 0

    @pytest.mark.asyncio
    async def test_hard_delete_entry_removes_all_data(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _hard_delete_entry removes entry, stream, and index."""
        entry_id = uuid.uuid4()
        cache_key = "test_key"

        # Mock hgetall to return cache key
        mock_redis_client.hgetall.return_value = {
            b"cache_key": cache_key.encode("utf-8")
        }

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.delete = MagicMock()
        mock_pipeline.execute = AsyncMock()

        await storage_with_mock_client._hard_delete_entry(entry_id)

        # Check index removal
        mock_redis_client.srem.assert_called_once()

        # Check entry and stream deletion
        assert mock_pipeline.delete.call_count == 2
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_hard_delete_entry_without_cache_key(
        self,
        storage_with_mock_client: AsyncRedisStorage,
        mock_redis_client: AsyncMock,
    ) -> None:
        """Test _hard_delete_entry handles missing cache_key."""
        entry_id = uuid.uuid4()

        # Mock hgetall to return no cache_key
        mock_redis_client.hgetall.return_value = {}

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_pipeline.delete = MagicMock()
        mock_pipeline.execute = AsyncMock()

        await storage_with_mock_client._hard_delete_entry(entry_id)

        # Check index removal was NOT called
        mock_redis_client.srem.assert_not_called()

        # Check entry and stream still deleted
        assert mock_pipeline.delete.call_count == 2


# ============================================================================
# Test Class: Close Connection
# ============================================================================


class TestClose:
    """Test connection closing."""

    @pytest.mark.asyncio
    async def test_close_closes_owned_client(
        self, mock_redis_client: AsyncMock
    ) -> None:
        """Test close() closes Redis client when storage owns it."""
        with patch("src.cache_manager.async_redis_storage.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis_client

            storage = AsyncRedisStorage()  # Creates own client

            await storage.close()

            mock_redis_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_not_close_external_client(
        self, mock_redis_client: AsyncMock
    ) -> None:
        """Test close() does not close external Redis client."""
        storage = AsyncRedisStorage(client=mock_redis_client)  # External client

        await storage.close()

        mock_redis_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_handles_exception_gracefully(self) -> None:
        """Test close() handles exceptions during close gracefully."""
        mock_client = AsyncMock()
        mock_client.aclose.side_effect = Exception("Close failed")

        with patch("src.cache_manager.async_redis_storage.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_client

            storage = AsyncRedisStorage()

            # Should not raise exception
            await storage.close()

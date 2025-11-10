"""
Comprehensive tests for src/cache_manager/manager.py

Tests cover:
- HTTPCacheManager initialization with Redis (async client)
- HTTPCacheManager initialization with SQLite
- get_aiohttp_session() with body-based caching
- Service-specific TTL configuration
- Redis connection failure behavior
- Cache statistics retrieval
- Session cleanup and closing (sync and async)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache_manager.config import CacheConfig
from src.cache_manager.manager import HTTPCacheManager


class TestHTTPCacheManagerInit:
    """Test HTTPCacheManager initialization."""

    def test_init_with_config_disabled(self) -> None:
        """Test initialization with caching disabled."""
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)

        assert manager.config == config
        assert manager._storage is None
        assert manager._async_redis_client is None

    def test_init_with_config_enabled_redis(self) -> None:
        """Test initialization with Redis storage enabled (lazy initialization)."""
        config = CacheConfig(enabled=True, storage_type="redis")
        manager = HTTPCacheManager(config)

        assert manager.config == config
        # Redis client is lazily initialized, not created during __init__
        assert manager._async_redis_client is None
        assert manager._redis_event_loop is None

    def test_init_with_config_enabled_sqlite(self, tmp_path) -> None:
        """Test initialization with SQLite storage."""
        cache_dir = str(tmp_path / "test_cache")
        config = CacheConfig(
            enabled=True, storage_type="sqlite", cache_dir=cache_dir
        )

        with patch(
            "src.cache_manager.manager.hishel.SyncSqliteStorage"
        ) as mock_storage:
            with patch("src.cache_manager.manager.Path.mkdir"):
                manager = HTTPCacheManager(config)

                assert manager.config == config
                assert manager._storage is not None
                assert manager._async_redis_client is None
                mock_storage.assert_called_once()

    def test_init_redis_url_missing_raises_error(self) -> None:
        """Test that missing redis_url raises ValueError."""
        config = CacheConfig(enabled=True, storage_type="redis", redis_url=None)

        manager = HTTPCacheManager(config)

        # Should log warning and not crash
        assert manager._async_redis_client is None

    def test_init_redis_connection_failure_logs_warning(self) -> None:
        """Test that Redis connection failure logs warning (no fallback)."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis_class.from_url.side_effect = Exception("Connection failed")

            manager = HTTPCacheManager(config)

            # Should not crash, just log warning
            assert manager._async_redis_client is None

    def test_init_invalid_storage_type_raises_error(self) -> None:
        """Test that invalid storage type raises ValueError."""
        config = CacheConfig(enabled=True)
        config.storage_type = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="Unknown storage type"):
            HTTPCacheManager(config)


class TestGetAiohttpSession:
    """Test get_aiohttp_session() method."""

    def test_get_aiohttp_session_cache_disabled(self) -> None:
        """Test that regular aiohttp session is returned when cache disabled."""
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)

        with patch(
            "src.cache_manager.manager.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            session = manager.get_aiohttp_session("jikan")

            assert session == mock_session
            mock_session_class.assert_called_once()

    def test_get_aiohttp_session_storage_not_redis(self) -> None:
        """Test that regular session returned when storage is not Redis."""
        config = CacheConfig(enabled=True, storage_type="sqlite")

        with patch("src.cache_manager.manager.hishel.SyncSqliteStorage"):
            with patch("src.cache_manager.manager.Path.mkdir"):
                with patch(
                    "src.cache_manager.manager.aiohttp.ClientSession"
                ) as mock_session_class:
                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")

                    assert session == mock_session
                    mock_session_class.assert_called_once()

    def test_get_aiohttp_session_no_async_redis_client(self) -> None:
        """Test regular session returned when async Redis client is None."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis_class.from_url.side_effect = Exception("Failed")

            with patch(
                "src.cache_manager.manager.aiohttp.ClientSession"
            ) as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                manager = HTTPCacheManager(config)
                session = manager.get_aiohttp_session("jikan")

                assert session == mock_session
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_with_redis_success(self) -> None:
        """Test aiohttp session creation with Redis caching (lazy initialization)."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "src.cache_manager.async_redis_storage.AsyncRedisStorage"
            ) as mock_async_storage:
                with patch(
                    "src.cache_manager.aiohttp_adapter.CachedAiohttpSession"
                ) as mock_cached_session:
                    mock_session_instance = MagicMock()
                    mock_cached_session.return_value = mock_session_instance

                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")

                    # Should create cached session
                    mock_cached_session.assert_called_once()
                    mock_async_storage.assert_called_once()
                    assert session == mock_session_instance
                    # Verify lazy initialization happened
                    mock_async_redis_class.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_body_based_caching_header(self) -> None:
        """Test that body-based caching header is added."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("src.cache_manager.async_redis_storage.AsyncRedisStorage"):
                with patch(
                    "src.cache_manager.aiohttp_adapter.CachedAiohttpSession"
                ) as mock_cached_session:
                    manager = HTTPCacheManager(config)
                    manager.get_aiohttp_session("anilist")

                    # Check that X-Hishel-Body-Key header was added
                    call_kwargs = mock_cached_session.call_args[1]
                    assert "headers" in call_kwargs
                    assert call_kwargs["headers"]["X-Hishel-Body-Key"] == "true"

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_service_specific_ttl(self) -> None:
        """Test that service-specific TTL is used."""
        config = CacheConfig(enabled=True, storage_type="redis", ttl_jikan=7200)

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "src.cache_manager.async_redis_storage.AsyncRedisStorage"
            ) as mock_async_storage:
                with patch("src.cache_manager.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)
                    manager.get_aiohttp_session("jikan")

                    # Check TTL passed to AsyncRedisStorage
                    call_kwargs = mock_async_storage.call_args[1]
                    assert call_kwargs["default_ttl"] == 7200.0

    def test_get_aiohttp_session_exception_in_cached_session_creation(self) -> None:
        """Test fallback to regular session when cached session creation fails."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
                side_effect=Exception("Creation failed"),
            ):
                with patch(
                    "src.cache_manager.manager.aiohttp.ClientSession"
                ) as mock_session_class:
                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")

                    # Should fall back to regular aiohttp session
                    assert session == mock_session
                    mock_session_class.assert_called_once()

    def test_get_aiohttp_session_import_error_fallback(self) -> None:
        """Test fallback to regular session when imports fail."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "src.cache_manager.async_redis_storage.AsyncRedisStorage",
                side_effect=ImportError("No module"),
            ):
                with patch(
                    "src.cache_manager.manager.aiohttp.ClientSession"
                ) as mock_session_class:
                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")

                    # Should fall back to regular aiohttp session
                    assert session == mock_session
                    mock_session_class.assert_called_once()


class TestServiceTTL:
    """Test _get_service_ttl() method."""

    def test_get_service_ttl_known_service(self) -> None:
        """Test TTL retrieval for known services."""
        config = CacheConfig(
            enabled=True,
            ttl_jikan=3600,
            ttl_anilist=7200,
            ttl_kitsu=14400,
        )
        manager = HTTPCacheManager(config)

        assert manager._get_service_ttl("jikan") == 3600
        assert manager._get_service_ttl("anilist") == 7200
        assert manager._get_service_ttl("kitsu") == 14400

    def test_get_service_ttl_unknown_service_default(self) -> None:
        """Test that unknown service returns default 24h TTL."""
        config = CacheConfig(enabled=True)
        manager = HTTPCacheManager(config)

        # Unknown service should return default 86400 (24 hours)
        ttl = manager._get_service_ttl("unknown_service")
        assert ttl == 86400

    def test_get_service_ttl_all_services(self) -> None:
        """Test TTL for all configured services."""
        config = CacheConfig(enabled=True)
        manager = HTTPCacheManager(config)

        services = [
            "jikan",
            "anilist",
            "anidb",
            "kitsu",
            "anime_planet",
            "anisearch",
            "animeschedule",
        ]

        for service in services:
            ttl = manager._get_service_ttl(service)
            assert ttl == 86400  # All default to 24 hours


class TestCacheManagerClose:
    """Test close() and close_async() methods."""

    def test_close_sync_does_nothing(self) -> None:
        """Test that sync close() does nothing (deprecated)."""
        config = CacheConfig(enabled=True, storage_type="sqlite")

        with patch("src.cache_manager.manager.hishel.SyncSqliteStorage"):
            with patch("src.cache_manager.manager.Path.mkdir"):
                manager = HTTPCacheManager(config)
                # Should not raise error
                manager.close()

    @pytest.mark.asyncio
    async def test_close_async_with_redis_client(self) -> None:
        """Test async closing manager with Redis client (lazy initialization)."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("src.cache_manager.async_redis_storage.AsyncRedisStorage"):
                with patch("src.cache_manager.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)
                    # Trigger lazy initialization by calling get_aiohttp_session
                    manager.get_aiohttp_session("jikan")
                    await manager.close_async()

                    # Redis client should be closed
                    mock_async_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async_without_redis_client(self) -> None:
        """Test async closing manager without Redis client."""
        config = CacheConfig(enabled=True, storage_type="sqlite")

        with patch("src.cache_manager.manager.hishel.SyncSqliteStorage"):
            with patch("src.cache_manager.manager.Path.mkdir"):
                manager = HTTPCacheManager(config)
                # Should not raise error
                await manager.close_async()

    @pytest.mark.asyncio
    async def test_close_async_with_redis_error(self) -> None:
        """Test that close_async() handles Redis errors gracefully."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis
            mock_async_redis.close.side_effect = Exception("Close error")

            manager = HTTPCacheManager(config)
            # Should not raise error
            await manager.close_async()


class TestGetStats:
    """Test get_stats() method."""

    def test_get_stats_disabled(self) -> None:
        """Test stats when caching is disabled."""
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)

        stats = manager.get_stats()

        assert stats == {"enabled": False}

    def test_get_stats_redis(self) -> None:
        """Test stats with Redis storage."""
        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://test:6379/0",
        )

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            manager = HTTPCacheManager(config)
            stats = manager.get_stats()

            assert stats["enabled"] is True
            assert stats["storage_type"] == "redis"
            assert stats["redis_url"] == "redis://test:6379/0"
            assert stats["cache_dir"] is None

    def test_get_stats_sqlite(self, tmp_path) -> None:
        """Test stats with SQLite storage."""
        cache_dir = str(tmp_path / "custom_cache")
        config = CacheConfig(
            enabled=True,
            storage_type="sqlite",
            cache_dir=cache_dir,
        )

        with patch("src.cache_manager.manager.hishel.SyncSqliteStorage"):
            with patch("src.cache_manager.manager.Path.mkdir"):
                manager = HTTPCacheManager(config)
                stats = manager.get_stats()

                assert stats["enabled"] is True
                assert stats["storage_type"] == "sqlite"
                assert stats["cache_dir"] == cache_dir
                assert stats["redis_url"] is None


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_production_redis_setup(self) -> None:
        """Test production-like Redis setup (lazy initialization)."""
        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://prod:6379/0",
            ttl_jikan=86400,
            ttl_anilist=86400,
        )

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("src.cache_manager.async_redis_storage.AsyncRedisStorage"):
                with patch("src.cache_manager.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)

                    # Redis client is lazily initialized
                    assert manager._async_redis_client is None

                    # Trigger lazy initialization
                    manager.get_aiohttp_session("jikan")

                    # Now Redis client should be initialized
                    assert manager._async_redis_client is not None
                    assert manager._get_service_ttl("jikan") == 86400

    def test_development_sqlite_setup(self, tmp_path) -> None:
        """Test development-like SQLite setup."""
        cache_dir = str(tmp_path / "dev_cache")
        config = CacheConfig(
            enabled=True,
            storage_type="sqlite",
            cache_dir=cache_dir,
        )

        with patch("src.cache_manager.manager.hishel.SyncSqliteStorage"):
            with patch("src.cache_manager.manager.Path.mkdir"):
                manager = HTTPCacheManager(config)

                assert manager._storage is not None
                assert manager._async_redis_client is None

    def test_multiple_services_session_creation(self) -> None:
        """Test creating sessions for multiple services."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "src.cache_manager.manager.aiohttp.ClientSession"
            ) as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                manager = HTTPCacheManager(config)

                # Create multiple aiohttp sessions
                session1 = manager.get_aiohttp_session("jikan")
                session2 = manager.get_aiohttp_session("kitsu")
                session3 = manager.get_aiohttp_session("anidb")

                # All should return aiohttp sessions (mocked)
                assert session1 is not None
                assert session2 is not None
                assert session3 is not None


class TestRedisClientEventLoopSwitching:
    """Test Redis client cleanup when event loops switch."""

    def test_old_redis_client_closed_on_event_loop_switch(self) -> None:
        """
        Test that old Redis clients are properly closed when switching event loops.

        This test verifies the fix for the resource leak where old Redis clients
        were not being closed when a new event loop requested a cached session.

        Expected behavior:
        - When event loop switches, old Redis client should be closed
        - New Redis client should be created for new event loop
        - Only one active Redis client per manager at a time
        """
        import asyncio
        import threading

        # Track Redis client close() calls
        close_calls = []

        # Mock Redis client that tracks close() calls
        class MockAsyncRedis:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def close(self):
                if not self.closed:
                    self.closed = True
                    close_calls.append(self)

        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"
        )

        # Create new instances each time from_url is called
        def mock_from_url(*args, **kwargs):
            return MockAsyncRedis()

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            manager = HTTPCacheManager(config)

            clients = {}

            # Run in thread 1 with its own event loop
            def run_in_thread1():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def task():
                    session = manager.get_aiohttp_session("test_service")
                    await asyncio.sleep(0.01)
                    if hasattr(session, 'close'):
                        await session.close()
                    clients['client1'] = manager._async_redis_client

                loop.run_until_complete(task())
                loop.close()

            # Run in thread 2 with a different event loop
            def run_in_thread2():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def task():
                    session = manager.get_aiohttp_session("test_service")
                    await asyncio.sleep(0.01)
                    if hasattr(session, 'close'):
                        await session.close()
                    clients['client2'] = manager._async_redis_client

                loop.run_until_complete(task())
                loop.close()

            # Execute in separate threads
            thread1 = threading.Thread(target=run_in_thread1)
            thread2 = threading.Thread(target=run_in_thread2)

            thread1.start()
            thread1.join()

            thread2.start()
            thread2.join()

            client1 = clients.get('client1')
            client2 = clients.get('client2')

            # Verify old client was closed
            assert client1 is not None, "First client should have been created"
            assert client2 is not None, "Second client should have been created"
            assert client1 is not client2, "Should create new client for new loop"
            assert client1.closed, "Old Redis client should be closed when loop switches"
            assert len(close_calls) >= 1, "At least one client should have been closed"

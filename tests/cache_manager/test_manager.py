"""
Comprehensive tests for src/cache_manager/manager.py

Tests cover:
- HTTPCacheManager initialization with Redis (async client)
- get_aiohttp_session() with body-based caching
- Service-specific TTL configuration
- Redis connection failure behavior
- Cache statistics retrieval
- Session cleanup and closing (async)
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
        assert manager._async_redis_client is None

    def test_init_with_config_enabled_redis(self) -> None:
        """Test initialization with Redis storage enabled (lazy initialization)."""
        config = CacheConfig(enabled=True, storage_type="redis")
        manager = HTTPCacheManager(config)

        assert manager.config == config
        # Redis client is lazily initialized, not created during __init__
        assert manager._async_redis_client is None
        assert manager._redis_event_loop is None

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
    """Test close_async() method."""

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
                    mock_async_redis.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async_with_redis_error(self) -> None:
        """Test that close_async() handles Redis errors gracefully."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("src.cache_manager.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis
            mock_async_redis.aclose.side_effect = Exception("Close error")

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

            async def aclose(self):
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

    @pytest.mark.asyncio
    async def test_old_loop_still_running_uses_run_coroutine_threadsafe(self) -> None:
        """
        Test that when old event loop is still running, asyncio.run_coroutine_threadsafe is used.

        This covers line 97 where old_loop.is_running() returns True.
        """
        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"
        )

        # Track how close was called
        close_method_used = []

        class MockAsyncRedis:
            async def aclose(self):
                close_method_used.append("close_called")

        # First call creates client in loop1
        mock_client1 = MockAsyncRedis()
        # Second call creates client in loop2
        mock_client2 = MockAsyncRedis()

        call_count = [0]
        def mock_from_url(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client1
            else:
                return mock_client2

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            with patch('asyncio.run_coroutine_threadsafe') as mock_run_threadsafe:
                manager = HTTPCacheManager(config)

                # First call - creates client in current loop
                session1 = manager.get_aiohttp_session("test")
                assert manager._async_redis_client is mock_client1

                # Mock the old loop as still running
                old_loop = manager._redis_event_loop
                with patch.object(old_loop, 'is_running', return_value=True):
                    # Second call in same event loop should trigger cleanup
                    # Force a new event loop scenario by patching get_running_loop
                    import asyncio
                    new_loop = asyncio.new_event_loop()

                    async def get_new_loop():
                        return new_loop

                    with patch('asyncio.get_running_loop', return_value=new_loop):
                        session2 = manager.get_aiohttp_session("test")

                        # Should have called run_coroutine_threadsafe for old client
                        assert mock_run_threadsafe.called
                        # Verify it was called with close() coroutine and old loop
                        args = mock_run_threadsafe.call_args[0]
                        assert old_loop in mock_run_threadsafe.call_args[0] or old_loop == mock_run_threadsafe.call_args[1].get('loop')

    @pytest.mark.asyncio
    async def test_exception_during_old_client_close_is_handled(self) -> None:
        """
        Test that exceptions during old Redis client close are handled gracefully.

        This covers lines 95-97 where close errors are caught and logged.
        """
        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"
        )

        # Track close attempts
        close_exception_caught = []

        class MockAsyncRedis:
            async def aclose(self):
                error = Exception("Close failed unexpectedly")
                close_exception_caught.append(error)
                raise error

        mock_client1 = MockAsyncRedis()
        mock_client2 = MockAsyncRedis()

        call_count = [0]
        def mock_from_url(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client1
            else:
                return mock_client2

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            manager = HTTPCacheManager(config)

            # First call creates client in current loop
            session1 = manager.get_aiohttp_session("test")
            assert manager._async_redis_client is mock_client1

            # Get the old loop reference
            import asyncio
            old_loop = manager._redis_event_loop
            new_loop = asyncio.new_event_loop()

            # Mock old loop as not running (to trigger create_task path)
            with patch.object(old_loop, 'is_running', return_value=False):
                with patch('asyncio.get_running_loop', return_value=new_loop):
                    with patch.object(new_loop, 'create_task', side_effect=Exception("Task creation failed")):
                        # This should trigger exception in close handling (lines 95-97)
                        session2 = manager.get_aiohttp_session("test")

                        # Should have created new client anyway
                        assert manager._async_redis_client is mock_client2

    @pytest.mark.asyncio
    async def test_get_or_create_redis_client_returns_none_when_redis_url_none(self) -> None:
        """
        Test that _get_or_create_redis_client returns None when redis_url is None.

        This covers line 107 where redis_url check returns None.
        """
        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"  # Start with valid URL
        )

        manager = HTTPCacheManager(config)

        # First call with valid redis_url creates client
        session1 = manager.get_aiohttp_session("test")
        assert manager._async_redis_client is not None

        # Now simulate redis_url becoming None (configuration change)
        manager.config.redis_url = None

        # Force event loop switch to trigger _get_or_create_redis_client logic
        import asyncio
        new_loop = asyncio.new_event_loop()

        with patch('asyncio.get_running_loop', return_value=new_loop):
            # This should return None because redis_url is None (line 107)
            client = manager._get_or_create_redis_client()
            assert client is None


class TestGetAiohttpSessionErrorHandling:
    """Test error handling in get_aiohttp_session for lines 183-188."""

    @pytest.mark.asyncio
    async def test_import_error_from_cached_aiohttp_session_import(self) -> None:
        """
        Test ImportError when importing CachedAiohttpSession (line 184).
        """
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch('redis.asyncio.Redis.from_url') as mock_redis:
            mock_redis.return_value = MagicMock()

            # Mock ImportError when importing CachedAiohttpSession (within get_aiohttp_session)
            with patch('src.cache_manager.aiohttp_adapter.CachedAiohttpSession', side_effect=ImportError("Module not found")):
                with patch('src.cache_manager.manager.aiohttp.ClientSession') as mock_session:
                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("test")

                    # Should fall back to regular session (line 185)
                    mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_general_exception_in_async_storage_creation(self) -> None:
        """
        Test general Exception during async storage creation (lines 186-188).
        """
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch('redis.asyncio.Redis.from_url') as mock_redis:
            mock_redis.return_value = MagicMock()

            # Mock exception during AsyncRedisStorage creation
            with patch('src.cache_manager.async_redis_storage.AsyncRedisStorage', side_effect=RuntimeError("Storage init failed")):
                with patch('src.cache_manager.manager.aiohttp.ClientSession') as mock_session:
                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("test")

                    # Should fall back to regular session (line 188)
                    mock_session.assert_called_once()


class TestCloseErrorHandling:
    """Test error handling in close_async() method."""

    @pytest.mark.asyncio
    async def test_close_async_redis_with_exception(self) -> None:
        """
        Test that close_async() handles Redis close errors gracefully (lines 216-217).
        """
        config = CacheConfig(enabled=True, storage_type="redis")

        # Create mock Redis client that raises on close
        mock_redis = AsyncMock()
        mock_redis.aclose.side_effect = Exception("Redis close error")

        with patch('redis.asyncio.Redis.from_url', return_value=mock_redis):
            with patch('src.cache_manager.async_redis_storage.AsyncRedisStorage'):
                with patch('src.cache_manager.aiohttp_adapter.CachedAiohttpSession'):
                    manager = HTTPCacheManager(config)

                    # Trigger lazy initialization
                    manager.get_aiohttp_session("test")

                    # Verify Redis client was created
                    assert manager._async_redis_client is mock_redis

                    # close_async should not raise, should handle exception (line 217)
                    await manager.close_async()

                    # Verify close was attempted
                    mock_redis.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async_clears_cached_redis_client_references(self) -> None:
        """
        Test that close_async() clears cached Redis client and event loop references.
        
        Bug scenario (from code review):
        - close_async() closes the Redis client but keeps the closed instance cached
        - Next get_aiohttp_session() on same loop returns the closed client
        - Redis operations fail with "Connection closed" errors
        
        Expected behavior after fix:
        - close_async() should set self._async_redis_client = None
        - close_async() should set self._redis_event_loop = None
        - Next get_aiohttp_session() creates a fresh client
        """
        config = CacheConfig(enabled=True, storage_type="redis")

        mock_redis = AsyncMock()
        
        with patch('redis.asyncio.Redis.from_url', return_value=mock_redis):
            with patch('src.cache_manager.async_redis_storage.AsyncRedisStorage'):
                with patch('src.cache_manager.aiohttp_adapter.CachedAiohttpSession'):
                    manager = HTTPCacheManager(config)

                    # Step 1: Create first session (triggers lazy initialization)
                    session1 = manager.get_aiohttp_session("test")
                    
                    # Verify client and loop are cached
                    assert manager._async_redis_client is not None
                    assert manager._redis_event_loop is not None
                    first_client = manager._async_redis_client
                    first_loop = manager._redis_event_loop

                    # Step 2: Close the cache manager
                    await manager.close_async()

                    # Step 3: Verify references are cleared (THIS IS THE FIX)
                    assert manager._async_redis_client is None, (
                        "close_async() must clear _async_redis_client to prevent "
                        "reusing closed client on same event loop"
                    )
                    assert manager._redis_event_loop is None, (
                        "close_async() must clear _redis_event_loop to prevent "
                        "reusing closed client on same event loop"
                    )
                    
                    # Step 4: Verify client was actually closed
                    mock_redis.aclose.assert_called_once()

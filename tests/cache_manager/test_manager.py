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

    def test_init_redis_url_missing_logs_warning(self) -> None:
        """Test that missing redis_url logs warning and does not crash."""
        config = CacheConfig(enabled=True, storage_type="redis", redis_url=None)

        manager = HTTPCacheManager(config)

        # Should log warning and not crash - Redis client remains None
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
        """
        Verify that HTTPCacheManager returns non-null aiohttp sessions for multiple service names when Redis caching is enabled.
        
        Ensures that calling get_aiohttp_session for several distinct services (e.g., "jikan", "kitsu", "anidb") produces a session object for each.
        """
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
        Verify that when the asyncio event loop changes, the manager closes the previous Redis client and creates a new client for the new loop.
        
        Asserts that the first and second Redis clients are distinct, the original client was closed, and at least one close call occurred to prevent resource leaks.
        """
        import asyncio
        import threading

        # Track Redis client close() calls
        close_calls = []

        # Mock Redis client that tracks close() calls
        class MockAsyncRedis:
            def __init__(self, *_args: object, **_kwargs: object):
                """
                Initialize the object and mark it as open.

                Accepts arbitrary positional and keyword arguments for subclassing or expanded initialization.
                """
                self.closed = False

            async def aclose(self):
                """
                Mark the client as closed and record the close call.
                
                If the client is not already marked closed, sets `self.closed = True` and appends `self` to the module-level `close_calls` list.
                """
                if not self.closed:
                    self.closed = True
                    close_calls.append(self)

        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"
        )

        # Create new instances each time from_url is called
        def mock_from_url(*_args: object, **_kwargs: object):
            """
            Create and return a new MockAsyncRedis instance, ignoring any provided arguments.

            Returns:
                MockAsyncRedis: A fresh mock asynchronous Redis client.
            """
            return MockAsyncRedis()

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            manager = HTTPCacheManager(config)

            clients = {}

            # Run in thread 1 with its own event loop
            def run_in_thread1():
                """
                Create and run a new asyncio event loop that obtains and closes an aiohttp session for "test_service" and records the manager's async Redis client.
                
                Runs a fresh event loop, calls manager.get_aiohttp_session("test_service"), awaits a short delay, closes the session if it exposes `close`, and stores the current `manager._async_redis_client` into `clients['client1']`.
                """
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def task():
                    """
                    Run a short-lived retrieval of an aiohttp session and record the manager's async Redis client in `clients['client1']`.
                    
                    This coroutine obtains a session from `manager.get_aiohttp_session("test_service")`, yields briefly, closes the session if it provides an asynchronous `close` method, and stores the current value of `manager._async_redis_client` into `clients['client1']`.
                    """
                    session = manager.get_aiohttp_session("test_service")
                    await asyncio.sleep(0.01)
                    if hasattr(session, 'close'):
                        await session.close()
                    clients['client1'] = manager._async_redis_client

                loop.run_until_complete(task())
                loop.close()

            # Run in thread 2 with a different event loop
            def run_in_thread2():
                """
                Run a short asyncio task on a new event loop that acquires and closes an aiohttp session for "test_service" and records the manager's async Redis client into clients['client2'].
                
                The function creates a fresh event loop, runs an asynchronous task which obtains a session from manager.get_aiohttp_session("test_service"), awaits a brief pause, closes the session if it has a close coroutine, stores manager._async_redis_client in clients['client2'], and then closes the loop.
                """
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def task():
                    """
                    Create and close an aiohttp session for the "test_service" and capture the manager's async Redis client.
                    
                    This coroutine obtains a session from manager.get_aiohttp_session("test_service"), awaits a short delay, closes the session if it implements `close`, and stores the manager's current `_async_redis_client` into `clients['client2']`.
                    """
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
                """
                Asynchronously close the cache manager and clean up any held resources.
                """
                close_method_used.append("close_called")

        # First call creates client in loop1
        mock_client1 = MockAsyncRedis()
        # Second call creates client in loop2
        mock_client2 = MockAsyncRedis()

        call_count = [0]
        def mock_from_url(*_args: object, **_kwargs: object):
            """
            Return `mock_client1` on the first invocation and `mock_client2` on subsequent invocations.

            Increments the external `call_count[0]` counter each time the function is called to track invocation count.

            Returns:
                A mock client object: `mock_client1` for the first call, `mock_client2` for all later calls.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client1
            else:
                return mock_client2

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            with patch('asyncio.run_coroutine_threadsafe') as mock_run_threadsafe:
                manager = HTTPCacheManager(config)

                # First call - creates client in current loop
                manager.get_aiohttp_session("test")
                assert manager._async_redis_client is mock_client1

                # Mock the old loop as still running
                old_loop = manager._redis_event_loop
                with patch.object(old_loop, 'is_running', return_value=True):
                    # Second call in same event loop should trigger cleanup
                    # Force a new event loop scenario by patching get_running_loop
                    import asyncio
                    new_loop = asyncio.new_event_loop()

                    async def get_new_loop():
                        """
                        Create and return a newly created asyncio event loop.
                        
                        Returns:
                            asyncio.AbstractEventLoop: The newly created event loop.
                        """
                        return new_loop

                    with patch('asyncio.get_running_loop', return_value=new_loop):
                        manager.get_aiohttp_session("test")

                        # Should have called run_coroutine_threadsafe for old client
                        assert mock_run_threadsafe.called
                        # Verify it was called with close() coroutine and old loop
                        assert old_loop in mock_run_threadsafe.call_args[0] or old_loop == mock_run_threadsafe.call_args[1].get('loop')

    @pytest.mark.asyncio
    async def test_exception_during_old_client_close_is_handled(self) -> None:
        """
        Ensure that exceptions raised when closing a previous Redis client are handled without propagating and that a new Redis client is created.
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
                """
                Attempt to asynchronously close and signal an unexpected failure.
                
                Appends an Exception instance to the external `close_exception_caught` list and then raises it to indicate the close operation failed.
                
                Raises:
                    Exception: Raised with message "Close failed unexpectedly" when the close procedure fails.
                """
                error = Exception("Close failed unexpectedly")
                close_exception_caught.append(error)
                raise error

        mock_client1 = MockAsyncRedis()
        mock_client2 = MockAsyncRedis()

        call_count = [0]
        def mock_from_url(*_args: object, **_kwargs: object):
            """
            Return `mock_client1` on the first invocation and `mock_client2` on subsequent invocations.

            Increments the external `call_count[0]` counter each time the function is called to track invocation count.

            Returns:
                A mock client object: `mock_client1` for the first call, `mock_client2` for all later calls.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client1
            else:
                return mock_client2

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            manager = HTTPCacheManager(config)

            # First call creates client in current loop
            manager.get_aiohttp_session("test")
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
                        manager.get_aiohttp_session("test")

                        # Should have created new client anyway
                        assert manager._async_redis_client is mock_client2

    @pytest.mark.asyncio
    async def test_cleanup_task_has_done_callback_attached(self) -> None:
        """
        Test that cleanup task has done_callback attached for exception tracking.

        Bug scenario (from code review):
        - Line 94: current_loop.create_task(old_client.aclose())
        - If aclose() raises exception inside task, it's silently lost
        - Python emits "Task exception was never retrieved" warning
        - No visibility into cleanup failures

        Expected behavior after fix:
        - Store cleanup task reference (not fire-and-forget)
        - Add done_callback to track success/failure
        - Callback should log exceptions
        """
        import asyncio

        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0"
        )

        # Track create_task calls
        created_tasks = []
        original_create_task = asyncio.BaseEventLoop.create_task

        def track_create_task(self, coro, *args, **kwargs):
            """
            Wraps the event loop's task creation to record each created Task in the surrounding `created_tasks` list.
            
            Parameters:
                coro: Coroutine or awaitable to schedule as a Task.
                *args: Positional arguments forwarded to the original `create_task`.
                **kwargs: Keyword arguments forwarded to the original `create_task`.
            
            Returns:
                The newly created `asyncio.Task` instance. The task is also appended to `created_tasks`.
            """
            task = original_create_task(self, coro, *args, **kwargs)
            created_tasks.append(task)
            return task

        class MockAsyncRedis:
            async def aclose(self):
                """
                Perform any cleanup associated with closing the object.
                
                This implementation is a no-op and does not perform any actions.
                """
                pass  # No-op for this test

        mock_client1 = MockAsyncRedis()
        mock_client2 = MockAsyncRedis()

        call_count = [0]
        def mock_from_url(*_args: object, **_kwargs: object):
            """
            Return `mock_client1` on the first invocation and `mock_client2` on subsequent invocations.

            Increments the external `call_count[0]` counter each time the function is called to track invocation count.

            Returns:
                A mock client object: `mock_client1` for the first call, `mock_client2` for all later calls.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client1
            else:
                return mock_client2

        with patch('redis.asyncio.Redis.from_url', side_effect=mock_from_url):
            with patch.object(asyncio.BaseEventLoop, 'create_task', track_create_task):
                manager = HTTPCacheManager(config)

                # First call creates client in current loop
                manager.get_aiohttp_session("test")
                assert manager._async_redis_client is mock_client1

                # Get current loop reference
                old_loop = manager._redis_event_loop

                # Create new event loop and switch to it
                new_loop = asyncio.new_event_loop()

                # Mock old loop as not running (triggers create_task path on line 94)
                with patch.object(old_loop, 'is_running', return_value=False):
                    with patch('asyncio.get_running_loop', return_value=new_loop):
                        # Second call should trigger cleanup of old client via create_task
                        manager.get_aiohttp_session("test")

                        # Verify new client was created
                        assert manager._async_redis_client is mock_client2

                        # Verify a cleanup task was created
                        assert len(created_tasks) > 0, "Should create cleanup task for old Redis client"

                        # Verify the cleanup task has a done_callback attached
                        cleanup_task = created_tasks[-1]  # Most recent task
                        assert hasattr(cleanup_task, '_callbacks'), "Task should support callbacks"
                        assert cleanup_task._callbacks is not None and len(cleanup_task._callbacks) > 0, (
                            "Cleanup task MUST have done_callback attached to track exceptions. "
                            "Fire-and-forget tasks (without callbacks) silently lose exceptions."
                        )

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
        manager.get_aiohttp_session("test")
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
        Verifies that HTTPCacheManager falls back to a regular aiohttp ClientSession when creating the async Redis storage raises a general exception.
        
        Patches Redis client creation to succeed and forces AsyncRedisStorage to raise a RuntimeError, then asserts that get_aiohttp_session returns a normal ClientSession instead of a cached session.
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
                    manager.get_aiohttp_session("test")

                    # Verify client and loop are cached
                    assert manager._async_redis_client is not None
                    assert manager._redis_event_loop is not None

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
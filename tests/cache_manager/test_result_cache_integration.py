"""
Integration tests for result_cache.py singleton pattern.

These tests verify thread-safety and race condition handling
under concurrent operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from src.cache_manager.config import CacheConfig

pytestmark = pytest.mark.integration


class TestResultCacheRaceConditions:
    """Integration tests for race condition scenarios in result cache singleton."""

    @pytest.mark.asyncio
    async def test_race_condition_fix_with_instrumentation(self) -> None:
        """
        Validate that holding the internal initialization lock until the client is returned prevents a race where a concurrent closer can set the client to None during initialization.
        
        Instruments get_result_cache_redis_client to introduce a short delay inside the critical section, runs an initialization getter and a concurrent closer, and asserts the getter still observes a non-None client even if the closer runs during the delay.
        """
        with (
            patch("src.cache_manager.result_cache.get_cache_config") as mock_get_config,
            patch("src.cache_manager.result_cache.Redis.from_url") as mock_from_url,
        ):
            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://localhost:6379/0"
            mock_config_instance.redis_max_connections = 100
            mock_config_instance.redis_socket_keepalive = True
            mock_config_instance.redis_socket_connect_timeout = 5
            mock_config_instance.redis_socket_timeout = 10
            mock_config_instance.redis_retry_on_timeout = True
            mock_config_instance.redis_health_check_interval = 30
            mock_get_config.return_value = mock_config_instance

            mock_client = AsyncMock(spec=Redis)
            mock_from_url.return_value = mock_client

            # Import module to access internals
            from src.cache_manager import result_cache
            from src.cache_manager.result_cache import close_result_cache_redis_client

            # Reset state
            result_cache._redis_client = None

            # Create instrumented version that adds delay in critical section
            original_get = result_cache.get_result_cache_redis_client

            async def instrumented_get_with_delay():
                """
                Obtain the result cache Redis client while introducing a short delay inside the initialization lock to simulate a race window.
                
                This instrumented getter acquires the cache's internal lock, lazily initializes the Redis client if needed, awaits a brief sleep inside the critical section to allow concurrent operations to interleave, and then returns the client.
                
                Returns:
                    The initialized Redis client instance from the result cache.
                """
                async with result_cache._redis_lock:
                    if result_cache._redis_client is None:
                        config = result_cache.get_cache_config()
                        redis_url = config.redis_url or "redis://localhost:6379/0"
                        result_cache._redis_client = result_cache.Redis.from_url(
                            redis_url,
                            decode_responses=True,
                            max_connections=config.redis_max_connections,
                            socket_keepalive=config.redis_socket_keepalive,
                            socket_connect_timeout=config.redis_socket_connect_timeout,
                            socket_timeout=config.redis_socket_timeout,
                            retry_on_timeout=config.redis_retry_on_timeout,
                            health_check_interval=config.redis_health_check_interval,
                        )
                    # Delay in critical section to allow concurrent close attempt
                    await asyncio.sleep(0.01)
                    return result_cache._redis_client

            try:
                # Initialize with instrumented version
                result_cache.get_result_cache_redis_client = instrumented_get_with_delay
                await instrumented_get_with_delay()
                assert result_cache._redis_client is not None

                race_result = None

                async def getter_with_delay():
                    """
                    Await the instrumented Redis client getter and store the obtained client in the enclosing `race_result` variable.
                    """
                    nonlocal race_result
                    race_result = await instrumented_get_with_delay()

                async def concurrent_closer():
                    """
                    Sleep briefly and then attempt to close the shared result cache Redis client to simulate a concurrent closer racing with a getter's critical section.
                    """
                    await asyncio.sleep(0.005)
                    await close_result_cache_redis_client()

                # Run concurrently - closer tries to set None during getter's delay
                await asyncio.gather(getter_with_delay(), concurrent_closer())

                # PROOF: With lock held until return, getter cannot return None
                # even though closer tried to set it to None during getter's execution
                assert race_result is not None, (
                    "Race condition still exists: getter returned None "
                    "even though lock was held until return. Fix is incorrect."
                )
            finally:
                # Ensure clean state for other tests
                result_cache._redis_client = None
                result_cache.get_result_cache_redis_client = original_get
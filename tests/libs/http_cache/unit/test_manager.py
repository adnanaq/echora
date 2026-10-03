import asyncio
import logging
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
from hishel import FilterPolicy, Response
from http_cache.aiohttp_adapter import CachedAiohttpSession
from http_cache.async_redis_storage import AsyncRedisStorage
from http_cache.config import CacheConfig
from http_cache.exceptions import StorageConfigurationError
from http_cache.manager import (
    _MAX_ERROR_BODY_BYTES,
    HTTPCacheManager,
    NeverCacheErrorsFilter,
)
from redis.asyncio import Redis as AsyncRedis


def _response(status: int) -> Response:
    return Response(status_code=status)


def _old_loop() -> asyncio.AbstractEventLoop:
    return create_autospec(asyncio.AbstractEventLoop, instance=True)


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_http_cache_manager_init_sets_error_filter_without_global_body_key(
    cache_enabled: bool,
) -> None:
    config = CacheConfig(cache_enabled=cache_enabled, storage_type="redis")
    manager = HTTPCacheManager(config)

    assert manager.config == config
    assert manager._async_redis_client is None
    assert manager._redis_event_loop is None
    assert isinstance(manager.policy, FilterPolicy)
    assert manager.policy.use_body_key is False
    assert len(manager.policy.response_filters) == 1
    assert isinstance(manager.policy.response_filters[0], NeverCacheErrorsFilter)


def test_http_cache_manager_init_redis_url_missing_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = CacheConfig(cache_enabled=True, storage_type="redis", redis_url=None)

    with caplog.at_level(logging.WARNING, logger="http_cache.manager"):
        manager = HTTPCacheManager(config)

    assert manager._async_redis_client is None
    assert "redis_url required" in caplog.text


def test_http_cache_manager_init_unknown_storage_type_raises_storage_configuration_error() -> (
    None
):
    config = CacheConfig.model_construct(
        cache_enabled=True,
        storage_type="invalid",
        force_cache=False,
        always_revalidate=False,
    )

    with pytest.raises(StorageConfigurationError, match="Unknown storage type"):
        HTTPCacheManager(config)


async def test_get_aiohttp_session_cache_disabled_returns_plain_session() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=False))

    session = manager.get_aiohttp_session("mal")

    assert type(session) is aiohttp.ClientSession
    await session.close()


def test_get_aiohttp_session_no_running_event_loop_returns_plain_session() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))

    with patch(
        "http_cache.manager.aiohttp.ClientSession", autospec=True
    ) as session_class:
        session = manager.get_aiohttp_session("mal")

    assert session is session_class.return_value
    assert manager._async_redis_client is None


async def test_get_aiohttp_session_redis_available_returns_cached_session_with_service_ttl() -> (
    None
):
    config = CacheConfig(cache_enabled=True, storage_type="redis", ttl_mal=7200)
    manager = HTTPCacheManager(config)

    session = manager.get_aiohttp_session("mal")

    assert isinstance(session, CachedAiohttpSession)
    assert isinstance(session.storage, AsyncRedisStorage)
    assert session.storage.default_ttl == 7200.0
    assert session.policy is manager.policy
    await session.close()
    await manager.close_async()


@pytest.mark.parametrize(
    "storage_error",
    [ImportError("missing dep"), RuntimeError("init failed")],
    ids=["import_error", "other_error"],
)
async def test_get_aiohttp_session_storage_error_returns_plain_session(
    storage_error: Exception,
) -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))

    with patch(
        "http_cache.async_redis_storage.AsyncRedisStorage",
        autospec=True,
        side_effect=storage_error,
    ):
        session = manager.get_aiohttp_session("mal")

    assert type(session) is aiohttp.ClientSession
    await session.close()
    await manager.close_async()


def test_get_service_ttl_known_service_returns_configured_ttl() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, ttl_mal=3600))

    assert manager._get_service_ttl("mal") == 3600


def test_get_service_ttl_unknown_service_returns_one_day() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True))

    assert manager._get_service_ttl("unknown") == 86400


async def test_close_async_redis_client_open_closes_client_and_clears_it() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))
    client = create_autospec(AsyncRedis, instance=True)
    manager._async_redis_client = client
    manager._redis_event_loop = asyncio.get_running_loop()

    await manager.close_async()

    client.aclose.assert_awaited_once_with()
    assert manager._async_redis_client is None
    assert manager._redis_event_loop is None


async def test_close_async_aclose_failure_logs_warning_and_clears_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=False))
    client = create_autospec(AsyncRedis, instance=True)
    client.aclose.side_effect = RuntimeError("connection lost")
    manager._async_redis_client = client
    manager._redis_event_loop = asyncio.get_running_loop()

    with caplog.at_level(logging.WARNING, logger="http_cache.manager"):
        await manager.close_async()

    assert manager._async_redis_client is None
    assert manager._redis_event_loop is None
    assert "Error closing async Redis client" in caplog.text


def test_get_stats_cache_disabled_returns_only_disabled_flag() -> None:
    manager = HTTPCacheManager(CacheConfig(cache_enabled=False))

    assert manager.get_stats() == {"cache_enabled": False}


def test_get_stats_redis_storage_returns_redis_url() -> None:
    config = CacheConfig(
        cache_enabled=True, storage_type="redis", redis_url="redis://test"
    )
    stats = HTTPCacheManager(config).get_stats()

    assert stats["cache_enabled"] is True
    assert stats["redis_url"] == "redis://test"


async def test_get_or_create_redis_client_no_redis_url_returns_none() -> None:
    config = CacheConfig(cache_enabled=True, storage_type="redis", redis_url=None)

    assert HTTPCacheManager(config)._get_or_create_redis_client() is None


async def test_get_or_create_redis_client_event_loop_changed_creates_client_for_current_loop() -> (
    None
):
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))
    old_client = create_autospec(AsyncRedis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.return_value = False
    manager._async_redis_client = old_client
    manager._redis_event_loop = old_loop

    client = manager._get_or_create_redis_client()
    await asyncio.sleep(0)

    assert isinstance(client, AsyncRedis)
    assert client is not old_client
    assert manager._redis_event_loop is asyncio.get_running_loop()
    old_client.aclose.assert_awaited_once_with()
    await manager.close_async()


async def test_get_or_create_redis_client_old_loop_running_closes_old_client_on_that_loop() -> (
    None
):
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))
    old_client = create_autospec(AsyncRedis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.return_value = True
    manager._async_redis_client = old_client
    manager._redis_event_loop = old_loop

    with patch(
        "http_cache.manager.asyncio.run_coroutine_threadsafe", autospec=True
    ) as run_on_loop:
        client = manager._get_or_create_redis_client()

    close_coroutine, target_loop = run_on_loop.call_args.args
    close_coroutine.close()
    assert target_loop is old_loop
    assert client is not old_client
    assert isinstance(client, AsyncRedis)
    await manager.close_async()


async def test_get_or_create_redis_client_old_client_cleanup_failure_creates_new_client() -> (
    None
):
    manager = HTTPCacheManager(CacheConfig(cache_enabled=True, storage_type="redis"))
    old_client = create_autospec(AsyncRedis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.side_effect = RuntimeError("loop gone")
    manager._async_redis_client = old_client
    manager._redis_event_loop = old_loop

    client = manager._get_or_create_redis_client()

    assert isinstance(client, AsyncRedis)
    assert client is not old_client
    await manager.close_async()


def test_never_cache_errors_filter_needs_body_returns_true() -> None:
    assert NeverCacheErrorsFilter().needs_body() is True


@pytest.mark.parametrize("status", [200, 201, 301, 304])
def test_never_cache_errors_filter_apply_success_status_returns_true(
    status: int,
) -> None:
    assert NeverCacheErrorsFilter().apply(_response(status), None) is True


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_never_cache_errors_filter_apply_error_status_returns_false(
    status: int,
) -> None:
    assert NeverCacheErrorsFilter().apply(_response(status), None) is False


@pytest.mark.parametrize(
    "body",
    [
        b'<error code="500">banned</error>',
        b'<?xml version="1.0" encoding="UTF-8"?><error code="302">Client Outdated</error>',
        b"\n  <error>unknown</error>",
    ],
)
def test_never_cache_errors_filter_apply_xml_error_body_on_200_returns_false(
    body: bytes,
) -> None:
    assert NeverCacheErrorsFilter().apply(_response(200), body) is False


@pytest.mark.parametrize(
    "body",
    [
        b"<anime><titles><title>One Piece</title></titles></anime>",
        b'{"data": {"error": null}}',
        b"<html><body>no error here</body></html>",
    ],
)
def test_never_cache_errors_filter_apply_valid_body_on_200_returns_true(
    body: bytes,
) -> None:
    assert NeverCacheErrorsFilter().apply(_response(200), body) is True


def test_never_cache_errors_filter_apply_body_past_scan_limit_returns_true() -> None:
    body = b'<error code="500">banned</error>' + b"x" * _MAX_ERROR_BODY_BYTES

    assert NeverCacheErrorsFilter().apply(_response(200), body) is True

import asyncio
import logging
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
from common.config import CacheConfig, RedisConfig
from hishel import FilterPolicy, Response
from http_cache.aiohttp_adapter import CachedAiohttpSession
from http_cache.async_redis_storage import AsyncRedisStorage
from http_cache.manager import (
    _MAX_ERROR_BODY_BYTES,
    HTTPCacheManager,
    NeverCacheErrorsFilter,
)
from redis.asyncio import Redis

REDIS_URL = "redis://localhost:6379/15"


def _response(status: int) -> Response:
    return Response(status_code=status)


def _old_loop() -> asyncio.AbstractEventLoop:
    return create_autospec(asyncio.AbstractEventLoop, instance=True)


def _manager(
    *, cache_enabled: bool = True, redis_url: str | None = REDIS_URL, **cache
) -> HTTPCacheManager:
    return HTTPCacheManager(
        CacheConfig(cache_enabled=cache_enabled, **cache),
        RedisConfig(redis_url=redis_url),
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_http_cache_manager_init_sets_error_filter_without_global_body_key(
    cache_enabled: bool,
) -> None:
    manager = _manager(cache_enabled=cache_enabled)

    assert isinstance(manager.policy, FilterPolicy)
    assert manager.policy.use_body_key is False
    assert len(manager.policy.response_filters) == 1
    assert isinstance(manager.policy.response_filters[0], NeverCacheErrorsFilter)


def test_http_cache_manager_init_enabled_without_redis_url_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="http_cache.manager"):
        manager = _manager(redis_url=None)

    assert manager.enabled is False
    assert "REDIS_URL is not set" in caplog.text


def test_http_cache_manager_init_enabled_logs_redis_url_without_password(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="http_cache.manager"):
        _manager(redis_url="redis://user:secret@cache-host:6379/0")

    assert "redis://user:***@cache-host:6379/0" in caplog.text
    assert "secret" not in caplog.text


def test_http_cache_manager_init_disabled_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="http_cache.manager"):
        _manager(cache_enabled=False)

    assert caplog.text == ""


@pytest.mark.parametrize(
    ("cache_enabled", "redis_url", "expected"),
    [(True, REDIS_URL, True), (False, REDIS_URL, False), (True, None, False)],
)
def test_enabled_needs_cache_enabled_and_redis_url(
    cache_enabled: bool, redis_url: str | None, expected: bool
) -> None:
    assert (
        _manager(cache_enabled=cache_enabled, redis_url=redis_url).enabled is expected
    )


async def test_get_aiohttp_session_cache_disabled_returns_plain_session() -> None:
    session = _manager(cache_enabled=False).get_aiohttp_session("mal")

    assert type(session) is aiohttp.ClientSession
    await session.close()


def test_get_aiohttp_session_no_running_event_loop_returns_plain_session() -> None:
    with patch(
        "http_cache.manager.aiohttp.ClientSession", autospec=True
    ) as session_class:
        session = _manager().get_aiohttp_session("mal")

    assert session is session_class.return_value


async def test_get_aiohttp_session_enabled_returns_cached_session_with_service_ttl() -> (
    None
):
    manager = _manager(ttl_mal=7200, force_cache=False, always_revalidate=True)

    session = manager.get_aiohttp_session("mal")

    assert isinstance(session, CachedAiohttpSession)
    assert isinstance(session.storage, AsyncRedisStorage)
    assert session.storage.default_ttl == 7200.0
    assert session.storage.key_prefix == "hishel_cache"
    assert (
        session.storage.client.connection_pool.connection_kwargs["decode_responses"]
        is False
    )
    assert session.policy is manager.policy
    assert session.force_cache is False
    assert session.always_revalidate is True
    await session.close()
    await manager.close()


async def test_get_aiohttp_session_unknown_service_uses_default_ttl() -> None:
    manager = _manager()

    session = manager.get_aiohttp_session("unknown")

    assert isinstance(session, CachedAiohttpSession)
    assert session.storage.default_ttl == 86400.0
    await session.close()
    await manager.close()


async def test_get_result_client_enabled_returns_text_client_separate_from_response_client() -> (
    None
):
    manager = _manager()

    result_client = manager.get_result_client()
    session = manager.get_aiohttp_session("mal")

    assert isinstance(result_client, Redis)
    assert result_client.connection_pool.connection_kwargs["decode_responses"] is True
    assert isinstance(session, CachedAiohttpSession)
    assert session.storage.client is not result_client
    assert manager.get_result_client() is result_client
    await session.close()
    await manager.close()


@pytest.mark.parametrize(
    ("cache_enabled", "redis_url"), [(False, REDIS_URL), (True, None)]
)
async def test_get_result_client_caching_off_returns_none(
    cache_enabled: bool, redis_url: str | None
) -> None:
    assert (
        _manager(cache_enabled=cache_enabled, redis_url=redis_url).get_result_client()
        is None
    )


def test_get_result_client_no_running_event_loop_returns_none() -> None:
    assert _manager().get_result_client() is None


async def test_get_result_client_event_loop_changed_creates_new_clients() -> None:
    manager = _manager()
    old_client = create_autospec(Redis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.return_value = False
    manager._result_client = old_client
    manager._response_client = old_client
    manager._event_loop = old_loop

    client = manager.get_result_client()

    assert isinstance(client, Redis)
    assert client is not old_client
    assert manager._response_client is None
    old_client.aclose.assert_not_called()
    await manager.close()


async def test_get_result_client_old_loop_running_closes_old_clients_on_that_loop() -> (
    None
):
    manager = _manager()
    old_response_client = create_autospec(Redis, instance=True)
    old_result_client = create_autospec(Redis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.return_value = True
    manager._response_client = old_response_client
    manager._result_client = old_result_client
    manager._event_loop = old_loop

    with patch(
        "http_cache.manager.asyncio.run_coroutine_threadsafe", autospec=True
    ) as run_on_loop:
        client = manager.get_result_client()

    target_loops = []
    for call in run_on_loop.call_args_list:
        close_coroutine, target_loop = call.args
        close_coroutine.close()
        target_loops.append(target_loop)
    assert target_loops == [old_loop, old_loop]
    old_response_client.aclose.assert_called_once_with()
    old_result_client.aclose.assert_called_once_with()
    assert client is not old_result_client
    await manager.close()


async def test_get_result_client_old_loop_stopped_during_close_creates_new_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _manager()
    old_client = create_autospec(Redis, instance=True)
    old_loop = _old_loop()
    old_loop.is_running.return_value = True
    manager._result_client = old_client
    manager._event_loop = old_loop

    with (
        patch(
            "http_cache.manager.asyncio.run_coroutine_threadsafe",
            autospec=True,
            side_effect=RuntimeError("Event loop is closed"),
        ),
        caplog.at_level(logging.DEBUG, logger="http_cache.manager"),
    ):
        client = manager.get_result_client()

    assert isinstance(client, Redis)
    assert client is not old_client
    assert "Could not close previous Redis client" in caplog.text
    await manager.close()


async def test_close_open_clients_closes_both_and_clears_them() -> None:
    manager = _manager()
    response_client = create_autospec(Redis, instance=True)
    result_client = create_autospec(Redis, instance=True)
    manager._response_client = response_client
    manager._result_client = result_client
    manager._event_loop = asyncio.get_running_loop()

    await manager.close()

    response_client.aclose.assert_awaited_once_with()
    result_client.aclose.assert_awaited_once_with()
    assert manager._response_client is None
    assert manager._result_client is None
    assert manager._event_loop is None


async def test_close_aclose_failure_logs_warning_and_closes_other_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _manager()
    failing_client = create_autospec(Redis, instance=True)
    failing_client.aclose.side_effect = RuntimeError("connection lost")
    result_client = create_autospec(Redis, instance=True)
    manager._response_client = failing_client
    manager._result_client = result_client

    with caplog.at_level(logging.WARNING, logger="http_cache.manager"):
        await manager.close()

    result_client.aclose.assert_awaited_once_with()
    assert manager._response_client is None
    assert "Error closing Redis client" in caplog.text


async def test_close_without_clients_does_nothing() -> None:
    manager = _manager()

    await manager.close()

    assert manager._event_loop is None


async def test_close_then_request_creates_new_client() -> None:
    manager = _manager()
    first = manager.get_result_client()
    await manager.close()

    second = manager.get_result_client()

    assert second is not None
    assert second is not first
    await manager.close()


@pytest.mark.parametrize(
    ("cache_enabled", "redis_url"), [(False, REDIS_URL), (True, None)]
)
def test_get_stats_caching_off_returns_only_disabled_flag(
    cache_enabled: bool, redis_url: str | None
) -> None:
    stats = _manager(cache_enabled=cache_enabled, redis_url=redis_url).get_stats()

    assert stats == {"cache_enabled": False}


def test_get_stats_enabled_returns_storage_and_masked_redis_url() -> None:
    stats = _manager(redis_url="redis://user:secret@cache-host:6379/0").get_stats()

    assert stats == {
        "cache_enabled": True,
        "storage_type": "redis",
        "redis_url": "redis://user:***@cache-host:6379/0",
    }


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

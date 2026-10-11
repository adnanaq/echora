import logging
from unittest.mock import patch

import pytest
from common.config import CacheConfig, RedisConfig
from http_cache import close_cache, configure_cache, get_cache
from http_cache.manager import HTTPCacheManager

REDIS_URL = "redis://localhost:6379/15"


def test_get_cache_unconfigured_returns_disabled_manager(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="http_cache.setup"):
        cache = get_cache()

    assert cache.enabled is False
    assert "Cache not configured" in caplog.text


def test_get_cache_unconfigured_ignores_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", REDIS_URL)

    assert get_cache().enabled is False


def test_get_cache_repeated_call_returns_same_manager() -> None:
    assert get_cache() is get_cache()


def test_configure_cache_returns_manager_get_cache_returns() -> None:
    cache = configure_cache(
        CacheConfig(cache_enabled=True), RedisConfig(redis_url=REDIS_URL)
    )

    assert isinstance(cache, HTTPCacheManager)
    assert cache.enabled is True
    assert get_cache() is cache


def test_get_cache_before_configure_cache_later_calls_return_configured_manager() -> (
    None
):
    unconfigured = get_cache()

    configured = configure_cache(
        CacheConfig(cache_enabled=True), RedisConfig(redis_url=REDIS_URL)
    )

    assert unconfigured.enabled is False
    assert get_cache() is configured


def test_configure_cache_again_replaces_earlier_manager() -> None:
    first = configure_cache(CacheConfig(), RedisConfig(redis_url=REDIS_URL))

    second = configure_cache(CacheConfig(cache_enabled=False), RedisConfig())

    assert get_cache() is second
    assert second is not first


async def test_close_cache_configured_closes_manager() -> None:
    cache = configure_cache(CacheConfig(), RedisConfig(redis_url=REDIS_URL))

    with patch.object(cache, "close", autospec=True) as close:
        await close_cache()

    close.assert_awaited_once_with()


async def test_close_cache_unconfigured_does_nothing() -> None:
    with patch.object(HTTPCacheManager, "close", autospec=True) as close:
        await close_cache()

    close.assert_not_called()

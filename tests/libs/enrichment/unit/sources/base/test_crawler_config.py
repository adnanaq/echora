"""Unit tests for CrawlerRateLimiter and config factories in crawler_config."""

from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.base.crawler_config import (
    CrawlerRateLimiter,
    get_ap_rate_limiter,
    get_docker_browser_config,
    get_docker_crawler_config,
)

# ---------------------------------------------------------------------------
# CrawlerRateLimiter — core behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_acquire_does_not_sleep():
    limiter = CrawlerRateLimiter(min_interval_seconds=0.5, max_per_minute=60)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_acquire_enforces_min_interval():
    limiter = CrawlerRateLimiter(min_interval_seconds=0.5, max_per_minute=60)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        await limiter.acquire()
        mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_enforces_max_per_minute():
    limiter = CrawlerRateLimiter(min_interval_seconds=0.0, max_per_minute=2)

    # Force time so 3 acquires happen within the same minute window.
    times = iter([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 61.0])

    def fake_time():
        return next(times)

    with patch("time.time", side_effect=fake_time):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            await limiter.acquire()
            await limiter.acquire()
            mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_zero_min_interval_no_sleep_between_acquires():
    limiter = CrawlerRateLimiter(min_interval_seconds=0.0, max_per_minute=0)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        await limiter.acquire()
        mock_sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_ap_rate_limiter — singleton + parameters
# ---------------------------------------------------------------------------


def test_get_ap_rate_limiter_returns_singleton():
    a = get_ap_rate_limiter()
    b = get_ap_rate_limiter()
    assert a is b


def test_get_ap_rate_limiter_correct_parameters():
    limiter = get_ap_rate_limiter()
    assert limiter._min_interval == 5.0
    assert limiter._max_per_minute == 12


# ---------------------------------------------------------------------------
# get_docker_browser_config
# ---------------------------------------------------------------------------


def test_docker_browser_config_structure():
    cfg = get_docker_browser_config()
    assert cfg["type"] == "BrowserConfig"
    params = cfg["params"]
    assert params["headless"] is True
    assert params["enable_stealth"] is True


def test_docker_browser_config_extra_headers_merged():
    cfg = get_docker_browser_config(extra_headers={"X-Test": "1"})
    assert cfg["params"]["headers"]["X-Test"] == "1"
    # Default headers still present
    assert "User-Agent" in cfg["params"]["headers"]


def test_docker_browser_config_stealth_off():
    cfg = get_docker_browser_config(stealth=False)
    assert cfg["params"]["enable_stealth"] is False


# ---------------------------------------------------------------------------
# get_docker_crawler_config
# ---------------------------------------------------------------------------


def test_docker_crawler_config_structure():
    schema = {"name": "Test", "baseSelector": "//body", "fields": []}
    cfg = get_docker_crawler_config(schema)
    assert cfg["type"] == "CrawlerRunConfig"
    params = cfg["params"]
    assert params["extraction_strategy"]["type"] == "JsonXPathExtractionStrategy"
    assert params["extraction_strategy"]["params"]["schema"] is schema


def test_docker_crawler_config_delay_included_when_set():
    schema = {}
    cfg = get_docker_crawler_config(schema, delay=2.5)
    assert cfg["params"]["delay_before_return_html"] == 2.5


def test_docker_crawler_config_no_delay_key_when_not_set():
    schema = {}
    cfg = get_docker_crawler_config(schema)
    assert "delay_before_return_html" not in cfg["params"]

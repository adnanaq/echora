from unittest.mock import AsyncMock, patch

import pytest

from enrichment.api_helpers.mal_rate_limiter import MalRateLimiter


@pytest.mark.asyncio
async def test_first_acquire_does_not_sleep():
    limiter = MalRateLimiter(min_interval_seconds=0.5, max_per_minute=60)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_acquire_enforces_min_interval():
    limiter = MalRateLimiter(min_interval_seconds=0.5, max_per_minute=60)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        await limiter.acquire()
        mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_enforces_max_per_minute():
    limiter = MalRateLimiter(min_interval_seconds=0.0, max_per_minute=2)

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


def test_get_shared_limiter_returns_singleton_instance():
    from enrichment.api_helpers.mal_rate_limiter import get_shared_mal_rate_limiter

    a = get_shared_mal_rate_limiter()
    b = get_shared_mal_rate_limiter()
    assert a is b

"""
Integration tests for MalEnrichmentHelper.

NOTE: These tests make REAL HTTP calls to the public MAL API and may be rate-limited.
Set ENABLE_LIVE_API_TESTS=1 environment variable to run them.
"""

from __future__ import annotations

import os
import time

import pytest

from enrichment.api_helpers.mal_enrichment_helper import MalEnrichmentHelper
from enrichment.api_helpers.mal_rate_limiter import get_shared_mal_rate_limiter

# Mark all tests in this module as integration tests.
pytestmark = pytest.mark.integration

# Skip all tests in this module unless explicitly enabled via environment variable.
if not os.getenv("ENABLE_LIVE_API_TESTS"):
    pytestmark = [
        pytestmark,
        pytest.mark.skip(
            reason="Live API tests disabled. Set ENABLE_LIVE_API_TESTS=1 to run these tests. "
            "These tests make real HTTP calls to the public MAL API and may be rate-limited."
        ),
    ]


def test_shared_rate_limiter_is_singleton() -> None:
    """The shared limiter must be per-process singleton (option 1)."""
    a = get_shared_mal_rate_limiter()
    b = get_shared_mal_rate_limiter()
    assert a is b


@pytest.mark.asyncio
async def test_episode_detail_is_cacheable() -> None:
    """
    Call the same episode detail twice and assert the payload is stable.

    If redis/http cache is enabled, the second call is typically much faster, but we only
    assert the payload equality to keep this test robust across environments.
    """
    anime_id = "21"  # One Piece

    async with MalEnrichmentHelper(anime_id) as helper:
        t1 = time.monotonic()
        first = await helper.fetch_episode_detail(1)
        d1 = time.monotonic() - t1

        t2 = time.monotonic()
        second = await helper.fetch_episode_detail(1)
        d2 = time.monotonic() - t2

    assert first is not None
    assert second is not None
    assert first == second

    # Timing is informational and best-effort (do not make this flaky).
    if d1 > 0.1:
        assert d2 <= d1


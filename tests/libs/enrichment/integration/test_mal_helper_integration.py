"""
Integration tests for MalHelper.

NOTE: These tests make REAL HTTP calls to the public MAL API and may be rate-limited.
Set ENABLE_LIVE_API_TESTS=1 environment variable to run them.
"""

from __future__ import annotations

import os

import pytest

from enrichment.sources.mal.mal_helper import MalHelper
from enrichment.crawlers.mal_crawler.mal_base import get_shared_mal_rate_limiter

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

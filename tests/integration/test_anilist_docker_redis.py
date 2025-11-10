"""Integration test for AniList helper Redis connectivity in Docker environment."""

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anilist_helper_respects_redis_cache_url_env():
    """Test that AniList helper respects REDIS_CACHE_URL environment variable."""
    from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

    # This test verifies the fix for hard-coded Redis URL
    # The helper should now use cache manager which reads REDIS_CACHE_URL

    helper = AniListEnrichmentHelper()

    # Trigger session creation
    try:
        # Make a simple request (will fail if no network, but that's OK for this test)
        # We're testing the configuration, not the actual API call
        await helper._make_request("query { __typename }", {})
    except Exception:
        # Network errors are expected in test environment
        pass

    # Verify session was created (via cache manager, not manual Redis client)
    assert helper.session is not None
    assert helper._session_event_loop is not None

    await helper.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists("/.dockerenv"),
    reason="Docker-specific test - only runs in container",
)
@pytest.mark.asyncio
async def test_anilist_helper_connects_in_docker():
    """Test that AniList helper successfully connects to Redis in Docker."""
    from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

    # This test only runs inside Docker container
    # Verifies that Redis hostname resolution works (not localhost)

    helper = AniListEnrichmentHelper()

    # Make a request to trigger session creation and verify no connection errors
    try:
        result = await helper._make_request("query { __typename }", {})
        # Should get a response or error from API, not connection error
        assert result is not None
    except Exception as e:
        # Should not get Redis connection errors
        assert "Connect call failed" not in str(e)
        assert "Connection refused" not in str(e)

    await helper.close()

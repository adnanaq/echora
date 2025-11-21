"""
Test that all API helpers use centralized cache manager correctly.

This test verifies that no helper has hard-coded Redis URLs that would
fail in Docker environments.
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_all_helpers_use_cache_manager_not_hardcoded_redis(mocker):
    """Test that ALL API helpers use cache manager, not hard-coded Redis URLs.

    This test verifies helpers don't create their own Redis clients with hard-coded
    URLs. Instead, they should use http_cache_manager.get_aiohttp_session().

    The cache manager ITSELF will call Redis.from_url, but with a configurable URL
    from environment variables, not a hard-coded localhost URL.
    """

    # Mock cache manager to provide working sessions WITHOUT initializing Redis
    mock_session = mocker.MagicMock()
    mock_session.get = mocker.AsyncMock()
    mock_session.post = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = mocker.AsyncMock(
        return_value={}
    )
    mock_session.get.return_value.__aenter__.return_value.text = mocker.AsyncMock(
        return_value=""
    )
    mock_session.get.return_value.__aenter__.return_value.from_cache = False
    mock_session.post.return_value.__aenter__.return_value.status = 200
    mock_session.post.return_value.__aenter__.return_value.json = mocker.AsyncMock(
        return_value={}
    )
    mock_session.post.return_value.__aenter__.return_value.from_cache = False
    mock_session.post.return_value.__aenter__.return_value.headers = {}
    mock_session.close = mocker.AsyncMock()

    mocker.patch(
        "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
        return_value=mock_session,
    )

    # Track what Redis URLs were used (if any)
    redis_calls = []
    original_redis_from_url = mocker.AsyncMock()

    def track_redis_call(url, **kwargs):
        """
        Record a Redis connection URL and return the captured `original_redis_from_url` callable.
        
        Parameters:
            url (str): The Redis connection URL that was requested.
        
        Returns:
            callable: The `original_redis_from_url` callable used to delegate the actual Redis creation.
        """
        redis_calls.append(url)
        return original_redis_from_url

    mocker.patch("redis.asyncio.Redis.from_url", side_effect=track_redis_call)

    # List of all helpers to test
    helpers_to_test = [
        ("src.enrichment.api_helpers.anilist_helper", "AniListEnrichmentHelper"),
        ("src.enrichment.api_helpers.jikan_helper", "JikanDetailedFetcher"),
        ("src.enrichment.api_helpers.kitsu_helper", "KitsuEnrichmentHelper"),
        ("src.enrichment.api_helpers.anidb_helper", "AniDBEnrichmentHelper"),
    ]

    failed_helpers = []

    for module_path, class_name in helpers_to_test:
        try:
            # Import the helper
            module = __import__(module_path, fromlist=[class_name])
            helper_class = getattr(module, class_name)

            # Clear Redis calls before each helper
            redis_calls_before = len(redis_calls)

            # Create instance (JikanDetailedFetcher needs special args)
            if class_name == "JikanDetailedFetcher":
                helper = helper_class(anime_id="1", data_type="episodes")
            else:
                helper = helper_class()

            # Trigger session creation (different helpers have different methods)
            try:
                if class_name == "AniListEnrichmentHelper":
                    await helper.fetch_anime_by_anilist_id(1)
                elif class_name == "JikanDetailedFetcher":
                    await helper.fetch_episode_detail(1)
                elif class_name == "KitsuEnrichmentHelper":
                    await helper.get_anime_by_id(1)
                elif class_name == "AniDBEnrichmentHelper":
                    await helper.get_anime_by_id(1)
            except Exception:
                # API errors are OK, we're testing cache setup not API calls
                pass

            # Check if THIS helper created any Redis clients
            redis_calls_after = len(redis_calls)
            if redis_calls_after > redis_calls_before:
                new_calls = redis_calls[redis_calls_before:redis_calls_after]
                failed_helpers.append(
                    (class_name, f"Created Redis client directly: {new_calls}")
                )

            # Close helper
            if hasattr(helper, "close"):
                await helper.close()

        except Exception as e:
            # Track which helpers failed
            failed_helpers.append((class_name, f"Exception: {str(e)}"))

    # Report failures
    if failed_helpers:
        pytest.fail(
            f"Some helpers are creating their own Redis clients!\n"
            f"Helpers should use http_cache_manager.get_aiohttp_session() instead.\n"
            f"Failed helpers:\n"
            + "\n".join([f"  - {name}: {err}" for name, err in failed_helpers])
        )


@pytest.mark.asyncio
async def test_anilist_helper_no_hardcoded_redis(mocker):
    """
    Verify that AniListEnrichmentHelper obtains its HTTP session from the centralized cache manager and does not create a Redis client via a hard-coded Redis URL.
    
    This test patches the HTTP cache manager to return a mocked aiohttp session, patches `redis.asyncio.Redis.from_url` to detect any direct Redis usage, triggers a representative fetch to cause session initialization, and asserts that `Redis.from_url` was never called. The helper is closed at the end of the test.
    """
    from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

    # Mock Redis.from_url to detect calls
    mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

    # Mock cache manager
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.AsyncMock()
    mock_session.post.return_value.__aenter__.return_value.status = 200
    mock_session.post.return_value.__aenter__.return_value.json = mocker.AsyncMock(
        return_value={"data": {"Media": {"id": 1}}}
    )
    mock_session.post.return_value.__aenter__.return_value.from_cache = False
    mock_session.post.return_value.__aenter__.return_value.headers = {}

    mocker.patch(
        "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
        return_value=mock_session,
    )

    helper = AniListEnrichmentHelper()

    # Make request to trigger session creation
    await helper.fetch_anime_by_anilist_id(1)

    # Verify no hard-coded Redis URL was used
    mock_redis_from_url.assert_not_called()

    await helper.close()


@pytest.mark.asyncio
async def test_jikan_helper_no_hardcoded_redis(mocker):
    """Specific test for Jikan helper."""
    from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher

    # Mock Redis.from_url
    mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

    # Mock cache manager
    mock_session = mocker.AsyncMock()
    mock_session.get = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = mocker.AsyncMock(
        return_value={"data": {"mal_id": 1}}
    )

    mocker.patch(
        "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
        return_value=mock_session,
    )

    helper = JikanDetailedFetcher(anime_id="1", data_type="episodes")

    # Make request
    try:
        await helper.fetch_episode_detail(1)
    except Exception:
        pass  # API errors OK

    # Verify no hard-coded Redis
    mock_redis_from_url.assert_not_called()


@pytest.mark.asyncio
async def test_kitsu_helper_no_hardcoded_redis(mocker):
    """
    Verify that KitsuEnrichmentHelper uses the centralized HTTP cache session and does not instantiate a Redis client via `redis.asyncio.Redis.from_url`.
    
    Patches the HTTP cache manager to return a mocked aiohttp session, patches `Redis.from_url` to observe calls, invokes `get_anime_by_id`, and asserts that no direct Redis connection was created. API errors during the fetch are tolerated.
    """
    from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    # Mock Redis.from_url
    mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

    # Mock cache manager
    mock_session = mocker.AsyncMock()
    mock_session.get = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = mocker.AsyncMock(
        return_value={"data": {"id": "1"}}
    )

    mocker.patch(
        "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
        return_value=mock_session,
    )

    helper = KitsuEnrichmentHelper()

    # Make request
    try:
        await helper.get_anime_by_id(1)
    except Exception:
        pass  # API errors OK

    # Verify no hard-coded Redis
    mock_redis_from_url.assert_not_called()


@pytest.mark.asyncio
async def test_anidb_helper_no_hardcoded_redis(mocker):
    """
    Verifies that AniDBEnrichmentHelper obtains its HTTP session from the centralized cache manager and does not create a Redis client with a hard-coded URL.
    
    This test patches the HTTP cache manager to return a mocked session and monitors `redis.asyncio.Redis.from_url`; API errors raised during the fetch are tolerated.
    """
    from src.enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    # Mock Redis.from_url
    mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

    # Mock cache manager
    mock_session = mocker.AsyncMock()
    mock_session.get = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.text = mocker.AsyncMock(
        return_value='<?xml version="1.0"?><anime></anime>'
    )

    mocker.patch(
        "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
        return_value=mock_session,
    )

    helper = AniDBEnrichmentHelper()

    # Make request
    try:
        await helper.get_anime_by_id(1)
    except Exception:
        pass  # API errors OK

    # Verify no hard-coded Redis
    mock_redis_from_url.assert_not_called()

    await helper.close()
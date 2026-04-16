"""
Comprehensive unit tests for AniListHelper.

Tests cover:
- Event loop management and session recreation
- GraphQL request handling with caching
- Rate limiting (normal and 429 retry)
- Pagination for characters, staff, episodes
- Error handling and edge cases
- Cache hit detection and rate limiting optimization
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from enrichment.api_helpers.anilist_helper import AniListHelper
from enrichment.exceptions import (
    AniListGraphQLError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)


class TestAniListHelperInit:
    """Test initialization and configuration."""

    def test_init_default_values(self):
        """Test helper initializes with correct default values."""
        helper = AniListHelper()

        assert helper.base_url == "https://graphql.anilist.co"
        assert helper.session is None
        assert helper.rate_limit_remaining == 90
        assert helper._session_event_loop is None

    def test_init_no_session_created(self):
        """Test that session is not created during init."""
        helper = AniListHelper()
        assert helper.session is None


class TestAniListHelperSessionManagement:
    """Test session creation and event loop management."""

    @pytest.mark.asyncio
    async def test_session_created_on_first_request(self):
        """Test that session is created on first request."""
        helper = AniListHelper()
        assert helper.session is None

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        # Patch at the module where it's imported
        with patch(
            "http_cache.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Cache setup failed"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                await helper._make_request("query { test }")

        assert helper.session is not None
        assert helper._session_event_loop is not None

    @pytest.mark.asyncio
    async def test_cached_session_creation(self):
        """Test that cached session is created successfully."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_cached_session = MagicMock()
        mock_cached_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        # Patch at the source modules
        with patch(
            "http_cache.aiohttp_adapter.CachedAiohttpSession",
            return_value=mock_cached_session,
        ):
            with patch("redis.asyncio.Redis.from_url"):
                with patch("http_cache.async_redis_storage.AsyncRedisStorage"):
                    await helper._make_request("query { test }")

        assert helper.session is mock_cached_session

    @pytest.mark.asyncio
    async def test_session_recreated_for_new_event_loop(self):
        """Test that session is recreated when event loop changes."""
        helper = AniListHelper()

        # Create first session
        mock_session_1 = MagicMock()
        mock_session_1.close = AsyncMock()
        mock_session_1.post = MagicMock()
        helper.session = mock_session_1
        helper._session_event_loop = "old_loop"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session_2 = MagicMock()
        mock_session_2.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch(
            "http_cache.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Cache failed"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_session_2):
                await helper._make_request("query { test }")

        # Old session should be closed
        mock_session_1.close.assert_awaited_once()
        # New session should be created
        assert helper.session is mock_session_2
        assert helper._session_event_loop == asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_session_close_error_ignored(self):
        """Test that errors closing old session are ignored."""
        helper = AniListHelper()

        # Create session that fails to close
        mock_session_1 = MagicMock()
        mock_session_1.close = AsyncMock(side_effect=Exception("Close failed"))
        helper.session = mock_session_1
        helper._session_event_loop = "old_loop"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session_2 = MagicMock()
        mock_session_2.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch(
            "http_cache.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Cache failed"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_session_2):
                # Should not raise exception
                await helper._make_request("query { test }")

        # New session created despite close error
        assert helper.session is mock_session_2

    @pytest.mark.asyncio
    async def test_cached_session_fallback_on_error(self):
        """Test fallback to uncached session when cache setup fails."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_uncached_session = MagicMock()
        mock_uncached_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch(
            "http_cache.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Redis unavailable"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_uncached_session):
                await helper._make_request("query { test }")

        # Should create uncached session
        assert helper.session is mock_uncached_session

    @pytest.mark.asyncio
    async def test_session_initialization_failure_raises_runtime_error(self, mocker):
        """Test that RuntimeError is raised if session fails to initialize."""
        helper = AniListHelper()

        # Mock get_aiohttp_session to return None
        mocker.patch(
            "http_cache.instance.http_cache_manager.get_aiohttp_session",
            return_value=None,
        )

        with pytest.raises(RuntimeError, match="Failed to initialize AniList session"):
            await helper._make_request("query { test }")


class TestAniListHelperMakeRequest:
    """Test GraphQL request making with various scenarios."""

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful GraphQL request."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"data": {"Media": {"id": 1, "title": "Test"}}}
        )
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { Media { id title } }", {"id": 1})

        assert result == {"Media": {"id": 1, "title": "Test"}, "_from_cache": False}
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_with_variables(self):
        """Test GraphQL request with variables."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 123}}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        variables = {"id": 123, "type": "ANIME"}
        await helper._make_request("query { test }", variables)

        # Verify variables were passed
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["json"]["variables"] == variables

    @pytest.mark.asyncio
    async def test_make_request_cache_hit(self):
        """Test request with cache hit."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response.from_cache = True
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { test }")

        assert result["_from_cache"] is True

    @pytest.mark.asyncio
    async def test_make_request_from_cache_missing(self):
        """Test request when from_cache attribute is missing."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        # Remove from_cache attribute entirely
        if hasattr(mock_response, "from_cache"):
            delattr(mock_response, "from_cache")
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { test }")

        # Should default to False when attribute is missing
        assert result["_from_cache"] is False

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_tracking(self):
        """Test that rate limit headers are tracked."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {"X-RateLimit-Remaining": "45"}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        await helper._make_request("query { test }")

        assert helper.rate_limit_remaining == 45

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_low_waits(self):
        """Test that low rate limit triggers wait."""
        helper = AniListHelper()
        helper.rate_limit_remaining = 3  # Below threshold of 5

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await helper._make_request("query { test }")

            # Should wait 60 seconds
            mock_sleep.assert_awaited_once_with(60)
            # Rate limit reset to 90
            assert helper.rate_limit_remaining == 90

    @pytest.mark.asyncio
    async def test_make_request_429_retry(self):
        """Test that 429 status triggers retry after wait, and exhausts after max attempts."""
        helper = AniListHelper()

        # First response: 429 rate limit
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Retry-After": "30"}
        mock_response_429.from_cache = False

        # Second response: success
        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"data": {"Media": {"id": 1}}})
        mock_response_ok.from_cache = False
        mock_response_ok.headers = {}

        mock_session = MagicMock()
        # First call returns 429, second call returns success
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock()

        cm_ok = AsyncMock()
        cm_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        cm_ok.__aexit__ = AsyncMock()

        mock_session.post = MagicMock(side_effect=[cm_429, cm_ok])

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await helper._make_request("query { test }")

            # Should wait for Retry-After value
            mock_sleep.assert_awaited_once_with(30)
            # Should succeed on retry
            assert result["Media"]["id"] == 1
            # Should make 2 requests
            assert mock_session.post.call_count == 2

        # Test max retries exhaustion
        helper2 = AniListHelper()
        cm_429_persistent = AsyncMock()
        cm_429_persistent.__aenter__ = AsyncMock(return_value=mock_response_429)
        # Must return False so exceptions raised inside async-with propagate
        cm_429_persistent.__aexit__ = AsyncMock(return_value=False)

        mock_session2 = MagicMock()
        mock_session2.post = MagicMock(return_value=cm_429_persistent)

        helper2.session = mock_session2
        helper2._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ServiceRateLimitedError):
                await helper2._make_request("query { test }")

            # Should give up after 3 attempts
            assert mock_session2.post.call_count == 3

    @pytest.mark.asyncio
    async def test_make_request_429_no_retry_after_header(self):
        """Test 429 handling when Retry-After header is missing."""
        helper = AniListHelper()

        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {}  # No Retry-After

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"data": {}})
        mock_response_ok.from_cache = False
        mock_response_ok.headers = {}

        mock_session = MagicMock()
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock()

        cm_ok = AsyncMock()
        cm_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        cm_ok.__aexit__ = AsyncMock()

        mock_session.post = MagicMock(side_effect=[cm_429, cm_ok])

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await helper._make_request("query { test }")

            # Should use default 60 seconds
            mock_sleep.assert_awaited_once_with(60)

    @pytest.mark.asyncio
    async def test_make_request_graphql_errors(self):
        """Test handling of GraphQL errors in response."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"errors": [{"message": "Field not found"}], "data": None}
        )
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Should raise AniListGraphQLError on GraphQL errors
        with pytest.raises(AniListGraphQLError):
            await helper._make_request("query { test }")

    @pytest.mark.asyncio
    async def test_make_request_4xx_client_error_not_retried(self):
        """4xx client errors raise immediately without retry."""
        helper = AniListHelper()

        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404
        )

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.from_cache = False
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock(side_effect=error)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with pytest.raises(ServiceNetworkError):
            await helper._make_request("query { test }")

        # Should only attempt once (no retry for 4xx)
        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_make_request_http_error(self):
        """Test handling of HTTP errors."""
        helper = AniListHelper()

        # Create exception for raise_for_status to raise (4xx → ServiceNetworkError)
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=400
        )

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.from_cache = False
        mock_response.headers = {}
        # Make raise_for_status() call raise the exception
        mock_response.raise_for_status = MagicMock(side_effect=error)

        # Setup context manager properly with exception handling
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        # __aexit__ should return False to propagate the exception
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Should raise ServiceNetworkError on HTTP errors
        with pytest.raises(ServiceNetworkError):
            await helper._make_request("query { test }")

    @pytest.mark.asyncio
    async def test_make_request_exception(self):
        """Test handling of request exceptions."""
        import aiohttp

        helper = AniListHelper()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Network error"))

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Should raise ServiceNetworkError on network exception
        with pytest.raises(ServiceNetworkError):
            await helper._make_request("query { test }")


class TestAniListHelperQueryBuilders:
    """Test query builder methods."""

    def test_get_media_query_fields(self):
        """Test media query fields are returned."""
        helper = AniListHelper()
        fields = helper._get_media_query_fields()

        # Should contain key fields
        assert "id" in fields
        assert "idMal" in fields
        assert "title" in fields
        assert "description" in fields
        # Note: characters and staff are not in the query fields, they are fetched separately
        assert "genres" in fields

    def test_build_query_by_anilist_id(self):
        """Test AniList ID query builder."""
        helper = AniListHelper()
        query = helper._build_query_by_anilist_id()

        assert "query ($id: Int)" in query
        assert "Media(id: $id, type: ANIME)" in query
        # Should include media fields
        assert "title" in query


class TestAniListHelperFetchMethods:
    """Test data fetching methods."""

    @pytest.mark.asyncio
    async def test_fetch_anime_success(self):
        """Test successful anime fetch by AniList ID."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {"id": 21, "title": {"romaji": "One Piece"}},
                "_from_cache": False,
            }
        )

        result = await helper.fetch_anime(21)

        assert result == {"id": 21, "title": {"romaji": "One Piece"}}
        helper._make_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_anime_not_found(self):
        """Test anime fetch when Media is not in response."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime(99999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anime_empty_response(self):
        """Test anime fetch with empty response."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={})

        result = await helper.fetch_anime(21)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anime_by_mal_id_success(self):
        """Test successful anime fetch by MAL ID."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {"id": 21, "idMal": 21, "title": {"romaji": "One Piece"}},
                "_from_cache": False,
            }
        )

        result = await helper.fetch_anime_by_mal_id(21)

        assert result == {"id": 21, "idMal": 21, "title": {"romaji": "One Piece"}}
        call_args = helper._make_request.call_args
        assert call_args[0][1] == {"idMal": 21}

    @pytest.mark.asyncio
    async def test_fetch_anime_by_mal_id_not_found(self):
        """Test anime fetch by MAL ID when not found."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime_by_mal_id(99999)

        assert result is None

    @pytest.mark.asyncio
    async def test_build_query_by_mal_id(self):
        """Test MAL ID query builder produces valid GraphQL."""
        helper = AniListHelper()
        query = helper._build_query_by_mal_id()

        assert "idMal: $idMal" in query
        assert "type: ANIME" in query


class TestAniListHelperPagination:
    """Test pagination handling."""

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_single_page(self):
        """Test fetching single page of data."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {
                    "characters": {
                        "edges": [{"node": {"id": 1}}, {"node": {"id": 2}}],
                        "pageInfo": {"hasNextPage": False},
                    }
                },
                "_from_cache": False,
            }
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert len(result) == 2
        assert result[0] == {"node": {"id": 1}}
        helper._make_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_multiple_pages(self):
        """Test fetching multiple pages of data."""
        helper = AniListHelper()

        # Page 1: has next page
        response_page1 = {
            "Media": {
                "characters": {
                    "edges": [{"node": {"id": 1}}, {"node": {"id": 2}}],
                    "pageInfo": {"hasNextPage": True},
                }
            },
            "_from_cache": False,
        }

        # Page 2: no next page
        response_page2 = {
            "Media": {
                "characters": {
                    "edges": [{"node": {"id": 3}}, {"node": {"id": 4}}],
                    "pageInfo": {"hasNextPage": False},
                }
            },
            "_from_cache": False,
        }

        helper._make_request = AsyncMock(side_effect=[response_page1, response_page2])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await helper._fetch_paginated_data(21, "query", "characters")

            assert len(result) == 4
            assert result[2] == {"node": {"id": 3}}
            assert helper._make_request.await_count == 2
            # Sleep happens between pages only (not after the last page)
            assert mock_sleep.await_count == 1
            mock_sleep.assert_awaited_with(0.5)

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_cache_hit_no_sleep(self):
        """Test that cache hits don't trigger rate limiting sleep."""
        helper = AniListHelper()

        response_page1 = {
            "Media": {
                "characters": {
                    "edges": [{"node": {"id": 1}}],
                    "pageInfo": {"hasNextPage": True},
                }
            },
            "_from_cache": True,  # Cache hit
        }

        response_page2 = {
            "Media": {
                "characters": {
                    "edges": [{"node": {"id": 2}}],
                    "pageInfo": {"hasNextPage": False},
                }
            },
            "_from_cache": True,  # Cache hit
        }

        helper._make_request = AsyncMock(side_effect=[response_page1, response_page2])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await helper._fetch_paginated_data(21, "query", "characters")

            assert len(result) == 2
            # Should NOT rate limit for cache hits
            mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_empty_response(self):
        """Test pagination with empty response."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={})

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_missing_media(self):
        """Test pagination when Media is missing."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_missing_data_key(self):
        """Test pagination when data key is missing."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={"Media": {}, "_from_cache": False}
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_empty_edges(self):
        """Test pagination with empty edges array."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {
                    "characters": {"edges": [], "pageInfo": {"hasNextPage": False}}
                },
                "_from_cache": False,
            }
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_missing_page_info(self):
        """Test pagination when pageInfo is missing."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {
                    "characters": {
                        "edges": [{"node": {"id": 1}}]
                        # No pageInfo
                    }
                },
                "_from_cache": False,
            }
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        # Should stop after first page
        assert len(result) == 1
        helper._make_request.assert_awaited_once()


class TestAniListHelperSpecificFetchers:
    """Test specific data type fetchers (characters, staff, episodes)."""

    @pytest.mark.asyncio
    async def test_fetch_characters(self):
        """Test fetching all characters."""
        helper = AniListHelper()
        helper._fetch_paginated_data = AsyncMock(
            return_value=[
                {"node": {"id": 1, "name": {"full": "Monkey D. Luffy"}}, "role": "MAIN"}
            ]
        )

        result = await helper.fetch_characters(21)

        assert len(result) == 1
        assert result[0]["node"]["name"]["full"] == "Monkey D. Luffy"
        helper._fetch_paginated_data.assert_awaited_once()
        # Verify correct query was passed
        call_args = helper._fetch_paginated_data.call_args
        assert "characters" in call_args[0][2]


class TestAniListHelperFetchAllData:
    """Test fetch_all and _fetch_all_data_by_mal_id."""

    @pytest.mark.asyncio
    async def test_fetch_all_returns_none_when_no_anilist_url(self):
        """Returns None immediately when anilist_url missing from ids."""
        result = await AniListHelper().fetch_all({}, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_returns_none_when_both_empty(self):
        """Returns None when both anime and characters are falsy."""
        helper = AniListHelper()
        helper.fetch_anime_canonical = AsyncMock(return_value=None)
        helper.fetch_characters_canonical = AsyncMock(return_value=[])

        result = await helper.fetch_all(
            {"anilist_url": "https://anilist.co/anime/21"}, {}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_returns_dict_when_anime_present(self):
        """Returns dict with anime and characters when anime resolves."""
        helper = AniListHelper()
        helper.fetch_anime_canonical = AsyncMock(return_value={"title": "One Piece"})
        helper.fetch_characters_canonical = AsyncMock(return_value=[{"name": "Luffy"}])

        result = await helper.fetch_all(
            {"anilist_url": "https://anilist.co/anime/21"}, {}
        )
        assert result == {
            "anime": {"title": "One Piece"},
            "characters": [{"name": "Luffy"}],
        }

    @pytest.mark.asyncio
    async def test__fetch_all_data_by_mal_id_success(self):
        """Fetches by MAL ID, resolves AniList ID, injects characters."""
        helper = AniListHelper()

        helper.fetch_anime_by_mal_id = AsyncMock(
            return_value={"id": 21, "idMal": 21, "title": {"romaji": "One Piece"}}
        )
        helper.fetch_characters = AsyncMock(
            return_value=[{"node": {"id": 1}, "role": "MAIN"}]
        )

        result = await helper._fetch_all_data_by_mal_id(21)

        assert result is not None
        assert result["id"] == 21
        assert "characters" in result

    @pytest.mark.asyncio
    async def test__fetch_all_data_by_mal_id_not_found(self):
        """Returns None when MAL lookup finds nothing."""
        helper = AniListHelper()
        helper.fetch_anime_by_mal_id = AsyncMock(return_value=None)

        result = await helper._fetch_all_data_by_mal_id(99999)

        assert result is None


class TestAniListHelperClose:
    """Test cleanup methods."""

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test closing helper with active session."""
        helper = AniListHelper()

        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        helper.session = mock_session

        await helper.close()

        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test closing helper without session."""
        helper = AniListHelper()
        helper.session = None

        # Should not raise exception
        await helper.close()


class TestAniListHelperEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_unicode_in_graphql_query(self):
        """Test handling of Unicode characters in GraphQL queries and responses."""
        helper = AniListHelper()

        unicode_data = {
            "Media": {
                "id": 1,
                "title": {
                    "romaji": "進撃の巨人",
                    "native": "進撃の巨人",
                    "english": "Attack on Titan",
                },
            },
            "_from_cache": False,
        }

        helper._make_request = AsyncMock(return_value=unicode_data)

        result = await helper.fetch_anime(1)

        assert result is not None
        assert "進撃の巨人" in str(result)

    @pytest.mark.asyncio
    async def test_rate_limit_exactly_five(self):
        """Test rate limiting boundary at exactly 5 remaining."""
        helper = AniListHelper()
        helper.rate_limit_remaining = 5  # Exactly at threshold

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await helper._make_request("query { test }")

            # Should NOT wait at exactly 5 (threshold is < 5)
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_exactly_four(self):
        """Test rate limiting boundary at exactly 4 remaining."""
        helper = AniListHelper()
        helper.rate_limit_remaining = 4  # Below threshold

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await helper._make_request("query { test }")

            # Should wait at 4
            mock_sleep.assert_called_once_with(60)

    @pytest.mark.asyncio
    async def test_empty_string_variables(self):
        """Test GraphQL request with empty string variables."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Empty string variables should be handled
        await helper._make_request("query { test }", {"name": "", "id": 0})

        # Verify request was made
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON in response."""
        import json

        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
        )
        mock_response.from_cache = False
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()  # Should not raise

        # Setup context manager that raises during json()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Should raise on JSON parse error
        with pytest.raises(ServiceNetworkError):
            await helper._make_request("query { test }")

    @pytest.mark.asyncio
    async def test_very_large_paginated_results(self):
        """Test handling of very large paginated result sets."""
        helper = AniListHelper()

        # Simulate 10 pages with 25 items each
        responses = []
        for page in range(10):
            has_next = page < 9
            responses.append(
                {
                    "Media": {
                        "characters": {
                            "edges": [
                                {"node": {"id": i}}
                                for i in range(page * 25, (page + 1) * 25)
                            ],
                            "pageInfo": {"hasNextPage": has_next},
                        }
                    },
                    "_from_cache": False,
                }
            )

        helper._make_request = AsyncMock(side_effect=responses)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await helper._fetch_paginated_data(21, "query", "characters")

            # Should have all 250 items
            assert len(result) == 250
            assert helper._make_request.await_count == 10

    @pytest.mark.asyncio
    async def test_negative_anilist_id(self):
        """Test fetching with negative AniList ID."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime(-1)

        # Should handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_anilist_id(self):
        """Test fetching with zero AniList ID."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime(0)

        # Should handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_pagination_with_missing_edges_key(self):
        """Test pagination when edges key is missing from response."""
        helper = AniListHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {
                    "characters": {
                        # No edges key
                        "pageInfo": {"hasNextPage": False}
                    }
                },
                "_from_cache": False,
            }
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        # Should return empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_rate_limit_header_missing(self):
        """Test when rate limit header is missing from response."""
        helper = AniListHelper()
        initial_rate_limit = helper.rate_limit_remaining

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {}})
        mock_response.from_cache = False
        mock_response.headers = {}  # No rate limit header

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        await helper._make_request("query { test }")

        # Rate limit should remain unchanged
        assert helper.rate_limit_remaining == initial_rate_limit

    @pytest.mark.asyncio
    async def test_graphql_error_with_data(self):
        """Test GraphQL response with both errors and partial data."""
        helper = AniListHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "errors": [{"message": "Some field failed"}],
                "data": {"Media": {"id": 1}},  # Partial data
            }
        )
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        # Should raise on GraphQL errors
        with pytest.raises(AniListGraphQLError):
            await helper._make_request("query { test }")

    @pytest.mark.asyncio
    async def test_429_retry_with_very_long_wait(self):
        """Test 429 handling with very long Retry-After value."""
        helper = AniListHelper()

        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Retry-After": "3600"}  # 1 hour

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"data": {}})
        mock_response_ok.from_cache = False
        mock_response_ok.headers = {}

        mock_session = MagicMock()
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock()

        cm_ok = AsyncMock()
        cm_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        cm_ok.__aexit__ = AsyncMock()

        mock_session.post = MagicMock(side_effect=[cm_429, cm_ok])

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await helper._make_request("query { test }")

            # Should wait exactly the Retry-After value
            mock_sleep.assert_awaited_once_with(3600)


class TestAniListHelperErrorPaths:
    """Error and boundary paths: invalid URL, 403, 5xx."""

    def test_extract_anilist_id_invalid_url(self):
        """_extract_anilist_id raises ValueError for non-numeric last segment."""
        from enrichment.api_helpers.anilist_helper import _extract_anilist_id

        with pytest.raises(ValueError, match="Cannot extract AniList ID"):
            _extract_anilist_id("https://anilist.co/anime/not-a-number")

    @pytest.mark.asyncio
    async def test_execute_request_403_raises_service_blocked(self):
        """403 response raises ServiceBlockedError."""
        from enrichment.exceptions import ServiceBlockedError

        helper = AniListHelper()
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.from_cache = False
        mock_response.headers = {}

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with pytest.raises(ServiceBlockedError):
            await helper._execute_request("query { test }")

    @pytest.mark.asyncio
    async def test_execute_request_5xx_reraises(self):
        """5xx ClientResponseError is re-raised directly (not wrapped)."""
        helper = AniListHelper()
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503
        )
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.from_cache = False
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock(side_effect=error)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await helper._execute_request("query { test }")
        assert exc_info.value.status == 503


class TestAniListHelperCacheIntegration:
    """Test cache manager integration."""

    @pytest.mark.asyncio
    async def test_anilist_helper_uses_cache_manager(self, mocker):
        """Test that AniListHelper uses centralized cache manager.

        Body-key caching (for GraphQL POST requests) is enabled globally via
        FilterPolicy.use_body_key = True on the HTTPCacheManager policy, not via
        per-session X-Hishel-Body-Key headers.
        """
        from enrichment.api_helpers.anilist_helper import AniListHelper
        from http_cache.instance import http_cache_manager

        # Create proper mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"data": {"Media": {"id": 1, "title": {"romaji": "Test"}}}}
        )
        mock_response.from_cache = False
        mock_response.headers = {}

        # Create mock session with proper context manager for post
        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )
        mock_session.close = AsyncMock()

        # Mock the cache manager's get_aiohttp_session method
        mock_get_session = mocker.patch.object(
            http_cache_manager, "get_aiohttp_session", return_value=mock_session
        )

        helper = AniListHelper()

        # Trigger session creation by making a request
        result = await helper.fetch_anime(1)

        # Verify cache manager was called with correct parameters
        mock_get_session.assert_called_once()
        call_args = mock_get_session.call_args
        assert call_args[0][0] == "anilist"  # service name
        assert "timeout" in call_args[1]
        # Body-key caching is handled globally by FilterPolicy.use_body_key = True
        # on the HTTPCacheManager, not via per-session X-Hishel-Body-Key headers.
        assert "headers" not in call_args[1]

        # Verify the session was used for the request
        assert result is not None

        await helper.close()

    @pytest.mark.asyncio
    async def test_anilist_helper_does_not_create_manual_redis_client(self, mocker):
        """Test that AniListHelper does NOT manually create Redis clients."""
        from enrichment.api_helpers.anilist_helper import AniListHelper

        # Mock Redis.from_url to detect if it's called
        mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

        # Mock cache manager to provide a working session
        mock_session = mocker.AsyncMock()

        # Create async context manager for post()
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.json = mocker.AsyncMock(
            return_value={"data": {"Media": {"id": 1, "title": {"romaji": "Test"}}}}
        )
        mock_response.from_cache = False
        mock_response.headers = {}
        mock_response.raise_for_status = mocker.MagicMock()

        mock_cm = mocker.AsyncMock()
        mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

        mock_session.post = mocker.MagicMock(return_value=mock_cm)

        mocker.patch(
            "http_cache.instance.http_cache_manager.get_aiohttp_session",
            return_value=mock_session,
        )

        helper = AniListHelper()

        # Make a request to trigger session creation
        await helper.fetch_anime(1)

        # Verify Redis.from_url was NOT called (no manual Redis client creation)
        mock_redis_from_url.assert_not_called()

        await helper.close()


class TestAniListHelperCLI:
    """Test CLI main function."""

    @pytest.mark.asyncio
    async def test_main_with_anilist_id_success(self, tmp_path):
        """CLI with --url calls fetch_all and returns 0."""
        import sys
        from unittest.mock import patch

        test_args = [
            "script_name",
            "--url",
            "https://anilist.co/anime/21",
            "--output",
            str(tmp_path),
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all = AsyncMock(
                    return_value={"anime": {"title": "One Piece"}, "characters": []}
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 0
                mock_helper_instance.fetch_all.assert_awaited_once_with(
                    {"anilist_url": "https://anilist.co/anime/21"}, {}, str(tmp_path)
                )
                mock_helper_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_with_no_data_found(self, tmp_path):
        """CLI returns 1 when fetch_anime_canonical returns None."""
        import sys
        from unittest.mock import patch

        test_args = [
            "script_name",
            "--url",
            "https://anilist.co/anime/99999",
            "--output",
            str(tmp_path),
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all = AsyncMock(return_value=None)
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 1
                mock_helper_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_with_exception_still_closes(self, tmp_path):
        """CLI ensures helper.close() is called even on exception."""
        import sys
        from unittest.mock import patch

        test_args = [
            "script_name",
            "--url",
            "https://anilist.co/anime/21",
            "--output",
            str(tmp_path),
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all = AsyncMock(
                    side_effect=Exception("API Error")
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 1
                mock_helper_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_with_mal_id_success(self, tmp_path):
        """CLI with --mal-id resolves AniList ID and calls fetch_all."""
        import sys
        from unittest.mock import patch

        test_args = ["script_name", "--mal-id", "21", "--output", str(tmp_path)]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_anime_by_mal_id = AsyncMock(
                    return_value={"id": 21, "idMal": 21}
                )
                mock_helper_instance.fetch_all = AsyncMock(
                    return_value={"anime": {"title": "One Piece"}, "characters": []}
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 0
                mock_helper_instance.fetch_anime_by_mal_id.assert_awaited_once_with(21)
                mock_helper_instance.fetch_all.assert_awaited_once_with(
                    {"anilist_url": "https://anilist.co/anime/21"}, {}, str(tmp_path)
                )

    @pytest.mark.asyncio
    async def test_main_with_mal_id_not_found(self):
        """CLI with --mal-id returns 1 when MAL lookup fails."""
        import sys
        from unittest.mock import patch

        test_args = ["script_name", "--mal-id", "99999"]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_anime_by_mal_id = AsyncMock(
                    return_value=None
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 1
                mock_helper_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_with_mal_id_no_anilist_id_in_response(self):
        """CLI returns 1 when MAL response has no 'id' field for AniList."""
        import sys
        from unittest.mock import patch

        test_args = ["script_name", "--mal-id", "21"]

        with patch.object(sys, "argv", test_args):
            with patch(
                "enrichment.api_helpers.anilist_helper.AniListHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                # Response has no 'id' field → anilist_id is falsy
                mock_helper_instance.fetch_anime_by_mal_id = AsyncMock(
                    return_value={"idMal": 21}
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from enrichment.api_helpers.anilist_helper import main

                exit_code = await main()

                assert exit_code == 1
                mock_helper_instance.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anilist_helper.AniListHelper")
async def test_main_function_success(mock_helper_class, tmp_path):
    """Test main() function handles successful execution."""
    from enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(
        return_value={"anime": {"title": "Test"}, "characters": []}
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch(
        "sys.argv",
        [
            "script.py",
            "--url",
            "https://anilist.co/anime/21",
            "--output",
            str(tmp_path),
        ],
    ):
        exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_all.assert_awaited_once_with(
        {"anilist_url": "https://anilist.co/anime/21"}, {}, str(tmp_path)
    )
    mock_helper.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anilist_helper.AniListHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--url", "https://anilist.co/anime/99999"]):
        exit_code = await main()

    assert exit_code == 1
    mock_helper.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anilist_helper.AniListHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(side_effect=Exception("API error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--url", "https://anilist.co/anime/21"]):
        exit_code = await main()

    assert exit_code == 1
    mock_helper.close.assert_awaited_once()


# --- Tests for context manager protocol ---


class TestAniListHelperContextManager:
    """Test async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_protocol(self):
        """Test AniListHelper implements async context manager protocol."""
        async with AniListHelper() as helper:
            assert helper is not None
            assert isinstance(helper, AniListHelper)
            assert helper.session is None  # Lazy init - not created yet
        # Should exit cleanly, closing session if it was created

    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self):
        """Test that context manager closes session on exit."""
        helper = AniListHelper()

        # Create a mock session
        mock_session = AsyncMock()
        helper.session = mock_session

        async with helper:
            assert helper.session is mock_session

        # Session should be closed after context exit
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        helper = AniListHelper()
        mock_session = AsyncMock()
        helper.session = mock_session

        with pytest.raises(ValueError, match="Test error"):
            async with helper:
                raise ValueError("Test error")

        # Session should still be closed despite exception
        mock_session.close.assert_awaited_once()


class TestAniListHelperCanonicalMethods:
    """Tests for fetch_anime_canonical and fetch_characters_canonical."""

    # ------------------------------------------------------------------
    # fetch_anime_canonical
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_anime_canonical_success(self):
        """Returns canonical dict when raw data is found and valid."""
        from unittest.mock import patch

        helper = AniListHelper()
        helper.fetch_anime = AsyncMock(
            return_value={"id": 21, "title": {"romaji": "One Piece"}}
        )

        canonical = {"title": "One Piece", "type": "TV", "status": "ONGOING"}

        with patch(
            "enrichment.api_helpers.anilist_helper.anime_from_anilist",
            return_value=canonical,
        ) as mock_map:
            result = await helper.fetch_anime_canonical("https://anilist.co/anime/21")

        assert result == canonical
        mock_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_anime_canonical_extracts_id_from_url(self):
        """fetch_anime is called with the numeric ID extracted from the URL."""
        helper = AniListHelper()
        helper.fetch_anime = AsyncMock(return_value=None)

        await helper.fetch_anime_canonical("https://anilist.co/anime/21")

        helper.fetch_anime.assert_awaited_once_with(21)

    @pytest.mark.asyncio
    async def test_fetch_anime_canonical_not_found(self):
        """Returns None when fetch_anime returns None."""
        helper = AniListHelper()
        helper.fetch_anime = AsyncMock(return_value=None)

        result = await helper.fetch_anime_canonical("https://anilist.co/anime/99999")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anime_canonical_writes_jsonl(self, tmp_path):
        """Writes canonical dict as JSONL when temp_dir is given."""
        import json
        from unittest.mock import patch

        helper = AniListHelper()
        helper.fetch_anime = AsyncMock(
            return_value={"id": 21, "title": {"romaji": "One Piece"}}
        )

        canonical = {"title": "One Piece", "episode_count": 1100}

        with patch(
            "enrichment.api_helpers.anilist_helper.anime_from_anilist",
            return_value=canonical,
        ):
            result = await helper.fetch_anime_canonical(
                "https://anilist.co/anime/21", temp_dir=str(tmp_path)
            )

        assert result == canonical
        out_file = tmp_path / "anilist.jsonl"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["title"] == "One Piece"

    @pytest.mark.asyncio
    async def test_fetch_anime_canonical_no_output_dir(self):
        """Does not write any file when temp_dir is None."""
        from unittest.mock import patch

        helper = AniListHelper()
        helper.fetch_anime = AsyncMock(
            return_value={"id": 21, "title": {"romaji": "One Piece"}}
        )

        with patch(
            "enrichment.api_helpers.anilist_helper.anime_from_anilist",
            return_value={"title": "One Piece"},
        ):
            with patch(
                "enrichment.api_helpers.anilist_helper.append_jsonl"
            ) as mock_append:
                await helper.fetch_anime_canonical(
                    "https://anilist.co/anime/21", temp_dir=None
                )
                mock_append.assert_not_called()

    # ------------------------------------------------------------------
    # fetch_characters_canonical
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_characters_canonical_success(self):
        """Returns list of canonical dicts for each valid edge."""
        from unittest.mock import patch

        helper = AniListHelper()
        raw_edges = [
            {"node": {"id": 1, "name": {"full": "Luffy"}}, "role": "MAIN"},
            {"node": {"id": 2, "name": {"full": "Zoro"}}, "role": "MAIN"},
        ]
        helper.fetch_characters = AsyncMock(return_value=raw_edges)

        char_canonical = [{"name": "Luffy"}, {"name": "Zoro"}]

        with patch(
            "enrichment.api_helpers.anilist_helper.character_from_anilist",
            side_effect=char_canonical,
        ):
            result = await helper.fetch_characters_canonical(
                "https://anilist.co/anime/21"
            )

        assert len(result) == 2
        assert result[0]["name"] == "Luffy"

    @pytest.mark.asyncio
    async def test_fetch_characters_canonical_empty(self):
        """Returns empty list when no character edges exist."""
        helper = AniListHelper()
        helper.fetch_characters = AsyncMock(return_value=[])

        result = await helper.fetch_characters_canonical("https://anilist.co/anime/21")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_characters_canonical_skips_invalid_edges(self):
        """Invalid edges are silently skipped; valid ones still returned."""
        from unittest.mock import patch

        helper = AniListHelper()
        helper.fetch_characters = AsyncMock(
            return_value=[
                {"node": {"id": 1}, "role": "MAIN"},
                {"bad": "data"},  # will fail model_validate
                {"node": {"id": 3}, "role": "SUPPORTING"},
            ]
        )

        def _side_effect(edge):
            if not hasattr(edge, "node") or edge.node is None:
                raise ValueError("bad edge")
            return {"name": f"char_{edge.node.id}"}

        with patch(
            "enrichment.api_helpers.anilist_helper.AniListCharacterEdge"
        ) as MockEdge:
            # First call succeeds, second raises, third succeeds
            mock_edge1 = MagicMock()
            mock_edge3 = MagicMock()
            MockEdge.model_validate = MagicMock(
                side_effect=[mock_edge1, ValueError("bad"), mock_edge3]
            )

            with patch(
                "enrichment.api_helpers.anilist_helper.character_from_anilist",
                side_effect=[{"name": "Luffy"}, {"name": "Nami"}],
            ):
                result = await helper.fetch_characters_canonical(
                    "https://anilist.co/anime/21"
                )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_characters_canonical_writes_jsonl(self, tmp_path):
        """Writes each canonical character as a JSONL line."""
        import json
        from unittest.mock import patch

        helper = AniListHelper()
        helper.fetch_characters = AsyncMock(
            return_value=[
                {"node": {"id": 1}, "role": "MAIN"},
                {"node": {"id": 2}, "role": "SUPPORTING"},
            ]
        )

        chars = [{"name": "Luffy"}, {"name": "Zoro"}]

        with patch(
            "enrichment.api_helpers.anilist_helper.character_from_anilist",
            side_effect=chars,
        ):
            result = await helper.fetch_characters_canonical(
                "https://anilist.co/anime/21", temp_dir=str(tmp_path)
            )

        assert len(result) == 2
        out_file = tmp_path / "anilist_characters.jsonl"
        assert out_file.exists()
        lines = [json.loads(l) for l in out_file.read_text().splitlines()]
        assert lines[0]["name"] == "Luffy"
        assert lines[1]["name"] == "Zoro"

    @pytest.mark.asyncio
    async def test_fetch_characters_canonical_no_output_when_empty(self, tmp_path):
        """Does not write JSONL file when canonical list is empty."""
        helper = AniListHelper()
        helper.fetch_characters = AsyncMock(return_value=[])

        await helper.fetch_characters_canonical(
            "https://anilist.co/anime/21", temp_dir=str(tmp_path)
        )

        out_file = tmp_path / "anilist_characters.jsonl"
        assert not out_file.exists()

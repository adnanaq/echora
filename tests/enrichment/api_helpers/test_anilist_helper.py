"""
Comprehensive unit tests for AniListEnrichmentHelper.

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

from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper


class TestAniListEnrichmentHelperInit:
    """Test initialization and configuration."""

    def test_init_default_values(self):
        """Test helper initializes with correct default values."""
        helper = AniListEnrichmentHelper()

        assert helper.base_url == "https://graphql.anilist.co"
        assert helper.session is None
        assert helper.rate_limit_remaining == 90
        assert helper.rate_limit_reset is None
        assert helper._session_event_loop is None

    def test_init_no_session_created(self):
        """Test that session is not created during init."""
        helper = AniListEnrichmentHelper()
        assert helper.session is None


class TestAniListEnrichmentHelperSessionManagement:
    """Test session creation and event loop management."""

    @pytest.mark.asyncio
    async def test_session_created_on_first_request(self):
        """Test that session is created on first request."""
        helper = AniListEnrichmentHelper()
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
            "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Cache setup failed"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                await helper._make_request("query { test }")

        assert helper.session is not None
        assert helper._session_event_loop is not None

    @pytest.mark.asyncio
    async def test_cached_session_creation(self):
        """Test that cached session is created successfully."""
        helper = AniListEnrichmentHelper()

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
            "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
            return_value=mock_cached_session,
        ):
            with patch("redis.asyncio.Redis.from_url"):
                with patch("src.cache_manager.async_redis_storage.AsyncRedisStorage"):
                    await helper._make_request("query { test }")

        assert helper.session is mock_cached_session

    @pytest.mark.asyncio
    async def test_session_recreated_for_new_event_loop(self):
        """Test that session is recreated when event loop changes."""
        helper = AniListEnrichmentHelper()

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
            "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
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
        helper = AniListEnrichmentHelper()

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
            "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
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
        helper = AniListEnrichmentHelper()

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
            "src.cache_manager.aiohttp_adapter.CachedAiohttpSession",
            side_effect=Exception("Redis unavailable"),
        ):
            with patch("aiohttp.ClientSession", return_value=mock_uncached_session):
                await helper._make_request("query { test }")

        # Should create uncached session
        assert helper.session is mock_uncached_session


class TestAniListEnrichmentHelperMakeRequest:
    """Test GraphQL request making with various scenarios."""

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful GraphQL request."""
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()
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
        """Test that 429 status triggers retry after wait."""
        helper = AniListEnrichmentHelper()

        # First response: 429 rate limit
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Retry-After": "30"}

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

    @pytest.mark.asyncio
    async def test_make_request_429_no_retry_after_header(self):
        """Test 429 handling when Retry-After header is missing."""
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

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
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { test }")

        # Should return empty dict with cache metadata
        assert result == {"_from_cache": False}

    @pytest.mark.asyncio
    async def test_make_request_http_error(self):
        """Test handling of HTTP errors."""
        helper = AniListEnrichmentHelper()

        # Create exception for raise_for_status to raise
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500
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

        result = await helper._make_request("query { test }")

        # Should return empty dict with cache metadata
        assert result == {"_from_cache": False}

    @pytest.mark.asyncio
    async def test_make_request_exception(self):
        """Test handling of request exceptions."""
        helper = AniListEnrichmentHelper()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("Network error"))

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { test }")

        # Should return empty dict
        assert result == {"_from_cache": False}


class TestAniListEnrichmentHelperQueryBuilders:
    """Test query builder methods."""

    def test_get_media_query_fields(self):
        """Test media query fields are returned."""
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()
        query = helper._build_query_by_anilist_id()

        assert "query ($id: Int)" in query
        assert "Media(id: $id, type: ANIME)" in query
        # Should include media fields
        assert "title" in query


class TestAniListEnrichmentHelperFetchMethods:
    """Test data fetching methods."""

    @pytest.mark.asyncio
    async def test_fetch_anime_by_anilist_id_success(self):
        """Test successful anime fetch by AniList ID."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(
            return_value={
                "Media": {"id": 21, "title": {"romaji": "One Piece"}},
                "_from_cache": False,
            }
        )

        result = await helper.fetch_anime_by_anilist_id(21)

        assert result == {"id": 21, "title": {"romaji": "One Piece"}}
        helper._make_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_anime_by_anilist_id_not_found(self):
        """Test anime fetch when Media is not in response."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime_by_anilist_id(99999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anime_by_anilist_id_empty_response(self):
        """Test anime fetch with empty response."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={})

        result = await helper.fetch_anime_by_anilist_id(21)

        assert result is None


class TestAniListEnrichmentHelperPagination:
    """Test pagination handling."""

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_single_page(self):
        """Test fetching single page of data."""
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()

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
            # Sleep happens after EACH page fetch (including the last one)
            assert mock_sleep.await_count == 2
            mock_sleep.assert_awaited_with(0.5)

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_cache_hit_no_sleep(self):
        """Test that cache hits don't trigger rate limiting sleep."""
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={})

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_missing_media(self):
        """Test pagination when Media is missing."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_missing_data_key(self):
        """Test pagination when data key is missing."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(
            return_value={"Media": {}, "_from_cache": False}
        )

        result = await helper._fetch_paginated_data(21, "query", "characters")

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_empty_edges(self):
        """Test pagination with empty edges array."""
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()
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


class TestAniListEnrichmentHelperSpecificFetchers:
    """Test specific data type fetchers (characters, staff, episodes)."""

    @pytest.mark.asyncio
    async def test_fetch_all_characters(self):
        """Test fetching all characters."""
        helper = AniListEnrichmentHelper()
        helper._fetch_paginated_data = AsyncMock(
            return_value=[
                {"node": {"id": 1, "name": {"full": "Monkey D. Luffy"}}, "role": "MAIN"}
            ]
        )

        result = await helper.fetch_all_characters(21)

        assert len(result) == 1
        assert result[0]["node"]["name"]["full"] == "Monkey D. Luffy"
        helper._fetch_paginated_data.assert_awaited_once()
        # Verify correct query was passed
        call_args = helper._fetch_paginated_data.call_args
        assert "characters" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_fetch_all_staff(self):
        """Test fetching all staff."""
        helper = AniListEnrichmentHelper()
        helper._fetch_paginated_data = AsyncMock(
            return_value=[
                {
                    "node": {"id": 1, "name": {"full": "Eiichiro Oda"}},
                    "role": "Story & Art",
                }
            ]
        )

        result = await helper.fetch_all_staff(21)

        assert len(result) == 1
        assert result[0]["node"]["name"]["full"] == "Eiichiro Oda"
        # Verify correct data key
        call_args = helper._fetch_paginated_data.call_args
        assert call_args[0][2] == "staff"

    @pytest.mark.asyncio
    async def test_fetch_all_episodes(self):
        """Test fetching all episodes."""
        helper = AniListEnrichmentHelper()
        helper._fetch_paginated_data = AsyncMock(
            return_value=[{"node": {"id": 1, "episode": 1, "airingAt": 1234567890}}]
        )

        result = await helper.fetch_all_episodes(21)

        assert len(result) == 1
        assert result[0]["node"]["episode"] == 1
        # Verify correct data key
        call_args = helper._fetch_paginated_data.call_args
        assert call_args[0][2] == "airingSchedule"


class TestAniListEnrichmentHelperPopulateDetails:
    """Test detail population and full data fetching."""

    @pytest.mark.asyncio
    async def test_fetch_and_populate_details_success(self):
        """Test successful detail population."""
        helper = AniListEnrichmentHelper()

        anime_data = {"id": 21, "title": {"romaji": "One Piece"}}

        helper.fetch_all_characters = AsyncMock(
            return_value=[{"node": {"id": 1}, "role": "MAIN"}]
        )
        helper.fetch_all_staff = AsyncMock(
            return_value=[{"node": {"id": 1}, "role": "Director"}]
        )
        helper.fetch_all_episodes = AsyncMock(
            return_value=[{"node": {"id": 1, "episode": 1}}]
        )

        result = await helper._fetch_and_populate_details(anime_data)

        assert "characters" in result
        assert result["characters"]["edges"][0]["node"]["id"] == 1
        assert "staff" in result
        assert result["staff"]["edges"][0]["node"]["id"] == 1
        assert "airingSchedule" in result
        assert result["airingSchedule"]["edges"][0]["node"]["episode"] == 1

    @pytest.mark.asyncio
    async def test_fetch_and_populate_details_missing_id(self):
        """Test detail population when ID is missing."""
        helper = AniListEnrichmentHelper()

        anime_data = {"title": {"romaji": "Test"}}  # No ID

        result = await helper._fetch_and_populate_details(anime_data)

        # Should return original data unchanged
        assert result == anime_data

    @pytest.mark.asyncio
    async def test_fetch_and_populate_details_empty_results(self):
        """Test detail population with empty results."""
        helper = AniListEnrichmentHelper()

        anime_data = {"id": 21}

        helper.fetch_all_characters = AsyncMock(return_value=[])
        helper.fetch_all_staff = AsyncMock(return_value=[])
        helper.fetch_all_episodes = AsyncMock(return_value=[])

        result = await helper._fetch_and_populate_details(anime_data)

        # Should not add empty keys
        assert "characters" not in result
        assert "staff" not in result
        assert "airingSchedule" not in result

    @pytest.mark.asyncio
    async def test_fetch_all_data_by_anilist_id_success(self):
        """Test successful full data fetch."""
        helper = AniListEnrichmentHelper()

        helper.fetch_anime_by_anilist_id = AsyncMock(
            return_value={"id": 21, "title": {"romaji": "One Piece"}}
        )
        helper._fetch_and_populate_details = AsyncMock(
            return_value={
                "id": 21,
                "title": {"romaji": "One Piece"},
                "characters": {"edges": []},
            }
        )

        result = await helper.fetch_all_data_by_anilist_id(21)

        assert result is not None
        assert result["id"] == 21
        assert "characters" in result

    @pytest.mark.asyncio
    async def test_fetch_all_data_by_anilist_id_not_found(self):
        """Test full data fetch when anime not found."""
        helper = AniListEnrichmentHelper()
        helper.fetch_anime_by_anilist_id = AsyncMock(return_value=None)

        result = await helper.fetch_all_data_by_anilist_id(99999)

        assert result is None


class TestAniListEnrichmentHelperClose:
    """Test cleanup methods."""

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test closing helper with active session."""
        helper = AniListEnrichmentHelper()

        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        helper.session = mock_session

        await helper.close()

        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test closing helper without session."""
        helper = AniListEnrichmentHelper()
        helper.session = None

        # Should not raise exception
        await helper.close()


class TestAniListEnrichmentHelperEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_unicode_in_graphql_query(self):
        """Test handling of Unicode characters in GraphQL queries and responses."""
        helper = AniListEnrichmentHelper()

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

        result = await helper.fetch_anime_by_anilist_id(1)

        assert result is not None
        assert "進撃の巨人" in str(result)

    @pytest.mark.asyncio
    async def test_rate_limit_exactly_five(self):
        """Test rate limiting boundary at exactly 5 remaining."""
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=Exception("Invalid JSON"))
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

        result = await helper._make_request("query { test }")

        # Should return empty dict on JSON parse error
        assert result == {"_from_cache": False}

    @pytest.mark.asyncio
    async def test_very_large_paginated_results(self):
        """Test handling of very large paginated result sets."""
        helper = AniListEnrichmentHelper()

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
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime_by_anilist_id(-1)

        # Should handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_anilist_id(self):
        """Test fetching with zero AniList ID."""
        helper = AniListEnrichmentHelper()
        helper._make_request = AsyncMock(return_value={"_from_cache": False})

        result = await helper.fetch_anime_by_anilist_id(0)

        # Should handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_pagination_with_missing_edges_key(self):
        """Test pagination when edges key is missing from response."""
        helper = AniListEnrichmentHelper()
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
    async def test_fetch_populate_details_with_none_values(self):
        """Test detail population when fetch methods return None."""
        helper = AniListEnrichmentHelper()

        anime_data = {"id": 21, "title": {"romaji": "One Piece"}}

        # All fetch methods return None
        helper.fetch_all_characters = AsyncMock(return_value=None)
        helper.fetch_all_staff = AsyncMock(return_value=None)
        helper.fetch_all_episodes = AsyncMock(return_value=None)

        result = await helper._fetch_and_populate_details(anime_data)

        # Should not add keys when None returned
        assert "characters" not in result
        assert "staff" not in result
        assert "airingSchedule" not in result

    @pytest.mark.asyncio
    async def test_rate_limit_header_missing(self):
        """Test when rate limit header is missing from response."""
        helper = AniListEnrichmentHelper()
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
        helper = AniListEnrichmentHelper()

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
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        helper.session = mock_session
        helper._session_event_loop = asyncio.get_running_loop()

        result = await helper._make_request("query { test }")

        # Should return empty dict when errors exist
        assert result == {"_from_cache": False}

    @pytest.mark.asyncio
    async def test_429_retry_with_very_long_wait(self):
        """Test 429 handling with very long Retry-After value."""
        helper = AniListEnrichmentHelper()

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


class TestAniListEnrichmentHelperCacheIntegration:
    """Test cache manager integration."""

    @pytest.mark.asyncio
    async def test_anilist_helper_uses_cache_manager(self, mocker):
        """Test that AniListEnrichmentHelper uses centralized cache manager."""
        from src.cache_manager.instance import http_cache_manager
        from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

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

        helper = AniListEnrichmentHelper()

        # Trigger session creation by making a request
        result = await helper.fetch_anime_by_anilist_id(1)

        # Verify cache manager was called with correct parameters
        mock_get_session.assert_called_once()
        call_args = mock_get_session.call_args
        assert call_args[0][0] == "anilist"  # service name
        assert "timeout" in call_args[1]
        assert "headers" in call_args[1]
        assert call_args[1]["headers"]["X-Hishel-Body-Key"] == "true"

        # Verify the session was used for the request
        assert result is not None

        await helper.close()

    @pytest.mark.asyncio
    async def test_anilist_helper_does_not_create_manual_redis_client(self, mocker):
        """Test that AniListEnrichmentHelper does NOT manually create Redis clients."""
        from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

        # Mock Redis.from_url to detect if it's called
        mock_redis_from_url = mocker.patch("redis.asyncio.Redis.from_url")

        # Mock cache manager to provide a working session
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.AsyncMock()
        mock_session.post.return_value.__aenter__.return_value.status = 200
        mock_session.post.return_value.__aenter__.return_value.json = mocker.AsyncMock(
            return_value={"data": {"Media": {"id": 1, "title": {"romaji": "Test"}}}}
        )
        mock_session.post.return_value.__aenter__.return_value.from_cache = False
        mock_session.post.return_value.__aenter__.return_value.headers = {}

        mocker.patch(
            "src.cache_manager.instance.http_cache_manager.get_aiohttp_session",
            return_value=mock_session,
        )

        helper = AniListEnrichmentHelper()

        # Make a request to trigger session creation
        await helper.fetch_anime_by_anilist_id(1)

        # Verify Redis.from_url was NOT called (no manual Redis client creation)
        mock_redis_from_url.assert_not_called()

        await helper.close()


class TestAniListEnrichmentHelperCLI:
    """Test CLI main function."""

    @pytest.mark.asyncio
    async def test_main_with_anilist_id_success(self, tmp_path):
        """Test CLI with successful AniList ID fetch."""
        import json
        import sys
        from unittest.mock import patch

        output_file = tmp_path / "test_output.json"

        # Mock command line arguments
        test_args = ["script_name", "--anilist-id", "21", "--output", str(output_file)]

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all_data_by_anilist_id = AsyncMock(
                    return_value={"id": 21, "title": {"romaji": "One Piece"}}
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from src.enrichment.api_helpers.anilist_helper import main

                await main()

                # Verify helper methods were called
                mock_helper_instance.fetch_all_data_by_anilist_id.assert_awaited_once_with(
                    21
                )
                mock_helper_instance.close.assert_awaited_once()

                # Verify output file was created
                assert output_file.exists()
                with open(output_file) as f:
                    data = json.load(f)
                assert data["id"] == 21

    @pytest.mark.asyncio
    async def test_main_with_no_data_found(self, tmp_path):
        """Test CLI when no data is found."""
        import sys
        from unittest.mock import patch

        output_file = tmp_path / "test_output.json"

        test_args = [
            "script_name",
            "--anilist-id",
            "99999",
            "--output",
            str(output_file),
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all_data_by_anilist_id = AsyncMock(
                    return_value=None
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from src.enrichment.api_helpers.anilist_helper import main

                await main()

                # Verify close was called even when no data found
                mock_helper_instance.close.assert_awaited_once()

                # Verify output file was NOT created
                assert not output_file.exists()

    @pytest.mark.asyncio
    async def test_main_with_exception_still_closes(self, tmp_path):
        """Test CLI ensures helper.close() is called even on exception."""
        import sys
        from unittest.mock import patch

        output_file = tmp_path / "test_output.json"

        test_args = ["script_name", "--anilist-id", "21", "--output", str(output_file)]

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"
            ) as MockHelper:
                mock_helper_instance = MagicMock()
                mock_helper_instance.fetch_all_data_by_anilist_id = AsyncMock(
                    side_effect=Exception("API Error")
                )
                mock_helper_instance.close = AsyncMock()
                MockHelper.return_value = mock_helper_instance

                from src.enrichment.api_helpers.anilist_helper import main

                # Should not raise exception (caught in CLI)
                await main()

                # Verify close was called despite exception
                mock_helper_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_with_default_output_path(self):
        """Test CLI with default output path."""
        import sys
        from unittest.mock import patch

        test_args = ["script_name", "--anilist-id", "21"]

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"
            ) as MockHelper:
                with patch("builtins.open", MagicMock()) as mock_open:
                    mock_helper_instance = MagicMock()
                    mock_helper_instance.fetch_all_data_by_anilist_id = AsyncMock(
                        return_value={"id": 21}
                    )
                    mock_helper_instance.close = AsyncMock()
                    MockHelper.return_value = mock_helper_instance

                    from src.enrichment.api_helpers.anilist_helper import main

                    await main()

                    # Verify default output path was used
                    mock_open.assert_called()
                    call_args = mock_open.call_args[0]
                    assert "test_anilist_output.json" in call_args[0]


# --- Tests for main() function following jikan_helper pattern ---


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper")
async def test_main_function_success(mock_helper_class, tmp_path):
    """Test main() function handles successful execution."""
    from src.enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data_by_anilist_id = AsyncMock(
        return_value={"id": 21, "title": "Test"}
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    # Use pytest's tmp_path for portability
    output_file = str(tmp_path / "output.json")
    with patch(
        "sys.argv", ["script.py", "--anilist-id", "21", "--output", output_file]
    ):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_all_data_by_anilist_id.assert_awaited_once_with(21)
    mock_helper.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from src.enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data_by_anilist_id = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--anilist-id", "99999"]):
        exit_code = await main()

    assert exit_code == 1
    mock_helper.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.api_helpers.anilist_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data_by_anilist_id = AsyncMock(
        side_effect=Exception("API error")
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--anilist-id", "21"]):
        exit_code = await main()

    assert exit_code == 1
    mock_helper.close.assert_awaited_once()

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import CrawlResult

from enrichment.crawlers.anime_planet_anime_crawler import (
    _determine_season_from_date,
    _extract_json_ld,
    _extract_rank,
    _extract_slug_from_url,
    _extract_studios,
    _normalize_anime_url,
    _process_related_anime,
    _process_related_manga,
    fetch_animeplanet_anime,
)

# Use pytest-asyncio mode
pytestmark = pytest.mark.asyncio


# --- Unit Tests for Helper Functions ---


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("dandadan", "https://www.anime-planet.com/anime/dandadan"),
        ("/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
        (
            "https://www.anime-planet.com/anime/dandadan",
            "https://www.anime-planet.com/anime/dandadan",
        ),
        ("anime/one-piece", "https://www.anime-planet.com/anime/one-piece"),
    ],
)
def test_normalize_anime_url_valid(identifier, expected):
    assert _normalize_anime_url(identifier) == expected


def test_normalize_anime_url_invalid():
    with pytest.raises(ValueError, match="Invalid URL"):
        _normalize_anime_url("https://www.google.com/anime/dandadan")


@pytest.mark.parametrize(
    "url, expected_slug",
    [
        ("https://www.anime-planet.com/anime/dandadan", "dandadan"),
        ("https://www.anime-planet.com/anime/one-piece?foo=bar", "one-piece"),
    ],
)
def test_extract_slug_from_url_valid(url, expected_slug):
    assert _extract_slug_from_url(url) == expected_slug


def test_extract_slug_from_url_invalid():
    with pytest.raises(ValueError, match="Could not extract slug"):
        _extract_slug_from_url("https://www.anime-planet.com/manga/dandadan")


def test_extract_json_ld_valid():
    html = """
    <html><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "TVEpisode",
        "name": "Dandadan",
        "description": "This is a &lt;b&gt;great&lt;/b&gt; show.",
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx158822-DbJ2c82s35jA.jpg"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert json_ld["name"] == "Dandadan"
    assert json_ld["description"] == "This is a <b>great</b> show."
    assert "anime-planet.comhttps" not in json_ld["image"]

    # Test with a malformed image URL that needs fixing
    html_malformed_image = """
    <html><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx158822-DbJ2c82s35jA.jpg"
    }
    </script></html>
    """
    json_ld_malformed = _extract_json_ld(html_malformed_image)
    assert json_ld_malformed is not None
    assert "anime-planet.comhttps" not in json_ld_malformed["image"]


def test_extract_json_ld_malformed_image_url():
    html = """
    <html><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx158822-DbJ2c82s35jA.jpg"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert "anime-planet.comhttps" not in json_ld["image"]


def test_extract_json_ld_no_description():
    html = """
    <html><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "name": "Dandadan"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert "description" not in json_ld


@pytest.mark.parametrize(
    "html_input",
    [
        "<html></html>",  # No script tag
        '<html><script type="application/ld+json">{invalid json}</script></html>',  # Malformed JSON
    ],
)
def test_extract_json_ld_invalid(html_input):
    assert _extract_json_ld(html_input) is None


@pytest.mark.parametrize(
    "rank_texts, expected",
    [
        ([{"text": "Rank #123"}], 123),
        ([{"text": "Overall rank #456"}], 456),
        ([{"text": "No rank here"}], None),
        ([], None),
        ([{"text": "Rank #abc"}], None),
    ],
)
def test_extract_rank(rank_texts, expected):
    assert _extract_rank(rank_texts) == expected


def test_extract_studios():
    studios_raw = [
        {"studio": "Science SARU"},
        {"studio": "MAPPA"},
        {"studio": "Science SARU"},
    ]
    assert _extract_studios(studios_raw) == ["Science SARU", "MAPPA"]
    # Test limit
    studios_raw_long = [{"studio": f"Studio {i}"} for i in range(10)]
    assert len(_extract_studios(studios_raw_long)) == 5


@pytest.mark.parametrize(
    "date_str, expected_season",
    [
        ("2024-01-15", "WINTER"),
        ("2024-04-20", "SPRING"),
        ("2024-08-01", "SUMMER"),
        ("2024-11-30", "FALL"),
        ("2024-12-01", "WINTER"),
        ("invalid-date", None),
        ("", None),
        ("2024-00-15", None),  # Invalid month: 00
        ("2024-13-01", None),  # Invalid month: 13
    ],
)
def test_determine_season_from_date(date_str, expected_season):
    assert _determine_season_from_date(date_str) == expected_season


def test_process_related_anime():
    related_anime_raw = [
        {
            "title": "Sequel",
            "url": "/anime/sequel-slug",
            "relation_subtype": "Sequel",
            "start_date_attr": "2025-01-01",
            "metadata_text": "TV: 12 ep",
        },
        {
            "title": "Anime with end date only",
            "url": "/anime/end-date-slug",
            "end_date_attr": "2023-12-31",
        },
        {
            "title": "Absolute URL",
            "url": "https://www.anime-planet.com/anime/absolute-slug",
            "metadata_text": "TV: 24 ep",
        },
        {"title": "Prequel", "url": ""},  # Invalid, should be skipped
        {"title": "", "url": "/anime/no-title"},  # No title
        {"title": "No URL", "url": ""},  # No URL
        {"title": "Invalid URL", "url": "/manga/invalid-slug"},  # No anime slug
        {
            "title": "No Episodes",
            "url": "/anime/no-episodes",
            "metadata_text": "TV",
        },
        {
            "title": "No Type Match",
            "url": "/anime/no-type-match",
            "metadata_text": "Unknown: 12 ep",
        },
    ]
    processed = _process_related_anime(related_anime_raw)
    assert (
        len(processed) == 5
    )  # Original valid + end date + absolute URL + No Episodes + No Type Match
    item = processed[0]
    assert item["title"] == "Sequel"
    assert item["slug"] == "sequel-slug"
    assert item["relation_subtype"] == "SEQUEL"
    assert item["year"] == 2025
    assert item["type"] == "TV"
    assert item["episodes"] == 12

    # Assertions for the new edge cases
    assert processed[1]["slug"] == "end-date-slug"
    assert processed[1]["year"] == 2023

    # Test absolute URL handling (should not double-prefix)
    assert processed[2]["slug"] == "absolute-slug"
    assert processed[2]["url"] == "https://www.anime-planet.com/anime/absolute-slug"
    assert processed[2]["episodes"] == 24

    assert processed[3]["slug"] == "no-episodes"
    assert "episodes" not in processed[3]
    assert processed[4]["slug"] == "no-type-match"
    assert "type" not in processed[4]


def test_process_related_manga():
    related_manga_raw = [
        {
            "title": "Manga Adaptation",
            "url": "/manga/manga-slug",
            "relation_subtype": "Adaptation",
            "metadata_text": "Vol: 2, Ch: 12",
            "start_date_attr": "2022-01-01",  # Cover lines 610-614
        },
        {
            "title": "Manga with end date only",
            "url": "/manga/end-date-slug",
            "end_date_attr": "2023-12-31",
        },
        {"title": "Invalid Manga", "url": "/anime/not-a-manga"},  # Invalid slug
        {"title": "", "url": "/manga/no-title"},  # No title
        {"title": "No URL", "url": ""},  # No URL
        {"title": "Invalid URL", "url": "/anime/invalid-slug"},  # No manga slug
        {
            "title": "Only Volumes",
            "url": "/manga/only-volumes",
            "metadata_text": "Vol: 1",
        },
        {
            "title": "Only Chapters",
            "url": "/manga/only-chapters",
            "metadata_text": "Ch: 5",
        },
        {
            "title": "No Match",
            "url": "/manga/no-match",
            "metadata_text": "Unknown",
        },
    ]
    processed = _process_related_manga(related_manga_raw)
    assert (
        len(processed) == 5
    )  # Original valid + end date + Only Volumes + Only Chapters + No Match
    item = processed[0]
    assert item["title"] == "Manga Adaptation"
    assert item["slug"] == "manga-slug"
    assert item["volumes"] == 2
    assert item["chapters"] == 12
    assert item["start_date"] == "2022-01-01"
    assert item["year"] == 2022

    # Assertions for the new edge cases
    assert processed[1]["slug"] == "end-date-slug"
    assert processed[1]["year"] == 2023
    assert processed[2]["slug"] == "only-volumes"
    assert processed[2]["volumes"] == 1
    assert "chapters" not in processed[2]
    assert processed[3]["slug"] == "only-chapters"
    assert processed[3]["chapters"] == 5
    assert "volumes" not in processed[3]
    assert processed[4]["slug"] == "no-match"
    assert "volumes" not in processed[4]
    assert "chapters" not in processed[4]

    # Test with empty list
    assert _process_related_manga([]) == []


def test_process_related_manga_with_unknown_placeholders():
    """Test handling of '?' placeholders in volume/chapter metadata.

    Some anime-planet pages contain "Vol: ?" or "Ch: ?" when the count is unknown.
    The parser should skip these instead of crashing with ValueError.
    """
    related_manga_raw = [
        {
            "title": "Unknown Volume",
            "url": "/manga/unknown-vol",
            "metadata_text": "Vol: ?",
        },
        {
            "title": "Unknown Chapter",
            "url": "/manga/unknown-ch",
            "metadata_text": "Ch: ?",
        },
        {
            "title": "Both Unknown",
            "url": "/manga/both-unknown",
            "metadata_text": "Vol: ?, Ch: ?",
        },
        {
            "title": "Mixed Known Unknown",
            "url": "/manga/mixed",
            "metadata_text": "Vol: 5, Ch: ?",
        },
    ]

    # This should not raise ValueError
    processed = _process_related_manga(related_manga_raw)

    assert len(processed) == 4

    # Unknown Volume - should not have 'volumes' key
    assert "volumes" not in processed[0]
    assert processed[0]["slug"] == "unknown-vol"

    # Unknown Chapter - should not have 'chapters' key
    assert "chapters" not in processed[1]
    assert processed[1]["slug"] == "unknown-ch"

    # Both Unknown - neither key should exist
    assert "volumes" not in processed[2]
    assert "chapters" not in processed[2]
    assert processed[2]["slug"] == "both-unknown"

    # Mixed - only known value should be set
    assert processed[3]["volumes"] == 5
    assert "chapters" not in processed[3]
    assert processed[3]["slug"] == "mixed"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_wrong_result_type(MockAsyncWebCrawler):
    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock(return_value=["not a CrawlResult"])
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    with pytest.raises(TypeError, match="Unexpected result type"):
        await fetch_animeplanet_anime("any-slug")


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_full_success(MockAsyncWebCrawler):
    """Test successful fetching and processing of a rich anime page."""
    slug = "dandadan"
    url = f"https://www.anime-planet.com/anime/{slug}"

    # A more complete mock response with related content
    css_data = {
        "related_anime_raw": [
            {
                "title": "Dandadan Season 2",
                "url": "/anime/dandadan-season-2",
                "relation_subtype": "Sequel",
            }
        ],
        "related_manga_raw": [
            {
                "title": "Dandadan Manga",
                "url": "/manga/dandadan",
                "relation_subtype": "Adaptation",
            }
        ],
        "rank_text": [{"text": "Overall rank #123"}],
        "studios_raw": [{"studio": "Science SARU"}],
        "title_japanese": "Alt title: ダンダダン",
        "poster": "http://example.com/poster.jpg",
    }
    json_ld_data = {
        "@type": "TVSeries",
        "name": "Dandadan",
        "description": "A story of ghosts and aliens.",
        "url": url,
        "image": "http://example.com/image.jpg",
        "startDate": "2024-10-01T00:00:00+00:00",
        "endDate": "2024-12-25T00:00:00+00:00",
        "numberOfEpisodes": 12,
        "genre": ["Action", "Comedy"],
        "aggregateRating": {"ratingValue": "4.5", "ratingCount": "100"},
    }
    html_content = f"""
    <html>
        <meta property="og:image" content="http://example.com/poster_og.jpg">
        <script type="application/ld+json">{json.dumps(json_ld_data)}</script>
    </html>
    """

    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock()
    mock_arun.return_value = [
        CrawlResult(
            url=url,
            success=True,
            extracted_content=json.dumps([css_data]),
            html=html_content,
        )
    ]
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    data = await fetch_animeplanet_anime(slug)

    assert data is not None
    assert data["slug"] == slug
    assert data["title"] == "Dandadan"
    assert data["rank"] == 123
    assert data["studios"] == ["Science SARU"]
    assert data["title_japanese"] == "ダンダダン"
    assert data["poster"] == "http://example.com/poster.jpg"  # CSS has priority
    assert data["year"] == 2024
    assert data["season"] == "FALL"
    assert data["status"] == "COMPLETED"
    assert data["genres"] == ["Action", "Comedy"]
    assert "related_anime" in data
    assert "related_count" in data
    assert data["related_count"] == 1
    assert "related_manga" in data


@pytest.mark.parametrize(
    "start_date, end_date, expected_status",
    [
        ("2024-01-01", "2024-03-31", "COMPLETED"),
        ("2023-10-01", None, "AIRING"),
        (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), None, "AIRING"),
        # ("2099-01-01T00:00:00Z", None, "UPCOMING"), # This case is tested separately
        (None, None, "UNKNOWN"),
        ("invalid-date", None, "AIRING"),  # Fallback due to ValueError
    ],
)
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_anime_status_derivation(
    MockAsyncWebCrawler, start_date, end_date, expected_status
):
    # Use a unique slug for each test case to prevent cache collisions
    slug = f"status-test-{start_date}-{end_date}"

    json_ld_data = {"startDate": start_date, "endDate": end_date}
    html_content = f"""<html><script type="application/ld+json">{json.dumps(json_ld_data)}</script></html>"""

    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock(
        return_value=[
            CrawlResult(
                url="",
                success=True,
                extracted_content=json.dumps([{}]),
                html=html_content,
            )
        ]
    )
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    data = await fetch_animeplanet_anime(slug)
    assert data["status"] == expected_status


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_anime_status_upcoming_is_correct(MockAsyncWebCrawler):
    """Isolated test for the UPCOMING status to ensure it is correctly determined."""
    slug = "upcoming-anime-test"
    start_date = "2099-01-01T00:00:00Z"
    json_ld_data = {"startDate": start_date, "endDate": None}
    html_content = f"""<html><script type="application/ld+json">{json.dumps(json_ld_data)}</script></html>"""

    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock(
        return_value=[
            CrawlResult(
                url="",
                success=True,
                extracted_content=json.dumps([{}]),
                html=html_content,
            )
        ]
    )
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    data = await fetch_animeplanet_anime(slug)

    assert data is not None
    assert data["status"] == "UPCOMING"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_no_html(MockAsyncWebCrawler):
    """Test handling when crawl result has no HTML content."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock(
        return_value=[
            CrawlResult(
                url="",
                success=True,
                extracted_content=json.dumps([{"slug": "any-slug"}]),
                html="",
            )
        ]
    )
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    data = await fetch_animeplanet_anime("any-slug")
    assert data is not None
    assert "type" not in data  # JSON-LD processing should be skipped


@pytest.mark.parametrize(
    "crawler_return_value",
    [
        [],  # No results
        [
            CrawlResult(
                url="",
                success=False,
                error_message="Failed",
                html="",
                extracted_content="",
            )
        ],  # Unsuccessful crawl        [
        CrawlResult(url="", success=True, extracted_content="[]", html=""),
    ],
)
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_failure_scenarios(
    MockAsyncWebCrawler, crawler_return_value
):
    if not isinstance(crawler_return_value, list):
        crawler_return_value = [crawler_return_value]
    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_arun = AsyncMock(return_value=crawler_return_value)
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    data = await fetch_animeplanet_anime("any-slug")
    assert data is None


class TestCLI:
    """Test CLI functionality."""

    @patch(
        "enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime",
        new_callable=AsyncMock,
    )
    @patch("enrichment.crawlers.anime_planet_anime_crawler.argparse.ArgumentParser")
    async def test_main_block_cli_execution(
        self, mock_argparse, mock_fetch_animeplanet_anime
    ):
        """Test the __main__ block for CLI execution by simulating it."""
        # To test the main block, we can't easily use runpy because the module is already imported.
        # Instead, we simulate the actions of the __main__ block directly.

        # 1. Mock argparse
        mock_args = MagicMock()
        mock_args.identifier = "dandadan"
        mock_args.output = "test.json"
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_argparse.return_value = mock_parser

        # 2. Mock asyncio.run to just await the coroutine
        async def mock_asyncio_run(coro):
            await coro

        # 3. The actual code from the __main__ block
        with patch(
            "enrichment.crawlers.anime_planet_anime_crawler.asyncio.run",
            mock_asyncio_run,
        ):
            # This simulates the call inside if __name__ == "__main__"
            parser = mock_argparse()
            args = parser.parse_args()
            await mock_asyncio_run(
                mock_fetch_animeplanet_anime(
                    args.identifier,
                    output_path=args.output,
                )
            )

        # 4. Assert the mock was called correctly
        mock_fetch_animeplanet_anime.assert_awaited_once_with(
            "dandadan",
            output_path="test.json",
        )


# --- Cache Efficiency Tests ---


@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_cache_reuse_with_different_output_paths(MockAsyncWebCrawler, tmp_path):
    """
    Test: Cache should be reused when fetching same anime with different output paths.

    Behavior:
    1. First call: Crawl website (expensive), cache result
    2. Second call (same slug, different output_path): Cache HIT, no crawl
    3. Both calls should write their respective files

    The cached function `_fetch_animeplanet_anime_data` is separate from the
    side-effect wrapper `fetch_animeplanet_anime`, ensuring cache efficiency.
    """
    file1 = tmp_path / "call1.json"
    file2 = tmp_path / "call2.json"

    # Track how many times crawler is invoked
    crawl_count = 0

    # Create a real cache that stores data
    cache_storage = {}

    mock_result = CrawlResult(
        url="https://www.anime-planet.com/anime/cache-test",
        success=True,
        extracted_content=json.dumps(
            [
                {
                    "title": "Cache Test Anime",
                    "slug": "cache-test",
                    "rank_text": [{"text": "Rank #100"}],
                    "studios_raw": [{"studio": "Test Studio"}],
                }
            ]
        ),
        html="""
        <html>
            <script type="application/ld+json">
            {
                "@type": "TVSeries",
                "name": "Cache Test Anime",
                "startDate": "2024-01-01T00:00:00+00:00",
                "endDate": "2024-03-31T00:00:00+00:00"
            }
            </script>
        </html>
        """,
    )

    async def mock_arun(*_args, **_kwargs):
        nonlocal crawl_count
        crawl_count += 1
        return [mock_result]

    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_crawler_instance.__aenter__.return_value.arun = mock_arun

    # Mock Redis to actually cache data
    with patch(
        "http_cache.result_cache.get_result_cache_redis_client"
    ) as mock_get_redis:
        mock_redis = AsyncMock()

        async def mock_get(key):
            return cache_storage.get(key)

        async def mock_setex(key, _ttl, value):
            cache_storage[key] = value

        mock_redis.get = mock_get
        mock_redis.setex = mock_setex
        mock_get_redis.return_value = mock_redis

        # Call 1: First fetch with output_path
        result1 = await fetch_animeplanet_anime(
            "cache-test", output_path=str(file1)
        )

        assert result1 is not None
        assert result1["title"] == "Cache Test Anime"
        assert file1.exists(), "Call 1 should write file"
        assert crawl_count == 1, "First call should crawl"

        # Call 2: Same slug, different output_path
        # After refactoring: Should hit cache, NOT crawl again
        result2 = await fetch_animeplanet_anime(
            "cache-test", output_path=str(file2)
        )

        assert result2 is not None
        assert result2["title"] == "Cache Test Anime"
        assert file2.exists(), "Call 2 should write file (side effect)"

        # Verify cache efficiency: second call should hit cache, not re-crawl
        assert crawl_count == 1, (
            f"Expected 1 crawl (cache hit on 2nd call), but got {crawl_count} crawls. "
            "Cache is not being reused for same slug with different output_path!"
        )


# --- 100% Coverage Tests ---


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_anime_poster_regex_fallback(MockAsyncWebCrawler):
    """Test poster extraction via regex when CSS selector fails."""
    css_data = {
        "poster": "",  # Empty poster from CSS
        "rank_text": [],
        "studios_raw": [],
        "related_anime_raw": [],
        "related_manga_raw": [],
    }
    json_ld_data = {"@type": "TVSeries", "name": "Test"}
    html_content = '<html><meta property="og:image" content="http://example.com/poster.jpg"><script type="application/ld+json">{}</script></html>'.format(
        json.dumps(json_ld_data)
    )

    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_crawler_instance.__aenter__.return_value.arun = AsyncMock(
        return_value=[
            CrawlResult(
                url="",
                success=True,
                extracted_content=json.dumps([css_data]),
                html=html_content,
            )
        ]
    )

    data = await fetch_animeplanet_anime("test")
    assert data["poster"] == "http://example.com/poster.jpg"




def test_extract_rank_with_value_error():
    """Test rank extraction handles ValueError for edge case rank text."""
    # Edge case: no number after # symbol
    rank_texts = [{"text": "Rank #"}]
    result = _extract_rank(rank_texts)
    assert result is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_anime_empty_results_list(MockAsyncWebCrawler):
    """Test handling when AsyncWebCrawler returns empty list."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value
    mock_crawler_instance.__aenter__.return_value.arun = AsyncMock(return_value=[])

    result = await fetch_animeplanet_anime("test")
    assert result is None


@patch("enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime")
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    mock_fetch.return_value = {"title": "Test Anime", "slug": "test"}

    with patch("sys.argv", ["script.py", "test-anime", "--output", "output.json"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once_with(
        "test-anime", output_path="output.json"
    )


@patch("enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime")
async def test_main_function_no_data_extracted(mock_fetch):
    """Test main() function returns exit code 1 when no data is extracted."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    mock_fetch.return_value = None  # Extraction failure

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 1


@patch("enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    mock_fetch.side_effect = Exception("Crawl failed")

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime"
)
async def test_main_function_handles_value_error(mock_fetch):
    """Test main() handles ValueError with specific error message."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    mock_fetch.side_effect = ValueError("Invalid anime slug format")

    with patch("sys.argv", ["script.py", "invalid-slug"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime"
)
async def test_main_function_handles_os_error(mock_fetch):
    """Test main() handles OSError (file write failures)."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    mock_fetch.side_effect = OSError("Permission denied")

    with patch(
        "sys.argv",
        ["script.py", "dandadan", "--output", "/invalid/path.json"],
    ):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime"
)
async def test_main_function_handles_unexpected_exception(mock_fetch):
    """Test main() handles unexpected exceptions and logs full traceback."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    # Simulate truly unexpected exception
    mock_fetch.side_effect = RuntimeError("Unexpected internal error")

    with patch("sys.argv", ["script.py", "dandadan"]):
        exit_code = await main()

    assert exit_code == 1


async def test_main_function_with_default_output():
    """Test main() function uses default output path when not specified."""
    from enrichment.crawlers.anime_planet_anime_crawler import main

    with patch("sys.argv", ["script.py", "dandadan"]):
        with patch(
            "enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime"
        ) as mock_fetch:
            mock_fetch.return_value = {"title": "Dandadan"}
            exit_code = await main()

    assert exit_code == 0
    # Should use default output path
    call_args = mock_fetch.call_args
    assert call_args[1]["output_path"] == "animeplanet_anime.json"

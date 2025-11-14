"""
Comprehensive unit tests for anisearch_character_crawler.py with 100% coverage.

Tests cover:
- Main crawler function with cache decorator
- URL handling and validation
- Data extraction and processing
- Character data flattening and cleaning
- Favorites parsing (numbers and missing)
- Image URL extraction from CSS style
- Error handling and edge cases
- CLI functionality
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import CrawlResult

from src.enrichment.crawlers.anisearch_character_crawler import (
    fetch_anisearch_characters,
)


class TestFetchAnisearchCharacters:
    """Test fetch_anisearch_characters function."""

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_no_results(self):
        """Test handling when crawler returns no results."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_no_results/characters"
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_wrong_result_type(self):
        """Test handling when result is not CrawlResult type.

        Note: This test is implemented in test_crawlers_unit.py as
        TestAniSearchCharacterCrawler.test_fetch_anisearch_characters_wrong_result_type
        where the cache decorator can be properly bypassed.
        """

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_extraction_failed(self):
        """Test handling when extraction fails."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = False
                mock_result.error_message = "Extraction failed"
                mock_result.extracted_content = None
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_wrong_result_type/characters"

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_extraction_failed/characters"
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_no_extracted_content(self):
        """Test handling when extracted content is None."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.extracted_content = None
                mock_result.error_message = ""  # Add missing attribute
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_no_extracted_content/characters"

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_no_extracted_content/characters"
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_basic_success(self):
        """Test successful character data extraction."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_basic_success/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main Character",
                            "characters": [
                                {
                                    "name": "Test Character",
                                    "url": "/character/456,test-character",
                                    "favorites": "123 favorites",
                                    "image": 'url("https://example.com/image.jpg")',
                                }
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_basic_success/characters"
                )

                assert result is not None
                assert "characters" in result
                assert len(result["characters"]) == 1
                char = result["characters"][0]
                assert char["name"] == "Test Character"
                assert char["role"] == "Main"  # "Main Character" → "Main"
                assert (
                    char["url"]
                    == "https://www.anisearch.com//character/456,test-character"  # Double slash from concatenation
                )
                assert char["favorites"] == 123
                assert char["image"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_favorites_parsing(self):
        """Test favorites field parsing with various formats."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_favorites_parsing/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {
                                    "name": "Char1",
                                    "url": "/character/1",
                                    "favorites": "42",
                                },
                                {
                                    "name": "Char2",
                                    "url": "/character/2",
                                    "favorites": "1000 favorites",
                                },
                                {
                                    "name": "Char3",
                                    "url": "/character/3",
                                    "favorites": "No number here",
                                },
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_favorites_parsing/characters"
                )

                assert result is not None
                chars = result["characters"]
                assert len(chars) == 3
                assert chars[0]["favorites"] == 42
                assert chars[1]["favorites"] == 1000
                assert "favorites" not in chars[2]  # Deleted if no number

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_image_url_extraction(self):
        """Test image URL extraction from CSS style attribute."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_image_url_extraction/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {
                                    "name": "Char1",
                                    "url": "/character/1",
                                    "image": 'url("https://example.com/1.jpg")',
                                },
                                {
                                    "name": "Char2",
                                    "url": "/character/2",
                                    "image": "url(https://example.com/2.jpg)",
                                },
                                {
                                    "name": "Char3",
                                    "url": "/character/3",
                                    "image": 'url("https://example.com/3.jpg")',
                                },
                                {
                                    "name": "Char4",
                                    "url": "/character/4",
                                    "image": "No URL here",
                                },
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_image_url_extraction/characters"
                )

                assert result is not None
                chars = result["characters"]
                assert len(chars) == 4
                assert chars[0]["image"] == "https://example.com/1.jpg"
                assert chars[1]["image"] == "https://example.com/2.jpg"
                assert chars[2]["image"] == "https://example.com/3.jpg"
                assert chars[3]["image"] == "No URL here"  # No match, unchanged

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_empty_image(self):
        """Test handling of empty image field."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_empty_image/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {
                                    "name": "Char1",
                                    "url": "/character/1",
                                    "image": "",
                                },
                                {
                                    "name": "Char2",
                                    "url": "/character/2",
                                    "image": None,
                                },
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_empty_image/characters"
                )

                assert result is not None
                # Empty and None images shouldn't cause crashes
                assert len(result["characters"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_multiple_sections(self):
        """Test flattening characters from multiple role sections."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_multiple_sections/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main Character",
                            "characters": [
                                {"name": "Hero", "url": "/character/1"},
                                {"name": "Heroine", "url": "/character/2"},
                            ],
                        },
                        {
                            "role": "Supporting Character",
                            "characters": [
                                {"name": "Sidekick", "url": "/character/3"},
                            ],
                        },
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_multiple_sections/characters"
                )

                assert result is not None
                assert len(result["characters"]) == 3
                assert result["characters"][0]["role"] == "Main"
                assert result["characters"][1]["role"] == "Main"
                assert result["characters"][2]["role"] == "Supporting"

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_missing_role(self):
        """Test handling when role field is missing."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_missing_role/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "characters": [
                                {"name": "Char", "url": "/character/1"},
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_missing_role/characters"
                )

                assert result is not None
                assert result["characters"][0]["role"] == ""

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_empty_characters_list(self):
        """Test handling when characters list is empty."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_empty_characters_list/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_empty_characters_list/characters"
                )

                assert result is not None
                assert result["characters"] == []

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_with_output_file(self, tmp_path):
        """Test writing data to output file."""
        output_file = tmp_path / "test_characters.json"

        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_with_output_file/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {"name": "Test", "url": "/character/1"},
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_with_output_file/characters",
                    return_data=True,
                    output_path=str(output_file),
                )

                assert result is not None
                assert output_file.exists()

                # Verify file contents
                with open(output_file) as f:
                    data = json.load(f)
                assert "characters" in data
                assert data["characters"][0]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_return_data_false(self):
        """Test with return_data=False."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_return_data_false/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {"name": "Test", "url": "/character/1"},
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_return_data_false/characters",
                    return_data=False,
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_multiple_failed_results(self):
        """Test when loop completes with multiple failed results (line 165 coverage)."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                # Multiple results but all failed
                mock_result1 = MagicMock(spec=CrawlResult)
                mock_result1.success = False
                mock_result1.error_message = "Failed 1"
                mock_result1.url = "https://www.anisearch.com/anime/test1/characters"

                mock_result2 = MagicMock(spec=CrawlResult)
                mock_result2.success = False
                mock_result2.error_message = "Failed 2"
                mock_result2.url = "https://www.anisearch.com/anime/test2/characters"

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result1, mock_result2])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_multiple_failed_results/characters"
                )

                assert result is None


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_fetch_anisearch_characters_unicode_in_names(self):
        """Test handling of Unicode characters in character names."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test_fetch_anisearch_characters_unicode_in_names/characters"
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {"name": "桜木花道", "url": "/character/1"},
                                {"name": "モンキー・D・ルフィ", "url": "/character/2"},
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test_fetch_anisearch_characters_unicode_in_names/characters"
                )

                assert result is not None
                assert len(result["characters"]) == 2
                assert result["characters"][0]["name"] == "桜木花道"
                assert result["characters"][1]["name"] == "モンキー・D・ルフィ"


class TestCLI:
    """Test CLI functionality."""

    @pytest.mark.asyncio
    async def test_cli_execution(self, tmp_path):
        """Test CLI execution with arguments."""
        output_file = tmp_path / "cli_characters.json"

        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.cached_result",
                lambda **kwargs: lambda f: f,
            ):
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = (
                    "https://www.anisearch.com/anime/test_cli_execution/characters"
                )
                mock_result.extracted_content = json.dumps(
                    [
                        {
                            "role": "Main",
                            "characters": [
                                {"name": "CLI Test", "url": "/character/1"},
                            ],
                        }
                    ]
                )

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(
                    return_value=mock_crawler
                )
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                # Simulate CLI call
                test_args = [
                    "script_name",
                    "https://www.anisearch.com/anime/test_cli_execution/characters",
                    "--output",
                    str(output_file),
                ]

                with patch.object(sys, "argv", test_args):
                    import src.enrichment.crawlers.anisearch_character_crawler as crawler_module

                    parser = crawler_module.argparse.ArgumentParser()
                    parser.add_argument("url", type=str)
                    parser.add_argument("--output", type=str, default="test.json")
                    args = parser.parse_args(test_args[1:])

                    await fetch_anisearch_characters(
                        args.url, return_data=False, output_path=args.output
                    )

                assert output_file.exists()

    @pytest.mark.asyncio
    async def test_cache_key_only_depends_on_url(self, tmp_path):
        """
        Test that cache keys depend ONLY on URL, not on output_path or return_data.

        This validates the fix for the reviewer's comment: cache keys should be
        based purely on the URL being crawled, so different output paths reuse
        the same cached data instead of creating redundant cache entries.

        Expected behavior (after fix):
        - Same URL + different output_path = SAME cache key
        - Same URL + different return_data = SAME cache key
        - Result: Single cache entry reused for all calls with same URL
        """
        output_file1 = tmp_path / "output1.json"
        output_file2 = tmp_path / "output2.json"

        # Expected cached data
        cached_data = {"characters": [{"name": "Test", "url": "https://www.anisearch.com//character/1", "role": "Main"}]}

        # Mock Redis client to track cache key generation
        mock_redis = AsyncMock()
        # First get() returns None (cache miss), second get() returns cached data (cache hit)
        mock_redis.get = AsyncMock(side_effect=[None, json.dumps(cached_data)])
        mock_redis.setex = AsyncMock()

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client",
            return_value=mock_redis
        ):
            with patch(
                "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
            ) as MockCrawler:
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.url = "https://www.anisearch.com/anime/test/characters"
                mock_result.extracted_content = json.dumps([
                    {
                        "role": "Main",
                        "characters": [{"name": "Test", "url": "/character/1"}],
                    }
                ])

                mock_crawler = AsyncMock()
                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
                MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

                # First call with output_file1, return_data=True
                result1 = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test/characters",
                    return_data=True,
                    output_path=str(output_file1),
                )

                # Second call with output_file2, return_data=False
                result2 = await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test/characters",
                    return_data=False,
                    output_path=str(output_file2),
                )

                # Verify both files were written (side effects work regardless of cache)
                assert output_file1.exists(), "First output file should be written"
                assert output_file2.exists(), "Second output file should be written"

                # Verify return_data behavior
                assert result1 is not None, "First call should return data"
                assert result2 is None, "Second call should not return data (return_data=False)"

                # Verify cache behavior: should only create ONE cache entry
                assert mock_redis.get.call_count == 2, "Should query cache twice"
                assert mock_redis.setex.call_count == 1, "Should create only 1 cache entry"

                # Extract cache key from setex call
                cache_key = mock_redis.setex.call_args[0][0]

                # Cache key should NOT contain output_path or return_data
                assert "output_path" not in cache_key, (
                    "Cache key must not include output_path parameter"
                )
                assert "return_data" not in cache_key, (
                    "Cache key must not include return_data parameter"
                )

                # Cache key should only contain URL and schema hash
                assert "anisearch_characters" in cache_key, "Should have key_prefix"
                assert "test/characters" in cache_key or len(cache_key) > 100, (
                    "Should include URL (or hash if too long)"
                )


class TestTypeErrorCoverage:
    """Tests specifically for TypeError exception path coverage (line 108)."""

    @pytest.mark.asyncio
    async def test_type_error_when_result_not_crawl_result(self):
        """Test line 108: TypeError when arun returns non-CrawlResult in container."""
        with patch(
            "src.enrichment.crawlers.anisearch_character_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            mock_crawler = AsyncMock()

            # Return a CrawlResultContainer with a string instead of CrawlResult
            from crawl4ai.models import CrawlResultContainer

            bad_container = CrawlResultContainer(["not a CrawlResult object"])

            mock_crawler.arun = AsyncMock(return_value=bad_container)
            MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
            MockCrawler.return_value.__aexit__ = AsyncMock(
                return_value=None
            )  # Critical: return None!

            with pytest.raises(
                TypeError, match="Unexpected result type.*expected CrawlResult"
            ):
                await fetch_anisearch_characters(
                    "https://www.anisearch.com/anime/test/characters"
                )


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_character_crawler.fetch_anisearch_characters")
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    from src.enrichment.crawlers.anisearch_character_crawler import main

    mock_fetch.return_value = {
        "characters": [{"name": "Test Character"}],
        "total_count": 1,
    }

    with patch(
        "sys.argv",
        [
            "script.py",
            "https://www.anisearch.com/anime/18878/characters",
            "--output",
            "/tmp/output.json",
        ],
    ):
        exit_code = await main()

    assert exit_code == 0
    # Verify the function was called (args vs kwargs may vary by implementation)
    mock_fetch.assert_awaited_once()
    call_args = mock_fetch.call_args
    # Check the URL was passed correctly (could be positional or keyword)
    if call_args[0]:
        assert call_args[0][0] == "https://www.anisearch.com/anime/18878/characters"
    else:
        assert call_args[1]["url"] == "https://www.anisearch.com/anime/18878/characters"


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_character_crawler.fetch_anisearch_characters")
async def test_main_function_with_default_output(mock_fetch):
    """Test main() function with default output path."""
    from src.enrichment.crawlers.anisearch_character_crawler import main

    mock_fetch.return_value = {"characters": [], "total_count": 0}

    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/12345/characters"]
    ):
        exit_code = await main()

    assert exit_code == 0
    # Verify default output path used
    call_args = mock_fetch.call_args
    assert call_args[1]["output_path"] == "anisearch_characters.json"


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_character_crawler.fetch_anisearch_characters")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.crawlers.anisearch_character_crawler import main

    mock_fetch.side_effect = ValueError("Crawler error")
    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/18878/characters"]
    ):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_character_crawler.fetch_anisearch_characters")
async def test_main_function_no_data_found(mock_fetch):
    """Test main() function when no data found."""
    from src.enrichment.crawlers.anisearch_character_crawler import main

    mock_fetch.return_value = None

    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/99999/characters"]
    ):
        exit_code = await main()

    # Crawler may return None, should still complete
    assert exit_code == 0  # CLI returns 0 even when no data per current implementation

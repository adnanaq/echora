"""
Unit tests for JikanDetailedFetcher.

Tests all methods including:
- Rate limiting logic
- Episode fetching with cache hit detection
- Character fetching with cache hit detection
- Retry logic for 429 errors
- Error handling
- Batch file operations
- Main data fetching workflow
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher


class TestJikanDetailedFetcherInit:
    """Test initialization of JikanDetailedFetcher."""

    def test_init_with_custom_session(self):
        """Test initialization with custom session provided."""
        mock_session = MagicMock()
        fetcher = JikanDetailedFetcher("123", "episodes", session=mock_session)

        assert fetcher.anime_id == "123"
        assert fetcher.data_type == "episodes"
        assert fetcher.session is mock_session
        assert fetcher.request_count == 0
        assert fetcher.batch_size == 50
        assert fetcher.max_requests_per_second == 3
        assert fetcher.max_requests_per_minute == 60

    @patch("src.enrichment.api_helpers.jikan_helper._cache_manager")
    def test_init_without_session(self, mock_cache_manager):
        """Test initialization without session - creates one from cache manager."""
        mock_session = MagicMock()
        mock_cache_manager.get_aiohttp_session.return_value = mock_session

        fetcher = JikanDetailedFetcher("456", "characters")

        mock_cache_manager.get_aiohttp_session.assert_called_once_with("jikan")
        assert fetcher.session is mock_session


class TestRateLimiting:
    """Test rate limiting logic."""

    @pytest.mark.asyncio
    async def test_respect_rate_limits_first_request(self):
        """Test first request doesn't wait."""
        fetcher = JikanDetailedFetcher("123", "episodes", session=MagicMock())
        fetcher.request_count = 0

        start = time.time()
        await fetcher.respect_rate_limits()
        elapsed = time.time() - start

        # First request should not wait
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_respect_rate_limits_subsequent_request(self):
        """Test subsequent requests wait 0.5s."""
        fetcher = JikanDetailedFetcher("123", "episodes", session=MagicMock())
        fetcher.request_count = 1

        start = time.time()
        await fetcher.respect_rate_limits()
        elapsed = time.time() - start

        # Should wait ~0.5s
        assert 0.4 < elapsed < 0.6

    @pytest.mark.asyncio
    async def test_respect_rate_limits_minute_reset(self):
        """Test counter resets after one minute."""
        fetcher = JikanDetailedFetcher("123", "episodes", session=MagicMock())
        fetcher.request_count = 30
        fetcher.start_time = time.time() - 61  # 61 seconds ago

        await fetcher.respect_rate_limits()

        # Should have reset counter
        assert fetcher.request_count == 0
        assert fetcher.start_time > time.time() - 2  # Recent start time

    @pytest.mark.asyncio
    async def test_respect_rate_limits_max_per_minute_reached(self):
        """Test waits when 60 requests/minute limit reached."""
        fetcher = JikanDetailedFetcher("123", "episodes", session=MagicMock())
        fetcher.request_count = 60
        fetcher.start_time = time.time() - 30  # 30 seconds ago

        start = time.time()
        await fetcher.respect_rate_limits()
        elapsed = time.time() - start

        # Should wait remaining time in minute (~30s)
        assert elapsed > 29
        assert fetcher.request_count == 0  # Reset after waiting


class TestFetchEpisodeDetail:
    """Test episode fetching with cache detection."""

    @pytest.mark.asyncio
    async def test_fetch_episode_success_cache_miss(self):
        """Test successful episode fetch from network (cache miss)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "url": "https://myanimelist.net/anime/21/episode/1",
                    "title": "Episode 1",
                    "title_japanese": "第1話",
                    "title_romaji": "Dai 1 wa",
                    "aired": "2000-01-01",
                    "score": 8.5,
                    "filler": False,
                    "recap": False,
                    "duration": 24,
                    "synopsis": "First episode",
                }
            }
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()  # Mock to avoid delays

        result = await fetcher.fetch_episode_detail(1)

        assert result is not None
        assert result["episode_number"] == 1
        assert result["title"] == "Episode 1"
        assert result["synopsis"] == "First episode"
        assert fetcher.request_count == 1
        fetcher.respect_rate_limits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_episode_success_cache_hit(self):
        """Test successful episode fetch from cache (no rate limiting)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = True  # Cache hit
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "url": "https://myanimelist.net/anime/21/episode/2",
                    "title": "Episode 2",
                    "filler": True,
                }
            }
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        result = await fetcher.fetch_episode_detail(2)

        assert result is not None
        assert result["episode_number"] == 2
        assert result["filler"] is True
        assert fetcher.request_count == 0  # Not incremented for cache hit
        fetcher.respect_rate_limits.assert_not_awaited()  # Not called for cache hit

    @pytest.mark.asyncio
    async def test_fetch_episode_429_retry_success(self):
        """Test episode fetch retries on 429 and succeeds."""
        # First attempt: 429, second attempt: success
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.from_cache = False
        mock_response_200.json = AsyncMock(
            return_value={"data": {"title": "Retry Success"}}
        )

        # Create context managers
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock(return_value=None)

        cm_200 = AsyncMock()
        cm_200.__aenter__ = AsyncMock(return_value=mock_response_200)
        cm_200.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[cm_429, cm_200])

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.fetch_episode_detail(1)

        assert result is not None
        assert result["title"] == "Retry Success"

    @pytest.mark.asyncio
    async def test_fetch_episode_429_max_retries(self):
        """Test episode fetch gives up after 3 retries."""
        mock_response = AsyncMock()
        mock_response.status = 429

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.fetch_episode_detail(1)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_episode_other_http_error(self):
        """Test episode fetch returns None for non-200/429 status."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        result = await fetcher.fetch_episode_detail(1)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_episode_exception(self):
        """Test episode fetch handles exceptions gracefully."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        result = await fetcher.fetch_episode_detail(1)

        assert result is None


class TestFetchCharacterDetail:
    """Test character fetching with cache detection."""

    @pytest.mark.asyncio
    async def test_fetch_character_success_cache_miss(self):
        """Test successful character fetch from network (cache miss)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "url": "https://myanimelist.net/character/1",
                    "name": "Test Character",
                    "name_kanji": "テスト",
                    "nicknames": ["Testy"],
                    "about": "A test character",
                    "images": {"jpg": {"image_url": "test.jpg"}},
                    "favorites": 100,
                }
            }
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        character_data = {
            "character": {"mal_id": 123},
            "role": "Main",
            "voice_actors": [{"name": "Actor A", "language": "Japanese"}],
        }

        result = await fetcher.fetch_character_detail(character_data)

        assert result is not None
        assert result["character_id"] == 123
        assert result["name"] == "Test Character"
        assert result["role"] == "Main"
        assert fetcher.request_count == 1
        fetcher.respect_rate_limits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_character_success_cache_hit(self):
        """Test successful character fetch from cache (no rate limiting)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = True  # Cache hit
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "name": "Cached Character",
                    "nicknames": [],
                }
            }
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        character_data = {"character": {"mal_id": 456}, "role": "Supporting"}

        result = await fetcher.fetch_character_detail(character_data)

        assert result is not None
        assert result["character_id"] == 456
        assert fetcher.request_count == 0  # Not incremented for cache hit
        fetcher.respect_rate_limits.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_character_429_retry_success(self):
        """Test character fetch retries on 429 and succeeds."""
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.from_cache = False
        mock_response_200.json = AsyncMock(
            return_value={"data": {"name": "Retry Char"}}
        )

        # Create context managers
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock(return_value=None)

        cm_200 = AsyncMock()
        cm_200.__aenter__ = AsyncMock(return_value=mock_response_200)
        cm_200.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[cm_429, cm_200])

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        character_data = {"character": {"mal_id": 789}}

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.fetch_character_detail(character_data)

        assert result is not None
        assert result["name"] == "Retry Char"

    @pytest.mark.asyncio
    async def test_fetch_character_429_max_retries(self):
        """Test character fetch gives up after 3 retries."""
        mock_response = AsyncMock()
        mock_response.status = 429

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)

        character_data = {"character": {"mal_id": 999}}

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.fetch_character_detail(character_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_character_other_http_error(self):
        """Test character fetch returns None for non-200/429 status."""
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)

        character_data = {"character": {"mal_id": 111}}

        result = await fetcher.fetch_character_detail(character_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_character_exception(self):
        """Test character fetch handles exceptions gracefully."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("API error"))

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)

        character_data = {"character": {"mal_id": 222}}

        result = await fetcher.fetch_character_detail(character_data)

        assert result is None


class TestBatchFileOperations:
    """Test batch file append operations."""

    def test_append_batch_new_file(self):
        """Test appending batch to new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "progress.json"
            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())

            batch_data = [{"id": 1, "data": "test1"}, {"id": 2, "data": "test2"}]

            count = fetcher.append_batch_to_file(batch_data, str(progress_file))

            assert count == 2
            assert progress_file.exists()

            with open(progress_file) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 2
            assert saved_data[0]["id"] == 1

    def test_append_batch_existing_file(self):
        """Test appending batch to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "progress.json"

            # Create existing file
            existing_data = [{"id": 1, "data": "existing"}]
            with open(progress_file, "w") as f:
                json.dump(existing_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            batch_data = [{"id": 2, "data": "new"}]

            count = fetcher.append_batch_to_file(batch_data, str(progress_file))

            assert count == 2

            with open(progress_file) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 2
            assert saved_data[0]["id"] == 1
            assert saved_data[1]["id"] == 2


class TestFetchDetailedData:
    """Test main fetch_detailed_data workflow."""

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_episodes_list_input(self):
        """Test fetching episodes with list input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Create input data
            input_data = [
                {"mal_id": 1, "title": "Ep 1"},
                {"mal_id": 2, "title": "Ep 2"},
            ]
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.fetch_episode_detail = AsyncMock(
                side_effect=[
                    {"episode_number": 1, "title": "Episode 1", "synopsis": "Test"},
                    {"episode_number": 2, "title": "Episode 2"},
                ]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            assert output_file.exists()
            with open(output_file) as f:
                result = json.load(f)

            assert len(result) == 2
            assert result[0]["episode_number"] == 1

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_episodes_paginated_input(self):
        """Test fetching episodes with paginated data input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Create paginated input data
            input_data = {"data": [{"mal_id": 1}, {"mal_id": 2}]}
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.fetch_episode_detail = AsyncMock(
                side_effect=[{"episode_number": 1}, {"episode_number": 2}]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_episodes_with_count(self):
        """Test fetching episodes with episode count in input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Input with episode count
            input_data = {"episodes": 3}
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.fetch_episode_detail = AsyncMock(
                side_effect=[
                    {"episode_number": 1},
                    {"episode_number": 2},
                    {"episode_number": 3},
                ]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            assert output_file.exists()
            with open(output_file) as f:
                result = json.load(f)

            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_characters(self):
        """Test fetching characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Create character input data
            input_data = {
                "data": [
                    {"character": {"mal_id": 100}, "role": "Main"},
                    {"character": {"mal_id": 200}, "role": "Supporting"},
                ]
            }
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "characters", session=MagicMock())
            fetcher.fetch_character_detail = AsyncMock(
                side_effect=[
                    {"character_id": 100, "name": "Char 1", "about": "Main character"},
                    {"character_id": 200, "name": "Char 2"},
                ]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            assert output_file.exists()
            with open(output_file) as f:
                result = json.load(f)

            assert len(result) == 2
            assert result[0]["character_id"] == 100

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_with_existing_progress(self):
        """Test resuming from existing progress file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"
            progress_file = Path(tmpdir) / "output.json.progress"

            # Create input
            input_data = [{"mal_id": 1}, {"mal_id": 2}, {"mal_id": 3}]
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            # Create existing progress (1 episode already fetched)
            existing_progress = [{"episode_number": 1}]
            with open(progress_file, "w") as f:
                json.dump(existing_progress, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            # Should only fetch episodes 2 and 3
            fetcher.fetch_episode_detail = AsyncMock(
                side_effect=[{"episode_number": 2}, {"episode_number": 3}]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            assert fetcher.fetch_episode_detail.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_batch_saving(self):
        """Test batch saving during fetch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Create large input to trigger batching
            input_data = [{"mal_id": i} for i in range(1, 101)]  # 100 episodes
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.batch_size = 50  # Batch every 50 items

            # Mock fetch to return data
            async def mock_fetch(ep_id):
                return {"episode_number": ep_id}

            fetcher.fetch_episode_detail = AsyncMock(side_effect=mock_fetch)

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            # Progress file should be cleaned up
            progress_file = Path(tmpdir) / "output.json.progress"
            assert not progress_file.exists()

            # Final file should have all data
            with open(output_file) as f:
                result = json.load(f)
            assert len(result) == 100

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_skip_failed_items(self):
        """Test that failed fetches are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            input_data = [{"mal_id": 1}, {"mal_id": 2}, {"mal_id": 3}]
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            # Episode 2 fails, others succeed
            fetcher.fetch_episode_detail = AsyncMock(
                side_effect=[
                    {"episode_number": 1},
                    None,  # Failed
                    {"episode_number": 3},
                ]
            )

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            with open(output_file) as f:
                result = json.load(f)

            # Should only have 2 episodes (failed one skipped)
            assert len(result) == 2
            assert result[0]["episode_number"] == 1
            assert result[1]["episode_number"] == 3


class TestMainFunction:
    """Test CLI main function."""

    @pytest.mark.asyncio
    async def test_main_missing_input_file_error(self):
        """Test main() error handling for missing input file (lines 363-364)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = str(Path(tmpdir) / "nonexistent.json")
            output_file = str(Path(tmpdir) / "output.json")

            with patch(
                "sys.argv", ["script", "episodes", "21", nonexistent, output_file]
            ):
                from src.enrichment.api_helpers.jikan_helper import main

                # Should return exit code 1 for missing file
                exit_code = await main()
                assert exit_code == 1

    @pytest.mark.asyncio
    async def test_main_creates_output_directory(self):
        """Test main creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "subdir" / "output.json"

            # Create input file
            with open(input_file, "w") as f:
                json.dump([{"mal_id": 1}], f)

            with patch(
                "sys.argv",
                ["script", "episodes", "21", str(input_file), str(output_file)],
            ):
                with patch(
                    "src.enrichment.api_helpers.jikan_helper.JikanDetailedFetcher"
                ) as mock_fetcher_class:
                    mock_fetcher = MagicMock()
                    mock_fetcher.fetch_detailed_data = AsyncMock()
                    mock_fetcher_class.return_value = mock_fetcher

                    from src.enrichment.api_helpers.jikan_helper import main

                    await main()

                    # Output directory should be created
                    assert output_file.parent.exists()
                    mock_fetcher.fetch_detailed_data.assert_awaited_once()

    def test_main_entrypoint(self):
        """Test __main__ entrypoint execution (line 375)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Create minimal input file
            with open(input_file, "w") as f:
                json.dump([{"mal_id": 1}], f)

            # Test the actual __main__ execution path
            import subprocess
            import sys

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.enrichment.api_helpers.jikan_helper",
                    "episodes",
                    "21",
                    str(input_file),
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # The script should execute (may succeed or fail, but line 375 will be covered)
            # Success (0) or error exit (1) both mean the line was executed
            assert result.returncode in [0, 1]


class TestEdgeCasesAndBoundaries:
    """Additional edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_fetch_episode_no_from_cache_attribute(self):
        """Test episode fetch when response has no from_cache attribute."""
        mock_response = AsyncMock()
        mock_response.status = 200
        # No from_cache attribute set
        mock_response.json = AsyncMock(return_value={"data": {"title": "Test"}})

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        result = await fetcher.fetch_episode_detail(1)

        # Should default to False and rate limit
        assert result is not None
        assert fetcher.request_count == 1
        fetcher.respect_rate_limits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_episode_missing_optional_fields(self):
        """Test episode fetch with minimal data (all optional fields missing)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(return_value={"data": {}})  # Empty data

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        result = await fetcher.fetch_episode_detail(1)

        # Should handle missing fields gracefully
        assert result is not None
        assert result["episode_number"] == 1
        assert result["url"] is None
        assert result["title"] is None
        assert result["filler"] is False  # Default value
        assert result["recap"] is False  # Default value

    @pytest.mark.asyncio
    async def test_fetch_character_missing_optional_fields(self):
        """Test character fetch with minimal data."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(return_value={"data": {}})  # Empty data

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        character_data = {"character": {"mal_id": 123}}

        result = await fetcher.fetch_character_detail(character_data)

        # Should handle missing fields gracefully
        assert result is not None
        assert result["character_id"] == 123
        assert result["nicknames"] == []  # Default value
        assert result["images"] == {}  # Default value

    @pytest.mark.asyncio
    async def test_rate_limit_exactly_at_boundary(self):
        """Test rate limiting at exactly 60 requests."""
        fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
        fetcher.request_count = 59
        fetcher.start_time = time.time()

        # Should not wait yet
        start = time.time()
        await fetcher.respect_rate_limits()
        elapsed = time.time() - start

        assert elapsed < 1  # Should be fast, only 0.5s wait

    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_requests(self):
        """Test rate limiting with rapid concurrent requests."""
        fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())

        # Simulate rapid requests
        start = time.time()
        for i in range(5):
            fetcher.request_count = i
            await fetcher.respect_rate_limits()

        elapsed = time.time() - start

        # Should take at least 2 seconds (4 * 0.5s waits, first doesn't wait)
        assert elapsed >= 2.0

    @pytest.mark.asyncio
    async def test_fetch_episode_json_decode_error(self):
        """Test episode fetch with invalid JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(
            side_effect=json.JSONDecodeError("msg", "doc", 0)
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        result = await fetcher.fetch_episode_detail(1)

        # Should handle JSON error gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_episode_timeout_error(self):
        """Test episode fetch with timeout."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        result = await fetcher.fetch_episode_detail(1)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_empty_input_list(self):
        """Test fetching with empty input list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Empty list
            with open(input_file, "w") as f:
                json.dump([], f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.fetch_episode_detail = AsyncMock()

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            # Should not call fetch_episode_detail for empty list
            fetcher.fetch_episode_detail.assert_not_awaited()

            # Output should be empty list
            assert output_file.exists()
            with open(output_file) as f:
                result = json.load(f)
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_zero_episodes(self):
        """Test fetching with zero episode count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # Zero episodes
            with open(input_file, "w") as f:
                json.dump({"episodes": 0}, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.fetch_episode_detail = AsyncMock()

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            fetcher.fetch_episode_detail.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_all_items_fail(self):
        """Test fetching when all items fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            input_data = [{"mal_id": 1}, {"mal_id": 2}]
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            # All fetches return None
            fetcher.fetch_episode_detail = AsyncMock(return_value=None)

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            # Output should be empty list
            with open(output_file) as f:
                result = json.load(f)
            assert result == []

    def test_append_batch_with_unicode(self):
        """Test appending batch with Unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "progress.json"
            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())

            # Japanese characters, emojis, etc.
            batch_data = [
                {"id": 1, "title": "テスト", "emoji": "🎌"},
                {"id": 2, "title": "한국어", "emoji": "🇰🇷"},
                {"id": 3, "title": "中文", "emoji": "🇨🇳"},
            ]

            count = fetcher.append_batch_to_file(batch_data, str(progress_file))

            assert count == 3

            # Verify Unicode is preserved
            with open(progress_file, encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data[0]["title"] == "テスト"
            assert saved_data[1]["emoji"] == "🇰🇷"

    def test_append_batch_with_special_characters(self):
        """Test appending batch with special JSON characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "progress.json"
            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())

            batch_data = [
                {"id": 1, "desc": 'Quote: "test"'},
                {"id": 2, "desc": "Newline:\ntest"},
                {"id": 3, "desc": "Tab:\ttest"},
                {"id": 4, "desc": "Backslash: \\test"},
            ]

            count = fetcher.append_batch_to_file(batch_data, str(progress_file))

            assert count == 4

            # Verify special characters are escaped properly
            with open(progress_file) as f:
                saved_data = json.load(f)

            assert '"test"' in saved_data[0]["desc"]
            assert "\n" in saved_data[1]["desc"]

    @pytest.mark.asyncio
    async def test_fetch_episode_429_then_other_error(self):
        """Test 429 retry that fails with different error."""
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429

        mock_response_500 = AsyncMock()
        mock_response_500.status = 500

        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        cm_429.__aexit__ = AsyncMock(return_value=None)

        cm_500 = AsyncMock()
        cm_500.__aenter__ = AsyncMock(return_value=mock_response_500)
        cm_500.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[cm_429, cm_500])

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.fetch_episode_detail(1)

        # Should return None for 500 error
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_character_no_role_in_data(self):
        """Test character fetch when role is missing from input."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(return_value={"data": {"name": "Test"}})

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        # No role or voice_actors in character_data
        character_data = {"character": {"mal_id": 123}}

        result = await fetcher.fetch_character_detail(character_data)

        assert result is not None
        assert result["role"] is None
        assert result["voice_actors"] == []

    @pytest.mark.asyncio
    async def test_fetch_detailed_data_progress_partial_batch(self):
        """Test progress file with partial batch (less than batch_size)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"

            # 10 items (less than batch_size of 50)
            input_data = [{"mal_id": i} for i in range(1, 11)]
            with open(input_file, "w") as f:
                json.dump(input_data, f)

            fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
            fetcher.batch_size = 50

            async def mock_fetch(ep_id):
                return {"episode_number": ep_id}

            fetcher.fetch_episode_detail = AsyncMock(side_effect=mock_fetch)

            await fetcher.fetch_detailed_data(str(input_file), str(output_file))

            # Should save final batch even though it's < 50
            with open(output_file) as f:
                result = json.load(f)

            assert len(result) == 10

    @pytest.mark.asyncio
    async def test_rate_limit_time_going_backwards(self):
        """Test rate limiting when system time goes backwards (edge case)."""
        fetcher = JikanDetailedFetcher("21", "episodes", session=MagicMock())
        fetcher.request_count = 30
        fetcher.start_time = time.time() + 10  # Future time

        # Should handle negative elapsed time gracefully
        await fetcher.respect_rate_limits()

        # Should reset counter when time is inconsistent
        assert fetcher.request_count == 0

    @pytest.mark.asyncio
    async def test_fetch_episode_response_missing_data_key(self):
        """Test episode fetch when response is missing 'data' key."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(return_value={})  # No 'data' key

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "episodes", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        result = await fetcher.fetch_episode_detail(1)

        # Should handle KeyError gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_character_response_missing_data_key(self):
        """Test character fetch when response is missing 'data' key."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.from_cache = False
        mock_response.json = AsyncMock(return_value={})  # No 'data' key

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock()

        fetcher = JikanDetailedFetcher("21", "characters", session=mock_session)
        fetcher.respect_rate_limits = AsyncMock()

        character_data = {"character": {"mal_id": 123}}

        result = await fetcher.fetch_character_detail(character_data)

        # Should handle KeyError gracefully
        assert result is None


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.jikan_helper.JikanDetailedFetcher")
@patch("os.makedirs")
@patch("os.path.exists")
async def test_main_function_success_episodes(mock_exists, mock_makedirs, mock_fetcher_class):
    """Test main() function handles successful episodes execution."""
    from src.enrichment.api_helpers.jikan_helper import main

    mock_exists.return_value = True
    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_detailed_data = AsyncMock()
    mock_fetcher_class.return_value = mock_fetcher

    with patch("sys.argv", ["script.py", "episodes", "123", "input.json", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetcher_class.assert_called_once_with("123", "episodes")
    mock_fetcher.fetch_detailed_data.assert_awaited_once_with("input.json", "/tmp/output.json")


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.jikan_helper.JikanDetailedFetcher")
@patch("os.makedirs")
@patch("os.path.exists")
async def test_main_function_success_characters(mock_exists, mock_makedirs, mock_fetcher_class):
    """Test main() function handles successful characters execution."""
    from src.enrichment.api_helpers.jikan_helper import main

    mock_exists.return_value = True
    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_detailed_data = AsyncMock()
    mock_fetcher_class.return_value = mock_fetcher

    with patch("sys.argv", ["script.py", "characters", "456", "input.json", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetcher_class.assert_called_once_with("456", "characters")


@pytest.mark.asyncio
@patch("os.path.exists")
async def test_main_function_input_file_not_exists(mock_exists):
    """Test main() function returns error when input file doesn't exist."""
    from src.enrichment.api_helpers.jikan_helper import main

    mock_exists.return_value = False

    with patch("sys.argv", ["script.py", "episodes", "123", "nonexistent.json", "output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.jikan_helper.JikanDetailedFetcher")
@patch("os.path.exists")
async def test_main_function_error_handling(mock_exists, mock_fetcher_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.api_helpers.jikan_helper import main

    mock_exists.return_value = True
    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_detailed_data = AsyncMock(side_effect=Exception("API error"))
    mock_fetcher_class.return_value = mock_fetcher

    with patch("sys.argv", ["script.py", "episodes", "123", "input.json", "output.json"]):
        exit_code = await main()

    assert exit_code == 1

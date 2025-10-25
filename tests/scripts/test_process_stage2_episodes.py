#!/usr/bin/env python3
"""
Test suite for process_stage2_episodes.py script.
Tests episode processing with multi-source integration (Jikan, Kitsu, AniSearch).
Achieves 100% code coverage including all edge cases.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from process_stage2_episodes import (
    auto_detect_temp_dir,
    convert_jst_to_utc,
    load_anisearch_episode_data,
    load_kitsu_episode_data,
    process_all_episodes,
)


class TestTimezoneConversion:
    """Test JST to UTC timezone conversion."""

    def test_convert_jst_to_utc_basic(self):
        """Test basic JST to UTC conversion."""
        jst_time = "1999-10-20T00:00:00+09:00"
        expected = "1999-10-19T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_non_midnight(self):
        """Test JST to UTC conversion with non-midnight time."""
        jst_time = "2024-10-04T09:30:00+09:00"
        expected = "2024-10-04T00:30:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_with_milliseconds(self):
        """Test JST to UTC conversion with milliseconds."""
        jst_time = "2024-10-04T09:30:00.123+09:00"
        expected = "2024-10-04T00:30:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_none(self):
        """Test handling of None input."""
        assert convert_jst_to_utc(None) is None

    def test_convert_jst_to_utc_empty_string(self):
        """Test handling of empty string."""
        assert convert_jst_to_utc("") is None

    def test_convert_jst_to_utc_date_boundary(self):
        """Test date boundary shift (JST to UTC crosses date)."""
        jst_time = "2024-01-01T00:00:00+09:00"
        expected = "2023-12-31T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_leap_year(self):
        """Test leap year date handling."""
        jst_time = "2024-02-29T00:00:00+09:00"
        expected = "2024-02-28T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_year_boundary(self):
        """Test year boundary crossing."""
        jst_time = "2024-01-01T08:00:00+09:00"
        expected = "2023-12-31T23:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_invalid_format(self):
        """Test handling of invalid datetime format."""
        invalid_time = "not-a-datetime"
        # Should return original string on error
        result = convert_jst_to_utc(invalid_time)
        assert result == invalid_time

    def test_convert_jst_to_utc_partial_datetime(self):
        """Test handling of partial datetime string (date only)."""
        partial_time = "2024-10-04"
        # Python's fromisoformat can parse date-only strings, treats as local midnight
        result = convert_jst_to_utc(partial_time)
        # Date-only gets interpreted as local time, will be converted to UTC
        assert result is not None and result.startswith("2024-10-04")


class TestKitsuDataLoading:
    """Test Kitsu episode data loading and processing."""

    @pytest.fixture
    def mock_kitsu_data(self):
        """Create mock Kitsu data with all possible fields."""
        return {
            "anime": {"attributes": {"slug": "one-piece"}},
            "episodes": [
                {
                    "attributes": {
                        "number": 1,
                        "thumbnail": {"original": "https://example.com/thumb1.jpg"},
                        "description": "  Episode 1 description  ",  # Test strip
                        "synopsis": "  Episode 1 synopsis  ",  # Test strip
                        "seasonNumber": 1,
                        "titles": {
                            "en": "English Title",
                            "en_us": "US English Title",
                            "ja_jp": "日本語タイトル",
                            "en_jp": "Romaji Title",
                        },
                    }
                },
                {
                    "attributes": {
                        "number": 2,
                        "thumbnail": {"original": "https://example.com/thumb2.jpg"},
                        "titles": {"en_us": "US Only Title"},
                    }
                },
                {
                    "attributes": {
                        "number": 3,
                        "thumbnail": {},  # Empty thumbnail object
                        "description": "   ",  # Whitespace only
                        "synopsis": "",  # Empty string
                        "seasonNumber": 0,  # Zero is valid
                        "titles": {},  # No titles
                    }
                },
                {
                    "attributes": {
                        # Missing number - should be skipped
                        "titles": {"en": "No Episode Number"}
                    }
                },
            ],
        }

    @pytest.fixture
    def temp_dir_with_kitsu(self, mock_kitsu_data, tmp_path):
        """Create temporary directory with Kitsu data."""
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(mock_kitsu_data, f)
        return str(tmp_path)

    def test_load_kitsu_episode_data_success(self, temp_dir_with_kitsu):
        """Test successful loading of Kitsu episode data."""
        result = load_kitsu_episode_data(temp_dir_with_kitsu)

        (
            thumbnails,
            descriptions,
            synopses,
            titles,
            titles_jp,
            titles_romaji,
            season_nums,
            episode_urls,
        ) = result

        # Check thumbnails
        assert len(thumbnails) == 2
        assert thumbnails[1] == "https://example.com/thumb1.jpg"
        assert thumbnails[2] == "https://example.com/thumb2.jpg"

        # Check descriptions (should be stripped)
        assert len(descriptions) == 1
        assert descriptions[1] == "Episode 1 description"

        # Check synopses (should be stripped)
        assert len(synopses) == 1
        assert synopses[1] == "Episode 1 synopsis"

        # Check titles
        assert len(titles) == 2
        assert titles[1] == "English Title"  # Should prefer 'en' over 'en_us'
        assert titles[2] == "US Only Title"  # Falls back to 'en_us'

        # Check season numbers (including 0)
        assert len(season_nums) == 2
        assert season_nums[1] == 1
        assert season_nums[3] == 0  # Zero is valid

        # Check episode URLs
        assert len(episode_urls) == 3  # Episodes 1, 2, 3 (not 4 - missing number)
        assert episode_urls[1] == "https://kitsu.app/anime/one-piece/episodes/1"

    def test_load_kitsu_episode_data_missing_file(self, tmp_path):
        """Test handling of missing Kitsu file."""
        result = load_kitsu_episode_data(str(tmp_path))

        # Should return 8 empty dicts
        assert len(result) == 8
        assert all(r == {} for r in result)

    def test_load_kitsu_episode_data_en_priority(self, temp_dir_with_kitsu):
        """Test that 'en' locale is prioritized over 'en_us'."""
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        titles = result[3]  # titles is the 4th element (index 3)

        # Episode 1 has both 'en' and 'en_us', should use 'en'
        assert titles[1] == "English Title"

        # Episode 2 only has 'en_us', should use that
        assert titles[2] == "US Only Title"

    def test_load_kitsu_episode_url_construction(self, temp_dir_with_kitsu):
        """Test Kitsu episode URL construction."""
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        episode_urls = result[7]  # episode_urls is the 8th element (index 7)

        assert episode_urls[1] == "https://kitsu.app/anime/one-piece/episodes/1"
        assert episode_urls[2] == "https://kitsu.app/anime/one-piece/episodes/2"

    def test_load_kitsu_episode_data_no_slug(self, tmp_path):
        """Test handling when anime slug is missing."""
        data = {
            "anime": {"attributes": {}},
            "episodes": [{"attributes": {"number": 1, "titles": {"en": "Test"}}}],
        }

        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(data, f)

        result = load_kitsu_episode_data(str(tmp_path))
        episode_urls = result[7]

        # No URLs should be created without slug
        assert len(episode_urls) == 0

    def test_load_kitsu_episode_data_japanese_romaji_titles(self, temp_dir_with_kitsu):
        """Test Japanese and Romaji title extraction."""
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        titles_jp = result[4]
        titles_romaji = result[5]

        assert titles_jp[1] == "日本語タイトル"
        assert titles_romaji[1] == "Romaji Title"

    def test_load_kitsu_episode_data_exception_handling(self, tmp_path):
        """Test exception handling for corrupted JSON."""
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            f.write("{invalid json")

        result = load_kitsu_episode_data(str(tmp_path))
        # Should return 8 empty dicts on exception
        assert len(result) == 8
        assert all(r == {} for r in result)

    def test_load_kitsu_episode_data_empty_thumbnail(self, tmp_path):
        """Test handling of empty thumbnail object."""
        data = {
            "anime": {"attributes": {"slug": "test"}},
            "episodes": [
                {
                    "attributes": {
                        "number": 1,
                        "thumbnail": {},  # Empty thumbnail
                    }
                },
                {
                    "attributes": {
                        "number": 2,
                        "thumbnail": {"original": None},  # None original
                    }
                },
            ],
        }

        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(data, f)

        result = load_kitsu_episode_data(str(tmp_path))
        thumbnails = result[0]

        # No thumbnails should be extracted
        assert len(thumbnails) == 0


class TestAniSearchDataLoading:
    """Test AniSearch episode data loading."""

    @pytest.fixture
    def mock_anisearch_data(self):
        """Create mock AniSearch data."""
        return {
            "episodes": [
                {"episodeNumber": 1, "title": "AniSearch Episode 1"},
                {"episodeNumber": 2, "title": "AniSearch Episode 2"},
                {
                    "episodeNumber": 3,
                    "title": "",  # Empty title should be skipped
                },
                {
                    "episodeNumber": None,  # None episode number
                    "title": "No Episode Number",
                },
                {
                    # Missing episodeNumber
                    "title": "Missing Episode Number"
                },
            ]
        }

    @pytest.fixture
    def temp_dir_with_anisearch(self, mock_anisearch_data, tmp_path):
        """Create temporary directory with AniSearch data."""
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(mock_anisearch_data, f)
        return str(tmp_path)

    def test_load_anisearch_episode_data_success(self, temp_dir_with_anisearch):
        """Test successful loading of AniSearch episode data."""
        titles = load_anisearch_episode_data(temp_dir_with_anisearch)

        # Only episodes 1 and 2 should be included
        assert len(titles) == 2
        assert titles[1] == "AniSearch Episode 1"
        assert titles[2] == "AniSearch Episode 2"

    def test_load_anisearch_episode_data_missing_file(self, tmp_path):
        """Test handling of missing AniSearch file."""
        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_empty_episodes(self, tmp_path):
        """Test handling of empty episodes list."""
        data = {"episodes": []}

        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(data, f)

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_exception_handling(self, tmp_path):
        """Test exception handling for corrupted JSON."""
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            f.write("{malformed json")

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_missing_episodes_key(self, tmp_path):
        """Test handling when 'episodes' key is missing."""
        data = {"other_key": "value"}

        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(data, f)

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}


class TestEpisodeProcessing:
    """Test full episode processing pipeline."""

    @pytest.fixture
    def mock_episodes_detailed(self):
        """Create mock episodes_detailed.json data."""
        return [
            {
                "episode_number": 1,
                "title": "Jikan Title 1",
                "title_japanese": "Jikan Japanese 1",
                "title_romaji": "Jikan Romaji 1",
                "synopsis": "Jikan synopsis",
                "aired": "1999-10-20T00:00:00+09:00",
                "duration": 1440,
                "score": 8.5,
                "filler": False,
                "recap": False,
                "url": "https://myanimelist.net/anime/21/One_Piece/episode/1",
            },
            {
                "episode_number": 2,
                "title": None,  # Missing Jikan title, should fallback
                "title_japanese": None,
                "title_romaji": None,
                "synopsis": None,
                "aired": "1999-10-27T00:00:00+09:00",
                "duration": 1440,
                "score": None,
                "filler": True,  # Test filler flag
                "recap": False,
                "url": "https://myanimelist.net/anime/21/One_Piece/episode/2",
            },
            {
                "episode_number": 3,
                "title": None,  # Will test AniSearch fallback
                "synopsis": None,
                "aired": "1999-11-03T00:00:00+09:00",
                "duration": 1440,
                "filler": False,
                "recap": True,  # Test recap flag
                # No URL
            },
        ]

    @pytest.fixture
    def complete_test_env(self, mock_episodes_detailed, tmp_path):
        """Create complete test environment with all data sources."""
        # Episodes detailed
        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(mock_episodes_detailed, f)

        # Kitsu data
        kitsu_data = {
            "anime": {"attributes": {"slug": "one-piece"}},
            "episodes": [
                {
                    "attributes": {
                        "number": 2,
                        "thumbnail": {"original": "https://kitsu.io/thumb2.jpg"},
                        "titles": {
                            "en": "Kitsu Title 2",
                            "ja_jp": "Kitsu Japanese 2",
                            "en_jp": "Kitsu Romaji 2",
                        },
                        "synopsis": "Kitsu synopsis 2",
                        "description": "Kitsu description 2",
                        "seasonNumber": 1,
                    }
                }
            ],
        }
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(kitsu_data, f)

        # AniSearch data
        anisearch_data = {
            "episodes": [{"episodeNumber": 3, "title": "AniSearch Title 3"}]
        }
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(anisearch_data, f)

        return str(tmp_path)

    def test_process_all_episodes_success(self, complete_test_env):
        """Test full episode processing with all data sources."""
        process_all_episodes(complete_test_env)

        # Verify output file was created
        output_file = Path(complete_test_env) / "stage2_episodes.json"
        assert output_file.exists()

        # Load and verify output
        with open(output_file) as f:
            output = json.load(f)

        assert "episode_details" in output
        episodes = output["episode_details"]
        assert len(episodes) == 3

        # Test Episode 1: Jikan data
        ep1 = episodes[0]
        assert ep1["episode_number"] == 1
        assert ep1["title"] == "Jikan Title 1"
        assert ep1["title_japanese"] == "Jikan Japanese 1"
        assert ep1["title_romaji"] == "Jikan Romaji 1"
        assert ep1["synopsis"] == "Jikan synopsis"
        assert ep1["aired"] == "1999-10-19T15:00:00Z"  # Converted to UTC
        assert ep1["duration"] == 1440
        assert ep1["score"] == 8.5
        assert ep1["filler"] is False
        assert ep1["recap"] is False
        assert (
            ep1["episode_pages"]["mal"]
            == "https://myanimelist.net/anime/21/One_Piece/episode/1"
        )
        assert ep1["streaming"] == {}

        # Test Episode 2: Kitsu fallback for title
        ep2 = episodes[1]
        assert ep2["episode_number"] == 2
        assert ep2["title"] == "Kitsu Title 2"  # Fallback to Kitsu
        assert ep2["title_japanese"] == "Kitsu Japanese 2"  # Fallback to Kitsu
        assert ep2["title_romaji"] == "Kitsu Romaji 2"  # Fallback to Kitsu
        assert ep2["synopsis"] == "Kitsu synopsis 2"  # Kitsu fallback
        assert ep2["description"] == "Kitsu description 2"  # From Kitsu
        assert ep2["season_number"] == 1
        assert ep2["aired"] == "1999-10-26T15:00:00Z"  # Converted to UTC
        assert ep2["filler"] is True
        assert ep2["thumbnails"] == ["https://kitsu.io/thumb2.jpg"]
        assert "kitsu" in ep2["episode_pages"]
        assert (
            ep2["episode_pages"]["kitsu"]
            == "https://kitsu.app/anime/one-piece/episodes/2"
        )

        # Test Episode 3: AniSearch fallback for title
        ep3 = episodes[2]
        assert ep3["episode_number"] == 3
        assert ep3["title"] == "AniSearch Title 3"  # Fallback to AniSearch
        assert ep3["aired"] == "1999-11-02T15:00:00Z"  # Converted to UTC
        assert ep3["recap"] is True
        assert ep3["episode_pages"] == {}  # No URL

    def test_process_all_episodes_title_fallback_priority(self, complete_test_env):
        """Test that title fallback follows Jikan → Kitsu → AniSearch priority."""
        process_all_episodes(complete_test_env)

        output_file = Path(complete_test_env) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episode_details"]

        # Episode 1: Has Jikan title
        assert episodes[0]["title"] == "Jikan Title 1"

        # Episode 2: No Jikan, has Kitsu
        assert episodes[1]["title"] == "Kitsu Title 2"

        # Episode 3: No Jikan or Kitsu, has AniSearch
        assert episodes[2]["title"] == "AniSearch Title 3"

    def test_process_all_episodes_episode_pages(self, complete_test_env):
        """Test episode_pages object construction."""
        process_all_episodes(complete_test_env)

        output_file = Path(complete_test_env) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episode_details"]

        # Episode 1: Has MAL URL
        assert "mal" in episodes[0]["episode_pages"]

        # Episode 2: Has both MAL and Kitsu URLs
        assert "mal" in episodes[1]["episode_pages"]
        assert "kitsu" in episodes[1]["episode_pages"]

        # Episode 3: No URL in original data
        assert episodes[2]["episode_pages"] == {}

    def test_process_all_episodes_timezone_conversion(self, complete_test_env):
        """Test that all episode aired dates are converted to UTC."""
        process_all_episodes(complete_test_env)

        output_file = Path(complete_test_env) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episode_details"]

        for episode in episodes:
            aired = episode.get("aired")
            if aired:
                # Should end with 'Z' (UTC)
                assert aired.endswith("Z")
                # Should not contain timezone offset
                assert "+09:00" not in aired

    def test_process_all_episodes_minimal_data(self, tmp_path):
        """Test processing with minimal data (no Kitsu or AniSearch)."""
        # Only episodes_detailed
        episodes_data = [
            {
                "episode_number": 1,
                "title": "Episode 1",
                "aired": "2024-01-01T00:00:00+09:00",
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        assert output_file.exists()

        with open(output_file) as f:
            output = json.load(f)

        assert len(output["episode_details"]) == 1
        ep = output["episode_details"][0]
        assert ep["title"] == "Episode 1"
        assert ep["aired"] == "2023-12-31T15:00:00Z"

    def test_process_all_episodes_output_structure(self, complete_test_env):
        """Test output JSON structure and file creation."""
        process_all_episodes(complete_test_env)

        output_file = Path(complete_test_env) / "stage2_episodes.json"
        assert output_file.exists()

        # Verify file is valid JSON
        with open(output_file) as f:
            output = json.load(f)

        # Verify structure
        assert isinstance(output, dict)
        assert "episode_details" in output
        assert isinstance(output["episode_details"], list)

    def test_process_all_episodes_with_four_episodes(self, tmp_path):
        """Test timezone conversion examples print (covers lines 236-239)."""
        episodes_data = [
            {
                "episode_number": i,
                "title": f"Ep {i}",
                "aired": f"1999-10-{20+i}T00:00:00+09:00",
                "filler": False,
                "recap": False,
            }
            for i in range(1, 5)
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        # This will print examples for first 3 episodes (line 236-239)
        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        assert len(output["episode_details"]) == 4


class TestAutoDetectTempDir:
    """Test auto-detection of temp directory."""

    def test_auto_detect_single_directory(self, tmp_path, monkeypatch):
        """Test auto-detection with single directory."""
        # Create temp directory structure
        temp_base = tmp_path / "temp"
        temp_base.mkdir()
        anime_dir = temp_base / "One_agent1"
        anime_dir.mkdir()

        # Change to temp directory parent
        monkeypatch.chdir(tmp_path)

        result = auto_detect_temp_dir()
        assert result == "temp/One_agent1"

    def test_auto_detect_multiple_directories(self, tmp_path, monkeypatch):
        """Test auto-detection with multiple directories (should exit)."""
        # Create temp directory structure with multiple anime dirs
        temp_base = tmp_path / "temp"
        temp_base.mkdir()
        (temp_base / "One_agent1").mkdir()
        (temp_base / "Naruto_agent1").mkdir()

        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1

    def test_auto_detect_no_temp_directory(self, tmp_path, monkeypatch):
        """Test auto-detection when temp directory doesn't exist."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1

    def test_auto_detect_empty_temp_directory(self, tmp_path, monkeypatch):
        """Test auto-detection with empty temp directory."""
        temp_base = tmp_path / "temp"
        temp_base.mkdir()

        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_episodes_detailed_file(self, tmp_path):
        """Test handling when episodes_detailed.json is missing."""
        with pytest.raises(FileNotFoundError):
            process_all_episodes(str(tmp_path))

    def test_empty_episodes_list(self, tmp_path):
        """Test processing empty episodes list."""
        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump([], f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        assert output["episode_details"] == []

    def test_malformed_json_graceful_handling(self, tmp_path):
        """Test graceful handling of malformed JSON files."""
        # Create malformed Kitsu file
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            f.write("{invalid json")

        # Should handle error and return empty dicts
        result = load_kitsu_episode_data(str(tmp_path))
        assert all(r == {} for r in result)

    def test_episode_with_missing_fields(self, tmp_path):
        """Test processing episode with many missing fields."""
        episodes_data = [
            {
                "episode_number": 1
                # Missing all other fields
            }
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        ep = output["episode_details"][0]
        assert ep["episode_number"] == 1
        assert ep["title"] is None
        assert ep["aired"] is None
        assert ep["filler"] is False  # Default value
        assert ep["recap"] is False  # Default value
        assert ep["thumbnails"] == []
        assert ep["episode_pages"] == {}

    def test_episode_with_null_aired_date(self, tmp_path):
        """Test episode with null aired date."""
        episodes_data = [
            {
                "episode_number": 1,
                "title": "Test",
                "aired": None,
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        ep = output["episode_details"][0]
        assert ep["aired"] is None

    def test_ensure_ascii_false(self, tmp_path):
        """Test that ensure_ascii=False preserves unicode characters."""
        episodes_data = [
            {
                "episode_number": 1,
                "title": "テスト",  # Japanese characters
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump(episodes_data, f, ensure_ascii=False)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"

        # Read raw file content
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
            # Should contain actual Japanese characters, not unicode escapes
            assert "テスト" in content

    def test_synopsis_fallback_with_empty_jikan(self, tmp_path):
        """Test synopsis fallback when Jikan has empty string."""
        episodes_data = [
            {
                "episode_number": 1,
                "synopsis": "",  # Empty string
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        # Kitsu data with synopsis
        kitsu_data = {
            "anime": {"attributes": {"slug": "test"}},
            "episodes": [{"attributes": {"number": 1, "synopsis": "Kitsu synopsis"}}],
        }
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(kitsu_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        # Empty string is falsy, should fallback to Kitsu
        ep = output["episode_details"][0]
        assert ep["synopsis"] == "Kitsu synopsis"


class TestMainExecution:
    """Test main script execution paths (argparse and main block)."""

    def test_main_with_agent_id_absolute_path(self, tmp_path):
        """Test main execution with agent_id and absolute path."""
        # Create test environment
        anime_dir = tmp_path / "test_agent"
        anime_dir.mkdir()

        episodes_file = anime_dir / "episodes_detailed.json"
        with open(episodes_file, "w") as f:
            json.dump(
                [
                    {
                        "episode_number": 1,
                        "title": "Test",
                        "filler": False,
                        "recap": False,
                    }
                ],
                f,
            )

        # Simulate command line arguments
        test_args = ["script_name", "test_agent", "--temp-dir", str(tmp_path)]

        with patch("sys.argv", test_args):
            with patch("process_stage2_episodes.PROJECT_ROOT", tmp_path):
                # Import and run main block
                import subprocess

                result = subprocess.run(
                    [
                        "python",
                        str(SCRIPTS_DIR / "process_stage2_episodes.py"),
                        "test_agent",
                        "--temp-dir",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    text=True,
                )

        assert result.returncode == 0
        assert (anime_dir / "stage2_episodes.json").exists()

    def test_main_with_nonexistent_directory(self, tmp_path):
        """Test main execution with non-existent directory."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                str(SCRIPTS_DIR / "process_stage2_episodes.py"),
                "nonexistent_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "does not exist" in result.stdout

    def test_main_with_missing_episodes_file(self, tmp_path):
        """Test main execution when episodes_detailed.json is missing."""
        anime_dir = tmp_path / "test_agent"
        anime_dir.mkdir()

        import subprocess

        result = subprocess.run(
            [
                "python",
                str(SCRIPTS_DIR / "process_stage2_episodes.py"),
                "test_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Required file not found" in result.stdout


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=process_stage2_episodes", "--cov-report=term-missing"]
    )

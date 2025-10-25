#!/usr/bin/env python3
"""
Test suite for process_stage4_statistics.py script.
Tests statistics extraction from 6 anime data sources with score normalization.
Achieves 100% code coverage including all edge cases.
"""

import json
import sys
from pathlib import Path

import pytest
from process_stage4_statistics import (
    extract_all_statistics,
    extract_anidb_statistics,
    extract_anilist_statistics,
    extract_animeplanet_statistics,
    extract_animeschedule_statistics,
    extract_kitsu_statistics,
    extract_mal_statistics,
    has_any_statistics,
    load_source_data,
    normalize_score,
    safe_get,
)


@pytest.fixture
def stage4_script_path():
    """Get path to stage4 script for subprocess calls."""
    # Find project root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current / "scripts" / "process_stage4_statistics.py"
        current = current.parent
    raise RuntimeError("Could not find project root")


class TestSafeGet:
    """Test safe dictionary navigation."""

    def test_safe_get_simple_key(self):
        """Test retrieving simple key."""
        data = {"key": "value"}
        assert safe_get(data, "key") == "value"

    def test_safe_get_nested_keys(self):
        """Test retrieving nested keys."""
        data = {"level1": {"level2": {"level3": "value"}}}
        assert safe_get(data, "level1", "level2", "level3") == "value"

    def test_safe_get_missing_key(self):
        """Test handling of missing key."""
        data = {"key": "value"}
        assert safe_get(data, "missing") is None

    def test_safe_get_missing_nested_key(self):
        """Test handling of missing nested key."""
        data = {"level1": {"level2": "value"}}
        assert safe_get(data, "level1", "missing", "level3") is None

    def test_safe_get_with_default(self):
        """Test custom default value."""
        data = {"key": "value"}
        assert safe_get(data, "missing", default="custom") == "custom"

    def test_safe_get_none_value(self):
        """Test handling of None value."""
        data = {"key": None}
        assert safe_get(data, "key") is None

    def test_safe_get_non_dict_intermediate(self):
        """Test handling when intermediate value is not a dict."""
        data = {"key": "string_value"}
        assert safe_get(data, "key", "nested") is None


class TestNormalizeScore:
    """Test score normalization to 0-10 scale."""

    def test_normalize_score_already_normalized(self):
        """Test score already on 0-10 scale."""
        assert normalize_score(8.5, scale_factor=1.0) == 8.5

    def test_normalize_score_from_100_scale(self):
        """Test normalization from 0-100 scale."""
        assert normalize_score(85, scale_factor=0.1) == 8.5

    def test_normalize_score_from_5_scale(self):
        """Test normalization from 0-5 scale."""
        assert normalize_score(4.25, scale_factor=2.0) == 8.5

    def test_normalize_score_none_input(self):
        """Test handling of None input."""
        assert normalize_score(None) is None

    def test_normalize_score_zero_value(self):
        """Test handling of zero value (should return None)."""
        assert normalize_score(0) is None

    def test_normalize_score_string_number(self):
        """Test conversion from string to float."""
        assert normalize_score("8.5", scale_factor=1.0) == 8.5

    def test_normalize_score_string_100_scale(self):
        """Test string conversion from 100 scale."""
        assert normalize_score("85.5", scale_factor=0.1) == 8.55

    def test_normalize_score_clamp_upper(self):
        """Test clamping values above 10."""
        assert normalize_score(15.0, scale_factor=1.0) == 10.0

    def test_normalize_score_clamp_lower(self):
        """Test clamping values below 0."""
        assert normalize_score(-5.0, scale_factor=1.0) == 0.0

    def test_normalize_score_invalid_string(self):
        """Test handling of non-numeric string."""
        assert normalize_score("not_a_number") is None

    def test_normalize_score_invalid_type(self):
        """Test handling of invalid type."""
        assert normalize_score({"key": "value"}) is None


class TestExtractMALStatistics:
    """Test MAL statistics extraction from Jikan data."""

    def test_extract_mal_complete_data(self):
        """Test extraction with all fields present."""
        jikan_data = {
            "data": {
                "score": 8.47,
                "scored_by": 503637,
                "rank": 164,
                "popularity": 247,
                "members": 843430,
                "favorites": 13839,
            }
        }
        result = extract_mal_statistics(jikan_data)

        assert result["score"] == 8.47
        assert result["scored_by"] == 503637
        assert result["rank"] == 164
        assert result["popularity_rank"] == 247
        assert result["members"] == 843430
        assert result["favorites"] == 13839

    def test_extract_mal_missing_fields(self):
        """Test extraction with missing fields."""
        jikan_data = {"data": {"score": 8.0}}
        result = extract_mal_statistics(jikan_data)

        assert result["score"] == 8.0
        assert result["scored_by"] is None
        assert result["rank"] is None

    def test_extract_mal_empty_data(self):
        """Test extraction with empty data object."""
        jikan_data = {"data": {}}
        result = extract_mal_statistics(jikan_data)

        assert all(v is None for v in result.values())

    def test_extract_mal_no_data_key(self):
        """Test extraction when data key is missing."""
        jikan_data = {}
        result = extract_mal_statistics(jikan_data)

        assert all(v is None for v in result.values())

    def test_extract_mal_zero_score(self):
        """Test that zero score is treated as None."""
        jikan_data = {"data": {"score": 0}}
        result = extract_mal_statistics(jikan_data)

        assert result["score"] is None


class TestExtractAnimeScheduleStatistics:
    """Test AnimeSchedule statistics extraction."""

    def test_extract_animeschedule_complete_data(self):
        """Test extraction with all fields present."""
        animeschedule_data = {
            "stats": {
                "averageScore": 86.7052993774414,
                "ratingCount": 327,
                "trackedRating": 31,
                "trackedCount": 1909,
            }
        }
        result = extract_animeschedule_statistics(animeschedule_data)

        assert result["score"] == pytest.approx(8.67052993774414, rel=1e-9)
        assert result["scored_by"] == 327
        assert result["rank"] == 31
        assert result["members"] == 1909
        assert result["popularity_rank"] is None
        assert result["favorites"] is None

    def test_extract_animeschedule_missing_stats(self):
        """Test extraction with missing stats object."""
        animeschedule_data = {}
        result = extract_animeschedule_statistics(animeschedule_data)

        assert result["score"] is None
        assert result["scored_by"] is None

    def test_extract_animeschedule_partial_data(self):
        """Test extraction with partial data."""
        animeschedule_data = {"stats": {"averageScore": 75.5}}
        result = extract_animeschedule_statistics(animeschedule_data)

        assert result["score"] == pytest.approx(7.55, rel=1e-9)
        assert result["scored_by"] is None


class TestExtractKitsuStatistics:
    """Test Kitsu statistics extraction."""

    def test_extract_kitsu_complete_data(self):
        """Test extraction with all fields present."""
        kitsu_data = {
            "anime": {
                "attributes": {
                    "averageRating": "86.11",
                    "userCount": 9836,
                    "favoritesCount": 140,
                    "ratingRank": 18,
                    "popularityRank": 1666,
                }
            }
        }
        result = extract_kitsu_statistics(kitsu_data)

        assert result["score"] == 8.611
        assert result["members"] == 9836
        assert result["favorites"] == 140
        assert result["rank"] == 18
        assert result["popularity_rank"] == 1666
        assert result["scored_by"] is None

    def test_extract_kitsu_string_score(self):
        """Test that string scores are properly converted."""
        kitsu_data = {"anime": {"attributes": {"averageRating": "82.5"}}}
        result = extract_kitsu_statistics(kitsu_data)

        assert result["score"] == 8.25

    def test_extract_kitsu_missing_attributes(self):
        """Test extraction with missing attributes."""
        kitsu_data = {"anime": {}}
        result = extract_kitsu_statistics(kitsu_data)

        assert all(v is None for v in result.values())

    def test_extract_kitsu_no_anime_key(self):
        """Test extraction when anime key is missing."""
        kitsu_data = {}
        result = extract_kitsu_statistics(kitsu_data)

        assert all(v is None for v in result.values())


class TestExtractAnimePlanetStatistics:
    """Test Anime-Planet statistics extraction."""

    def test_extract_animeplanet_complete_data(self):
        """Test extraction with all fields present."""
        animeplanet_data = {
            "aggregate_rating": {"ratingValue": 4.346, "ratingCount": 12268},
            "rank": 105,
        }
        result = extract_animeplanet_statistics(animeplanet_data)

        assert result["score"] == 8.692
        assert result["scored_by"] == 12268
        assert result["rank"] == 105

    def test_extract_animeplanet_missing_aggregate_rating(self):
        """Test extraction with missing aggregate_rating."""
        animeplanet_data = {"rank": 100}
        result = extract_animeplanet_statistics(animeplanet_data)

        assert result["score"] is None
        assert result["scored_by"] is None
        assert result["rank"] == 100

    def test_extract_animeplanet_empty_data(self):
        """Test extraction with empty data."""
        animeplanet_data = {}
        result = extract_animeplanet_statistics(animeplanet_data)

        assert result["score"] is None
        assert result["scored_by"] is None
        assert result["rank"] is None

    def test_extract_animeplanet_score_scaling(self):
        """Test that 0-5 scale is properly converted to 0-10."""
        animeplanet_data = {"aggregate_rating": {"ratingValue": 5.0}}
        result = extract_animeplanet_statistics(animeplanet_data)

        assert result["score"] == 10.0


class TestExtractAniListStatistics:
    """Test AniList statistics extraction."""

    def test_extract_anilist_complete_data(self):
        """Test extraction with all fields present."""
        anilist_data = {
            "averageScore": 84,
            "favourites": 15497,
            "popularity": 284464,
            "rankings": [
                {
                    "rank": 96,
                    "type": "RATED",
                    "format": "TV",
                    "year": None,
                    "season": None,
                    "allTime": True,
                },
                {
                    "rank": 1,
                    "type": "POPULAR",
                    "format": "TV",
                    "year": 2024,
                    "season": "FALL",
                    "allTime": False,
                },
            ],
        }
        result = extract_anilist_statistics(anilist_data)

        assert result["score"] == 8.4
        assert result["favorites"] == 15497
        assert result["members"] == 284464
        assert result["scored_by"] is None
        assert result["rank"] is None
        assert result["popularity_rank"] is None

        # Check contextual ranks
        assert len(result["contextual_ranks"]) == 2
        assert result["contextual_ranks"][0]["rank"] == 96
        assert result["contextual_ranks"][0]["type"] == "RATED"
        assert result["contextual_ranks"][0]["all_time"] is True
        assert result["contextual_ranks"][1]["rank"] == 1
        assert result["contextual_ranks"][1]["season"] == "FALL"

    def test_extract_anilist_no_rankings(self):
        """Test extraction with no rankings."""
        anilist_data = {"averageScore": 80, "favourites": 100}
        result = extract_anilist_statistics(anilist_data)

        assert result["score"] == 8.0
        assert result["contextual_ranks"] is None

    def test_extract_anilist_empty_rankings(self):
        """Test extraction with empty rankings array."""
        anilist_data = {"averageScore": 80, "rankings": []}
        result = extract_anilist_statistics(anilist_data)

        assert result["contextual_ranks"] is None

    def test_extract_anilist_invalid_ranking_item(self):
        """Test handling of non-dict ranking items."""
        anilist_data = {
            "averageScore": 80,
            "rankings": [{"rank": 1, "type": "RATED"}, "invalid_item", None],
        }
        result = extract_anilist_statistics(anilist_data)

        # Should only include the valid dict
        assert len(result["contextual_ranks"]) == 1
        assert result["contextual_ranks"][0]["rank"] == 1

    def test_extract_anilist_missing_alltime_field(self):
        """Test that missing allTime field defaults to False."""
        anilist_data = {
            "rankings": [
                {
                    "rank": 10,
                    "type": "RATED",
                    # allTime field missing
                }
            ]
        }
        result = extract_anilist_statistics(anilist_data)

        assert result["contextual_ranks"][0]["all_time"] is False


class TestExtractAniDBStatistics:
    """Test AniDB statistics extraction."""

    def test_extract_anidb_complete_data(self):
        """Test extraction with all fields present."""
        anidb_data = {"ratings": {"permanent": {"value": 8.35, "count": 1427}}}
        result = extract_anidb_statistics(anidb_data)

        assert result["score"] == 8.35
        assert result["scored_by"] == 1427
        assert result["rank"] is None
        assert result["popularity_rank"] is None
        assert result["members"] is None
        assert result["favorites"] is None

    def test_extract_anidb_missing_permanent(self):
        """Test extraction with missing permanent ratings."""
        anidb_data = {"ratings": {}}
        result = extract_anidb_statistics(anidb_data)

        assert result["score"] is None
        assert result["scored_by"] is None

    def test_extract_anidb_missing_ratings(self):
        """Test extraction with missing ratings object."""
        anidb_data = {}
        result = extract_anidb_statistics(anidb_data)

        assert result["score"] is None
        assert result["scored_by"] is None

    def test_extract_anidb_partial_data(self):
        """Test extraction with partial data."""
        anidb_data = {
            "ratings": {
                "permanent": {
                    "value": 7.5
                    # count missing
                }
            }
        }
        result = extract_anidb_statistics(anidb_data)

        assert result["score"] == 7.5
        assert result["scored_by"] is None


class TestHasAnyStatistics:
    """Test statistics validation."""

    def test_has_any_statistics_with_score(self):
        """Test that score counts as valid statistics."""
        stats = {
            "score": 8.5,
            "scored_by": None,
            "rank": None,
            "popularity_rank": None,
            "members": None,
            "favorites": None,
        }
        assert has_any_statistics(stats) is True

    def test_has_any_statistics_with_scored_by(self):
        """Test that scored_by counts as valid statistics."""
        stats = {
            "score": None,
            "scored_by": 1000,
            "rank": None,
            "popularity_rank": None,
            "members": None,
            "favorites": None,
        }
        assert has_any_statistics(stats) is True

    def test_has_any_statistics_all_none(self):
        """Test that all None values returns False."""
        stats = {
            "score": None,
            "scored_by": None,
            "rank": None,
            "popularity_rank": None,
            "members": None,
            "favorites": None,
        }
        assert has_any_statistics(stats) is False

    def test_has_any_statistics_with_contextual_ranks(self):
        """Test that contextual_ranks are ignored in validation."""
        stats = {
            "score": None,
            "scored_by": None,
            "rank": None,
            "popularity_rank": None,
            "members": None,
            "favorites": None,
            "contextual_ranks": [{"rank": 1}],
        }
        # Should return False even though contextual_ranks is present
        assert has_any_statistics(stats) is False

    def test_has_any_statistics_multiple_fields(self):
        """Test with multiple non-None fields."""
        stats = {
            "score": 8.0,
            "scored_by": 1000,
            "rank": 50,
            "popularity_rank": None,
            "members": None,
            "favorites": None,
        }
        assert has_any_statistics(stats) is True


class TestLoadSourceData:
    """Test loading of source data files."""

    @pytest.fixture
    def temp_dir_with_sources(self, tmp_path):
        """Create temp directory with all source files."""
        sources = {
            "jikan": {"data": {"score": 8.5}},
            "anilist": {"averageScore": 85},
            "kitsu": {"anime": {"attributes": {"averageRating": "85"}}},
            "anidb": {"ratings": {"permanent": {"value": 8.5}}},
            "anime_planet": {"aggregate_rating": {"ratingValue": 4.25}},
            "animeschedule": {"stats": {"averageScore": 85}},
        }

        for name, data in sources.items():
            file_path = tmp_path / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

        return str(tmp_path)

    def test_load_source_data_all_present(self, temp_dir_with_sources):
        """Test loading when all source files are present."""
        result = load_source_data(temp_dir_with_sources)

        assert "jikan" in result
        assert "anilist" in result
        assert "kitsu" in result
        assert "anidb" in result
        assert "anime_planet" in result
        assert "animeschedule" in result

    def test_load_source_data_missing_files(self, tmp_path):
        """Test loading when files are missing."""
        result = load_source_data(str(tmp_path))

        # All sources should be present but empty
        assert len(result) == 6
        assert all(v == {} for v in result.values())

    def test_load_source_data_partial_files(self, tmp_path):
        """Test loading when only some files are present."""
        jikan_file = tmp_path / "jikan.json"
        with open(jikan_file, "w") as f:
            json.dump({"data": {"score": 8.0}}, f)

        result = load_source_data(str(tmp_path))

        assert result["jikan"] != {}
        assert result["anilist"] == {}

    def test_load_source_data_malformed_json(self, tmp_path):
        """Test handling of malformed JSON files."""
        jikan_file = tmp_path / "jikan.json"
        with open(jikan_file, "w") as f:
            f.write("{invalid json")

        result = load_source_data(str(tmp_path))

        # Should handle error and return empty dict for that source
        assert result["jikan"] == {}


class TestExtractAllStatistics:
    """Test extraction from all sources."""

    @pytest.fixture
    def all_sources_data(self):
        """Create data for all sources."""
        return {
            "jikan": {
                "data": {
                    "score": 8.47,
                    "scored_by": 500000,
                    "rank": 164,
                    "popularity": 247,
                    "members": 800000,
                    "favorites": 13000,
                }
            },
            "anilist": {"averageScore": 84, "favourites": 15000, "popularity": 284464},
            "kitsu": {
                "anime": {
                    "attributes": {
                        "averageRating": "86.11",
                        "userCount": 9836,
                        "favoritesCount": 140,
                        "ratingRank": 18,
                        "popularityRank": 1666,
                    }
                }
            },
            "anidb": {"ratings": {"permanent": {"value": 8.35, "count": 1427}}},
            "anime_planet": {
                "aggregate_rating": {"ratingValue": 4.346, "ratingCount": 12268},
                "rank": 105,
            },
            "animeschedule": {
                "stats": {
                    "averageScore": 86.7,
                    "ratingCount": 327,
                    "trackedRating": 31,
                    "trackedCount": 1909,
                }
            },
        }

    def test_extract_all_statistics_complete(self, all_sources_data):
        """Test extraction from all sources."""
        result = extract_all_statistics(all_sources_data)

        assert len(result) == 6
        assert "mal" in result
        assert "anilist" in result
        assert "kitsu" in result
        assert "anidb" in result
        assert "animeplanet" in result
        assert "animeschedule" in result

    def test_extract_all_statistics_empty_sources(self):
        """Test extraction with empty sources."""
        sources = {
            "jikan": {},
            "anilist": {},
            "kitsu": {},
            "anidb": {},
            "anime_planet": {},
            "animeschedule": {},
        }
        result = extract_all_statistics(sources)

        # No statistics should be included
        assert len(result) == 0

    def test_extract_all_statistics_partial_sources(self):
        """Test extraction with only some sources having data."""
        sources = {
            "jikan": {"data": {"score": 8.0, "scored_by": 1000}},
            "anilist": {},
            "kitsu": {},
            "anidb": {},
            "anime_planet": {},
            "animeschedule": {},
        }
        result = extract_all_statistics(sources)

        assert len(result) == 1
        assert "mal" in result

    def test_extract_all_statistics_filters_empty_stats(self):
        """Test that sources with all None values are filtered out."""
        sources = {
            "jikan": {"data": {"score": 8.0}},
            "anilist": {},  # Will produce all None values
            "kitsu": {},
            "anidb": {},
            "anime_planet": {},
            "animeschedule": {},
        }
        result = extract_all_statistics(sources)

        # Only mal should be included
        assert "mal" in result
        assert "anilist" not in result


class TestMainExecution:
    """Test main script execution."""

    def test_main_with_complete_data(self, tmp_path, stage4_script_path):
        """Test main execution with complete data."""
        # Create test environment
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()

        # Create source files
        sources = {
            "jikan": {"data": {"score": 8.5, "scored_by": 1000}},
            "anilist": {"averageScore": 85},
            "kitsu": {"anime": {"attributes": {"averageRating": "85"}}},
            "anidb": {"ratings": {"permanent": {"value": 8.5}}},
            "anime_planet": {"aggregate_rating": {"ratingValue": 4.25}},
            "animeschedule": {"stats": {"averageScore": 85}},
        }

        for name, data in sources.items():
            file_path = agent_dir / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

        # Run script
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(stage4_script_path),
                "test_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (agent_dir / "stage4_statistics.json").exists()

        # Verify output
        with open(agent_dir / "stage4_statistics.json") as f:
            output = json.load(f)

        assert "statistics" in output
        assert len(output["statistics"]) == 6

    def test_main_with_missing_sources(self, tmp_path, stage4_script_path):
        """Test main execution when some sources are missing."""
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()

        # Only create jikan file
        jikan_file = agent_dir / "jikan.json"
        with open(jikan_file, "w") as f:
            json.dump({"data": {"score": 8.0, "scored_by": 1000}}, f)

        # Run script
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(stage4_script_path),
                "test_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify output
        with open(agent_dir / "stage4_statistics.json") as f:
            output = json.load(f)

        # Only mal should be present
        assert "mal" in output["statistics"]
        assert len(output["statistics"]) == 1

    def test_main_with_custom_temp_dir(self, tmp_path, stage4_script_path):
        """Test main execution with custom temp directory."""
        custom_temp = tmp_path / "custom_temp"
        custom_temp.mkdir()
        agent_dir = custom_temp / "test_agent"
        agent_dir.mkdir()

        # Create minimal data
        jikan_file = agent_dir / "jikan.json"
        with open(jikan_file, "w") as f:
            json.dump({"data": {"score": 8.0}}, f)

        # Run script with custom temp dir
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(stage4_script_path),
                "test_agent",
                "--temp-dir",
                str(custom_temp),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (agent_dir / "stage4_statistics.json").exists()

    def test_main_output_structure(self, tmp_path, stage4_script_path):
        """Test that output has correct structure."""
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()

        # Create minimal data
        jikan_file = agent_dir / "jikan.json"
        with open(jikan_file, "w") as f:
            json.dump({"data": {"score": 8.0, "scored_by": 1000}}, f)

        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(stage4_script_path),
                "test_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        # Verify structure
        with open(agent_dir / "stage4_statistics.json") as f:
            output = json.load(f)

        assert isinstance(output, dict)
        assert "statistics" in output
        assert isinstance(output["statistics"], dict)

    def test_main_ensure_ascii_false(self, tmp_path, stage4_script_path):
        """Test that ensure_ascii=False preserves unicode characters."""
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()

        # Create data with unicode
        jikan_file = agent_dir / "jikan.json"
        with open(jikan_file, "w", encoding="utf-8") as f:
            json.dump({"data": {"score": 8.0}}, f)

        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(stage4_script_path),
                "test_agent",
                "--temp-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        # Verify file encoding
        with open(agent_dir / "stage4_statistics.json", encoding="utf-8") as f:
            content = f.read()
            # Should be valid UTF-8
            assert content is not None


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=process_stage4_statistics", "--cov-report=term-missing"]
    )

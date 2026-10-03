import json
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pytest
from process_stage2_episodes import (
    auto_detect_temp_dir,
    convert_jst_to_utc,
    load_anisearch_episode_data,
    load_kitsu_episode_data,
    process_all_episodes,
)


class TestTimezoneConversion:
    def test_convert_jst_to_utc_midnight_jst_returns_previous_day_utc(self):
        jst_time = "1999-10-20T00:00:00+09:00"
        expected = "1999-10-19T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_morning_jst_returns_same_day_utc(self):
        jst_time = "2024-10-04T09:30:00+09:00"
        expected = "2024-10-04T00:30:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_milliseconds_are_dropped(self):
        jst_time = "2024-10-04T09:30:00.123+09:00"
        expected = "2024-10-04T00:30:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_none_returns_none(self):
        assert convert_jst_to_utc(None) is None

    def test_convert_jst_to_utc_empty_string_returns_none(self):
        assert convert_jst_to_utc("") is None

    def test_convert_jst_to_utc_new_year_midnight_returns_previous_year(self):
        jst_time = "2024-01-01T00:00:00+09:00"
        expected = "2023-12-31T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_leap_day_returns_february_28(self):
        jst_time = "2024-02-29T00:00:00+09:00"
        expected = "2024-02-28T15:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_new_year_morning_returns_previous_year(self):
        jst_time = "2024-01-01T08:00:00+09:00"
        expected = "2023-12-31T23:00:00Z"
        assert convert_jst_to_utc(jst_time) == expected

    def test_convert_jst_to_utc_invalid_format_returns_input(self):
        invalid_time = "not-a-datetime"
        result = convert_jst_to_utc(invalid_time)
        assert result == invalid_time

    def test_convert_jst_to_utc_date_only_returns_utc_on_same_date(self):
        partial_time = "2024-10-04"
        result = convert_jst_to_utc(partial_time)
        assert result is not None and result.startswith("2024-10-04")


class TestKitsuDataLoading:
    @pytest.fixture
    def kitsu_data(self):
        return {
            "anime": {"attributes": {"slug": "one-piece"}},
            "episodes": [
                {
                    "attributes": {
                        "number": 1,
                        "thumbnail": {"original": "https://example.com/thumb1.jpg"},
                        "description": "  Episode 1 description  ",
                        "synopsis": "  Episode 1 synopsis  ",
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
                        "thumbnail": {},
                        "description": "   ",
                        "synopsis": "",
                        "seasonNumber": 0,
                        "titles": {},
                    }
                },
                {"attributes": {"titles": {"en": "No Episode Number"}}},
            ],
        }

    @pytest.fixture
    def temp_dir_with_kitsu(self, kitsu_data, tmp_path):
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(kitsu_data, f)
        return str(tmp_path)

    def test_load_kitsu_episode_data_returns_fields_per_episode(
        self, temp_dir_with_kitsu
    ):
        result = load_kitsu_episode_data(temp_dir_with_kitsu)

        (
            thumbnails,
            descriptions,
            synopses,
            titles,
            _titles_jp,
            _titles_romaji,
            season_nums,
            episode_urls,
        ) = result

        assert len(thumbnails) == 2
        assert thumbnails[1] == "https://example.com/thumb1.jpg"
        assert thumbnails[2] == "https://example.com/thumb2.jpg"

        assert len(descriptions) == 1
        assert descriptions[1] == "Episode 1 description"

        assert len(synopses) == 1
        assert synopses[1] == "Episode 1 synopsis"

        assert len(titles) == 2
        assert titles[1] == "English Title"
        assert titles[2] == "US Only Title"

        assert len(season_nums) == 2
        assert season_nums[1] == 1
        assert season_nums[3] == 0

        assert len(episode_urls) == 3
        assert episode_urls[1] == "https://kitsu.app/anime/one-piece/episodes/1"

    def test_load_kitsu_episode_data_missing_file_returns_empty_maps(self, tmp_path):
        result = load_kitsu_episode_data(str(tmp_path))

        assert len(result) == 8
        assert all(r == {} for r in result)

    def test_load_kitsu_episode_data_en_title_preferred_over_en_us(
        self, temp_dir_with_kitsu
    ):
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        titles = result[3]

        assert titles[1] == "English Title"

        assert titles[2] == "US Only Title"

    def test_load_kitsu_episode_data_builds_episode_urls_from_slug(
        self, temp_dir_with_kitsu
    ):
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        episode_urls = result[7]

        assert episode_urls[1] == "https://kitsu.app/anime/one-piece/episodes/1"
        assert episode_urls[2] == "https://kitsu.app/anime/one-piece/episodes/2"

    def test_load_kitsu_episode_data_no_slug_returns_no_urls(self, tmp_path):
        data = {
            "anime": {"attributes": {}},
            "episodes": [{"attributes": {"number": 1, "titles": {"en": "Test"}}}],
        }

        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(data, f)

        result = load_kitsu_episode_data(str(tmp_path))
        episode_urls = result[7]

        assert len(episode_urls) == 0

    def test_load_kitsu_episode_data_returns_japanese_and_romaji_titles(
        self, temp_dir_with_kitsu
    ):
        result = load_kitsu_episode_data(temp_dir_with_kitsu)
        titles_jp = result[4]
        titles_romaji = result[5]

        assert titles_jp[1] == "日本語タイトル"
        assert titles_romaji[1] == "Romaji Title"

    def test_load_kitsu_episode_data_invalid_json_returns_empty_maps(self, tmp_path):
        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            f.write("{invalid json")

        result = load_kitsu_episode_data(str(tmp_path))
        assert len(result) == 8
        assert all(r == {} for r in result)

    def test_load_kitsu_episode_data_empty_thumbnail_returns_no_thumbnails(
        self, tmp_path
    ):
        data = {
            "anime": {"attributes": {"slug": "test"}},
            "episodes": [
                {"attributes": {"number": 1, "thumbnail": {}}},
                {
                    "attributes": {
                        "number": 2,
                        "thumbnail": {"original": None},
                    }
                },
            ],
        }

        kitsu_file = tmp_path / "kitsu.json"
        with open(kitsu_file, "w") as f:
            json.dump(data, f)

        result = load_kitsu_episode_data(str(tmp_path))
        thumbnails = result[0]

        assert len(thumbnails) == 0


class TestAniSearchDataLoading:
    @pytest.fixture
    def anisearch_data(self):
        return {
            "episodes": [
                {"episodeNumber": 1, "title": "AniSearch Episode 1"},
                {"episodeNumber": 2, "title": "AniSearch Episode 2"},
                {"episodeNumber": 3, "title": ""},
                {
                    "episodeNumber": None,
                    "title": "No Episode Number",
                },
                {"title": "Missing Episode Number"},
            ]
        }

    @pytest.fixture
    def temp_dir_with_anisearch(self, anisearch_data, tmp_path):
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(anisearch_data, f)
        return str(tmp_path)

    def test_load_anisearch_episode_data_returns_titles_with_episode_number(
        self, temp_dir_with_anisearch
    ):
        titles = load_anisearch_episode_data(temp_dir_with_anisearch)

        assert len(titles) == 2
        assert titles[1] == "AniSearch Episode 1"
        assert titles[2] == "AniSearch Episode 2"

    def test_load_anisearch_episode_data_missing_file_returns_empty(self, tmp_path):
        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_empty_episodes_returns_empty(self, tmp_path):
        data = {"episodes": []}

        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(data, f)

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_invalid_json_returns_empty(self, tmp_path):
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            f.write("{malformed json")

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}

    def test_load_anisearch_episode_data_missing_episodes_key_returns_empty(
        self, tmp_path
    ):
        data = {"other_key": "value"}

        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(data, f)

        titles = load_anisearch_episode_data(str(tmp_path))
        assert titles == {}


class TestEpisodeProcessing:
    @pytest.fixture
    def mal_episodes(self):
        return [
            {
                "episode_number": 1,
                "title": "MAL Title 1",
                "title_japanese": "MAL Japanese 1",
                "title_romaji": "MAL Romaji 1",
                "synopsis": "MAL synopsis",
                "aired": "1999-10-20T00:00:00+09:00",
                "duration": 1440,
                "score": 8.5,
                "filler": False,
                "recap": False,
                "url": "https://myanimelist.net/anime/21/One_Piece/episode/1",
            },
            {
                "episode_number": 2,
                "title": None,
                "title_japanese": None,
                "title_romaji": None,
                "synopsis": None,
                "aired": "1999-10-27T00:00:00+09:00",
                "duration": 1440,
                "score": None,
                "filler": True,
                "recap": False,
                "url": "https://myanimelist.net/anime/21/One_Piece/episode/2",
            },
            {
                "episode_number": 3,
                "title": None,
                "synopsis": None,
                "aired": "1999-11-03T00:00:00+09:00",
                "duration": 1440,
                "filler": False,
                "recap": True,
            },
        ]

    @pytest.fixture
    def agent_dir_with_all_sources(self, mal_episodes, tmp_path):
        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(mal_episodes, f)

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

        anisearch_data = {
            "episodes": [{"episodeNumber": 3, "title": "AniSearch Title 3"}]
        }
        anisearch_file = tmp_path / "anisearch.json"
        with open(anisearch_file, "w") as f:
            json.dump(anisearch_data, f)

        return str(tmp_path)

    def test_process_all_episodes_all_sources_merges_each_episode(
        self, agent_dir_with_all_sources
    ):
        process_all_episodes(agent_dir_with_all_sources)

        output_file = Path(agent_dir_with_all_sources) / "stage2_episodes.json"
        assert output_file.exists()

        with open(output_file) as f:
            output = json.load(f)

        assert "episodes" in output
        episodes = output["episodes"]
        assert len(episodes) == 3

        ep1 = episodes[0]
        assert ep1["episode_number"] == 1
        assert ep1["title"] == "MAL Title 1"
        assert ep1["title_japanese"] == "MAL Japanese 1"
        assert ep1["title_romaji"] == "MAL Romaji 1"
        assert ep1["synopsis"] == "MAL synopsis"
        assert ep1["aired"] == "1999-10-19T15:00:00Z"
        assert ep1["duration"] == 1440
        assert ep1["score"] == 8.5
        assert ep1["filler"] is False
        assert ep1["recap"] is False
        assert (
            ep1["episode_pages"]["mal"]
            == "https://myanimelist.net/anime/21/One_Piece/episode/1"
        )
        assert ep1["streaming"] == {}

        ep2 = episodes[1]
        assert ep2["episode_number"] == 2
        assert ep2["title"] == "Kitsu Title 2"
        assert ep2["title_japanese"] == "Kitsu Japanese 2"
        assert ep2["title_romaji"] == "Kitsu Romaji 2"
        assert ep2["synopsis"] == "Kitsu synopsis 2"
        assert ep2["description"] == "Kitsu description 2"
        assert ep2["season_number"] == 1
        assert ep2["aired"] == "1999-10-26T15:00:00Z"
        assert ep2["filler"] is True
        assert ep2["thumbnails"] == ["https://kitsu.io/thumb2.jpg"]
        assert "kitsu" in ep2["episode_pages"]
        assert (
            ep2["episode_pages"]["kitsu"]
            == "https://kitsu.app/anime/one-piece/episodes/2"
        )

        ep3 = episodes[2]
        assert ep3["episode_number"] == 3
        assert ep3["title"] == "AniSearch Title 3"
        assert ep3["aired"] == "1999-11-02T15:00:00Z"
        assert ep3["recap"] is True
        assert ep3["episode_pages"] == {}

    def test_process_all_episodes_title_falls_back_from_mal_to_kitsu_to_anisearch(
        self, agent_dir_with_all_sources
    ):
        process_all_episodes(agent_dir_with_all_sources)

        output_file = Path(agent_dir_with_all_sources) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episodes"]

        assert episodes[0]["title"] == "MAL Title 1"

        assert episodes[1]["title"] == "Kitsu Title 2"

        assert episodes[2]["title"] == "AniSearch Title 3"

    def test_process_all_episodes_episode_pages_list_known_urls(
        self, agent_dir_with_all_sources
    ):
        process_all_episodes(agent_dir_with_all_sources)

        output_file = Path(agent_dir_with_all_sources) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episodes"]

        assert "mal" in episodes[0]["episode_pages"]

        assert "mal" in episodes[1]["episode_pages"]
        assert "kitsu" in episodes[1]["episode_pages"]

        assert episodes[2]["episode_pages"] == {}

    def test_process_all_episodes_aired_dates_are_utc(self, agent_dir_with_all_sources):
        process_all_episodes(agent_dir_with_all_sources)

        output_file = Path(agent_dir_with_all_sources) / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        episodes = output["episodes"]

        for episode in episodes:
            aired = episode.get("aired")
            if aired:
                assert aired.endswith("Z")
                assert "+09:00" not in aired

    def test_process_all_episodes_mal_only_writes_mal_fields(self, tmp_path):
        episodes_data = [
            {
                "episode_number": 1,
                "title": "Episode 1",
                "aired": "2024-01-01T00:00:00+09:00",
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        assert output_file.exists()

        with open(output_file) as f:
            output = json.load(f)

        assert len(output["episodes"]) == 1
        ep = output["episodes"][0]
        assert ep["title"] == "Episode 1"
        assert ep["aired"] == "2023-12-31T15:00:00Z"

    def test_process_all_episodes_writes_episodes_list(
        self, agent_dir_with_all_sources
    ):
        process_all_episodes(agent_dir_with_all_sources)

        output_file = Path(agent_dir_with_all_sources) / "stage2_episodes.json"
        assert output_file.exists()

        with open(output_file) as f:
            output = json.load(f)

        assert isinstance(output, dict)
        assert "episodes" in output
        assert isinstance(output["episodes"], list)

    def test_process_all_episodes_four_episodes_writes_all_four(self, tmp_path):
        episodes_data = [
            {
                "episode_number": i,
                "title": f"Ep {i}",
                "aired": f"1999-10-{20 + i}T00:00:00+09:00",
                "filler": False,
                "recap": False,
            }
            for i in range(1, 5)
        ]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        assert len(output["episodes"]) == 4


class TestAutoDetectTempDir:
    def test_auto_detect_temp_dir_single_directory_returns_its_path(
        self, tmp_path, monkeypatch
    ):
        temp_base = tmp_path / "temp"
        temp_base.mkdir()
        anime_dir = temp_base / "One_agent1"
        anime_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        result = auto_detect_temp_dir()
        assert result == "temp/One_agent1"

    def test_auto_detect_temp_dir_multiple_directories_exits_with_error(
        self, tmp_path, monkeypatch
    ):
        temp_base = tmp_path / "temp"
        temp_base.mkdir()
        (temp_base / "One_agent1").mkdir()
        (temp_base / "Naruto_agent1").mkdir()

        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1

    def test_auto_detect_temp_dir_no_temp_directory_exits_with_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1

    def test_auto_detect_temp_dir_empty_temp_directory_exits_with_error(
        self, tmp_path, monkeypatch
    ):
        temp_base = tmp_path / "temp"
        temp_base.mkdir()

        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            auto_detect_temp_dir()
        assert exc_info.value.code == 1


class TestEdgeCases:
    def test_process_all_episodes_missing_mal_episodes_raises_file_not_found(
        self, tmp_path
    ):
        with pytest.raises(FileNotFoundError):
            process_all_episodes(str(tmp_path))

    def test_process_all_episodes_empty_list_writes_no_episodes(self, tmp_path):
        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump([], f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        assert output["episodes"] == []

    def test_process_all_episodes_missing_fields_writes_defaults(self, tmp_path):
        episodes_data = [{"episode_number": 1}]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        ep = output["episodes"][0]
        assert ep["episode_number"] == 1
        assert ep["title"] is None
        assert ep["aired"] is None
        assert ep["filler"] is False
        assert ep["recap"] is False
        assert ep["thumbnails"] == []
        assert ep["episode_pages"] == {}

    def test_process_all_episodes_null_aired_writes_null_aired(self, tmp_path):
        episodes_data = [
            {
                "episode_number": 1,
                "title": "Test",
                "aired": None,
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"
        with open(output_file) as f:
            output = json.load(f)

        ep = output["episodes"][0]
        assert ep["aired"] is None

    def test_process_all_episodes_japanese_title_written_unescaped(self, tmp_path):
        episodes_data = [
            {
                "episode_number": 1,
                "title": "テスト",
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump(episodes_data, f, ensure_ascii=False)

        process_all_episodes(str(tmp_path))

        output_file = tmp_path / "stage2_episodes.json"

        with open(output_file, encoding="utf-8") as f:
            content = f.read()
            assert "テスト" in content

    def test_process_all_episodes_empty_mal_synopsis_falls_back_to_kitsu(
        self, tmp_path
    ):
        episodes_data = [
            {
                "episode_number": 1,
                "synopsis": "",
                "filler": False,
                "recap": False,
            }
        ]

        episodes_file = tmp_path / "mal_episodes.jsonl"
        with open(episodes_file, "w") as f:
            json.dump(episodes_data, f)

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

        ep = output["episodes"][0]
        assert ep["synopsis"] == "Kitsu synopsis"


class TestMainExecution:
    def _run_script(self, monkeypatch, *arguments: str) -> None:
        monkeypatch.setattr(sys, "argv", ["process_stage2_episodes.py", *arguments])
        runpy.run_path(
            str(SCRIPTS_DIR / "process_stage2_episodes.py"), run_name="__main__"
        )

    def test_main_agent_id_with_temp_dir_writes_stage2_file(
        self, tmp_path, monkeypatch
    ):
        anime_dir = tmp_path / "test_agent"
        anime_dir.mkdir()
        (anime_dir / "mal_episodes.jsonl").write_text(
            json.dumps(
                [
                    {
                        "episode_number": 1,
                        "title": "Test",
                        "filler": False,
                        "recap": False,
                    }
                ]
            )
        )

        self._run_script(monkeypatch, "test_agent", "--temp-dir", str(tmp_path))

        assert (anime_dir / "stage2_episodes.json").exists()

    def test_main_missing_agent_directory_exits_with_error(
        self, tmp_path, monkeypatch, capsys
    ):
        with pytest.raises(SystemExit) as exc_info:
            self._run_script(
                monkeypatch, "nonexistent_agent", "--temp-dir", str(tmp_path)
            )

        assert exc_info.value.code == 1
        assert "does not exist" in capsys.readouterr().out

    def test_main_missing_mal_episodes_exits_with_error(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / "test_agent").mkdir()

        with pytest.raises(SystemExit) as exc_info:
            self._run_script(monkeypatch, "test_agent", "--temp-dir", str(tmp_path))

        assert exc_info.value.code == 1
        assert "Required file not found" in capsys.readouterr().out

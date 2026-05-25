"""Unit tests for anisearch_mapper.py — episode_from_anisearch value normalization."""

from enrichment.sources.anisearch.anisearch_anime_models import AniSearchEpisode
from enrichment.sources.anisearch.anisearch_mapper import episode_from_anisearch


def _ep(**kwargs) -> AniSearchEpisode:
    # Model fields are already parsed by the crawler — mapper just maps.
    defaults = {
        "episode_number": 1,
        "is_filler": False,
        "duration": 1440,
        "aired": "1999-10-20",
        "title": "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!",
        "title_romaji": "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!",
        "title_japanese": "俺はルフィ!海賊王になる男だ!",
        "source": "https://www.anisearch.com/anime/2227,one-piece/episodes",
    }
    return AniSearchEpisode(**{**defaults, **kwargs})


def test_episode_number_mapped() -> None:
    assert episode_from_anisearch(_ep())["episode_number"] == 1


def test_filler_false_by_default() -> None:
    assert episode_from_anisearch(_ep())["filler"] is False


def test_filler_true_preserved() -> None:
    assert episode_from_anisearch(_ep(is_filler=True))["filler"] is True


def test_duration_passed_through() -> None:
    assert episode_from_anisearch(_ep(duration=1440))["duration"] == 1440


def test_duration_none_omits_key() -> None:
    assert "duration" not in episode_from_anisearch(_ep(duration=None))


def test_aired_normalized_to_utc_iso() -> None:
    # "1999-10-20" JST midnight → "1999-10-19T15:00:00Z" UTC (Midnight JST rule)
    aired = episode_from_anisearch(_ep()).get("aired")
    assert aired is not None
    assert "1999-10-19" in aired


def test_aired_none_omits_key() -> None:
    assert "aired" not in episode_from_anisearch(_ep(aired=None))


def test_title_romaji_and_japanese_passed_through() -> None:
    result = episode_from_anisearch(_ep())
    assert result["title_romaji"] == "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!"
    assert result["title_japanese"] == "俺はルフィ!海賊王になる男だ!"


def test_title_japanese_none_omits_key() -> None:
    assert "title_japanese" not in episode_from_anisearch(_ep(title_japanese=None))


def test_title_none_falls_back_to_episode_n() -> None:
    assert episode_from_anisearch(_ep(title=None))["title"] == "Episode 1"


def test_source_url_in_sources() -> None:
    assert (
        "https://www.anisearch.com/anime/2227,one-piece/episodes"
        in episode_from_anisearch(_ep())["sources"]
    )


def test_source_none_omits_sources() -> None:
    result = episode_from_anisearch(_ep(source=None))
    assert "sources" not in result or result.get("sources") == []


def test_anime_id_absent_from_mapped_output() -> None:
    # anime_id is a UUID assigned during assembly, not available at crawl time
    assert "anime_id" not in episode_from_anisearch(_ep())

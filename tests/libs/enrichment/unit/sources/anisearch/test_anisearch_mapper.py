"""Unit tests for anisearch_mapper.py — value normalization during mapping."""

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchEpisode,
    AniSearchStatistics,
)
from enrichment.sources.anisearch.anisearch_mapper import (
    anime_from_anisearch,
    episode_from_anisearch,
)


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


def test_anime_from_anisearch_full_start_date_sets_year_season_and_date() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1999-10-20", start_year=1999)
    )
    assert (mapped["year"], mapped["season"], mapped["aired_dates"]["aired_from"]) == (
        1999,
        "FALL",
        "1999-10-19T15:00:00Z",
    )
    assert "month" not in mapped


def test_anime_from_anisearch_year_only_start_sets_year_without_season_month_or_date() -> (
    None
):
    mapped = anime_from_anisearch(AniSearchAnime(start_year=2027, status="Upcoming"))
    assert mapped["year"] == 2027
    assert not {"season", "month", "aired_dates"} & mapped.keys()


def test_anime_from_anisearch_month_and_year_start_sets_year_and_month() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_year=2008, start_month="November")
    )
    assert (mapped["year"], mapped["month"]) == (2008, "November")
    assert "aired_dates" not in mapped


def test_anime_from_anisearch_undated_takes_anisearch_status() -> None:
    assert (
        anime_from_anisearch(AniSearchAnime(status="Upcoming"))["status"] == "UPCOMING"
    )


def test_anime_from_anisearch_stated_status_wins_over_dates() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1994-04-28", status="Completed")
    )
    assert mapped["status"] == "FINISHED"


def test_anime_from_anisearch_without_stated_status_derives_status_from_dates() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(start_date="1999-10-20"))
    assert mapped["status"] == "ONGOING"


def test_anime_from_anisearch_unrecognised_status_derives_status_from_dates() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1999-10-20", status="Something New")
    )
    assert mapped["status"] == "ONGOING"


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


def test_statistics_score_rescaled_from_five_stars() -> None:
    # The page states "Calculated Value 4.18 = 84%" — 4.18 of 5, so 8.36 of 10.
    stats = anime_from_anisearch(
        AniSearchAnime(
            statistics=AniSearchStatistics(score=4.18, scored_by=7902, rank=124)
        )
    )["statistics"]["anisearch"]
    assert stats["score"] == 8.36
    assert stats["scored_by"] == 7902
    assert stats["rank"] == 124


def test_statistics_omitted_without_values() -> None:
    result = anime_from_anisearch(AniSearchAnime(statistics=AniSearchStatistics()))
    assert "statistics" not in result or not result["statistics"]


def test_studio_becomes_a_company() -> None:
    result = anime_from_anisearch(
        AniSearchAnime(
            studio="Toei Animation Co., Ltd.",
            studio_url="https://www.anisearch.com/company/412,toei-animation-co-ltd",
        )
    )
    assert result["companies"] == [
        {
            "name": "Toei Animation Co., Ltd.",
            "roles": ["STUDIO"],
            "sources": ["https://www.anisearch.com/company/412,toei-animation-co-ltd"],
        }
    ]
    assert "studios" not in result

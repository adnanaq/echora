"""Unit tests for mal_base.py — sidebar parsers, number utils, model diffing."""

import pytest

from enrichment.crawlers.mal_crawler.mal_base import (
    diff_model_lists,
    diff_models,
    extract_id_from_url,
    parse_aired_string,
    parse_broadcast_string,
    parse_duration_seconds,
    parse_iso_date,
    parse_number,
    parse_premiered,
    parse_sidebar_field,
    parse_sidebar_link_texts,
)
from enrichment.crawlers.mal_crawler.mal_models import MalScrapedAnime


# =============================================================================
# parse_number
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2,644,378", 2644378),
        ("#17", 17),
        ("#54", 54),
        ("123", 123),
        ("N/A", None),
        (None, None),
        ("", None),
        ("?", None),
    ],
)
def test_parse_number(raw: str | None, expected: int | None) -> None:
    assert parse_number(raw) == expected


# =============================================================================
# parse_duration_seconds
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("24 min.", 1440),
        ("1 hr. 30 min.", 5400),
        ("00:24:37", 1477),
        ("2 min.", 120),
        ("1 hr.", 3600),
        (None, None),
        ("", None),
        ("Unknown", None),
    ],
)
def test_parse_duration_seconds(raw: str | None, expected: int | None) -> None:
    assert parse_duration_seconds(raw) == expected


# =============================================================================
# parse_iso_date
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Oct 20, 1999", "1999-10-20"),
        ("Apr 5, 2003", "2003-04-05"),
        ("Jan 1, 2000", "2000-01-01"),
        ("?", None),
        ("N/A", None),
        (None, None),
        ("", None),
        ("1999-10-20", "1999-10-20"),  # Already ISO
    ],
)
def test_parse_iso_date(raw: str | None, expected: str | None) -> None:
    assert parse_iso_date(raw) == expected


# =============================================================================
# parse_aired_string
# =============================================================================


def test_parse_aired_string_range() -> None:
    from_d, to_d = parse_aired_string("Oct 20, 1999 to ?")
    assert from_d == "1999-10-20"
    assert to_d is None


def test_parse_aired_string_closed_range() -> None:
    from_d, to_d = parse_aired_string("Oct 20, 1999 to Nov 5, 2000")
    assert from_d == "1999-10-20"
    assert to_d == "2000-11-05"


def test_parse_aired_string_single_date() -> None:
    from_d, to_d = parse_aired_string("Apr 5, 2003")
    assert from_d == "2003-04-05"
    assert to_d is None


def test_parse_aired_string_none() -> None:
    from_d, to_d = parse_aired_string(None)
    assert from_d is None
    assert to_d is None


# =============================================================================
# parse_premiered
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected_season, expected_year",
    [
        ("Fall 1999", "fall", 1999),
        ("Spring 2024", "spring", 2024),
        ("WINTER 2020", "winter", 2020),
        (None, None, None),
        ("", None, None),
    ],
)
def test_parse_premiered(
    raw: str | None, expected_season: str | None, expected_year: int | None
) -> None:
    season, year = parse_premiered(raw)
    assert season == expected_season
    assert year == expected_year


# =============================================================================
# parse_broadcast_string
# =============================================================================


def test_parse_broadcast_string_full() -> None:
    day, time, tz = parse_broadcast_string("Sundays at 23:15 (JST)")
    assert day == "Sundays"
    assert time == "23:15"
    assert tz == "JST"


def test_parse_broadcast_string_unknown() -> None:
    day, time, tz = parse_broadcast_string("Unknown")
    assert day is None
    assert time is None
    assert tz is None


def test_parse_broadcast_string_none() -> None:
    day, time, tz = parse_broadcast_string(None)
    assert day is None
    assert time is None
    assert tz is None


# =============================================================================
# extract_id_from_url
# =============================================================================


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://myanimelist.net/anime/21/One_Piece", 21),
        ("https://myanimelist.net/character/40", 40),
        ("https://myanimelist.net/people/123/Name", 123),
        ("https://myanimelist.net/manga/456", 456),
        ("https://myanimelist.net/producer/3", 3),
        ("https://example.com/something", None),
    ],
)
def test_extract_id_from_url(url: str, expected: int | None) -> None:
    assert extract_id_from_url(url) == expected


# =============================================================================
# parse_sidebar_field
# =============================================================================


_SAMPLE_SIDEBAR = """
<div>
    <span class="dark_text">Episodes:</span>
    1122
    <br>
    <span class="dark_text">Status:</span>
    Currently Airing
    <br>
    <span class="dark_text">Aired:</span>
    Oct 20, 1999 to ?
    <br>
    <span class="dark_text">Ranked:</span>
    #54
    <br>
</div>
"""


def test_parse_sidebar_field_episodes() -> None:
    result = parse_sidebar_field(_SAMPLE_SIDEBAR, "Episodes")
    assert result == "1122"


def test_parse_sidebar_field_status() -> None:
    result = parse_sidebar_field(_SAMPLE_SIDEBAR, "Status")
    assert result == "Currently Airing"


def test_parse_sidebar_field_ranked() -> None:
    result = parse_sidebar_field(_SAMPLE_SIDEBAR, "Ranked")
    assert result == "#54"


def test_parse_sidebar_field_missing() -> None:
    result = parse_sidebar_field(_SAMPLE_SIDEBAR, "Nonexistent")
    assert result is None


# =============================================================================
# diff_models
# =============================================================================


def _make_anime(score: float = 8.7, title: str = "One Piece") -> MalScrapedAnime:
    return MalScrapedAnime(
        anime_id=21,
        url="https://myanimelist.net/anime/21",
        title=title,
        score=score,
    )


def test_diff_models_no_changes() -> None:
    old = _make_anime()
    new = _make_anime()
    diff = diff_models(old, new, "anime")
    assert not diff.has_changes
    assert diff.changes == []


def test_diff_models_changed_field() -> None:
    old = _make_anime(score=8.7)
    new = _make_anime(score=8.8)
    diff = diff_models(old, new, "anime")
    assert diff.has_changes
    assert len(diff.changes) == 1
    assert diff.changes[0].field == "score"
    assert diff.changes[0].old_value == 8.7
    assert diff.changes[0].new_value == 8.8


def test_diff_models_new_entity() -> None:
    new = _make_anime()
    diff = diff_models(None, new, "anime")
    assert diff.is_new
    assert diff.has_changes


def test_diff_model_lists_added_and_removed() -> None:
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedCharacter

    old_chars = [
        MalScrapedCharacter(mal_id=40, url="...", name="Luffy"),
        MalScrapedCharacter(mal_id=41, url="...", name="Zoro"),
    ]
    new_chars = [
        MalScrapedCharacter(mal_id=40, url="...", name="Luffy"),
        MalScrapedCharacter(mal_id=42, url="...", name="Nami"),
    ]
    list_diff = diff_model_lists(old_chars, new_chars, "character")
    assert 42 in list_diff.added
    assert 41 in list_diff.removed
    assert not list_diff.updated  # Luffy unchanged

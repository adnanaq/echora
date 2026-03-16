"""Unit tests for mal_base.py — sidebar parsers, number utils, model diffing."""

import pytest

from enrichment.crawlers.mal_crawler.mal_base import (
    diff_model_lists,
    diff_models,
    extract_id_from_url,
    normalize_mal_anime_url,
    parse_aired_string,
    parse_broadcast_string,
    parse_duration_seconds,
    parse_episode_ranges,
    parse_iso_date,
    parse_number,
    parse_premiered,
    parse_sidebar_field,
    parse_sidebar_link_texts,
    parse_sidebar_links,
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
# normalize_mal_anime_url
# =============================================================================


def test_normalize_mal_anime_url_slugless_no_slug() -> None:
    url, has_slug = normalize_mal_anime_url("https://myanimelist.net/anime/21")
    assert url == "https://myanimelist.net/anime/21"
    assert has_slug is False


def test_normalize_mal_anime_url_slugged_has_slug() -> None:
    url, has_slug = normalize_mal_anime_url("https://myanimelist.net/anime/21/One_Piece")
    assert url == "https://myanimelist.net/anime/21/One_Piece"
    assert has_slug is True


def test_normalize_mal_anime_url_passes_through_unchanged() -> None:
    original = "https://myanimelist.net/anime/57334/Dandadan"
    url, _ = normalize_mal_anime_url(original)
    assert url == original


def test_normalize_mal_anime_url_wrong_base_raises() -> None:
    with pytest.raises(ValueError):
        normalize_mal_anime_url("https://example.com/anime/21")


def test_normalize_mal_anime_url_non_anime_path_raises() -> None:
    with pytest.raises(ValueError):
        normalize_mal_anime_url("https://myanimelist.net/character/40")


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
# parse_sidebar_links / parse_sidebar_link_texts
# =============================================================================


_SAMPLE_SIDEBAR_WITH_LINKS = """
<div>
  <span class="dark_text">Studios:</span>
  <a href="https://myanimelist.net/anime/producer/18/Toei_Animation">Toei Animation</a>,
  <a href="https://myanimelist.net/anime/producer/99/Other_Studio">Other Studio</a>
  <br>
  <span class="dark_text">Genres:</span>
  <a href="https://myanimelist.net/anime/genre/1/Action">Action</a>
  <br>
  <span class="dark_text">Next:</span>
  nothing here
</div>
"""


def test_parse_sidebar_links_extracts_multiple() -> None:
    result = parse_sidebar_links(_SAMPLE_SIDEBAR_WITH_LINKS, "Studios")
    assert len(result) == 2
    assert result[0]["name"] == "Toei Animation"
    assert result[0]["url"] == "https://myanimelist.net/anime/producer/18/Toei_Animation"
    assert result[0]["mal_id"] == 18
    assert result[1]["name"] == "Other Studio"
    assert result[1]["mal_id"] == 99


def test_parse_sidebar_links_single_entry() -> None:
    result = parse_sidebar_links(_SAMPLE_SIDEBAR_WITH_LINKS, "Genres")
    assert len(result) == 1
    assert result[0]["name"] == "Action"
    assert result[0]["url"] == "https://myanimelist.net/anime/genre/1/Action"
    # Genre URLs use /anime/genre/N format — extract_id_from_url only parses
    # /anime/, /character/, /people/, /manga/, /producer/ paths, not /genre/
    assert result[0]["mal_id"] is None


def test_parse_sidebar_links_missing_label_returns_empty() -> None:
    result = parse_sidebar_links(_SAMPLE_SIDEBAR_WITH_LINKS, "Nonexistent")
    assert result == []


def test_parse_sidebar_link_texts_returns_names_only() -> None:
    result = parse_sidebar_link_texts(_SAMPLE_SIDEBAR_WITH_LINKS, "Studios")
    assert result == ["Toei Animation", "Other Studio"]


def test_parse_sidebar_link_texts_genres() -> None:
    result = parse_sidebar_link_texts(_SAMPLE_SIDEBAR_WITH_LINKS, "Genres")
    assert result == ["Action"]


def test_parse_sidebar_link_texts_missing_label_returns_empty() -> None:
    result = parse_sidebar_link_texts(_SAMPLE_SIDEBAR_WITH_LINKS, "Nonexistent")
    assert result == []


# =============================================================================
# parse_episode_ranges
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1-26", [(1, 26)]),
        ("492", [(492, 492)]),
        ("1139-", [(1139, None)]),
        ("1-30, 492, 1139-", [(1, 30), (492, 492), (1139, None)]),
        (None, []),
        ("", []),
        ("(eps 1-13)", [(1, 13)]),
    ],
)
def test_parse_episode_ranges(
    raw: str | None, expected: list[dict]
) -> None:
    assert parse_episode_ranges(raw) == expected


# =============================================================================
# diff_models
# =============================================================================


def _make_anime(score: float = 8.7, title: str = "One Piece") -> MalScrapedAnime:
    return MalScrapedAnime(
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


def test_diff_models_entity_id_correct_for_anime() -> None:
    """diff_models uses url as entity_id for MalScrapedAnime (anime_id field removed)."""
    old = _make_anime(score=8.7)
    new = _make_anime(score=8.8)
    diff = diff_models(old, new, "anime")
    assert diff.entity_id == "https://myanimelist.net/anime/21"


def test_diff_models_entity_id_correct_for_new_anime() -> None:
    """entity_id is the URL even for new (old=None) anime diffs."""
    new = _make_anime()
    diff = diff_models(None, new, "anime")
    assert diff.entity_id == "https://myanimelist.net/anime/21"


# =============================================================================
# diff_model_lists
# =============================================================================


def test_diff_model_lists_added_and_removed() -> None:
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedCharacter

    old_chars = [
        MalScrapedCharacter(url="https://myanimelist.net/character/40/Luffy", name="Luffy"),
        MalScrapedCharacter(url="https://myanimelist.net/character/41/Zoro", name="Zoro"),
    ]
    new_chars = [
        MalScrapedCharacter(url="https://myanimelist.net/character/40/Luffy", name="Luffy"),
        MalScrapedCharacter(url="https://myanimelist.net/character/42/Nami", name="Nami"),
    ]
    list_diff = diff_model_lists(old_chars, new_chars, "character")
    assert "https://myanimelist.net/character/42/Nami" in list_diff.added
    assert "https://myanimelist.net/character/41/Zoro" in list_diff.removed
    assert not list_diff.updated  # Luffy unchanged

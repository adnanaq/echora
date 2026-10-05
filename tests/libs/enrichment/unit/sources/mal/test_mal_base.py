import pytest
from enrichment.sources.mal.mal_base import (
    _get_entity_id,
    diff_model_lists,
    diff_models,
    normalize_mal_anime_url,
    parse_duration_seconds,
    parse_episode_ranges,
    parse_number,
    parse_premiered,
    parse_score,
    parse_sidebar_field,
)
from enrichment.sources.mal.mal_models import MalAnime, MalCharacter, MalEpisode
from pydantic import BaseModel

ONE_PIECE_URL = "https://myanimelist.net/anime/21"
SIDEBAR = """
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


class _ModelWithoutId(BaseModel):
    pass


def _anime(score: float = 8.7) -> MalAnime:
    return MalAnime(source=ONE_PIECE_URL, title="One Piece", score=score)


def _character(number: int, name: str) -> MalCharacter:
    return MalCharacter(source=f"https://myanimelist.net/character/{number}", name=name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2,644,378", 2644378), ("#17", 17), ("#54", 54), ("123", 123)],
)
def test_parse_number_number_text_gives_integer(raw: str, expected: int) -> None:
    assert parse_number(raw) == expected


@pytest.mark.parametrize("raw", ["N/A", None, "", "?"])
def test_parse_number_without_number_returns_none(raw: str | None) -> None:
    assert parse_number(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("8.73", 8.73), ("10", 10.0), (" 6.06 ", 6.06)]
)
def test_parse_score_score_text_gives_float(raw: str, expected: float) -> None:
    assert parse_score(raw) == expected


@pytest.mark.parametrize("raw", ["N/A", None, "", "   "])
def test_parse_score_below_vote_threshold_or_missing_returns_none(
    raw: str | None,
) -> None:
    assert parse_score(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("24 min.", 1440),
        ("1 hr. 30 min.", 5400),
        ("00:24:37", 1477),
        ("2 min.", 120),
        ("1 hr.", 3600),
    ],
)
def test_parse_duration_seconds_duration_text_gives_seconds(
    raw: str, expected: int
) -> None:
    assert parse_duration_seconds(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "Unknown"])
def test_parse_duration_seconds_without_duration_returns_none(
    raw: str | None,
) -> None:
    assert parse_duration_seconds(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fall 1999", ("fall", 1999)),
        ("Spring 2024", ("spring", 2024)),
        ("WINTER 2020", ("winter", 2020)),
    ],
)
def test_parse_premiered_season_and_year_give_lowercase_season_and_year(
    raw: str, expected: tuple[str, int]
) -> None:
    assert parse_premiered(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "Not a season string"])
def test_parse_premiered_without_season_returns_nothing(raw: str | None) -> None:
    assert parse_premiered(raw) == (None, None)


def test_normalize_mal_anime_url_bare_address_reports_no_title() -> None:
    assert normalize_mal_anime_url(ONE_PIECE_URL) == (ONE_PIECE_URL, False)


@pytest.mark.parametrize(
    "url",
    [
        "https://myanimelist.net/anime/21/One_Piece",
        "https://myanimelist.net/anime/57334/Dandadan",
    ],
)
def test_normalize_mal_anime_url_address_with_title_returned_unchanged(
    url: str,
) -> None:
    assert normalize_mal_anime_url(url) == (url, True)


@pytest.mark.parametrize(
    "url",
    ["https://example.com/anime/21", "https://myanimelist.net/character/40"],
)
def test_normalize_mal_anime_url_non_mal_anime_address_raises_value_error(
    url: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_mal_anime_url(url)


@pytest.mark.parametrize(
    ("label", "expected"),
    [("Episodes", "1122"), ("Status", "Currently Airing"), ("Ranked", "#54")],
)
def test_parse_sidebar_field_label_present_returns_its_value(
    label: str, expected: str
) -> None:
    assert parse_sidebar_field(SIDEBAR, label) == expected


def test_parse_sidebar_field_label_absent_returns_none() -> None:
    assert parse_sidebar_field(SIDEBAR, "Nonexistent") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1-26", [(1, 26)]),
        ("492", [(492, 492)]),
        ("1139-", [(1139, None)]),
        ("1-30, 492, 1139-", [(1, 30), (492, 492), (1139, None)]),
        ("(eps 1-13)", [(1, 13)]),
    ],
)
def test_parse_episode_ranges_range_text_gives_start_and_end(
    raw: str, expected: list[tuple[int, int | None]]
) -> None:
    assert parse_episode_ranges(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "1 2-3", "1 2", "-5"])
def test_parse_episode_ranges_unreadable_parts_skipped(raw: str | None) -> None:
    assert parse_episode_ranges(raw) == []


def test_diff_models_same_values_report_no_changes() -> None:
    diff = diff_models(_anime(), _anime(), "anime")
    assert not diff.has_changes
    assert diff.changes == []


def test_diff_models_changed_field_reported_with_old_and_new_values() -> None:
    diff = diff_models(_anime(score=8.7), _anime(score=8.8), "anime")
    assert diff.has_changes
    assert [
        (change.field, change.old_value, change.new_value) for change in diff.changes
    ] == [("score", 8.7, 8.8)]
    assert diff.entity_id == ONE_PIECE_URL


def test_diff_models_without_old_model_reports_new_entity() -> None:
    diff = diff_models(None, _anime(), "anime")
    assert (diff.is_new, diff.has_changes, diff.entity_id) == (
        True,
        True,
        ONE_PIECE_URL,
    )


def test_diff_model_lists_reports_added_and_removed_entities() -> None:
    old = [_character(40, "Luffy"), _character(41, "Zoro")]
    new = [_character(40, "Luffy"), _character(42, "Nami")]
    list_diff = diff_model_lists(old, new, "character")
    assert list_diff.added == ["https://myanimelist.net/character/42"]
    assert list_diff.removed == ["https://myanimelist.net/character/41"]
    assert not list_diff.updated


def test_diff_model_lists_changed_entity_reported_as_updated() -> None:
    list_diff = diff_model_lists(
        [_character(40, "Luffy")], [_character(40, "Luffy Updated")], "character"
    )
    assert len(list_diff.updated) == 1
    assert list_diff.updated[0].has_changes


def test_get_entity_id_episode_returns_episode_number() -> None:
    episode = MalEpisode(
        source=f"{ONE_PIECE_URL}/One_Piece/episode/1",
        episode_number=1,
        title="Romance Dawn",
    )
    assert _get_entity_id(episode) == 1


def test_get_entity_id_model_without_id_returns_zero() -> None:
    assert _get_entity_id(_ModelWithoutId()) == 0

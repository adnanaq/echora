from datetime import UTC, datetime

import pytest
from match_suggestions.pair_questions import (
    PAIR_QUESTIONS,
    QUESTION_VERSION,
    build_pair_state,
    build_request_fingerprint,
    resolve_page_year,
)
from work_matching import ProviderPage


def _page(
    page_key: str,
    address: str,
    *,
    episode_count: int | None = 12,
    aired_from: datetime | None = datetime(2020, 1, 5, tzinfo=UTC),
    aired_to: datetime | None = datetime(2020, 3, 22, tzinfo=UTC),
    year: int | None = 2020,
) -> ProviderPage:
    return ProviderPage(
        key=page_key,
        address=address,
        titles=("Show",),
        type="TV",
        year=year,
        season="WINTER",
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
    )


MAL_PAGE = _page("mal:anime:1", "https://myanimelist.net/anime/1")
ANILIST_PAGE = _page("anilist:anime:2", "https://anilist.co/anime/2")


def test_pair_questions_ask_four_yes_no_judgments() -> None:
    assert sorted(PAIR_QUESTIONS) == [
        "is_first_work_part_of_second_work",
        "is_same_franchise",
        "is_same_work",
        "is_second_work_part_of_first_work",
    ]


def test_build_pair_state_names_pages_by_site_and_states_details() -> None:
    pair_state = build_pair_state(MAL_PAGE, ANILIST_PAGE, {"mal:anime:1": "A story."})

    assert pair_state["page_a"] == {
        "site": "MyAnimeList",
        "titles": ["Show"],
        "type": "TV",
        "season": "WINTER 2020",
        "episodes": 12,
        "first_aired": "2020-01-05",
        "last_aired": "2020-03-22",
        "synopsis": "A story.",
    }
    assert "synopsis" not in pair_state["page_b"]
    assert pair_state["comparison"]["episodes"] == "Both pages list 12 episodes."


@pytest.mark.parametrize(
    ("page", "expected_sentence"),
    [
        (
            _page(
                "kitsu:anime:3",
                "https://kitsu.app/anime/3",
                episode_count=6,
                aired_to=datetime(2020, 2, 9, tzinfo=UTC),
            ),
            "`page_a` lists fewer episodes than `page_b`, "
            "and all of `page_a`'s air dates fall within `page_b`'s run.",
        ),
        (
            _page(
                "kitsu:anime:3",
                "https://kitsu.app/anime/3",
                aired_from=datetime(2023, 1, 5, tzinfo=UTC),
                aired_to=datetime(2023, 3, 22, tzinfo=UTC),
            ),
            "The episode counts are similar, but the first air dates are more than a month apart.",
        ),
        (
            _page("kitsu:anime:3", "https://kitsu.app/anime/3", aired_from=None),
            "One page gives no episode count or first air date, so their air spans cannot be compared.",
        ),
        (
            _page(
                "kitsu:anime:3",
                "https://kitsu.app/anime/3",
                aired_from=datetime(2020, 1, 20, tzinfo=UTC),
            ),
            "Both pages cover about the same span: similar episode counts, first aired within a month of each other.",
        ),
    ],
)
def test_build_pair_state_air_spans_states_runs(
    page: ProviderPage, expected_sentence: str
) -> None:
    assert build_pair_state(page, MAL_PAGE, {})["comparison"]["air_spans"] == (
        expected_sentence
    )


def test_build_pair_state_long_synopsis_cut() -> None:
    pair_state = build_pair_state(MAL_PAGE, ANILIST_PAGE, {"mal:anime:1": "x" * 5000})

    assert len(pair_state["page_a"]["synopsis"]) == 1500


def test_build_request_fingerprint_changed_synopsis_changes_fingerprint() -> None:
    first_fingerprint = build_request_fingerprint(
        build_pair_state(MAL_PAGE, ANILIST_PAGE, {"mal:anime:1": "A story."})
    )

    assert first_fingerprint == build_request_fingerprint(
        build_pair_state(MAL_PAGE, ANILIST_PAGE, {"mal:anime:1": "A story."})
    )
    assert first_fingerprint != build_request_fingerprint(
        build_pair_state(MAL_PAGE, ANILIST_PAGE, {"mal:anime:1": "Another story."})
    )


def test_question_version_names_wording_and_model() -> None:
    assert len(QUESTION_VERSION) == 16


def test_resolve_page_year_missing_year_reads_first_air_date() -> None:
    assert (
        resolve_page_year(
            _page("mal:anime:1", "https://myanimelist.net/anime/1", year=None)
        )
        == 2020
    )


def test_build_pair_state_provider_without_site_name_named_by_host() -> None:
    simkl_page = _page("simkl:anime:5", "https://simkl.com/anime/5")

    assert build_pair_state(simkl_page, MAL_PAGE, {})["page_a"]["site"] == "simkl.com"


def test_build_pair_state_comparison_edge_cases() -> None:
    movie_missing_year = _page(
        "kitsu:anime:10",
        "https://kitsu.app/anime/10",
        year=None,
        aired_from=None,
    ).model_copy(update={"type": "MOVIE"})
    pair_state_missing = build_pair_state(movie_missing_year, MAL_PAGE, {})[
        "comparison"
    ]
    assert "types differ" in pair_state_missing["type"]
    assert pair_state_missing["years"] == "One page gives no year."

    fall_next_year = _page(
        "kitsu:anime:11",
        "https://kitsu.app/anime/11",
        year=2021,
        aired_from=datetime(2020, 1, 8, tzinfo=UTC),
    ).model_copy(update={"season": "FALL"})
    pair_state_diff_year = build_pair_state(fall_next_year, MAL_PAGE, {})["comparison"]
    assert "years differ" in pair_state_diff_year["years"]
    assert (
        pair_state_diff_year["start_dates"]
        == "The first air dates are less than a week apart."
    )

    fall_same_year = _page(
        "kitsu:anime:12",
        "https://kitsu.app/anime/12",
        year=2020,
        aired_from=datetime(2020, 7, 1, tzinfo=UTC),
    ).model_copy(update={"season": "FALL"})
    pair_state_diff_season = build_pair_state(fall_same_year, MAL_PAGE, {})[
        "comparison"
    ]
    assert "seasons differ" in pair_state_diff_season["years"]
    assert "months apart" in pair_state_diff_season["start_dates"]

    shorter_with_end = _page(
        "kitsu:anime:13",
        "https://kitsu.app/anime/13",
        episode_count=6,
        aired_from=datetime(2020, 1, 1, tzinfo=UTC),
        aired_to=datetime(2020, 2, 15, tzinfo=UTC),
    )
    longer_no_end = _page(
        "mal:anime:13",
        "https://myanimelist.net/anime/13",
        episode_count=24,
        aired_from=datetime(2020, 1, 1, tzinfo=UTC),
        aired_to=None,
    )
    assert (
        "missing end date"
        in build_pair_state(shorter_with_end, longer_no_end, {})["comparison"][
            "air_spans"
        ]
    )

    longer_with_end = longer_no_end.model_copy(
        update={"aired_to": datetime(2020, 6, 30, tzinfo=UTC)}
    )
    shorter_outside = _page(
        "kitsu:anime:14",
        "https://kitsu.app/anime/14",
        episode_count=6,
        aired_from=datetime(2022, 1, 1, tzinfo=UTC),
        aired_to=datetime(2022, 2, 15, tzinfo=UTC),
    )
    assert (
        "airs outside"
        in build_pair_state(shorter_outside, longer_with_end, {})["comparison"][
            "air_spans"
        ]
    )

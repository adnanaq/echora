import pytest
from work_matching.answer_rules import PlacementRule
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.named_entries import (
    MAX_TARGET_ENTRY_SUGGESTIONS,
    find_covered_outside_entries,
    list_named_entry_page_keys,
    resolve_named_entries,
    suggest_target_entries,
)
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage
from work_matching.review_cases import TargetEntrySuggestion

SERIES_MAL = "mal:anime:21273"


@pytest.fixture
def series_seed() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            {
                "title": "Gochuumon wa Usagi desu ka?",
                "type": "TV",
                "sources": [
                    "https://myanimelist.net/anime/21273",
                    "https://anidb.net/anime/10583",
                ],
            },
            {
                "title": "Gochuumon wa Usagi desu ka? Movie: We Are Family!",
                "type": "MOVIE",
                "sources": ["https://myanimelist.net/anime/61115"],
            },
            {
                "title": "Other Show",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/1"],
            },
        ]
    )


@pytest.fixture
def named_answer() -> dict[str, str]:
    return {
        "anidb:anime:19188": f"other work {SERIES_MAL}",
        "anidb:anime:19792": "work 1",
        "mal:anime:61115": "work 1",
    }


def test_list_named_entry_page_keys_reads_keys_after_other_work(named_answer) -> None:
    assert list_named_entry_page_keys(
        named_answer
        | {"mal:anime:61115": "work 1, partial in work 2, other work mal:anime:9"}
    ) == [SERIES_MAL, "mal:anime:9"]


@pytest.mark.parametrize(
    "answer",
    [None, "work 1", {"mal:anime:1": "other work"}, {"mal:anime:1": 1}],
)
def test_list_named_entry_page_keys_answer_naming_no_entry_gives_none(answer) -> None:
    assert list_named_entry_page_keys(answer) == []


def test_resolve_named_entries_maps_named_key_to_seed_entry(
    review_cases, series_seed, named_answer
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(update={"answer": named_answer})
    resolved_case = resolve_named_entries(case, series_seed)
    accepted_answer = resolved_case.accepted_answer
    assert resolved_case.named_entry_reference_pages == {SERIES_MAL: SERIES_MAL}
    assert isinstance(accepted_answer, dict)
    assert accepted_answer["anidb:anime:19188"] == PagePlacement(
        "other work", named_entry_page_key=SERIES_MAL
    )


def test_resolve_named_entries_key_in_no_seed_entry_left_unresolved(
    review_cases, named_answer
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(update={"answer": named_answer})
    resolved_case = resolve_named_entries(case, KnownWorkEntriesFromSeed([]))
    assert resolved_case.named_entry_reference_pages == {}
    assert resolved_case.accepted_answer is None
    assert resolved_case.list_broken_answer_rules(named_answer) == (
        PlacementRule.UNKNOWN_NAMED_ENTRY,
    )


def test_resolve_named_entries_given_answer_resolves_that_answer(
    review_cases, series_seed, named_answer
) -> None:
    resolved_case = resolve_named_entries(
        review_cases["gochuumon_movie"], series_seed, named_answer
    )
    assert resolved_case.list_broken_answer_rules(named_answer) == ()


def test_suggest_target_entries_lists_series_entries_outside_case(
    review_cases, series_seed
) -> None:
    suggestions = suggest_target_entries(review_cases["gochuumon_movie"], series_seed)
    assert [
        (suggestion.entry_reference_page, suggestion.title, suggestion.providers)
        for suggestion in suggestions
    ] == [(SERIES_MAL, "Gochuumon wa Usagi desu ka?", ("anidb", "mal"))]


def test_suggest_target_entries_closest_titles_first_across_case_entries(
    review_cases,
) -> None:
    seed_titles = (
        "Alpha Beta Gamma Delta",
        "Zeta Eta Theta",
        "Alpha",
        "Alpha Beta",
        "Alpha Beta Gamma",
        "Zeta Eta Theta 2nd Season",
    )
    seed = KnownWorkEntriesFromSeed(
        [
            {
                "title": title,
                "type": "TV",
                "sources": [f"https://myanimelist.net/anime/{number}"],
            }
            for number, title in enumerate(seed_titles, start=1)
        ]
    )
    case_entries = {
        work_entry.reference_page_key: work_entry
        for work_entry in (
            seed.find_by_page_key("mal:anime:1"),
            seed.find_by_page_key("mal:anime:2"),
        )
        if work_entry
    }
    case = review_cases["gochuumon_movie"].model_copy(update={"entries": case_entries})
    assert [suggestion.title for suggestion in suggest_target_entries(case, seed)] == [
        "Alpha Beta Gamma",
        "Zeta Eta Theta 2nd Season",
        "Alpha Beta",
        "Alpha",
    ]


def test_suggest_target_entries_keeps_closest_titles_up_to_cap(review_cases) -> None:
    seed_titles = (
        "Alpha Beta Gamma Delta",
        "Alpha Beta",
        *(f"Alpha Beta Gamma Delta {number}" for number in range(1, 14)),
    )
    seed = KnownWorkEntriesFromSeed(
        [
            {
                "title": title,
                "type": "TV",
                "sources": [f"https://myanimelist.net/anime/{number}"],
            }
            for number, title in enumerate(seed_titles, start=1)
        ]
    )
    case_entry = seed.find_by_page_key("mal:anime:1")
    assert case_entry is not None
    case = review_cases["gochuumon_movie"].model_copy(
        update={"entries": {case_entry.reference_page_key: case_entry}}
    )
    suggested_titles = [
        suggestion.title for suggestion in suggest_target_entries(case, seed)
    ]
    assert len(suggested_titles) == MAX_TARGET_ENTRY_SUGGESTIONS
    assert "Alpha Beta" not in suggested_titles


def _build_dated_page(
    page_key: str, page_type: str, episode_count: int, aired_from: str, aired_to: str
) -> ProviderPage:
    return ProviderPage(
        key=page_key,
        address=f"https://example.org/{page_key}",
        type=page_type,
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
    )


def test_find_covered_outside_entries_keeps_entries_page_run_covers(
    review_cases,
) -> None:
    season_four = TargetEntrySuggestion(
        entry_reference_page="anilist:anime:199073",
        title="Xianwu Zhuan 4",
        providers=("anilist",),
    )
    season_four_on_mal = season_four.model_copy(
        update={"entry_reference_page": "mal:anime:9", "providers": ("mal",)}
    )
    movie = TargetEntrySuggestion(
        entry_reference_page="anilist:anime:9", title="Movie", providers=("anilist",)
    )
    case = review_cases["gochuumon_movie"].model_copy(
        update={
            "pages": (
                _build_dated_page(
                    "mal:anime:56717",
                    "ONA",
                    147,
                    "2023-09-23T15:00:00Z",
                    "2026-05-23T15:00:00Z",
                ),
            ),
            "target_entry_suggestions": (season_four, season_four_on_mal, movie),
        }
    )
    season_four_page = _build_dated_page(
        "anilist:anime:199073",
        "ONA",
        43,
        "2025-09-05T15:00:00Z",
        "2026-05-22T15:00:00Z",
    )
    covered_entries = find_covered_outside_entries(
        case,
        {
            "anilist:anime:199073": [season_four_page],
            "mal:anime:9": [season_four_page],
            "anilist:anime:9": [
                _build_dated_page(
                    "anilist:anime:9",
                    "MOVIE",
                    1,
                    "2025-01-10T15:00:00Z",
                    "2025-01-10T15:00:00Z",
                )
            ],
        },
    )
    assert covered_entries == (season_four,)

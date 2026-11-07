import pytest
from work_matching.answer_rules import PlacementRule
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.named_entries import (
    list_named_entry_page_keys,
    resolve_named_entries,
    suggest_target_entries,
)
from work_matching.page_placement import PagePlacement

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
    assert list_named_entry_page_keys(named_answer) == [SERIES_MAL]


@pytest.mark.parametrize("answer", [None, "work 1", {"mal:anime:1": "other work"}])
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

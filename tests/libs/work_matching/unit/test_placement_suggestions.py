import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.placement_suggestions import suggest_page_placements

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"


@pytest.fixture
def part_pages(linked_page, work_entry):
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(KITSU_ONE, MAL).model_copy(update={"episode_count": 1}),
        linked_page(KITSU_TWO, MAL).model_copy(update={"episode_count": 4}),
    ]
    return entry, decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))


def test_suggest_page_placements_linked_groups_and_other_work(one_linked) -> None:
    assert suggest_page_placements(*one_linked[:2]) == {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "other work",
        "mal:anime:1": "work 1",
    }


def test_suggest_page_placements_works_linked_apart_numbered_by_trust(
    linked_apart,
) -> None:
    assert suggest_page_placements(*linked_apart[:2]) == {
        "mal:anime:1": "work 1",
        "mal:anime:2": "work 2",
        "anilist:anime:1": "work 1",
        "anilist:anime:2": "work 2",
    }


def test_suggest_page_placements_works_joined_through_shared_page_left_out(
    matching_case_decisions,
) -> None:
    assert suggest_page_placements(*matching_case_decisions("puu_neko")[:2]) == {}


def test_suggest_page_placements_unlinked_page_left_out(
    matching_case_decisions,
) -> None:
    suggestion = suggest_page_placements(
        *matching_case_decisions("ao_haru_ride_ova")[:2]
    )
    assert suggestion == {
        "anilist:anime:20837": "work 1, part",
        "anilist:anime:20900": "work 1, part",
        "animeplanet:anime:blue-spring-ride-page-13": "work 1",
        "kitsu:anime:8488": "work 1",
        "mal:anime:24151": "work 1",
    }


def test_suggest_page_placements_part_pages_marked_part(part_pages) -> None:
    assert suggest_page_placements(*part_pages) == {
        "kitsu:anime:1": "work 1, part",
        "kitsu:anime:2": "work 1, part",
        "mal:anime:1": "work 1",
    }


def test_suggest_page_placements_details_differ_pages_left_out(
    linked_page, work_entry
) -> None:
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(address, MAL).model_copy(update={"year": year})
        for address, year in ((KITSU_ONE, 2013), (KITSU_TWO, 2021))
    ]
    decisions = decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))
    assert suggest_page_placements(entry, decisions) == {"mal:anime:1": "work 1"}

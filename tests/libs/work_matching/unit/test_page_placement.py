import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.page_placement import (
    PagePlacement,
    list_placeable_page_keys,
    place_pages_without_review,
    read_placement,
    write_placement,
)

ENTRY = PagePlacement("work 1")
OTHER_WORK = PagePlacement("other work")
MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
OTHER_MAL = "https://myanimelist.net/anime/9"
OTHER_ANILIST = "https://anilist.co/anime/9"


@pytest.fixture
def known_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            {"title": "Other", "type": "TV", "sources": [OTHER_MAL]},
            {"title": "Third", "type": "TV", "sources": [OTHER_ANILIST]},
        ]
    )


def test_place_pages_without_review_renamed_page_keeps_current_page(
    matching_case_decisions,
) -> None:
    entry, decisions, pages = matching_case_decisions("cocoon")
    assert place_pages_without_review(decisions) == {
        "animeplanet:anime:cocoon-one-summer-of-girlhood": ENTRY
    }


def test_place_pages_without_review_one_linked_page_places_other_page_elsewhere(
    one_linked,
) -> None:
    entry, decisions, pages = one_linked
    assert place_pages_without_review(decisions) == {
        "kitsu:anime:1": ENTRY,
        "kitsu:anime:2": OTHER_WORK,
    }


def test_place_pages_without_review_pages_sharing_link_gives_none(shared_link) -> None:
    assert place_pages_without_review(shared_link[1]) is None


@pytest.mark.parametrize("case_name", ["puu_neko", "genshin_sigewinne"])
def test_place_pages_without_review_separate_works_or_undecided_gives_none(
    matching_case_decisions, case_name: str
) -> None:
    assert place_pages_without_review(matching_case_decisions(case_name)[1]) is None


def test_place_pages_without_review_one_provider_needing_review_places_nothing(
    linked_page, work_entry, known_work_entries
) -> None:
    planet_one = "https://www.anime-planet.com/anime/one"
    planet_two = "https://www.anime-planet.com/anime/two"
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO, planet_one, planet_two)
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, OTHER_MAL)]
    decisions = decide_doubled_providers(entry, pages, known_work_entries)
    assert [decision.is_review_needed for decision in decisions] == [False, True]
    assert place_pages_without_review(decisions) is None


def test_list_placeable_page_keys_doubled_pages_first_then_other_pages(
    matching_case_decisions,
) -> None:
    entry, decisions, _ = matching_case_decisions("cocoon")
    assert list_placeable_page_keys(entry, decisions) == (
        "animeplanet:anime:cocoon-one-summer-of-girlhood",
        "anidb:anime:18084",
        "anilist:anime:166527",
        "animecountdown:anime:2196692",
        "ann:anime:28792",
        "anisearch:anime:18492",
        "kitsu:anime:47658",
        "livechart:anime:12223",
        "mal:anime:55779",
        "simkl:anime:2196692",
    )


def test_read_placement_other_work_naming_page_key_keeps_key() -> None:
    assert read_placement("other work mal:anime:62841") == PagePlacement(
        "other work", named_entry_page_key="mal:anime:62841"
    )


@pytest.mark.parametrize(
    "placement_text",
    [
        "other work Kusuriya",
        "other work mal:anime:",
        "other work ",
        "work 1, partial in other work mal:anime:",
    ],
)
def test_read_placement_other_work_naming_no_page_key_gives_none(
    placement_text: str,
) -> None:
    assert read_placement(placement_text) is None


@pytest.mark.parametrize(
    "placement_text",
    [
        "work 1",
        "work 2, partial in work 1, other work mal:anime:9",
        "dropped",
        "other work",
        "other work mal:anime:9",
    ],
)
def test_write_placement_gives_text_read(placement_text: str) -> None:
    placement = read_placement(placement_text)
    assert placement is not None
    assert write_placement(placement) == placement_text

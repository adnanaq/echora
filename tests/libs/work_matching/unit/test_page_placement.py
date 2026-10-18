import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.page_placement import (
    PagePlacement,
    list_placeable_pages,
    place_pages_without_review,
    read_answer_placements,
    suggest_page_placements,
)
from work_matching.provider_page import ProviderPage

ENTRY = PagePlacement("entry")
OTHER_WORK = PagePlacement("other work")
MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
OTHER_MAL = "https://myanimelist.net/anime/9"
OTHER_ANILIST = "https://anilist.co/anime/9"


@pytest.fixture
def known_works() -> KnownWorksFromSeed:
    return KnownWorksFromSeed(
        [
            {"title": "Other", "type": "TV", "sources": [OTHER_MAL]},
            {"title": "Third", "type": "TV", "sources": [OTHER_ANILIST]},
        ]
    )


@pytest.fixture
def one_linked(linked_page, work_entry, known_works):
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(MAL),
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, OTHER_MAL),
    ]
    return entry, decide_doubled_providers(entry, pages, known_works), pages


@pytest.fixture
def shared_link(linked_page, work_entry):
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, MAL)]
    return entry, decide_doubled_providers(entry, pages, KnownWorksFromSeed([])), pages


@pytest.fixture
def linked_apart(linked_page, work_entry):
    entry = work_entry(
        MAL,
        "https://myanimelist.net/anime/2",
        "https://anilist.co/anime/1",
        "https://anilist.co/anime/2",
        "https://anisearch.com/anime/1",
    )
    pages = [
        linked_page("https://anilist.co/anime/1", MAL),
        linked_page("https://anilist.co/anime/2", "https://myanimelist.net/anime/2"),
    ]
    return entry, decide_doubled_providers(entry, pages, KnownWorksFromSeed([])), pages


def _matching_case(matching_cases, name: str):
    entries, pages = matching_cases[name]
    return (
        entries[0],
        decide_doubled_providers(entries[0], pages, KnownWorksFromSeed([])),
        pages,
    )


def test_place_pages_without_review_renamed_page_keeps_current_page(
    matching_cases,
) -> None:
    entry, decisions, pages = _matching_case(matching_cases, "cocoon")
    assert place_pages_without_review(entry, decisions, pages) == {
        "animeplanet:anime:cocoon-one-summer-of-girlhood": ENTRY
    }


def test_place_pages_without_review_one_linked_page_places_other_page_elsewhere(
    one_linked,
) -> None:
    entry, decisions, pages = one_linked
    assert place_pages_without_review(entry, decisions, pages) == {
        "kitsu:anime:1": ENTRY,
        "kitsu:anime:2": OTHER_WORK,
    }


def test_place_pages_without_review_pages_sharing_link_gives_none(shared_link) -> None:
    assert place_pages_without_review(*shared_link) is None


@pytest.mark.parametrize("case_name", ["puu_neko", "genshin_sigewinne"])
def test_place_pages_without_review_mixed_entry_or_review_gives_none(
    matching_cases, case_name: str
) -> None:
    assert (
        place_pages_without_review(*_matching_case(matching_cases, case_name)) is None
    )


def test_place_pages_without_review_page_linking_two_works_gives_none(
    linked_page, work_entry, known_works
) -> None:
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, OTHER_MAL, OTHER_ANILIST),
    ]
    decisions = decide_doubled_providers(entry, pages, known_works)
    assert place_pages_without_review(entry, decisions, pages) is None


def test_place_pages_without_review_page_linking_work_outside_seed_gives_none(
    linked_page, work_entry, known_works
) -> None:
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, "https://myanimelist.net/anime/77"),
    ]
    decisions = decide_doubled_providers(entry, pages, known_works)
    assert place_pages_without_review(entry, decisions, pages) is None


@pytest.mark.parametrize(
    ("moved_page_details", "is_placed"),
    [
        ({"type": "MOVIE", "year": 2020}, False),
        ({"type": "TV", "year": 1987}, False),
        ({"type": "TV", "year": 2020}, True),
        ({"type": "TV SHORT"}, True),
    ],
)
def test_place_pages_without_review_moved_page_details_must_agree_with_target(
    work_entry, moved_page_details: dict, is_placed: bool
) -> None:
    known_works = KnownWorksFromSeed(
        [
            {
                "title": "Other",
                "type": "TV",
                "sources": [OTHER_MAL],
                "animeSeason": {"year": 2020},
            }
        ]
    )
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [
        ProviderPage.from_canonical_record(
            KITSU_ONE, {"title": "Show", "sources": [KITSU_ONE, MAL]}
        ),
        ProviderPage.from_canonical_record(
            KITSU_TWO,
            {"title": "Other", "sources": [KITSU_TWO, OTHER_MAL]} | moved_page_details,
        ),
    ]
    decisions = decide_doubled_providers(entry, pages, known_works)
    placements = place_pages_without_review(entry, decisions, pages)
    assert (placements is not None) is is_placed


def test_place_pages_without_review_current_page_not_listed_gives_none(
    work_entry,
) -> None:
    new_address = "https://www.anime-planet.com/anime/new"
    old_addresses = [f"https://www.anime-planet.com/anime/old-{n}" for n in (1, 2)]
    entry = work_entry(MAL, *old_addresses)
    pages = [
        ProviderPage.from_canonical_record(
            address, {"title": "Show", "sources": [address, new_address]}
        )
        for address in old_addresses
    ]
    decisions = decide_doubled_providers(entry, pages, KnownWorksFromSeed([]))
    assert place_pages_without_review(entry, decisions, pages) is None


def test_list_placeable_pages_doubled_pages_first_then_other_pages(
    matching_cases,
) -> None:
    entry, decisions, _ = _matching_case(matching_cases, "cocoon")
    assert list_placeable_pages(entry, decisions) == (
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


def test_suggest_page_placements_linked_groups_and_other_work(one_linked) -> None:
    assert suggest_page_placements(*one_linked[:2]) == {
        "kitsu:anime:1": "entry",
        "kitsu:anime:2": "other work",
        "mal:anime:1": "entry",
    }


def test_suggest_page_placements_works_linked_apart_numbered_by_trust(
    linked_apart,
) -> None:
    assert suggest_page_placements(*linked_apart[:2]) == {
        "mal:anime:1": "entry",
        "mal:anime:2": "work 2",
        "anilist:anime:1": "entry",
        "anilist:anime:2": "work 2",
    }


def test_suggest_page_placements_works_joined_through_shared_page_left_out(
    matching_cases,
) -> None:
    assert (
        suggest_page_placements(*_matching_case(matching_cases, "puu_neko")[:2]) == {}
    )


def test_suggest_page_placements_unlinked_page_left_out(matching_cases) -> None:
    suggestion = suggest_page_placements(
        *_matching_case(matching_cases, "ao_haru_ride_ova")[:2]
    )
    assert suggestion == {
        "anilist:anime:20837": "entry",
        "anilist:anime:20900": "entry",
        "animeplanet:anime:blue-spring-ride-page-13": "entry",
        "kitsu:anime:8488": "entry",
        "mal:anime:24151": "entry",
    }


@pytest.mark.parametrize(
    ("answer", "placements"),
    [
        (
            {"kitsu:anime:1": "entry", "kitsu:anime:2": "other work"},
            {"kitsu:anime:1": ENTRY, "kitsu:anime:2": OTHER_WORK},
        ),
        (
            {"kitsu:anime:1": "entry, main", "kitsu:anime:2": "entry"},
            {
                "kitsu:anime:1": PagePlacement("entry", is_main_page=True),
                "kitsu:anime:2": ENTRY,
            },
        ),
        (
            {
                "kitsu:anime:1": "entry",
                "kitsu:anime:2": "other work",
                "mal:anime:1": "other work",
            },
            {
                "kitsu:anime:1": ENTRY,
                "kitsu:anime:2": OTHER_WORK,
                "mal:anime:1": OTHER_WORK,
            },
        ),
        (
            {
                "kitsu:anime:1": "entry",
                "kitsu:anime:2": "work 2",
                "mal:anime:1": "entry",
            },
            {
                "kitsu:anime:1": ENTRY,
                "kitsu:anime:2": PagePlacement("work 2"),
                "mal:anime:1": ENTRY,
            },
        ),
    ],
)
def test_read_answer_placements_valid_answer_gives_placements(
    shared_link, answer: dict[str, str], placements: dict[str, PagePlacement]
) -> None:
    assert read_answer_placements(answer, *shared_link[:2]) == placements


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "one work",
        {"kitsu:anime:1": "entry"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "entry"},
        {"kitsu:anime:1": "entry, main", "kitsu:anime:2": "entry, main"},
        {"kitsu:anime:1": "entry, main", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "other work, main"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "work 2"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "work 1", "mal:anime:1": "entry"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "work 01", "mal:anime:1": "entry"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "maybe"},
        {
            "kitsu:anime:1": "entry",
            "kitsu:anime:2": "other work",
            "mal:anime:5": "entry",
        },
        {"kitsu:anime:1": 1, "kitsu:anime:2": ["entry"]},
    ],
)
def test_read_answer_placements_invalid_answer_gives_none(
    shared_link, answer: object
) -> None:
    assert read_answer_placements(answer, *shared_link[:2]) is None

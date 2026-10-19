import json
import logging

import pytest
from seed_cleaning.linked_page_additions import (
    LinkedPageCorrections,
    correct_linked_pages,
    find_entries_without_mal_page,
    find_linked_pages,
    list_entries_with_placed_pages,
)
from seed_cleaning.seed_corrections import (
    CorrectedEntry,
    SeedCorrection,
    correct_doubled_provider_entries,
)
from work_matching import (
    KnownWorksFromSeed,
    ProviderPage,
    ProviderPageStore,
    ReviewFile,
    WorkEntry,
)

BOXER_ANILIST = "https://anilist.co/anime/163794"
BOXER_MAL = "https://myanimelist.net/anime/65144"
YAMATO_ANILIST = "https://anilist.co/anime/182615"
YAMATO_ANIDB = "https://anidb.net/anime/18902"
YAMATO_MAL = "https://myanimelist.net/anime/65124"
MORI_NO_YOUSEI_KITSU = "https://kitsu.app/anime/40741"


def _seed_entry(title: str, work_type: str, *addresses: str) -> dict:
    return {
        "title": title,
        "type": work_type,
        "sources": list(addresses),
        "animeSeason": {"season": "UNDEFINED"},
    }


BOXER_SEED_ENTRY = _seed_entry("The Boxer", "TV", BOXER_ANILIST)
YAMATO_SEED_ENTRY = _seed_entry(
    "Space Battleship Yamato Movie (New)", "MOVIE", YAMATO_ANIDB, YAMATO_ANILIST
)
MORI_NO_YOUSEI_SEED_ENTRY = _seed_entry(
    "Mori no Yousei: Kinoko no Musume", "TV", MORI_NO_YOUSEI_KITSU
)


def _page(address: str, *links: str, **details) -> ProviderPage:
    return ProviderPage.from_canonical_record(
        address,
        {"title": "Show", "external_sources": [{"source": link} for link in links]}
        | details,
    )


@pytest.fixture
def known_works() -> KnownWorksFromSeed:
    return KnownWorksFromSeed(
        [BOXER_SEED_ENTRY, YAMATO_SEED_ENTRY, MORI_NO_YOUSEI_SEED_ENTRY]
    )


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


@pytest.fixture
def review_file(tmp_path) -> ReviewFile:
    return ReviewFile(tmp_path / "review.jsonl")


def _unchanged(known_works: KnownWorksFromSeed) -> list[CorrectedEntry]:
    return [CorrectedEntry(entry.reference_page, entry) for entry in known_works]


def _correct(
    known_works: KnownWorksFromSeed, store: ProviderPageStore, review_file: ReviewFile
) -> LinkedPageCorrections:
    return correct_linked_pages(
        find_linked_pages(_unchanged(known_works), store, known_works),
        store,
        review_file,
    )


def test_find_entries_without_mal_page_keeps_entries_lacking_mal(known_works) -> None:
    entries = [
        *known_works,
        WorkEntry.from_seed_entry(_seed_entry("Mal", "TV", YAMATO_MAL)),
    ]
    assert [entry.title for entry in find_entries_without_mal_page(entries)] == [
        "The Boxer",
        "Space Battleship Yamato Movie (New)",
        "Mori no Yousei: Kinoko no Musume",
    ]


def test_list_entries_with_placed_pages_corrected_entry_replaces_seed_entry(
    store, known_works, review_file
) -> None:
    doubled_entry = WorkEntry.from_seed_entry(
        _seed_entry("The Boxer", "TV", BOXER_ANILIST, "https://anilist.co/anime/9")
    )
    store.add_pages(
        _page(BOXER_ANILIST, sources=[BOXER_ANILIST, "https://anilist.co/anime/9"]),
        _page("https://anilist.co/anime/9"),
    )
    works = KnownWorksFromSeed(
        [
            _seed_entry("The Boxer", "TV", BOXER_ANILIST, "https://anilist.co/anime/9"),
            YAMATO_SEED_ENTRY,
        ]
    )
    doubled_corrections = correct_doubled_provider_entries(
        [doubled_entry], store, works, review_file
    )
    placed_entries = list_entries_with_placed_pages(
        list(works), doubled_corrections, works, store
    )
    assert [
        (placed.seed_entry_reference_page, list(placed.entry.pages))
        for placed in placed_entries
    ] == [
        ("anilist:anime:163794", ["anilist:anime:9"]),
        ("anilist:anime:182615", ["anidb:anime:18902", "anilist:anime:182615"]),
    ]


def test_list_entries_with_placed_pages_entry_with_open_case_left_out(
    store, review_file
) -> None:
    works = KnownWorksFromSeed(
        [
            _seed_entry(
                "Show", "TV", "https://kitsu.app/anime/1", "https://kitsu.app/anime/2"
            ),
            BOXER_SEED_ENTRY,
        ]
    )
    doubled_corrections = correct_doubled_provider_entries(
        [next(iter(works))], store, works, review_file
    )
    placed_entries = list_entries_with_placed_pages(
        list(works), doubled_corrections, works, store
    )
    assert [placed.seed_entry_reference_page for placed in placed_entries] == [
        "anilist:anime:163794"
    ]


def test_find_linked_pages_groups_entries_linking_each_unheld_page(
    store, known_works
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL),
        _page(YAMATO_ANILIST, BOXER_MAL, YAMATO_ANIDB),
        _page(
            MORI_NO_YOUSEI_KITSU,
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=1",
        ),
    )
    linked_pages = find_linked_pages(_unchanged(known_works), store, known_works)
    assert {
        key: [linking.seed_entry_reference_page for linking in linkings]
        for key, linkings in linked_pages.items()
    } == {"mal:anime:65144": ["anilist:anime:163794", "anilist:anime:182615"]}


def test_find_linked_pages_new_entry_links_nothing(store, known_works) -> None:
    store.add_pages(_page(BOXER_ANILIST, BOXER_MAL))
    new_entry = CorrectedEntry(None, WorkEntry.from_seed_entry(BOXER_SEED_ENTRY))
    assert find_linked_pages([new_entry], store, known_works) == {}


def test_correct_linked_pages_page_passing_checks_added_to_entry(
    store, known_works, review_file
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, episode_count=12),
        _page(BOXER_MAL, type="TV"),
    )
    linked_page_corrections = _correct(known_works, store, review_file)
    assert linked_page_corrections.corrections == (
        SeedCorrection(
            entry_reference_page="anilist:anime:163794",
            page_key="mal:anime:65144",
            address=BOXER_MAL,
            work="entry",
            is_added_page=True,
            reason="CHECKS_PASSED",
        ),
    )
    assert linked_page_corrections.review_cases == ()


def test_correct_linked_pages_page_not_stored_waits_for_fetch(
    store, known_works, review_file
) -> None:
    store.add_pages(_page(BOXER_ANILIST, BOXER_MAL))
    linked_page_corrections = _correct(known_works, store, review_file)
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.waiting_page_keys,
    ) == ((), ("mal:anime:65144",))


def test_correct_linked_pages_dead_page_not_added(
    store, known_works, review_file
) -> None:
    store.add_pages(
        _page(MORI_NO_YOUSEI_KITSU, "https://myanimelist.net/anime/35362"),
        ProviderPage.from_dead_address("https://myanimelist.net/anime/35362"),
    )
    linked_page_corrections = _correct(known_works, store, review_file)
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.review_cases,
        linked_page_corrections.not_found_page_keys,
    ) == ((), (), ("mal:anime:35362",))


def test_correct_linked_pages_page_needing_person_becomes_review_case(
    store, known_works, review_file
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, episode_count=12),
        _page(BOXER_MAL, type="TV", episode_count=1),
    )
    linked_page_corrections = _correct(known_works, store, review_file)
    [review_case] = linked_page_corrections.review_cases
    assert (linked_page_corrections.corrections, review_case.case) == (
        (),
        "linked page: mal:anime:65144 from anilist:anime:163794",
    )


def test_correct_linked_pages_kept_answer_adds_part_page(
    tmp_path, store, known_works
) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        json.dumps(
            {
                "case": "linked page: mal:anime:65144 from anilist:anime:163794",
                "answer": {"mal:anime:65144": "add as part"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, episode_count=12),
        _page(BOXER_MAL, type="TV", episode_count=1),
    )
    [correction] = _correct(known_works, store, ReviewFile(review_path)).corrections
    assert (correction.is_added_page, correction.is_part_page, correction.reason) == (
        True,
        True,
        "review answer",
    )


def test_correct_linked_pages_answers_adding_two_pages_of_provider_add_neither(
    tmp_path, store, known_works, caplog
) -> None:
    other_mal = "https://myanimelist.net/anime/65145"
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(
                {
                    "case": f"linked page: {key} from anilist:anime:163794",
                    "answer": {key: "add"},
                }
            )
            + "\n"
            for key in ("mal:anime:65144", "mal:anime:65145")
        ),
        encoding="utf-8",
    )
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, other_mal),
        _page(BOXER_MAL, type="TV"),
        _page(other_mal, type="TV"),
    )
    with caplog.at_level(logging.WARNING):
        linked_page_corrections = _correct(known_works, store, ReviewFile(review_path))
    assert linked_page_corrections.corrections == ()
    assert "anilist:anime:163794" in caplog.text

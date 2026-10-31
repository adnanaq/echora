import json
import logging

import pytest
from seed_cleaning.corrected_entries import (
    CorrectedEntry,
)
from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.linked_page_additions import (
    LinkedPageCorrections,
    correct_linked_pages,
    find_entries_without_mal_page,
    find_linking_seed_entries,
    list_entries_with_placed_pages,
)
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import (
    KnownWorkEntriesFromSeed,
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


def _page(address: str, *linked_addresses: str, **details) -> ProviderPage:
    return ProviderPage.from_canonical_record(
        address,
        {
            "title": "Show",
            "external_sources": [
                {"source": linked_address} for linked_address in linked_addresses
            ],
        }
        | details,
    )


@pytest.fixture
def known_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [BOXER_SEED_ENTRY, YAMATO_SEED_ENTRY, MORI_NO_YOUSEI_SEED_ENTRY]
    )


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


@pytest.fixture
def review_file(tmp_path) -> ReviewFile:
    return ReviewFile(tmp_path / "review.jsonl")


def _unchanged(known_work_entries: KnownWorkEntriesFromSeed) -> list[CorrectedEntry]:
    return [CorrectedEntry(entry.reference_page, entry) for entry in known_work_entries]


def _correct(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> LinkedPageCorrections:
    return correct_linked_pages(
        find_linking_seed_entries(
            _unchanged(known_work_entries), store, known_work_entries
        ),
        store,
        known_work_entries,
        review_file,
    )


def test_correct_linked_pages_slug_of_held_page_not_added(
    store, known_work_entries, review_file
) -> None:
    kitsu_slug = "https://kitsu.app/anime/mori-no-yousei-kinoko-no-musume"
    store.add_pages(
        _page(BOXER_ANILIST, kitsu_slug),
        _page(kitsu_slug, type="TV", sources=[kitsu_slug, MORI_NO_YOUSEI_KITSU]),
    )
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.review_cases,
        linked_page_corrections.held_page_keys,
    ) == ((), (), ("kitsu:anime:40741",))


def test_correct_linked_pages_slug_of_unheld_page_added_under_number(
    store, known_work_entries, review_file
) -> None:
    kitsu_slug = "https://kitsu.app/anime/the-boxer"
    store.add_pages(
        _page(BOXER_ANILIST, kitsu_slug),
        _page(
            kitsu_slug, type="TV", sources=[kitsu_slug, "https://kitsu.app/anime/51119"]
        ),
    )
    [correction] = _correct(known_work_entries, store, review_file).corrections
    assert (correction.page_key, correction.address) == (
        "kitsu:anime:51119",
        "https://kitsu.app/anime/51119",
    )


def test_correct_linked_pages_two_spellings_from_two_entries_decided_together(
    store, known_work_entries, review_file
) -> None:
    kitsu_slug = "https://kitsu.app/anime/the-boxer"
    kitsu_number = "https://kitsu.app/anime/51119"
    store.add_pages(
        _page(BOXER_ANILIST, kitsu_slug),
        _page(YAMATO_ANILIST, kitsu_number),
        _page(kitsu_slug, type="TV", sources=[kitsu_slug, kitsu_number]),
        _page(kitsu_number, type="TV"),
    )
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    [review_case] = linked_page_corrections.review_cases
    assert (linked_page_corrections.corrections, review_case.case_name) == (
        (),
        "linked page: kitsu:anime:51119 from anilist:anime:163794 anilist:anime:182615",
    )


def test_find_entries_without_mal_page_keeps_entries_lacking_mal(
    known_work_entries,
) -> None:
    entries = [
        *known_work_entries,
        WorkEntry.from_seed_entry(_seed_entry("Mal", "TV", YAMATO_MAL)),
    ]
    assert [entry.title for entry in find_entries_without_mal_page(entries)] == [
        "The Boxer",
        "Space Battleship Yamato Movie (New)",
        "Mori no Yousei: Kinoko no Musume",
    ]


def test_list_entries_with_placed_pages_corrected_entry_replaces_seed_entry(
    store, known_work_entries, review_file
) -> None:
    doubled_entry = WorkEntry.from_seed_entry(
        _seed_entry("The Boxer", "TV", BOXER_ANILIST, "https://anilist.co/anime/9")
    )
    store.add_pages(
        _page(BOXER_ANILIST, sources=[BOXER_ANILIST, "https://anilist.co/anime/9"]),
        _page("https://anilist.co/anime/9"),
    )
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("The Boxer", "TV", BOXER_ANILIST, "https://anilist.co/anime/9"),
            YAMATO_SEED_ENTRY,
        ]
    )
    doubled_corrections = correct_entry_groups(
        [EntryGroup((doubled_entry,))], store, works, review_file
    )
    placed_entries = list_entries_with_placed_pages(
        list(works), doubled_corrections, works, store
    )
    assert [
        (placed.seed_entry_reference_page, list(placed.entry.addresses_by_page_key))
        for placed in placed_entries
    ] == [
        ("anilist:anime:163794", ["anilist:anime:9"]),
        ("anilist:anime:182615", ["anidb:anime:18902", "anilist:anime:182615"]),
    ]


def test_list_entries_with_placed_pages_entry_with_open_case_left_out(
    store, review_file
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry(
                "Show", "TV", "https://kitsu.app/anime/1", "https://kitsu.app/anime/2"
            ),
            BOXER_SEED_ENTRY,
        ]
    )
    doubled_corrections = correct_entry_groups(
        [EntryGroup((next(iter(works)),))], store, works, review_file
    )
    placed_entries = list_entries_with_placed_pages(
        list(works), doubled_corrections, works, store
    )
    assert [placed.seed_entry_reference_page for placed in placed_entries] == [
        "anilist:anime:163794"
    ]


def test_find_linking_seed_entries_groups_entries_linking_each_unheld_page(
    store, known_work_entries
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL),
        _page(YAMATO_ANILIST, BOXER_MAL, YAMATO_ANIDB),
        _page(
            MORI_NO_YOUSEI_KITSU,
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=1",
        ),
    )
    linked_pages = find_linking_seed_entries(
        _unchanged(known_work_entries), store, known_work_entries
    )
    assert {
        key: [linking.seed_entry_reference_page for linking in linkings]
        for key, linkings in linked_pages.items()
    } == {"mal:anime:65144": ["anilist:anime:163794", "anilist:anime:182615"]}


def test_find_linking_seed_entries_new_entry_links_nothing(
    store, known_work_entries
) -> None:
    store.add_pages(_page(BOXER_ANILIST, BOXER_MAL))
    new_entry = CorrectedEntry(None, WorkEntry.from_seed_entry(BOXER_SEED_ENTRY))
    assert find_linking_seed_entries([new_entry], store, known_work_entries) == {}


def test_correct_linked_pages_page_passing_checks_added_to_entry(
    store, known_work_entries, review_file
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, episode_count=12),
        _page(BOXER_MAL, type="TV"),
    )
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    assert linked_page_corrections.corrections == (
        SeedCorrection(
            entry_reference_page="anilist:anime:163794",
            page_key="mal:anime:65144",
            address=BOXER_MAL,
            placement_word="work 1",
            is_added_page=True,
            reason="CHECKS_PASSED",
        ),
    )
    assert linked_page_corrections.review_cases == ()


def test_correct_linked_pages_page_not_stored_waits_for_fetch(
    store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(BOXER_ANILIST, BOXER_MAL))
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.waiting_page_keys,
    ) == ((), ("mal:anime:65144",))


def test_correct_linked_pages_dead_page_not_added(
    store, known_work_entries, review_file
) -> None:
    store.add_pages(
        _page(MORI_NO_YOUSEI_KITSU, "https://myanimelist.net/anime/35362"),
        ProviderPage.from_dead_address("https://myanimelist.net/anime/35362"),
    )
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.review_cases,
        linked_page_corrections.not_found_page_keys,
    ) == ((), (), ("mal:anime:35362",))


def test_correct_linked_pages_page_needing_person_becomes_review_case(
    store, known_work_entries, review_file
) -> None:
    store.add_pages(
        _page(BOXER_ANILIST, BOXER_MAL, episode_count=12),
        _page(BOXER_MAL, type="TV", episode_count=1),
    )
    linked_page_corrections = _correct(known_work_entries, store, review_file)
    [review_case] = linked_page_corrections.review_cases
    assert (linked_page_corrections.corrections, review_case.case_name) == (
        (),
        "linked page: mal:anime:65144 from anilist:anime:163794",
    )


def test_correct_linked_pages_kept_answer_adds_part_page(
    tmp_path, store, known_work_entries
) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        json.dumps(
            {
                "case_name": "linked page: mal:anime:65144 from anilist:anime:163794",
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
    [correction] = _correct(
        known_work_entries, store, ReviewFile(review_path)
    ).corrections
    assert (
        correction.is_added_page,
        correction.is_part_page,
        correction.reason,
    ) == (True, True, "review answer")


def test_correct_linked_pages_answers_adding_two_pages_of_provider_add_neither(
    tmp_path, store, known_work_entries, caplog
) -> None:
    other_mal = "https://myanimelist.net/anime/65145"
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_name": f"linked page: {key} from anilist:anime:163794",
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
        linked_page_corrections = _correct(
            known_work_entries, store, ReviewFile(review_path)
        )
    assert (
        linked_page_corrections.corrections,
        linked_page_corrections.conflicting_page_keys,
    ) == ((), ("mal:anime:65144", "mal:anime:65145"))
    assert "anilist:anime:163794" in caplog.text

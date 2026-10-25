import pytest
from seed_cleaning.corrected_entries import (
    CorrectedEntry,
    UnknownSeedEntryError,
    build_corrected_entries,
)
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPage,
    ProviderPageStore,
    ReviewFile,
    WorkEntry,
)

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
TARGET_MAL = "https://myanimelist.net/anime/9"
OLD_PLANET = "https://www.anime-planet.com/anime/old-name"
NEW_PLANET = "https://www.anime-planet.com/anime/new-name"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {
        "title": title,
        "type": "TV",
        "sources": list(addresses),
        "animeSeason": {"season": "SPRING", "year": 2020},
    }


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
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL),
        ]
    )


@pytest.fixture
def entry(known_work_entries) -> WorkEntry:
    return next(iter(known_work_entries))


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


@pytest.fixture
def review_file(tmp_path) -> ReviewFile:
    return ReviewFile(tmp_path / "review.jsonl")


def _correct(entry, store, known_work_entries, review_file):
    return correct_entry_groups([[entry]], store, known_work_entries, review_file)


def test_build_corrected_entries_moved_page_leaves_entry_and_joins_seed_entry(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    corrections = _correct(entry, store, known_work_entries, review_file).corrections
    assert build_corrected_entries(corrections, known_work_entries, store) == [
        CorrectedEntry(
            seed_entry_reference_page="mal:anime:1",
            entry=entry.model_copy(
                update={
                    "addresses_by_page_key": {
                        "mal:anime:1": MAL,
                        "kitsu:anime:1": KITSU_ONE,
                    }
                }
            ),
        ),
        CorrectedEntry(
            seed_entry_reference_page="mal:anime:9",
            entry=WorkEntry(
                title="Show Movie",
                type="TV",
                year=2020,
                season="SPRING",
                addresses_by_page_key={
                    "mal:anime:9": TARGET_MAL,
                    "kitsu:anime:2": KITSU_TWO,
                },
            ),
        ),
    ]


def test_build_corrected_entries_page_moved_into_entry_corrected_itself_joins_corrected_entry(
    store, review_file
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL, NEW_PLANET, OLD_PLANET),
        ]
    )
    store.add_pages(
        _page(KITSU_ONE, MAL),
        _page(KITSU_TWO, TARGET_MAL),
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]),
        _page(NEW_PLANET),
    )
    corrections = correct_entry_groups(
        [[work] for work in works], store, works, review_file
    ).corrections
    corrected = build_corrected_entries(corrections, works, store)
    [target] = [
        corrected_entry.entry
        for corrected_entry in corrected
        if corrected_entry.seed_entry_reference_page == "mal:anime:9"
    ]
    assert list(target.addresses_by_page_key) == [
        "mal:anime:9",
        "animeplanet:anime:new-name",
        "kitsu:anime:2",
    ]


def test_build_corrected_entries_renamed_old_address_dropped(
    store, review_file
) -> None:
    works = KnownWorkEntriesFromSeed([_seed_entry("Show", MAL, NEW_PLANET, OLD_PLANET)])
    store.add_pages(
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]), _page(NEW_PLANET)
    )
    corrections = _correct(next(iter(works)), store, works, review_file).corrections
    [corrected] = build_corrected_entries(corrections, works, store)
    assert list(corrected.entry.addresses_by_page_key) == [
        "mal:anime:1",
        "animeplanet:anime:new-name",
    ]


def test_build_corrected_entries_main_page_listed_first(
    entry, store, known_work_entries
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key=key,
            address=address,
            placement_word="work 1",
            is_main_page=key == "kitsu:anime:2",
            reason="review answer",
        )
        for key, address in (("kitsu:anime:1", KITSU_ONE), ("kitsu:anime:2", KITSU_TWO))
    ]
    [corrected] = build_corrected_entries(corrections, known_work_entries, store)
    assert list(corrected.entry.addresses_by_page_key) == [
        "kitsu:anime:2",
        "mal:anime:1",
        "kitsu:anime:1",
    ]


def test_build_corrected_entries_added_page_joins_entry(
    store, known_work_entries
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:9",
            page_key="anilist:anime:9",
            address="https://anilist.co/anime/9",
            placement_word="work 1",
            is_added_page=True,
            is_part_page=True,
            reason="review answer",
        )
    ]
    [corrected] = build_corrected_entries(corrections, known_work_entries, store)
    assert (
        corrected.seed_entry_reference_page,
        corrected.entry.addresses_by_page_key,
    ) == (
        "mal:anime:9",
        {"mal:anime:9": TARGET_MAL, "anilist:anime:9": "https://anilist.co/anime/9"},
    )


def test_build_corrected_entries_work_pages_become_new_entry_from_trusted_page(
    entry, store, known_work_entries
) -> None:
    store.add_pages(
        _page(
            KITSU_TWO, title="Show: The Movie", type="MOVIE", year=2021, season="FALL"
        )
    )
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:2",
            address=KITSU_TWO,
            placement_word="work 2",
            new_entry_key="mal:anime:1 work 2",
            reason="review answer",
        )
    ]
    assert build_corrected_entries(corrections, known_work_entries, store)[
        1
    ] == CorrectedEntry(
        seed_entry_reference_page=None,
        entry=WorkEntry(
            title="Show: The Movie",
            type="MOVIE",
            year=2021,
            season="FALL",
            addresses_by_page_key={"kitsu:anime:2": KITSU_TWO},
        ),
    )


def test_build_corrected_entries_page_without_details_new_entry_takes_entry_title(
    entry, store, known_work_entries, caplog
) -> None:
    ann = "https://animenewsnetwork.com/encyclopedia/anime.php?id=5"
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="ann:anime:5",
            address=ann,
            placement_word="other work",
            new_entry_key="mal:anime:1 ann:anime:5",
            reason="review answer",
        )
    ]
    assert build_corrected_entries(corrections, known_work_entries, store)[
        1
    ] == CorrectedEntry(
        seed_entry_reference_page=None,
        entry=WorkEntry(
            title="Show", type="UNKNOWN", addresses_by_page_key={"ann:anime:5": ann}
        ),
    )
    assert "ann:anime:5" in caplog.text


def test_build_corrected_entries_entry_missing_from_seed_raises(
    store, known_work_entries
) -> None:
    correction = SeedCorrection(
        entry_reference_page="mal:anime:404",
        page_key="kitsu:anime:1",
        address=KITSU_ONE,
        placement_word="other work",
        reason="review answer",
    )
    with pytest.raises(UnknownSeedEntryError):
        build_corrected_entries([correction], known_work_entries, store)


CUE_ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=24629"
CUE_MAL = "https://myanimelist.net/anime/43735"
CUE_KITSU = "https://kitsu.app/anime/44979"


@pytest.fixture
def split_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [_seed_entry("CUE!", CUE_ANN), _seed_entry("Cue!", CUE_MAL, CUE_KITSU)]
    )


def test_build_corrected_entries_entry_left_without_pages_gives_no_entry(
    split_work_entries, store
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="ann:anime:24629",
            page_key="ann:anime:24629",
            address=CUE_ANN,
            placement_word="work 1",
            target_entry_reference_page="mal:anime:43735",
            reason="review answer",
        )
    ]
    corrected_entries = build_corrected_entries(corrections, split_work_entries, store)
    assert corrected_entries[0] == CorrectedEntry(
        seed_entry_reference_page="ann:anime:24629", entry=None
    )
    assert corrected_entries[1].entry is not None
    assert list(corrected_entries[1].entry.addresses_by_page_key) == [
        "mal:anime:43735",
        "kitsu:anime:44979",
        "ann:anime:24629",
    ]


def test_build_corrected_entries_pages_of_two_entries_with_one_new_entry_key_form_one_entry(
    split_work_entries, store
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page=reference_page,
            page_key=page_key,
            address=address,
            placement_word="work 2",
            new_entry_key="mal:anime:43735 work 2",
            reason="review answer",
        )
        for reference_page, page_key, address in [
            ("ann:anime:24629", "ann:anime:24629", CUE_ANN),
            ("mal:anime:43735", "kitsu:anime:44979", CUE_KITSU),
        ]
    ]
    new_entries = [
        corrected.entry
        for corrected in build_corrected_entries(corrections, split_work_entries, store)
        if corrected.seed_entry_reference_page is None
    ]
    assert [
        list(new_entry.addresses_by_page_key)
        for new_entry in new_entries
        if new_entry is not None
    ] == [["ann:anime:24629", "kitsu:anime:44979"]]


def test_build_corrected_entries_later_work_page_without_move_stays(
    split_work_entries, store
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:43735",
            page_key="kitsu:anime:44979",
            address=CUE_KITSU,
            placement_word="work 2",
            reason="review answer",
        )
    ]
    [corrected] = build_corrected_entries(corrections, split_work_entries, store)
    assert corrected.entry is not None
    assert "kitsu:anime:44979" in corrected.entry.addresses_by_page_key


def test_build_corrected_entries_new_entry_without_details_takes_most_trusted_entry_title(
    split_work_entries, store
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page=reference_page,
            page_key=page_key,
            address=address,
            placement_word="work 2",
            new_entry_key="mal:anime:43735 work 2",
            reason="review answer",
        )
        for reference_page, page_key, address in [
            ("ann:anime:24629", "ann:anime:24629", CUE_ANN),
            ("mal:anime:43735", "kitsu:anime:44979", CUE_KITSU),
        ]
    ]
    [new_entry] = [
        corrected.entry
        for corrected in build_corrected_entries(corrections, split_work_entries, store)
        if corrected.seed_entry_reference_page is None
    ]
    assert new_entry is not None
    assert new_entry.title == "Cue!"

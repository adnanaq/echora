import pytest
from seed_cleaning.corrected_seed import (
    EntryIdentity,
    PageOrigin,
    build_corrected_seed,
)
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import KnownWorkEntriesFromSeed, ProviderPageStore

CUE_ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=24629"
CUE_MAL = "https://myanimelist.net/anime/43735"
CUE_KITSU = "https://kitsu.app/anime/44979"
OTHER_MAL = "https://myanimelist.net/anime/9"
ADDED_ANILIST = "https://anilist.co/anime/9"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {"title": title, "type": "TV", "sources": list(addresses)}


@pytest.fixture
def known_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            _seed_entry("CUE!", CUE_ANN),
            _seed_entry("Cue!", CUE_MAL, CUE_KITSU),
            _seed_entry("Other", OTHER_MAL),
        ]
    )


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


def _correction(**fields) -> SeedCorrection:
    return SeedCorrection(
        **{"placement_word": "work 1", "reason": "review answer"} | fields
    )


def test_build_corrected_seed_without_corrections_keeps_seed_entries(
    known_work_entries, store
) -> None:
    corrected_seed = build_corrected_seed([], known_work_entries, store)
    assert [entry.reference_page for entry in corrected_seed.known_work_entries] == [
        "ann:anime:24629",
        "mal:anime:43735",
        "mal:anime:9",
    ]
    assert corrected_seed.identity_by_reference_page["mal:anime:9"] == EntryIdentity(
        "mal:anime:9", None
    )
    assert corrected_seed.find_page_origin(
        "mal:anime:43735", "kitsu:anime:44979"
    ) == PageOrigin("mal:anime:43735", CUE_KITSU)


def test_build_corrected_seed_moved_page_keeps_entry_listing_it_as_origin(
    known_work_entries, store
) -> None:
    corrections = [
        _correction(
            entry_reference_page="ann:anime:24629",
            page_key="ann:anime:24629",
            address=CUE_ANN,
            target_entry_reference_page="mal:anime:43735",
        )
    ]
    corrected_seed = build_corrected_seed(corrections, known_work_entries, store)
    assert [entry.reference_page for entry in corrected_seed.known_work_entries] == [
        "mal:anime:43735",
        "mal:anime:9",
    ]
    assert corrected_seed.find_page_origin(
        "mal:anime:43735", "ann:anime:24629"
    ) == PageOrigin("ann:anime:24629", CUE_ANN)


def test_build_corrected_seed_new_entry_identified_by_new_entry_key(
    known_work_entries, store
) -> None:
    corrections = [
        _correction(
            entry_reference_page="mal:anime:43735",
            page_key="kitsu:anime:44979",
            address=CUE_KITSU,
            placement_word="work 2",
            new_entry_key="mal:anime:43735 work 2",
        )
    ]
    corrected_seed = build_corrected_seed(corrections, known_work_entries, store)
    assert corrected_seed.identity_by_reference_page[
        "kitsu:anime:44979"
    ] == EntryIdentity(None, "mal:anime:43735 work 2")
    assert corrected_seed.find_page_origin(
        "kitsu:anime:44979", "kitsu:anime:44979"
    ) == PageOrigin("mal:anime:43735", CUE_KITSU)


def test_build_corrected_seed_added_page_marked_added(
    known_work_entries, store
) -> None:
    corrections = [
        _correction(
            entry_reference_page="mal:anime:9",
            page_key="anilist:anime:9",
            address=ADDED_ANILIST,
            is_added_page=True,
        )
    ]
    corrected_seed = build_corrected_seed(corrections, known_work_entries, store)
    assert corrected_seed.find_page_origin(
        "mal:anime:9", "anilist:anime:9"
    ) == PageOrigin("mal:anime:9", ADDED_ANILIST, is_added_page=True)

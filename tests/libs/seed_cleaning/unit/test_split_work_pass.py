import pytest
from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.seed_corrections import SeedCorrection
from seed_cleaning.split_work_pass import prepare_split_work_pass
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPage,
    ProviderPageStore,
    ReviewFile,
)

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=2"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {
        "title": title,
        "type": "TV",
        "sources": list(addresses),
        "animeSeason": {"season": "SPRING", "year": 2020},
    }


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


def test_prepare_split_work_pass_entry_with_open_first_pass_case_left_out(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO), _seed_entry("Show", ANN)]
    )
    first_pass = correct_entry_groups(
        [EntryGroup((next(iter(works)),))],
        store,
        works,
        ReviewFile(tmp_path / "review.jsonl"),
    )
    split_work_pass = prepare_split_work_pass(
        first_pass.corrections, first_pass.review_cases, works, store
    )
    assert [entry.reference_page for entry in split_work_pass.known_work_entries] == [
        "ann:anime:2"
    ]
    assert split_work_pass.candidates == []


def test_prepare_split_work_pass_new_entry_of_first_pass_becomes_candidate(
    store,
) -> None:
    works = KnownWorkEntriesFromSeed([_seed_entry("Show", MAL, KITSU_ONE)])
    store.add_pages(
        ProviderPage.from_canonical_record(
            KITSU_ONE, {"title": "Show", "type": "TV", "year": 2020}
        )
    )
    first_pass_corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:1",
            address=KITSU_ONE,
            placement_word="work 2",
            new_entry_key="mal:anime:1 work 2",
            reason="review answer",
        )
    ]
    split_work_pass = prepare_split_work_pass(first_pass_corrections, (), works, store)
    [candidates] = split_work_pass.candidates
    assert [entry.reference_page for entry in candidates.entries] == [
        "mal:anime:1",
        "kitsu:anime:1",
    ]
    assert (
        split_work_pass.corrected_seed.identity_by_reference_page[
            "kitsu:anime:1"
        ].new_entry_key
        == "mal:anime:1 work 2"
    )

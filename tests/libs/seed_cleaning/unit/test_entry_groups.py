import pytest
from seed_cleaning.entry_groups import EntryNotInSeedError, find_entry_groups
from seed_cleaning.split_work_candidates import SplitWorkCandidates
from work_matching import KnownWorkEntriesFromSeed


def _seed_entry(*addresses: str) -> dict:
    return {"title": "Show", "type": "TV", "sources": list(addresses)}


def _reference_pages(entry_groups) -> list[list[str]]:
    return [[entry.reference_page for entry in group] for group in entry_groups]


def test_find_entry_groups_doubled_entry_alone_forms_group() -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [_seed_entry("https://myanimelist.net/anime/1")]
    )
    [entry] = known_work_entries
    assert _reference_pages(find_entry_groups([entry], [], known_work_entries)) == [
        ["mal:anime:1"]
    ]


def test_find_entry_groups_candidates_sharing_entry_merge_in_seed_order() -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [
            _seed_entry("https://anilist.co/anime/1"),
            _seed_entry("https://myanimelist.net/anime/2"),
            _seed_entry("https://kitsu.app/anime/3"),
            _seed_entry("https://myanimelist.net/anime/4"),
        ]
    )
    first, second, third, fourth = known_work_entries
    candidate_groups = [
        SplitWorkCandidates(entries=(second, third)),
        SplitWorkCandidates(entries=(first, third)),
    ]
    assert _reference_pages(
        find_entry_groups([fourth, third], candidate_groups, known_work_entries)
    ) == [
        ["anilist:anime:1", "mal:anime:2", "kitsu:anime:3"],
        ["mal:anime:4"],
    ]


def test_find_entry_groups_entries_listing_same_page_stay_apart() -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [
            _seed_entry("https://myanimelist.net/anime/1", "https://kitsu.app/anime/1"),
            _seed_entry("https://myanimelist.net/anime/1"),
        ]
    )
    first, second = known_work_entries
    assert _reference_pages(
        find_entry_groups([first, second], [], known_work_entries)
    ) == [["mal:anime:1"], ["mal:anime:1"]]


def test_find_entry_groups_copy_of_entry_raises() -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [_seed_entry("https://myanimelist.net/anime/1")]
    )
    [entry] = known_work_entries
    with pytest.raises(EntryNotInSeedError):
        find_entry_groups([entry.model_copy()], [], known_work_entries)

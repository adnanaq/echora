from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.split_work_candidates import SplitWorkCandidates
from work_matching import WorkEntry


def _entry(address: str) -> WorkEntry:
    return WorkEntry.from_seed_entry(
        {"title": "Show", "type": "TV", "sources": [address]}
    )


def test_entry_group_from_candidates_keeps_evidence_and_shared_link_flag() -> None:
    entries = (
        _entry("https://myanimelist.net/anime/1"),
        _entry("https://anidb.net/anime/2"),
    )
    candidates = SplitWorkCandidates(
        entries=entries,
        candidate_evidence=("anidb:anime:2 links mal:anime:1",),
        has_shared_tying_link=True,
    )
    assert EntryGroup.from_candidates(candidates) == EntryGroup(
        entries, ("anidb:anime:2 links mal:anime:1",), has_shared_tying_link=True
    )


def test_entry_group_single_entry_has_no_evidence() -> None:
    entry = _entry("https://myanimelist.net/anime/1")
    assert EntryGroup((entry,)) == EntryGroup((entry,), (), has_shared_tying_link=False)

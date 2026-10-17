import pytest
from work_matching.split_work_check import SplitWorkReason, decide_split_work


def test_decide_split_work_linked_entries_one_work(matching_cases) -> None:
    entries, pages = matching_cases["cue"]
    decision = decide_split_work(entries, pages)
    assert (decision.is_one_work, decision.reason) == (True, SplitWorkReason.LINKED)


def test_decide_split_work_unlinked_entries_review(linked_page, work_entry) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        linked_page("https://myanimelist.net/anime/1", "https://kitsu.app/anime/1")
    ]
    decision = decide_split_work(entries, pages)
    assert (decision.is_one_work, decision.reason) == (
        False,
        SplitWorkReason.NOT_LINKED,
    )


def test_decide_split_work_entry_linked_through_middle_entry_one_work(
    linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
        work_entry("https://kitsu.app/anime/1"),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1"),
        linked_page("https://kitsu.app/anime/1", "https://anilist.co/anime/1"),
    ]
    assert decide_split_work(entries, pages).is_one_work


def test_decide_split_work_types_differ_review(linked_page, work_entry) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1", work_type="TV"),
        work_entry("https://anilist.co/anime/1", work_type="MOVIE"),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1")
    ]
    assert decide_split_work(entries, pages).reason is SplitWorkReason.TYPE_DIFFERS


def test_decide_split_work_years_differ_review(linked_page, work_entry) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1", year=2020),
        work_entry("https://anilist.co/anime/1", year=2021),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1")
    ]
    assert decide_split_work(entries, pages).reason is SplitWorkReason.YEAR_DIFFERS


def test_decide_split_work_unknown_type_and_missing_year_agree(
    linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1", work_type="UNKNOWN", year=None),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1")
    ]
    assert decide_split_work(entries, pages).is_one_work


@pytest.mark.parametrize("entry_count", [0, 1])
def test_decide_split_work_fewer_than_two_entries_review(
    work_entry, entry_count: int
) -> None:
    entries = [work_entry("https://myanimelist.net/anime/1")][:entry_count]
    decision = decide_split_work(entries, [])
    assert (decision.is_one_work, decision.reason) == (
        False,
        SplitWorkReason.TOO_FEW_ENTRIES,
    )


def test_decide_split_work_shared_page_without_link_review(work_entry) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1", "https://kitsu.app/anime/1"),
        work_entry("https://anilist.co/anime/1", "https://kitsu.app/anime/1"),
    ]
    assert decide_split_work(entries, []).reason is SplitWorkReason.NOT_LINKED

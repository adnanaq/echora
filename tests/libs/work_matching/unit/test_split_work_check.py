import pytest
from work_matching.provider_page import ProviderPage
from work_matching.split_work_check import (
    SplitWorkReviewReason,
    list_split_work_review_reasons,
)

MAL = "https://myanimelist.net/anime/1"
ANILIST = "https://anilist.co/anime/1"
KITSU = "https://kitsu.app/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"


def test_list_split_work_review_reasons_linked_entries_give_none(
    matching_cases,
) -> None:
    entries, pages = matching_cases["cue"]
    assert list_split_work_review_reasons(entries, pages) == ()


def test_list_split_work_review_reasons_unlinked_entries_not_tied(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST)]
    assert list_split_work_review_reasons(entries, [linked_page(MAL, KITSU)]) == (
        SplitWorkReviewReason.NOT_TIED_BY_LINKS,
    )


def test_list_split_work_review_reasons_entry_linked_through_middle_entry_gives_none(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST), work_entry(KITSU)]
    pages = [linked_page(ANILIST, MAL), linked_page(KITSU, ANILIST)]
    assert list_split_work_review_reasons(entries, pages) == ()


def test_list_split_work_review_reasons_types_differ(linked_page, work_entry) -> None:
    entries = [work_entry(MAL, work_type="TV"), work_entry(ANILIST, work_type="MOVIE")]
    assert list_split_work_review_reasons(entries, [linked_page(ANILIST, MAL)]) == (
        SplitWorkReviewReason.TYPE_DIFFERS,
    )


def test_list_split_work_review_reasons_years_differ(linked_page, work_entry) -> None:
    entries = [work_entry(MAL, year=2020), work_entry(ANILIST, year=2021)]
    assert list_split_work_review_reasons(entries, [linked_page(ANILIST, MAL)]) == (
        SplitWorkReviewReason.YEAR_DIFFERS,
    )


def test_list_split_work_review_reasons_every_failed_check_listed(work_entry) -> None:
    entries = [
        work_entry(MAL, work_type="TV", year=2020),
        work_entry(ANILIST, work_type="MOVIE", year=2021),
    ]
    assert list_split_work_review_reasons(entries, []) == (
        SplitWorkReviewReason.NOT_TIED_BY_LINKS,
        SplitWorkReviewReason.TYPE_DIFFERS,
        SplitWorkReviewReason.YEAR_DIFFERS,
        SplitWorkReviewReason.NO_PAGE_CHECKED,
    )


def test_list_split_work_review_reasons_unknown_type_and_missing_year_agree(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, work_type="UNKNOWN", year=None), work_entry(ANILIST)]
    assert list_split_work_review_reasons(entries, [linked_page(ANILIST, MAL)]) == ()


@pytest.mark.parametrize("entry_count", [0, 1])
def test_list_split_work_review_reasons_fewer_than_two_entries_give_none(
    work_entry, entry_count: int
) -> None:
    entries = [work_entry(MAL)][:entry_count]
    assert list_split_work_review_reasons(entries, []) == ()


def test_list_split_work_review_reasons_shared_page_without_link_not_tied(
    work_entry,
) -> None:
    entries = [work_entry(MAL, KITSU), work_entry(ANILIST, KITSU)]
    assert list_split_work_review_reasons(entries, []) == (
        SplitWorkReviewReason.NOT_TIED_BY_LINKS,
        SplitWorkReviewReason.NO_PAGE_CHECKED,
    )


def test_list_split_work_review_reasons_only_dead_pages_nothing_checked(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST)]
    pages = [
        ProviderPage.from_dead_address(MAL),
        ProviderPage.from_dead_address(ANILIST),
    ]
    assert SplitWorkReviewReason.NO_PAGE_CHECKED in list_split_work_review_reasons(
        entries, pages
    )


def test_list_split_work_review_reasons_shared_tying_link_listed(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST)]
    assert list_split_work_review_reasons(
        entries, [linked_page(ANILIST, MAL)], has_shared_tying_link=True
    ) == (SplitWorkReviewReason.LINK_SHARED_WITH_OTHER_ENTRIES,)


def test_list_split_work_review_reasons_mal_page_outside_group_listed(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST), work_entry(KITSU)]
    pages = [linked_page(ANILIST, MAL), linked_page(KITSU, ANILIST, OTHER_MAL)]
    assert list_split_work_review_reasons(entries, pages) == (
        SplitWorkReviewReason.LINKS_MAL_PAGE_OUTSIDE_GROUP,
    )


def test_list_split_work_review_reasons_partly_linked_group_not_tied(
    matching_cases,
) -> None:
    entries, pages = matching_cases["shin_oishinbo"]
    assert list_split_work_review_reasons(entries, pages) == (
        SplitWorkReviewReason.NOT_TIED_BY_LINKS,
    )

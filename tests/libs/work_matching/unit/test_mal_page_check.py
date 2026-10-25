from work_matching.mal_page_check import check_with_mal_pages
from work_matching.page_details_comparison import MalPageComparison
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_key

AWARE_MAL = "https://myanimelist.net/anime/32956"
AWARE_ANISEARCH = "https://anisearch.com/anime/11255"
AWARE_ANIDB = "https://anidb.net/anime/12006"


def _detailed_page(
    address: str,
    page_type: str,
    episode_count: int | None,
    aired_from: str,
    aired_to: str | None,
) -> ProviderPage:
    return ProviderPage(
        key=get_page_key(address),
        address=address,
        type=page_type,
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
    )


def test_check_with_mal_pages_part_page_marked_and_unfetched_page_left_out(
    work_entry,
) -> None:
    entries = [work_entry(AWARE_ANIDB, AWARE_ANISEARCH), work_entry(AWARE_MAL)]
    pages = [
        _detailed_page(AWARE_MAL, "TV", 230, "2016-04-07", "2022-03-03"),
        _detailed_page(AWARE_ANISEARCH, "TV", 39, "2016-04-07", "2017-02-16"),
    ]
    mal_page_check = check_with_mal_pages(
        entries,
        [],
        dict.fromkeys(
            ["anidb:anime:12006", "anisearch:anime:11255", "mal:anime:32956"],
            "work 1",
        ),
        pages,
    )
    assert mal_page_check == (
        {"anisearch:anime:11255": "work 1, part", "mal:anime:32956": "work 1"},
        (MalPageComparison.DETAILS_UNKNOWN,),
    )


def test_check_with_mal_pages_work_without_mal_page_left_out(work_entry) -> None:
    entries = [
        work_entry("https://anilist.co/anime/1"),
        work_entry("https://kitsu.app/anime/1"),
    ]
    mal_page_check = check_with_mal_pages(
        entries,
        [],
        {"anilist:anime:1": "work 1", "kitsu:anime:1": "work 1"},
        [],
    )
    assert mal_page_check == ({}, (MalPageComparison.NO_MAL_PAGE,))


def test_check_with_mal_pages_other_work_kept_unchecked(work_entry) -> None:
    entries = [work_entry(AWARE_MAL), work_entry("https://kitsu.app/anime/2")]
    suggested_placements = {"mal:anime:32956": "work 1", "kitsu:anime:2": "other work"}
    assert check_with_mal_pages(entries, [], suggested_placements, []) == (
        suggested_placements,
        (),
    )

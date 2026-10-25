import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import (
    combine_group_entries,
    list_mal_page_review_reasons,
    read_group_answer_placements,
    suggest_group_page_placements,
)
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.page_details_comparison import MalPageComparison
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_key

MAL = "https://myanimelist.net/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"
ANILIST = "https://anilist.co/anime/1"
KITSU = "https://kitsu.app/anime/1"
ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=1"
SIMKL = "https://simkl.com/anime/1"
OTHER_KITSU = "https://kitsu.app/anime/2"


def _detailed_page(
    address: str,
    page_type: str,
    episode_count: int | None,
    aired_from: str,
    aired_to: str | None,
    *linked_addresses: str,
) -> ProviderPage:
    return ProviderPage(
        key=get_page_key(address),
        address=address,
        type=page_type,
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
        linked_page_keys=tuple(sorted(map(get_page_key, linked_addresses))),
    )


def _season_page(address: str, *linked_addresses: str) -> ProviderPage:
    return _detailed_page(
        address, "TV", 12, "2020-01-05", "2020-03-29", *linked_addresses
    )


def _decide(entries, pages):
    return decide_doubled_providers(
        combine_group_entries(entries), pages, KnownWorkEntriesFromSeed([])
    )


def test_combine_group_entries_single_entry_returned_unchanged(work_entry) -> None:
    entry = work_entry(MAL, KITSU)
    assert combine_group_entries([entry]) is entry


def test_combine_group_entries_takes_details_of_most_trusted_entry(
    work_entry,
) -> None:
    entries = [
        work_entry(ANN, work_type="MOVIE", year=2021),
        work_entry(MAL, KITSU, work_type="TV", year=2020),
    ]
    combined_entry = combine_group_entries(entries)
    assert (
        combined_entry.type,
        combined_entry.year,
        list(combined_entry.addresses_by_page_key),
    ) == ("TV", 2020, ["ann:anime:1", "mal:anime:1", "kitsu:anime:1"])


def test_suggest_group_page_placements_linked_entries_one_work(
    linked_page, work_entry
) -> None:
    entries = [work_entry(ANILIST), work_entry(MAL, KITSU)]
    pages = [_season_page(ANILIST, MAL), _season_page(MAL)]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "anilist:anime:1": "work 1",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 1",
    }


def test_suggest_group_page_placements_two_mal_pages_left_out(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, KITSU), work_entry(OTHER_MAL)]
    pages = [linked_page(MAL, KITSU), linked_page(OTHER_MAL, KITSU)]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {}


def test_suggest_group_page_placements_separate_links_numbered_by_trust(
    linked_page, work_entry
) -> None:
    entries = [work_entry(ANILIST, MAL), work_entry(KITSU, OTHER_MAL)]
    pages = [linked_page(ANILIST, MAL), linked_page(KITSU, OTHER_MAL)]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "anilist:anime:1": "work 1",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 2",
        "mal:anime:2": "work 2",
    }


def test_suggest_group_page_placements_unlinked_page_follows_agreeing_entry(
    linked_page, work_entry
) -> None:
    entries = [work_entry(ANILIST), work_entry(MAL, KITSU, SIMKL)]
    pages = [
        _season_page(ANILIST),
        _season_page(MAL, ANILIST),
        _season_page(KITSU, MAL),
    ]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "anilist:anime:1": "work 1",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 1",
        "simkl:anime:1": "work 1",
    }


def test_suggest_group_page_placements_unlinked_page_of_mixed_entry_left_out(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, KITSU, SIMKL), work_entry(ANILIST, OTHER_MAL)]
    pages = [
        _season_page(MAL),
        _season_page(OTHER_MAL),
        _season_page(ANILIST, MAL),
        _season_page(KITSU, OTHER_MAL),
    ]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 2",
        "anilist:anime:1": "work 1",
        "mal:anime:2": "work 2",
    }


def test_suggest_group_page_placements_unlinked_doubled_page_left_out(
    linked_page, work_entry
) -> None:
    entries = [work_entry(ANN), work_entry(MAL, KITSU, OTHER_KITSU)]
    pages = [linked_page(MAL, ANN), linked_page(KITSU, MAL)]
    assert "kitsu:anime:2" not in suggest_group_page_placements(
        entries, _decide(entries, pages), pages
    )


@pytest.fixture
def cue_like_group(linked_page, work_entry):
    entries = [work_entry(ANN), work_entry(MAL, KITSU)]
    return entries, _decide(entries, [linked_page(MAL, ANN)])


def test_read_group_answer_placements_every_page_placed_gives_placements(
    cue_like_group,
) -> None:
    entries, decisions = cue_like_group
    answer = {
        "ann:anime:1": "work 1",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 1",
    }
    assert read_group_answer_placements(
        answer, entries, decisions, {}
    ) == dict.fromkeys(answer, PagePlacement("work 1"))


@pytest.mark.parametrize(
    "answer",
    [
        {"ann:anime:1": "work 1", "mal:anime:1": "work 1"},
        {"ann:anime:1": "maybe", "mal:anime:1": "work 1", "kitsu:anime:1": "work 1"},
    ],
)
def test_read_group_answer_placements_invalid_answer_gives_none(
    cue_like_group, answer: dict[str, str]
) -> None:
    entries, decisions = cue_like_group
    assert read_group_answer_placements(answer, entries, decisions, {}) is None


def test_read_group_answer_placements_two_mal_pages_in_one_work_gives_none(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    answer = {"mal:anime:1": "work 1", "mal:anime:2": "work 1"}
    assert (
        read_group_answer_placements(answer, entries, _decide(entries, []), {}) is None
    )


def test_read_group_answer_placements_two_mal_pages_in_separate_works_accepted(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    answer = {"mal:anime:1": "work 1", "mal:anime:2": "work 2"}
    assert read_group_answer_placements(answer, entries, _decide(entries, []), {}) == {
        "mal:anime:1": PagePlacement("work 1"),
        "mal:anime:2": PagePlacement("work 2"),
    }


def test_read_group_answer_placements_single_entry_needs_only_doubled_pages(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, KITSU, "https://kitsu.app/anime/2")]
    pages = [linked_page(KITSU, MAL), linked_page("https://kitsu.app/anime/2", MAL)]
    answer = {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "work 1"}
    assert read_group_answer_placements(
        answer, entries, _decide(entries, pages), {}
    ) == {
        "kitsu:anime:1": PagePlacement("work 1", is_main_page=True),
        "kitsu:anime:2": PagePlacement("work 1"),
    }


def test_read_group_answer_placements_work_of_two_entries_without_reference_page_gives_none(
    linked_page, work_entry
) -> None:
    entries = [work_entry(ANN, SIMKL), work_entry(MAL, KITSU)]
    decisions = _decide(entries, [linked_page(MAL, ANN)])
    answer = {
        "ann:anime:1": "work 1",
        "simkl:anime:1": "work 2",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 2",
    }
    assert read_group_answer_placements(answer, entries, decisions, {}) is None


def test_read_group_answer_placements_part_reference_pages_only_accepted(
    cue_like_group,
) -> None:
    entries, decisions = cue_like_group
    answer = {
        "ann:anime:1": "work 1, part",
        "mal:anime:1": "work 1, part",
        "kitsu:anime:1": "work 1",
    }
    assert read_group_answer_placements(answer, entries, decisions, {}) is not None


def test_suggest_group_page_placements_shorter_run_marked_part_and_unfetched_page_left_out(
    work_entry,
) -> None:
    entries = [
        work_entry(
            "https://anidb.net/anime/12006", "https://anisearch.com/anime/11255"
        ),
        work_entry(
            "https://myanimelist.net/anime/32956", "https://kitsu.app/anime/12168"
        ),
    ]
    pages = [
        _detailed_page(
            "https://anisearch.com/anime/11255", "TV", 39, "2016-04-07", "2017-02-16"
        ),
        _detailed_page(
            "https://myanimelist.net/anime/32956", "TV", 230, "2016-04-07", "2022-03-03"
        ),
        _detailed_page(
            "https://kitsu.app/anime/12168",
            "TV",
            230,
            "2016-04-07",
            "2022-03-03",
            "https://anidb.net/anime/12006",
            "https://myanimelist.net/anime/32956",
        ),
    ]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "anisearch:anime:11255": "work 1, part",
        "mal:anime:32956": "work 1",
        "kitsu:anime:12168": "work 1",
    }


def _dirty_pair_flash_group(work_entry):
    entries = [
        work_entry(
            "https://myanimelist.net/anime/1802", "https://kitsu.app/anime/1621"
        ),
        work_entry("https://anidb.net/anime/402", "https://anisearch.com/anime/1296"),
    ]
    pages = [
        _detailed_page(
            "https://myanimelist.net/anime/1802", "OVA", 6, "1994-01-20", "1994-06-20"
        ),
        _detailed_page(
            "https://kitsu.app/anime/1621",
            "OVA",
            6,
            "1994-01-20",
            "1994-06-20",
            "https://anidb.net/anime/402",
            "https://myanimelist.net/anime/1802",
        ),
        _detailed_page(
            "https://anisearch.com/anime/1296", "OVA", 16, "1994-01-20", "1996-04-24"
        ),
    ]
    return entries, pages


def test_suggest_group_page_placements_page_covering_several_works_left_out(
    work_entry,
) -> None:
    entries, pages = _dirty_pair_flash_group(work_entry)
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "mal:anime:1802": "work 1",
        "kitsu:anime:1621": "work 1",
    }


def test_list_mal_page_review_reasons_lists_why_suggestions_left_out(
    work_entry,
) -> None:
    entries, pages = _dirty_pair_flash_group(work_entry)
    assert list_mal_page_review_reasons(entries, _decide(entries, pages), pages) == (
        MalPageComparison.SEVERAL_WORKS,
        MalPageComparison.DETAILS_UNKNOWN,
    )


def test_suggest_group_page_placements_page_listed_with_mal_page_kept_when_count_unknown(
    work_entry,
) -> None:
    entries = [
        work_entry(
            "https://myanimelist.net/anime/63219",
            "https://anilist.co/anime/205116",
            "https://kitsu.app/anime/50804",
        ),
        work_entry("https://anisearch.com/anime/21800"),
    ]
    pages = [
        _detailed_page(
            "https://myanimelist.net/anime/63219", "TV", 12, "2026-07-05", "2026-09-20"
        ),
        _detailed_page(
            "https://anilist.co/anime/205116",
            "TV SHORT",
            12,
            "2026-07-04",
            "2026-09-20",
            "https://myanimelist.net/anime/63219",
        ),
        _detailed_page(
            "https://kitsu.app/anime/50804",
            "TV",
            None,
            "2026-07-05",
            None,
            "https://anilist.co/anime/205116",
        ),
        _detailed_page(
            "https://anisearch.com/anime/21800", "TV", None, "2026-07-04", None
        ),
    ]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "mal:anime:63219": "work 1",
        "anilist:anime:205116": "work 1",
        "kitsu:anime:50804": "work 1",
    }


def test_suggest_group_page_placements_doubled_pages_keep_decision_marks(
    work_entry,
) -> None:
    entries = [
        work_entry(MAL, ANILIST),
        work_entry(OTHER_KITSU, "https://anilist.co/anime/2"),
    ]
    pages = [
        _detailed_page(MAL, "TV", 24, "2020-01-01", "2020-06-30"),
        _detailed_page(ANILIST, "TV", 24, "2020-01-01", "2020-06-30", MAL),
        _detailed_page(OTHER_KITSU, "TV", 12, "2020-04-01", "2020-06-30", MAL),
        _detailed_page(
            "https://anilist.co/anime/2", "TV", 12, "2020-04-01", "2020-06-30", MAL
        ),
    ]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "mal:anime:1": "work 1",
        "anilist:anime:1": "work 1, part",
        "kitsu:anime:2": "work 1, part",
        "anilist:anime:2": "work 1, part",
    }


def test_suggest_group_page_placements_doubled_parts_of_mal_work_marked_part(
    matching_cases,
) -> None:
    entries, pages = matching_cases["douluo_dalu_arcs"]
    suggested_placements = suggest_group_page_placements(
        entries, _decide(entries, pages), pages
    )
    assert {
        key: placement
        for key, placement in suggested_placements.items()
        if key.startswith("kitsu:")
    } == {
        "kitsu:anime:42037": "work 1, part",
        "kitsu:anime:46536": "work 1, part",
        "kitsu:anime:46535": "work 1, part",
    }


def test_suggest_group_page_placements_several_works_page_left_out_films_kept_apart(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ajin_films"]
    suggested_placements = suggest_group_page_placements(
        entries, _decide(entries, pages), pages
    )
    work_by_mal_page = {
        key: suggested_placements.get(key)
        for key in ("mal:anime:30868", "mal:anime:30869", "mal:anime:30870")
    }
    assert len(set(work_by_mal_page.values())) == 3
    assert None not in work_by_mal_page.values()
    assert (
        suggested_placements["kitsu:anime:10949"] == work_by_mal_page["mal:anime:30868"]
    )
    assert "anidb:anime:11265" not in suggested_placements
    assert not any(
        placement.endswith(", part") for placement in suggested_placements.values()
    )

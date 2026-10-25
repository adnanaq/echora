import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import (
    combine_group_entries,
    read_group_answer_placements,
    suggest_group_page_placements,
)
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.page_placement import PagePlacement

MAL = "https://myanimelist.net/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"
ANILIST = "https://anilist.co/anime/1"
KITSU = "https://kitsu.app/anime/1"
ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=1"
SIMKL = "https://simkl.com/anime/1"
OTHER_KITSU = "https://kitsu.app/anime/2"


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
    pages = [linked_page(ANILIST, MAL)]
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
    entries = [work_entry(ANN), work_entry(MAL, KITSU, SIMKL)]
    pages = [linked_page(MAL, ANN), linked_page(KITSU, MAL)]
    assert suggest_group_page_placements(entries, _decide(entries, pages), pages) == {
        "ann:anime:1": "work 1",
        "mal:anime:1": "work 1",
        "kitsu:anime:1": "work 1",
        "simkl:anime:1": "work 1",
    }


def test_suggest_group_page_placements_unlinked_page_of_mixed_entry_left_out(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, KITSU, SIMKL), work_entry(ANILIST, OTHER_MAL)]
    pages = [linked_page(ANILIST, MAL), linked_page(KITSU, OTHER_MAL)]
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
    assert read_group_answer_placements(answer, entries, decisions) == dict.fromkeys(
        answer, PagePlacement("work 1")
    )


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
    assert read_group_answer_placements(answer, entries, decisions) is None


def test_read_group_answer_placements_two_mal_pages_in_one_work_gives_none(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    answer = {"mal:anime:1": "work 1", "mal:anime:2": "work 1"}
    assert read_group_answer_placements(answer, entries, _decide(entries, [])) is None


def test_read_group_answer_placements_two_mal_pages_in_separate_works_accepted(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    answer = {"mal:anime:1": "work 1", "mal:anime:2": "work 2"}
    assert read_group_answer_placements(answer, entries, _decide(entries, [])) == {
        "mal:anime:1": PagePlacement("work 1"),
        "mal:anime:2": PagePlacement("work 2"),
    }


def test_read_group_answer_placements_single_entry_needs_only_doubled_pages(
    linked_page, work_entry
) -> None:
    entries = [work_entry(MAL, KITSU, "https://kitsu.app/anime/2")]
    pages = [linked_page(KITSU, MAL), linked_page("https://kitsu.app/anime/2", MAL)]
    answer = {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "work 1"}
    assert read_group_answer_placements(answer, entries, _decide(entries, pages)) == {
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
    assert read_group_answer_placements(answer, entries, decisions) is None


def test_read_group_answer_placements_part_reference_pages_only_accepted(
    cue_like_group,
) -> None:
    entries, decisions = cue_like_group
    answer = {
        "ann:anime:1": "work 1, part",
        "mal:anime:1": "work 1, part",
        "kitsu:anime:1": "work 1",
    }
    assert read_group_answer_placements(answer, entries, decisions) is not None

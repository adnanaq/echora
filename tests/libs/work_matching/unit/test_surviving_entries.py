from work_matching.page_placement import PagePlacement
from work_matching.surviving_entries import find_surviving_entries

MAL = "https://myanimelist.net/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"
ANILIST = "https://anilist.co/anime/1"
OTHER_ANILIST = "https://anilist.co/anime/2"
ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=1"


def test_find_surviving_entries_unplaced_reference_page_survives_as_first_work(
    work_entry,
) -> None:
    assert find_surviving_entries([work_entry(MAL, ANILIST)], {}) == {
        "work 1": "mal:anime:1"
    }


def test_find_surviving_entries_most_trusted_reference_page_survives(
    work_entry,
) -> None:
    entries = [work_entry(ANN), work_entry(ANILIST), work_entry(MAL)]
    placements = {
        "ann:anime:1": PagePlacement("work 1"),
        "anilist:anime:1": PagePlacement("work 1"),
        "mal:anime:1": PagePlacement("work 1"),
    }
    assert find_surviving_entries(entries, placements) == {"work 1": "mal:anime:1"}


def test_find_surviving_entries_same_provider_tie_broken_by_reference_page(
    work_entry,
) -> None:
    entries = [work_entry(OTHER_ANILIST), work_entry(ANILIST)]
    placements = {
        "anilist:anime:2": PagePlacement("work 1"),
        "anilist:anime:1": PagePlacement("work 1"),
    }
    assert find_surviving_entries(entries, placements) == {"work 1": "anilist:anime:1"}


def test_find_surviving_entries_each_work_gets_own_survivor(work_entry) -> None:
    entries = [work_entry(MAL), work_entry(OTHER_MAL), work_entry(ANILIST)]
    placements = {
        "mal:anime:1": PagePlacement("work 2"),
        "mal:anime:2": PagePlacement("work 1"),
        "anilist:anime:1": PagePlacement("work 2"),
    }
    assert find_surviving_entries(entries, placements) == {
        "work 1": "mal:anime:2",
        "work 2": "mal:anime:1",
    }


def test_find_surviving_entries_part_reference_page_never_survives(
    work_entry,
) -> None:
    entries = [work_entry(MAL), work_entry(ANILIST)]
    placements = {
        "mal:anime:1": PagePlacement("work 1", is_part_page=True),
        "anilist:anime:1": PagePlacement("work 1"),
    }
    assert find_surviving_entries(entries, placements) == {"work 1": "anilist:anime:1"}


def test_find_surviving_entries_work_without_whole_reference_page_has_no_survivor(
    work_entry,
) -> None:
    entries = [work_entry(MAL, OTHER_ANILIST), work_entry(ANILIST)]
    placements = {
        "mal:anime:1": PagePlacement("work 1"),
        "anilist:anime:2": PagePlacement("work 2"),
        "anilist:anime:1": PagePlacement("other work"),
    }
    assert find_surviving_entries(entries, placements) == {"work 1": "mal:anime:1"}


def test_find_surviving_entries_only_part_reference_pages_most_trusted_part_survives(
    work_entry,
) -> None:
    entries = [work_entry(ANILIST), work_entry(MAL)]
    placements = {
        "anilist:anime:1": PagePlacement("work 1", is_part_page=True),
        "mal:anime:1": PagePlacement("work 1", is_part_page=True),
    }
    assert find_surviving_entries(entries, placements) == {"work 1": "mal:anime:1"}

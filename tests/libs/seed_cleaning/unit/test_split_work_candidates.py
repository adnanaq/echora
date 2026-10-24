import logging

import pytest
from seed_cleaning.split_work_candidates import find_split_work_candidates
from work_matching import KnownWorkEntriesFromSeed


def _candidates_by_title(seed_works) -> dict[str, tuple[list[str], str | None]]:
    return {
        candidates.entries[0].title.casefold(): (
            [entry.title for entry in candidates.entries],
            candidates.shared_provider,
        )
        for candidates in find_split_work_candidates(seed_works)
    }


@pytest.mark.parametrize(
    ("title", "entry_titles"),
    [
        ("cue!", ["CUE!", "Cue!"]),
        ("ghost meets gal!", ["Ghost Meets Gal!", "Ghost Meets Gal!"]),
        ("phantom busters", ["Phantom Busters", "Phantom Busters"]),
    ],
)
def test_find_split_work_candidates_entries_sharing_no_provider_kept_unmarked(
    seed_works, title: str, entry_titles: list[str]
) -> None:
    assert _candidates_by_title(seed_works)[title] == (entry_titles, None)


@pytest.mark.parametrize(
    ("title", "shared_provider"),
    [("dirty pair flash", "ann"), ("liar game", "ann"), ("hokuto no ken", "simkl")],
)
def test_find_split_work_candidates_entries_sharing_one_provider_kept_and_marked(
    seed_works, title: str, shared_provider: str
) -> None:
    assert _candidates_by_title(seed_works)[title][1] == shared_provider


def test_find_split_work_candidates_entries_sharing_two_providers_dropped_and_logged(
    seed_works, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        candidates = _candidates_by_title(seed_works)
    assert "hanasaka jiisan" not in candidates
    assert any("Hanasaka" in message for message in caplog.messages)


def test_find_split_work_candidates_seasons_far_apart_not_grouped(seed_works) -> None:
    assert "gelpiyo" not in _candidates_by_title(seed_works)


def test_find_split_work_candidates_entries_joined_through_shared_match_form_one_group() -> (
    None
):
    works = KnownWorkEntriesFromSeed(
        [
            {
                "title": "Show",
                "type": "TV",
                "animeSeason": {"season": "WINTER", "year": 2020},
                "sources": ["https://anilist.co/anime/1"],
            },
            {
                "title": "Show",
                "type": "TV",
                "animeSeason": {"season": "UNDEFINED"},
                "sources": ["https://kitsu.app/anime/2"],
            },
            {
                "title": "Show",
                "type": "TV",
                "animeSeason": {"season": "WINTER", "year": 2021},
                "sources": ["https://myanimelist.net/anime/3"],
            },
        ]
    )
    [candidates] = find_split_work_candidates(works)
    assert [entry.reference_page for entry in candidates.entries] == [
        "anilist:anime:1",
        "kitsu:anime:2",
        "mal:anime:3",
    ]


def test_find_split_work_candidates_single_entry_titles_give_nothing() -> None:
    works = KnownWorkEntriesFromSeed(
        [{"title": "Alone", "type": "TV", "sources": ["https://anilist.co/anime/1"]}]
    )
    assert find_split_work_candidates(works) == []

import logging

import pytest
from seed_cleaning.split_work_candidates import find_split_work_candidates
from seed_cleaning.split_work_links import SplitWorkLink, SplitWorkLinks
from work_matching import KnownWorkEntriesFromSeed

NO_SPLIT_WORK_LINKS = SplitWorkLinks({}, {})


def _candidates_by_title(seed_works) -> dict[str, tuple[list[str], str | None]]:
    return {
        candidates.entries[0].title.casefold(): (
            [entry.title for entry in candidates.entries],
            candidates.shared_provider,
        )
        for candidates in find_split_work_candidates(seed_works, NO_SPLIT_WORK_LINKS)
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


def test_find_split_work_candidates_entries_sharing_mal_dropped_and_logged(
    caplog,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            {
                "title": "Show",
                "type": "TV",
                "sources": [f"https://myanimelist.net/anime/{number}"],
            }
            for number in (1, 2)
        ]
    )
    with caplog.at_level(logging.WARNING):
        assert find_split_work_candidates(works, NO_SPLIT_WORK_LINKS) == []
    assert any("mal" in message for message in caplog.messages)


def test_find_split_work_candidates_entries_listing_same_page_marked_with_provider() -> (
    None
):
    shared_kitsu = "https://kitsu.app/anime/7"
    works = KnownWorkEntriesFromSeed(
        [
            {"title": "Twin", "type": "TV", "sources": [address, shared_kitsu]}
            for address in (
                "https://myanimelist.net/anime/5",
                "https://anilist.co/anime/5",
            )
        ]
    )
    [candidates] = find_split_work_candidates(works, NO_SPLIT_WORK_LINKS)
    assert candidates.shared_provider == "kitsu"


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
    [candidates] = find_split_work_candidates(works, NO_SPLIT_WORK_LINKS)
    assert [entry.reference_page for entry in candidates.entries] == [
        "anilist:anime:1",
        "kitsu:anime:2",
        "mal:anime:3",
    ]


def test_find_split_work_candidates_single_entry_titles_give_nothing() -> None:
    works = KnownWorkEntriesFromSeed(
        [{"title": "Alone", "type": "TV", "sources": ["https://anilist.co/anime/1"]}]
    )
    assert find_split_work_candidates(works, NO_SPLIT_WORK_LINKS) == []


def _entries(*titles_and_addresses: tuple[str, str]) -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            {"title": title, "type": "TV", "sources": [address]}
            for title, address in titles_and_addresses
        ]
    )


def test_find_split_work_candidates_linked_entries_with_different_titles_grouped() -> (
    None
):
    works = _entries(
        ("Aware! Meisaku-kun", "https://myanimelist.net/anime/32956"),
        ("Aware! Meisaku-kun 2", "https://anidb.net/anime/14760"),
    )
    links = SplitWorkLinks(
        {(0, 1): (SplitWorkLink("anidb:anime:14760", "mal:anime:32956"),)},
        {"mal:anime:32956": frozenset({1})},
    )
    [candidates] = find_split_work_candidates(works, links)
    assert (
        [entry.title for entry in candidates.entries],
        candidates.candidate_evidence,
    ) == (
        ["Aware! Meisaku-kun", "Aware! Meisaku-kun 2"],
        ("anidb:anime:14760 links mal:anime:32956",),
    )


def test_find_split_work_candidates_title_and_link_groups_sharing_entry_merge() -> None:
    works = _entries(
        ("Show", "https://myanimelist.net/anime/1"),
        ("Show", "https://kitsu.app/anime/1"),
        ("Show Part 2", "https://anidb.net/anime/2"),
    )
    links = SplitWorkLinks(
        {(0, 2): (SplitWorkLink("anidb:anime:2", "mal:anime:1"),)},
        {"mal:anime:1": frozenset({2})},
    )
    [candidates] = find_split_work_candidates(works, links)
    assert candidates.candidate_evidence == (
        "same title: 'Show' and 'Show'",
        "anidb:anime:2 links mal:anime:1",
    )


def test_find_split_work_candidates_linked_group_sharing_two_providers_kept() -> None:
    works = KnownWorkEntriesFromSeed(
        [
            {
                "title": title,
                "type": "TV",
                "sources": [
                    address,
                    "https://animenewsnetwork.com/encyclopedia/anime.php?id=5",
                    "https://simkl.com/anime/5",
                ],
            }
            for title, address in [
                ("Show", "https://myanimelist.net/anime/1"),
                ("Show Season", "https://anidb.net/anime/2"),
            ]
        ]
    )
    links = SplitWorkLinks(
        {(0, 1): (SplitWorkLink("anidb:anime:2", "mal:anime:1"),)},
        {"mal:anime:1": frozenset({1})},
    )
    assert len(find_split_work_candidates(works, links)) == 1


@pytest.mark.parametrize(
    ("linking_entries", "has_shared_tying_link"),
    [(frozenset({1}), False), (frozenset({1, 2}), True)],
)
def test_find_split_work_candidates_tying_page_linked_from_outside_group_flagged(
    linking_entries: frozenset[int], has_shared_tying_link: bool
) -> None:
    works = _entries(
        ("Show", "https://myanimelist.net/anime/1"),
        ("Show Part", "https://anidb.net/anime/2"),
        ("Unrelated", "https://kitsu.app/anime/3"),
    )
    links = SplitWorkLinks(
        {(0, 1): (SplitWorkLink("anidb:anime:2", "mal:anime:1"),)},
        {"mal:anime:1": linking_entries},
    )
    [candidates] = find_split_work_candidates(works, links)
    assert candidates.has_shared_tying_link is has_shared_tying_link


def _ajin_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            {
                "title": title,
                "type": "MOVIE",
                "animeSeason": {"season": season, "year": year},
                "sources": sources,
            }
            for title, season, year, sources in [
                (
                    "Ajin",
                    "FALL",
                    2015,
                    [
                        "https://anidb.net/anime/11265",
                        "https://anisearch.com/anime/10462",
                    ],
                ),
                (
                    "Ajin Part 1: Shoudou",
                    "FALL",
                    2015,
                    [
                        "https://myanimelist.net/anime/30868",
                        "https://kitsu.app/anime/10949",
                    ],
                ),
                (
                    "Ajin Part 2: Shoutotsu",
                    "SPRING",
                    2016,
                    ["https://myanimelist.net/anime/30869"],
                ),
                (
                    "Ajin Part 3: Shougeki",
                    "SUMMER",
                    2016,
                    ["https://myanimelist.net/anime/30870"],
                ),
            ]
        ]
    )


def test_find_split_work_candidates_entry_holding_several_works_page_grouped_with_linking_mal_entries() -> (
    None
):
    links = SplitWorkLinks(
        {},
        {"anidb:anime:11265": frozenset({1, 2, 3})},
        {"anidb:anime:11265": frozenset({1, 2, 3})},
    )
    [candidates] = find_split_work_candidates(_ajin_entries(), links)
    assert (
        [entry.title for entry in candidates.entries],
        candidates.candidate_evidence,
    ) == (
        [
            "Ajin",
            "Ajin Part 1: Shoudou",
            "Ajin Part 2: Shoutotsu",
            "Ajin Part 3: Shougeki",
        ],
        (
            "anidb:anime:11265 is linked from 3 entries holding their own MAL page, "
            "so it covers several works",
        ),
    )


def test_find_split_work_candidates_several_works_page_held_by_mal_entry_gives_nothing() -> (
    None
):
    links = SplitWorkLinks(
        {},
        {"kitsu:anime:10949": frozenset({2, 3})},
        {"kitsu:anime:10949": frozenset({2, 3})},
    )
    assert find_split_work_candidates(_ajin_entries(), links) == []

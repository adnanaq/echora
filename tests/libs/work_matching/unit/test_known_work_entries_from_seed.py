from work_matching.known_work_entries import KnownWorkEntries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed


def test_known_work_entries_from_seed_find_by_page_address_decorated_address_gives_work_entry(
    seed_entries,
) -> None:
    works = KnownWorkEntriesFromSeed(seed_entries)
    work = works.find_by_page_address("https://www.myanimelist.net/anime/21/One_Piece")
    assert work is not None
    assert work.title == "One Piece"


def test_known_work_entries_from_seed_find_by_page_address_old_anidb_address_gives_work_entry(
    seed_entries,
) -> None:
    works = KnownWorkEntriesFromSeed(seed_entries)
    work = works.find_by_page_address(
        "https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69"
    )
    assert work is not None
    assert work.title == "One Piece"


def test_known_work_entries_from_seed_find_by_page_address_unknown_page_gives_none(
    seed_entries,
) -> None:
    assert (
        KnownWorkEntriesFromSeed(seed_entries).find_by_page_address(
            "https://anilist.co/anime/1"
        )
        is None
    )


def test_known_work_entries_from_seed_find_by_page_key_gives_work_entry(
    seed_entries,
) -> None:
    work = KnownWorkEntriesFromSeed(seed_entries).find_by_page_key("mal:anime:21")
    assert work is not None
    assert work.title == "One Piece"


def test_known_work_entries_from_seed_find_by_title_cosmetic_difference_finds_every_entry(
    seed_entries,
) -> None:
    works = KnownWorkEntriesFromSeed(seed_entries)
    assert {work.title for work in works.find_by_title("ranma ½", "TV", 1989)} == {
        "Ranma 1/2",
        "Ranma ½",
    }
    assert {work.title for work in works.find_by_title("Cue", "TV", 2022)} == {
        "CUE!",
        "Cue!",
    }


def test_known_work_entries_from_seed_find_by_title_other_type_or_year_finds_nothing(
    seed_entries,
) -> None:
    works = KnownWorkEntriesFromSeed(seed_entries)
    assert works.find_by_title("One Piece", "MOVIE", 1999) == []
    assert works.find_by_title("One Piece", "TV", 2000) == []


def test_known_work_entries_from_seed_find_by_title_missing_year_on_either_side_matches(
    seed_entries,
) -> None:
    works = KnownWorkEntriesFromSeed(seed_entries)
    assert [w.title for w in works.find_by_title("One Piece", "TV", None)] == [
        "One Piece"
    ]
    assert [
        w.title for w in works.find_by_title("2.5-jigen no Ririsa Season 2", "TV", 2026)
    ] == ["2.5-jigen no Ririsa 2nd Season"]


def test_known_work_entries_from_seed_page_in_two_entries_reported_and_first_kept() -> (
    None
):
    shared = "https://myanimelist.net/anime/1"
    works = KnownWorkEntriesFromSeed(
        [
            {"title": "First", "type": "TV", "sources": [shared]},
            {
                "title": "Second",
                "type": "TV",
                "sources": [shared, "https://anilist.co/anime/2"],
            },
        ]
    )
    assert works.page_keys_in_several_entries == frozenset({"mal:anime:1"})
    work = works.find_by_page_address(shared)
    assert work is not None
    assert work.title == "First"


def test_known_work_entries_from_seed_entry_without_sources_skipped_and_reported() -> (
    None
):
    works = KnownWorkEntriesFromSeed(
        [
            {"title": "No Pages", "type": "TV", "sources": []},
            {
                "title": "One Piece",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/21"],
            },
        ]
    )
    assert (len(works), works.titles_of_entries_without_pages) == (1, ("No Pages",))


def test_known_work_entries_from_seed_from_file_counts_every_entry(
    seed_file, seed_entries
) -> None:
    works = KnownWorkEntriesFromSeed.from_file(seed_file)
    assert len(works) == len(seed_entries)
    assert works.page_keys_in_several_entries == frozenset()


def test_known_work_entries_from_seed_satisfies_known_work_entries_lookup(
    seed_entries,
) -> None:
    assert isinstance(KnownWorkEntriesFromSeed(seed_entries), KnownWorkEntries)


def test_known_work_entries_from_seed_iteration_gives_work_entries_with_pages_in_seed_order() -> (
    None
):
    works = KnownWorkEntriesFromSeed(
        [
            {"title": "First", "type": "TV", "sources": ["https://anilist.co/anime/1"]},
            {"title": "No Pages", "type": "TV", "sources": []},
            {"title": "Second", "type": "TV", "sources": ["https://kitsu.app/anime/2"]},
        ]
    )
    assert [work.title for work in works] == ["First", "Second"]

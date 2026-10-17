from work_matching.known_works import KnownWorks
from work_matching.known_works_from_seed import KnownWorksFromSeed


def test_known_works_from_seed_find_by_page_decorated_address_gives_work(
    seed_entries,
) -> None:
    works = KnownWorksFromSeed(seed_entries)
    work = works.find_by_page("https://www.myanimelist.net/anime/21/One_Piece")
    assert work is not None
    assert work.title == "One Piece"


def test_known_works_from_seed_find_by_page_old_anidb_address_gives_work(
    seed_entries,
) -> None:
    works = KnownWorksFromSeed(seed_entries)
    work = works.find_by_page("https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69")
    assert work is not None
    assert work.title == "One Piece"


def test_known_works_from_seed_find_by_page_unknown_page_gives_none(
    seed_entries,
) -> None:
    assert (
        KnownWorksFromSeed(seed_entries).find_by_page("https://anilist.co/anime/1")
        is None
    )


def test_known_works_from_seed_find_by_title_cosmetic_difference_finds_every_entry(
    seed_entries,
) -> None:
    works = KnownWorksFromSeed(seed_entries)
    assert {work.title for work in works.find_by_title("ranma ½", "TV", 1989)} == {
        "Ranma 1/2",
        "Ranma ½",
    }
    assert {work.title for work in works.find_by_title("Cue", "TV", 2022)} == {
        "CUE!",
        "Cue!",
    }


def test_known_works_from_seed_find_by_title_other_type_or_year_finds_nothing(
    seed_entries,
) -> None:
    works = KnownWorksFromSeed(seed_entries)
    assert works.find_by_title("One Piece", "MOVIE", 1999) == []
    assert works.find_by_title("One Piece", "TV", 2000) == []


def test_known_works_from_seed_find_by_title_missing_year_on_either_side_matches(
    seed_entries,
) -> None:
    works = KnownWorksFromSeed(seed_entries)
    assert [w.title for w in works.find_by_title("One Piece", "TV", None)] == [
        "One Piece"
    ]
    assert [
        w.title for w in works.find_by_title("2.5-jigen no Ririsa Season 2", "TV", 2026)
    ] == ["2.5-jigen no Ririsa 2nd Season"]


def test_known_works_from_seed_page_in_two_entries_reported_and_first_kept() -> None:
    shared = "https://myanimelist.net/anime/1"
    works = KnownWorksFromSeed(
        [
            {"title": "First", "type": "TV", "sources": [shared]},
            {
                "title": "Second",
                "type": "TV",
                "sources": [shared, "https://anilist.co/anime/2"],
            },
        ]
    )
    assert works.duplicate_pages == frozenset({"mal:anime:1"})
    work = works.find_by_page(shared)
    assert work is not None
    assert work.title == "First"


def test_known_works_from_seed_from_file_counts_every_entry(
    seed_file, seed_entries
) -> None:
    works = KnownWorksFromSeed.from_file(seed_file)
    assert len(works) == len(seed_entries)
    assert works.duplicate_pages == frozenset()


def test_known_works_from_seed_satisfies_known_works_lookup(seed_entries) -> None:
    assert isinstance(KnownWorksFromSeed(seed_entries), KnownWorks)

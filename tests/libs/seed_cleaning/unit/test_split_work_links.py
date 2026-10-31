from seed_cleaning.split_work_links import SplitWorkLink, find_split_work_links
from work_matching import KnownWorkEntriesFromSeed, ProviderPage, ProviderPageStore

MAIN_MAL = "https://myanimelist.net/anime/32956"
OTHER_MAL = "https://myanimelist.net/anime/40000"
SEASON_ANIDB = "https://anidb.net/anime/14760"
SEASON_ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=24132"
OTHER_ANIDB = "https://anidb.net/anime/16366"
KITSU = "https://kitsu.app/anime/12168"
SEASON_KITSU = "https://kitsu.app/anime/14241"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {"title": title, "type": "TV", "sources": list(addresses)}


def _page(address: str, *linked_addresses: str, **details) -> ProviderPage:
    return ProviderPage.from_canonical_record(
        address,
        {
            "title": "Show",
            "external_sources": [{"source": linked} for linked in linked_addresses],
        }
        | details,
    )


def _store(tmp_path, *pages: ProviderPage) -> ProviderPageStore:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    store.add_pages(*pages)
    return store


def test_find_split_work_links_page_linking_other_entry_mal_page_ties_entries(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Main", MAIN_MAL, KITSU), _seed_entry("Season 2", SEASON_ANIDB)]
    )
    split_work_links = find_split_work_links(
        works, _store(tmp_path, _page(SEASON_ANIDB, MAIN_MAL))
    )
    assert split_work_links.links_by_entry_pair == {
        (0, 1): (SplitWorkLink("anidb:anime:14760", "mal:anime:32956"),)
    }


def test_find_split_work_links_link_between_two_mal_entries_ignored(tmp_path) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Main", MAIN_MAL), _seed_entry("Other", OTHER_MAL, SEASON_ANIDB)]
    )
    split_work_links = find_split_work_links(
        works, _store(tmp_path, _page(SEASON_ANIDB, MAIN_MAL))
    )
    assert split_work_links.links_by_entry_pair == {}


def test_find_split_work_links_page_linking_two_pages_of_one_provider_ties_nothing(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Main", MAIN_MAL),
            _seed_entry("Other", OTHER_MAL),
            _seed_entry("Anthology", SEASON_ANIDB),
        ]
    )
    split_work_links = find_split_work_links(
        works, _store(tmp_path, _page(SEASON_ANIDB, MAIN_MAL, OTHER_MAL))
    )
    assert split_work_links.links_by_entry_pair == {}


def test_find_split_work_links_link_through_non_mal_page_ties_entries(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Winter Star", SEASON_ANN),
            _seed_entry("Fuyu no Star", SEASON_ANIDB),
        ]
    )
    split_work_links = find_split_work_links(
        works, _store(tmp_path, _page(SEASON_ANIDB, SEASON_ANN))
    )
    assert list(split_work_links.links_by_entry_pair) == [(0, 1)]


def test_find_split_work_links_dead_page_ties_nothing(tmp_path) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Main", MAIN_MAL), _seed_entry("Season 2", SEASON_ANIDB)]
    )
    store = _store(tmp_path, ProviderPage.from_dead_address(SEASON_ANIDB))
    assert find_split_work_links(works, store).links_by_entry_pair == {}


def test_find_split_work_links_records_every_entry_linking_page(tmp_path) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Main", MAIN_MAL),
            _seed_entry("Season 2", SEASON_ANIDB),
            _seed_entry("Season 6", OTHER_ANIDB),
        ]
    )
    store = _store(
        tmp_path, _page(SEASON_ANIDB, MAIN_MAL), _page(OTHER_ANIDB, MAIN_MAL)
    )
    split_work_links = find_split_work_links(works, store)
    assert sorted(split_work_links.links_by_entry_pair) == [(0, 1), (0, 2)]
    assert split_work_links.linking_entries_by_page_key == {
        "mal:anime:32956": frozenset({1, 2})
    }


def test_find_split_work_links_entry_with_own_mal_page_recorded_as_linking(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Main", MAIN_MAL),
            _seed_entry("Season 2", SEASON_ANIDB),
            _seed_entry("Sequel", OTHER_MAL, OTHER_ANIDB),
        ]
    )
    store = _store(
        tmp_path, _page(SEASON_ANIDB, MAIN_MAL), _page(OTHER_ANIDB, MAIN_MAL)
    )
    split_work_links = find_split_work_links(works, store)
    assert list(split_work_links.links_by_entry_pair) == [(0, 1)]
    assert split_work_links.linking_entries_by_page_key == {
        "mal:anime:32956": frozenset({1, 2})
    }


def test_find_split_work_links_page_linked_from_two_mal_entries_ties_nothing(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Series", SEASON_ANIDB),
            _seed_entry("Series", MAIN_MAL, KITSU),
            _seed_entry("Special", OTHER_MAL),
        ]
    )
    store = _store(
        tmp_path, _page(MAIN_MAL, SEASON_ANIDB), _page(OTHER_MAL, SEASON_ANIDB)
    )
    split_work_links = find_split_work_links(works, store)
    assert split_work_links.links_by_entry_pair == {}
    assert split_work_links.linking_entries_by_page_key == {
        "anidb:anime:14760": frozenset({1, 2})
    }
    assert split_work_links.mal_entries_by_several_works_page_key == {
        "anidb:anime:14760": frozenset({1, 2})
    }


def test_find_split_work_links_page_linked_from_one_mal_entry_ties_entries(
    tmp_path,
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Series", SEASON_ANIDB),
            _seed_entry("Series", MAIN_MAL, KITSU),
            _seed_entry("Season 2", SEASON_KITSU),
        ]
    )
    store = _store(
        tmp_path, _page(MAIN_MAL, SEASON_ANIDB), _page(SEASON_KITSU, SEASON_ANIDB)
    )
    split_work_links = find_split_work_links(works, store)
    assert sorted(split_work_links.links_by_entry_pair) == [(0, 1), (0, 2)]

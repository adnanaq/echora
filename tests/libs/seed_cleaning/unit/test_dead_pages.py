from seed_cleaning.dead_pages import list_dead_page_corrections
from seed_cleaning.seed_corrections import DEAD_PAGE_REASON, SeedCorrection
from work_matching import KnownWorkEntriesFromSeed, ProviderPage, ProviderPageStore

MAL = "https://myanimelist.net/anime/1"
KITSU = "https://kitsu.app/anime/1"
UNFETCHED_ANIDB = "https://anidb.net/anime/1"


def test_list_dead_page_corrections_drops_only_dead_pages(tmp_path) -> None:
    works = KnownWorkEntriesFromSeed(
        [{"title": "Show", "type": "TV", "sources": [MAL, KITSU, UNFETCHED_ANIDB]}]
    )
    store = ProviderPageStore(tmp_path / "provider_pages.jsonl")
    store.add_pages(
        ProviderPage.build_dead_page(KITSU),
        ProviderPage.read_canonical_record(MAL, {"title": "Show"}),
    )
    assert list_dead_page_corrections(works, store) == [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:1",
            address=KITSU,
            placement_word=None,
            is_dead_page=True,
            reason=DEAD_PAGE_REASON,
        )
    ]

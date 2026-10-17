from work_matching.provider_page import ProviderPage
from work_matching.provider_page_store import ProviderPageStore

MAL = "https://myanimelist.net/anime/21"
KITSU = "https://kitsu.app/anime/12"


def _page(address: str, title: str) -> ProviderPage:
    return ProviderPage.from_canonical_record(address, {"title": title, "year": 1999})


def test_provider_page_store_missing_file_starts_empty(tmp_path) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    assert len(store) == 0
    assert store.get("mal:anime:21") is None


def test_provider_page_store_added_pages_read_back_after_reopening(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    pages = [_page(MAL, "One Piece"), _page(KITSU, "One Piece")]
    ProviderPageStore(path).add(*pages)
    reopened = ProviderPageStore(path)
    assert [reopened.get(page.key) for page in pages] == pages


def test_provider_page_store_same_page_added_again_keeps_latest(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    store = ProviderPageStore(path)
    latest = _page(MAL, "One Piece")
    store.add(_page(MAL, "Old Title"))
    store.add(latest)
    assert store.get("mal:anime:21") == latest
    assert ProviderPageStore(path).get("mal:anime:21") == latest
    assert len(store) == 1


def test_provider_page_store_contains_page_key_after_add(tmp_path) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    store.add(_page(MAL, "One Piece"))
    assert "mal:anime:21" in store
    assert "kitsu:anime:12" not in store


def test_provider_page_store_creates_missing_folder(tmp_path) -> None:
    path = tmp_path / "seed_cleaning" / "pages.jsonl"
    ProviderPageStore(path).add(_page(MAL, "One Piece"))
    assert path.exists()

import pytest
from pydantic import ValidationError
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_store import ProviderPageStore
from work_matching.work_entry import WorkEntry

MAL = "https://myanimelist.net/anime/21"
KITSU = "https://kitsu.app/anime/12"


def _page(address: str, title: str) -> ProviderPage:
    return ProviderPage.from_canonical_record(address, {"title": title, "year": 1999})


def test_provider_page_store_missing_file_starts_empty(tmp_path) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    assert len(store) == 0
    assert store.get_page("mal:anime:21") is None


def test_provider_page_store_add_pages_read_back_after_reopening(
    tmp_path,
) -> None:
    path = tmp_path / "pages.jsonl"
    pages = [_page(MAL, "One Piece"), _page(KITSU, "One Piece")]
    ProviderPageStore(path).add_pages(*pages)
    reopened = ProviderPageStore(path)
    assert [reopened.get_page(page.key) for page in pages] == pages


def test_provider_page_store_same_page_added_again_keeps_latest(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    store = ProviderPageStore(path)
    latest = _page(MAL, "One Piece")
    store.add_pages(_page(MAL, "Old Title"))
    store.add_pages(latest)
    assert store.get_page("mal:anime:21") == latest
    assert ProviderPageStore(path).get_page("mal:anime:21") == latest
    assert len(store) == 1


def test_provider_page_store_contains_page_key_after_add(tmp_path) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    store.add_pages(_page(MAL, "One Piece"))
    assert "mal:anime:21" in store
    assert "kitsu:anime:12" not in store


def test_provider_page_store_creates_missing_folder(tmp_path) -> None:
    path = tmp_path / "seed_cleaning" / "pages.jsonl"
    ProviderPageStore(path).add_pages(_page(MAL, "One Piece"))
    assert path.exists()


def test_provider_page_store_partial_last_line_left_until_next_add(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    page = _page(MAL, "One Piece")
    ProviderPageStore(path).add_pages(page)
    with path.open("a", encoding="utf-8") as output:
        output.write('{"key":"kitsu:an')
    written = path.read_bytes()
    store = ProviderPageStore(path)
    assert (path.read_bytes(), len(store)) == (written, 1)
    store.add_pages(_page(KITSU, "One Piece"))
    reopened = ProviderPageStore(path)
    assert (len(reopened), reopened.get_page(page.key)) == (2, page)


def test_provider_page_store_broken_complete_line_raises_naming_file_and_line(
    tmp_path,
) -> None:
    path = tmp_path / "pages.jsonl"
    path.write_text('{"key":"kitsu:an\n', encoding="utf-8")
    with pytest.raises(ValidationError) as raised:
        ProviderPageStore(path)
    assert raised.value.__notes__ == [f"{path}, line 1"]


def test_provider_page_store_add_pages_without_pages_creates_no_file(tmp_path) -> None:
    path = tmp_path / "pages.jsonl"
    ProviderPageStore(path).add_pages()
    assert not path.exists()


def test_provider_page_store_has_page_searched_by_after_add(tmp_path) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    store.add_pages(
        ProviderPage.from_canonical_record(
            "https://animeschedule.net/anime/one-piece",
            {"title": "One Piece"},
            searched_by_address=MAL,
        )
    )
    assert store.has_page_searched_by("mal:anime:21")
    assert not store.has_page_searched_by("kitsu:anime:12")


def test_provider_page_store_get_pages_searched_by_gives_pages_after_reopening(
    tmp_path,
) -> None:
    path = tmp_path / "pages.jsonl"
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/one-piece",
        {"title": "One Piece"},
        searched_by_address=MAL,
    )
    ProviderPageStore(path).add_pages(found, _page(KITSU, "One Piece"))
    reopened = ProviderPageStore(path)
    assert reopened.get_pages_searched_by("mal:anime:21") == (found,)
    assert reopened.get_pages_searched_by("kitsu:anime:12") == ()


def test_provider_page_store_searched_page_added_through_second_search_keeps_both(
    tmp_path,
) -> None:
    path = tmp_path / "pages.jsonl"
    record = {"title": "Ranma", "sources": [MAL, "https://anilist.co/anime/210"]}
    store = ProviderPageStore(path)
    for searched_by_address in (MAL, "https://anilist.co/anime/210"):
        store.add_pages(
            ProviderPage.from_canonical_record(
                "https://animeschedule.net/anime/ranma",
                record,
                searched_by_address=searched_by_address,
            )
        )
    reopened = ProviderPageStore(path).get_page("https://animeschedule.net/anime/ranma")
    assert reopened is not None
    assert reopened.searched_by == ("anilist:anime:210", "mal:anime:21")


def test_get_entry_pages_gives_stored_pages_then_pages_found_through_them(
    tmp_path,
) -> None:
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    searched_page = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/one-piece",
        {"title": "One Piece", "sources": [MAL]},
        searched_by_address=MAL,
    )
    mal_page = _page(MAL, "One Piece")
    store.add_pages(searched_page, mal_page, _page(KITSU, "One Piece"))
    entry = WorkEntry.from_seed_entry(
        {
            "title": "One Piece",
            "type": "TV",
            "sources": [MAL, "https://anidb.net/anime/69"],
        }
    )
    assert store.get_entry_pages(entry) == [mal_page, searched_page]


def test_provider_page_store_searched_page_added_twice_in_one_call_keeps_both_searched_by_keys(
    tmp_path,
) -> None:
    path = tmp_path / "pages.jsonl"
    record = {"title": "Ranma", "sources": [MAL, "https://anilist.co/anime/210"]}
    ProviderPageStore(path).add_pages(
        *(
            ProviderPage.from_canonical_record(
                "https://animeschedule.net/anime/ranma",
                record,
                searched_by_address=searched_by_address,
            )
            for searched_by_address in (MAL, "https://anilist.co/anime/210")
        )
    )
    reopened = ProviderPageStore(path).get_page("https://animeschedule.net/anime/ranma")
    assert reopened is not None
    assert reopened.searched_by == ("anilist:anime:210", "mal:anime:21")

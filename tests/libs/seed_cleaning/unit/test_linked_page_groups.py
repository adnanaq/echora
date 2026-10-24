import pytest
from seed_cleaning.linked_page_groups import (
    LinkingSeedEntry,
    group_linked_pages_by_current_key,
    rename_linked_pages_to_current_keys,
)
from work_matching import (
    LinkingEntryDetails,
    ProviderPage,
    ProviderPageStore,
    WorkEntry,
)

BOXER_ANILIST = "https://anilist.co/anime/163794"
YAMATO_ANILIST = "https://anilist.co/anime/182615"
KITSU_SLUG = "https://kitsu.app/anime/the-boxer"
KITSU_NUMBER = "https://kitsu.app/anime/51119"


def _linking_seed_entry(address: str, *linked_page_keys: str) -> LinkingSeedEntry:
    entry = WorkEntry.from_seed_entry(
        {"title": "Show", "type": "TV", "sources": [address]}
    )
    return LinkingSeedEntry(
        entry.reference_page,
        LinkingEntryDetails(entry, None, frozenset(linked_page_keys)),
    )


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    store = ProviderPageStore(tmp_path / "provider_pages.jsonl")
    store.add_pages(
        ProviderPage.from_canonical_record(
            KITSU_SLUG, {"title": "Show", "sources": [KITSU_SLUG, KITSU_NUMBER]}
        ),
        ProviderPage.from_canonical_record(KITSU_NUMBER, {"title": "Show"}),
    )
    return store


def test_group_linked_pages_by_current_key_two_spellings_form_one_group(store) -> None:
    boxer = _linking_seed_entry(BOXER_ANILIST, "kitsu:anime:the-boxer")
    yamato = _linking_seed_entry(YAMATO_ANILIST, "kitsu:anime:51119")
    groups, waiting_page_keys = group_linked_pages_by_current_key(
        {"kitsu:anime:the-boxer": [boxer], "kitsu:anime:51119": [yamato]}, store
    )
    [(page, linking_seed_entries)] = groups.values()
    assert (list(groups), page.key, waiting_page_keys) == (
        ["kitsu:anime:51119"],
        "kitsu:anime:51119",
        (),
    )
    assert [
        linking_seed_entry.seed_entry_reference_page
        for linking_seed_entry in linking_seed_entries
    ] == ["anilist:anime:163794", "anilist:anime:182615"]


def test_group_linked_pages_by_current_key_page_not_stored_waits(store) -> None:
    boxer = _linking_seed_entry(BOXER_ANILIST, "mal:anime:65144")
    groups, waiting_page_keys = group_linked_pages_by_current_key(
        {"mal:anime:65144": [boxer]}, store
    )
    assert (groups, waiting_page_keys) == ({}, ("mal:anime:65144",))


def test_rename_linked_pages_to_current_keys_unfetched_pages_keep_spelling() -> None:
    boxer = _linking_seed_entry(
        BOXER_ANILIST, "kitsu:anime:the-boxer", "mal:anime:65144"
    )
    renamed = rename_linked_pages_to_current_keys(
        boxer, {"kitsu:anime:the-boxer": "kitsu:anime:51119"}
    )
    assert renamed.entry_details.linked_page_keys == frozenset(
        {"kitsu:anime:51119", "mal:anime:65144"}
    )
    assert renamed.seed_entry_reference_page == boxer.seed_entry_reference_page

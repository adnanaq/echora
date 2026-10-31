from typing import Any
from unittest.mock import create_autospec

import pytest
from enrichment.pipeline.api_fetcher import ApiFetcher
from enrichment.pipeline.fetch_summary import FetchOutcome
from enrichment.pipeline.provider_page_fetcher import ProviderPageFetcher
from seed_cleaning.seed_fetch import fetch_seed_and_linked_pages, read_seed_release_name
from work_matching import KnownWorkEntriesFromSeed, ProviderPageStore

ONE_PIECE_MAL = "https://myanimelist.net/anime/21"
ONE_PIECE_KITSU = "https://kitsu.app/anime/12"
ONE_PIECE_FILM_ANIDB = "https://anidb.net/anime/1530"
ONE_PIECE_FILM_ANIDB_KEY = "anidb:anime:1530"


def _seed_entry(title: str, *addresses: str) -> dict[str, Any]:
    return {"title": title, "type": "TV", "sources": list(addresses)}


def _api_fetcher(
    records_by_service_and_address: dict[tuple[str, str], dict[str, Any]],
) -> ApiFetcher:
    api_fetcher = create_autospec(ApiFetcher, instance=True)
    api_fetcher.api_errors = {}

    async def fetch_all_data(
        ids: dict[str, str], offline_data: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        [service] = kwargs["only_services"]
        record = records_by_service_and_address.get((service, next(iter(ids.values()))))
        return {service: None if record is None else {"anime": record}}

    api_fetcher.fetch_all_data.side_effect = fetch_all_data
    return api_fetcher


async def test_fetch_seed_and_linked_pages_linked_page_of_entry_holding_mal_page_stored(
    tmp_path,
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [_seed_entry("One Piece", ONE_PIECE_MAL, ONE_PIECE_KITSU)]
    )
    api_fetcher = _api_fetcher(
        {
            ("mal", ONE_PIECE_MAL): {
                "title": "One Piece",
                "sources": [ONE_PIECE_FILM_ANIDB],
            },
            ("kitsu", ONE_PIECE_KITSU): {"title": "One Piece"},
            ("anidb", ONE_PIECE_FILM_ANIDB): {"title": "One Piece Film"},
        }
    )
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    seed_fetch_summary = await fetch_seed_and_linked_pages(
        known_work_entries,
        store,
        ProviderPageFetcher(lambda: api_fetcher, store),
    )
    linked_page = store.get_page(ONE_PIECE_FILM_ANIDB_KEY)
    assert linked_page is not None
    assert not linked_page.is_dead
    assert seed_fetch_summary.linked_page_summary.outcomes == {FetchOutcome.FETCHED: 1}


async def test_fetch_seed_and_linked_pages_page_seed_holds_not_fetched_as_linked_page(
    tmp_path,
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [
            _seed_entry("One Piece", ONE_PIECE_MAL),
            _seed_entry("One Piece Film", ONE_PIECE_FILM_ANIDB),
        ]
    )
    api_fetcher = _api_fetcher(
        {
            ("mal", ONE_PIECE_MAL): {
                "title": "One Piece",
                "sources": [ONE_PIECE_FILM_ANIDB],
            }
        }
    )
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    seed_fetch_summary = await fetch_seed_and_linked_pages(
        known_work_entries,
        store,
        ProviderPageFetcher(lambda: api_fetcher, store),
    )
    assert seed_fetch_summary.linked_page_summary.outcomes == {}


@pytest.mark.parametrize(
    ("seed_document", "expected"),
    [
        (
            {
                "$schema": "https://raw.githubusercontent.com/cedya77/anime-offline-database/refs/tags/2026-43/schemas/anime-offline-database-minified.schema.json",
                "lastUpdate": "2026-10-25",
            },
            "2026-43",
        ),
        ({"lastUpdate": "2026-10-25"}, "2026-10-25"),
    ],
)
def test_read_seed_release_name_tag_or_update_date_returned(
    seed_document: dict[str, str], expected: str
) -> None:
    assert read_seed_release_name(seed_document) == expected

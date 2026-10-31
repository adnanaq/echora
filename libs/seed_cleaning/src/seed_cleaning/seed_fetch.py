"""Fetch every page of a seed release, then every page they link that no seed entry holds.

Seed cleaning reads only the page store, so the store must hold every page the
rules can read before either pass runs.

Each item gives the rule, then the reason for it.

- Every seed entry's pages are fetched, not only the entries checked. Split
  works are found by links, and a link is only seen when the store holds the
  page carrying it.
- Then every page those pages link that no seed entry holds is fetched, for
  every entry. Deciding whether such a page joins an entry needs its details,
  and later steps, such as discovery, read the same pages.
- Linked pages are listed after the seed pages are stored, because their links
  are read from the stored pages.
- A linked page is fetched once, with the title of the first entry in seed
  order linking it; some helpers search by title.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from enrichment.pipeline.fetch_summary import FetchSummary
from enrichment.pipeline.page_requests import FETCHED_PROVIDERS
from enrichment.pipeline.provider_page_fetcher import ProviderPageFetcher
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    find_unheld_linked_page_keys,
)

# The release tag in the seed's schema address, such as ``2026-43`` in
# ``.../refs/tags/2026-43/schemas/...``.
_RELEASE_TAG_IN_SCHEMA_ADDRESS = re.compile(r"/refs/tags/([^/]+)/")


@dataclass(frozen=True)
class SeedFetchSummary:
    """How fetching a seed release's pages and their linked pages went."""

    seed_page_summary: FetchSummary
    linked_page_summary: FetchSummary


async def fetch_seed_and_linked_pages(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    page_fetcher: ProviderPageFetcher,
) -> SeedFetchSummary:
    """Fetch every seed entry's pages, then every unheld page they link.

    Args:
        known_work_entries: Every entry of the seed release.
        store: The provider page store the fetcher adds pages to.
        page_fetcher: The fetcher adding pages to that store.

    Returns:
        The summary of each of the two fetches.
    """
    seed_page_summary = await page_fetcher.fetch_entry_pages(known_work_entries)
    linked_page_summary = await page_fetcher.fetch_linked_pages(
        list_unheld_linked_page_titles(known_work_entries, store)
    )
    return SeedFetchSummary(seed_page_summary, linked_page_summary)


def list_unheld_linked_page_titles(
    known_work_entries: KnownWorkEntriesFromSeed, store: ProviderPageStore
) -> dict[str, str]:
    """List the pages stored seed pages link that no seed entry holds.

    Args:
        known_work_entries: Every entry of the seed release.
        store: The provider page store holding the seed entries' pages.

    Returns:
        Each linked page's key, sorted, mapped to the title of the first entry
        in seed order that links it.
    """
    linking_entry_title_by_page_key: dict[str, str] = {}
    for entry in known_work_entries:
        for linked_page_key in find_unheld_linked_page_keys(
            entry, store.get_entry_pages(entry), known_work_entries, FETCHED_PROVIDERS
        ):
            linking_entry_title_by_page_key.setdefault(linked_page_key, entry.title)
    return dict(sorted(linking_entry_title_by_page_key.items()))


def read_seed_release_name(seed_document: Mapping[str, Any]) -> str:
    """Read which release a downloaded seed file is, for naming its run log.

    Args:
        seed_document: The seed file's top-level object.

    Returns:
        The release tag in the schema address, such as ``2026-43``, or the
        release's ``lastUpdate`` date when the address names no tag.
    """
    if schema_address_match := _RELEASE_TAG_IN_SCHEMA_ADDRESS.search(
        seed_document.get("$schema", "")
    ):
        return schema_address_match[1]
    return seed_document["lastUpdate"]

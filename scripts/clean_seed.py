"""Find the seed entries that need checking, and fetch every page of them.

The first step of seed cleaning. It reads a downloaded seed release, never
changes it, and writes to the data folder:

- ``entries_to_check.json``: the entries with a doubled provider, and the
  groups of split-work candidates;
- ``provider_pages.jsonl``: every page of those entries, fetched through the
  enrichment helpers. A rerun fetches only what is not stored yet.

Usage:
    uv run python scripts/clean_seed.py [SEED_FILE] [--data-dir DIR] [--no-fetch]
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from common.config import get_settings
from enrichment.pipeline.api_fetcher import ApiFetcher
from http_cache import close_cache, configure_cache
from seed_cleaning import (
    FetchSummary,
    ProviderPageFetcher,
    SplitWorkCandidates,
    doubled_provider_entries,
    split_work_candidates,
)
from work_matching import (
    KnownWorksFromSeed,
    ProviderPageStore,
    WorkEntry,
    doubled_provider_pages,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def parse_arguments() -> argparse.Namespace:
    """Read the seed file, the data folder and whether to fetch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "seed_file",
        nargs="?",
        default="assets/seed_data/anime-offline-database.json",
        type=Path,
        help="The downloaded seed release (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/seed_cleaning",
        type=Path,
        help="Where the entries to check and pages are written (default: %(default)s)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Only list the entries to check; fetch no pages",
    )
    return parser.parse_args()


def entries_to_check_document(
    seed_schema: str | None,
    doubled: list[WorkEntry],
    candidate_groups: list[SplitWorkCandidates],
) -> dict[str, Any]:
    """Describe the entries to check for ``entries_to_check.json``.

    Each entry is named by its reference page, which stays the same across seed
    releases while its position in the file does not. ``seed_schema`` is the
    seed file's ``$schema``, whose address names the release, or None when the
    file has none.
    """
    return {
        "seed_schema": seed_schema,
        "doubled_provider_entries": [
            {
                "reference_page": entry.reference_page,
                "title": entry.title,
                "doubled_providers": doubled_provider_pages(entry.pages),
            }
            for entry in doubled
        ],
        "split_work_candidates": [
            {
                "shared_provider": group.shared_provider,
                "entries": [
                    {"reference_page": entry.reference_page}
                    | entry.model_dump(mode="json")
                    for entry in group.entries
                ],
            }
            for group in candidate_groups
        ],
    }


async def fetch_pages(entries: list[WorkEntry], store_path: Path) -> FetchSummary:
    """Fetch every page of the entries that the store does not hold yet."""
    settings = get_settings()
    cache = configure_cache(settings.cache, settings.redis)
    try:
        fetcher = ProviderPageFetcher(
            lambda: ApiFetcher(settings.enrichment, cache),
            ProviderPageStore(store_path),
        )
        return await fetcher.fetch_entry_pages(entries)
    finally:
        await close_cache()


async def main() -> None:
    """List the entries to check, write them, and fetch their pages."""
    arguments = parse_arguments()
    document = json.loads(arguments.seed_file.read_text(encoding="utf-8"))
    known_works = KnownWorksFromSeed(document["data"])
    doubled = doubled_provider_entries(known_works)
    candidate_groups = split_work_candidates(known_works)

    arguments.data_dir.mkdir(parents=True, exist_ok=True)
    entries_to_check = entries_to_check_document(
        document.get("$schema"), doubled, candidate_groups
    )
    (arguments.data_dir / "entries_to_check.json").write_text(
        json.dumps(entries_to_check, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    marked = sum(group.shared_provider is not None for group in candidate_groups)
    print(f"Entries with a doubled provider: {len(doubled)}")
    print(f"Split-work candidate groups: {len(candidate_groups)} ({marked} marked)")
    if arguments.no_fetch:
        return

    # Every entry is passed on: the fetcher makes each request once, so an entry
    # sharing a page with another still gets its other pages fetched.
    summary = await fetch_pages(
        [*doubled, *(entry for group in candidate_groups for entry in group.entries)],
        arguments.data_dir / "provider_pages.jsonl",
    )
    for outcome, count in sorted(summary.outcomes.items()):
        print(f"Pages {outcome}: {count}")
    for page, error in summary.failed_pages.items():
        print(f"Failed, retried next run: {page}: {error[:160]}")


if __name__ == "__main__":
    asyncio.run(main())

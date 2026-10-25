"""Find the seed entries that need checking, fetch their pages, and correct them.

Seed cleaning reads a downloaded seed release, never changes it, and writes to
the data folder:

- ``entries_to_check.json``: the entries with a doubled provider, and the
  groups of split-work candidates;
- ``provider_pages.jsonl``: every page of those entries, the AniList, Kitsu
  and AniDB pages of entries without a MAL page (the pages that carry links),
  and the pages they link that no seed entry holds, fetched through the
  enrichment helpers. A rerun fetches only what is not stored yet;
- ``review.jsonl``: one case per entry group the rules cannot place (an
  entry with doubled providers, or split-work candidates, merged when they
  share an entry), and per linked page they cannot add on their own, where a
  person writes the answer; answers are kept across runs;
- ``corrections.jsonl``: where each page of an entry group ends up, and each
  linked page added, from the rules or an accepted answer. Rewritten on each
  run.

AniDB caps requests per day, so a long run sets a wide gap and stops at the
first block, and the next day's run resumes: ``ANIDB_MIN_REQUEST_INTERVAL=8``
and ``ANIDB_BAN_COOLDOWN=86400``.

Usage:
    uv run python scripts/clean_seed.py [SEED_FILE] [--data-dir DIR]
        [--no-fetch | --refetch-stored]
"""

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from common.config import get_settings
from enrichment.pipeline.api_fetcher import ApiFetcher
from http_cache import close_cache, configure_cache
from seed_cleaning import (
    LINK_CARRYING_PROVIDERS,
    FetchSummary,
    LinkedPageCorrections,
    PlacementCorrections,
    ProviderPageFetcher,
    SeedCorrection,
    SplitWorkCandidates,
    build_corrected_entries,
    correct_entry_groups,
    correct_linked_pages,
    find_doubled_provider_entries,
    find_entries_without_mal_page,
    find_entry_groups,
    find_linking_seed_entries,
    find_split_work_candidates,
    list_entries_with_placed_pages,
    write_corrections,
)
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    find_doubled_provider_page_keys,
    replace_file_lines,
)


def parse_arguments() -> argparse.Namespace:
    """Read the seed file, the data folder and how to fetch from the command line.

    Returns:
        The arguments: ``seed_file``, ``data_dir``, ``no_fetch`` and
        ``refetch_stored``.
    """
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
    fetch_choice = parser.add_mutually_exclusive_group()
    fetch_choice.add_argument(
        "--no-fetch",
        action="store_true",
        help="Fetch no pages; correct the entries from the pages already stored",
    )
    fetch_choice.add_argument(
        "--refetch-stored",
        action="store_true",
        help=(
            "Fetch stored pages again, so their details gain fields added since "
            "they were stored; the cache serves pages fetched in the last week"
        ),
    )
    return parser.parse_args()


def build_entries_to_check_document(
    seed_schema: str | None,
    doubled_provider_entries: list[WorkEntry],
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
                "doubled_providers": find_doubled_provider_page_keys(
                    entry.addresses_by_page_key
                ),
            }
            for entry in doubled_provider_entries
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


# The counts ``count_corrections`` gives, in printed order.
CORRECTION_COUNT_NAMES = (
    "pages kept as main page",
    "pages kept as part page",
    "old addresses dropped",
    "pages moved to a seed entry",
    "pages moved to a new entry",
    "pages added",
    "pages added as parts",
    "review cases answered",
    "review cases open",
)


def count_corrections(
    corrections: Iterable[SeedCorrection], review_cases: Iterable[ReviewCase]
) -> dict[str, int]:
    """Count what the corrections change, and how many review cases are open.

    Each correction that changes the seed is counted once, under one of these:

    Each item reads ``<count names>: <correction counted>``.

    - Pages added, pages added as parts: a page the entry's pages link and no
      seed entry holds joins the entry.
    - Old addresses dropped: the entry also lists the page's current address.
    - Pages moved to a seed entry, pages moved to a new entry: the page
      belongs to another work.
    - Pages kept as main page: the page stays and is its provider's main
      page, so the page is listed first.
    - Pages kept as part page: the page stays and covers part of its work, so
      enrichment combines it with the work's other parts.

    A page that stays in its entry, neither main page nor part page, is not counted.

    Args:
        corrections: Every seed correction of the run.
        review_cases: Every review case of the run, with its kept answer.

    Returns:
        Each count by its printed name.
    """
    counts = dict.fromkeys(CORRECTION_COUNT_NAMES, 0)
    for correction in corrections:
        if correction.is_added_page:
            counts[
                "pages added as parts" if correction.is_part_page else "pages added"
            ] += 1
        elif correction.renamed_to:
            counts["old addresses dropped"] += 1
        elif correction.target_entry_reference_page:
            counts["pages moved to a seed entry"] += 1
        elif correction.new_entry_key:
            counts["pages moved to a new entry"] += 1
        elif correction.is_main_page:
            counts["pages kept as main page"] += 1
        elif correction.is_part_page:
            counts["pages kept as part page"] += 1
    for review_case in review_cases:
        is_answered = review_case.accepted_answer is not None
        counts["review cases answered" if is_answered else "review cases open"] += 1
    return counts


def list_entries_to_check(
    seed_document: dict[str, Any],
    known_work_entries: KnownWorkEntriesFromSeed,
    data_dir: Path,
) -> list[tuple[WorkEntry, ...]]:
    """Write ``entries_to_check.json`` and return the entry groups to check.

    Args:
        seed_document: The downloaded seed release, as read from its file.
        known_work_entries: Every seed entry.
        data_dir: The folder ``entries_to_check.json`` is written to.

    Returns:
        The entry groups: each entry with a doubled provider and each group of
        split-work candidates, merged when they share an entry.
    """
    doubled_provider_entries = find_doubled_provider_entries(known_work_entries)
    candidate_groups = find_split_work_candidates(known_work_entries)
    data_dir.mkdir(parents=True, exist_ok=True)
    entries_to_check = build_entries_to_check_document(
        seed_document.get("$schema"), doubled_provider_entries, candidate_groups
    )
    replace_file_lines(
        data_dir / "entries_to_check.json",
        [json.dumps(entries_to_check, ensure_ascii=False, indent=1) + "\n"],
    )
    marked_group_count = sum(
        group.shared_provider is not None for group in candidate_groups
    )
    print(f"Entries with a doubled provider: {len(doubled_provider_entries)}")
    print(
        f"Split-work candidate groups: {len(candidate_groups)} ({marked_group_count} marked)"
    )
    entry_groups = find_entry_groups(
        doubled_provider_entries, candidate_groups, known_work_entries
    )
    print(f"Entry groups: {len(entry_groups)}")
    return entry_groups


@asynccontextmanager
async def open_page_fetcher(
    store: ProviderPageStore, arguments: argparse.Namespace
) -> AsyncIterator[ProviderPageFetcher | None]:
    """Open the HTTP cache and a page fetcher for the run.

    Args:
        store: The provider page store the fetcher adds pages to.
        arguments: The command-line arguments, for ``no_fetch`` and
            ``refetch_stored``.

    Yields:
        The page fetcher, or None when the run fetches nothing. The cache is
        closed when the run leaves the block.
    """
    if arguments.no_fetch:
        yield None
        return
    settings = get_settings()
    cache = configure_cache(settings.cache, settings.redis)
    try:
        yield ProviderPageFetcher(
            lambda: ApiFetcher(settings.enrichment, cache),
            store,
            is_stored_page_fetched_again=arguments.refetch_stored,
        )
    finally:
        await close_cache()


def print_fetch_summary(pages_name: str, fetch_summary: FetchSummary) -> None:
    """Print how a fetch went: a count per outcome, and each failure.

    Args:
        pages_name: The name of the pages fetched, starting each printed line.
        fetch_summary: The fetch's summary.
    """
    for outcome, page_count in sorted(fetch_summary.outcomes.items()):
        print(f"{pages_name} {outcome}: {page_count}")
    for request_label, fetch_error in fetch_summary.errors_by_request_label.items():
        print(f"Failed, retried next run: {request_label}: {fetch_error[:160]}")


async def add_linked_pages(
    entries: list[WorkEntry],
    group_corrections: PlacementCorrections,
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    page_fetcher: ProviderPageFetcher | None,
    review_file: ReviewFile,
) -> LinkedPageCorrections:
    """Fetch the pages entries link that no seed entry holds, and decide each.

    Args:
        entries: The entries whose links to read: the entries to check and the
            entries without a MAL page.
        group_corrections: The entry group corrections and
            review cases, which say which entries have placed pages.
        known_work_entries: Every seed entry.
        store: The provider page store.
        page_fetcher: The page fetcher, or None when the run fetches nothing.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections adding pages, the linked-page review cases, and the
        linked pages not added.
    """
    linking_seed_entries_by_page_key = find_linking_seed_entries(
        list_entries_with_placed_pages(
            entries, group_corrections, known_work_entries, store
        ),
        store,
        known_work_entries,
    )
    if page_fetcher is not None:
        print_fetch_summary(
            "Linked pages",
            # Every linked page is passed: the fetcher skips stored pages itself,
            # and still fetches the current key of one stored under a slug.
            await page_fetcher.fetch_linked_pages(
                {
                    key: linking_seed_entries[0].entry_details.entry.title
                    for key, linking_seed_entries in (
                        linking_seed_entries_by_page_key.items()
                    )
                }
            ),
        )
    return correct_linked_pages(
        linking_seed_entries_by_page_key, store, known_work_entries, review_file
    )


def write_results(
    data_dir: Path,
    review_file: ReviewFile,
    group_corrections: PlacementCorrections,
    linked_page_corrections: LinkedPageCorrections,
) -> None:
    """Write the review and corrections files, and print what they hold.

    Args:
        data_dir: The folder ``corrections.jsonl`` is written to.
        review_file: The review file the cases are written to.
        group_corrections: The entry group corrections and cases.
        linked_page_corrections: The linked-page corrections and cases.
    """
    review_cases = [
        *group_corrections.review_cases,
        *linked_page_corrections.review_cases,
    ]
    corrections = [
        *group_corrections.corrections,
        *linked_page_corrections.corrections,
    ]
    unused_answer_case_names = review_file.write_cases(review_cases)
    write_corrections(data_dir / "corrections.jsonl", corrections)
    for count_name, correction_count in count_corrections(
        corrections, review_cases
    ).items():
        print(f"{count_name.capitalize()}: {correction_count}")
    print(
        f"Linked pages waiting for a fetch: {len(linked_page_corrections.waiting_page_keys)}"
    )
    print(f"Linked pages not found: {len(linked_page_corrections.not_found_page_keys)}")
    print(
        f"Linked pages held under another address: {len(linked_page_corrections.held_page_keys)}"
    )
    for page_key in linked_page_corrections.conflicting_page_keys:
        print(
            f"Answer not applied, another page of its provider is added to the same entry: {page_key}"
        )
    for case_name in unused_answer_case_names:
        print(f"Answer kept, its case no longer comes up: {case_name}")


async def main() -> None:
    """List the entries to check, fetch their pages, correct them and add linked pages."""
    arguments = parse_arguments()
    seed_document = json.loads(arguments.seed_file.read_text(encoding="utf-8"))
    known_work_entries = KnownWorkEntriesFromSeed(seed_document["data"])
    entry_groups = list_entries_to_check(
        seed_document, known_work_entries, arguments.data_dir
    )
    entries_to_check = [entry for entries in entry_groups for entry in entries]
    entries_without_mal_page = find_entries_without_mal_page(known_work_entries)
    print(f"Entries without a MAL page: {len(entries_without_mal_page)}")
    store = ProviderPageStore(arguments.data_dir / "provider_pages.jsonl")
    review_file = ReviewFile(arguments.data_dir / "review.jsonl")
    async with open_page_fetcher(store, arguments) as page_fetcher:
        if page_fetcher is not None:
            # Every entry is passed on: the fetcher makes each request once, so
            # an entry sharing a page with another still gets its other pages.
            print_fetch_summary(
                "Pages",
                await page_fetcher.fetch_entry_pages(entries_to_check),
            )
            print_fetch_summary(
                "Pages of entries without a MAL page",
                await page_fetcher.fetch_entry_pages(
                    entries_without_mal_page, providers=LINK_CARRYING_PROVIDERS
                ),
            )
        group_corrections = correct_entry_groups(
            entry_groups, store, known_work_entries, review_file
        )
        linked_page_corrections = await add_linked_pages(
            [*entries_to_check, *entries_without_mal_page],
            group_corrections,
            known_work_entries,
            store,
            page_fetcher,
            review_file,
        )
    write_results(
        arguments.data_dir, review_file, group_corrections, linked_page_corrections
    )
    removed_entry_count = sum(
        corrected.entry is None
        for corrected in build_corrected_entries(
            group_corrections.corrections, known_work_entries, store
        )
    )
    print(f"Entries removed, every page moved out: {removed_entry_count}")


if __name__ == "__main__":
    # Set here rather than at import, so importing the script, as the tests do,
    # leaves logging as the importer set it.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    asyncio.run(main())

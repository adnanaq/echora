"""Find the seed entries that need checking, fetch their pages, and correct them.

Seed cleaning reads a downloaded seed release, never changes it, and runs in
two passes:

- First pass: entries with a doubled provider are placed, and pages entries
  link that no seed entry holds are added.
- Second pass: split-work candidates are found on the entries as the first
  pass leaves them, leaving out entries with a first-pass case still open, and
  placed. Its corrections stand in for the first pass's for the same pages.

It writes to the data folder:

- ``entries_to_check.json``: the entries with a doubled provider, and the
  groups of split-work candidates with the titles and links that made them;
- ``provider_pages.jsonl``: every page of those entries, the AniList, Kitsu
  and AniDB pages of entries without a MAL page (the pages that carry links),
  and the pages they link that no seed entry holds, fetched through the
  enrichment helpers. A rerun fetches only what is not stored yet. Split works
  are found by links too, so they are found among every entry whose pages the
  store holds: ``--store`` reads a store holding every seed entry's pages;
- ``review.jsonl``: one case per entry group the rules cannot place (an
  entry with doubled providers, or a group of split-work candidates), and per
  linked page they cannot add on their own, where a person writes the answer;
  answers are kept across runs;
- ``corrections.jsonl``: where each page of an entry group ends up, and each
  linked page added, from the rules or an accepted answer. Rewritten on each
  run.

AniDB caps requests per day, so a long run sets a wide gap and stops at the
first block, and the next day's run resumes: ``ANIDB_MIN_REQUEST_INTERVAL=8``
and ``ANIDB_BAN_COOLDOWN=86400``.

Usage:
    uv run python scripts/clean_seed.py [SEED_FILE] [--data-dir DIR]
        [--store PATH] [--no-fetch | --refetch-stored]
"""

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from common.config import get_settings
from enrichment.pipeline.api_fetcher import ApiFetcher
from http_cache import close_cache, configure_cache
from seed_cleaning import (
    LINK_CARRYING_PROVIDERS,
    EntryGroup,
    FetchSummary,
    LinkedPageCorrections,
    PlacementCorrections,
    ProviderPageFetcher,
    SeedCorrection,
    SplitWorkCandidates,
    SplitWorkPass,
    build_corrected_entries,
    build_entries_to_check_document,
    correct_entry_groups,
    correct_linked_pages,
    count_corrections,
    find_doubled_provider_entries,
    find_entries_without_mal_page,
    find_linking_seed_entries,
    list_entries_with_placed_pages,
    merge_corrections,
    prepare_split_work_pass,
    write_corrections,
)
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    replace_file_lines,
)


def parse_arguments() -> argparse.Namespace:
    """Read the seed file, the data folder and how to fetch from the command line.

    Returns:
        The arguments: ``seed_file``, ``data_dir``, ``store``, ``no_fetch``
        and ``refetch_stored``.
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
    parser.add_argument(
        "--store",
        type=Path,
        help=(
            "The provider page store to read and add to (default: "
            "provider_pages.jsonl in the data folder); a store holding every "
            "seed entry's pages finds split works by links among all entries"
        ),
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


def write_entries_to_check(
    seed_document: dict[str, Any],
    doubled_provider_entries: list[WorkEntry],
    candidate_groups: list[SplitWorkCandidates],
    data_dir: Path,
) -> None:
    """Write ``entries_to_check.json`` and print how many entries each pass checks.

    Args:
        seed_document: The downloaded seed release, as read from its file.
        doubled_provider_entries: The first pass's entries.
        candidate_groups: The second pass's groups of split-work candidates.
        data_dir: The folder ``entries_to_check.json`` is written to.
    """
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
    corrections: list[SeedCorrection],
    review_cases: list[ReviewCase],
    linked_page_corrections: LinkedPageCorrections,
) -> None:
    """Write the review and corrections files, and print what they hold.

    Args:
        data_dir: The folder ``corrections.jsonl`` is written to.
        review_file: The review file the cases are written to.
        corrections: Both passes' corrections, merged.
        review_cases: Both passes' review cases.
        linked_page_corrections: The linked-page outcomes, for their counts.
    """
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


def print_corrected_entry_counts(
    corrections: list[SeedCorrection],
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
) -> None:
    """Print how many entries the corrections remove, and how many new entries lack a type.

    A new entry without fetched page details gets an unknown type, so the count
    keeps such entries from being missed before indexing.

    Args:
        corrections: Both passes' corrections, merged.
        known_work_entries: Every seed entry.
        store: The fetched pages, for the new entries' details.
    """
    corrected_entries = build_corrected_entries(corrections, known_work_entries, store)
    removed_entry_count = sum(
        corrected.entry is None for corrected in corrected_entries
    )
    unknown_type_new_entry_count = sum(
        corrected.seed_entry_reference_page is None
        and corrected.entry is not None
        and corrected.entry.type == "UNKNOWN"
        for corrected in corrected_entries
    )
    print(f"Entries removed, every page moved out: {removed_entry_count}")
    print(f"New entries with an unknown type: {unknown_type_new_entry_count}")


async def run_first_pass(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    page_fetcher: ProviderPageFetcher | None,
    review_file: ReviewFile,
) -> tuple[PlacementCorrections, LinkedPageCorrections]:
    """Place the entries with a doubled provider and add the pages entries link.

    Args:
        known_work_entries: Every seed entry.
        store: The provider page store.
        page_fetcher: The page fetcher, or None when the run fetches nothing.
        review_file: The review file, for the answers kept in it.

    Returns:
        The doubled-provider corrections and cases, and the linked-page ones.
    """
    doubled_provider_entries = find_doubled_provider_entries(known_work_entries)
    entries_without_mal_page = find_entries_without_mal_page(known_work_entries)
    print(f"Entries without a MAL page: {len(entries_without_mal_page)}")
    if page_fetcher is not None:
        print_fetch_summary(
            "Pages", await page_fetcher.fetch_entry_pages(doubled_provider_entries)
        )
        print_fetch_summary(
            "Pages of entries without a MAL page",
            await page_fetcher.fetch_entry_pages(
                entries_without_mal_page, providers=LINK_CARRYING_PROVIDERS
            ),
        )
    group_corrections = correct_entry_groups(
        [EntryGroup((entry,)) for entry in doubled_provider_entries],
        store,
        known_work_entries,
        review_file,
    )
    linked_page_corrections = await add_linked_pages(
        [*doubled_provider_entries, *entries_without_mal_page],
        group_corrections,
        known_work_entries,
        store,
        page_fetcher,
        review_file,
    )
    return group_corrections, linked_page_corrections


async def run_second_pass(
    split_work_pass: SplitWorkPass,
    store: ProviderPageStore,
    page_fetcher: ProviderPageFetcher | None,
    review_file: ReviewFile,
) -> PlacementCorrections:
    """Fetch the split-work candidates' pages and place each group.

    Args:
        split_work_pass: The second pass's entries and candidate groups.
        store: The provider page store.
        page_fetcher: The page fetcher, or None when the run fetches nothing.
        review_file: The review file, for the answers kept in it.

    Returns:
        The split-work corrections and cases.
    """
    if page_fetcher is not None:
        print_fetch_summary(
            "Pages of split-work candidates",
            await page_fetcher.fetch_entry_pages(
                [
                    entry
                    for candidates in split_work_pass.candidates
                    for entry in candidates.entries
                ]
            ),
        )
    return correct_entry_groups(
        [EntryGroup.from_candidates(group) for group in split_work_pass.candidates],
        store,
        split_work_pass.known_work_entries,
        review_file,
        split_work_pass.corrected_seed,
    )


async def main() -> None:
    """Run both passes over the seed and write what they decided."""
    arguments = parse_arguments()
    seed_document = json.loads(arguments.seed_file.read_text(encoding="utf-8"))
    known_work_entries = KnownWorkEntriesFromSeed(seed_document["data"])
    arguments.data_dir.mkdir(parents=True, exist_ok=True)
    store = ProviderPageStore(
        arguments.store or arguments.data_dir / "provider_pages.jsonl"
    )
    review_file = ReviewFile(arguments.data_dir / "review.jsonl")
    async with open_page_fetcher(store, arguments) as page_fetcher:
        group_corrections, linked_page_corrections = await run_first_pass(
            known_work_entries, store, page_fetcher, review_file
        )
        first_pass_corrections = [
            *group_corrections.corrections,
            *linked_page_corrections.corrections,
        ]
        first_pass_cases = [
            *group_corrections.review_cases,
            *linked_page_corrections.review_cases,
        ]
        split_work_pass = prepare_split_work_pass(
            first_pass_corrections, first_pass_cases, known_work_entries, store
        )
        split_work_corrections = await run_second_pass(
            split_work_pass, store, page_fetcher, review_file
        )
    write_entries_to_check(
        seed_document,
        find_doubled_provider_entries(known_work_entries),
        split_work_pass.candidates,
        arguments.data_dir,
    )
    corrections = merge_corrections(
        first_pass_corrections, split_work_corrections.corrections
    )
    write_results(
        arguments.data_dir,
        review_file,
        corrections,
        [*first_pass_cases, *split_work_corrections.review_cases],
        linked_page_corrections,
    )
    print_corrected_entry_counts(corrections, known_work_entries, store)


if __name__ == "__main__":
    # Set here rather than at import, so importing the script, as the tests do,
    # leaves logging as the importer set it.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    asyncio.run(main())

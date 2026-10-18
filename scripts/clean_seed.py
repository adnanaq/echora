"""Find the seed entries that need checking, fetch their pages, and correct them.

Seed cleaning reads a downloaded seed release, never changes it, and writes to
the data folder:

- ``entries_to_check.json``: the entries with a doubled provider, and the
  groups of split-work candidates;
- ``provider_pages.jsonl``: every page of those entries, fetched through the
  enrichment helpers. A rerun fetches only what is not stored yet;
- ``review.jsonl``: one case per entry the rules cannot place, where a person
  writes the answer; answers are kept across runs;
- ``corrections.jsonl``: where each page of an entry with doubled providers
  ends up, from the rules or an accepted answer. Rewritten on each run.

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
    DoubledProviderCorrections,
    FetchSummary,
    ProviderPageFetcher,
    SplitWorkCandidates,
    correct_doubled_provider_entries,
    find_doubled_provider_entries,
    find_split_work_candidates,
    write_corrections,
)
from work_matching import (
    ENTRY_WORK,
    KnownWorksFromSeed,
    ProviderPageStore,
    ReviewFile,
    WorkEntry,
    find_doubled_provider_pages,
    replace_file_lines,
)


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
        help="Fetch no pages; correct the entries from the pages already stored",
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
                "doubled_providers": find_doubled_provider_pages(entry.pages),
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


def count_corrections(
    doubled_provider_corrections: DoubledProviderCorrections,
) -> dict[str, int]:
    """Count what the corrections change, and how many review cases are open.

    Each correction that changes the seed is counted once, under one of these:

    - an old address dropped, because the entry also lists the page's current
      address;
    - a page moved to another seed entry, or to a new entry;
    - a page that stays in its entry but is its provider's main page, so it is
      moved to the front of the entry's pages.

    A page that stays in its entry and is not the main page changes nothing,
    so it is not counted.
    """
    counts = {
        "pages kept as main page": 0,
        "old addresses dropped": 0,
        "pages moved to a seed entry": 0,
        "pages moved to a new entry": 0,
        "review cases answered": 0,
        "review cases open": 0,
    }
    for correction in doubled_provider_corrections.corrections:
        if correction.renamed_to:
            counts["old addresses dropped"] += 1
        elif correction.target_entry_reference_page:
            counts["pages moved to a seed entry"] += 1
        elif correction.work != ENTRY_WORK:
            counts["pages moved to a new entry"] += 1
        elif correction.is_main_page:
            counts["pages kept as main page"] += 1
    for review_case in doubled_provider_corrections.review_cases:
        is_answered = review_case.accepted_answer is not None
        counts["review cases answered" if is_answered else "review cases open"] += 1
    return counts


def correct_entries(
    entries: list[WorkEntry], known_works: KnownWorksFromSeed, data_dir: Path
) -> None:
    """Correct the entries from the stored pages, and write the review and corrections."""
    review_file = ReviewFile(data_dir / "review.jsonl")
    doubled_provider_corrections = correct_doubled_provider_entries(
        entries,
        ProviderPageStore(data_dir / "provider_pages.jsonl"),
        known_works,
        review_file,
    )
    unused_answer_case_names = review_file.write_cases(
        doubled_provider_corrections.review_cases
    )
    write_corrections(
        data_dir / "corrections.jsonl", doubled_provider_corrections.corrections
    )
    for count_name, correction_count in count_corrections(
        doubled_provider_corrections
    ).items():
        print(f"{count_name.capitalize()}: {correction_count}")
    for case_name in unused_answer_case_names:
        print(f"Answer kept, its case no longer comes up: {case_name}")


async def main() -> None:
    """List the entries to check, write them, and fetch their pages."""
    arguments = parse_arguments()
    seed_document = json.loads(arguments.seed_file.read_text(encoding="utf-8"))
    known_works = KnownWorksFromSeed(seed_document["data"])
    doubled_provider_entries = find_doubled_provider_entries(known_works)
    candidate_groups = find_split_work_candidates(known_works)

    arguments.data_dir.mkdir(parents=True, exist_ok=True)
    entries_to_check = build_entries_to_check_document(
        seed_document.get("$schema"), doubled_provider_entries, candidate_groups
    )
    replace_file_lines(
        arguments.data_dir / "entries_to_check.json",
        [json.dumps(entries_to_check, ensure_ascii=False, indent=1) + "\n"],
    )
    marked_group_count = sum(
        group.shared_provider is not None for group in candidate_groups
    )
    print(f"Entries with a doubled provider: {len(doubled_provider_entries)}")
    print(
        f"Split-work candidate groups: {len(candidate_groups)} ({marked_group_count} marked)"
    )
    if not arguments.no_fetch:
        # Every entry is passed on: the fetcher makes each request once, so an
        # entry sharing a page with another still gets its other pages fetched.
        fetch_summary = await fetch_pages(
            [
                *doubled_provider_entries,
                *(entry for group in candidate_groups for entry in group.entries),
            ],
            arguments.data_dir / "provider_pages.jsonl",
        )
        for outcome, page_count in sorted(fetch_summary.outcomes.items()):
            print(f"Pages {outcome}: {page_count}")
        for failed_page, fetch_error in fetch_summary.failed_pages.items():
            print(f"Failed, retried next run: {failed_page}: {fetch_error[:160]}")
    correct_entries(doubled_provider_entries, known_works, arguments.data_dir)


if __name__ == "__main__":
    # Set here rather than at import, so importing the script, as the tests do,
    # leaves logging as the importer set it.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    asyncio.run(main())

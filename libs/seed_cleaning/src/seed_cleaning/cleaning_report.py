"""Describe a seed-cleaning run: the entries it checks and what its corrections change.

``scripts/clean_seed.py`` writes the entries to check to a file and prints the
corrections' counts; both are built here so they can be tested without a run.
"""

from collections.abc import Iterable
from typing import Any

from seed_cleaning.seed_corrections import SeedCorrection
from seed_cleaning.split_work_candidates import SplitWorkCandidates
from work_matching import ReviewCase, WorkEntry, find_doubled_provider_page_keys


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
                "candidate_evidence": list(group.candidate_evidence),
                "has_shared_tying_link": group.has_shared_tying_link,
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

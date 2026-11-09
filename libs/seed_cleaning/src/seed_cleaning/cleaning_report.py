"""Describe a seed-cleaning run: the entries it checks and what its corrections change.

``scripts/clean_seed.py`` writes the entries to check to a file and prints the
corrections' counts; both are built here so they can be tested without a run.
"""

from collections.abc import Iterable
from typing import Any

from seed_cleaning.corrected_entries import CorrectedEntry
from seed_cleaning.linked_page_additions import LinkedPageCorrections
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
    releases while its position in the file does not.

    Args:
        seed_schema: The seed file's ``$schema`` address naming the release, or
            None when the file has none.
        doubled_provider_entries: The seed entries holding a doubled provider.
        candidate_groups: The split-work candidate groups found in the second pass.

    Returns:
        The JSON-serialisable dictionary for ``entries_to_check.json``.
    """
    return {
        "seed_schema": seed_schema,
        "doubled_provider_entries": [
            {
                "reference_page": entry.reference_page_key,
                "title": entry.title,
                "doubled_providers": find_doubled_provider_page_keys(
                    entry.addresses_by_page_key
                ),
            }
            for entry in doubled_provider_entries
        ],
        "split_work_candidates": [
            {
                "shared_provider": candidate_group.shared_provider,
                "candidate_evidence": list(candidate_group.candidate_evidence),
                "has_shared_tying_link": candidate_group.has_shared_tying_link,
                "entries": [
                    {"reference_page": entry.reference_page_key}
                    | entry.model_dump(mode="json")
                    for entry in candidate_group.entries
                ],
            }
            for candidate_group in candidate_groups
        ],
    }


# The counts ``count_corrections`` gives, in printed order.
CORRECTION_COUNT_NAMES = (
    "partial sources added",
    "duplicate pages dropped",
    "old addresses dropped",
    "dead pages dropped",
    "pages moved to a seed entry",
    "pages moved to a new entry",
    "pages added",
    "pages added as partial sources",
    "review cases answered",
    "review cases open",
    "open cases placed by a previous answer",
)


def count_corrections(
    corrections: Iterable[SeedCorrection], review_cases: Iterable[ReviewCase]
) -> dict[str, int]:
    """Count what the corrections change, and how many review cases are open.

    Each correction that changes the seed is counted once, under one of these:

    Each item reads ``<count names>: <correction counted>``.

    - Pages added, pages added as partial sources: a page the entry's pages
      link and no seed entry holds joins the entry's own pages or its partial
      sources.
    - Old addresses dropped: the entry also lists the page's current address.
    - Dead pages dropped: the provider says the page does not exist.
    - Duplicate pages dropped: an answer drops the page as a duplicate page.
    - Pages moved to a seed entry, pages moved to a new entry: the page
      belongs to another work.
    - Partial sources added: the page stays and also joins other entries as a
      partial source, because it covers their works too.

    A page that stays in its entry and is no partial source is not counted.
    Open cases placed by a previous answer are open cases a seed release
    renamed whose previous answer still places the pages it saw; they are
    counted among the open cases too.

    Args:
        corrections: Every seed correction of the run.
        review_cases: Every review case of the run, with its kept answer.

    Returns:
        Each count by its printed name.
    """
    counts = dict.fromkeys(CORRECTION_COUNT_NAMES, 0)
    for correction in corrections:
        if change_name := find_correction_change(correction):
            counts[change_name] += 1
    for review_case in review_cases:
        is_answered = review_case.accepted_answer is not None
        counts["review cases answered" if is_answered else "review cases open"] += 1
        if (
            not is_answered
            and review_case.previous_answer is not None
            and review_case.previous_answer.is_applied
        ):
            counts["open cases placed by a previous answer"] += 1
    return counts


def find_correction_change(correction: SeedCorrection) -> str | None:
    """Return the change a correction makes, named as ``count_corrections`` counts it.

    The first matching change wins, in this order, so each correction has one:
    added page (as a partial source or a source), old address dropped, dead
    page dropped, duplicate page dropped, moved to a seed entry, moved to a
    new entry, partial source added.

    Args:
        correction: One seed correction.

    Returns:
        The count name, or None for a page that stays in its entry unchanged.
    """
    is_partial_source = bool(
        correction.partial_entry_reference_pages or correction.partial_new_entry_keys
    )
    if correction.is_added_page:
        return "pages added as partial sources" if is_partial_source else "pages added"
    if correction.renamed_to:
        return "old addresses dropped"
    if correction.is_dead_page:
        return "dead pages dropped"
    if correction.is_dropped_page:
        return "duplicate pages dropped"
    if correction.target_entry_reference_page:
        return "pages moved to a seed entry"
    if correction.new_entry_key:
        return "pages moved to a new entry"
    if is_partial_source:
        return "partial sources added"
    return None


def count_corrected_entries(
    corrected_entries: Iterable[CorrectedEntry],
) -> dict[str, int]:
    """Count the entries the corrections remove, and the new entries without a type.

    A new entry without fetched page details gets an unknown type, so the count
    keeps such entries from being missed before indexing.

    Args:
        corrected_entries: The entries both passes' corrections change or create.

    Returns:
        Each count's name mapped to its count, in the order the run reports them.
    """
    corrected_entries = list(corrected_entries)
    return {
        "entries removed, no page left": sum(
            corrected_entry.entry is None for corrected_entry in corrected_entries
        ),
        "new entries with an unknown type": sum(
            corrected_entry.seed_entry_reference_page is None
            and corrected_entry.entry is not None
            and corrected_entry.entry.type == "UNKNOWN"
            for corrected_entry in corrected_entries
        ),
    }


def describe_linked_page_outcomes(
    linked_page_corrections: LinkedPageCorrections,
) -> list[str]:
    """Describe the linked pages not added, in report lines.

    Args:
        linked_page_corrections: The linked-page step's outcomes.

    Returns:
        A count of the pages waiting for a fetch, not found, and held under
        another address, then one line per page an answer could not add
        because another page of its provider joins the same entry.
    """
    return [
        f"Linked pages waiting for a fetch: {len(linked_page_corrections.waiting_page_keys)}",
        f"Linked pages not found: {len(linked_page_corrections.not_found_page_keys)}",
        "Linked pages held under another address: "
        f"{len(linked_page_corrections.held_page_keys)}",
        *(
            "Answer not applied, another page of its provider is added to the "
            f"same entry: {page_key}"
            for page_key in linked_page_corrections.conflicting_page_keys
        ),
    ]

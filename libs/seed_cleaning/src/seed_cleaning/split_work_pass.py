"""Prepare the second pass: split-work candidates on the first pass's entries.

The first pass resolves doubled providers and adds linked pages. Split works
are then looked for on the entries as that pass leaves them, so a page the
first pass moved or added is grouped where it now is, and a new entry an answer
made can be a candidate too. The rules:

Each item gives the rule, then the reason for it.

- Entries with an open first-pass case are left out. Their pages are not
  placed yet, so grouping them would ask about pages twice at once; they join
  the second pass once their case is answered.
- Links and titles are read from the corrected entries
  (``split_work_links``, ``split_work_candidates``). Corrections are
  rebuilt from the seed every run, so the second pass always sees the first
  pass's latest answers.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from seed_cleaning.corrected_seed import CorrectedSeed, build_corrected_seed
from seed_cleaning.seed_corrections import SeedCorrection
from seed_cleaning.split_work_candidates import (
    SplitWorkCandidates,
    find_split_work_candidates,
)
from seed_cleaning.split_work_links import find_split_work_links
from work_matching import KnownWorkEntriesFromSeed, ProviderPageStore, ReviewCase


@dataclass(frozen=True)
class SplitWorkPass:
    """The second pass's entries and candidates, and how corrections name them."""

    # Every entry as the first pass leaves it, for naming pages and entries.
    corrected_seed: CorrectedSeed
    # The entries the second pass groups: the corrected seed without the
    # entries of open first-pass cases.
    known_work_entries: KnownWorkEntriesFromSeed
    candidates: list[SplitWorkCandidates]


def prepare_split_work_pass(
    first_pass_corrections: Iterable[SeedCorrection],
    first_pass_review_cases: Iterable[ReviewCase],
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
) -> SplitWorkPass:
    """Find the split-work candidates among the entries the first pass leaves.

    Args:
        first_pass_corrections: The first pass's corrections.
        first_pass_review_cases: The first pass's review cases, with their
            kept answers; the entries of each open case are left out.
        known_work_entries: Every seed entry.
        store: The fetched pages, for the corrected entries' links.

    Returns:
        The corrected seed, the entries grouped, and their candidate groups.
    """
    corrected_seed = build_corrected_seed(
        first_pass_corrections, known_work_entries, store
    )
    open_case_reference_pages = {
        reference_page
        for review_case in first_pass_review_cases
        if review_case.accepted_answer is None
        for reference_page in review_case.entries
    }
    grouped_entries = KnownWorkEntriesFromSeed.from_work_entries(
        entry
        for entry in corrected_seed.known_work_entries
        if corrected_seed.identity_by_reference_page[
            entry.reference_page
        ].seed_entry_reference_page
        not in open_case_reference_pages
    )
    return SplitWorkPass(
        corrected_seed,
        grouped_entries,
        find_split_work_candidates(
            grouped_entries, find_split_work_links(grouped_entries, store)
        ),
    )

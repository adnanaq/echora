"""Prepare the second pass: split-work candidates on the first pass's entries.

The first pass resolves doubled providers and adds linked pages. Split works
are then looked for on the entries as that pass leaves them, so a page the
first pass moved or added is grouped where it now is, and a new entry an answer
made can be a candidate too. The rules:

Each item gives the rule, then the reason for it.

- Entries with an open first-pass case are left out. The pages of such
  entries are not placed yet, so grouping them would ask about pages twice at
  once. These entries join the second pass once their case is answered.
- Two entries a first-pass answer separated, by moving a page from one into
  the other, are not grouped by their own title match or link
  (``split_work_candidates``). Otherwise every answered split would come back
  as a split-work case asking the same question.
- Links and titles are read from the corrected entries
  (``split_work_links``, ``split_work_candidates``). Corrections are
  rebuilt from the seed every run, so the second pass always sees the first
  pass's latest answers.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from seed_cleaning.corrected_seed import (
    CorrectedSeed,
    EntryIdentity,
    build_corrected_seed,
    find_move_destination,
)
from seed_cleaning.seed_corrections import REVIEW_ANSWER_REASON, SeedCorrection
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
    grouped_entries = KnownWorkEntriesFromSeed.index_work_entries(
        entry
        for entry in corrected_seed.known_work_entries
        if corrected_seed.identity_by_reference_page[
            entry.reference_page_key
        ].seed_entry_reference_page
        not in open_case_reference_pages
    )
    return SplitWorkPass(
        corrected_seed,
        grouped_entries,
        find_split_work_candidates(
            grouped_entries,
            find_split_work_links(grouped_entries, store),
            _find_separated_reference_page_pairs(
                first_pass_corrections, corrected_seed
            ),
        ),
    )


def _find_separated_reference_page_pairs(
    first_pass_corrections: Iterable[SeedCorrection], corrected_seed: CorrectedSeed
) -> frozenset[frozenset[str]]:
    """Find each two corrected entries a first-pass answer separated.

    An answer that moves a page out of its seed entry, into a new entry or
    another seed entry, says the page is a different work from the entry it
    leaves.

    Args:
        first_pass_corrections: The first pass's corrections.
        corrected_seed: The entries the first pass leaves, with how
            corrections name them.

    Returns:
        The reference pages of the entry each answered page left and the
        entry it joined, one pair per move.
    """
    reference_page_by_identity = {
        identity: reference_page
        for reference_page, identity in corrected_seed.identity_by_reference_page.items()
    }
    return frozenset(
        frozenset((origin_reference_page, destination_reference_page))
        for correction in first_pass_corrections
        if correction.reason == REVIEW_ANSWER_REASON
        and (destination := find_move_destination(correction))
        and (
            origin_reference_page := reference_page_by_identity.get(
                EntryIdentity(correction.entry_reference_page, None)
            )
        )
        and (destination_reference_page := reference_page_by_identity.get(destination))
        and origin_reference_page != destination_reference_page
    )

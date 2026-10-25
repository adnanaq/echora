"""Surviving entries: which seed entry each work of a placement keeps.

When pages of one or more seed entries are placed into works, each work keeps
at most one seed entry, and the other entries' pages in that work move into it.
Keeping a seed entry rather than building a new one keeps its anime id, so
search entries stored under that id stay valid. The rules:

- Only an entry whose reference page is placed in the work can survive as that
  work. The reference page is the page the entry is known by.
- An entry whose reference page is a whole page wins over one whose reference
  page is a part page. A part entry stands for only part of the work, so a
  whole entry names the work better. A part entry still survives when no
  whole entry is placed in the work: building a new entry instead would lose
  the entry's anime id and split a single entry's pages between two entries.
- Among the entries left, the entry whose reference page has the most trusted
  provider survives (MAL first), because that provider names the work best.
  Two reference pages of the same provider are told apart by their page key,
  so every run picks the same entry.
- A reference page with no placement counts as ``work 1``. Only rule placements
  leave pages unplaced, and those keep the entry as its first work.
- A work in which no entry's reference page is placed has no surviving entry,
  and becomes a new entry.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import NamedTuple

from work_matching.page_placement import FIRST_WORK, OTHER_WORK, PagePlacement
from work_matching.provider_page_key import rank_by_provider_trust
from work_matching.work_entry import WorkEntry


class _SurvivorCandidate(NamedTuple):
    """An entry that may survive as a work, ordered so the survivor sorts first."""

    # False sorts first, so a whole reference page wins over a part page.
    is_part_page: bool
    # How far the reference page's provider is trusted, MAL first.
    provider_rank: int
    reference_page: str


def find_surviving_entries(
    entries: Iterable[WorkEntry], placements: Mapping[str, PagePlacement]
) -> dict[str, str]:
    """Find the seed entry each work keeps, by the rules in the module docstring.

    Args:
        entries: The seed entries whose pages are placed.
        placements: Each placed page's placement, by its page key.

    Returns:
        Each work's placement word (``work 1``, ``work 2``…), mapped to the
        reference page of the entry the work keeps. A work without a
        surviving entry is left out; ``other work`` is never included.
    """
    candidates_by_work: defaultdict[str, list[_SurvivorCandidate]] = defaultdict(list)
    for entry in entries:
        placement = placements.get(entry.reference_page, PagePlacement(FIRST_WORK))
        if placement.placement_word != OTHER_WORK:
            candidates_by_work[placement.placement_word].append(
                _SurvivorCandidate(
                    placement.is_part_page,
                    rank_by_provider_trust(entry.reference_page),
                    entry.reference_page,
                )
            )
    return {
        placement_word: min(candidates).reference_page
        for placement_word, candidates in candidates_by_work.items()
    }

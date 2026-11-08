"""Find which seed entry each work of a placement keeps.

When pages of one or more seed entries are placed into works, each work keeps
at most one seed entry, and the other entries' pages in that work move into it.
Keeping a seed entry rather than building a new one keeps its anime id, so
search entries stored under that id stay valid. The rules:

- Only an entry whose reference page is placed in the work can survive as that
  work. The reference page is the page the entry is known by.
- Among the entries placed in the work, the entry whose reference page has
  the most trusted provider survives (MAL first), because that provider names
  the work best.
  Two reference pages of the same provider are told apart by their page key,
  so every run picks the same entry.
- A renamed reference page is looked up under its current key first, since a
  doubled page is placed under its current key.
- A reference page with no placement counts as ``work 1``. Only rule placements
  leave pages unplaced, and those keep the entry as its first work.
- A work in which no entry's reference page is placed has no surviving entry,
  and becomes a new entry.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import NamedTuple

from work_matching.page_placement import (
    FIRST_WORK,
    PagePlacement,
    find_page_placement,
)
from work_matching.provider_page_key import rank_by_provider_trust
from work_matching.work_entry import WorkEntry


class _SurvivorCandidate(NamedTuple):
    """Hold an entry that may survive as a work, ordered so the survivor sorts first."""

    # How far the reference page's provider is trusted, MAL first.
    provider_rank: int
    reference_page: str


def find_surviving_entries(
    entries: Iterable[WorkEntry],
    placements: Mapping[str, PagePlacement],
    current_page_key_by_page_key: Mapping[str, str],
) -> dict[str, str]:
    """Find the seed entry each work keeps, by the rules in the module docstring.

    Args:
        entries: The seed entries whose pages are placed.
        placements: Each placed page's placement, by its page key.
        current_page_key_by_page_key: Each renamed page's current key, by the
            key an entry lists it under.

    Returns:
        Each work's placement word (``work 1``, ``work 2`` and so on), mapped to the
        reference page of the entry the work keeps. A work without a
        surviving entry is left out, and neither ``other work`` nor
        ``dropped`` is ever included.
    """
    candidates_by_work: defaultdict[str, list[_SurvivorCandidate]] = defaultdict(list)
    for entry in entries:
        placed_page = find_page_placement(
            entry.reference_page_key, placements, current_page_key_by_page_key
        )
        placement = placed_page.placement if placed_page else PagePlacement(FIRST_WORK)
        if placement.is_case_work:
            candidates_by_work[placement.placement_word].append(
                _SurvivorCandidate(
                    rank_by_provider_trust(entry.reference_page_key),
                    entry.reference_page_key,
                )
            )
    return {
        placement_word: min(candidates).reference_page
        for placement_word, candidates in candidates_by_work.items()
    }

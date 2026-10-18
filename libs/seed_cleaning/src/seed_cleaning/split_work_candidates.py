"""Find groups of seed entries that may be one work split over several entries.

The seed sometimes lists one work as two entries: a site adds a page before the
seed joins it to the work's main entry, so the page becomes an entry of its
own. Such entries share a title, so candidates are found by title, type, year
and season. A title alone does not prove one work: the split-work rules merge a
group only when links between its entries' pages tie them together.
"""

import logging
from collections import Counter
from collections.abc import Sequence

from networkx.utils import UnionFind
from pydantic import BaseModel, ConfigDict
from work_matching import KnownWorksFromSeed, WorkEntry, decided_provider

logger = logging.getLogger(__name__)

# Seasons in calendar order, so neighbouring seasons differ by one.
_SEASON_ORDER = ("WINTER", "SPRING", "SUMMER", "FALL")


class SplitWorkCandidates(BaseModel):
    """Seed entries that may be one work, for the split-work rules to decide."""

    model_config = ConfigDict(frozen=True)

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # The one provider with pages in two or more of the entries, if any. That
    # provider lists the entries as separate pages, which alone does not decide:
    # such groups are sometimes one work and sometimes not.
    shared_provider: str | None = None


def split_work_candidates(known_works: KnownWorksFromSeed) -> list[SplitWorkCandidates]:
    """Return every group of entries that may be one work split in several.

    Rules:

    - Title, type and year: two entries match when the known-works title
      lookup finds one for the other (title equal once case, punctuation,
      accents and season wording are folded; same type; equal years, or a
      missing year).
    - Season: their seasons are equal, next to each other, or one is missing.
      A small entry often has no season yet; seasons far apart are different
      works ("Gelpiyo", winter and fall); neighbouring ones go either way.
    - Groups: entries joined through a shared match form one group, and the
      split-work rules send a group whose years or types clash to review.
    - Providers: a group whose entries share no provider is a candidate; one
      sharing exactly one provider is kept but marked with it; one sharing two
      or more is dropped and logged, since two providers each listing the
      entries separately points to different works. Simkl and AnimeCountdown
      count as one provider.

    Args:
        known_works: Every entry of the seed, by page and by title.

    Returns:
        The kept groups, ordered by their first entry's place in the seed.
    """
    candidates: list[SplitWorkCandidates] = []
    for entries in _title_groups(known_works):
        shared = _shared_providers(entries)
        if len(shared) > 1:
            logger.warning(
                f"Split-work group dropped, its entries share {', '.join(shared)}: "
                f"{', '.join(entry.title for entry in entries)}"
            )
            continue
        candidates.append(
            SplitWorkCandidates(
                entries=entries, shared_provider=shared[0] if shared else None
            )
        )
    return candidates


def _title_groups(known_works: KnownWorksFromSeed) -> list[tuple[WorkEntry, ...]]:
    """Group the entries that match by title, type, year and season.

    Two matching entries join their groups, so entries joined through a shared
    match end in one group.

    Returns:
        Each group of two or more entries, in seed file order.
    """
    works = list(known_works)
    # By object, not reference page: two entries can list the same page, and
    # find_by_title returns the very entries the seed lookup holds.
    position_of = {id(work): position for position, work in enumerate(works)}
    groups = UnionFind(range(len(works)))
    for position, work in enumerate(works):
        for match in known_works.find_by_title(work.title, work.type, work.year):
            if match is not work and _is_season_close(work.season, match.season):
                groups.union(position, position_of[id(match)])
    positions_by_group = sorted(
        sorted(positions) for positions in groups.to_sets() if len(positions) > 1
    )
    return [
        tuple(works[position] for position in positions)
        for positions in positions_by_group
    ]


def _is_season_close(first: str | None, second: str | None) -> bool:
    """Report whether two seasons are equal, next to each other, or one is missing.

    Examples:
        >>> _is_season_close("SUMMER", "FALL")
        True
        >>> _is_season_close("WINTER", "FALL")
        False
        >>> _is_season_close(None, "FALL")
        True
    """
    if first is None or second is None:
        return True
    return abs(_SEASON_ORDER.index(first) - _SEASON_ORDER.index(second)) <= 1


def _shared_providers(entries: Sequence[WorkEntry]) -> list[str]:
    """Return the providers with pages in two or more of the entries, sorted."""
    entries_per_provider = Counter(
        provider for entry in entries for provider in _providers_of(entry)
    )
    return sorted(
        provider for provider, count in entries_per_provider.items() if count > 1
    )


def _providers_of(entry: WorkEntry) -> set[str]:
    """Return the providers an entry has pages from, Simkl and AnimeCountdown as one."""
    return {provider for key in entry.pages if (provider := decided_provider(key))}

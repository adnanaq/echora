"""Find groups of seed entries that may be one work split over several entries.

The seed sometimes lists one work as two entries: a site adds a page before the
seed joins it to the work's main entry, so the page becomes an entry of its
own. Such entries share a title, so candidates are found by title, type, year
and season. A title alone does not prove one work, so every candidate group
goes to review as an entry group (``entry_groups``).
"""

import logging
from collections import Counter
from collections.abc import Sequence

from networkx.utils import UnionFind
from pydantic import BaseModel, ConfigDict
from work_matching import KnownWorkEntriesFromSeed, WorkEntry, get_decided_provider

logger = logging.getLogger(__name__)

# Seasons in calendar order, so neighbouring seasons differ by one.
_SEASON_ORDER = ("WINTER", "SPRING", "SUMMER", "FALL")
# Entries sharing this provider are never one work: a work is what MAL lists.
_MAL_PROVIDER = "mal"


class SplitWorkCandidates(BaseModel):
    """Seed entries that may be one work, for a person to review as one group."""

    model_config = ConfigDict(frozen=True)

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # The one provider with pages in two or more of the entries, if any. That
    # provider lists the entries as separate pages, which alone does not decide:
    # such groups are sometimes one work and sometimes not.
    shared_provider: str | None = None


def find_split_work_candidates(
    known_work_entries: KnownWorkEntriesFromSeed,
) -> list[SplitWorkCandidates]:
    """Return every group of entries that may be one work split in several.

    Rules:

    - Title, type and year: two entries match when the title lookup of the
      known work entries finds one for the other. Titles are equal once case,
      punctuation, accents and season wording are folded; types are the same;
      years are equal, or one is missing.
    - Season: their seasons are equal, next to each other, or one is missing.
      A small entry often has no season yet; seasons far apart are different
      works ("Gelpiyo", winter and fall); neighbouring ones go either way.
    - Groups: entries joined through a shared match form one group. The
      review case lists a group whose years or types clash.
    - Providers: a group whose entries share no provider is a candidate; one
      sharing exactly one provider is kept but marked with it; one sharing two
      or more is dropped and logged, since two providers each listing the
      entries separately points to different works. Simkl and AnimeCountdown
      count as one provider.
    - MAL: a group whose entries share MAL is dropped and logged. A work is
      what MAL lists, so entries holding different MAL pages are never one
      work.

    Args:
        known_work_entries: Every entry of the seed, by page and by title.

    Returns:
        The kept groups, ordered by their first entry's place in the seed.
    """
    candidates: list[SplitWorkCandidates] = []
    for entries in _group_entries_by_title(known_work_entries):
        shared_providers = _find_shared_providers(entries)
        if len(shared_providers) > 1 or _MAL_PROVIDER in shared_providers:
            logger.warning(
                f"Split-work group dropped, its entries share {', '.join(shared_providers)}: "
                f"{', '.join(entry.title for entry in entries)}"
            )
            continue
        candidates.append(
            SplitWorkCandidates(
                entries=entries,
                shared_provider=shared_providers[0] if shared_providers else None,
            )
        )
    return candidates


def _group_entries_by_title(
    known_work_entries: KnownWorkEntriesFromSeed,
) -> list[tuple[WorkEntry, ...]]:
    """Group the entries that match by title, type, year and season.

    Two matching entries join their groups, so entries joined through a shared
    match end in one group.

    Returns:
        Each group of two or more entries, in seed file order.
    """
    work_entries = list(known_work_entries)
    # By object, not reference page: two entries can list the same page, and
    # find_by_title returns the very entries the seed lookup holds.
    position_by_entry_id = {
        id(work_entry): position for position, work_entry in enumerate(work_entries)
    }
    entry_groups = UnionFind(range(len(work_entries)))
    for position, work_entry in enumerate(work_entries):
        for matching_work_entry in known_work_entries.find_by_title(
            work_entry.title, work_entry.type, work_entry.year
        ):
            if matching_work_entry is not work_entry and _is_season_close(
                work_entry.season, matching_work_entry.season
            ):
                entry_groups.union(
                    position, position_by_entry_id[id(matching_work_entry)]
                )
    sorted_group_positions = sorted(
        sorted(positions) for positions in entry_groups.to_sets() if len(positions) > 1
    )
    return [
        tuple(work_entries[position] for position in positions)
        for positions in sorted_group_positions
    ]


def _is_season_close(first_season: str | None, second_season: str | None) -> bool:
    """Report whether two seasons are equal, next to each other, or one is missing.

    Examples:
        >>> _is_season_close("SUMMER", "FALL")
        True
        >>> _is_season_close("WINTER", "FALL")
        False
        >>> _is_season_close(None, "FALL")
        True
    """
    if first_season is None or second_season is None:
        return True
    return (
        abs(_SEASON_ORDER.index(first_season) - _SEASON_ORDER.index(second_season)) <= 1
    )


def _find_shared_providers(entries: Sequence[WorkEntry]) -> list[str]:
    """Return the providers with pages in two or more of the entries, sorted."""
    entry_count_by_provider = Counter(
        provider for entry in entries for provider in _list_entry_providers(entry)
    )
    return sorted(
        provider
        for provider, entry_count in entry_count_by_provider.items()
        if entry_count > 1
    )


def _list_entry_providers(entry: WorkEntry) -> set[str]:
    """Return the providers an entry has pages from, Simkl and AnimeCountdown as one."""
    return {
        provider
        for key in entry.addresses_by_page_key
        if (provider := get_decided_provider(key))
    }

"""Group the seed entries to check into entry groups, each placed by one case.

An entry group is what one review case places (``work_matching.entry_group``).
The groups come from the two lists of entries to check:

- Each entry with a doubled provider forms a group of its own.
- Each group of split-work candidates forms a group.
- Groups sharing an entry merge into one. An entry can hold doubled providers
  and also be a split-work candidate, and its pages can only be placed once.

Entries are told apart by object, not by reference page, because two seed
entries can list the same page.
"""

from collections.abc import Iterable

from networkx.utils import UnionFind
from seed_cleaning.split_work_candidates import SplitWorkCandidates
from work_matching import KnownWorkEntriesFromSeed, WorkEntry


class EntryNotInSeedError(LookupError):
    """An entry given is not one of the entries the known work entries hold."""

    def __init__(self, reference_page: str) -> None:
        """Name the entry, and why a copy of an entry does not count."""
        super().__init__(
            f"Entry {reference_page} is not one of the seed's entries; entries "
            "are told apart by object, so a copy of an entry does not count"
        )


def find_entry_groups(
    doubled_provider_entries: Iterable[WorkEntry],
    split_work_candidates: Iterable[SplitWorkCandidates],
    known_work_entries: KnownWorkEntriesFromSeed,
) -> list[tuple[WorkEntry, ...]]:
    """Group the entries to check, merging groups that share an entry.

    Args:
        doubled_provider_entries: The seed entries with a doubled provider, as
            the objects ``known_work_entries`` holds.
        split_work_candidates: The groups of split-work candidates, of the
            same objects.
        known_work_entries: Every seed entry, for the seed order.

    Returns:
        Each entry group, its entries in seed order, the groups ordered by
        their first entry's place in the seed.

    Raises:
        EntryNotInSeedError: An entry is not one ``known_work_entries`` holds.
    """
    work_entries = list(known_work_entries)
    seed_position_by_entry_id = {
        id(work_entry): position for position, work_entry in enumerate(work_entries)
    }
    entry_groups = UnionFind()
    for group_entries in [
        *((entry,) for entry in doubled_provider_entries),
        *(candidates.entries for candidates in split_work_candidates),
    ]:
        entry_groups.union(
            *(
                _find_seed_position(entry, seed_position_by_entry_id)
                for entry in group_entries
            )
        )
    return [
        tuple(work_entries[position] for position in positions)
        for positions in sorted(sorted(group) for group in entry_groups.to_sets())
    ]


def _find_seed_position(
    entry: WorkEntry, seed_position_by_entry_id: dict[int, int]
) -> int:
    """Return an entry's place in the seed, or raise when the seed holds no such object."""
    position = seed_position_by_entry_id.get(id(entry))
    if position is None:
        raise EntryNotInSeedError(entry.reference_page)
    return position

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


def find_entry_groups(
    doubled_provider_entries: Iterable[WorkEntry],
    split_work_candidates: Iterable[SplitWorkCandidates],
    known_work_entries: KnownWorkEntriesFromSeed,
) -> list[tuple[WorkEntry, ...]]:
    """Group the entries to check, merging groups that share an entry.

    Args:
        doubled_provider_entries: The seed entries with a doubled provider.
        split_work_candidates: The groups of split-work candidates.
        known_work_entries: Every seed entry, for the seed order.

    Returns:
        Each entry group, its entries in seed order, the groups ordered by
        their first entry's place in the seed.
    """
    seed_position_by_entry_id = {
        id(entry): position for position, entry in enumerate(known_work_entries)
    }
    entry_by_id: dict[int, WorkEntry] = {}
    entry_groups = UnionFind()
    for group_entries in [
        *((entry,) for entry in doubled_provider_entries),
        *(candidates.entries for candidates in split_work_candidates),
    ]:
        entry_by_id |= {id(entry): entry for entry in group_entries}
        entry_groups.union(*map(id, group_entries))
    sorted_entry_ids = sorted(
        (
            sorted(entry_ids, key=seed_position_by_entry_id.__getitem__)
            for entry_ids in entry_groups.to_sets()
        ),
        key=lambda entry_ids: seed_position_by_entry_id[entry_ids[0]],
    )
    return [
        tuple(entry_by_id[entry_id] for entry_id in entry_ids)
        for entry_ids in sorted_entry_ids
    ]

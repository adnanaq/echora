"""Load and cache seed work entries for validating answers that reference seed entries.

Answers that assign a page to ``other work <page key>`` are validated against
the seed release so the review page rejects the same unknown or conflicting
target entries as the seed-cleaning pipeline.

Each caching rule reads `<rule>: <reason>`:
- Lazy loading: The seed file is parsed only when a saved answer explicitly
  names a target seed entry because loading a full release takes several
  seconds and roughly 100 MB of memory.
- Modification-time invalidation: The cached lookup is keyed by both the file
  path and its nanosecond modification timestamp so repeated saves reuse the
  in-memory lookup while a replaced seed file is reloaded automatically.
- Single-release retention: Only the most recently loaded seed release is
  retained in the LRU cache so older releases are garbage-collected.
"""

from functools import lru_cache
from pathlib import Path

from work_matching import KnownWorkEntriesFromSeed


def load_seed_entries(seed_file: Path) -> KnownWorkEntriesFromSeed:
    """Load work entries from a seed file, reusing the cached lookup when unchanged.

    Args:
        seed_file: Path to the seed release JSON file.

    Returns:
        The indexed lookup of known work entries from the seed file.
    """
    return _load_seed_entries_at(seed_file, seed_file.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def _load_seed_entries_at(
    seed_file: Path, modified_at_ns: int
) -> KnownWorkEntriesFromSeed:
    """Parse work entries from a seed file, cached by path and modification timestamp.

    Args:
        seed_file: Path to the seed release JSON file.
        modified_at_ns: File modification time in nanoseconds used as part of
            the cache key.

    Returns:
        The indexed lookup of known work entries from the seed file.
    """
    del modified_at_ns
    return KnownWorkEntriesFromSeed.from_file(seed_file)

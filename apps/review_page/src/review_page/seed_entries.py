"""Load the seed's entries for answers that name one, once per seed release.

An ``other work <page key>`` answer is checked against the seed, so the page
refuses the same named entries as the run. Each item gives the rule, then the
reason for it.

- The seed is loaded only when an answer names an entry, because loading a
  release takes a few seconds and about 100 MB, and most saves name none.
- The loaded entries are kept while the seed file's path and modification
  time stay the same, so many saves load the seed once, and a replaced
  release is read again.
- Only the last release loaded is kept, so an old release's entries are freed.
"""

from functools import lru_cache
from pathlib import Path

from work_matching import KnownWorkEntriesFromSeed


def load_seed_entries(seed_file: Path) -> KnownWorkEntriesFromSeed:
    """Return the seed file's entries, loading them only when the file is new or changed.

    Args:
        seed_file: The seed release file.

    Returns:
        The lookup over every entry in the file.
    """
    return _load_seed_entries_at(seed_file, seed_file.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def _load_seed_entries_at(
    seed_file: Path, modified_at_ns: int
) -> KnownWorkEntriesFromSeed:
    """Load a seed file's entries, cached by the file and its modification time.

    Args:
        seed_file: The seed release file.
        modified_at_ns: The file's modification time, in nanoseconds; part of
            the cache key only.

    Returns:
        The lookup over every entry in the file.
    """
    del modified_at_ns
    return KnownWorkEntriesFromSeed.from_file(seed_file)

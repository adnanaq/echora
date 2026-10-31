"""Entry groups: the entries one review case places together, and why.

An entry group is what one review case places (``work_matching.entry_group``).
Seed cleaning builds the groups in two passes:

Each item reads ``<pass>: <groups>``.

- First pass: each entry with a doubled provider forms a group of its own.
- Second pass: each group of split-work candidates, found on the entries as
  the first pass leaves them, forms a group, with the titles and links that
  made it.

Split-work candidate groups never share an entry (``split_work_candidates``
joins every group sharing one), and the second pass leaves out entries with a
first-pass case still open, so no page is placed by two groups at once.
"""

from typing import NamedTuple, Self

from seed_cleaning.split_work_candidates import SplitWorkCandidates
from work_matching import WorkEntry


class EntryGroup(NamedTuple):
    """Entries one review case places, with why they are grouped."""

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # Why the entries are candidates, shown to the reviewer; empty for a
    # single entry.
    candidate_evidence: tuple[str, ...] = ()
    # Whether a page tying the entries is also linked from entries outside the
    # group, which the review case warns about.
    has_shared_tying_link: bool = False

    @classmethod
    def from_candidates(cls, candidates: SplitWorkCandidates) -> Self:
        """Make the group of one set of split-work candidates, keeping its evidence."""
        return cls(
            candidates.entries,
            candidates.candidate_evidence,
            candidates.has_shared_tying_link,
        )

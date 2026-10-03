"""How closely two runs' search results agree, query by query."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ResultAgreement:
    identical: int
    total: int
    overlap: float


def compare_top_results(
    first: Sequence[Sequence[Hashable]], second: Sequence[Sequence[Hashable]]
) -> ResultAgreement:
    """Queries with the same IDs in the same order, and the share of IDs in common.

    ``overlap`` is the number of IDs found in both result lists of a query,
    summed over queries, divided by the total length of the first run's lists.
    """
    identical = sum(list(a) == list(b) for a, b in zip(first, second, strict=True))
    shared = sum(len(set(a) & set(b)) for a, b in zip(first, second, strict=True))
    listed = sum(len(a) for a in first)
    return ResultAgreement(
        identical=identical,
        total=len(first),
        overlap=shared / listed if listed else 1.0,
    )

"""Deduplication helpers used by strict batch update APIs."""

from collections.abc import Callable, Hashable
from typing import Literal, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
DedupPolicy = Literal["last-wins", "fail"]


class DuplicateKeyError(ValueError):
    """Raised when deduplication policy is fail and a duplicate key is found."""

    def __init__(self, key: Hashable) -> None:
        super().__init__(f"Duplicate key found: {key!r}")
        self.key = key


def deduplicate_items(
    items: list[T],
    key_fn: Callable[[T], K],
    dedup_policy: DedupPolicy = "last-wins",
) -> tuple[list[T], int]:
    """Deduplicate items using configured duplicate policy.

    Args:
        items: Input items to deduplicate.
        key_fn: Function that derives a hashable deduplication key.
        dedup_policy: Duplicate handling policy.

    Returns:
        Tuple of ``(deduplicated_items, duplicates_removed_count)``.

    Raises:
        ValueError: If policy value is unsupported.
        DuplicateKeyError: If duplicates are present under ``fail`` policy.
    """
    if dedup_policy not in {"last-wins", "fail"}:
        raise ValueError("dedup_policy must be one of: last-wins, fail")

    if dedup_policy == "fail":
        seen: set[K] = set()
        for item in items:
            key = key_fn(item)
            if key in seen:
                raise DuplicateKeyError(key)
            seen.add(key)
        return list(items), 0

    deduplicated: dict[K, T] = {}
    duplicates_removed = 0

    for item in items:
        key = key_fn(item)
        if key in deduplicated:
            duplicates_removed += 1
            deduplicated.pop(key)
        deduplicated[key] = item

    return list(deduplicated.values()), duplicates_removed

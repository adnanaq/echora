"""Deduplication helpers for Qdrant batch update operations."""

from collections.abc import Callable, Hashable
from typing import Literal, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
DedupPolicy = Literal["last-wins", "first-wins", "warn", "fail"]


class DuplicateKeyError(ValueError):
    """Raised when deduplication policy is ``fail`` and duplicate key is found."""

    def __init__(self, key: Hashable) -> None:
        """Initialize duplicate key error.

        Args:
            key: Duplicate key encountered during deduplication.
        """
        super().__init__(f"Duplicate key found: {key!r}")
        self.key = key


def deduplicate_items(
    items: list[T],
    key_fn: Callable[[T], K],
    dedup_policy: DedupPolicy = "last-wins",
    on_warn: Callable[[K], None] | None = None,
) -> tuple[list[T], int]:
    """Deduplicate a list of items by a computed key.

    Args:
        items: Items to deduplicate.
        key_fn: Function that computes deduplication key for each item.
        dedup_policy: Duplicate handling policy:
            - ``"last-wins"``: keep most recent duplicate.
            - ``"first-wins"``: keep first occurrence.
            - ``"warn"``: same as ``"last-wins"`` and invokes ``on_warn``.
            - ``"fail"``: raises ``DuplicateKeyError``.
        on_warn: Optional callback invoked with duplicate key when policy is ``warn``.

    Returns:
        Tuple ``(deduplicated_items, duplicates_removed)``.

    Raises:
        ValueError: If ``dedup_policy`` is invalid.
        DuplicateKeyError: If ``dedup_policy`` is ``"fail"`` and duplicates exist.
    """
    if dedup_policy not in {"last-wins", "first-wins", "warn", "fail"}:
        raise ValueError(  # noqa: TRY003
            "dedup_policy must be one of: last-wins, first-wins, warn, fail"
        )

    deduplicated: list[T] = []
    seen_keys: set[K] = set()

    for item in items:
        key = key_fn(item)
        if key in seen_keys:
            if dedup_policy == "first-wins":
                continue
            if dedup_policy == "fail":
                raise DuplicateKeyError(key)
            if dedup_policy == "warn" and on_warn is not None:
                on_warn(key)
            # last-wins and warn: drop prior occurrence and append latest one.
            deduplicated = [existing for existing in deduplicated if key_fn(existing) != key]

        else:
            seen_keys.add(key)

        deduplicated.append(item)

    duplicates_removed = len(items) - len(deduplicated)
    return deduplicated, duplicates_removed


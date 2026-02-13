"""Unit tests for deduplication utility helpers."""

from dataclasses import dataclass

import pytest
from qdrant_db.utils.dedup import DuplicateKeyError, deduplicate_items


@dataclass(frozen=True)
class _Item:
    """Simple test item used for deduplication behavior checks."""

    key: str
    value: int


def test_deduplicate_items_rejects_invalid_policy() -> None:
    """Invalid dedup policy must raise a validation error."""
    with pytest.raises(ValueError, match="dedup_policy must be one of"):
        deduplicate_items(
            items=[_Item(key="a", value=1)],
            key_fn=lambda item: item.key,
            dedup_policy="unknown",  # type: ignore[arg-type]
        )


def test_deduplicate_items_fail_policy_without_duplicates() -> None:
    """Fail policy should return all items unchanged when keys are unique."""
    items = [_Item(key="a", value=1), _Item(key="b", value=2)]
    deduplicated, duplicates_removed = deduplicate_items(
        items=items,
        key_fn=lambda item: item.key,
        dedup_policy="fail",
    )

    assert deduplicated == items
    assert duplicates_removed == 0


def test_deduplicate_items_fail_policy_raises_on_duplicate() -> None:
    """Fail policy should raise DuplicateKeyError on repeated keys."""
    items = [_Item(key="a", value=1), _Item(key="a", value=2)]

    with pytest.raises(DuplicateKeyError, match="Duplicate key found") as exc_info:
        deduplicate_items(items=items, key_fn=lambda item: item.key, dedup_policy="fail")

    assert exc_info.value.key == "a"


def test_deduplicate_items_last_wins_keeps_latest_and_order() -> None:
    """Last-wins policy keeps latest values and reinserts in latest-order."""
    items = [
        _Item(key="a", value=1),
        _Item(key="b", value=10),
        _Item(key="a", value=2),
        _Item(key="c", value=100),
        _Item(key="b", value=20),
    ]

    deduplicated, duplicates_removed = deduplicate_items(
        items=items,
        key_fn=lambda item: item.key,
        dedup_policy="last-wins",
    )

    assert deduplicated == [
        _Item(key="a", value=2),
        _Item(key="c", value=100),
        _Item(key="b", value=20),
    ]
    assert duplicates_removed == 2


def test_deduplicate_items_last_wins_with_empty_input() -> None:
    """Empty input should return empty output and zero duplicates removed."""
    deduplicated, duplicates_removed = deduplicate_items(
        items=[],
        key_fn=lambda item: item,
        dedup_policy="last-wins",
    )

    assert deduplicated == []
    assert duplicates_removed == 0

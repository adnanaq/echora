from dataclasses import dataclass

import pytest
from qdrant_db.utils.dedup import DuplicateKeyError, deduplicate_items


@dataclass(frozen=True)
class _Item:
    key: str
    value: int


def test_deduplicate_items_invalid_policy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dedup_policy must be one of"):
        deduplicate_items(
            items=[_Item(key="a", value=1)],
            key_fn=lambda item: item.key,
            dedup_policy="unknown",  # type: ignore[arg-type]
        )


def test_deduplicate_items_fail_policy_without_duplicates_returns_all_items() -> None:
    items = [_Item(key="a", value=1), _Item(key="b", value=2)]
    deduplicated, duplicates_removed = deduplicate_items(
        items=items,
        key_fn=lambda item: item.key,
        dedup_policy="fail",
    )

    assert deduplicated == items
    assert duplicates_removed == 0


def test_deduplicate_items_fail_policy_with_duplicate_raises_duplicate_key_error() -> (
    None
):
    items = [_Item(key="a", value=1), _Item(key="a", value=2)]

    with pytest.raises(DuplicateKeyError, match="Duplicate key found") as exc_info:
        deduplicate_items(
            items=items, key_fn=lambda item: item.key, dedup_policy="fail"
        )

    assert exc_info.value.key == "a"


def test_deduplicate_items_last_wins_keeps_latest_values_and_order() -> None:
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


def test_deduplicate_items_last_wins_empty_input_returns_empty_and_zero() -> None:
    deduplicated, duplicates_removed = deduplicate_items(
        items=[],
        key_fn=lambda item: item,
        dedup_policy="last-wins",
    )

    assert deduplicated == []
    assert duplicates_removed == 0

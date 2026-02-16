"""Unit tests for deduplication utility helpers."""

import pytest
from qdrant_db.utils.dedup import DuplicateKeyError, deduplicate_items


class TestDeduplicateItems:
    """Test suite for generic deduplication helper."""

    def test_last_wins_policy(self):
        """Keep the last occurrence when duplicates are present."""
        items = [
            {"point_id": "p1", "vector_name": "text", "value": 1},
            {"point_id": "p1", "vector_name": "text", "value": 2},
            {"point_id": "p2", "vector_name": "text", "value": 3},
        ]

        deduped, removed = deduplicate_items(
            items=items,
            key_fn=lambda i: (i["point_id"], i["vector_name"]),
            dedup_policy="last-wins",
        )

        assert removed == 1
        assert len(deduped) == 2
        assert deduped[0]["point_id"] == "p1"
        assert deduped[0]["value"] == 2
        assert deduped[1]["point_id"] == "p2"

    def test_first_wins_policy(self):
        """Keep the first occurrence when duplicates are present."""
        items = [
            {"point_id": "p1", "value": 1},
            {"point_id": "p1", "value": 2},
        ]

        deduped, removed = deduplicate_items(
            items=items,
            key_fn=lambda i: i["point_id"],
            dedup_policy="first-wins",
        )

        assert removed == 1
        assert len(deduped) == 1
        assert deduped[0]["value"] == 1

    def test_warn_policy_invokes_callback(self):
        """Invoke warning callback for each duplicate key."""
        items = [
            {"point_id": "p1", "value": 1},
            {"point_id": "p1", "value": 2},
            {"point_id": "p1", "value": 3},
        ]
        warned: list[str] = []

        deduped, removed = deduplicate_items(
            items=items,
            key_fn=lambda i: i["point_id"],
            dedup_policy="warn",
            on_warn=warned.append,
        )

        assert removed == 2
        assert len(deduped) == 1
        assert deduped[0]["value"] == 3
        assert warned == ["p1", "p1"]

    def test_fail_policy_raises_duplicate_key_error(self):
        """Raise duplicate key error when policy is fail."""
        items = [{"point_id": "p1"}, {"point_id": "p1"}]

        with pytest.raises(DuplicateKeyError) as exc_info:
            deduplicate_items(
                items=items,
                key_fn=lambda i: i["point_id"],
                dedup_policy="fail",
            )

        assert exc_info.value.key == "p1"

    def test_invalid_policy_raises_value_error(self):
        """Raise ValueError for unsupported dedup policy."""
        with pytest.raises(ValueError, match="dedup_policy must be one of"):
            deduplicate_items(
                items=[{"point_id": "p1"}],
                key_fn=lambda i: i["point_id"],
                dedup_policy="replace",  # type: ignore[arg-type]
            )

    def test_no_duplicates_returns_original_count(self):
        """Return zero duplicates removed when no duplicate keys exist."""
        items = [{"point_id": "p1"}, {"point_id": "p2"}]

        deduped, removed = deduplicate_items(
            items=items,
            key_fn=lambda i: i["point_id"],
            dedup_policy="last-wins",
        )

        assert removed == 0
        assert deduped == items


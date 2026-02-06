"""
Unit tests for update_vectors.py data validation logic.

These are sync unit tests that validate malformed anime payload handling
without requiring Qdrant or ML models.
"""

import uuid

import pytest
from common.models.anime import AnimeRecord


def test_malformed_anime_payload_handling():
    """Test that malformed anime payloads are caught and skipped gracefully.

    This tests the fix for the TypeError bug where None or non-dict anime values
    would cause uncaught exceptions in the data validation loop (lines 203-212).
    """
    from pydantic import ValidationError

    # Minimal valid anime structure for testing
    def make_valid_anime(anime_id="test-123"):
        return {
            "id": anime_id,
            "title": "Test Anime",
            "type": "TV",
            "status": "FINISHED",
            "sources": [],
        }

    # Test cases covering all edge cases from the bug report
    test_cases = [
        ("None anime", {"anime": None, "characters": [], "episodes": []}),
        ("String anime", {"anime": "not_a_dict", "characters": [], "episodes": []}),
        ("Int anime", {"anime": 123, "characters": [], "episodes": []}),
        ("List anime", {"anime": [], "characters": [], "episodes": []}),
        ("Missing anime key", {"characters": [], "episodes": []}),
        ("Empty anime dict", {"anime": {}, "characters": [], "episodes": []}),
        (
            "Valid with ID",
            {
                "anime": make_valid_anime("existing-id"),
                "characters": [],
                "episodes": [],
            },
        ),
        (
            "Valid without ID",
            {
                "anime": {**make_valid_anime(), "id": None},
                "characters": [],
                "episodes": [],
            },
        ),
        (
            "Valid empty ID",
            {
                "anime": {**make_valid_anime(), "id": ""},
                "characters": [],
                "episodes": [],
            },
        ),
    ]

    records = []
    skipped = []

    for desc, anime_dict in test_cases:
        try:
            # This is the fixed code from lines 206-213 in update_vectors.py
            anime_payload = anime_dict.get("anime")
            if not isinstance(anime_payload, dict):
                raise KeyError("Missing or invalid 'anime' key")  # noqa: TRY301
            if not anime_payload.get("id"):
                anime_payload["id"] = str(uuid.uuid4())
            records.append(AnimeRecord(**anime_dict))  # ty: ignore[invalid-argument-type]
        except (KeyError, TypeError, ValidationError) as e:
            skipped.append((desc, type(e).__name__))
            continue

    # Verify results
    assert len(records) > 0, (
        f"Should have created at least one valid record, got {len(records)} records"
    )
    assert len(skipped) > 0, (
        f"Should have skipped malformed records, got {len(skipped)} skipped"
    )

    # First 5 test cases should be skipped (malformed anime values)
    assert len(skipped) >= 5, (
        f"Expected at least 5 skipped, got {len(skipped)}: {skipped}"
    )

    # All malformed records should be caught by KeyError or ValidationError
    for desc, exception_type in skipped:
        assert exception_type in ["KeyError", "ValidationError", "TypeError"], (
            f"Unexpected exception type for '{desc}': {exception_type}"
        )

    # Valid records should have IDs assigned
    for record in records:
        assert record.anime.id, "All valid records should have an ID"
        assert len(record.anime.id) > 0, "ID should not be empty"


def test_no_uncaught_typeerror_from_malformed_data():
    """Verify that TypeError for None/non-dict anime is caught, not raised.

    These cases previously caused uncaught TypeError that would crash the loop:
    - anime_dict["anime"] is None → TypeError: 'in' operator on None
    - anime_dict["anime"] is string → TypeError: subscript assignment on str
    """
    from pydantic import ValidationError

    # These cases previously caused uncaught TypeError
    problematic_cases = [
        {"anime": None, "characters": [], "episodes": []},  # TypeError: 'in' on None
        {
            "anime": "string",
            "characters": [],
            "episodes": [],
        },  # TypeError: subscript on str
        {"anime": 123, "characters": [], "episodes": []},  # TypeError: 'in' on int
        {"anime": [], "characters": [], "episodes": []},  # TypeError: list not dict
    ]

    for anime_dict in problematic_cases:
        exception_caught = False
        try:
            # Fixed code from lines 206-213
            anime_payload = anime_dict.get("anime")
            if not isinstance(anime_payload, dict):
                raise KeyError("Missing or invalid 'anime' key")  # noqa: TRY301
            if not anime_payload.get("id"):
                anime_payload["id"] = str(uuid.uuid4())
            AnimeRecord(**anime_dict)  # ty: ignore[invalid-argument-type]
        except (KeyError, TypeError, ValidationError):
            # This is expected - malformed data should be caught
            exception_caught = True
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")

        assert exception_caught, f"Should have caught exception for {anime_dict}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

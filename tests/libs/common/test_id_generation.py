"""Tests for ID generation utility."""

import uuid

import pytest
from common.utils.id_generation import (
    generate_deterministic_id,
    generate_entity_id,
)


def _is_valid_uuid(val: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False


def test_generate_entity_id_format():
    """Test that generated entity IDs are valid UUID strings."""
    id1 = generate_entity_id()
    assert _is_valid_uuid(id1)
    # UUID format: 8-4-4-4-12 (36 chars with hyphens)
    assert len(id1) == 36
    assert id1.count("-") == 4


def test_generate_entity_id_randomness():
    """Test that entity IDs are unique (UUIDv4)."""
    id1 = generate_entity_id()
    id2 = generate_entity_id()

    assert id1 != id2


def test_generate_deterministic_id_consistency():
    """Test that deterministic IDs remain constant for same input."""
    seed = "One Piece_1071"
    id1 = generate_deterministic_id(seed)
    id2 = generate_deterministic_id(seed)

    assert id1 == id2
    assert _is_valid_uuid(id1)


def test_generate_deterministic_id_uniqueness():
    """Test that different inputs produce different deterministic IDs."""
    id1 = generate_deterministic_id("One Piece_1071")
    id2 = generate_deterministic_id("One Piece_1072")

    assert id1 != id2


def test_generate_deterministic_id_episode_pattern():
    """Test that manual episode seed generation works as expected."""
    anime_id = "01934f6c-8a2b-4c3d-9e5f-1a2b3c4d5e6f"
    ep_num = 1
    seed = f"{anime_id}_{ep_num}"

    ep_id = generate_deterministic_id(seed)
    assert _is_valid_uuid(ep_id)

    # Consistency check
    ep_id_2 = generate_deterministic_id(seed)
    assert ep_id == ep_id_2


def test_generate_deterministic_id_empty_seed_raises():
    """Test that empty seed raises ValueError."""
    with pytest.raises(ValueError, match="seed must be a non-empty string"):
        generate_deterministic_id("")


def test_generate_deterministic_id_whitespace_seed_raises():
    """Test that whitespace-only seed raises ValueError."""
    with pytest.raises(ValueError, match="seed must be a non-empty string"):
        generate_deterministic_id("   \t\n  ")

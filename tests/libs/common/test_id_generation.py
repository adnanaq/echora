"""Tests for ID generation utility."""

import time

from common.utils.id_generation import (
    generate_deterministic_id,
    generate_episode_id,
    generate_ulid,
)


def test_generate_ulid_format():
    """Test that generated ULIDs have correct prefix and length."""
    anime_id = generate_ulid("anime")
    assert anime_id.startswith("anime_")
    assert len(anime_id) == len("anime_") + 26  # ULID is 26 chars

    char_id = generate_ulid("character")
    assert char_id.startswith("char_")
    assert len(char_id) == len("char_") + 26


def test_generate_ulid_sorting():
    """Test that ULIDs are time-sortable."""
    id1 = generate_ulid("anime")
    time.sleep(0.001)  # Ensure minimal time difference
    id2 = generate_ulid("anime")

    assert id1 < id2


def test_generate_deterministic_id_consistency():
    """Test that deterministic IDs remain constant for same input."""
    seed = "One Piece_1071"
    id1 = generate_deterministic_id(seed, "episode")
    id2 = generate_deterministic_id(seed, "episode")

    assert id1 == id2
    assert id1.startswith("ep_")
    assert len(id1) == len("ep_") + 16  # 16 char hash


def test_generate_deterministic_id_uniqueness():
    """Test that different inputs produce different deterministic IDs."""
    id1 = generate_deterministic_id("One Piece_1071", "episode")
    id2 = generate_deterministic_id("One Piece_1072", "episode")

    assert id1 != id2


def test_generate_deterministic_id_no_prefix():
    """Test deterministic ID generation without prefix."""
    seed = "test_seed"
    id1 = generate_deterministic_id(seed)

    # Should be just the 16-char hash
    assert len(id1) == 16
    assert "_" not in id1


def test_generate_episode_id_structure():
    """Test that generic episode ID helper works as expected."""
    anime_id = "anime_01ARZ3NDEKTSV4RRFFQ69G5FAV"
    ep_num = 1

    ep_id = generate_episode_id(anime_id, ep_num)

    # Check prefix
    assert ep_id.startswith("ep_")

    # Check consistency (re-generating gives same ID)
    ep_id_2 = generate_episode_id(anime_id, ep_num)
    assert ep_id == ep_id_2

    # Check uniqueness (different ep number = different ID)
    ep_id_3 = generate_episode_id(anime_id, 2)
    assert ep_id != ep_id_3

    # Check uniqueness (different anime = different ID)
    ep_id_4 = generate_episode_id("anime_DIFFERENT", ep_num)
    assert ep_id != ep_id_4

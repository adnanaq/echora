"""Identity generation utilities using ULID and deterministic hashing.

This module provides standard functions for generating:
1. Unique Lexicographically Sortable Identifiers (ULID) for primary entities.
2. Deterministic SHA-256 IDs for sub-entities (e.g. Episodes) based on content.
"""

import hashlib
from typing import Literal

from ulid import ULID

# Common entity prefixes for easy identification
EntityType = Literal["anime", "character", "episode", "image", "staff", "voice_actor", "studio"]

ENTITY_PREFIXES: dict[EntityType, str] = {
    "anime": "anime_",
    "character": "char_",
    "episode": "ep_",
    "image": "img_",
    "staff": "staff_",
    "voice_actor": "va_",
    "studio": "studio_",
}


def generate_ulid(entity_type: EntityType) -> str:
    """Generate a new random, time-sortable ULID with entity prefix.

    Args:
        entity_type: The type of entity (e.g. 'anime', 'character')

    Returns:
        Prefixed ULID string (e.g., 'anime_01ARZ3NDEKTSV4RRFFQ69G5FAV')
    """
    prefix = ENTITY_PREFIXES.get(entity_type, f"{entity_type}_")
    return f"{prefix}{ULID()}"


def generate_deterministic_id(seed: str, entity_type: EntityType | None = None) -> str:
    """Generate a deterministic ID based on a unique seed string.

    Useful for entities that must have the same ID across multiple runs
    if their content hasn't changed (e.g., Episodes derived from Anime ID + Number).

    Args:
        seed: Unique string content to hash (e.g. "anime_id_episode_1")
        entity_type: Optional prefix to add to the hash

    Returns:
        Prefixed short hash (16 chars)
    """
    # Use SHA-256 for stability
    hash_object = hashlib.sha256(seed.encode("utf-8"))
    # Take first 16 chars of hex digest for a reasonably short but unique ID
    # (64 bits of entropy is sufficient for this scope)
    short_hash = hash_object.hexdigest()[:16]

    if entity_type:
        prefix = ENTITY_PREFIXES.get(entity_type, f"{entity_type}_")
        return f"{prefix}{short_hash}"

    return short_hash


def generate_episode_id(anime_id: str, episode_number: int | float) -> str:
    """Generate a deterministic ID for an episode.

    Standardizes the "anime_id + episode_number" seed pattern.

    Args:
        anime_id: The ULID of the parent anime.
        episode_number: The episode number (can be float for 1.5 etc).

    Returns:
        Deterministic ID string (e.g. 'ep_12345abcdef')
    """
    seed = f"{anime_id}_{episode_number}"
    return generate_deterministic_id(seed, "episode")

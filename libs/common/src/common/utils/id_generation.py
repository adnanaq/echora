"""Identity generation utilities using standard UUID versions.

This module provides standard functions for generating:
1. Random UUIDs (UUIDv4) for primary entities.
2. Deterministic UUIDs (UUIDv5) for sub-entities based on content.
"""

import uuid

# Namespace for Echora to ensure global uniqueness of our deterministic IDs
# Generated once: uuid.uuid4()
NAMESPACE_ECHORA = uuid.UUID("979fe79f-4bcb-499b-841f-3b9a075ae30f")


def generate_entity_id() -> str:
    """Generate a new random UUID (UUIDv4).

    Returns:
        UUID string (e.g., '01934f6c-8a2b-4c3d-9e5f-1a2b3c4d5e6f')
    """
    return str(uuid.uuid4())


def generate_deterministic_id(seed: str) -> str:
    """Generate a deterministic UUID (UUIDv5) based on a unique seed string.

    Useful for entities that must have the same ID across multiple runs
    if their content hasn't changed.

    Args:
        seed: Unique string content to hash (e.g. "anime_id_episode_1")

    Returns:
        UUID string
    """
    return str(uuid.uuid5(NAMESPACE_ECHORA, seed))

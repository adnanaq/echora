"""Mapping helpers for converting internal agent types to proto values."""

from __future__ import annotations

from agent.v1 import agent_search_pb2
from agent_core.schemas import EntityType


def entity_type_to_proto(entity_type: EntityType) -> int:
    """Map internal ``EntityType`` to ``agent_search_pb2.EntityType``.

    Args:
        entity_type: Internal entity type from agent-core schemas.

    Returns:
        Integer enum value defined by ``agent_search_pb2.EntityType``.
    """
    match entity_type:
        case EntityType.ANIME:
            return agent_search_pb2.ENTITY_TYPE_ANIME
        case EntityType.CHARACTER:
            return agent_search_pb2.ENTITY_TYPE_CHARACTER
        case EntityType.EPISODE:
            return agent_search_pb2.ENTITY_TYPE_EPISODE
        case EntityType.MANGA:
            return agent_search_pb2.ENTITY_TYPE_MANGA
    return agent_search_pb2.ENTITY_TYPE_UNSPECIFIED


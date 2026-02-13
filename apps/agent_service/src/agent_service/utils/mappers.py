"""Mapping helpers for converting internal agent types to proto values."""

from __future__ import annotations

from typing import Any, Mapping

from agent.v1 import agent_search_pb2
from agent_core.schemas import EntityType, SearchAIEvidence


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


def evidence_to_proto(
    evidence: Mapping[str, Any] | SearchAIEvidence,
) -> agent_search_pb2.SearchAIEvidence:
    """Map internal evidence payload to typed ``SearchAIEvidence`` proto.

    Args:
        evidence: Internal evidence payload from agent-core.

    Returns:
        Typed protobuf evidence message with stable, flat fields.
    """
    if isinstance(evidence, SearchAIEvidence):
        evidence_data: Mapping[str, Any] = evidence.model_dump()
    else:
        evidence_data = evidence

    def to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return agent_search_pb2.SearchAIEvidence(
        search_similarity_score=to_float(evidence_data.get("search_similarity_score"), 0.0),
        termination_reason=str(evidence_data.get("termination_reason") or ""),
        last_summary=str(evidence_data.get("last_summary") or ""),
    )

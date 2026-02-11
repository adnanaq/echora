"""Schema contracts exchanged by planner, executors, and gRPC service."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from atomic_agents import BaseIOSchema
from pydantic import Field


class EntityType(str, Enum):
    """Supported canonical entity types in the anime domain."""

    ANIME = "anime"
    CHARACTER = "character"
    EPISODE = "episode"
    MANGA = "manga"


class EntityRef(BaseIOSchema):
    """Reference to a canonical entity ID (UUID) plus its entity type."""

    entity_type: EntityType = Field(...)
    id: str = Field(..., description="Canonical UUID string (also used as Qdrant point id).")


class SearchIntent(BaseIOSchema):
    """Planner instruction for a bounded Qdrant retrieval step."""

    rationale: str = Field(..., description="Short operational reason for this step (for logs).")
    entity_type: EntityType = Field(..., description="Which entity type to search (maps to Qdrant payload entity_type).")
    query: str | None = Field(None, description="Semantic search text. None means 'filter-only' retrieval.")
    image_query: str | None = Field(
        None,
        description="Image query (URL, data-url, or raw base64).",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Qdrant payload filters (bounded by indexed payload keys).",
    )
    # TODO(vector): If we want stricter contracts, replace image_query with a structured
    # payload (mime_type + bytes or URL) instead of a free-form string.


class GraphIntent(BaseIOSchema):
    """Planner instruction for a bounded PostgreSQL graph primitive (stubbed for now)."""

    rationale: str = Field(..., description="Short operational reason for this traversal/comparison (for logs).")
    query_type: Literal["neighbors", "k_hop", "path", "compare"] = Field(...)
    start: EntityRef = Field(...)
    end: EntityRef | None = Field(None)
    max_hops: int = Field(default=3)
    edge_types: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=lambda: {"max_results": 50, "max_fanout_per_hop": 50})
    # TODO(postgres): Enforce hard caps in executor code regardless of LLM output
    # (upper bound max_hops, max_results, max_fanout_per_hop).


class RetrievalResult(BaseIOSchema):
    """Standardized output of a retrieval step (Qdrant)."""

    summary: str = Field(..., description="Compact summary of the retrieved results for the LLM.")
    raw_data: list[dict[str, Any]] = Field(..., description="Structured payloads returned by the executor.")
    count: int = Field(..., description="Number of items returned.")


class GraphPath(BaseIOSchema):
    """A single graph path represented as node IDs and relationship labels."""

    nodes: list[EntityRef] = Field(default_factory=list, description="Ordered node IDs in the path.")
    rels: list[str] = Field(default_factory=list, description="Ordered relationship types between nodes.")


class GraphResult(BaseIOSchema):
    """Standardized output of a graph step (PostgreSQL)."""

    summary: str = Field(..., description="Compact summary of the traversal/comparison results for the LLM.")
    paths: list[GraphPath] = Field(default_factory=list, description="Optional explanation paths.")
    node_ids: list[EntityRef] = Field(default_factory=list, description="All unique nodes referenced by this result.")
    count: int = Field(..., description="Number of paths or nodes returned.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata for debugging/UI.")


class SufficiencyReport(BaseIOSchema):
    """Sufficiency decision: whether retrieved context is enough to answer."""

    sufficient: bool = Field(..., description="Whether the current context is sufficient to answer.")
    rationale: str = Field(..., description="Short operational reason (no chain-of-thought).")
    missing: list[str] = Field(default_factory=list, description="Key missing facts/entities needed.")
    suggested_next_action: Literal["qdrant_search", "pg_graph"] | None = Field(
        None,
        description="Optional hint for the orchestrator; planner remains the source of truth.",
    )


class AgentResponse(BaseIOSchema):
    """Final response returned to the BFF: answer + entity IDs + evidence."""

    answer: str = Field(..., description="Final answer in markdown.")
    source_entities: list[EntityRef] = Field(
        default_factory=list,
        description="All canonical entities used as supporting evidence for the answer.",
    )
    result_entities: list[EntityRef] = Field(
        default_factory=list,
        description="Primary entities to display as the final result set (can be a subset of source_entities).",
    )
    evidence: dict[str, Any] = Field(default_factory=dict, description="Structured evidence (paths, rankings, etc.).")
    citations: list[str] = Field(default_factory=list, description="Optional source strings/urls.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class NextStep(BaseIOSchema):
    """Planner output: one bounded action (qdrant_search / pg_graph / final) plus intent payload."""

    action: Literal["qdrant_search", "pg_graph", "final"] = Field(...)
    rationale: str = Field(..., description="Short operational reason (for logs).")
    search_intent: SearchIntent | None = Field(None)
    graph_intent: GraphIntent | None = Field(None)
    final: AgentResponse | None = Field(None)

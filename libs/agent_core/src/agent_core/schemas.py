"""Schema contracts exchanged by staged agents, executors, and gRPC service."""

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

    entity_type: EntityType = Field(..., description="Canonical entity category for the referenced ID.")
    id: str = Field(..., description="Canonical UUID string (also used as Qdrant point id).")


class RewriteOutput(BaseIOSchema):
    """Normalized query rewrite and retrieval gate decision."""

    rewritten_query: str = Field(..., description="Normalized query used by downstream stages.")
    needs_external_context: bool = Field(..., description="Whether retrieval is required before answering.")
    rationale: str = Field(..., description="Short operational reason (no chain-of-thought).")


class RewriteInput(BaseIOSchema):
    """Structured input for the rewrite stage."""

    user_query: str = Field(..., description="Original user query text.")
    has_image_query: bool = Field(
        default=False,
        description="Whether the request also includes an image query payload.",
    )


class SourceSelectionInput(BaseIOSchema):
    """Structured input for source-selection at a specific turn."""

    user_query: str = Field(..., description="Original user query text.")
    rewritten_query: str = Field(..., description="Normalized query emitted by rewrite stage.")
    turn: int = Field(..., ge=1, description="Current 1-based loop turn.")
    has_image_query: bool = Field(
        default=False,
        description="Whether the request includes an image query payload.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Recent warning messages accumulated in orchestration.",
    )
    last_action: Literal["qdrant_search", "pg_graph"] | None = Field(
        None,
        description="Previous retrieval action, if any.",
    )
    last_summary: str | None = Field(
        None,
        description="Summary of the previous retrieval result, if any.",
    )


class AnswerInput(BaseIOSchema):
    """Structured input for answer drafting."""

    user_query: str = Field(..., description="Original user query text.")
    rewritten_query: str = Field(..., description="Normalized query emitted by rewrite stage.")


class SufficiencyInput(BaseIOSchema):
    """Structured input for sufficiency validation."""

    user_query: str = Field(..., description="Original user query text.")
    draft_answer: str = Field(..., description="Latest draft answer to evaluate.")


class SearchIntent(BaseIOSchema):
    """Source-selection instruction for a bounded Qdrant retrieval step."""

    rationale: str = Field(..., description="Short operational reason for this step (for logs).")
    entity_type: EntityType = Field(..., description="Which entity type to search (maps to Qdrant payload entity_type).")
    query: str | None = Field(
        None,
        description=(
            "Semantic text query. Use this for text intent only; keep natural user/entity phrasing."
        ),
    )
    image_query: str | None = Field(
        None,
        description="Image query payload (data URL or raw base64 only).",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Qdrant payload filters (bounded by indexed payload keys); do not encode filter semantics into query text.",
    )
    # TODO(vector): If we want stricter contracts, replace image_query with a structured
    # payload (mime_type + bytes) instead of a free-form string.


class GraphIntent(BaseIOSchema):
    """Source-selection instruction for a bounded PostgreSQL graph primitive."""

    rationale: str = Field(..., description="Short operational reason for this traversal/comparison (for logs).")
    query_type: Literal["neighbors", "k_hop", "path", "compare"] = Field(
        ...,
        description="Graph primitive to execute (neighbors, k-hop expansion, path, or comparison).",
    )
    start: EntityRef = Field(..., description="Start node reference for traversal/path/comparison.")
    end: EntityRef | None = Field(None, description="Optional end node for path/comparison queries.")
    max_hops: int = Field(default=3, description="Maximum traversal depth for k-hop/path operations.")
    edge_types: list[str] = Field(
        default_factory=list,
        description="Optional relationship type allowlist to constrain traversal.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional graph-level filters applied by the executor.",
    )
    limits: dict[str, int] = Field(
        default_factory=lambda: {"max_results": 50, "max_fanout_per_hop": 50},
        description="Execution limits (result cap and per-hop fanout cap).",
    )
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


class AnswerOutput(BaseIOSchema):
    """Intermediate answer candidate generated before sufficiency validation."""

    answer: str = Field(..., description="Candidate natural language answer.")
    source_entities: list[EntityRef] = Field(
        default_factory=list,
        description="All entities used as supporting context for the draft answer.",
    )
    result_entities: list[EntityRef] = Field(
        default_factory=list,
        description="Primary entities to return as final user-facing results.",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured draft evidence. Canonical keys are "
            "`search_similarity_score`, `llm_confidence`, "
            "`termination_reason`, and `last_summary`."
        ),
    )
    llm_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM self-rated confidence for this draft answer.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Draft-level warnings about ambiguity, insufficiency, or caveats.",
    )


class SufficiencyOutput(BaseIOSchema):
    """Sufficiency decision: whether retrieved context is enough to answer."""

    sufficient: bool = Field(..., description="Whether the current context is sufficient to answer.")
    rationale: str = Field(..., description="Short operational reason (no chain-of-thought).")
    missing: list[str] = Field(default_factory=list, description="Key missing facts/entities needed.")


class AgentResponse(BaseIOSchema):
    """Internal final response model produced by the orchestrator."""

    answer: str = Field(..., description="Final answer in markdown.")
    source_entities: list[EntityRef] = Field(
        default_factory=list,
        description="All canonical entities used as supporting evidence for the answer.",
    )
    result_entities: list[EntityRef] = Field(
        default_factory=list,
        description="Primary entities to display as the final result set (can be a subset of source_entities).",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured final evidence. Canonical keys are "
            "`search_similarity_score`, `llm_confidence`, "
            "`termination_reason`, and `last_summary`."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="User-visible caveats describing limitations of the final answer.",
    )


class SourceSelectionOutput(BaseIOSchema):
    """Source-selector output: one bounded retrieval action plus intent payload."""

    action: Literal["qdrant_search", "pg_graph"] = Field(
        ...,
        description="Next action to execute for this turn; current orchestrator accepts qdrant_search/pg_graph.",
    )
    rationale: str = Field(..., description="Short operational reason (for logs).")
    search_intent: SearchIntent | None = Field(
        None,
        description="Bounded Qdrant retrieval intent when action=qdrant_search.",
    )
    graph_intent: GraphIntent | None = Field(
        None,
        description="Bounded graph intent when action=pg_graph.",
    )

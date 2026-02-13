"""Schema contracts exchanged by staged agents, executors, and gRPC service."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from atomic_agents import BaseIOSchema
from pydantic import Field


# ============================================================================
# Common/Shared Schemas
# ============================================================================


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


class SearchAIEvidence(BaseIOSchema):
    """Structured evidence and metrics for a search response.

    Uses objective metrics instead of subjective LLM self-ratings.
    """

    search_similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Top vector similarity score from retrieval.",
    )
    num_sources: int = Field(
        default=0,
        ge=0,
        description="Number of retrieved sources used in the answer.",
    )
    termination_reason: str = Field(
        default="",
        description="Reason the orchestration loop stopped (e.g. 'sufficient', 'max_turns', 'no_match').",
    )


# ============================================================================
# Rewrite Agent Schemas
# ============================================================================


class RewriteInput(BaseIOSchema):
    """Structured input for the rewrite stage."""

    user_query: str = Field(..., description="Original user query text.")
    has_image_query: bool = Field(
        default=False,
        description="Whether the request also includes an image query payload.",
    )
    current_rewritten_query: str | None = Field(
        None,
        description="Latest rewritten query to refine after insufficient results.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Concrete missing facts or constraints from the previous attempt.",
    )
    last_retrieval_summary: str | None = Field(
        None,
        description="Summary from the previous retrieval step to guide query refinement.",
    )


class RewriteOutput(BaseIOSchema):
    """Normalized query rewrite, retrieval gate, and intent classification."""

    rewritten_query: str = Field(
        ...,
        description="Normalized query optimized for retrieval. Examples: 'Boruto character relationships' → 'Boruto AND relationship'; 'anime like Death Note' → 'psychological thriller anime mystery'"
    )
    needs_external_context: bool = Field(
        ...,
        description="Whether retrieval is required. Set to False only for general knowledge questions (e.g., 'What is anime?'). Set to True for specific entity queries, comparisons, recommendations, or relationship questions."
    )
    requires_graph_traversal: bool = Field(
        default=False,
        description=(
            "Whether this query EXPLICITLY requires PostgreSQL graph traversal for deep relationships/comparisons. "
            "Set to True ONLY for: explicit relationship questions ('how is X related to Y', 'what connects A to B'), "
            "cross-entity comparisons ('compare Naruto vs Bleach'), or family trees ('Naruto's family'). "
            "Set to False for: semantic/content search ('anime with X and Y', 'anime about Z'), "
            "character search ('shows with dog character'), entity lookups, or recommendations. "
            "DEFAULT to False - be conservative, most queries are semantic search."
        )
    )
    rationale: str = Field(
        ...,
        description="One-sentence operational reason for decisions (max 20 words). Example: 'Requires graph traversal to compare relationships across two separate anime universes.'"
    )


# ============================================================================
# Source Selector Agent Schemas
# ============================================================================


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
    attempted_actions: list[Literal["qdrant_search", "pg_graph"]] = Field(
        default_factory=list,
        description="Recent retrieval actions already attempted in this request.",
    )
    requires_graph_traversal: bool = Field(
        default=False,
        description=(
            "Whether the rewrite agent detected that this query requires graph traversal for relationships/comparisons. "
            "Use this as a hint: if True and pg_graph not yet attempted, consider trying pg_graph."
        )
    )


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
    """Source-selection instruction for a bounded PostgreSQL graph primitive.

    Safety limits (max_hops, max_results, max_fanout) are enforced by the
    executor, not specified by the LLM.
    """

    rationale: str = Field(..., description="Short operational reason for this traversal/comparison (for logs).")
    query_type: Literal["neighbors", "k_hop", "path", "compare"] = Field(
        ...,
        description="Graph primitive to execute (neighbors, k-hop expansion, path, or comparison).",
    )
    start: EntityRef = Field(..., description="Start node reference for traversal/path/comparison.")
    end: EntityRef | None = Field(None, description="Optional end node for path/comparison queries.")
    edge_types: list[str] = Field(
        default_factory=list,
        description="Optional relationship type allowlist to constrain traversal.",
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


# ============================================================================
# Answer Agent Schemas
# ============================================================================


class AnswerInput(BaseIOSchema):
    """Structured input for answer drafting."""

    user_query: str = Field(..., description="Original user query text.")
    rewritten_query: str = Field(..., description="Normalized query emitted by rewrite stage.")


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
    evidence: SearchAIEvidence = Field(
        default_factory=SearchAIEvidence,
        description="Structured evidence and metrics for this draft.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Draft-level warnings about ambiguity, insufficiency, or caveats.",
    )


# ============================================================================
# Sufficiency Agent Schemas
# ============================================================================


class SufficiencyInput(BaseIOSchema):
    """Input for sufficiency validation with retrieval context."""

    user_query: str = Field(
        ...,
        description="Original user query text exactly as the user asked it."
    )
    draft_answer: str = Field(
        ...,
        description="Latest draft answer generated from retrieved context. Evaluate whether this answer fully addresses the user_query."
    )
    last_search_similarity_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Top similarity score from the most recent Qdrant search (0.0-1.0). "
            "Scores >0.7 indicate high semantic relevance. "
            "Scores <0.5 suggest weak matches. "
            "Use this to assess retrieval confidence, but do NOT use score alone to decide sufficiency—"
            "always check if the draft_answer actually addresses the user_query."
        )
    )
    attempted_actions: list[Literal["qdrant_search", "pg_graph"]] = Field(
        default_factory=list,
        description=(
            "List of retrieval methods already attempted in this request: 'qdrant_search', 'pg_graph'. "
            "If query EXPLICITLY asks about relationships ('how is X related to Y') but only 'qdrant_search' was tried, request pg_graph. "
            "Do NOT request pg_graph for content/theme searches ('anime with X and Y') - those are semantic search. "
            "Example: ['qdrant_search', 'qdrant_search'] means only semantic search was used."
        )
    )


class SufficiencyOutput(BaseIOSchema):
    """Sufficiency decision with explicit graph traversal requests."""

    sufficient: bool = Field(
        ...,
        description=(
            "Whether the current context is sufficient to answer the user's query. "
            "Set to True if: (1) draft_answer addresses user_query with score >0.7, OR "
            "(2) draft_answer fully addresses user_query even with lower score. "
            "Set to False ONLY if: answer is clearly incomplete, vague, or query EXPLICITLY asks for relationships but pg_graph not attempted. "
            "Be generous - if Qdrant found good matches (score >0.7), accept the answer."
        )
    )
    rationale: str = Field(
        ...,
        description=(
            "One-sentence operational reason (max 25 words). "
            "Examples: 'Sufficient: high similarity (0.85) and draft covers all entities.' OR "
            "'Insufficient: relationship query needs pg_graph but only qdrant_search attempted.'"
        )
    )
    missing: list[str] = Field(
        default_factory=list,
        max_items=5,
        description=(
            "Concrete missing facts, entities, or retrieval methods. "
            "Examples: 'relationship between Boruto and Naruto characters', 'pg_graph traversal', "
            "'episode-level details', 'character family tree'. Keep each item under 10 words."
        )
    )
    need_graph_traversal: bool = Field(
        default=False,
        description=(
            "Set to True ONLY if the query EXPLICITLY asks for relationships ('how is X related to Y', 'compare A vs B') "
            "AND pg_graph was not yet attempted (check attempted_actions). "
            "Do NOT set to True for content/theme searches - those are semantic search, not graph traversal. "
            "DEFAULT to False - be very conservative."
        )
    )


# ============================================================================
# Executor Result Schemas
# ============================================================================


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
